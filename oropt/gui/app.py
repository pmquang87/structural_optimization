"""Streamlit dashboard: configure, launch, and live-monitor a BESO run.

Launch with::

    streamlit run oropt/gui/app.py

The GUI is fully decoupled from the solver: it edits the YAML config, starts
``python -m oropt.run`` as a detached subprocess, and then only *reads* the
status files the loop writes (``status.json`` / ``history.csv`` /
``topology_latest.vtu``). Closing the browser never stops the run; reopening
re-attaches to it.

The page is laid out as a sidebar (config selection + run/queue control) and six
tabs; each tab's body lives in its own ``render_*_tab`` function below so this
script stays readable as orchestration rather than one long top-level block.
"""
from __future__ import annotations

import dataclasses
import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from oropt import queue_runner
from oropt import status as st_io
from oropt.animate import (VIEWS as _BUILTIN_VIEWS, frame_count,
                           make_animation, selectable_views)
from oropt.config import AnimateOpts, Config, LoadCase
from oropt.gui import queue_store as qs
from oropt.gui.boxes import (BOX_COLUMNS, growth_boxes_from_records,
                             records_from_growth_boxes)
from oropt.gui.cases import (CASE_COLUMNS, load_cases_from_records,
                             records_from_load_cases)
from oropt.gui.colors import COMMON_COLORS, OTHER, is_valid_color
from oropt.gui.runstate import find_active_run
from oropt.gui.views import (VIEW_COLUMNS, custom_views_from_records,
                             records_from_custom_views)
from oropt.validate import ERROR, check_config, has_errors

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = PROJECT_ROOT / "configs" / "elevator_linkage.yaml"
QUEUE_PATH = qs.default_queue_path(PROJECT_ROOT)
QUEUE_BADGE = {qs.PENDING: "⏳ pending", qs.RUNNING: "🟢 running",
               qs.DONE: "✅ done", qs.FAILED: "❌ failed", qs.SKIPPED: "⤳ skipped"}

st.set_page_config(page_title="oropt — OpenRadioss BESO", layout="wide")


# ---- run control -----------------------------------------------------------
def launch_run(cfg_path: Path, resume: bool) -> None:
    # Reuse the queue runner's detached-launch helpers so ↻ Resume and a queued run
    # share one definition of "launch oropt.run detached" (same command, same
    # CREATE_NEW_PROCESS_GROUP | DETACHED_PROCESS flags) — no drift.
    queue_runner.spawn_detached(
        queue_runner.run_argv(cfg_path, resume), PROJECT_ROOT)


def request_stop(work: Path) -> None:
    (work / "stop.flag").write_text("stop", encoding="utf-8")


def snapshot_into_case_dir(source: str | Path) -> str:
    """Freeze *source* as the queue's immutable run config and return its path.

    Stored in the model's case directory (under ``queue_configs/``) so the frozen
    config travels with the run/case data; falls back to beside the source when the
    case dir can't be resolved. A queued run is launched from this copy, so later
    edits to the working config can't change a run already in the queue.
    """
    case_dir = qs.resolve_case_dir(source, PROJECT_ROOT)
    dest = str(Path(case_dir) / qs.QUEUE_CONFIG_DIRNAME) if case_dir else None
    return qs.snapshot_config(source, dest)


def force_kill(work: Path) -> None:
    pid = st_io.read_pid(work)
    if pid and sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    elif pid:
        import os
        import signal
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def _fmt_limit(v, fmt: str = "{:.0f}") -> str:
    """Format a feasibility limit for display; a blank limit (None/NaN) -> '—'."""
    return "—" if v is None or v != v else fmt.format(v)


# ---- Input tab -------------------------------------------------------------
def render_input_tab(cfg: Config) -> None:
    st.subheader("Model")
    cfg.model.case_dir = st.text_input("Case directory", cfg.model.case_dir)
    cfg.work_dir = st.text_input(
        "Run / output folder", cfg.work_dir,
        placeholder=cfg.run_folder(),
        help="Scratch, checkpoints and status files go here. Leave blank to use "
             f"the case directory itself (→ `{cfg.run_folder()}`); the mutated "
             "deck is isolated in its solve/ sub-folder so the source decks are "
             "never clobbered.")
    cc = st.columns(2)
    cfg.model.design_part_id = int(cc[0].number_input(
        "Design part id", value=cfg.model.design_part_id, step=1))
    cfg.model.bc_group_id = int(cc[1].number_input(
        "BC node-group id", value=cfg.model.bc_group_id, step=1))
    cases = cfg.load_case_list()
    stems = ", ".join(c.stem for c in cases) if cases else "—"
    st.caption(f"OpenRadioss root: `{cfg.or_paths.root}`  ·  np={cfg.run.np} "
               f"nt={cfg.run.nt}  ·  deck(s) `{stems}`")
    st.caption("The deck stem, displacement node and σ/d limits are defined "
               "**per load case** on the 🔀 Load cases tab.")

    st.subheader("Solver backend")
    cfg.docker.enabled = st.checkbox(
        "Run OpenRadioss via Docker (MUMPS implicit — no Intel MPI needed)",
        value=cfg.docker.enabled,
        help="Use the Dockerised OpenRadioss build instead of the native Windows "
             "binaries (works on AMD or Intel). Requires Docker Desktop running "
             "and the image loaded; outputs are written into the run folder, "
             "exactly like the native backend.")
    if cfg.docker.enabled:
        dk = st.columns([2, 1, 1])
        cfg.docker.image = dk[0].text_input("Docker image", cfg.docker.image)
        cfg.docker.np = int(dk[1].number_input(
            "MPI np", value=int(cfg.docker.np), min_value=1, step=1))
        cfg.docker.nt = int(dk[2].number_input(
            "Threads nt", value=int(cfg.docker.nt), min_value=1, step=1))
        st.caption("Keep np × nt ≤ CPU cores. The container bind-mounts the run "
                   "folder to /data and writes results back there.")


# ---- Load cases tab --------------------------------------------------------
def render_load_cases_tab(cfg: Config, cfg_path: Path) -> None:
    st.subheader("Load cases")
    st.caption(
        "Define the load case(s) the part is optimised against — the **single "
        "source of truth** for each deck's stem, the constrained displacement "
        "node and the σ/d limits. A single-load run is just **one** row. Add more "
        "rows to optimise a **weighted-sum compliance** over several loads (the "
        "linkage pulled in different directions); each row is a separate deck pair "
        "in the case directory that shares the same mesh — only its load differs. "
        "*Deck stem* is required; leave *σ_allow* / *d_allow* blank to leave that "
        "quantity unconstrained.")
    if not cfg.load_cases:               # always offer at least one row to fill in
        cfg.load_cases = [LoadCase(name="case", weight=1.0,
                                   sigma_allow=250.0, d_allow=1.0)]
    lc_df = pd.DataFrame(records_from_load_cases(cfg.load_cases),
                         columns=CASE_COLUMNS)
    lc_edited = st.data_editor(
        lc_df, num_rows="dynamic", width="stretch",
        key="load_cases_editor", column_config={
            "name": st.column_config.TextColumn(
                "Name", help="Label for the load case, e.g. pull_z."),
            "stem": st.column_config.TextColumn(
                "Deck stem", help="<stem>_0000.rad / _0001.rad in the case "
                                  "directory. Required."),
            "weight": st.column_config.NumberColumn(
                "Weight", min_value=0.0, step=0.1, format="%.3f",
                help="wᵢ in s_e = Σ wᵢ·(energyᵢ / max energyᵢ). Blank → 1."),
            "disp_node_id": st.column_config.NumberColumn(
                "Disp node id", step=1, format="%d",
                help="Constrained node for this case. Blank → none tracked."),
            "sigma_allow": st.column_config.NumberColumn(
                "σ_allow [MPa]", min_value=0.0, step=1.0,
                help="Per-case von-Mises stress limit. Blank → no stress limit."),
            "d_allow": st.column_config.NumberColumn(
                "d_allow [mm]", min_value=0.0, step=0.1,
                help="Per-case displacement limit. Blank → no displacement limit."),
        })
    cfg.load_cases = load_cases_from_records(lc_edited.to_dict("records"))
    n = len(cfg.load_cases)
    if n == 0:
        st.warning("Define at least one load case — fill in a row above "
                   "(a deck stem is required; σ_allow / d_allow may be blank).")
    elif n == 1:
        st.info("Single load case (classic single-load BESO). Add a row above to "
                "optimise several loads.")
    else:
        st.success(
            f"{n} load cases: every iteration solves all of them (≈ {n}× a "
            "single-case run, each under `solve/case_<i>/`); the design is "
            "feasible only when **every** case is.")
    if st.button("💾 Save config", key="lc_save"):
        cfg.to_yaml(cfg_path)
        st.success(f"Saved to {cfg_path}")


# ---- shared evolution-animation camera/playback widgets --------------------
def render_camera_settings(opts: AnimateOpts, key_prefix: str) -> None:
    """Render the evolution-animation camera & playback widgets into *opts*.

    Shared by the Optimiser / Output tab (settings the post-run animation will
    use) and the 🎬 Re-animate tab (settings for an on-demand re-render), so the two
    can never drift. Every widget writes straight back into the passed *opts*
    (an :class:`~oropt.config.AnimateOpts`); *key_prefix* keeps the two instances'
    Streamlit widget keys distinct so both can live on the page at once.
    """
    st.caption("Custom camera angles — name your own viewpoints (a built-in base "
               "+ azimuth/elevation offsets) to reuse them in the dropdown below.")
    cv_df = pd.DataFrame(records_from_custom_views(opts.custom_views),
                         columns=VIEW_COLUMNS)
    cv_edited = st.data_editor(
        cv_df, num_rows="dynamic", width="stretch",
        key=f"{key_prefix}_custom_views_editor", column_config={
            "name": st.column_config.TextColumn(
                "name", help="Pick this angle by name in 'Camera angle'."),
            "base": st.column_config.SelectboxColumn(
                "base", options=list(_BUILTIN_VIEWS), default="iso",
                help="Built-in preset this angle starts from."),
            "azimuth": st.column_config.NumberColumn(
                "azimuth [°]", step=15.0, help="Offset about the vertical."),
            "elevation": st.column_config.NumberColumn(
                "elevation [°]", step=15.0, help="Up/down tilt offset."),
        })
    opts.custom_views = custom_views_from_records(cv_edited.to_dict("records"))

    ac = st.columns(3)
    _views = selectable_views(opts)                # custom names + built-in presets
    opts.view = ac[0].selectbox(
        "Camera angle", _views,
        index=_views.index(opts.view) if opts.view in _views else 0,
        key=f"{key_prefix}_view",
        help="Viewpoint for every frame: a custom angle (above), iso (3D), or a "
             "straight-on front/back/left/right/top/bottom. The azimuth/elevation "
             "here are added on top as a final nudge.")
    opts.azimuth = float(ac[1].number_input(
        "Azimuth [°]", value=float(opts.azimuth), step=15.0,
        key=f"{key_prefix}_azimuth",
        help="Extra camera rotation about the vertical, applied after the preset."))
    opts.elevation = float(ac[2].number_input(
        "Elevation [°]", value=float(opts.elevation), step=15.0,
        key=f"{key_prefix}_elevation",
        help="Extra camera tilt up/down, applied after the preset."))
    ac2 = st.columns(2)
    opts.fps = float(ac2[0].number_input(
        "Frames per second", value=float(opts.fps), min_value=0.5, step=1.0,
        key=f"{key_prefix}_fps"))
    opts.show_labels = ac2[1].checkbox(
        "Stamp 'iter N' on each frame", value=opts.show_labels,
        key=f"{key_prefix}_show_labels")
    opts.opacity = float(st.slider(
        "Surface opacity", min_value=0.0, max_value=1.0,
        value=float(opts.opacity), step=0.05, key=f"{key_prefix}_opacity",
        help="1.0 = solid; lower makes the design see-through so internal "
             "structure shows. Transparency uses depth peeling when available."))


def color_picker(container, label: str, current: str, key_prefix: str
                 ) -> tuple[str, bool]:
    """A named-colour dropdown + an "Other…" hex/name escape hatch in *container*.

    Returns ``(value, valid)``. A dropdown pick is always valid; a custom entry is
    checked with :func:`oropt.gui.colors.is_valid_color` so a typo is caught in the
    form rather than only as a failed render. *current* pre-selects the matching
    name, else the "Other…" box pre-filled with it. *key_prefix* namespaces the
    widgets so several pickers coexist on the page.

    The controls are keyed, so Streamlit persists the user's pick across reruns and
    ignores ``index``/``value`` after the first render — what we want while editing
    one config, but it means loading a *different* config in the sidebar would leave
    the box showing the previous pick (a lie the Save button would then write back
    over the new config). So we re-seed both controls from *current* whenever it
    changes underneath us, which leaves in-session edits untouched.
    """
    current = (current or "").strip()
    options = list(COMMON_COLORS) + [OTHER]
    preset = current if current in COMMON_COLORS else OTHER
    name_key, custom_key = f"{key_prefix}_name", f"{key_prefix}_custom"
    seed_key = f"{key_prefix}_seed"
    if st.session_state.get(seed_key) != current:      # config changed (or first run)
        st.session_state[seed_key] = current
        st.session_state[name_key] = preset
        st.session_state[custom_key] = current if preset == OTHER else ""

    choice = container.selectbox(label, options, key=name_key)
    if choice != OTHER:
        return choice, True
    value = container.text_input(
        f"{label} (hex or name)", key=custom_key, placeholder="#b0c4de").strip()
    ok = is_valid_color(value)
    if value and not ok:
        container.caption(f"⚠ '{value}' isn't a colour pyvista recognises.")
    return value, ok


def render_appearance_settings(opts: AnimateOpts, key_prefix: str
                               ) -> tuple[bool, bool]:
    """Render the evolution-animation appearance & resolution widgets into *opts*.

    Shared by the Optimiser / Output tab (the look the post-run animation will use)
    and the 🎬 Re-animate tab (an on-demand re-render), so the two can never drift — the
    same split as :func:`render_camera_settings`. Returns ``(color_ok, bg_ok)`` from
    the two colour pickers so the caller can gate its action button on valid
    colours. *key_prefix* namespaces the widgets so both instances coexist.
    """
    st.markdown("**Appearance & resolution**")
    st.caption("Colours take a named colour (e.g. `steelblue`), a hex code "
               "(`#b0c4de`), or a matplotlib `tab:` name — pick a common one or "
               "choose *Other…* to type your own.")
    rc = st.columns(3)
    opts.color, color_ok = color_picker(rc[0], "Surface colour", opts.color,
                                        f"{key_prefix}_color")
    opts.background, bg_ok = color_picker(rc[1], "Background", opts.background,
                                          f"{key_prefix}_bg")
    opts.show_edges = rc[2].checkbox("Show mesh edges", value=opts.show_edges,
                                     key=f"{key_prefix}_edges")
    rc2 = st.columns(3)
    opts.window_w = int(rc2[0].number_input(
        "Width [px]", value=int(opts.window_w), min_value=160, step=80,
        key=f"{key_prefix}_w",
        help="Render width — higher = sharper but slower/larger."))
    opts.window_h = int(rc2[1].number_input(
        "Height [px]", value=int(opts.window_h), min_value=120, step=80,
        key=f"{key_prefix}_h"))
    opts.hold_last = int(rc2[2].number_input(
        "Hold last frame (×)", value=int(opts.hold_last), min_value=1, step=1,
        key=f"{key_prefix}_hold",
        help="Linger on the final design, in multiples of one frame's duration."))
    opts.render_timeout_s = float(st.number_input(
        "Render timeout [s]", value=float(opts.render_timeout_s),
        min_value=10.0, step=30.0, key=f"{key_prefix}_timeout",
        help="Cap on the off-screen render subprocess (all frames). Raise it for "
             "many high-resolution frames."))
    return color_ok, bg_ok


# ---- Optimiser / Output tab ------------------------------------------------
def render_constraints_tab(cfg: Config, cfg_path: Path) -> None:
    st.caption("Feasibility limits (σ_allow / d_allow) are now set **per load "
               "case** on the 🔀 Load cases tab.")
    st.subheader("Optimiser")
    _opts = ["beso", "levelset", "tobs", "hca"]
    _opt_labels = {"beso": "BESO — bi-directional element removal",
                   "levelset": "Level-set — smoother boundaries",
                   "tobs": "TOBS — binary ILP flips (Sivapuram & Picelli 2018)",
                   "hca": "HCA — hybrid cellular automata (LS-TaSC-style)"}
    opt_name = st.selectbox(
        "Topology optimiser", _opts,
        index=_opts.index(cfg.optimizer_name()) if cfg.optimizer_name() in _opts else 0,
        format_func=lambda k: _opt_labels[k],
        help="Which algorithm drives the loop. All share the sensitivity pipeline "
             "(filter + Huang-Xie history) and volume/feasibility bookkeeping; only "
             "the per-iteration design update differs.")
    cfg.optimizer = opt_name
    # The active block carries the shared knobs below (so a level-set / TOBS run is
    # fully specified by its own block); for BESO this is cfg.beso — unchanged.
    aopt = cfg.active_opts()

    st.subheader("Keep-out / non-design regions")
    st.caption("Design elements touching these nodes are frozen (never deleted).")
    fg = st.text_input("Freeze /GRNOD/NODE group ids (comma-sep, e.g. 99999999)",
                       ",".join(str(x) for x in cfg.model.freeze_group_ids))
    fn = st.text_input("Freeze explicit node ids (comma-sep)",
                       ",".join(str(x) for x in cfg.model.freeze_node_ids))
    cfg.model.freeze_group_ids = [int(x) for x in fg.replace(" ", "").split(",") if x]
    cfg.model.freeze_node_ids = [int(x) for x in fn.replace(" ", "").split(",") if x]
    allow_del_bc = st.checkbox(
        "Allow deleting elements at BC nodes",
        value=not aopt.protect_bc_nodes,
        help="By default the BC node-group (model.bc_group_id) is frozen. Enable "
             "this to let the optimiser remove material there too — the BC nodes "
             "stay fixed via their /BCS and still anchor connectivity.")
    aopt.protect_bc_nodes = not allow_del_bc

    st.subheader("Stress-exclusion regions")
    st.caption("Design elements touching these nodes have their von-Mises "
               "**ignored** — left out of σ_max, the feasibility check and the "
               "Monitor/report stress. Use for a known hot-spot a later design "
               "phase will fix (they still take part in the optimisation; add "
               "them to the keep-out set above too if you also want them frozen).")
    sg = st.text_input(
        "Ignore-stress /GRNOD/NODE group ids (comma-sep, e.g. 999999998)",
        ",".join(str(x) for x in cfg.model.stress_exclude_group_ids))
    sn = st.text_input("Ignore-stress explicit node ids (comma-sep)",
                       ",".join(str(x) for x in cfg.model.stress_exclude_node_ids))
    cfg.model.stress_exclude_group_ids = [
        int(x) for x in sg.replace(" ", "").split(",") if x]
    cfg.model.stress_exclude_node_ids = [
        int(x) for x in sn.replace(" ", "").split(",") if x]

    st.subheader("Growth boxes — add material")
    st.caption(
        "Axis-aligned boxes (two opposite corners, like LS-DYNA `*DEFINE_BOX` / "
        "Radioss `/BOX/RECTA`) marking **candidate growth material**: design "
        "elements whose centroid lies inside a box start the run **void**, and "
        "the optimiser may *add* them where the load path wants — so the design "
        "can grow material where the original part had none. The box volume must "
        "be **pre-meshed** into the design part (same `/TETRA4` block, "
        "node-conformal interface, node ids ≥ design_node_min); a box over "
        "unmeshed space is an error at run start. Multiple boxes act as a union; "
        "volume fractions are then relative to the enlarged (part + boxes) space.")
    gb_df = pd.DataFrame(records_from_growth_boxes(cfg.model.growth_boxes),
                         columns=BOX_COLUMNS)
    gb_edited = st.data_editor(
        gb_df, num_rows="dynamic", width="stretch",
        key="growth_boxes_editor", column_config={
            "name": st.column_config.TextColumn(
                "Name", help="Label for run-log / validation messages."),
            **{k: st.column_config.NumberColumn(
                k, format="%.3f",
                help="Box bound in model units (e.g. mm). All six are required.")
               for k in BOX_COLUMNS[1:]},
        })
    cfg.model.growth_boxes = growth_boxes_from_records(
        gb_edited.to_dict("records"))
    if cfg.model.growth_boxes:
        st.info(f"{len(cfg.model.growth_boxes)} growth box(es): their elements "
                "start void and may be grown into. With BESO, keep "
                "`max_add_ratio` ≥ `evolution_rate` so back-off growth isn't "
                "throttled (validation warns otherwise).")

    _opt_short = {"beso": "BESO", "levelset": "Level-set", "tobs": "TOBS",
                  "hca": "HCA"}
    st.subheader(f"{_opt_short[opt_name]} parameters")
    g = st.columns(3)
    aopt.evolution_rate = g[0].number_input(
        "Evolution rate (vol/iter)", value=float(aopt.evolution_rate),
        step=0.005, format="%.3f", key=f"evo_{opt_name}")
    aopt.target_volume_fraction = g[1].number_input(
        "Target volume fraction", value=float(aopt.target_volume_fraction),
        min_value=0.05, max_value=1.0, step=0.05, key=f"tvf_{opt_name}")
    aopt.filter_radius = g[2].number_input(
        "Filter radius [mm]", value=float(aopt.filter_radius), step=0.5,
        key=f"fr_{opt_name}")
    h = st.columns(3)
    aopt.history_weight = h[0].slider(
        "History weight", 0.0, 1.0, float(aopt.history_weight), key=f"hw_{opt_name}")
    aopt.max_iter = int(h[1].number_input(
        "Max iterations", value=int(aopt.max_iter), step=10, key=f"mi_{opt_name}"))
    aopt.sensitivity = h[2].selectbox(
        "Sensitivity", ["energy", "vonmises", "blend"],
        index=["energy", "vonmises", "blend"].index(aopt.sensitivity),
        key=f"sens_{opt_name}")

    st.markdown("**Feasibility back-off** — how the volume target reacts to the "
                "constraint values. v is the worst utilisation ratio across load "
                "cases (σ_max/σ_allow and d/d_allow; v ≤ 1 = feasible). Defaults "
                "= the classic binary gate (fixed ±ER step from feasible/"
                "infeasible alone), which is known to ping-pong across the limit.")
    b = st.columns(4)
    aopt.backoff_gain = b[0].number_input(
        "Back-off gain", value=float(aopt.backoff_gain), min_value=0.0,
        step=0.5, format="%.2f", key=f"bogain_{opt_name}",
        help="0 = classic gate: any violation grows the target by one full "
             "evolution-rate step. > 0 = proportional: the growth step is "
             "ER·min(gain·(v−1), cap), so a 1 % violation triggers a nudge and "
             "a large one a capped surge. Size it so gain·(typical overshoot) "
             "≈ 1, e.g. 10–20.")
    aopt.backoff_cap = b[1].number_input(
        "Back-off cap (×ER)", value=float(aopt.backoff_cap), min_value=0.1,
        step=0.5, format="%.1f", key=f"bocap_{opt_name}",
        help="Cap on the proportional growth step, in multiples of the "
             "evolution rate. Only used when the gain is > 0.")
    aopt.damping_threshold = b[2].number_input(
        "Damping threshold", value=float(aopt.damping_threshold),
        min_value=0.05, max_value=1.0, step=0.01, format="%.2f",
        key=f"damp_{opt_name}",
        help="While feasible with v above this, removal slows by "
             "(1−v)/(1−threshold) so the design glides into the limit instead "
             "of overshooting and oscillating. 1.0 = off (full rate until "
             "infeasible); 0.9–0.95 is typical.")
    aopt.addback_stress_bias = b[3].number_input(
        "Add-back stress bias", value=float(aopt.addback_stress_bias),
        min_value=0.0, step=0.5, format="%.2f", key=f"abbias_{opt_name}",
        help="When a stress limit is violated, scale the sensitivity driving "
             "the update by (1 + bias·σ_vm/σ_allow), spatially filtered, so "
             "the material added back lands near the overstressed region "
             "instead of wherever the energy ranking points. 0 = off.")

    # ---- optimiser-specific knobs -----------------------------------------
    if opt_name == "tobs":
        t = st.columns(2)
        cfg.tobs.flip_limit = t[0].number_input(
            "Flip move-limit β (frac/iter)", value=float(cfg.tobs.flip_limit),
            min_value=0.005, max_value=0.5, step=0.005, format="%.3f",
            help="Max fraction of elements the ILP may flip per iteration "
                 "(Σ|Δx| ≤ β·N). 0.01–0.05 is typical.")
        cfg.tobs.constraint_relaxation = t[1].number_input(
            "Constraint relaxation ε", value=float(cfg.tobs.constraint_relaxation),
            min_value=0.0, max_value=0.2, step=0.005, format="%.3f",
            help="Relaxation band (×V0) on the linearised volume constraint so the "
                 "binary ILP is always feasible (the paper's ε).")
    elif opt_name == "levelset":
        t = st.columns(3)
        cfg.levelset.dt = t[0].number_input(
            "Level-set dt", value=float(cfg.levelset.dt), step=0.1,
            help="Pseudo-time step for the φ evolution.")
        cfg.levelset.smoothing_passes = int(t[1].number_input(
            "Smoothing passes", value=int(cfg.levelset.smoothing_passes),
            min_value=0, step=1,
            help="Laplacian/Jacobi regularisation passes per iteration."))
        cfg.levelset.band_width = t[2].number_input(
            "Band width", value=float(cfg.levelset.band_width), step=0.5,
            help="Clamp |φ| to this each step to keep the field bounded.")
    elif opt_name == "hca":
        t = st.columns(3)
        cfg.hca.kp = t[0].number_input(
            "Controller gain Kp", value=float(cfg.hca.kp),
            min_value=0.05, max_value=5.0, step=0.05, format="%.2f",
            help="Proportional gain of the density controller "
                 "Δx = Kp·(S−S*)/S*. Keep min(Kp, move limit) > 0.5 or no "
                 "element can be removed in a single iteration (removal then "
                 "lags the volume target over extra solves).")
        cfg.hca.move_limit = t[1].number_input(
            "Move limit (Δx/iter)", value=float(cfg.hca.move_limit),
            min_value=0.05, max_value=1.0, step=0.05, format="%.2f",
            help="Cap on each element's virtual-density change per iteration. "
                 "1.0 = uncapped; lower it for smoother, more damped evolution.")
        cfg.hca.field_history_weight = t[2].slider(
            "Field history weight", 0.0, 1.0,
            float(cfg.hca.field_history_weight),
            help="Extra HCA-internal blend of the energy field with previous "
                 "iterations (LS-TaSC's multi-iteration weighted sum). 1.0 = "
                 "off — the shared history weight above already blends "
                 "iterations.")

    arch = st.columns(2)
    aopt.archive_iterations = arch[0].checkbox(
        "Archive each iteration", value=aopt.archive_iterations,
        key=f"arch_{opt_name}",
        help="Copy each iteration's deck, animation and listing into "
             "<run_folder>/iter_NNNN/ before solve/ is recycled.")
    aopt.archive_restart = arch[1].checkbox(
        "…incl. restart (~345 MB/iter)", value=aopt.archive_restart,
        key=f"archr_{opt_name}",
        help="Also copy the restart file, preserving the full solver state for "
             "every iteration. Applies only when 'Archive each iteration' is on.")

    st.subheader("Manufacturing (AM) constraints")
    st.caption("Printability constraints applied to the design each iteration "
               "(after the optimiser update). All default OFF — leave them off "
               "for an unconstrained run.")
    mfg = cfg.manufacturing
    mfg.min_member_layers = int(st.number_input(
        "Minimum member size (erode/dilate hops)", value=int(mfg.min_member_layers),
        min_value=0, step=1,
        help="Morphological open removing thin features / single-element slivers. "
             "0 = off; 1–2 is typical."))

    st.markdown("**Symmetry planes** — force the design symmetric across a plane "
                "(either side alive ⇒ both alive).")
    existing_sym = {str(p.get("axis", "")).lower(): float(p.get("offset", 0.0))
                    for p in (mfg.symmetry_planes or [])}
    planes: list[dict] = []
    for col, ax in zip(st.columns(3), ("x", "y", "z")):
        on = col.checkbox(f"Mirror {ax.upper()}", value=ax in existing_sym,
                          key=f"sym_{ax}")
        off = col.number_input(f"{ax.upper()} plane offset",
                               value=existing_sym.get(ax, 0.0), step=0.5,
                               key=f"sym_off_{ax}", disabled=not on)
        if on:
            planes.append({"axis": ax, "offset": float(off)})
    mfg.symmetry_planes = planes

    st.markdown("**Overhang / self-support** — forbid material unsupported below "
                "the critical angle along the build direction.")
    ov = st.columns(2)
    _axis_vec = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
    _cur = "off"
    if mfg.build_direction is not None:
        _t = [round(float(v), 6) for v in mfg.build_direction]
        _cur = next((a for a, v in _axis_vec.items() if v == _t), "z")
    _opts = ["off", "x", "y", "z"]
    _sel = ov[0].selectbox("Build direction", _opts, index=_opts.index(_cur),
                           help="Up/growth axis of the print. 'off' disables the "
                                "overhang constraint.")
    mfg.build_direction = None if _sel == "off" else _axis_vec[_sel]
    mfg.max_overhang_angle = float(ov[1].number_input(
        "Max overhang angle [deg]", value=float(mfg.max_overhang_angle),
        min_value=0.0, max_value=90.0, step=5.0,
        help="Cone half-angle from the build direction within which support must "
             "exist. ~45° is a common self-supporting limit; 0 = off."))

    st.subheader("Post-processing")
    st.markdown("**Animation → d3plot**")
    cfg.d3plot.enabled = st.checkbox(
        "Convert final animation to d3plot when the run finishes",
        value=cfg.d3plot.enabled,
        help="Runs the external Vortex-Radioss Anim_to_D3plot tool on the final "
             "design's OpenRadioss animation, writing <work>/d3plot/<stem>.d3plot "
             "(one per load case). Best-effort — a missing tool or dependency "
             "never fails the run.")
    dc = st.columns(2)
    cfg.d3plot.tool_root = dc[0].text_input(
        "Vortex-Radioss tool root", cfg.d3plot.tool_root,
        help="Folder containing the `vortex_radioss` package "
             "(the openradioss_tools repo root). Blank → the OROPT_VORTEX_ROOT "
             "environment variable.")
    cfg.d3plot.python_exe = dc[1].text_input(
        "Converter Python (optional)", cfg.d3plot.python_exe,
        placeholder=str(Path(cfg.d3plot.tool_root or ".") / ".venv" / "Scripts" / "python.exe"),
        help="Interpreter with lasso-python + tqdm installed. Blank → the tool "
             "root's .venv if present, else the oropt interpreter.")

    st.markdown("**Surface smoothing**")
    cfg.smooth.enabled = st.checkbox(
        "Smooth the final geometry surface when the run finishes",
        value=cfg.smooth.enabled,
        help="Extracts and smooths the design surface, writing "
             "<work>/topology_smoothed.<ext> — a clean mesh for CAD / 3D-print / "
             "review — plus one per iteration (topology_smoothed_iterNNNN.<ext>).")
    sc = st.columns(3)
    cfg.smooth.iterations = int(sc[0].number_input(
        "Smoothing passes", value=int(cfg.smooth.iterations), min_value=0, step=5))
    _methods = ["taubin", "laplacian"]
    cfg.smooth.method = sc[1].selectbox(
        "Method", _methods,
        index=_methods.index(cfg.smooth.method) if cfg.smooth.method in _methods else 0,
        help="Taubin preserves volume; Laplacian smooths more but shrinks.")
    _fmts = ["stl", "vtp", "both"]
    cfg.smooth.output_format = sc[2].selectbox(
        "Output format", _fmts,
        index=_fmts.index(cfg.smooth.output_format)
        if cfg.smooth.output_format in _fmts else 0)

    st.markdown("**Evolution animation**")
    cfg.animate.enabled = st.checkbox(
        "Build a topology-evolution GIF when the run finishes",
        value=cfg.animate.enabled,
        help="Renders the per-iteration smoothed surfaces (raw snapshots as a "
             "fallback) from one fixed camera into <work>/topology_evolution.gif — "
             "a quick visual of the optimisation. Best-effort; never fails the run.")

    render_camera_settings(cfg.animate, key_prefix="con")
    color_ok, bg_ok = render_appearance_settings(cfg.animate, key_prefix="con")

    if st.button("💾 Save config", disabled=not (color_ok and bg_ok)):
        cfg.to_yaml(cfg_path)
        st.success(f"Saved to {cfg_path}")


# ---- Monitor tab -----------------------------------------------------------
def render_monitor_tab(cfg: Config, work: Path, refresh_s: int) -> None:
    @st.fragment(run_every=refresh_s)
    def monitor():
        st.caption(f"📂 monitoring `{work}`")     # which run folder this view reads
        status = st_io.read_status(work)
        if status is None:
            st.info("No run here yet. Configure on the other tabs, then "
                    "➕ Add to queue and ▶ Start queue.")
            return

        feas = "✅ feasible" if status.feasible else "⚠️ infeasible"
        st.markdown(f"**{status.state.upper()}** · iter {status.iteration}/"
                    f"{status.max_iter} · {feas} · {status.message}")
        k = st.columns(4)
        k[0].metric("Volume fraction", f"{status.volume_fraction:.3f}")
        k[1].metric("σ_max [MPa]", f"{status.sigma_max:.1f}",
                    f"limit {_fmt_limit(status.sigma_allow, '{:.0f}')}",
                    delta_color="off")
        k[2].metric("disp [mm]", f"{status.disp:.4f}",
                    f"limit {_fmt_limit(status.d_allow, '{:.2f}')}",
                    delta_color="off")
        eta = status.eta_s / 60 if status.eta_s == status.eta_s else float("nan")
        k[3].metric("ETA [min]", "—" if eta != eta else f"{eta:.0f}",
                    f"elapsed {status.elapsed_s/60:.0f} min", delta_color="off")
        cases = getattr(status, "cases", None) or []
        if len(cases) > 1:
            st.caption(f"σ_max and disp above are the **worst across "
                       f"{len(cases)} load cases**, each shown with that case's "
                       "own limit; the design is feasible only when every case is "
                       "(below). Each case's animation is under `solve/case_<i>/`.")
            cdf = pd.DataFrame([{
                "case": c["name"],
                "σ_max [MPa]": c["sigma_max"],
                "σ_allow [MPa]": _fmt_limit(c["sigma_allow"], "{:.0f}"),
                "disp [mm]": c["disp"],
                "d_allow [mm]": _fmt_limit(c["d_allow"], "{:.2f}"),
                "feasible": "✅" if c["feasible"] else "⚠️",
            } for c in cases])
            st.dataframe(cdf, hide_index=True, use_container_width=True)
        if getattr(status, "stress_excluded_elems", 0):
            st.caption(f"σ_max ignores **{status.stress_excluded_elems} elements** "
                       "in the configured stress-exclusion region(s).")
        if getattr(status, "elements_candidate", 0):
            st.caption(
                f"Growth boxes: **{getattr(status, 'elements_grown', 0)}** of "
                f"**{status.elements_candidate}** candidate elements grown "
                "(material added beyond the original part).")

        hist = st_io.read_history(work)
        if hist:
            df = pd.DataFrame(hist)
            for c in ("iteration", "volume_fraction", "sigma_max", "disp"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.set_index("iteration")
            cols = st.columns(3)
            cols[0].caption("Volume fraction"); cols[0].line_chart(df[["volume_fraction"]])
            d2 = df[["sigma_max"]].copy(); d2["σ_allow"] = status.sigma_allow
            cols[1].caption("σ_max vs limit"); cols[1].line_chart(d2)
            d3 = df[["disp"]].copy(); d3["d_allow"] = status.d_allow
            cols[2].caption("disp vs limit"); cols[2].line_chart(d3)

    monitor()

    # The 3D topology preview lives in its OWN fragment, deliberately *without*
    # run_every: the Monitor's periodic auto-refresh reruns only the metrics/queue
    # fragments, so it never re-renders this one — and the panel pane keeps whatever
    # camera angle the user set instead of snapping back to isometric every refresh.
    # 🔄 Update view re-reads the latest geometry on demand (which does reset it).
    @st.fragment
    def topology_view():
        topo = work / st_io.TOPOLOGY
        if not topo.exists():
            return
        head = st.columns([3, 1])
        head[0].subheader("Current topology")
        # The button's only effect is to rerun this fragment, which re-reads the
        # geometry below; its return value is intentionally unused.
        head[1].button(
            "🔄 Update view", key="topo_refresh", width="stretch",
            help="Re-read the latest geometry. This resets the camera angle; "
                 "otherwise the view keeps your angle across the auto-refresh.")
        try:
            import pyvista as pv
            from stpyvista import stpyvista
            grid = pv.read(str(topo))
            scal = "sensitivity" if "sensitivity" in grid.cell_data else None
            pl = pv.Plotter(window_size=[700, 450], off_screen=True)
            pl.add_mesh(grid, scalars=scal, cmap="viridis", show_edges=False)
            pl.view_isometric(); pl.background_color = "white"
            # backend="panel" renders in-process. The default "trame" backend
            # exports the scene from a multiprocessing.Process whose spawned child
            # dies in DuplicateHandle under `streamlit run` on Windows (and would
            # then hang the parent on queue.get()).
            stpyvista(pl, backend="panel", key="topo")
        except Exception as exc:  # noqa: BLE001
            st.caption(f"(3D view unavailable: {exc})")

    topology_view()

    # Live read-only view of the serial run queue, on the same refresh tick (so a
    # queued run's progress shows here even before its config is selected above).
    @st.fragment(run_every=refresh_s)
    def queue_monitor():
        q = qs.load_queue(QUEUE_PATH)
        if not q.entries:
            return
        st.markdown("---")
        alive = bool(q.runner_pid) and st_io.pid_alive(q.runner_pid)
        c = qs.counts(q)
        state = ("🟢 runner active" if alive
                 else "⏸ paused" if q.paused else "⚪ idle")
        line = (f"**Run queue** — {state} · ⏳ {c['pending']} · 🟢 {c['running']} "
                f"· ✅ {c['done']} · ❌ {c['failed']}")
        if c["skipped"]:
            line += f" · ⤳ {c['skipped']}"
        st.markdown(line)
        for e in q.entries:
            st.caption(f"{QUEUE_BADGE.get(e.state, '•')} {Path(e.config).name}"
                       + (f" — {e.message}" if e.message else ""))

    queue_monitor()


# ---- Re-animate tab --------------------------------------------------------
def render_reanimate_tab(cfg: Config, default_folder: Path) -> None:
    """Re-render the topology-evolution GIF for an *existing* run with fresh
    camera / resolution / playback settings — no re-solving.

    A spin-off of the post-run animation: it reads the same per-iteration
    ``topology_smoothed_iter*`` (or raw ``topology_iter*.vtu``) snapshots a finished
    run already wrote and re-encodes them via :func:`oropt.animate.make_animation`,
    writing a **new** GIF into the run folder so the run's original is preserved
    unless its name is reused. The render runs in :mod:`oropt.animate`'s isolated
    off-screen subprocess (crash containment), so it is safe to drive synchronously
    from here.
    """
    st.subheader("Re-animate a finished run")
    st.caption(
        "Re-render the topology-evolution GIF from an **existing** run's "
        "per-iteration surfaces with new settings (camera angle, resolution, "
        "colours, playback) — without re-running the optimisation. Writes a new "
        "GIF into the run folder, leaving the original `topology_evolution.gif` "
        "untouched unless you reuse its name.")

    folder = Path(st.text_input(
        "Run folder", str(default_folder), key="reanim_folder",
        help="A finished run's output folder — the one holding the per-iteration "
             "topology_smoothed_iter*/topology_iter* snapshots. Defaults to the "
             "currently selected run."))
    n_frames, src = frame_count(folder) if str(folder).strip() else (0, "")
    ready = bool(str(folder).strip()) and folder.exists() and n_frames >= 2
    if not str(folder).strip() or not folder.exists():
        st.warning("Run folder not found.")
    elif n_frames < 2:
        st.warning(
            f"Need ≥2 per-iteration snapshots to animate (found {n_frames}). Run "
            "with surface smoothing or per-iteration snapshots enabled first.")
    else:
        st.success(f"{n_frames} frames found ({src} surfaces).")

    out_name = st.text_input(
        "Output GIF name", "topology_evolution_reanim.gif", key="reanim_out",
        help="Written into the run folder. Use `topology_evolution.gif` to "
             "overwrite the run's original (e.g. to refresh the one the report "
             "embeds).").strip()

    # Independent settings, seeded from the config's animate block so the tool
    # starts from the run's configured look — but kept on a SEPARATE AnimateOpts so
    # editing here never leaks into the cfg the queue would enqueue/save.
    opts = dataclasses.replace(cfg.animate)
    render_camera_settings(opts, key_prefix="reanim")
    color_ok, bg_ok = render_appearance_settings(opts, key_prefix="reanim")

    name_ok = out_name.endswith(".gif")
    if out_name and not name_ok:
        st.caption("Output name must end in `.gif`.")
    if st.button("🎬 Generate animation", type="primary",
                 disabled=not (ready and name_ok and color_ok and bg_ok)):
        opts.enabled = True
        tmp_cfg = Config()
        tmp_cfg.animate = opts                 # make_animation reads cfg.animate
        logs: list[str] = []
        with st.spinner(f"Rendering {n_frames} frames at "
                        f"{opts.window_w}×{opts.window_h}…"):
            out = make_animation(tmp_cfg, folder, logs.append, out_name=out_name)
        if logs:
            with st.expander("Render log", expanded=out is None):
                for line in logs:
                    st.text(line)
        if out is not None and out.is_file():
            st.success(f"Wrote `{out}`")
            st.image(str(out), caption=out.name)
            try:
                st.download_button("⬇ Download GIF", out.read_bytes(),
                                   file_name=out.name, mime="image/gif",
                                   key="reanim_dl")
            except OSError:
                pass
        else:
            st.error("No GIF produced — see the render log above.")


# ---- Queue tab -------------------------------------------------------------
def render_queue_tab(cfg_path: Path) -> None:
    st.subheader("Run queue (serial)")
    st.caption(
        "Enqueue several runs; a detached **queue runner** executes them strictly "
        "one at a time (never two solver processes at once) and starts the next "
        "automatically when the current finishes. The queue lives on disk and "
        "keeps draining after you close the browser — just like a single run. "
        "`is_running` stays the source of truth, so a run already live for a "
        "config is waited on, never double-launched. Each entry runs from a "
        "**frozen copy** of the config taken when you added it (saved in the "
        "model's case directory under `queue_configs/`), so later edits to the "
        "working config never change a run already queued.")

    queue = qs.load_queue(QUEUE_PATH)          # re-read (sidebar may have mutated)
    runner_alive = bool(queue.runner_pid) and st_io.pid_alive(queue.runner_pid)

    # ---- add a config -----------------------------------------------------
    add_cols = st.columns([5, 1, 1])
    new_cfg = add_cols[0].text_input("Config to enqueue", str(cfg_path),
                                     key="queue_add_path")
    add_resume = add_cols[1].checkbox("resume", value=False, key="queue_add_resume",
                                      help="Enqueue this run with --resume.")
    if add_cols[2].button("➕ Add", width="stretch"):
        p = Path(new_cfg)
        if not p.exists():
            st.warning(f"Config not found: {new_cfg}")
        else:
            # Freeze a snapshot now (in the case dir) and queue *that* — later edits
            # to the source config can't change a run already in the queue.
            snap = snapshot_into_case_dir(p)
            qs.mutate(QUEUE_PATH, lambda q: qs.add(
                q, snap, resume=add_resume,
                work_dir=qs.resolve_work_dir(snap, PROJECT_ROOT)))
            st.rerun()

    dups = qs.duplicate_work_dirs(queue)
    if dups:
        st.warning(
            "Multiple queued runs share a run/output folder, so they would "
            "overwrite each other's status and results (they still run serially, "
            "never at once). Give each its own `work_dir` / case directory before "
            "starting:\n" + "\n".join(f"- `{d}`" for d in sorted(dups)))

    # ---- queue-wide controls ----------------------------------------------
    ctl = st.columns(4)
    if ctl[0].button("▶ Start queue", width="stretch",
                     disabled=runner_alive or qs.counts(queue)["pending"] == 0):
        qs.mutate(QUEUE_PATH, lambda q: qs.set_paused(q, False))
        queue_runner.spawn_runner(QUEUE_PATH, PROJECT_ROOT)
        st.rerun()
    if ctl[1].button("⏸ Pause queue", width="stretch", disabled=not runner_alive,
                     help="Stop after the current run finishes; doesn't kill it."):
        qs.mutate(QUEUE_PATH, lambda q: qs.set_paused(q, True))
        st.rerun()
    if ctl[2].button("🧹 Clear finished", width="stretch"):
        qs.mutate(QUEUE_PATH, qs.clear_finished)
        st.rerun()
    if ctl[3].button("🗑 Clear all", width="stretch",
                     help="Remove every entry except one currently running."):
        qs.mutate(QUEUE_PATH, qs.clear_all)
        st.rerun()

    # ---- the entries ------------------------------------------------------
    if not queue.entries:
        st.info("Queue is empty. Add a config above, or use the sidebar's "
                "“➕ Add current config to queue”.")
    last = len(queue.entries) - 1
    for i, e in enumerate(queue.entries):
        row = st.columns([5, 2, 1, 1, 1])
        label = Path(e.config).name + (" · resume" if e.resume else "")
        detail = f"`{e.config}`"
        if e.work_dir:
            detail += f"  \n↳ `{e.work_dir}`"
        if e.message:
            detail += f"  \n_{e.message}_"
        row[0].markdown(f"**{label}**  \n{detail}")
        row[1].markdown(QUEUE_BADGE.get(e.state, e.state))
        # Reorder / remove only — entries run from an immutable config snapshot
        # (see snapshot_config), so there is nothing to edit in place anymore.
        # Capture e.id via default arg so the lambda binds this row, not the last.
        if row[2].button("⬆", key=f"q_up_{e.id}", disabled=i == 0,
                         help="Run earlier"):
            qs.mutate(QUEUE_PATH, lambda q, _id=e.id: qs.move(q, _id, -1))
            st.rerun()
        if row[3].button("⬇", key=f"q_down_{e.id}", disabled=i == last,
                         help="Run later"):
            qs.mutate(QUEUE_PATH, lambda q, _id=e.id: qs.move(q, _id, +1))
            st.rerun()
        if row[4].button("✖", key=f"q_rm_{e.id}", help="Remove from queue"):
            qs.mutate(QUEUE_PATH, lambda q, _id=e.id: qs.remove(q, _id))
            st.rerun()


# ---- sidebar: config selection --------------------------------------------
st.sidebar.title("oropt")
st.sidebar.caption("OpenRadioss-coupled BESO topology optimisation")
cfg_path = Path(st.sidebar.text_input("Config file", str(DEFAULT_CFG)))
if not cfg_path.exists():
    st.sidebar.error("Config not found.")
    st.stop()
cfg_raw = Config.read_yaml_dict(cfg_path)   # kept for unrecognised-key validation
cfg = Config.from_dict(cfg_raw)
work = Path(cfg.run_folder())          # work_dir, or the case dir when blank
if not work.is_absolute():
    work = PROJECT_ROOT / work

# Run state follows whatever run is actually live — the selected config's own
# folder, or a queued run in its (possibly de-duplicated) reserved folder — so the
# sidebar and Monitor stay in sync with the queue instead of showing idle.
queue = qs.load_queue(QUEUE_PATH)
active = find_active_run(work, queue)
running = active is not None
live_dir = active[0] if active else work    # the folder the Monitor should follow

if active is None:
    run_state = "⚪ idle"
elif active[0] == work:
    run_state = "🟢 running"
else:
    run_state = f"🟢 running — {active[1]} (via queue)"
st.sidebar.markdown(f"**Run state:** {run_state}")

# Fail-fast config check: same validation the headless run does, surfaced before
# launch. Hard errors block enqueuing the config (a queued run could not or must
# not start anyway).
problems = check_config(cfg, raw=cfg_raw)
cfg_errors = has_errors(problems)
if problems:
    n_err = sum(1 for p in problems if p.severity == ERROR)
    with st.sidebar.expander(
            f"⚠ Config check: {n_err} error(s), {len(problems) - n_err} warning(s)",
            expanded=cfg_errors):
        for p in problems:
            (st.error if p.severity == ERROR else st.warning)(p.message)
        if cfg_errors:
            st.caption("Fix the errors above to enable ➕ Add to queue.")

# Ad-hoc single-run launching was removed: a run is started *only* via the queue
# (➕ Add current config to queue → ▶ Start queue, below). Stop / Resume / Force
# kill act on whatever Run state shows as live (a queued run, or one launched
# earlier), so they stay; they target the live run's folder and don't touch `cfg`.
c1, c2 = st.sidebar.columns(2)
if c1.button("⏸ Stop", disabled=not running, width="stretch"):
    request_stop(live_dir)
    st.sidebar.info("Stop requested (after current solve).")
if c2.button("↻ Resume", disabled=running, width="stretch"):
    launch_run(cfg_path, resume=True)   # resumes the selected config's run from checkpoint
    st.sidebar.success("Resumed.")
if st.sidebar.button("⏹ Force kill", disabled=not running):
    force_kill(live_dir)

refresh_s = int(st.sidebar.number_input(
    "Refresh interval (s)", min_value=120, max_value=300, value=120, step=5,
    help="How often the Monitor tab re-reads the run's status files "
         "(default 120s; increase up to 300s to ease the polling load)."))

# ---- sidebar: run queue (serial) — the only way to start a run ------------
# Add/start/pause; full management lives in the 🧮 Queue tab. The detached serial
# runner launches one run at a time. (`queue` was loaded above for the run-state
# sync.)
runner_alive = bool(queue.runner_pid) and st_io.pid_alive(queue.runner_pid)
qcounts = qs.counts(queue)
st.sidebar.markdown("---")
queue_state = ("🟢 runner active" if runner_alive
               else "⏸ paused" if queue.paused else "⚪ idle")
st.sidebar.markdown(f"**Run queue:** {qcounts['pending']} pending · {queue_state}")
# Captured here but enqueued *after* the tabs populate `cfg`: a queued run reads the
# on-disk config at run time, so the current edits must be saved first or the queued
# run ignores them (e.g. the chosen optimiser). Blocked on config errors.
add_to_queue_clicked = st.sidebar.button("➕ Add current config to queue",
                                         width="stretch", disabled=cfg_errors)
qcol = st.sidebar.columns(2)
if qcol[0].button("▶ Start queue", width="stretch",
                  disabled=runner_alive or qcounts["pending"] == 0):
    qs.mutate(QUEUE_PATH, lambda q: qs.set_paused(q, False))
    queue_runner.spawn_runner(QUEUE_PATH, PROJECT_ROOT)
    st.rerun()
if qcol[1].button("⏸ Pause queue", width="stretch", disabled=not runner_alive):
    qs.mutate(QUEUE_PATH, lambda q: qs.set_paused(q, True))
    st.rerun()

# ---- tabs (each body lives in a render_*_tab function above) ---------------
tab_in, tab_lc, tab_con, tab_mon, tab_anim, tab_q = st.tabs(
    ["📥 Input", "🔀 Load cases", "🎚 Optimiser / Output", "📊 Monitor",
     "🎬 Re-animate", "🧮 Queue"])
with tab_in:
    render_input_tab(cfg)
with tab_lc:
    render_load_cases_tab(cfg, cfg_path)
with tab_con:
    render_constraints_tab(cfg, cfg_path)
with tab_mon:
    render_monitor_tab(cfg, live_dir, refresh_s)   # follow the live run's folder
with tab_anim:
    render_reanimate_tab(cfg, live_dir)            # re-render an existing run's GIF
with tab_q:
    render_queue_tab(cfg_path)

# ---- deferred enqueue action -----------------------------------------------
# Handled here, now that every tab above has written its widgets back into `cfg`,
# so enqueuing persists the *on-screen* config (incl. the selected optimiser)
# rather than the stale on-disk one. The button renders in the sidebar above; only
# its cfg-dependent effect is deferred to this point. We save the working config,
# then freeze an immutable snapshot and queue *that* — so later edits to the
# working config can't change a run already in the queue (it runs what you saw).
if add_to_queue_clicked:
    cfg.to_yaml(cfg_path)
    snap = snapshot_into_case_dir(cfg_path)        # frozen copy in the case dir
    qs.mutate(QUEUE_PATH, lambda q: qs.add(
        q, snap, resume=False,
        work_dir=qs.resolve_work_dir(snap, PROJECT_ROOT)))
    st.rerun()
