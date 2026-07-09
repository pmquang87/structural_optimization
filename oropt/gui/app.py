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
from oropt.deck import Deck
from oropt.gui import queue_store as qs
from oropt.gui.boxes import (BOX_COLUMNS, FRAME_COLUMNS, POINT_COLUMNS,
                             apply_frame_records, apply_point_records,
                             growth_boxes_from_records, records_from_frames,
                             records_from_growth_boxes, records_from_points)
from oropt.gui.cases import (CASE_COLUMNS, load_cases_from_records,
                             records_from_load_cases)
from oropt.gui.colors import COMMON_COLORS, OTHER, is_valid_color
from oropt.gui import growthprep
from oropt.gui.runstate import find_active_run
from oropt.growthmesh import GROWTH_MESH_DIRNAME, point_config_at
from oropt.loop import copy_iter0, preview_growth_boxes
from oropt.mesh import Mesh, overlay_primitives
from oropt.report import write_report
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


def _int_ids(text: str, label: str) -> list[int]:
    """Tolerant comma-separated id parse for the free-text id fields.

    A stray non-numeric token (``99999999;``, ``abc``) used to raise an uncaught
    ``ValueError`` and replace the whole tab render with a traceback -- and abort
    the script before the deferred Save ran. Flag bad tokens in a caption and
    keep the good ones instead (the table-based editors in ``cases.py`` /
    ``boxes.py`` already tolerate bad cells this way).
    """
    ids: list[int] = []
    bad: list[str] = []
    for tok in text.replace(";", ",").replace(" ", "").split(","):
        if not tok:
            continue
        try:
            ids.append(int(tok))
        except ValueError:
            bad.append(tok)
    if bad:
        st.caption(f"⚠ {label}: ignored non-numeric id(s) "
                   + ", ".join(repr(b) for b in bad))
    return ids


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

    n_cases = len(cases)
    cfg.run.solver_concurrency = int(st.number_input(
        "Concurrent solvers", value=min(int(cfg.run.solver_concurrency),
                                        max(1, n_cases)),
        min_value=1, max_value=max(1, n_cases), step=1, key="run_concurrency",
        disabled=n_cases <= 1,
        help="How many load-case solves to run at once each iteration (default 1 "
             "= sequential). On a strong PC set >1 to solve several load cases "
             "simultaneously; each solver still uses nt threads (and MPI with np), "
             "so keep concurrency × nt ≤ cores and watch RAM. Capped at the number "
             "of load cases; a single-case run is unaffected."))
    if n_cases <= 1:
        st.caption("Only one load case — concurrency has nothing to parallelise. "
                   "Add load cases on the 🔀 tab to solve several at once.")

    cfg.run.max_wall_hours = float(st.number_input(
        "Wall-clock budget [h] (0 = unlimited)",
        value=float(cfg.run.max_wall_hours), min_value=0.0, step=0.5,
        key="run_wall_budget",
        help="Whole-run wall-clock budget. Checked between iterations (a solve "
             "in flight is never cut short): once exceeded the run stops "
             "cleanly — state 'stopped', checkpoint kept, post-run steps "
             "(d3plot/smoothing/animation/report) still run — instead of being "
             "killed mid-solve by a shared-machine or cluster session limit. "
             "Resume later with --resume."))

    with st.expander("⏩ Seed iteration 0 from another run "
                     "(skip the first full-volume solve)"):
        cfg.run.reuse_iter0 = st.checkbox(
            "Reuse an iter_0000 already in this run folder", value=cfg.run.reuse_iter0,
            key="reuse_iter0",
            help="Iteration 0 solves the initial full-volume design — the most "
                 "expensive solve — and is identical across runs with the same "
                 "initial deck. If a matching iter_0000 is present, reuse its solve "
                 "instead of re-running it. Guarded by a byte-compare of the starter "
                 "deck, so a copy from a different model is refused and solved fresh.")
        st.caption(f"Target run folder: `{cfg.run_folder()}`")
        src = st.text_input(
            "Copy iter_0000 from another run's folder", key="copy_iter0_src",
            help="Path to an old run folder that contains an iter_0000. Its whole "
                 "iter_0000 (single- or multi-case) is copied here; the run then "
                 "validates and reuses it at iteration 0.")
        overwrite = st.checkbox("Overwrite an existing iter_0000 here",
                                key="copy_iter0_overwrite")
        if st.button("📋 Copy iter_0000 here", disabled=not src.strip(),
                     key="copy_iter0_btn"):
            ok, msg = copy_iter0(src.strip(), cfg.run_folder(), overwrite=overwrite)
            (st.success if ok else st.error)(msg)


# ---- Load cases tab --------------------------------------------------------
def render_load_cases_tab(cfg: Config, cfg_path: Path) -> None:
    st.subheader("Load cases")
    st.caption(
        "Define the load case(s) the part is optimised against — the **single "
        "source of truth** for each deck's stem, the constrained displacement "
        "node(s) and the σ/d limits. A single-load run is just **one** row. Add "
        "more rows to optimise a **weighted-sum compliance** over several loads "
        "(the linkage pulled in different directions); each row is a separate deck "
        "pair in the case directory that shares the same mesh — only its load "
        "differs. *Deck stem* is required; leave *σ_allow* blank to leave stress "
        "unconstrained. Constrain the displacement at **several nodes** by listing "
        "`node:limit` pairs separated by `;` (e.g. `10021367:1.0; 10021400:2.0`); "
        "a bare `node` tracks it without a limit — the design is feasible only "
        "when every one holds.")
    if not cfg.load_cases:               # always offer at least one row to fill in
        cfg.load_cases = [LoadCase(name="case", weight=1.0, sigma_allow=250.0)]
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
            "disp_constraints": st.column_config.TextColumn(
                "Disp constraints (node:limit)",
                help="Per-node displacement limits as `node:limit` pairs "
                     "separated by `;` (e.g. `10021367:1.0; 10021400:2.0`). A "
                     "bare `node` tracks it unconstrained. Blank → none tracked."),
            "sigma_allow": st.column_config.NumberColumn(
                "σ_allow [MPa]", min_value=0.0, step=1.0,
                help="Per-case von-Mises stress limit. Blank → no stress limit."),
            "fast_mode": st.column_config.CheckboxColumn(
                "Fast mode", default=False,
                help="Screen this case with a validated TIED LINEAR solve "
                     "(~35× faster) instead of the full nonlinear one. The "
                     "load/support contact patches are auto-tied so the linear "
                     "step has a real load+support path. A ranking/flagging "
                     "screen, not a certifying stress — it reads ~14% below the "
                     "nonlinear peak, so set σ_allow (used exactly as in normal "
                     "mode) to account for that bias (≈254 MPa screened near this "
                     "deck's 292 yield). Default off."),
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


# ---- shared evolution-animation camera/playback widgets --------------------
def render_camera_settings(opts: AnimateOpts, key_prefix: str) -> None:
    """Render the evolution-animation camera & playback widgets into *opts*.

    Shared by the Optimizer / Output tab (settings the post-run animation will
    use) and the 🛠 Re-postprocessing tab (settings for an on-demand re-render), so
    the two can never drift. Every widget writes straight back into the passed *opts*
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

    Shared by the Optimizer / Output tab (the look the post-run animation will use)
    and the 🛠 Re-postprocessing tab (an on-demand re-render), so the two can never
    drift — the same split as :func:`render_camera_settings`. Returns ``(color_ok,
    bg_ok)`` from
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


# ---- Optimizer / Output tab ------------------------------------------------
def render_constraints_tab(cfg: Config, cfg_path: Path) -> None:
    st.caption("Feasibility limits (σ_allow / d_allow) are now set **per load "
               "case** on the 🔀 Load cases tab.")
    st.subheader("Optimizer")
    _opts = ["beso", "levelset", "tobs", "hca"]
    _opt_labels = {"beso": "BESO — bi-directional element removal",
                   "levelset": "Level-set — smoother boundaries",
                   "tobs": "TOBS — binary ILP flips (Sivapuram & Picelli 2018)",
                   "hca": "HCA — hybrid cellular automata (LS-TaSC-style)"}
    opt_name = st.selectbox(
        "Topology optimizer", _opts,
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
    cfg.model.freeze_group_ids = _int_ids(fg, "freeze group ids")
    cfg.model.freeze_node_ids = _int_ids(fn, "freeze node ids")
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
    cfg.model.stress_exclude_group_ids = _int_ids(sg, "ignore-stress group ids")
    cfg.model.stress_exclude_node_ids = _int_ids(sn, "ignore-stress node ids")

    st.subheader("Growth regions — add material")
    cfg.model.growth_enabled = st.checkbox(
        "Enable growth regions",
        value=getattr(cfg.model, "growth_enabled", True),
        help="On: design elements whose centroid lies in a region below "
             "start VOID and the optimiser may add material there. Off: the "
             "regions are kept in the config (coordinates preserved) but "
             "ignored, so the run solves the part as-is. Leftover regions "
             "from an earlier run stay dormant until you tick this -- they "
             "can't silently drive a run.")
    st.caption(
        "Regions (like the LS-DYNA `*DEFINE_BOX` / Radioss `/BOX/RECTA` family) "
        "marking **candidate growth material**: design elements whose centroid "
        "lies inside a region start the run **void**, and the optimiser may *add* "
        "them where the load path wants — so the design can grow material where "
        "the original part had none. Pick a **shape** per row and fill only that "
        "shape's coordinates: `box` uses the six `*_min`/`*_max` bounds, `sphere` "
        "uses `cx,cy,cz` + `radius`, `cylinder` uses the two axis end-points "
        "`x1,y1,z1`–`x2,y2,z2` + `radius`, `polyhedron` takes an arbitrary node "
        "list (all coordinates explicit) in the *Polyhedron points* table below — "
        "its region is the points' convex hull. The region volume must contain "
        "candidate elements: **pre-mesh** it into the design part (same "
        "`/TETRA4` block, node-conformal interface, node ids ≥ design_node_min) "
        "— or **generate** it with the ⚙️ *Generate growth mesh* button below; "
        "a region over unmeshed space is an error at "
        "run start. Multiple regions act as a union; volume fractions are then "
        "relative to the enlarged (part + regions) space. A region **may overlap "
        "the part**: with **Carve part** unchecked (default) the original part "
        "stays intact and only expansion elements start void — so a region can "
        "be drawn generously into the part to guarantee the new material "
        "attaches with no gap (needs the original-part element-id boundary, "
        "recorded by the ⚙️ growth-mesh step; without it the region voids "
        "everything inside). Check it for deliberate carve-and-regrow: the "
        "overlapped original elements start void too. A row may reference a "
        "`/BOX/{RECTA,SPHER,CYLIN}` card in the deck by **Deck /BOX id** instead of "
        "coordinates; oriented (local-frame) boxes are set below.")
    if cfg.model.growth_enabled:
        # Capture the oriented-frame and polyhedron-point rows BEFORE the main table
        # below overwrites cfg.model.growth_boxes (which carries the loaded frames/
        # points): those editors are seeded from here and their results re-applied by
        # name, so frames and points survive the main table's shape/coordinate
        # round-trip instead of being silently dropped.
        frame_records = records_from_frames(cfg.model.growth_boxes)
        point_records = records_from_points(cfg.model.growth_boxes)
        _num_cols = [c for c in BOX_COLUMNS
                     if c not in ("name", "shape", "carve", "deck_box_id")]
        gb_df = pd.DataFrame(records_from_growth_boxes(cfg.model.growth_boxes),
                             columns=BOX_COLUMNS)
        gb_edited = st.data_editor(
            gb_df, num_rows="dynamic", width="stretch",
            key="growth_boxes_editor", column_config={
                "name": st.column_config.TextColumn(
                    "Name", help="Label for run-log / validation messages, and how a "
                                 "frame below is matched to a box."),
                "shape": st.column_config.SelectboxColumn(
                    "Shape", options=["box", "sphere", "cylinder", "polyhedron"],
                    default="box",
                    help="box = two corners; sphere = centre+radius; "
                         "cylinder = two axis end-points + radius; polyhedron = "
                         "convex hull of the node list in the points table below."),
                "carve": st.column_config.CheckboxColumn(
                    "Carve part", default=False,
                    help="Unchecked (default): the original part stays alive; "
                         "only expansion elements (ids above the original-part "
                         "boundary below) start void. Checked: original-part "
                         "elements inside the region start void too — deliberate "
                         "carve-and-regrow."),
                "deck_box_id": st.column_config.NumberColumn(
                    "Deck /BOX id", format="%d", step=1,
                    help="Reference a /BOX/{RECTA,SPHER,CYLIN} card in the deck by id "
                         "instead of coordinates; resolved at run start. Leave blank "
                         "to use the coordinates in this row."),
                **{k: st.column_config.NumberColumn(
                    k, format="%.3f",
                    help="Coordinate in model units (e.g. mm); fill the columns "
                         "the row's shape needs.")
                   for k in _num_cols},
            })
        cfg.model.growth_boxes = growth_boxes_from_records(
            gb_edited.to_dict("records"))
        if cfg.model.growth_boxes:
            # Oriented boxes (advanced): a local frame turns the box bounds into a skew
            # system (LS-DYNA *DEFINE_BOX_LOCAL). Edited in its own name-keyed table so
            # the 3-vectors fit; applied back onto the matching box rows.
            with st.expander("Oriented box frames (advanced)"):
                st.caption(
                    "Give a **box** a local frame: an origin, a local +x direction "
                    "(`ax,ay,az`) and a vector in the local +xy plane (`bx,by,bz`) — "
                    "Gram-Schmidt-orthonormalised, so the box bounds are measured in "
                    "that skew system. Match a row to a box by **Region** (its Name); "
                    "leave a row blank for an axis-aligned box. Sphere/cylinder regions "
                    "can't be oriented.")
                fr_df = pd.DataFrame(frame_records, columns=FRAME_COLUMNS)
                fr_edited = st.data_editor(
                    fr_df, num_rows="dynamic", width="stretch",
                    key="growth_frames_editor", column_config={
                        "name": st.column_config.TextColumn(
                            "Region", help="Must match a box row's Name."),
                        **{k: st.column_config.NumberColumn(k, format="%.3f")
                           for k in FRAME_COLUMNS[1:]},
                    })
                cfg.model.growth_boxes = apply_frame_records(
                    cfg.model.growth_boxes, fr_edited.to_dict("records"))
            # Polyhedron node lists: N points of 3 coordinates don't fit the fixed
            # columns above, so each polyhedron region's nodes are edited here — one
            # x/y/z row per node, matched to its region by Name (the frames pattern).
            if point_records or any(b.shape_kind() == "polyhedron"
                                    for b in cfg.model.growth_boxes):
                with st.expander("Polyhedron points (one row per node)",
                                 expanded=True):
                    st.caption(
                        "Define a **polyhedron** region by its nodes: one row per "
                        "node, **all three coordinates explicit** (a row missing "
                        "any coordinate is dropped — nothing is defaulted or "
                        "inferred). Match rows to a region by **Region** (its "
                        "Name); give at least 4 non-coplanar points. The region is "
                        "the points' **convex hull** (an arbitrary warped 8-node "
                        "brick is the convex case; a non-convex point set is "
                        "treated as its hull).")
                    pt_df = pd.DataFrame(point_records, columns=POINT_COLUMNS)
                    pt_edited = st.data_editor(
                        pt_df, num_rows="dynamic", width="stretch",
                        key="growth_points_editor", column_config={
                            "name": st.column_config.TextColumn(
                                "Region", help="Must match a polyhedron row's Name."),
                            **{k: st.column_config.NumberColumn(k, format="%.3f")
                               for k in POINT_COLUMNS[1:]},
                        })
                    cfg.model.growth_boxes = apply_point_records(
                        cfg.model.growth_boxes, pt_edited.to_dict("records"))
            if any(not b.carve for b in cfg.model.growth_boxes):
                # Carve-off regions need the original/expansion element-id boundary;
                # the ⚙️ growth-mesh "use these decks" button records it, the 🔍
                # preview button auto-fills it from the deck, and this input covers
                # hand-pre-meshed decks (and shows what was recorded/derived). The
                # widget is session-state-driven (no value=) so the preview's
                # auto-fill — stashed on click, because this widget is already
                # instantiated by then — can update what it shows on its rerun.
                _auto = st.session_state.pop("growth_orig_elem_max_autofill", None)
                if _auto is not None:
                    st.session_state["growth_orig_elem_max"] = int(_auto)
                elif "growth_orig_elem_max" not in st.session_state:
                    st.session_state["growth_orig_elem_max"] = int(
                        cfg.model.growth_original_elem_max or 0)
                _thr = st.number_input(
                    "Original part: highest element id (for carve-off regions)",
                    min_value=0, step=1, format="%d",
                    key="growth_orig_elem_max",
                    help="Elements with an id up to this are the ORIGINAL part and "
                         "stay alive inside regions with Carve part unchecked; ids "
                         "above are expansion material and start void. The ⚙️ "
                         "growth-mesh step's *use these decks* button records it, "
                         "and 🔍 *Preview region element counts* auto-fills it with "
                         "the deck's highest design element id while it is unset. "
                         "0 = unset: nothing is identifiable as original, so "
                         "carve-off regions void everything inside (validation "
                         "warns).")
                cfg.model.growth_original_elem_max = int(_thr) or None
            st.info(f"{len(cfg.model.growth_boxes)} growth region(s): their elements "
                    "start void and may be grown into. With BESO, keep "
                    "`max_add_ratio` ≥ `evolution_rate` so back-off growth isn't "
                    "throttled (validation warns otherwise). The Monitor's 3D view "
                    "outlines each region so you can place coordinates visually.")
            _render_keepout(cfg)
            _render_growth_preview(cfg)
            _render_growth_mesh(cfg, cfg_path)
    elif cfg.model.growth_boxes:
        st.caption(
            f"🚫 Growth is off -- {len(cfg.model.growth_boxes)} "
            "region(s) kept in the config but ignored this run. Tick "
            "**Enable growth regions** above to edit or use them.")

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
    b = st.columns(5)
    aopt.backoff_gain = b[0].number_input(
        "Back-off gain", value=float(aopt.backoff_gain), min_value=0.0,
        step=0.5, format="%.2f", key=f"bogain_{opt_name}",
        help="0 = classic gate: any violation grows the target by one full "
             "evolution-rate step. > 0 = proportional: the growth step is "
             "ER·max(floor, min(gain·(v−1), cap)), so a 1 % violation triggers "
             "a nudge and a large one a capped surge. Size it so "
             "gain·(typical overshoot) ≈ 1, e.g. 10–20.")
    aopt.backoff_cap = b[1].number_input(
        "Back-off cap (×ER)", value=float(aopt.backoff_cap), min_value=0.1,
        step=0.5, format="%.1f", key=f"bocap_{opt_name}",
        help="Cap on the proportional growth step, in multiples of the "
             "evolution rate. Only used when the gain is > 0.")
    aopt.backoff_floor = b[2].number_input(
        "Back-off floor (×ER)", value=float(aopt.backoff_floor), min_value=0.0,
        step=0.05, format="%.2f", key=f"boflr_{opt_name}",
        help="Floor on the proportional back-step, in fractions of the "
             "evolution rate: a persistent hair-above-the-limit violation "
             "still backs off by at least floor·ER instead of sitting in a "
             "limit cycle pinned just above the allowable. Only used when the "
             "gain is > 0.")
    aopt.damping_threshold = b[3].number_input(
        "Damping threshold", value=float(aopt.damping_threshold),
        min_value=0.05, max_value=1.0, step=0.01, format="%.2f",
        key=f"damp_{opt_name}",
        help="While feasible with v above this, removal slows by "
             "(1−v)/(1−threshold) so the design glides into the limit instead "
             "of overshooting and oscillating. 1.0 = off (full rate until "
             "infeasible); 0.9–0.95 is typical.")
    aopt.addback_stress_bias = b[4].number_input(
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
        t = st.columns(4)
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
        cfg.levelset.nucleation_rate = t[3].number_input(
            "Nucleation rate", value=float(cfg.levelset.nucleation_rate),
            min_value=0.0, step=0.1, format="%.2f",
            help="Reaction term (crude topological derivative): low-energy "
                 "regions sink by dt·rate·(1 − Vn) per iteration, so holes can "
                 "nucleate in the free interior instead of only at existing "
                 "void interfaces. 0 = off.")
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

    st.subheader("Manufacturing constraints")
    st.caption("Printability / castability constraints applied to the design each "
               "iteration (after the optimiser update), in the fixed order below. "
               "All default OFF — leave them off for an unconstrained run.")
    mfg = cfg.manufacturing
    _axis_vec = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}
    _opts = ["off", "x", "y", "z"]

    def _axis_name(vec):
        if vec is None:
            return "off"
        t = [round(float(v), 6) for v in vec]
        # A hand-edited vector that is no unit axis (e.g. [0,0,-1] -- sign
        # matters for casting/overhang) maps to "custom", which round-trips the
        # value untouched. The old "z" fallback silently REWROTE it to +z on the
        # next Save, inverting the constraint with no warning.
        return next((a for a, v in _axis_vec.items() if v == t), "custom")

    def _axis_opts(current):
        return _opts + (["custom"] if current == "custom" else [])

    mem = st.columns(2)
    mfg.min_member_layers = int(mem[0].number_input(
        "Minimum member size (erode/dilate hops)", value=int(mfg.min_member_layers),
        min_value=0, step=1,
        help="Morphological open removing thin features / single-element slivers. "
             "0 = off; 1–2 is typical."))
    mfg.max_member_layers = int(mem[1].number_input(
        "Maximum member size (hops to a void)", value=int(mfg.max_member_layers),
        min_value=0, step=1,
        help="OptiStruct MAXDIM: carve bulky lumps so every element lies within N "
             "hops of a void (least-useful material first). 0 = off."))

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

    st.markdown("**Casting / draw direction** — remove undercuts so a die can "
                "slide out along the draw axis.")
    cast = st.columns(2)
    _cur_draw = _axis_name(mfg.draw_direction)
    _sel_draw = cast[0].selectbox(
        "Draw direction", _axis_opts(_cur_draw),
        index=_axis_opts(_cur_draw).index(_cur_draw),
        key="draw_dir",
        help="Axis the die is pulled along. Single-sided keeps a solid bottom "
             "prefix (no material above a void); 'off' disables casting. "
             "'custom' = the config's own (non-axis) vector, kept as-is.")
    if _sel_draw != "custom":
        mfg.draw_direction = None if _sel_draw == "off" else _axis_vec[_sel_draw]
    mfg.draw_two_sided = cast[1].checkbox(
        "Two-sided (parting surface)", value=mfg.draw_two_sided, key="draw_two",
        disabled=_sel_draw == "off",
        help="Keep one contiguous run around a parting surface instead of a "
             "single-sided bottom prefix.")

    st.markdown("**Extrusion** — constant cross-section along an axis (each prism "
                "made uniform by a majority vote).")
    _cur_ext = _axis_name(mfg.extrusion_axis)
    _sel_ext = st.selectbox(
        "Extrusion axis", _axis_opts(_cur_ext),
        index=_axis_opts(_cur_ext).index(_cur_ext),
        key="ext_axis",
        help="Elements are binned into prisms by their footprint; a prism is solid "
             "iff ≥ half of it is alive. 'off' disables extrusion. "
             "'custom' = the config's own (non-axis) vector, kept as-is.")
    if _sel_ext != "custom":
        mfg.extrusion_axis = None if _sel_ext == "off" else _axis_vec[_sel_ext]

    st.markdown("**Overhang / self-support** — forbid material unsupported below "
                "the critical angle along the build direction (applied last).")
    ov = st.columns(2)
    _cur_bld = _axis_name(mfg.build_direction)
    _sel = ov[0].selectbox(
        "Build direction", _axis_opts(_cur_bld),
        index=_axis_opts(_cur_bld).index(_cur_bld),
        key="build_dir",
        help="Up/growth axis of the print. 'off' disables the overhang "
             "constraint. 'custom' = the config's own (non-axis) vector, kept "
             "as-is.")
    if _sel != "custom":
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
    render_appearance_settings(cfg.animate, key_prefix="con")   # shows inline colour validity


@st.cache_resource(show_spinner=False)
def _load_deck_mesh(starter_str: str, mtime: float, part_id: int, node_min: int):
    """Load + cache the (read-only) deck and mesh for the preview button.

    Keyed on the starter path, its mtime and the design-part ids, so repeated
    preview clicks (which only vary the regions) don't re-parse the deck — matters
    for the 575k-element model. ``st.cache_resource`` returns the same objects
    (no per-call copy); they are only read here, never mutated."""
    deck = Deck.load(starter_str, part_id, node_min)
    return deck, Mesh.from_deck(deck)


def _render_keepout(cfg: Config) -> None:
    """Growth keep-out: an additional Radioss deck of nearby parts (never solved)
    whose occupied volume forbids growth. A candidate inside the neighbour parts
    starts void but is held void every iteration, so the optimiser can never place
    material there. The path is resolved relative to ``case_dir`` like the decks;
    the counts show up in the 🔍 preview below.

    Written straight onto ``cfg.model`` (the deferred-save pattern -- app.py
    rebuilds cfg from YAML each rerun, so the sidebar 💾 Save persists what is on
    screen)."""
    with st.expander("🚧 Keep-out — forbid growth into neighbour parts"):
        st.caption(
            "An **additional** Radioss deck describing nearby parts that are "
            "**not simulated** (never solved). Their occupied volume (their "
            "actual mesh, not a bounding box) is **forbidden growth space**: a "
            "growth candidate whose centroid falls inside those parts is held "
            "**void** every iteration — it never becomes material, so the "
            "optimiser cannot grow into the neighbour parts. Applied to both the "
            "run and the ⚙️ growth-mesh step (no candidate tets are generated "
            "there). Path is relative to the case directory, like the load-case "
            "decks; leave blank to disable.")
        cfg.model.growth_keepout_rad = (st.text_input(
            "Keep-out deck (.rad, relative to case_dir)",
            value=cfg.model.growth_keepout_rad or "",
            placeholder="neighbour_parts_0000.rad",
            help="A Radioss starter/deck with the nearby parts' /NODE + "
                 "/TETRA4 (or /BRICK) blocks. Only geometry is read; it is "
                 "never added to any solve.").strip() or None)
        c = st.columns(2)
        pids = c[0].text_input(
            "Keep-out part ids (comma-sep, blank = all solid parts)",
            value=",".join(str(x) for x in cfg.model.growth_keepout_part_ids),
            help="Which /TETRA4 or /BRICK part ids in the keep-out deck form the "
                 "forbidden space. Blank = every solid part in the deck.")
        cfg.model.growth_keepout_part_ids = _int_ids(pids, "keep-out part ids")
        cfg.model.growth_keepout_clearance_mm = float(c[1].number_input(
            "Clearance [mm]", value=float(cfg.model.growth_keepout_clearance_mm),
            min_value=0.0, step=0.5, format="%.3f",
            help="Keep a gap around the neighbour parts: a candidate within this "
                 "distance of the neighbour geometry is forbidden too. 0 = the "
                 "parts' volume exactly."))


def _render_growth_preview(cfg: Config) -> None:
    """On-demand button: count how many design elements each growth region would
    start *void*, by loading the primary load case's starter deck in-process (pure
    Python, no VTK) and running :func:`oropt.loop.preview_growth_boxes`.

    Lets a user confirm a region is actually pre-meshed — and that the run-start
    guards pass — before committing to a multi-hour run, instead of finding out
    only when the loop aborts. Reads the live editor state on ``cfg``. The deck
    load is cached (:func:`_load_deck_mesh`) so re-clicking after only editing
    regions is instant.

    Clicking also **auto-fills the original-part element-id boundary**
    (``model.growth_original_elem_max``) while it is unset and a carve-off
    region needs it: the loaded deck's highest design element id. The preview
    is computed with the boundary applied, then one ``st.rerun`` refreshes the
    id input above (it was instantiated before the click could be handled);
    the computed preview rides across that rerun in session state."""
    clicked = st.button(
        "🔍 Preview region element counts", key="growth_preview",
        help="Load the deck and count the design elements inside each region "
             "(they start void). Also runs the run-start guards, so a "
             "mis-placed or un-meshed region — or a typo'd keep-out / "
             "stress-exclusion / BC /GRNOD group id — is caught now, not "
             "hours in. "
             "While the original-part element-id boundary is unset (0), it is "
             "auto-filled with the CURRENT deck's highest design element id — "
             "so run this on the ORIGINAL folder, not an extended growth_mesh "
             "deck (see the note below).")
    if any(not b.carve for b in cfg.model.growth_boxes):
        st.caption(
            "⚠️ **Getting *Original part: highest element id* right (carve-off "
            "regions).** It must be the **original** deck's highest element id, "
            "fixed *before* any expansion elements exist. This button auto-fills "
            "it **only while the field is 0/unset**, reading whichever deck "
            "`case_dir` points at now — so let it auto-fill only with `case_dir` "
            "on the **original** folder. Once `case_dir` points at a "
            "**growth_mesh / extended** deck, do **not** rely on this: it would "
            "capture the extended deck's (higher) max, so every generated "
            "candidate is misread as original part and the run aborts with "
            "*“contains only original part elements … nothing would start "
            "void.”* On an extended deck the boundary comes from the ⚙️ *use "
            "these decks* button — it records the exact boundary **and** fills "
            "this field with it on the rerun — or type it in by hand (this "
            "🔍 preview will not overwrite a field that is already non-zero).")
    preview, autofilled = None, None
    if clicked:
        cases = cfg.load_case_list()
        if not cases:
            st.warning("Define a load case (🔀 Load cases tab) first — the "
                       "preview reads that case's starter deck.")
            return
        starter = cases[0].starter
        if not starter.exists():
            st.warning(f"Starter deck not found: `{starter}`. Check the case "
                       "directory and the load case's stem.")
            return
        try:
            with st.spinner(f"Loading `{starter.name}` and counting …"):
                deck, mesh = _load_deck_mesh(
                    str(starter), starter.stat().st_mtime,
                    cfg.model.design_part_id, cfg.model.design_node_min)
                if (cfg.model.growth_original_elem_max is None
                        and deck.elem_ids.size
                        and any(not b.carve for b in cfg.model.growth_boxes)):
                    # Derive the original/expansion id boundary from the deck
                    # itself — right for the original decks; a deck already
                    # holding expansion elements needs it set by hand (the
                    # success message below says so).
                    autofilled = int(deck.elem_ids.max())
                    cfg.model.growth_original_elem_max = autofilled
                preview = preview_growth_boxes(deck, mesh, cfg.model)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not preview regions: {exc}")
            return
        if autofilled is not None:
            st.session_state["growth_orig_elem_max_autofill"] = autofilled
            st.session_state["growth_preview_stash"] = (preview, autofilled)
            st.rerun()
    else:
        stashed = st.session_state.pop("growth_preview_stash", None)
        if stashed is not None:
            preview, autofilled = stashed
    if preview is None:
        return
    if autofilled is not None:
        st.success(
            f"Auto-filled **Original part: highest element id** = {autofilled} "
            "— the highest design element id in the loaded starter deck. "
            "Save the config to keep it. Derived from the *current* deck: if "
            "it already contains expansion elements (hand-pre-meshed, or an "
            "extended growth-mesh deck), correct it manually — the ⚙️ "
            "growth-mesh *use these decks* button records the exact boundary.")
    st.dataframe(
        pd.DataFrame([{"region": r.name, "shape": r.shape,
                       "elements": r.count, "note": r.note or "✓"}
                      for r in preview.rows]),
        hide_index=True, use_container_width=True)
    pct = (100.0 * preview.total_candidates / preview.total_elements
           if preview.total_elements else 0.0)
    if preview.notice:
        st.warning(preview.notice)
    if preview.keepout:
        (st.error if "error" in preview.keepout else st.info)(
            f"🚧 {preview.keepout}")
    if preview.group_guard:
        st.error(f"⚠ Run-start guard would abort: {preview.group_guard}")
    if preview.guard:
        st.error(f"⚠ Run-start guard would abort: {preview.guard}")
    elif not preview.group_guard:
        st.success(
            f"{preview.total_candidates} of {preview.total_elements} design "
            f"elements ({pct:.1f}%) start void across all regions — regions are "
            "run-ready.")


def _render_growth_mesh_progress(prep: Path) -> None:
    """Live view of a running PREPARE subprocess: a fragment tails the log
    every couple of seconds (only this block reruns — the rest of the app
    stays interactive, like the Monitor's metrics fragment) and flips the
    whole panel over with a full rerun once the child ends."""
    @st.fragment(run_every=2.0)
    def _progress():
        status = growthprep.read_status(prep)
        if status.state != growthprep.RUNNING:
            st.rerun()          # full-app rerun → the panel renders the outcome
        st.info(f"⚙️ Generating the candidate mesh (TetGen) in isolated "
                f"subprocess {status.pid} — the dashboard stays responsive, "
                "and a page reload re-attaches to the same run.")
        if status.log_tail:
            st.code(status.log_tail, language=None)
        if st.button("🛑 Cancel generation", key="growth_mesh_cancel",
                     help="Force-kill the PREPARE subprocess. Decks are only "
                          "written after every guard passes, so cancelling "
                          "leaves no partial deck set behind."):
            growthprep.cancel(prep)
            st.rerun()
    _progress()


def _render_growth_mesh(cfg: Config, cfg_path: Path) -> None:
    """The growth-mesh PREPARE step (phase 2, :mod:`oropt.growthmesh`) as a
    button: TetGen-fill the regions with node-conformal candidate elements and
    write the extended deck set to ``<case_dir>/growth_mesh/``, instead of
    pre-meshing the region volume in a pre-processor.

    TetGen runs in an *isolated detached subprocess*
    (:mod:`oropt.gui.growthprep`), never in-process: on a large model it can
    allocate tens of GB (observed >50 GB) and a native crash is uncatchable,
    either of which used to freeze or OOM-kill the whole dashboard. The panel
    only launches the CLI and then *reads files* — the log tail while running,
    the ``--json`` report when done — so the Streamlit script is never blocked
    and a page reload re-attaches.

    Deliberately explicit, mirroring the CLI: nothing is written unless the
    phase-1 run-start guards pass on the extended deck, the result lands in its
    own folder for inspection/diffing, and a run only uses it after
    ``model.case_dir`` is pointed there — the second button, which rewrites the
    YAML and reruns so the Input tab shows the new folder."""
    if st.session_state.pop("growth_mesh_pointed", None):
        st.success(f"`model.case_dir` now points at the extended deck set "
                   f"(saved to `{cfg_path}`). Iteration 0 still solves exactly "
                   "the original part — the new elements start void.")
    with st.expander("⚙️ Generate growth mesh (TetGen) — no pre-meshing needed"):
        st.caption(
            "Fills each region with new TET4 candidate elements that share the "
            "part's surface **nodes** (TetGen `-Y` constrained "
            "tetrahedralisation: the part surface is preserved exactly — exact "
            "node conformity, no tied interface), then writes **extended "
            f"starter decks** for every load case to `{GROWTH_MESH_DIRNAME}/` "
            "inside the case directory (engine decks copied verbatim), for "
            "inspection and diffing. New node ids go above max(existing) and "
            "≥ `design_node_min`; element ids above max(existing). The "
            "run-start guards are re-run on the extended deck **before** "
            "anything is written. Needs the optional `tetgen` package "
            "(`pip install \"oropt[growthmesh]\"`; TetGen itself is "
            "AGPL-licensed). The generator only sees the design part — keep "
            "regions clear of other parts (rigid bodies, shells).")
        gcol = st.columns(2)
        size_factor = gcol[0].number_input(
            "Element size ×", value=1.0, min_value=0.1, step=0.1, format="%.2f",
            key="growth_mesh_size",
            help="Target element edge as a multiple of the part's mean surface "
                 "edge length. 1.0 matches the part's own sizing; larger = "
                 "coarser and faster.")
        min_ratio = gcol[1].number_input(
            "Quality bound (TetGen -q)", value=1.5, min_value=1.0, step=0.1,
            format="%.2f", key="growth_mesh_minratio",
            help="Radius-edge quality bound; lower = better-shaped tets, more "
                 "elements.")
        prep = growthprep.prepare_dir(cfg)
        status = growthprep.read_status(prep)
        if st.button("⚙️ Generate & write extended decks",
                     key="growth_mesh_generate",
                     disabled=status.state == growthprep.RUNNING,
                     help="Runs TetGen on the primary case's starter deck in "
                          "an isolated subprocess (a huge model can need tens "
                          "of GB — the dashboard stays alive either way) and "
                          "writes the extended deck set for every load case. "
                          "Nothing is written if a run-start guard fails."):
            saved = Config.from_dict(Config.read_yaml_dict(cfg_path))
            if saved.model.growth_boxes != cfg.model.growth_boxes:
                st.warning("The regions above have unsaved edits — generating "
                           "from the on-screen regions. 💾 Save config before "
                           "running, or the run will select candidates with "
                           "the stale saved regions.")
            try:
                growthprep.start(cfg, prep, float(size_factor),
                                 float(min_ratio), PROJECT_ROOT)
            except (RuntimeError, OSError) as exc:
                st.error(f"Could not launch the growth-mesh subprocess: {exc}")
            status = growthprep.read_status(prep)
        if status.state == growthprep.RUNNING:
            _render_growth_mesh_progress(prep)
            return
        if status.state == growthprep.FAILED:
            st.error(f"Growth-mesh generation failed: {status.error}")
            if status.log_tail:
                st.code(status.log_tail, language=None)
            st.caption(f"Full log: `{prep / growthprep.LOG_NAME}`")
            return
        rep = status.report
        if rep is None:
            return
        st.success(
            f"{rep.n_new_elems} new candidate elements / {rep.n_new_nodes} new "
            f"nodes (target edge {rep.target_edge:.3g}; quality min "
            f"{rep.quality_min:.2f}, median {rep.quality_median:.2f}). Guards "
            f"passed — {rep.total_candidates} elements start void on the "
            f"extended deck. Written to `{rep.out_dir}`.")
        st.dataframe(
            pd.DataFrame([{"region": lbl, "new elements": n}
                          for lbl, n in rep.per_region]),
            hide_index=True, use_container_width=True)
        if st.button("📁 Use the extended decks — point the config's case "
                     "directory at them", key="growth_mesh_use",
                     help="Rewrites model.case_dir in the config YAML to the "
                          "growth_mesh folder and records the original-part "
                          "element-id boundary (growth_original_elem_max) that "
                          "carve-off regions need (only those keys are "
                          "touched)."):
            point_config_at(cfg_path, rep.out_dir,
                            original_elem_max=rep.original_elem_max)
            st.session_state["growth_mesh_pointed"] = rep.out_dir
            # Push the recorded boundary through the same channel the 🔍 preview
            # uses so the "Original part: highest element id" field — a sticky
            # session-state widget — actually shows it on the rerun, instead of
            # keeping (and, worse, re-saving) its stale/0 value over the correct
            # boundary point_config_at just wrote to the YAML.
            st.session_state["growth_orig_elem_max_autofill"] = int(
                rep.original_elem_max)
            st.rerun()


def _add_growth_overlay(pl, pv, boxes) -> int:
    """Add a red wireframe outline of each growth region to plotter *pl*.

    Draws the same :func:`~oropt.mesh.overlay_primitives` outlines the report
    render uses — a box (12 edges), sphere, finite cylinder or polyhedron
    (convex-hull edges) — so a user can place region coordinates against the
    live topology instead of blind. Returns the number of regions drawn (0 when
    none have drawable geometry, e.g. a deck-referenced box whose corners
    aren't resolved without the deck)."""
    import numpy as np
    drawn = 0
    for pr in overlay_primitives(boxes):
        kind = pr["kind"]
        if kind in ("box", "polyhedron"):
            pts = np.asarray(pr["corners"], dtype=float)
            lines = np.hstack([[2, i, j] for i, j in pr["edges"]]).astype(int)
            mesh = pv.PolyData(pts, lines=lines)
        elif kind == "sphere":
            mesh = pv.Sphere(radius=pr["radius"], center=pr["center"])
        else:                                        # cylinder
            p1 = np.asarray(pr["p1"], dtype=float)
            p2 = np.asarray(pr["p2"], dtype=float)
            axis = p2 - p1
            mesh = pv.Cylinder(center=(p1 + p2) / 2.0, direction=axis,
                               radius=pr["radius"],
                               height=float(np.linalg.norm(axis)))
        pl.add_mesh(mesh, color="red", style="wireframe", line_width=2,
                    opacity=0.7)
        drawn += 1
    return drawn


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
        # Live "what is running right now" (set before each solve): the fast-mode
        # monitor — shows whether a fast tied-linear or full nonlinear solve is in
        # flight during the minutes it takes.
        activity = getattr(status, "activity", "")
        if status.state == "running" and activity:
            st.info(f"🔧 Now running: {activity}")
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
        fast_names = [c["name"] for c in cases if c.get("fast_mode")]
        if fast_names:
            st.caption(f"⚡ **Fast mode** (tied linear screen, ~35× faster — a "
                       f"ranking screen reading ~14% below nonlinear, not a "
                       f"certifying stress) for: {', '.join(fast_names)}.")
        if len(cases) > 1:
            st.caption(f"σ_max and disp above are the **worst across "
                       f"{len(cases)} load cases**, each shown with that case's "
                       "own limit; the design is feasible only when every case is "
                       "(below). Each case's animation is under `solve/case_<i>/`.")
            cdf = pd.DataFrame([{
                "case": c["name"],
                **({"mode": "⚡ fast" if c.get("fast_mode") else "nonlinear"}
                   if fast_names else {}),   # mode column only when a case is fast
                "σ_max [MPa]": c["sigma_max"],
                "σ_allow [MPa]": _fmt_limit(c["sigma_allow"], "{:.0f}"),
                "disp [mm]": c["disp"],
                "d_allow [mm]": _fmt_limit(c["d_allow"], "{:.2f}"),
                "feasible": "✅" if c["feasible"] else "⚠️",
            } for c in cases])
            st.dataframe(cdf, hide_index=True, use_container_width=True)
        # Per-node displacement breakdown: a load case may constrain several nodes,
        # each with its own limit. Shown whenever there's more than one such
        # constraint (a classic single-node run keeps just the headline metric).
        disp_rows = [(c["name"], dc) for c in cases
                     for dc in c.get("disp_constraints", [])]
        if len(disp_rows) > 1:
            st.caption("Displacement constraints — each node is checked against "
                       "its own limit; the design is feasible only when **every** "
                       "one holds. The disp metric above is the worst ratio.")
            ddf = pd.DataFrame([{
                "case": name,
                "node": str(dc["node_id"]),
                "disp [mm]": dc["disp"],
                "d_allow [mm]": _fmt_limit(dc["d_allow"], "{:.2f}"),
                "feasible": "✅" if dc["feasible"] else "⚠️",
            } for name, dc in disp_rows])
            st.dataframe(ddf, hide_index=True, use_container_width=True)
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

        # Run log tail: the loop's stdout is discarded by the detached launch, so
        # this is the only place the run's progress — and every best-effort
        # post-run step's skip reason (d3plot / smooth / animate / report) — is
        # visible in the browser (e.g. "animate: ... skipped"). Auto-expanded on
        # failure so the reason isn't buried; mirrors the PREPARE panel's tail.
        log_tail = st_io.read_log_tail(work)
        if log_tail:
            with st.expander("📜 Run log (tail) — progress & post-run "
                             "(d3plot / smooth / animate / report) messages",
                             expanded=status.state == "failed"):
                st.code(log_tail, language=None)

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
            # Overlay each growth region as a red wireframe outline so regions can
            # be placed against the live topology (best-effort; only regions with
            # resolved geometry are drawn). Skipped when growth is switched off --
            # the regions are dormant, so outlining them would misrepresent the run.
            overlay_boxes = (cfg.model.growth_boxes
                             if cfg.model.growth_enabled else [])
            n_overlay = _add_growth_overlay(pl, pv, overlay_boxes)
            pl.view_isometric(); pl.background_color = "white"
            # backend="panel" renders in-process. The default "trame" backend
            # exports the scene from a multiprocessing.Process whose spawned child
            # dies in DuplicateHandle under `streamlit run` on Windows (and would
            # then hang the parent on queue.get()).
            stpyvista(pl, backend="panel", key="topo")
            if n_overlay:
                st.caption(f"🟥 {n_overlay} growth region(s) outlined in red — "
                           "material may grow into these.")
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


# ---- Re-postprocessing tab -------------------------------------------------
def render_postprocess_tab(cfg: Config, default_folder: Path) -> None:
    """Re-run post-processing on an *existing* finished run — without re-solving.

    Groups the on-demand post-run tools that act on a run folder the loop already
    wrote: re-render the topology-evolution GIF (fresh camera / resolution /
    playback, via :func:`oropt.animate.make_animation`) and re-generate the report
    (:func:`oropt.report.write_report`). Both only *read* the run's per-iteration
    artefacts and re-write their own outputs, so they never touch the run; the
    heavy renders run in isolated off-screen subprocesses (crash containment), safe
    to drive synchronously from here. More tools can be added below over time.
    """
    st.subheader("Re-post-process a finished run")
    st.caption(
        "Re-run post-processing on an **existing** finished run — without "
        "re-solving. Set the run folder, then use the tools below (re-render the "
        "evolution animation, or re-generate the report). More tools may be added "
        "here over time.")

    # Keep the raw text: Path("") stringifies to "." (truthy), so a CLEARED
    # field would silently mean "operate on the GUI's CWD" — the blank-folder
    # guards below would never fire and the report tool would write into the
    # project root instead of a run folder.
    folder_txt = st.text_input(
        "Run folder", str(default_folder), key="reanim_folder",
        help="A finished run's output folder — the one holding its status.json / "
             "history.csv and the per-iteration topology snapshots. Defaults to the "
             "currently selected run.").strip()
    folder = Path(folder_txt)

    st.markdown("#### 🎬 Evolution animation")
    st.caption(
        "Re-render the topology-evolution GIF from the run's per-iteration surfaces "
        "with new settings (camera angle, resolution, colours, playback). Writes a "
        "new GIF into the run folder, leaving the original `topology_evolution.gif` "
        "untouched unless you reuse its name.")
    n_frames, src = frame_count(folder) if folder_txt else (0, "")
    ready = bool(folder_txt) and folder.exists() and n_frames >= 2
    if not folder_txt or not folder.exists():
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

    # ---- re-generate the report ------------------------------------------
    st.markdown("---")
    st.markdown("#### 📝 Report")
    st.caption(
        "Re-generate `report.html` / `report.md` for this run from its "
        "`status.json` + `history.csv` — the summary, the interactive convergence "
        "charts, and a render of the **last feasible** design. Handy to refresh a "
        "run whose report predates an oropt update.")
    report_ready = bool(folder_txt) and folder.exists()
    if not report_ready:
        st.caption("Set a valid run folder above to enable this.")
    if st.button("📝 Re-generate report", key="repost_report",
                 disabled=not report_ready):
        # Prefer the run's own frozen config (config_used.yaml — what it actually
        # ran with) so the report's optimiser/limits match the run, not the config
        # currently loaded in the GUI; fall back to the loaded one if it's absent.
        rcfg = cfg
        run_cfg = folder / "config_used.yaml"
        if run_cfg.is_file():
            try:
                rcfg = Config.from_yaml(str(run_cfg))
            except Exception:  # noqa: BLE001 - fall back to the loaded config
                rcfg = cfg
        logs: list[str] = []
        with st.spinner("Regenerating report (rendering the 3D view may take a "
                        "moment)…"):
            out = write_report(rcfg, folder, logs.append)
        if logs:
            with st.expander("Report log", expanded=out is None):
                for line in logs:
                    st.text(line)
        if out is not None and out.is_file():
            st.success(f"Wrote `{out.name}` (+ `report.md`) in {folder}")
        else:
            st.warning("No report written — see the log above.")


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

# 💾 Save config: captured here, written *after* the tabs render (see the deferred
# block below) so it persists the on-screen edits, not the stale on-disk config.
save_config_clicked = st.sidebar.button(
    "💾 Save config", width="stretch", key="save_config",
    help="Write the current on-screen settings to the config file. (Also done "
         "automatically when you add to the queue or resume.)")

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
# ↻ Resume = continue the selected config's run from its last checkpoint. The
# resume itself is *deferred* to after the tabs render (see below) so the loop
# re-reads the on-screen config — you can change parameters or even the optimiser
# and continue, instead of silently resuming the stale on-disk settings. Needs a
# checkpoint (a run that started) and no live run.
resume_next_iter = st_io.checkpoint_iteration(work)     # loop's start_iter, or None
can_resume = (not running) and resume_next_iter is not None
resume_clicked = c2.button(
    "↻ Resume", disabled=not can_resume, width="stretch", key="resume_run",
    help="Continue this run from its last checkpoint. On-screen parameter / "
         "optimiser edits are saved first, so you can change settings before "
         "continuing (switching optimiser reuses the current geometry; the "
         "level-set field re-initialises). Set '+ more iterations' to run past "
         "where it stopped.")
add_iters = int(st.sidebar.number_input(
    "↻ + more iterations", min_value=0, max_value=1000, value=0, step=10,
    key="resume_add_iters", disabled=not can_resume,
    help="Extend the run by this many iterations beyond where it stopped"
         + (f" (next iteration: {resume_next_iter})." if can_resume else ".")
         + " 0 = continue with the config's current max_iter; a run that already"
           " reached max_iter needs this or there is nothing left to do."))
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
    ["📥 Input", "🔀 Load cases", "🎚 Optimizer / Output", "📊 Monitor",
     "🛠 Re-postprocessing", "🧮 Queue"])
with tab_in:
    render_input_tab(cfg)
with tab_lc:
    render_load_cases_tab(cfg, cfg_path)
with tab_con:
    render_constraints_tab(cfg, cfg_path)
with tab_mon:
    render_monitor_tab(cfg, live_dir, refresh_s)   # follow the live run's folder
with tab_anim:
    render_postprocess_tab(cfg, live_dir)          # re-run post-processing on a run
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

# ---- deferred resume/continue action ---------------------------------------
# Also deferred to here so the loop resumes the *on-screen* config: every tab has
# now written its widgets back into `cfg`, so a changed parameter or optimiser is
# saved before the resume launches (the immediate-launch button used to drop
# them). Optionally extend the active optimiser's max_iter so the run continues
# past where it stopped — a run that already reached max_iter has an empty
# range(start_iter, max_iter) otherwise, so nothing would happen.
if resume_clicked:
    oc = cfg.active_opts()
    target = int(resume_next_iter) + add_iters if add_iters > 0 else oc.max_iter
    if resume_next_iter >= target:
        st.sidebar.warning(
            f"Already at iteration {resume_next_iter} ≥ max_iter {target} — "
            "nothing to continue. Set '↻ + more iterations' above (or raise "
            "Max iterations on the Optimiser tab) and click ↻ Resume again.")
    else:
        oc.max_iter = target                        # extension (or unchanged)
        cfg.to_yaml(cfg_path)                        # persist on-screen edits first
        launch_run(cfg_path, resume=True)
        st.sidebar.success(
            f"Continuing {cfg.optimizer_name()} from iteration {resume_next_iter} "
            f"→ max_iter {target}.")

# ---- deferred save action --------------------------------------------------
# Deferred like the two above so 💾 Save writes the on-screen config (every tab has
# now written its widgets into `cfg`), not the stale on-disk one.
if save_config_clicked:
    cfg.to_yaml(cfg_path)
    st.sidebar.success(f"Saved to {cfg_path}")
