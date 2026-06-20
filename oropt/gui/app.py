"""Streamlit dashboard: configure, launch, and live-monitor a BESO run.

Launch with::

    streamlit run oropt/gui/app.py

The GUI is fully decoupled from the solver: it edits the YAML config, starts
``python -m oropt.run`` as a detached subprocess, and then only *reads* the
status files the loop writes (``status.json`` / ``history.csv`` /
``topology_latest.vtu``). Closing the browser never stops the run; reopening
re-attaches to it.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

from oropt import status as st_io
from oropt.config import Config, DEFAULT_WORK_SUBDIR
from oropt.gui.cases import (CASE_COLUMNS, load_cases_from_records,
                             records_from_load_cases)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CFG = PROJECT_ROOT / "configs" / "elevator_linkage.yaml"

st.set_page_config(page_title="oropt — OpenRadioss BESO", layout="wide")


# ---- run control -----------------------------------------------------------
def launch_run(cfg_path: Path, resume: bool) -> None:
    cmd = [sys.executable, "-m", "oropt.run", "--config", str(cfg_path)]
    if resume:
        cmd.append("--resume")
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    subprocess.Popen(cmd, cwd=str(PROJECT_ROOT), creationflags=flags,
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def request_stop(work: Path) -> None:
    (work / "stop.flag").write_text("stop", encoding="utf-8")


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


# ---- sidebar: config selection --------------------------------------------
st.sidebar.title("oropt")
st.sidebar.caption("OpenRadioss-coupled BESO topology optimisation")
cfg_path = Path(st.sidebar.text_input("Config file", str(DEFAULT_CFG)))
if not cfg_path.exists():
    st.sidebar.error("Config not found.")
    st.stop()
cfg = Config.from_yaml(cfg_path)
work = Path(cfg.run_folder())          # work_dir, or <case_dir>/work when blank
if not work.is_absolute():
    work = PROJECT_ROOT / work

running = st_io.is_running(work)
st.sidebar.markdown(f"**Run state:** {'🟢 running' if running else '⚪ idle'}")
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("▶ Start", disabled=running, width="stretch"):
    cfg.to_yaml(cfg_path)
    launch_run(cfg_path, resume=False)
    st.sidebar.success("Launched.")
if c2.button("⏸ Stop", disabled=not running, width="stretch"):
    request_stop(work)
    st.sidebar.info("Stop requested (after current solve).")
if c3.button("↻ Resume", disabled=running, width="stretch"):
    launch_run(cfg_path, resume=True)
    st.sidebar.success("Resumed.")
if st.sidebar.button("⏹ Force kill", disabled=not running):
    force_kill(work)

refresh_s = int(st.sidebar.number_input(
    "Refresh interval (s)", min_value=1, max_value=3600, value=60, step=5,
    help="How often the Monitor tab re-reads the run's status files."))

tab_in, tab_lc, tab_con, tab_mon = st.tabs(
    ["📥 Input", "🔀 Load cases", "🎚 Constraints / BC", "📊 Monitor"])

# ---- Input tab -------------------------------------------------------------
with tab_in:
    st.subheader("Model")
    cfg.model.case_dir = st.text_input("Case directory", cfg.model.case_dir)
    cfg.model.stem = st.text_input("Deck stem", cfg.model.stem)
    cfg.work_dir = st.text_input(
        "Run / output folder", cfg.work_dir,
        placeholder=cfg.run_folder(),
        help="Scratch, checkpoints and status files go here. Leave blank to use "
             f"a `{DEFAULT_WORK_SUBDIR}/` sub-folder inside the case directory "
             f"(→ `{cfg.run_folder()}`); the mutated deck is isolated in its "
             "solve/ sub-folder.")
    cc = st.columns(3)
    cfg.model.design_part_id = int(cc[0].number_input(
        "Design part id", value=cfg.model.design_part_id, step=1))
    cfg.model.disp_node_id = int(cc[1].number_input(
        "Displacement node id", value=cfg.model.disp_node_id or 0, step=1)) or None
    cfg.model.bc_group_id = int(cc[2].number_input(
        "BC node-group id", value=cfg.model.bc_group_id, step=1))
    st.caption(f"OpenRadioss root: `{cfg.or_paths.root}`  ·  np={cfg.run.np} "
               f"nt={cfg.run.nt}  ·  starter `{cfg.model.starter().name}`")

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
with tab_lc:
    st.subheader("Load cases")
    st.caption(
        "Optimise the part against several loads (the linkage pulled in "
        "different directions) by minimising a **weighted-sum compliance**. "
        "Each row is a separate deck pair in the case directory that shares the "
        "same mesh — only its load differs. **Leave the table empty for a "
        "classic single-load run.** Blank *stem* → the model deck stem (Input "
        "tab); blank *disp/σ/d* cells inherit the model & constraints defaults.")
    lc_df = pd.DataFrame(records_from_load_cases(cfg.load_cases),
                         columns=CASE_COLUMNS)
    lc_edited = st.data_editor(
        lc_df, num_rows="dynamic", width="stretch",
        key="load_cases_editor", column_config={
            "name": st.column_config.TextColumn(
                "Name", help="Label for the load case, e.g. pull_z."),
            "stem": st.column_config.TextColumn(
                "Deck stem", help="<stem>_0000.rad / _0001.rad in the case "
                                  "directory. Blank → the model deck stem."),
            "weight": st.column_config.NumberColumn(
                "Weight", min_value=0.0, step=0.1, format="%.3f",
                help="wᵢ in s_e = Σ wᵢ·(energyᵢ / max energyᵢ). Blank → 1."),
            "disp_node_id": st.column_config.NumberColumn(
                "Disp node id", step=1, format="%d",
                help="Constrained node for this case. Blank → model disp node."),
            "sigma_allow": st.column_config.NumberColumn(
                "σ_allow [MPa]", min_value=0.0, step=1.0,
                help="Per-case stress limit. Blank → the global constraint."),
            "d_allow": st.column_config.NumberColumn(
                "d_allow [mm]", min_value=0.0, step=0.1,
                help="Per-case displacement limit. Blank → the global constraint."),
        })
    cfg.load_cases = load_cases_from_records(lc_edited.to_dict("records"))
    if cfg.load_cases:
        st.success(
            f"{len(cfg.load_cases)} load case(s): every iteration solves all of "
            "them (≈ N× a single-case run, each under `solve/case_<i>/`); the "
            "design is feasible only when **every** case is. Save the config "
            "(Constraints / BC tab) or ▶ Start to apply.")
    else:
        st.info("No load cases — the run uses the single model deck (classic "
                "single-load BESO). Add a row above to optimise several loads.")

# ---- Constraints / BC tab --------------------------------------------------
with tab_con:
    st.subheader("Constraints")
    a, b = st.columns(2)
    cfg.constraints.sigma_allow = a.number_input(
        "Max von-Mises σ_allow [MPa]", value=float(cfg.constraints.sigma_allow))
    cfg.constraints.d_allow = b.number_input(
        "Max displacement d_allow [mm]", value=float(cfg.constraints.d_allow))

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
        value=not cfg.beso.protect_bc_nodes,
        help="By default the BC node-group (model.bc_group_id) is frozen. Enable "
             "this to let the optimiser remove material there too — the BC nodes "
             "stay fixed via their /BCS and still anchor connectivity.")
    cfg.beso.protect_bc_nodes = not allow_del_bc

    st.subheader("BESO parameters")
    g = st.columns(3)
    cfg.beso.evolution_rate = g[0].number_input(
        "Evolution rate (vol/iter)", value=float(cfg.beso.evolution_rate),
        step=0.005, format="%.3f")
    cfg.beso.target_volume_fraction = g[1].number_input(
        "Target volume fraction", value=float(cfg.beso.target_volume_fraction),
        min_value=0.05, max_value=1.0, step=0.05)
    cfg.beso.filter_radius = g[2].number_input(
        "Filter radius [mm]", value=float(cfg.beso.filter_radius), step=0.5)
    h = st.columns(3)
    cfg.beso.history_weight = h[0].slider(
        "History weight", 0.0, 1.0, float(cfg.beso.history_weight))
    cfg.beso.max_iter = int(h[1].number_input(
        "Max iterations", value=int(cfg.beso.max_iter), step=10))
    cfg.beso.sensitivity = h[2].selectbox(
        "Sensitivity", ["energy", "vonmises", "blend"],
        index=["energy", "vonmises", "blend"].index(cfg.beso.sensitivity))
    arch = st.columns(2)
    cfg.beso.archive_iterations = arch[0].checkbox(
        "Archive each iteration", value=cfg.beso.archive_iterations,
        help="Copy each iteration's deck, animation and listing into "
             "<run_folder>/iter_NNNN/ before solve/ is recycled.")
    cfg.beso.archive_restart = arch[1].checkbox(
        "…incl. restart (~345 MB/iter)", value=cfg.beso.archive_restart,
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
             "design's OpenRadioss animation, writing <work>/d3plot/<stem>.d3plot. "
             "Best-effort — a missing tool or dependency never fails the run.")
    dc = st.columns(2)
    cfg.d3plot.tool_root = dc[0].text_input(
        "Vortex-Radioss tool root", cfg.d3plot.tool_root,
        help="Folder containing the `vortex_radioss` package "
             "(the openradioss_tools repo root).")
    cfg.d3plot.python_exe = dc[1].text_input(
        "Converter Python (optional)", cfg.d3plot.python_exe,
        placeholder=str(Path(cfg.d3plot.tool_root) / ".venv" / "Scripts" / "python.exe"),
        help="Interpreter with lasso-python + tqdm installed. Blank → the tool "
             "root's .venv if present, else the oropt interpreter.")

    st.markdown("**Surface smoothing**")
    cfg.smooth.enabled = st.checkbox(
        "Smooth the final geometry surface when the run finishes",
        value=cfg.smooth.enabled,
        help="Extracts and smooths the final design's surface, writing "
             "<work>/topology_smoothed.<ext> — a clean mesh for CAD / 3D-print / review.")
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

    if st.button("💾 Save config"):
        cfg.to_yaml(cfg_path)
        st.success(f"Saved to {cfg_path}")


# ---- Monitor tab -----------------------------------------------------------
with tab_mon:
    @st.fragment(run_every=refresh_s)
    def monitor():
        status = st_io.read_status(work)
        if status is None:
            st.info("No run yet. Configure on the other tabs, then ▶ Start.")
            return

        feas = "✅ feasible" if status.feasible else "⚠️ infeasible"
        st.markdown(f"**{status.state.upper()}** · iter {status.iteration}/"
                    f"{status.max_iter} · {feas} · {status.message}")
        k = st.columns(4)
        k[0].metric("Volume fraction", f"{status.volume_fraction:.3f}")
        k[1].metric("σ_max [MPa]", f"{status.sigma_max:.1f}",
                    f"limit {status.sigma_allow:.0f}", delta_color="off")
        k[2].metric("disp [mm]", f"{status.disp:.4f}",
                    f"limit {status.d_allow:.2f}", delta_color="off")
        eta = status.eta_s / 60 if status.eta_s == status.eta_s else float("nan")
        k[3].metric("ETA [min]", "—" if eta != eta else f"{eta:.0f}",
                    f"elapsed {status.elapsed_s/60:.0f} min", delta_color="off")
        if len(cfg.load_cases) > 1:
            st.caption(f"σ_max and disp are the **worst across "
                       f"{len(cfg.load_cases)} load cases**; the design is "
                       "feasible only when every case is. Each case's animation "
                       "is under `solve/case_<i>/`.")

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

        topo = work / st_io.TOPOLOGY
        if topo.exists():
            st.subheader("Current topology")
            try:
                import pyvista as pv
                from stpyvista import stpyvista
                grid = pv.read(str(topo))
                scal = "sensitivity" if "sensitivity" in grid.cell_data else None
                pl = pv.Plotter(window_size=[700, 450], off_screen=True)
                pl.add_mesh(grid, scalars=scal, cmap="viridis", show_edges=False)
                pl.view_isometric(); pl.background_color = "white"
                # backend="panel" renders in-process. The default "trame" backend
                # exports the scene from a multiprocessing.Process whose spawned
                # child dies in DuplicateHandle under `streamlit run` on Windows
                # (and would then hang the parent on queue.get()).
                stpyvista(pl, backend="panel", key="topo")
            except Exception as exc:  # noqa: BLE001
                st.caption(f"(3D view unavailable: {exc})")

    monitor()
