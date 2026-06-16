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
from oropt.config import Config

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
work = Path(cfg.work_dir)
if not work.is_absolute():
    work = PROJECT_ROOT / work

running = st_io.is_running(work)
st.sidebar.markdown(f"**Run state:** {'🟢 running' if running else '⚪ idle'}")
c1, c2, c3 = st.sidebar.columns(3)
if c1.button("▶ Start", disabled=running, use_container_width=True):
    cfg.to_yaml(cfg_path)
    launch_run(cfg_path, resume=False)
    st.sidebar.success("Launched.")
if c2.button("⏸ Stop", disabled=not running, use_container_width=True):
    request_stop(work)
    st.sidebar.info("Stop requested (after current solve).")
if c3.button("↻ Resume", disabled=running, use_container_width=True):
    launch_run(cfg_path, resume=True)
    st.sidebar.success("Resumed.")
if st.sidebar.button("⏹ Force kill", disabled=not running):
    force_kill(work)

tab_in, tab_con, tab_mon = st.tabs(["📥 Input", "🎚 Constraints / BC", "📊 Monitor"])

# ---- Input tab -------------------------------------------------------------
with tab_in:
    st.subheader("Model")
    cfg.model.case_dir = st.text_input("Case directory", cfg.model.case_dir)
    cfg.model.stem = st.text_input("Deck stem", cfg.model.stem)
    cc = st.columns(3)
    cfg.model.design_part_id = int(cc[0].number_input(
        "Design part id", value=cfg.model.design_part_id, step=1))
    cfg.model.disp_node_id = int(cc[1].number_input(
        "Displacement node id", value=cfg.model.disp_node_id or 0, step=1)) or None
    cfg.model.bc_group_id = int(cc[2].number_input(
        "BC node-group id", value=cfg.model.bc_group_id, step=1))
    st.caption(f"OpenRadioss root: `{cfg.or_paths.root}`  ·  np={cfg.run.np} "
               f"nt={cfg.run.nt}  ·  starter `{cfg.model.starter().name}`")

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

    if st.button("💾 Save config"):
        cfg.to_yaml(cfg_path)
        st.success(f"Saved to {cfg_path}")


# ---- Monitor tab -----------------------------------------------------------
with tab_mon:
    @st.fragment(run_every=5.0)
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
                pl = pv.Plotter(window_size=[700, 450])
                pl.add_mesh(grid, scalars=scal, cmap="viridis", show_edges=False)
                pl.view_isometric(); pl.background_color = "white"
                stpyvista(pl, key="topo")
            except Exception as exc:  # noqa: BLE001
                st.caption(f"(3D view unavailable: {exc})")

    monitor()
