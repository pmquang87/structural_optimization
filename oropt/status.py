"""Run state shared between the headless solver loop and the Streamlit GUI.

The loop owns the run and writes three artefacts into ``work_dir`` every
iteration; the GUI only ever *reads* them, so closing or crashing the GUI never
touches the run:

* ``status.json``          — latest scalar state (atomic write)
* ``history.csv``          — one row per iteration (appended)
* ``topology_latest.vtu``  — current alive elements for the 3D view

A ``run.pid`` file records the loop process so the GUI can tell whether a run is
live and offer Stop. ``checkpoint.npz`` holds the alive mask + sensitivity
history for ``--resume``.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

STATUS = "status.json"
HISTORY = "history.csv"
TOPOLOGY = "topology_latest.vtu"
PIDFILE = "run.pid"
CHECKPOINT = "checkpoint.npz"

_HISTORY_COLS = ["iteration", "volume_fraction", "sigma_max", "disp",
                 "elements_alive", "feasible", "iter_wall_s", "or_termination"]


@dataclass
class Status:
    state: str = "idle"               # idle | running | converged | failed | stopped
    iteration: int = 0
    max_iter: int = 0
    volume_fraction: float = 1.0
    sigma_max: float = float("nan")
    sigma_allow: float = float("nan")
    disp: float = float("nan")
    d_allow: float = float("nan")
    feasible: bool = True
    elements_alive: int = 0
    elements_total: int = 0
    or_termination: str = ""
    iter_wall_s: float = 0.0
    elapsed_s: float = 0.0
    eta_s: float = float("nan")
    message: str = ""
    updated: str = ""
    pid: int = 0


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_status(work_dir: str | Path, st: Status) -> None:
    import datetime as _dt
    st.updated = _dt.datetime.now().isoformat(timespec="seconds")
    _atomic_write(Path(work_dir) / STATUS, json.dumps(asdict(st), indent=2))


def read_status(work_dir: str | Path) -> Optional[Status]:
    p = Path(work_dir) / STATUS
    if not p.exists():
        return None
    try:
        return Status(**json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, TypeError):
        return None


def append_history(work_dir: str | Path, row: dict) -> None:
    p = Path(work_dir) / HISTORY
    new = not p.exists()
    with open(p, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_HISTORY_COLS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def read_history(work_dir: str | Path) -> list[dict]:
    p = Path(work_dir) / HISTORY
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_topology(work_dir: str | Path, node_xyz: np.ndarray,
                   conn_rows: np.ndarray, alive_mask: np.ndarray,
                   fields: Optional[dict] = None) -> None:
    """Write the current alive tetra mesh to ``topology_latest.vtu`` for the GUI."""
    import pyvista as pv
    idx = np.flatnonzero(alive_mask)
    if idx.size == 0:
        return
    conn = conn_rows[idx]
    cells = np.hstack([np.full((idx.size, 1), 4, np.int64), conn]).ravel()
    celltypes = np.full(idx.size, pv.CellType.TETRA, np.uint8)
    grid = pv.UnstructuredGrid(cells, celltypes, node_xyz.astype(float))
    for name, arr in (fields or {}).items():
        grid.cell_data[name] = np.asarray(arr)[idx]
    grid.save(str(Path(work_dir) / TOPOLOGY))


# ---- PID / liveness --------------------------------------------------------
def write_pid(work_dir: str | Path) -> int:
    pid = os.getpid()
    (Path(work_dir) / PIDFILE).write_text(str(pid), encoding="utf-8")
    return pid


def read_pid(work_dir: str | Path) -> Optional[int]:
    p = Path(work_dir) / PIDFILE
    if not p.exists():
        return None
    try:
        return int(p.read_text(encoding="utf-8").strip())
    except ValueError:
        return None


def clear_pid(work_dir: str | Path) -> None:
    (Path(work_dir) / PIDFILE).unlink(missing_ok=True)


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    if sys.platform == "win32":
        import ctypes
        SYNCHRONIZE = 0x00100000
        h = ctypes.windll.kernel32.OpenProcess(SYNCHRONIZE, False, int(pid))
        if h:
            ctypes.windll.kernel32.CloseHandle(h)
            return True
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def is_running(work_dir: str | Path) -> bool:
    return pid_alive(read_pid(work_dir))


# ---- checkpoint (resume) ---------------------------------------------------
def save_checkpoint(work_dir: str | Path, iteration: int, alive_mask: np.ndarray,
                    sens_prev: Optional[np.ndarray] = None) -> None:
    np.savez(Path(work_dir) / CHECKPOINT, iteration=iteration,
             alive_mask=alive_mask,
             sens_prev=(sens_prev if sens_prev is not None else np.array([])))


def load_checkpoint(work_dir: str | Path) -> Optional[dict]:
    p = Path(work_dir) / CHECKPOINT
    if not p.exists():
        return None
    d = np.load(p)
    sp = d["sens_prev"]
    return {"iteration": int(d["iteration"]), "alive_mask": d["alive_mask"],
            "sens_prev": (sp if sp.size else None)}
