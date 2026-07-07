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
RUN_LOG = "run.log"        # the loop's tee'd stdout (see oropt.run._tee_log)

_HISTORY_COLS = ["iteration", "volume_fraction", "sigma_max", "disp",
                 "elements_alive", "feasible", "iter_wall_s", "or_termination",
                 "optimizer"]


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
    stress_excluded_elems: int = 0    # design elements whose von-Mises is ignored (stress-exclusion region)
    elements_candidate: int = 0       # growth-box candidate elements (start void, growable)
    elements_grown: int = 0           # candidates currently alive -- material grown into the boxes
    # Per-load-case feasibility breakdown, one dict per case with keys
    # name/sigma_max/sigma_allow/disp/d_allow/feasible. Each case is checked
    # against its OWN limits (every load case defines its sigma_allow/d_allow), so
    # the GUI can show what actually gated feasibility. The headline
    # sigma_allow/d_allow above are the limits of the worst-stress / worst-disp
    # case respectively (for a single-case run this is just that case). Empty until
    # the first iteration.
    cases: list = field(default_factory=list)
    # Live "what is running right now" line, refreshed *before* each long solve so
    # the GUI shows the current activity during the minutes a solve takes (which
    # iteration, which load case, and whether it is the fast tied-linear screen or
    # the full nonlinear solve). Blank between solves (the per-iteration published
    # status carries the completed metrics, not a live activity).
    activity: str = ""
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
    # A fresh file gets the current column set (incl. later-added ones like
    # ``optimizer``); an existing file keeps *its own* header, so appending to a
    # history written before a column existed never misaligns the CSV (the extra
    # key is dropped by extrasaction="ignore"). This matters for a run resumed
    # across an oropt upgrade, or one that switched optimiser mid-run.
    cols = _HISTORY_COLS
    if not new:
        with open(p, newline="", encoding="utf-8") as fh:
            header = fh.readline().strip()
        if header:
            cols = header.split(",")
    with open(p, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow(row)


def read_log_tail(work_dir: str | Path, n: int = 200) -> str:
    """Last *n* non-blank lines of the run's ``run.log`` (``""`` when absent).

    The GUI launches the loop detached with its stdout discarded, so ``run.log``
    (written by :func:`oropt.run._tee_log`) is the only durable record of the
    run's progress and — the reason this exists — the skip reason of every
    best-effort post-run step (d3plot / smooth / animate / report), which each
    only *log* their outcome. The Monitor tails it so those surface in the
    browser, mirroring the PREPARE panel's log tail. Best-effort: an unreadable
    log reads as empty rather than raising into the GUI.
    """
    p = Path(work_dir) / RUN_LOG
    try:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(s for s in lines[-n:] if s.strip())


def read_history(work_dir: str | Path) -> list[dict]:
    p = Path(work_dir) / HISTORY
    if not p.exists():
        return []
    with open(p, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def topology_snapshot_name(iteration: int) -> str:
    """Filename of the immutable per-iteration topology snapshot."""
    return f"topology_iter{int(iteration):04d}.vtu"


def write_topology(work_dir: str | Path, node_xyz: np.ndarray,
                   conn_rows: np.ndarray, alive_mask: np.ndarray,
                   fields: Optional[dict] = None,
                   filename: str = TOPOLOGY,
                   iteration: Optional[int] = None) -> None:
    """Write the current alive tetra mesh to ``filename`` (default
    ``topology_latest.vtu``) for the GUI.

    When *iteration* is given, also write an immutable per-iteration snapshot
    ``topology_iter{iteration:04d}.vtu`` so the whole topology evolution can be
    replayed (the latest file is overwritten each iteration). The grid is built
    once and saved to both paths.
    """
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
    out = Path(work_dir)
    grid.save(str(out / filename))
    if iteration is not None:
        grid.save(str(out / topology_snapshot_name(iteration)))


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
        from ctypes import wintypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        k32 = ctypes.windll.kernel32
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        h = k32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return False
        # OpenProcess succeeding only proves the kernel object still exists; a
        # handle held elsewhere (e.g. the parent/GUI) keeps it alive after the
        # process has exited. Query the exit code to tell a live process apart
        # from an exited-but-lingering one. Caveat: a process that genuinely exits
        # with code 259 (== STILL_ACTIVE) is indistinguishable from a running one
        # here — accepted, as our runs never exit with that code.
        try:
            code = wintypes.DWORD()
            if not k32.GetExitCodeProcess(h, ctypes.byref(code)):
                return False
            return code.value == STILL_ACTIVE
        finally:
            k32.CloseHandle(h)
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def is_running(work_dir: str | Path) -> bool:
    return pid_alive(read_pid(work_dir))


# ---- checkpoint (resume) ---------------------------------------------------
def save_checkpoint(work_dir: str | Path, iteration: int, alive_mask: np.ndarray,
                    sens_prev: Optional[np.ndarray] = None,
                    phi: Optional[np.ndarray] = None,
                    x: Optional[np.ndarray] = None) -> None:
    """Persist the state a ``--resume`` needs to continue *without* perturbing the
    design: the alive mask, the history-blended sensitivity, and each field-carrying
    optimiser's *own* continuous field.

    *phi* is the level-set's nodal field and *x* is HCA's per-element virtual
    density. Without its field a resumed run re-initialises it from the alive mask,
    which both perturbs the design and re-orders it by the current sensitivity rank
    (level-set), or discards the controller's sub-threshold memory (HCA). BESO/TOBS
    are stateless and pass neither. On an optimiser *switch* the destination reloads
    only the field that matches its own kind (see the loop), so a phi never lands in
    HCA's ``x`` or vice versa — the mismatched field is dropped and re-initialised."""
    np.savez(Path(work_dir) / CHECKPOINT, iteration=iteration,
             alive_mask=alive_mask,
             sens_prev=(sens_prev if sens_prev is not None else np.array([])),
             phi=(phi if phi is not None else np.array([])),
             x=(x if x is not None else np.array([])))


def checkpoint_iteration(work_dir: str | Path) -> Optional[int]:
    """The iteration a resume would start from (``None`` when no checkpoint).

    Reads only the scalar ``iteration`` member of ``checkpoint.npz`` (not the
    multi-MB alive-mask / phi arrays), so the GUI can cheaply show "continues
    from iteration N" and size an "N more iterations" extension without loading
    the whole checkpoint every rerun. This equals the loop's ``start_iter``
    (``save_checkpoint`` stores ``it + 1`` after each completed iteration)."""
    p = Path(work_dir) / CHECKPOINT
    if not p.exists():
        return None
    try:
        with np.load(p) as d:
            return int(d["iteration"])
    except (OSError, KeyError, ValueError):
        return None


def load_checkpoint(work_dir: str | Path) -> Optional[dict]:
    p = Path(work_dir) / CHECKPOINT
    if not p.exists():
        return None
    d = np.load(p)
    sp = d["sens_prev"]
    phi = d["phi"] if "phi" in d.files else np.array([])   # pre-phi checkpoints
    x = d["x"] if "x" in d.files else np.array([])         # pre-x (no HCA field)
    return {"iteration": int(d["iteration"]), "alive_mask": d["alive_mask"],
            "sens_prev": (sp if sp.size else None),
            "phi": (phi if phi.size else None),
            "x": (x if x.size else None)}
