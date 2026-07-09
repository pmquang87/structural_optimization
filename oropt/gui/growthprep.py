"""Launch the growth-mesh PREPARE step in an *isolated subprocess* and read its
outcome back from files — how the dashboard's ⚙️ Generate button runs
:mod:`oropt.growthmesh` without hosting TetGen inside the Streamlit process.

TetGen allocates the whole candidate mesh in-process; on a large model that was
>50 GB inside the GUI, freezing and eventually OOM-killing the entire dashboard
— and a native TetGen crash is uncatchable in-process, the same reason
off-screen pyvista renders run isolated (:mod:`oropt._render`). So the GUI only
*launches* ``python -m oropt.growthmesh --config … --json …`` with the queue
runner's detached flags and then *reads files*, exactly like the run machinery:

* the on-screen config is frozen into the scratch folder first, so unsaved
  region edits are honoured and later edits can't change a launched PREPARE
  (the queue's snapshot-config idea);
* the child's stdout/stderr stream into a log the panel tails while running;
* on success the CLI's ``--json`` report is parsed back into the same
  :class:`~oropt.growthmesh.GrowthMeshReport` the panel has always rendered.

Liveness is a parent-written pid file checked with
:func:`oropt.status.pid_alive`, so a session that lost its process handle (page
reload, GUI restart) re-attaches from the files alone — closing the browser
never orphans the state, mirroring how runs are monitored.

Streamlit-free (like :mod:`oropt.gui.runstate` / :mod:`oropt.gui.queue_store`)
so it is fast to import and hermetically unit-testable.
"""
from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from oropt import status as st_io
from oropt.config import Config
from oropt.growthmesh import GrowthMeshReport, report_from_dict
from oropt.queue_runner import spawn_detached

#: scratch sub-folder of the config's run folder holding one PREPARE launch
PREPARE_DIRNAME = "growth_mesh_prepare"
CONFIG_NAME = "config.yaml"      # frozen on-screen config the child reads
LOG_NAME = "prepare.log"         # child stdout+stderr, tailed by the panel
REPORT_NAME = "report.json"      # the CLI's --json report (success only)
PIDFILE = "prepare.pid"          # parent-written child pid (liveness)

#: what the CLI's main() prefixes a caught setup/guard error with
_ERROR_PREFIX = "[oropt] growth-mesh: ERROR:"

IDLE, RUNNING, DONE, FAILED = "idle", "running", "done", "failed"


@dataclass(frozen=True)
class PrepareStatus:
    """One reading of a PREPARE scratch folder, derived from its files only."""
    state: str                            # IDLE | RUNNING | DONE | FAILED
    report: Optional[GrowthMeshReport]    # the parsed --json report when DONE
    error: str                            # short reason when FAILED
    log_tail: str                         # last lines of the child's output
    pid: int                              # live child pid when RUNNING


def prepare_dir(cfg: Config, anchor: str | Path | None = None) -> Path:
    """The PREPARE scratch folder for *cfg* — inside the run folder, following
    the run-folder model (blank ``work_dir`` → the case directory itself).

    A *relative* run folder is resolved against *anchor* (the GUI passes its
    PROJECT_ROOT — the cwd it launches the PREPARE child with). Without the
    anchor, the parent resolved the scratch folder against the *GUI process's*
    cwd while the child resolved ``--config``/``--json`` against PROJECT_ROOT:
    with the two cwds differing, the child couldn't find the frozen config (or
    wrote its report where :func:`read_status` never looks) and the panel
    reported "ended without writing a report"."""
    p = Path(cfg.run_folder()) / PREPARE_DIRNAME
    if anchor is not None and not p.is_absolute():
        p = Path(anchor) / p
    return p


def _read_pid(prep: Path) -> int:
    try:
        return int((prep / PIDFILE).read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        return 0


def _tail(prep: Path, max_lines: int = 30) -> str:
    try:
        text = (prep / LOG_NAME).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return "\n".join(text.strip().splitlines()[-max_lines:])


def _error_from_log(log_tail: str) -> str:
    """The CLI's own ERROR line when it caught the failure; otherwise a generic
    crash message (a native TetGen abort prints nothing useful)."""
    lines = log_tail.splitlines()
    for line in reversed(lines):
        if line.startswith(_ERROR_PREFIX):
            return line[len(_ERROR_PREFIX):].strip()
    last = next((ln for ln in reversed(lines) if ln.strip()), "")
    return ("the PREPARE subprocess ended without writing a report (crash, "
            "kill or out-of-memory)" + (f" — last output: {last}" if last
                                        else ""))


def start(cfg: Config, prep: Path, size_factor: float, min_ratio: float,
          cwd: str | Path) -> int:
    """Freeze *cfg* into *prep* and launch the PREPARE CLI on it, detached,
    with stdout/stderr streaming into the log file.

    Returns the child pid (also written to the pid file, so any session can
    re-attach). Raises ``RuntimeError`` when a PREPARE child is already live
    for this folder and ``OSError`` when the interpreter can't be launched.
    """
    if st_io.pid_alive(_read_pid(prep)):
        raise RuntimeError("a growth-mesh PREPARE subprocess is already "
                           f"running for {prep} — wait for it or cancel it")
    prep.mkdir(parents=True, exist_ok=True)
    for name in (REPORT_NAME, LOG_NAME, PIDFILE):   # no stale outcome survives
        (prep / name).unlink(missing_ok=True)
    cfg_path = prep / CONFIG_NAME
    cfg.to_yaml(cfg_path)
    cmd = [sys.executable, "-m", "oropt.growthmesh",
           "--config", str(cfg_path),
           "--size-factor", str(float(size_factor)),
           "--min-ratio", str(float(min_ratio)),
           "--json", str(prep / REPORT_NAME)]
    # The child inherits the handle and keeps writing after this one closes.
    with open(prep / LOG_NAME, "wb") as fh:
        proc = spawn_detached(cmd, cwd, stdout=fh, stderr=subprocess.STDOUT)
    (prep / PIDFILE).write_text(str(proc.pid), encoding="utf-8")
    return proc.pid


def read_status(prep: Path) -> PrepareStatus:
    """Classify the scratch folder: a live pid wins; else a report means
    success; else a pid file left behind means the child died reportless
    (``start`` clears all three files, so nothing here is stale); else
    nothing was ever launched."""
    pid = _read_pid(prep)
    tail = _tail(prep)
    if st_io.pid_alive(pid):
        return PrepareStatus(RUNNING, None, "", tail, pid)
    report_path = prep / REPORT_NAME
    if report_path.is_file():
        try:
            rep = report_from_dict(
                json.loads(report_path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            return PrepareStatus(FAILED, None,
                                 f"unreadable {REPORT_NAME}: {exc}", tail, 0)
        return PrepareStatus(DONE, rep, "", tail, 0)
    if pid:
        return PrepareStatus(FAILED, None, _error_from_log(tail), tail, 0)
    return PrepareStatus(IDLE, None, "", tail, 0)


def cancel(prep: Path) -> None:
    """Kill a live PREPARE child (tree-kill on Windows, SIGTERM elsewhere) —
    the same force-kill the dashboard offers for a stuck run. No-op when
    nothing is alive. PREPARE writes decks only after every guard passed, so
    killing it mid-flight leaves no partial deck set behind."""
    pid = _read_pid(prep)
    if not st_io.pid_alive(pid):
        return
    if sys.platform == "win32":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        import os
        import signal
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
