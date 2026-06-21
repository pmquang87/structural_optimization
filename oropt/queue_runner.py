"""Detached serial runner for the dashboard's run queue.

Launched (detached, like a single run) by :mod:`oropt.gui.app` as::

    python -m oropt.queue_runner <queue.json> [--project-root DIR]

It pops pending entries from the shared queue file and runs them strictly one at
a time via ``python -m oropt.run --config <cfg> [--resume]``, waiting for each to
finish *completely* (the child only returns after the loop's post-processing)
before starting the next — so only one solver process is ever live. Like a single
run today, the runner is detached, so it keeps draining the queue after the
browser is closed and exits when the queue empties or is paused.

:func:`oropt.status.is_running` stays the single source of truth for "a solver is
active": the runner waits on the child it launched and additionally refuses to
launch while that run's work dir is already busy, so it can never double-launch.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from oropt import status as st_io
from oropt.gui import queue_store as qs

POLL_S = 2.0   # cadence for re-checking liveness while a work dir is busy


# ---- process launch --------------------------------------------------------
def spawn_detached(cmd: list[str], cwd: str | Path) -> subprocess.Popen:
    """Start *cmd* as a process that outlives its parent (and the browser).

    Same detached flags the dashboard uses for a single run, so a queued run is
    indistinguishable from a manually launched one — and the runner can be killed
    or the machine rebooted mid-run without taking the solve down with it.
    """
    flags = 0
    if sys.platform == "win32":
        flags = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    return subprocess.Popen(cmd, cwd=str(cwd), creationflags=flags,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def spawn_runner(queue_path: str | Path, project_root: str | Path) -> subprocess.Popen:
    """Launch the detached queue runner for *queue_path* (called by the GUI)."""
    cmd = [sys.executable, "-m", "oropt.queue_runner", str(queue_path),
           "--project-root", str(project_root)]
    return spawn_detached(cmd, project_root)


def run_argv(config_path: str | Path, resume: bool,
             work_dir: str | Path | None = None) -> list[str]:
    cmd = [sys.executable, "-m", "oropt.run", "--config", str(config_path)]
    if resume:
        cmd.append("--resume")
    if work_dir:                       # the entry's (possibly de-duplicated) folder
        cmd += ["--work-dir", str(work_dir)]
    return cmd


# ---- result classification -------------------------------------------------
def classify(work: str, returncode: int | None) -> tuple[str, str]:
    """Map a finished run to a queue (state, message) from the status it wrote,
    falling back to the child's exit code.

    The loop writes a terminal ``status.json`` (``converged`` / ``stopped`` /
    ``failed``) for any run that started; a config rejected by the fail-fast
    validation never gets that far, so the exit code (``2``) is the only signal.
    """
    status = st_io.read_status(work) if work else None
    state = status.state if status else ""
    if state == "converged":
        return qs.DONE, "converged"
    if state == "stopped":
        return qs.DONE, status.message or "stopped"
    if state == "failed":
        return qs.FAILED, status.message or "run failed"
    if returncode is None:
        return qs.FAILED, "run ended without a terminal status"
    if returncode == 0:
        return qs.DONE, ""
    if returncode == 2:
        return qs.FAILED, "config rejected (validation errors)"
    return qs.FAILED, f"run exited with code {returncode}"


# ---- runner lifecycle ------------------------------------------------------
def _claim_singleton(queue_path: str | Path) -> bool:
    """Become the active runner, or return False if a live runner already owns
    the queue (so a double Start spawns a runner that immediately bows out)."""
    def fn(q: qs.RunQueue) -> bool:
        if (q.runner_pid and q.runner_pid != os.getpid()
                and st_io.pid_alive(q.runner_pid)):
            return False
        q.runner_pid = os.getpid()
        return True
    return bool(qs.mutate(queue_path, fn))


def _release_singleton(queue_path: str | Path) -> None:
    def fn(q: qs.RunQueue) -> None:
        if q.runner_pid == os.getpid():
            q.runner_pid = 0
    qs.mutate(queue_path, fn)


def _reconcile(queue_path: str | Path, project_root: str | Path) -> None:
    """Resolve entries a previous (crashed/killed) runner left in ``running``.

    The run itself is detached, so it kept going; classify it from its work dir
    if it has since finished, otherwise leave it ``running`` for the wait below.
    """
    def fn(q: qs.RunQueue) -> None:
        for e in q.entries:
            if e.state == qs.RUNNING:
                work = e.work_dir or qs.resolve_work_dir(e.config, project_root)
                e.work_dir = work
                if not work or not st_io.is_running(work):
                    e.state, e.message = classify(work, None)
    qs.mutate(queue_path, fn)


def _process_next(queue_path: str | Path, project_root: str | Path) -> bool:
    """Run the next pending entry to completion.

    Returns True if an entry was handled (caller should loop again), False when
    the queue is drained or paused.
    """
    # Claim the head pending entry atomically: mark it running (or skipped) under
    # the lock so neither the GUI nor a second runner can pick it up again.
    def claim(q: qs.RunQueue):
        if q.paused:
            return None
        e = qs.next_pending(q)
        if e is None:
            return None
        # Prefer the entry's own (possibly de-duplicated / user-edited) folder;
        # only fall back to resolving it from the config when it was never set.
        work = e.work_dir or qs.resolve_work_dir(e.config, project_root)
        e.work_dir = work
        if not Path(e.config).exists():
            e.state, e.message = qs.SKIPPED, "config file not found"
            return ("skip",)
        e.state, e.message = qs.RUNNING, ""
        return ("run", e.id, e.config, e.resume, work)

    claimed = qs.mutate(queue_path, claim)
    if claimed is None:
        return False
    if claimed[0] == "skip":
        return True
    _, entry_id, config, resume, work = claimed

    # Never double-launch: if that work dir is already busy (e.g. a manual ▶ Start
    # of the same config), wait for it to free up rather than start a second run.
    while work and st_io.is_running(work):
        time.sleep(POLL_S)

    # Pass the entry's folder as --work-dir so the run writes exactly where the
    # queue reserved (its own de-duplicated folder), not wherever the shared
    # config's work_dir points.
    proc = spawn_detached(run_argv(config, resume, work_dir=work), project_root)
    # The child writes its own PID once the loop starts; waiting on the handle we
    # own means we resume only after the run (incl. post-processing) fully ends.
    returncode = proc.wait()
    state, message = classify(work, returncode)
    qs.mutate(queue_path, lambda q: qs.mark(q, entry_id, state, message))
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="oropt.queue_runner",
        description="Serial runner for the oropt dashboard's run queue")
    ap.add_argument("queue_file", help="path to the queue JSON file")
    ap.add_argument("--project-root", default=os.getcwd(),
                    help="root the run subprocesses are launched from (their cwd)")
    args = ap.parse_args(argv)
    queue_path = Path(args.queue_file)
    project_root = Path(args.project_root)

    if not _claim_singleton(queue_path):
        return 0                       # another live runner already owns the queue
    try:
        _reconcile(queue_path, project_root)
        while _process_next(queue_path, project_root):
            pass
    finally:
        _release_singleton(queue_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
