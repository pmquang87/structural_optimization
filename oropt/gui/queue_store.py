"""Disk-backed serial run queue for the oropt dashboard (Streamlit-free).

The dashboard reruns on a timer and keeps no server-side state, so the queue of
optimisation runs lives in a small JSON file on disk rather than in
``st.session_state``. A detached *queue runner* process
(``python -m oropt.queue_runner <queue.json>``) pops pending entries and runs
them strictly one at a time via ``python -m oropt.run`` — so the queue keeps
draining after the browser is closed, exactly like a single detached run does
today. The GUI and the runner share this module as the single on-disk source of
truth for the queue.

This module owns only the queue *data*: the JSON shape, a locked atomic
read-modify-write, and the pure state transitions (add / remove / reorder /
mark). It deliberately imports neither Streamlit nor the solver, so it is fast
to import and hermetically unit-testable (mirroring :mod:`oropt.gui.cases`).
Run *liveness* is not tracked here — :func:`oropt.status.is_running` stays the
single source of truth for "a solver is active", so the queue can never
double-launch.
"""
from __future__ import annotations

import dataclasses
import json
import os
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

# Entry lifecycle states.
PENDING = "pending"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
SKIPPED = "skipped"
ACTIVE_STATES = (PENDING, RUNNING)        # still occupy a work dir
FINISHED_STATES = (DONE, FAILED, SKIPPED)

QUEUE_FILENAME = "run_queue.json"


@dataclass
class QueueEntry:
    """One queued optimisation run: a config path plus its lifecycle state."""
    id: str
    config: str                  # config path exactly as entered by the user
    resume: bool = False         # launch the run with --resume
    state: str = PENDING         # pending | running | done | failed | skipped
    message: str = ""            # short human note (failure reason, etc.)
    work_dir: str = ""           # resolved run folder (abs) — display + collision


@dataclass
class RunQueue:
    """The whole queue: ordered entries plus runner-coordination flags."""
    entries: list = field(default_factory=list)   # list[QueueEntry], in run order
    paused: bool = False                           # runner stops before next entry
    runner_pid: int = 0                            # active queue-runner pid (0 = none)


def default_queue_path(project_root: str | Path) -> Path:
    """Conventional queue-file location (shared by the GUI and the runner)."""
    return Path(project_root) / QUEUE_FILENAME


def new_id() -> str:
    return uuid.uuid4().hex[:8]


# ---- (de)serialisation -----------------------------------------------------
_ENTRY_FIELDS = {f.name for f in dataclasses.fields(QueueEntry)}


def _queue_from_dict(data: dict) -> RunQueue:
    entries: list[QueueEntry] = []
    for row in (data.get("entries") or []):
        if not isinstance(row, dict) or "id" not in row or "config" not in row:
            continue                                    # tolerate junk rows
        entries.append(QueueEntry(**{k: v for k, v in row.items()
                                     if k in _ENTRY_FIELDS}))
    return RunQueue(entries=entries,
                    paused=bool(data.get("paused", False)),
                    runner_pid=int(data.get("runner_pid", 0) or 0))


def _queue_to_dict(q: RunQueue) -> dict:
    return {"entries": [asdict(e) for e in q.entries],
            "paused": q.paused, "runner_pid": q.runner_pid}


def load_queue(path: str | Path) -> RunQueue:
    """Read the queue, returning an empty one for a missing/corrupt file."""
    p = Path(path)
    if not p.exists():
        return RunQueue()
    try:
        return _queue_from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError, TypeError, ValueError):
        return RunQueue()


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def save_queue(path: str | Path, q: RunQueue) -> None:
    _atomic_write(Path(path), json.dumps(_queue_to_dict(q), indent=2))


# ---- locked read-modify-write ----------------------------------------------
@contextmanager
def _lock(path: str | Path, timeout: float = 10.0, stale_after: float = 60.0):
    """Best-effort cross-process mutex guarding a queue read-modify-write.

    Both the GUI (membership edits) and the runner (state transitions) mutate the
    same file, so each whole-file update is serialised behind ``<path>.lock``
    (created with ``O_EXCL``). A lock older than *stale_after* — a crashed holder
    — is broken so the queue can never wedge permanently.
    """
    lock = Path(str(path) + ".lock")
    start = time.time()
    while True:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            break
        except FileExistsError:
            try:
                age = time.time() - lock.stat().st_mtime
            except OSError:
                age = 0.0
            if age > stale_after or (time.time() - start) > timeout:
                try:                                    # break a stale/contended lock
                    lock.unlink()
                except OSError:
                    pass
                continue
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def mutate(path: str | Path, fn: Callable[[RunQueue], object]) -> object:
    """Apply ``fn(queue)`` under the lock and persist; returns ``fn``'s result.

    ``fn`` mutates the loaded :class:`RunQueue` in place (e.g. via the helpers
    below) and may return a value (e.g. the new entry).
    """
    with _lock(path):
        q = load_queue(path)
        result = fn(q)
        save_queue(path, q)
    return result


# ---- pure transitions (operate on an in-memory RunQueue) -------------------
def find(q: RunQueue, entry_id: str) -> QueueEntry | None:
    return next((e for e in q.entries if e.id == entry_id), None)


def add(q: RunQueue, config: str, resume: bool = False,
        work_dir: str = "") -> QueueEntry:
    e = QueueEntry(id=new_id(), config=str(config), resume=bool(resume),
                   work_dir=str(work_dir))
    q.entries.append(e)
    return e


def remove(q: RunQueue, entry_id: str) -> None:
    q.entries = [e for e in q.entries if e.id != entry_id]


def move(q: RunQueue, entry_id: str, delta: int) -> None:
    """Shift an entry earlier (delta<0) or later (delta>0) in the run order."""
    idx = next((i for i, e in enumerate(q.entries) if e.id == entry_id), None)
    if idx is None:
        return
    j = max(0, min(len(q.entries) - 1, idx + delta))
    if j != idx:
        q.entries[idx], q.entries[j] = q.entries[j], q.entries[idx]


def clear_finished(q: RunQueue) -> None:
    q.entries = [e for e in q.entries if e.state not in FINISHED_STATES]


def clear_all(q: RunQueue) -> None:
    # Keep a running entry so its live run stays tracked; drop everything else.
    q.entries = [e for e in q.entries if e.state == RUNNING]


def mark(q: RunQueue, entry_id: str, state: str, message: str = "") -> None:
    e = find(q, entry_id)
    if e is not None:
        e.state, e.message = state, message


def set_paused(q: RunQueue, value: bool) -> None:
    q.paused = bool(value)


def next_pending(q: RunQueue) -> QueueEntry | None:
    """First entry still waiting to run (queue order is run order)."""
    return next((e for e in q.entries if e.state == PENDING), None)


def counts(q: RunQueue) -> dict:
    c = {PENDING: 0, RUNNING: 0, DONE: 0, FAILED: 0, SKIPPED: 0}
    for e in q.entries:
        c[e.state] = c.get(e.state, 0) + 1
    return c


def duplicate_work_dirs(q: RunQueue) -> set[str]:
    """Run folders shared by more than one not-yet-finished entry.

    Such entries would overwrite each other's status/results (they still run
    serially, never at once); the GUI surfaces this so the user gives each its
    own ``work_dir``.
    """
    seen: set[str] = set()
    dup: set[str] = set()
    for e in q.entries:
        if e.state in ACTIVE_STATES and e.work_dir:
            key = os.path.normcase(e.work_dir)
            if key in seen:
                dup.add(e.work_dir)
            seen.add(key)
    return dup


def resolve_work_dir(config_path: str | Path, project_root: str | Path) -> str:
    """Absolute run folder a config's run would use — the dir the loop writes its
    PID/status into. Matches the GUI's resolution (relative paths against
    *project_root*, since runs are launched with that cwd). Blank on a
    bad/missing config so a queue entry can still carry an unknown work dir.
    """
    from oropt.config import Config
    try:
        cfg = Config.from_yaml(config_path)
    except Exception:  # noqa: BLE001 - any read/parse error -> work dir unknown
        return ""
    w = Path(cfg.run_folder())
    if not w.is_absolute():
        w = Path(project_root) / w
    try:
        return str(w.resolve())
    except OSError:
        return str(w)
