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
import shutil
import sys
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
# Sub-folder (under the model's case directory) holding the immutable per-run
# config snapshots a queued run is launched from. See :func:`snapshot_config`.
QUEUE_CONFIG_DIRNAME = "queue_configs"


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
def _lock(path: str | Path, timeout: float = 30.0):
    """Cross-process mutex guarding a queue read-modify-write.

    Both the GUI (membership edits) and the runner (state transitions) mutate the
    same file, so each whole-file update is serialised behind an OS file lock
    (``flock`` on POSIX, ``msvcrt.locking`` on Windows) held on a persistent
    ``<path>.lock`` file.

    An OS lock (rather than the previous create-with-``O_EXCL`` + break-if-old
    scheme) because the kernel releases it automatically when the holder dies —
    no staleness heuristic at all, and therefore none of its races: two waiters
    could both judge a crashed holder's file stale, the second deleting the
    first's *fresh* lock so both entered the critical section and the last
    writer's queue update silently erased the other's; and after mere seconds of
    contention the old scheme deleted a perfectly *live* holder's lock. The lock
    file itself persists (it is never unlinked — unlinking is exactly what made
    breaking racy); genuinely stuck contention raises ``TimeoutError`` after
    *timeout* instead of corrupting the queue.
    """
    lock = Path(str(path) + ".lock")
    fd = os.open(str(lock), os.O_CREAT | os.O_RDWR)
    locked = False
    try:
        deadline = time.monotonic() + timeout
        while True:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"could not lock the run queue within {timeout:.0f}s "
                        f"({lock}): another oropt process is holding it")
                time.sleep(0.05)
        yield
    finally:
        if locked:
            try:
                if sys.platform == "win32":
                    import msvcrt
                    os.lseek(fd, 0, os.SEEK_SET)
                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
        os.close(fd)


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


def _unique_work_dir(base: str, taken: set[str]) -> str:
    """A folder named like *base* that is not already in *taken* (compared
    case-insensitively), suffixing ``_2`` / ``_3`` / … on collision. A blank
    *base* (run folder unknown) is returned unchanged."""
    if not base or os.path.normcase(base) not in taken:
        return base
    i = 2
    while True:
        cand = f"{base}_{i}"
        if os.path.normcase(cand) not in taken:
            return cand
        i += 1


def add(q: RunQueue, config: str, resume: bool = False,
        work_dir: str = "") -> QueueEntry:
    """Append a run to the queue.

    When *work_dir* would collide with another not-yet-finished entry's folder it
    is auto-suffixed (``…_2`` / ``…_3`` …) so each queued run gets its own run
    folder instead of overwriting another's status/results; the runner passes that
    folder to the run via ``--work-dir``. Pass a unique *work_dir* (or leave it
    blank when unknown) to opt out of the suffixing.
    """
    taken = {os.path.normcase(e.work_dir) for e in q.entries
             if e.state in ACTIVE_STATES and e.work_dir}
    e = QueueEntry(id=new_id(), config=str(config), resume=bool(resume),
                   work_dir=_unique_work_dir(str(work_dir), taken))
    q.entries.append(e)
    return e


def update_entry(q: RunQueue, entry_id: str, *, config: str | None = None,
                 resume: bool | None = None, work_dir: str | None = None) -> None:
    """Edit a queued entry's config path / resume flag / run folder in place.

    Only the fields passed (non-``None``) are changed. Intended for *pending*
    entries — the GUI restricts editing to those so a live run's folder can't move
    out from under it. The caller supplies the work_dir verbatim (no auto-suffix),
    so a user-entered collision still surfaces via :func:`duplicate_work_dirs`.
    """
    e = find(q, entry_id)
    if e is None:
        return
    if config is not None:
        e.config = str(config)
    if resume is not None:
        e.resume = bool(resume)
    if work_dir is not None:
        e.work_dir = str(work_dir)


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


def snapshot_config(source_path: str | Path,
                    dest_dir: str | Path | None = None) -> str:
    """Copy *source_path* to an immutable per-run snapshot and return its path.

    A queued run is launched from this frozen copy, never the working config, so a
    later edit to the original (or another enqueue that re-saves it) can't change a
    run already sitting in the queue — what you see when you add it is what runs.
    The copy lands in *dest_dir* (the GUI passes the model's case directory; falls
    back to a ``queue_configs/`` folder beside the source) under a unique
    ``<stem>_<id><suffix>`` name. It is a faithful *byte* copy (not a
    :class:`~oropt.config.Config` round-trip) so nothing in the file is normalised
    or dropped.
    """
    src = Path(source_path)
    d = Path(dest_dir) if dest_dir is not None else src.parent / QUEUE_CONFIG_DIRNAME
    d.mkdir(parents=True, exist_ok=True)
    snap = d / f"{src.stem}_{new_id()}{src.suffix or '.yaml'}"
    shutil.copyfile(src, snap)
    return str(snap)


def resolve_case_dir(config_path: str | Path, project_root: str | Path) -> str:
    """Absolute model case directory a config points at (``model.case_dir``).

    Resolved like the run folder (relative paths against *project_root*, since runs
    launch with that cwd); a blank/default ``"."`` resolves to *project_root*. Blank
    on a bad/missing config. The queue stores each run's frozen config snapshot here
    so it travels with the model/case data rather than the working config's folder.
    """
    from oropt.config import Config
    try:
        cfg = Config.from_yaml(config_path)
    except Exception:  # noqa: BLE001 - any read/parse error -> case dir unknown
        return ""
    c = Path(cfg.model.case_dir or ".")
    if not c.is_absolute():
        c = Path(project_root) / c
    try:
        return str(c.resolve())
    except OSError:
        return str(c)


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
