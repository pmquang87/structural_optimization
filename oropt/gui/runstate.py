"""Which run is currently live — shared by the dashboard's Run state + Monitor.

A queued run writes to its own (possibly de-duplicated) ``--work-dir`` folder, not
the selected config's, so the sidebar used to show *idle* and the Monitor showed
nothing while the queue was actually running. :func:`find_active_run` resolves the
folder of whatever run is genuinely live so both can follow it.

Streamlit-free (like :mod:`oropt.gui.cases` / :mod:`oropt.gui.queue_store`) so it
is fast to import and hermetically unit-testable; liveness is delegated to
:func:`oropt.status.is_running`, the single source of truth.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

from oropt import status as st_io
from oropt.gui import queue_store as qs


def find_active_run(selected_work: str | Path, queue: qs.RunQueue,
                    is_running: Callable[[str | Path], bool] = st_io.is_running
                    ) -> Optional[tuple[Path, str]]:
    """The ``(folder, label)`` of whatever run is currently live, else ``None``.

    Checks the selected config's own folder first (label ``"selected config"``);
    otherwise the first RUNNING queue entry whose reserved folder has a live PID
    (label = that entry's config filename). *is_running* is injectable for tests.
    """
    if is_running(selected_work):
        return Path(selected_work), "selected config"
    for e in queue.entries:
        if e.state == qs.RUNNING and e.work_dir and is_running(e.work_dir):
            return Path(e.work_dir), Path(e.config).name
    return None
