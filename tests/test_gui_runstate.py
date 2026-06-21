"""find_active_run: the GUI's Run state / Monitor follow whatever run is live."""
from __future__ import annotations

from pathlib import Path

from oropt.gui import queue_store as qs
from oropt.gui.runstate import find_active_run


def test_active_run_prefers_selected_config():
    q = qs.RunQueue()
    res = find_active_run("/sel", q, is_running=lambda w: str(w) == "/sel")
    assert res == (Path("/sel"), "selected config")


def test_active_run_follows_running_queue_entry_in_its_own_folder():
    # the bug: selected config's folder is idle, but a queued run is live in its
    # own (de-duplicated) folder -> find it so the sidebar/Monitor don't show idle.
    q = qs.RunQueue(entries=[
        qs.QueueEntry(id="a", config="/cfgs/pull.yaml",
                      work_dir="/x/run_2", state=qs.RUNNING),
    ])
    res = find_active_run("/sel", q, is_running=lambda w: str(w) == "/x/run_2")
    assert res == (Path("/x/run_2"), "pull.yaml")


def test_active_run_none_when_nothing_live():
    q = qs.RunQueue(entries=[
        qs.QueueEntry(id="a", config="c", work_dir="/x/run", state=qs.RUNNING),
    ])
    # RUNNING in the queue but the pid is gone (crashed runner) -> not active
    assert find_active_run("/sel", q, is_running=lambda w: False) is None


def test_active_run_skips_blank_folder_and_non_running_state():
    q = qs.RunQueue(entries=[
        qs.QueueEntry(id="a", config="c", work_dir="", state=qs.RUNNING),    # no folder
        qs.QueueEntry(id="b", config="d", work_dir="/y", state=qs.PENDING),  # not running
    ])
    assert find_active_run("/sel", q, is_running=lambda w: str(w) == "/y") is None
