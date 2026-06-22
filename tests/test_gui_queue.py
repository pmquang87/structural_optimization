"""Disk-backed serial run queue: pure transitions + persistence (Streamlit-free,
hermetic), the detached runner's serial-drain logic (patched so no real solver
launches), and a smoke test that the dashboard renders the new Queue tab.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import oropt
from oropt import queue_runner as qr
from oropt import status as st_io
from oropt.config import Config
from oropt.gui import queue_store as qs


# ---- pure transitions ------------------------------------------------------
def test_add_remove_and_reorder():
    q = qs.RunQueue()
    a = qs.add(q, "a.yaml")
    qs.add(q, "b.yaml")
    c = qs.add(q, "c.yaml")
    assert [e.config for e in q.entries] == ["a.yaml", "b.yaml", "c.yaml"]
    qs.move(q, c.id, -1)
    assert [e.config for e in q.entries] == ["a.yaml", "c.yaml", "b.yaml"]
    qs.move(q, a.id, -1)                       # already first -> clamped, no-op
    assert q.entries[0].id == a.id
    qs.remove(q, c.id)
    assert [e.config for e in q.entries] == ["a.yaml", "b.yaml"]


def test_next_pending_counts_and_clear():
    q = qs.RunQueue()
    a, b, c = (qs.add(q, n) for n in ("a", "b", "c"))
    qs.mark(q, a.id, qs.DONE)
    qs.mark(q, b.id, qs.RUNNING)
    assert qs.next_pending(q).id == c.id        # first still-pending entry
    assert qs.counts(q) == {qs.PENDING: 1, qs.RUNNING: 1, qs.DONE: 1,
                            qs.FAILED: 0, qs.SKIPPED: 0}
    qs.clear_finished(q)
    assert [e.id for e in q.entries] == [b.id, c.id]   # the done entry is dropped
    qs.clear_all(q)
    assert [e.id for e in q.entries] == [b.id]          # only the running one stays


def test_add_dedupes_colliding_work_dir():
    """Adding a run whose folder is already taken by an active entry gets its own
    suffixed folder, so queued runs never overwrite each other's results."""
    q = qs.RunQueue()
    a = qs.add(q, "a", work_dir="/x/work")
    b = qs.add(q, "b", work_dir="/x/work")      # collides with a -> suffixed
    c = qs.add(q, "c", work_dir="/x/work")      # collides with a & b -> next suffix
    assert a.work_dir == "/x/work"
    assert b.work_dir == "/x/work_2"
    assert c.work_dir == "/x/work_3"
    # a finished entry's folder is free to reuse (only active entries reserve one)
    qs.mark(q, a.id, qs.DONE)
    d = qs.add(q, "d", work_dir="/x/work")
    assert d.work_dir == "/x/work"              # a is done -> /x/work is free again
    # a blank work_dir (unknown folder) is never suffixed
    assert qs.add(q, "e").work_dir == ""


def test_duplicate_work_dirs_ignores_finished():
    # duplicate_work_dirs guards against collisions add() can't prevent (e.g. a
    # user editing an entry's folder), so build them directly rather than via add().
    q = qs.RunQueue(entries=[
        qs.QueueEntry(id="a", config="a", work_dir="/x/work"),
        qs.QueueEntry(id="b", config="b", work_dir="/x/work"),     # collides (active)
        qs.QueueEntry(id="c", config="c", work_dir="/x/work", state=qs.DONE),
        qs.QueueEntry(id="d", config="d", work_dir="/y/work"),     # unique
    ])
    assert qs.duplicate_work_dirs(q) == {"/x/work"}


def test_update_entry_edits_config_folder_and_resume():
    q = qs.RunQueue()
    e = qs.add(q, "a.yaml", work_dir="/x/work")
    qs.update_entry(q, e.id, config="b.yaml", work_dir="/y/work", resume=True)
    assert (e.config, e.work_dir, e.resume) == ("b.yaml", "/y/work", True)
    # only the fields passed change; a no-op id is ignored
    qs.update_entry(q, e.id, resume=False)
    assert e.config == "b.yaml" and e.resume is False
    qs.update_entry(q, "missing", config="z")    # unknown id -> no raise, no change
    assert e.config == "b.yaml"


# ---- persistence -----------------------------------------------------------
def test_save_load_roundtrip(tmp_path):
    qpath = tmp_path / "q.json"
    q = qs.RunQueue(paused=True, runner_pid=42)
    qs.add(q, "a.yaml", resume=True, work_dir="/w")
    qs.save_queue(qpath, q)
    r = qs.load_queue(qpath)
    assert r.paused is True and r.runner_pid == 42
    [e] = r.entries
    assert (e.config, e.resume, e.work_dir, e.state) == ("a.yaml", True, "/w",
                                                         qs.PENDING)


def test_load_missing_and_corrupt(tmp_path):
    assert qs.load_queue(tmp_path / "nope.json").entries == []
    bad = tmp_path / "bad.json"
    bad.write_text("{not json", encoding="utf-8")
    assert qs.load_queue(bad).entries == []
    junk = tmp_path / "junk.json"
    junk.write_text('{"entries": [{"nope": 1}, {"id": "x", "config": "c"}]}',
                    encoding="utf-8")
    assert [e.id for e in qs.load_queue(junk).entries] == ["x"]   # junk row dropped


def test_mutate_locked_read_modify_write(tmp_path):
    qpath = tmp_path / "q.json"
    e = qs.mutate(qpath, lambda q: qs.add(q, "a.yaml"))
    assert isinstance(e, qs.QueueEntry)
    qs.mutate(qpath, lambda q: qs.mark(q, e.id, qs.FAILED, "boom"))
    [r] = qs.load_queue(qpath).entries
    assert r.state == qs.FAILED and r.message == "boom"


# ---- work-dir resolution ---------------------------------------------------
def _write_cfg(tmp_path, name="cfg.yaml", work_dir_sub="w"):
    cfg = Config()
    cfg.model.case_dir = str(tmp_path)
    cfg.work_dir = str(tmp_path / work_dir_sub)
    p = tmp_path / name
    cfg.to_yaml(p)
    return p


def test_resolve_work_dir_blank_and_relative(tmp_path):
    cfg = Config()
    cfg.model.case_dir = str(tmp_path)
    cfg.work_dir = ""                                   # blank -> the case dir itself
    p = tmp_path / "c.yaml"
    cfg.to_yaml(p)
    assert qs.resolve_work_dir(p, tmp_path) == str(tmp_path.resolve())
    cfg.work_dir = "runs/r1"                             # relative -> vs project_root
    cfg.to_yaml(p)
    assert qs.resolve_work_dir(p, tmp_path) == str((tmp_path / "runs" / "r1").resolve())


def test_resolve_work_dir_missing_config_is_blank(tmp_path):
    assert qs.resolve_work_dir(tmp_path / "nope.yaml", tmp_path) == ""


# ---- runner: result classification -----------------------------------------
def test_classify_from_terminal_status(tmp_path):
    w = tmp_path / "w"
    w.mkdir()
    st_io.write_status(w, st_io.Status(state="converged"))
    assert qr.classify(str(w), 0)[0] == qs.DONE
    st_io.write_status(w, st_io.Status(state="stopped"))
    assert qr.classify(str(w), 0)[0] == qs.DONE
    st_io.write_status(w, st_io.Status(state="failed", message="solver died"))
    assert qr.classify(str(w), 1) == (qs.FAILED, "solver died")


def test_classify_without_status_falls_back_to_returncode(tmp_path):
    w = str(tmp_path / "never_ran")                     # no status.json
    assert qr.classify(w, 0)[0] == qs.DONE
    assert qr.classify(w, 2) == (qs.FAILED, "config rejected (validation errors)")
    assert qr.classify(w, 7)[0] == qs.FAILED
    assert qr.classify(w, None)[0] == qs.FAILED         # crashed runner reconcile


# ---- runner: serial drain (no real solver) ---------------------------------
class _FakeProc:
    """Stands in for the launched ``python -m oropt.run`` child: on wait() it
    writes a terminal status into the run's work dir, like the real loop, and
    exits 0."""
    def __init__(self, work, state="converged"):
        self._work, self._state = work, state

    def wait(self):
        Path(self._work).mkdir(parents=True, exist_ok=True)
        st_io.write_status(self._work, st_io.Status(state=self._state))
        return 0


def test_runner_drains_queue_one_at_a_time(tmp_path, monkeypatch):
    qpath = tmp_path / "q.json"
    c1 = _write_cfg(tmp_path, "a.yaml", "wa")
    c2 = _write_cfg(tmp_path, "b.yaml", "wb")
    for c in (c1, c2):
        qs.mutate(qpath, lambda q, c=c: qs.add(
            q, str(c), work_dir=qs.resolve_work_dir(c, tmp_path)))

    launched: list[str] = []

    def fake_spawn(cmd, cwd):
        cfg_path = cmd[cmd.index("--config") + 1]
        launched.append(cfg_path)
        return _FakeProc(qs.resolve_work_dir(cfg_path, tmp_path))

    monkeypatch.setattr(qr, "spawn_detached", fake_spawn)
    monkeypatch.setattr(qr.st_io, "is_running", lambda w: False)

    assert qr.main([str(qpath), "--project-root", str(tmp_path)]) == 0
    assert launched == [str(c1), str(c2)]               # ran in order, once each
    q = qs.load_queue(qpath)
    assert [e.state for e in q.entries] == [qs.DONE, qs.DONE]
    assert q.runner_pid == 0                            # released on exit


def test_run_argv_includes_work_dir_override():
    base = qr.run_argv("cfg.yaml", resume=False)
    assert "--work-dir" not in base                     # omitted when no override
    cmd = qr.run_argv("cfg.yaml", resume=True, work_dir="/x/run_2")
    assert cmd[cmd.index("--config") + 1] == "cfg.yaml"
    assert "--resume" in cmd
    assert cmd[cmd.index("--work-dir") + 1] == "/x/run_2"


def test_runner_launches_run_in_its_own_reserved_folder(tmp_path, monkeypatch):
    """Two queued configs whose folders collide are de-duplicated by add(), and
    the runner hands each its reserved folder via --work-dir so they never
    overwrite each other."""
    qpath = tmp_path / "q.json"
    c1 = _write_cfg(tmp_path, "a.yaml", "shared")       # both resolve to <tmp>/shared
    c2 = _write_cfg(tmp_path, "b.yaml", "shared")
    for c in (c1, c2):
        qs.mutate(qpath, lambda q, c=c: qs.add(
            q, str(c), work_dir=qs.resolve_work_dir(c, tmp_path)))

    work_dirs: list[str] = []

    def fake_spawn(cmd, cwd):
        work_dirs.append(cmd[cmd.index("--work-dir") + 1])
        return _FakeProc(cmd[cmd.index("--work-dir") + 1])

    monkeypatch.setattr(qr, "spawn_detached", fake_spawn)
    monkeypatch.setattr(qr.st_io, "is_running", lambda w: False)
    assert qr.main([str(qpath), "--project-root", str(tmp_path)]) == 0
    # second run was given its own suffixed folder, not the shared one
    assert len(work_dirs) == 2 and work_dirs[0] != work_dirs[1]
    assert work_dirs[1].endswith("_2")


def test_runner_skips_missing_config(tmp_path, monkeypatch):
    qpath = tmp_path / "q.json"
    qs.mutate(qpath, lambda q: qs.add(q, str(tmp_path / "gone.yaml")))
    monkeypatch.setattr(qr, "spawn_detached",
                        lambda *a, **k: pytest.fail("must not launch a missing cfg"))
    monkeypatch.setattr(qr.st_io, "is_running", lambda w: False)
    qr.main([str(qpath), "--project-root", str(tmp_path)])
    assert qs.load_queue(qpath).entries[0].state == qs.SKIPPED


def test_runner_paused_does_not_launch(tmp_path, monkeypatch):
    qpath = tmp_path / "q.json"
    c1 = _write_cfg(tmp_path)
    qs.mutate(qpath, lambda q: qs.add(q, str(c1)))
    qs.mutate(qpath, lambda q: qs.set_paused(q, True))
    monkeypatch.setattr(qr, "spawn_detached",
                        lambda *a, **k: pytest.fail("must not launch while paused"))
    monkeypatch.setattr(qr.st_io, "is_running", lambda w: False)
    qr.main([str(qpath), "--project-root", str(tmp_path)])
    assert qs.load_queue(qpath).entries[0].state == qs.PENDING


def test_second_runner_bows_out(tmp_path, monkeypatch):
    qpath = tmp_path / "q.json"

    def _seed(q):
        q.runner_pid = 999_999                          # another runner owns it
        qs.add(q, "x.yaml")
    qs.mutate(qpath, _seed)
    monkeypatch.setattr(qr.st_io, "pid_alive", lambda pid: pid == 999_999)
    monkeypatch.setattr(qr, "spawn_detached",
                        lambda *a, **k: pytest.fail("second runner must not run"))
    assert qr.main([str(qpath), "--project-root", str(tmp_path)]) == 0
    q = qs.load_queue(qpath)
    assert q.runner_pid == 999_999                      # left intact
    assert q.entries[0].state == qs.PENDING


def test_reconcile_resolves_orphaned_running_entry(tmp_path, monkeypatch):
    qpath = tmp_path / "q.json"
    c1 = _write_cfg(tmp_path)
    work = qs.resolve_work_dir(c1, tmp_path)
    Path(work).mkdir(parents=True, exist_ok=True)
    st_io.write_status(work, st_io.Status(state="converged"))

    def _seed(q):                                        # a crashed runner's leftover
        e = qs.add(q, str(c1), work_dir=work)
        e.state = qs.RUNNING
    qs.mutate(qpath, _seed)
    monkeypatch.setattr(qr.st_io, "is_running", lambda w: False)
    monkeypatch.setattr(qr, "spawn_detached",
                        lambda *a, **k: pytest.fail("nothing left to run"))
    qr.main([str(qpath), "--project-root", str(tmp_path)])
    assert qs.load_queue(qpath).entries[0].state == qs.DONE


# ---- dashboard smoke test --------------------------------------------------
def test_app_renders_with_queue_tab(tmp_path):
    """The Streamlit script renders end-to-end with the new Queue tab + sidebar
    queue controls (read-only render — no runner is spawned)."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

    cfg = Config()
    cfg.model.case_dir = str(tmp_path)
    cfg.work_dir = str(tmp_path / "work")
    cfg_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(cfg_path)

    app_file = Path(oropt.__file__).resolve().parent / "gui" / "app.py"
    # Cold-process import of streamlit/pyvista is slow on first run.
    at = AppTest.from_file(str(app_file), default_timeout=30)
    at.run()
    assert not at.exception
    assert any("Queue" in t.label for t in at.tabs)


def test_enqueue_persists_current_optimiser_selection(tmp_path, monkeypatch):
    """Regression: a run is started only via the queue, and picking an optimiser in
    a tab then clicking ➕ Add to queue must enqueue *that* optimiser. The sidebar
    action used to persist the config before the tabs wrote their widgets back into
    it, so a TOBS selection was silently enqueued/launched as BESO. Also guards that
    the ad-hoc ▶ Start button is gone (queue is the only launch path)."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

    # Don't touch the real project queue; don't let config validation disable the
    # ➕ Add to queue button.
    enqueued: list = []
    monkeypatch.setattr(qs, "mutate", lambda path, fn: enqueued.append(str(path)))
    monkeypatch.setattr("oropt.validate.check_config", lambda *a, **k: [])

    cfg = Config()
    cfg.model.case_dir = str(tmp_path)
    cfg.work_dir = str(tmp_path / "work")
    cfg.optimizer = "beso"                       # the on-disk default
    cfg_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(cfg_path)

    app_file = Path(oropt.__file__).resolve().parent / "gui" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=30)
    at.run()
    at.sidebar.text_input[0].set_value(str(cfg_path)).run()   # point at our config
    assert not at.exception

    # the ad-hoc single-run launcher is gone — only the queue's ▶ Start queue remains
    labels = [b.label for b in at.sidebar.button]
    assert "▶ Start" not in labels and "▶ Start queue" in labels

    # pick TOBS in the optimiser selectbox (rendered in the Constraints/BC tab)
    sb = next(s for s in at.selectbox if s.label == "Topology optimiser")
    sb.set_value("tobs").run()
    assert not at.exception

    # ➕ Add current config to queue -> must persist the on-screen TOBS selection
    add = next(b for b in at.sidebar.button
               if b.label == "➕ Add current config to queue")
    add.click().run()
    assert not at.exception
    assert Config.from_yaml(cfg_path).optimizer == "tobs"     # saved what's on screen
    assert enqueued, "➕ Add to queue did not enqueue the run"
