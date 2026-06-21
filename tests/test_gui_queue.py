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


def test_duplicate_work_dirs_ignores_finished():
    q = qs.RunQueue()
    qs.add(q, "a", work_dir="/x/work")
    qs.add(q, "b", work_dir="/x/work")          # collides with a (both active)
    done = qs.add(q, "c", work_dir="/x/work")
    qs.mark(q, done.id, qs.DONE)                # finished -> excluded
    qs.add(q, "d", work_dir="/y/work")          # unique
    assert qs.duplicate_work_dirs(q) == {"/x/work"}


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
    cfg.work_dir = ""                                   # blank -> <case_dir>/work
    p = tmp_path / "c.yaml"
    cfg.to_yaml(p)
    assert qs.resolve_work_dir(p, tmp_path) == str((tmp_path / "work").resolve())
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
