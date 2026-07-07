"""Sidebar ↻ Resume: continue a stopped run from its checkpoint, saving on-screen
parameter / optimiser edits first (deferred like ➕ Add to queue) and optionally
extending max_iter so the run continues past where it stopped."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from streamlit.testing.v1 import AppTest

import oropt
import oropt.gui.runstate as runstate
from oropt import queue_runner
from oropt import status as st_io
from oropt.config import Config, LoadCase

_APP = Path(oropt.__file__).resolve().parent / "gui" / "app.py"


def _make_stopped_run(tmp_path, *, optimizer="beso", max_iter=5, ckpt_iter=5) -> Path:
    """A config + work dir with a checkpoint, i.e. a run that can be continued."""
    work = tmp_path / "work"
    work.mkdir()
    cfg = Config()
    cfg.work_dir = str(work)
    cfg.optimizer = optimizer
    cfg.active_opts().max_iter = max_iter
    cfg.load_cases = [LoadCase(name="a", stem="gp", sigma_allow=1.0)]
    cfg_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(cfg_path)
    st_io.write_status(work, st_io.Status(state="stopped", iteration=ckpt_iter - 1,
                                          message="stop requested"))
    st_io.save_checkpoint(work, ckpt_iter, np.zeros(3, bool))
    return cfg_path


def _boot(cfg_path, monkeypatch):
    """Load the app on *cfg_path* with no live run and launches captured."""
    monkeypatch.setattr(runstate, "find_active_run", lambda *a, **k: None)
    calls: list[list[str]] = []
    monkeypatch.setattr(queue_runner, "spawn_detached",
                        lambda cmd, cwd, *a, **k: calls.append(list(cmd)))
    at = AppTest.from_file(str(_APP), default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(cfg_path)).run()
    return at, calls


def test_resume_extends_max_iter_and_launches(tmp_path, monkeypatch):
    # Run reached max_iter (5); ask for 10 more, continue.
    cfg_path = _make_stopped_run(tmp_path, max_iter=5, ckpt_iter=5)
    at, calls = _boot(cfg_path, monkeypatch)
    at.number_input(key="resume_add_iters").set_value(10).run()
    at.button(key="resume_run").click().run()
    assert not at.exception
    assert len(calls) == 1 and "--resume" in calls[0] and str(cfg_path) in calls[0]
    saved = Config.from_dict(Config.read_yaml_dict(cfg_path))
    assert saved.active_opts().max_iter == 15          # 5 (next iter) + 10 more


def test_resume_noop_guard_when_already_at_max_iter(tmp_path, monkeypatch):
    # On-screen Max iterations == where it stopped, no extension -> nothing to do;
    # warn, don't launch. (Set mi_beso explicitly: the keyed widget is sticky from
    # the first app render on DEFAULT_CFG, so we pin the on-screen value here.)
    cfg_path = _make_stopped_run(tmp_path, max_iter=5, ckpt_iter=5)
    at, calls = _boot(cfg_path, monkeypatch)
    at.number_input(key="mi_beso").set_value(5).run()
    at.button(key="resume_run").click().run()
    assert not at.exception
    assert calls == []                                 # never launched
    assert any("nothing to continue" in w.value for w in at.warning)


def test_resume_saves_onscreen_optimizer_change(tmp_path, monkeypatch):
    # Stopped before max_iter (5 < 20); switch optimiser on-screen, then continue:
    # the resume must persist the new optimiser, not the stale on-disk one.
    cfg_path = _make_stopped_run(tmp_path, optimizer="beso", max_iter=20, ckpt_iter=5)
    at, calls = _boot(cfg_path, monkeypatch)
    sel = [s for s in at.selectbox if s.label == "Topology optimizer"][0]
    sel.set_value("levelset").run()
    at.button(key="resume_run").click().run()
    assert not at.exception
    assert len(calls) == 1 and "--resume" in calls[0]
    saved = Config.from_dict(Config.read_yaml_dict(cfg_path))
    assert saved.optimizer_name() == "levelset"        # on-screen change persisted


def test_resume_disabled_without_checkpoint(tmp_path, monkeypatch):
    cfg_path = _make_stopped_run(tmp_path)
    (Path(cfg_path).parent / "work" / st_io.CHECKPOINT).unlink()   # no checkpoint
    at, calls = _boot(cfg_path, monkeypatch)
    assert not at.exception
    assert at.button(key="resume_run").disabled                    # nothing to resume
