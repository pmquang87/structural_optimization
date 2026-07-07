"""Sidebar 💾 Save config button: single, left-panel, and deferred so it persists
the on-screen edits (not the stale on-disk config)."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

import oropt
import oropt.gui.runstate as runstate
from oropt.config import Config, LoadCase

_APP = Path(oropt.__file__).resolve().parent / "gui" / "app.py"


def _boot(tmp_path, monkeypatch):
    monkeypatch.setattr(runstate, "find_active_run", lambda *a, **k: None)
    cfg = Config()
    cfg.work_dir = str(tmp_path / "work")
    Path(cfg.work_dir).mkdir()
    cfg.optimizer = "beso"
    cfg.load_cases = [LoadCase(name="a", stem="gp", sigma_allow=1.0)]
    cfg_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(cfg_path)
    at = AppTest.from_file(str(_APP), default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(cfg_path)).run()
    return at, cfg_path


def test_save_button_in_sidebar_only(tmp_path, monkeypatch):
    at, _ = _boot(tmp_path, monkeypatch)
    assert not at.exception
    assert any(b.key == "save_config" for b in at.sidebar.button)   # left panel
    # the two old in-tab "Save config" buttons are gone (only the sidebar one left)
    labels = [b.label for b in at.button]
    assert sum("Save config" in (lbl or "") for lbl in labels) == 1


def test_save_persists_onscreen_optimizer_change(tmp_path, monkeypatch):
    at, cfg_path = _boot(tmp_path, monkeypatch)
    sel = [s for s in at.selectbox if s.label == "Topology optimiser"][0]
    sel.set_value("levelset").run()                          # on-screen edit
    at.button(key="save_config").click().run()
    assert not at.exception
    saved = Config.from_dict(Config.read_yaml_dict(cfg_path))
    assert saved.optimizer_name() == "levelset"             # deferred save caught it
    assert any("Saved to" in s.value for s in at.sidebar.success)
