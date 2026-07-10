"""Switching configs in the sidebar re-seeds every cfg-bound keyed widget.

Keyed Streamlit widgets persist their state across reruns and ignore a changed
``value=`` default — right while editing ONE config, but loading a *different*
config used to leave every keyed widget (optimizer knobs, run settings,
manufacturing constraints, animation look) showing the previous config's
values, which 💾 Save / ➕ enqueue then silently wrote over the new config.
The `_seed_widget` helper (the color_picker pattern, generalised) resets a
widget exactly when the loaded config's value changes underneath it, keeping
in-session edits otherwise.
"""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

import oropt
import oropt.gui.runstate as runstate
from oropt.config import Config, GrowthBox, LoadCase

import pytest

pytestmark = pytest.mark.gui   # Streamlit AppTest / pandas-pyarrow: excluded on non-Windows CI

_APP = Path(oropt.__file__).resolve().parent / "gui" / "app.py"


def _write_cfg(path: Path, *, max_iter=150, evolution_rate=0.02,
               wall_hours=0.0, archive=True, orig_elem_max=None) -> Path:
    cfg = Config()
    work = path.parent / (path.stem + "_work")
    work.mkdir(exist_ok=True)
    cfg.work_dir = str(work)
    cfg.load_cases = [LoadCase(name="a", stem="gp", sigma_allow=1.0)]
    cfg.beso.max_iter = max_iter
    cfg.beso.evolution_rate = evolution_rate
    cfg.beso.archive_iterations = archive
    cfg.run.max_wall_hours = wall_hours
    if orig_elem_max is not None:
        cfg.model.growth_boxes = [GrowthBox(
            name="g", carve=False, x_min=0, x_max=1, y_min=0, y_max=1,
            z_min=0, z_max=1)]
        cfg.model.growth_original_elem_max = orig_elem_max
    cfg.to_yaml(path)
    return path


def _boot(cfg_path, monkeypatch) -> AppTest:
    monkeypatch.setattr(runstate, "find_active_run", lambda *a, **k: None)
    at = AppTest.from_file(str(_APP), default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(cfg_path)).run()
    return at


def test_switch_reseeds_optimizer_and_run_widgets(tmp_path, monkeypatch):
    a = _write_cfg(tmp_path / "a.yaml", max_iter=150, evolution_rate=0.02,
                   wall_hours=0.0, archive=True)
    b = _write_cfg(tmp_path / "b.yaml", max_iter=7, evolution_rate=0.011,
                   wall_hours=3.0, archive=False)
    at = _boot(a, monkeypatch)
    assert at.number_input(key="mi_beso").value == 150

    at.sidebar.text_input[0].set_value(str(b)).run()
    assert not at.exception
    # every keyed widget now shows config B, not the sticky config-A values
    assert at.number_input(key="mi_beso").value == 7
    assert abs(at.number_input(key="evo_beso").value - 0.011) < 1e-12
    assert at.number_input(key="run_wall_budget").value == 3.0
    assert at.checkbox(key="arch_beso").value is False


def test_switch_then_save_keeps_new_configs_values(tmp_path, monkeypatch):
    """The real damage of the stale state: Save used to write config A's knobs
    into config B. After the switch, Save must round-trip B unchanged."""
    a = _write_cfg(tmp_path / "a.yaml", max_iter=150, evolution_rate=0.02)
    b = _write_cfg(tmp_path / "b.yaml", max_iter=7, evolution_rate=0.011)
    at = _boot(a, monkeypatch)
    at.sidebar.text_input[0].set_value(str(b)).run()
    at.button(key="save_config").click().run()
    assert not at.exception
    saved = Config.from_yaml(b)
    assert saved.beso.max_iter == 7                    # NOT config A's 150
    assert abs(saved.beso.evolution_rate - 0.011) < 1e-12


def test_in_session_edits_survive_reruns(tmp_path, monkeypatch):
    """Seeding must not fight the user: an on-screen edit (config file
    unchanged) persists across reruns instead of snapping back."""
    a = _write_cfg(tmp_path / "a.yaml", max_iter=150)
    at = _boot(a, monkeypatch)
    at.number_input(key="mi_beso").set_value(42).run()
    assert at.number_input(key="mi_beso").value == 42  # rerun did not reset it
    at.run()
    assert at.number_input(key="mi_beso").value == 42


def test_switch_reseeds_growth_elem_boundary(tmp_path, monkeypatch):
    """The carve-off element-id boundary is model-specific: carrying config A's
    575000 into config B (boundary 120000) misclassifies every generated
    candidate as 'original part' and the run aborts / silently never grows."""
    a = _write_cfg(tmp_path / "a.yaml", orig_elem_max=575000)
    b = _write_cfg(tmp_path / "b.yaml", orig_elem_max=120000)
    at = _boot(a, monkeypatch)
    thr = [n for n in at.number_input if n.key == "growth_orig_elem_max"]
    assert thr and thr[0].value == 575000

    at.sidebar.text_input[0].set_value(str(b)).run()
    assert not at.exception
    thr = [n for n in at.number_input if n.key == "growth_orig_elem_max"]
    assert thr and thr[0].value == 120000              # B's boundary, not A's
