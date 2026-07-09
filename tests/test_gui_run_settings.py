"""Input-tab run controls: concurrent-solver count, the iteration-0 reuse toggle,
and the 'copy iter_0000 from another run' seed tool."""
from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest

import oropt
import oropt.gui.runstate as runstate
from oropt.config import Config, LoadCase

_APP = Path(oropt.__file__).resolve().parent / "gui" / "app.py"


def _cfg(tmp_path, n_cases=2) -> tuple[Path, Path]:
    work = tmp_path / "work"
    work.mkdir()
    cfg = Config()
    cfg.work_dir = str(work)
    cfg.load_cases = [LoadCase(name=f"c{i}", stem=f"c{i}", sigma_allow=1.0)
                      for i in range(n_cases)]
    cfg_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(cfg_path)
    return cfg_path, work


def _boot(cfg_path, monkeypatch):
    monkeypatch.setattr(runstate, "find_active_run", lambda *a, **k: None)
    at = AppTest.from_file(str(_APP), default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(cfg_path)).run()
    return at


def test_concurrency_enabled_with_multiple_cases(tmp_path, monkeypatch):
    cfg_path, _ = _cfg(tmp_path, n_cases=2)
    at = _boot(cfg_path, monkeypatch)
    assert not at.exception
    conc = at.number_input(key="run_concurrency")
    assert not conc.disabled and conc.max == 2            # capped at #load cases


def test_concurrency_disabled_for_single_case(tmp_path, monkeypatch):
    cfg_path, _ = _cfg(tmp_path, n_cases=1)
    at = _boot(cfg_path, monkeypatch)
    assert not at.exception
    assert at.number_input(key="run_concurrency").disabled  # nothing to parallelise


def test_reuse_iter0_checkbox_present_and_on(tmp_path, monkeypatch):
    cfg_path, _ = _cfg(tmp_path, n_cases=2)
    at = _boot(cfg_path, monkeypatch)
    assert at.checkbox(key="reuse_iter0").value is True     # default on


def test_copy_iter0_button_copies_into_run_folder(tmp_path, monkeypatch):
    cfg_path, work = _cfg(tmp_path, n_cases=2)
    src = tmp_path / "oldrun"
    (src / "iter_0000").mkdir(parents=True)
    (src / "iter_0000" / "c0A001").write_bytes(b"anim")
    at = _boot(cfg_path, monkeypatch)
    at.text_input(key="copy_iter0_src").set_value(str(src)).run()
    at.button(key="copy_iter0_btn").click().run()
    assert not at.exception
    assert (work / "iter_0000" / "c0A001").is_file()        # seeded into this run
    assert any("copied iter_0000" in s.value for s in at.success)

def test_wall_budget_input_roundtrips(tmp_path, monkeypatch):
    cfg_path, _ = _cfg(tmp_path, n_cases=1)
    at = _boot(cfg_path, monkeypatch)
    assert not at.exception
    budget = at.number_input(key="run_wall_budget")
    assert budget.value == 0.0                              # default: unlimited
    budget.set_value(12.0).run()
    assert not at.exception
