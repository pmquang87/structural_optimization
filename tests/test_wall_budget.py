"""The whole-run wall-clock budget (``run.max_wall_hours``).

Long runs on shared machines / clusters get killed mid-solve by session limits,
losing the in-flight iteration and leaving a stale 'running' status. The budget
lets the run stop CLEANLY at an iteration boundary instead: state ``stopped``
(not failed), the message names the budget, the checkpoint written at the end of
the previous iteration makes it resumable, and the post-run steps still execute.
Hermetic: ``run_solver`` / ``extract`` stubbed, the clock faked.
"""
from __future__ import annotations

import numpy as np
import pytest

from oropt import loop as loop_mod
from oropt import status as st
from oropt.config import Config, DispConstraint, LoadCase
from oropt.results import Results
from oropt.runner import RunResult
from oropt.validate import check_config, has_errors

ELEM_IDS = np.array([60000001, 60000002], dtype=np.int64)


def _results() -> Results:
    return Results(element_ids=ELEM_IDS.copy(),
                   energy=np.array([2.0, 1.0]),
                   vonmises=np.full(ELEM_IDS.size, 100.0),
                   sigma_max=100.0, disp=0.1, disp_node_id=60000001,
                   disps={60000001: 0.1})


def _cfg(case_dir, out_dir, max_iter=5) -> Config:
    cfg = Config()
    cfg.model.case_dir = str(case_dir)
    cfg.model.design_part_id = 60000000
    cfg.model.design_node_min = 60000000
    cfg.model.bc_group_id = 60000000
    cfg.beso.max_iter = max_iter
    cfg.beso.filter_radius = 0.0
    cfg.load_cases = [LoadCase(
        name="a", stem="lc_a", sigma_allow=250.0,
        disp_constraints=[DispConstraint(node_id=60000001, d_allow=1.0)])]
    cfg.work_dir = str(out_dir)
    return cfg


@pytest.fixture
def case_env(tmp_path, mini_deck_path, mini_engine_path):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "lc_a_0000.rad").write_text(
        mini_deck_path.read_text(encoding="utf-8"), encoding="utf-8")
    (case_dir / "lc_a_0001.rad").write_text(
        mini_engine_path.read_text(encoding="utf-8"), encoding="utf-8")
    return case_dir, tmp_path / "out"


class _FakeClock:
    """time.time() stand-in; each solve advances it by ``step_s``."""

    def __init__(self, step_s: float):
        self.now = 1000.0
        self.step_s = step_s

    def __call__(self) -> float:
        return self.now


def _stub_solver(monkeypatch, clock: _FakeClock, solver_calls: list):
    def fake_run_solver(cfg, run_dir, stem=None):
        solver_calls.append(stem)
        clock.now += clock.step_s
        return RunResult(True, "ok", "NORMAL TERMINATION")

    def fake_extract(cfg, run_dir, keep_vtk=False, stem=None, disp_node_id=None,
                     disp_node_ids=None, exclude_element_ids=None):
        return _results()

    monkeypatch.setattr(loop_mod, "run_solver", fake_run_solver)
    monkeypatch.setattr(loop_mod, "extract", fake_extract)
    monkeypatch.setattr(loop_mod.time, "time", clock)


def test_budget_stops_cleanly_at_iteration_boundary(case_env, monkeypatch):
    """Each fake solve takes 30 min against a 0.25 h budget: iteration 0 runs to
    completion (a solve in flight is never cut short), then the boundary check
    stops the run as ``stopped`` — resumable, not failed."""
    case_dir, out = case_env
    cfg = _cfg(case_dir, out, max_iter=5)
    cfg.run.max_wall_hours = 0.25
    clock = _FakeClock(step_s=1800.0)
    solver_calls: list = []
    _stub_solver(monkeypatch, clock, solver_calls)

    status = loop_mod.run_optimization(cfg, log=lambda *_: None)

    assert status.state == "stopped"
    assert "wall-clock budget" in status.message
    assert "max_wall_hours" in status.message
    assert len(solver_calls) == 1          # iter 0 finished; iter 1 never started
    assert len(st.read_history(cfg.work())) == 1
    assert (cfg.work() / "checkpoint.npz").exists()   # resumable


def test_budget_off_by_default(case_env, monkeypatch):
    """max_wall_hours=0 (the default) never stops a run, whatever the clock says."""
    case_dir, out = case_env
    cfg = _cfg(case_dir, out, max_iter=3)
    assert cfg.run.max_wall_hours == 0.0
    clock = _FakeClock(step_s=1e6)          # each solve 'takes' ~278 h
    solver_calls: list = []
    _stub_solver(monkeypatch, clock, solver_calls)

    status = loop_mod.run_optimization(cfg, log=lambda *_: None)

    assert "wall-clock budget" not in (status.message or "")
    assert len(solver_calls) == 3           # ran to max_iter


def test_budget_roundtrips_yaml(tmp_path):
    cfg = Config()
    cfg.run.max_wall_hours = 12.5
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    assert Config.from_yaml(p).run.max_wall_hours == 12.5


def test_negative_budget_is_a_config_error(tmp_path):
    cfg = Config()
    cfg.model.case_dir = str(tmp_path)
    cfg.run.max_wall_hours = -1.0
    problems = check_config(cfg, probe_docker_image=False)
    assert has_errors(problems)
    assert any("max_wall_hours" in str(p) for p in problems)
