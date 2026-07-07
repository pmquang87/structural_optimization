"""The null-solve guard.

A solve can reach NORMAL TERMINATION yet carry no load: zero von-Mises stress AND
zero strain energy across the whole design part -- the model never deformed. The
force landed on a constrained / rigid DOF, a contact interface never engaged, or
the deck was mis-exported. Every feasibility metric then reads 0 and passes every
limit trivially, and the BESO-family energy sensitivity is uniformly zero, so
without a guard the optimiser silently strips the part down to its protected
skeleton -- the ``opti_run5_Ti`` "lost continuity" failure. The loop must fail
loudly instead. Hermetic: ``run_solver`` / ``extract`` stubbed.
"""
from __future__ import annotations

import numpy as np
import pytest

from oropt import loop as loop_mod
from oropt import status as st
from oropt.config import Config, DispConstraint, LoadCase
from oropt.results import Results
from oropt.runner import RunResult

ELEM_IDS = np.array([60000001, 60000002], dtype=np.int64)


def _results(sigma: float, disp: float, energy) -> Results:
    return Results(element_ids=ELEM_IDS.copy(),
                   energy=np.asarray(energy, dtype=float),
                   vonmises=np.full(ELEM_IDS.size, sigma, dtype=float),
                   sigma_max=sigma, disp=disp, disp_node_id=60000001,
                   disps={60000001: disp})


# ---- Results.is_null_solve (unit) -------------------------------------------
def test_is_null_solve_true_when_all_zero():
    assert _results(0.0, 0.0, [0.0, 0.0]).is_null_solve is True


def test_is_null_solve_false_with_any_stress():
    assert _results(1e-6, 0.0, [0.0, 0.0]).is_null_solve is False


def test_is_null_solve_false_with_any_energy():
    assert _results(0.0, 0.0, [0.0, 1e-9]).is_null_solve is False


def test_is_null_solve_true_when_empty():
    """An empty design part (no surviving elements) is degenerate, not a valid
    design to keep optimising -- reported null."""
    r = Results(element_ids=np.array([], dtype=np.int64),
                energy=np.array([]), vonmises=np.array([]),
                sigma_max=float("nan"), disp=float("nan"), disp_node_id=None)
    assert r.is_null_solve is True


# ---- loop behaviour (hermetic) ----------------------------------------------
def _cfg(case_dir, out_dir, load_cases, max_iter=5) -> Config:
    cfg = Config()
    cfg.model.case_dir = str(case_dir)
    cfg.model.design_part_id = 60000000
    cfg.model.design_node_min = 60000000
    cfg.model.bc_group_id = 60000000
    cfg.beso.max_iter = max_iter
    cfg.beso.filter_radius = 0.0
    for lc in load_cases:
        if not lc.disp_constraints:
            lc.disp_constraints = [DispConstraint(node_id=60000001, d_allow=1.0)]
        if lc.sigma_allow is None:
            lc.sigma_allow = 250.0
    cfg.load_cases = load_cases
    cfg.work_dir = str(out_dir)
    return cfg


@pytest.fixture
def case_env(tmp_path, mini_deck_path, mini_engine_path):
    deck_text = mini_deck_path.read_text(encoding="utf-8")
    engine_text = mini_engine_path.read_text(encoding="utf-8")
    case_dir = tmp_path / "case"
    out_dir = tmp_path / "out"

    def make(stems):
        case_dir.mkdir(parents=True, exist_ok=True)
        for stem in stems:
            (case_dir / f"{stem}_0000.rad").write_text(deck_text, encoding="utf-8")
            (case_dir / f"{stem}_0001.rad").write_text(engine_text, encoding="utf-8")
        return case_dir, out_dir
    return make


def _stub_solver(monkeypatch, extract_fn, solver_calls):
    def fake_run_solver(cfg, run_dir, stem=None):
        solver_calls.append(stem)
        return RunResult(True, "ok", "NORMAL TERMINATION")
    monkeypatch.setattr(loop_mod, "run_solver", fake_run_solver)
    monkeypatch.setattr(loop_mod, "extract", extract_fn)


def test_null_solve_fails_run_at_iter0(case_env, monkeypatch):
    """A dead (zero-load) solve at iteration 0 fails the run with a clear message
    instead of carving material -- no sensitivity is built, no design update runs,
    and no successful iteration is recorded."""
    case_dir, out = case_env(("lc_a",))
    cfg = _cfg(case_dir, out, [LoadCase(name="a", stem="lc_a")], max_iter=5)
    solver_calls = []

    def fake_extract(cfg, run_dir, keep_vtk=False, stem=None, disp_node_id=None,
                     disp_node_ids=None, exclude_element_ids=None):
        return _results(0.0, 0.0, energy=[0.0, 0.0])
    _stub_solver(monkeypatch, fake_extract, solver_calls)

    status = loop_mod.run_optimization(cfg, log=lambda *_: None)

    assert status.state == "failed"
    assert "null solve" in status.message
    assert "carried no load" in status.message
    assert len(solver_calls) == 1                 # failed at iter 0, no further solves
    assert st.read_history(cfg.work()) == []      # nothing carved / no iter recorded


def test_null_solve_names_offending_case(case_env, monkeypatch):
    """Multi-case: only the push case is dead -> the failure message names it."""
    case_dir, out = case_env(("lc_pull", "lc_push"))
    cfg = _cfg(case_dir, out, [
        LoadCase(name="pull", stem="lc_pull", weight=1.0),
        LoadCase(name="push", stem="lc_push", weight=1.0)], max_iter=3)
    solver_calls = []

    def fake_extract(cfg, run_dir, keep_vtk=False, stem=None, disp_node_id=None,
                     disp_node_ids=None, exclude_element_ids=None):
        dead = stem == "lc_push"
        return _results(0.0 if dead else 100.0, 0.0 if dead else 0.1,
                        energy=[0.0, 0.0] if dead else [2.0, 1.0])
    _stub_solver(monkeypatch, fake_extract, solver_calls)

    status = loop_mod.run_optimization(cfg, log=lambda *_: None)

    assert status.state == "failed"
    assert "null solve" in status.message and "'push'" in status.message


def test_healthy_solve_not_flagged(case_env, monkeypatch):
    """A normal solve (positive stress + energy) is never flagged null: the run
    proceeds and does not fail with a null-solve message (no false positive)."""
    case_dir, out = case_env(("lc_a",))
    cfg = _cfg(case_dir, out, [LoadCase(name="a", stem="lc_a")], max_iter=1)
    solver_calls = []

    def fake_extract(cfg, run_dir, keep_vtk=False, stem=None, disp_node_id=None,
                     disp_node_ids=None, exclude_element_ids=None):
        return _results(100.0, 0.1, energy=[2.0, 1.0])
    _stub_solver(monkeypatch, fake_extract, solver_calls)

    status = loop_mod.run_optimization(cfg, log=lambda *_: None)

    assert status.state != "failed"
    assert "null solve" not in (status.message or "")
    assert len(solver_calls) == 1
