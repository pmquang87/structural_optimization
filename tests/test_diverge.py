"""Resilience to a non-converging implicit engine solve.

A severed load path (e.g. an over-carved design) makes the implicit engine
diverge forever without erroring out: "--ITERATION DIVERGE--" / timestep-cut
cycles until the (huge) hard timeout. Three layers are covered here:

* :class:`oropt.runner.DivergenceMonitor` — the streaming listing detector,
  against text modelled on a real elevator-linkage listing (where healthy
  solves also print *isolated* DIVERGE lines that recover);
* :func:`oropt.runner._run_engine` — the watchdog kills a diverging/overdue
  child process (a stand-in python script), while the hard timeout still raises;
* the loop — a diverged solve is treated as INFEASIBLE (back off from the
  previous alive mask, never parse the partial results), and N consecutive
  diverged iterations fail the run. Hermetic: ``run_solver``/``extract`` stubbed.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from oropt import loop as loop_mod
from oropt import runner as runner_mod
from oropt import status as st
from oropt.config import Config, DispConstraint, LoadCase
from oropt.results import Results
from oropt.runner import DivergenceMonitor, RunResult, _run_engine

ELEM_IDS = np.array([60000001, 60000002], dtype=np.int64)

# ---- real listing fragments (format observed on elevator-linkage runs) ------
DIVERGE_CYCLE = (
    "   --ITERATION DIVERGE with MAX_ITER REACHED--\n"
    " \n"
    "   --RESET ITERATION WITH NEW TIMESTEP--\n"
    " \n"
    "     --NEXT TIMESTEP IS DECREASED BY-- 0.6667E+00\n")
CONV_ROW = "         2              4.929E-01  4.653E-02  1.835E-02     C\n"
NONCONV_ROW = "        16              1.067E+00  5.917E-02  1.561E-02\n"
DT_INCREASE = "     --NEXT TIMESTEP IS INCREASED BY-- 0.1100E+01\n"
CYCLE_LINE = ("      16  0.3277      0.2785E-01 SOLID   60163238   0.0%"
              "   137.1       0.000       0.000       137.1       0.000"
              "      0.2900E-03   0.000    \n")


# ---- DivergenceMonitor -------------------------------------------------------
def test_monitor_ignores_recovered_diverges():
    """Isolated diverges that recover (converged row + dt increase after each,
    the healthy pattern) never trip, however many accumulate over the run."""
    mon = DivergenceMonitor(max_cycles=12)
    healthy = DIVERGE_CYCLE + CYCLE_LINE + CONV_ROW + DT_INCREASE
    for _ in range(50):
        assert mon.feed(healthy) is None
    assert mon.streak == 0


def test_monitor_trips_on_unbroken_diverge_streak():
    mon = DivergenceMonitor(max_cycles=12)
    # retries print cycle lines and non-converged iteration rows between
    # diverges -- neither counts as an accepted step
    stuck = DIVERGE_CYCLE + CYCLE_LINE + NONCONV_ROW
    reason = None
    for _ in range(12):
        reason = mon.feed(stuck)
    assert reason is not None
    assert "12 consecutive" in reason and "diverge_max_cycles=12" in reason


def test_monitor_converged_row_resets_streak_without_dt_increase():
    """An accepted step (Conv.stat 'C' row) resets the streak even when the
    timestep is not increased afterwards (dt already at its cap)."""
    mon = DivergenceMonitor(max_cycles=12)
    assert mon.feed(DIVERGE_CYCLE * 11) is None
    assert mon.feed(CYCLE_LINE + CONV_ROW) is None       # streak reset to 0
    assert mon.feed(DIVERGE_CYCLE * 11) is None          # 11 < 12: no trip
    assert mon.feed(DIVERGE_CYCLE) is not None           # 12th consecutive


def test_monitor_recovery_within_same_chunk_does_not_trip():
    """Only the streak still unbroken at the end of a chunk counts: a stall
    that already recovered by the time the file was polled is history."""
    mon = DivergenceMonitor(max_cycles=12)
    assert mon.feed(DIVERGE_CYCLE * 20 + CONV_ROW + DT_INCREASE) is None


def test_monitor_is_chunking_invariant():
    """Feeding byte-sized chunks (mid-line splits) trips at the same point as
    one big feed."""
    text = DIVERGE_CYCLE * 12
    mon = DivergenceMonitor(max_cycles=12)
    reasons = [mon.feed(text[i:i + 7]) for i in range(0, len(text), 7)]
    tripped = [r for r in reasons if r]
    assert tripped and "12 consecutive" in tripped[0]


def test_monitor_disabled_with_zero_max_cycles():
    mon = DivergenceMonitor(max_cycles=0)
    assert mon.feed(DIVERGE_CYCLE * 500) is None


# ---- _run_engine watchdog (real child processes, fast poll) ------------------
# The child stands in for the engine: writes the given listing content, then
# sleeps "forever" (diverging engines never exit) or exits cleanly.
_CHILD = (
    "import sys, time\n"
    "listing, mode = sys.argv[1], sys.argv[2]\n"
    "with open(listing, 'w') as f:\n"
    "    if mode == 'diverge':\n"
    "        f.write('   --ITERATION DIVERGE with MAX_ITER REACHED--\\n"
    "     --NEXT TIMESTEP IS DECREASED BY-- 0.6667E+00\\n' * 30)\n"
    "    if mode == 'ok':\n"
    "        f.write(' NORMAL TERMINATION\\n')\n"
    "    f.flush()\n"
    "    if mode != 'ok':\n"
    "        time.sleep(120)\n")


def _engine_cfg(**run_overrides) -> Config:
    cfg = Config()
    cfg.run.engine_timeout_s = 60.0
    cfg.run.engine_soft_timeout_s = 0.0
    cfg.run.diverge_max_cycles = 12
    for k, v in run_overrides.items():
        setattr(cfg.run, k, v)
    return cfg


def _child_cmd(listing: Path, mode: str) -> list[str]:
    return [sys.executable, "-c", _CHILD, str(listing), mode]


@pytest.fixture
def fast_poll(monkeypatch):
    monkeypatch.setattr(runner_mod, "_POLL_S", 0.1)


def test_run_engine_kills_diverging_solve(tmp_path, fast_poll):
    listing = tmp_path / "demo_0001.out"
    t0 = time.monotonic()
    rc, reason = _run_engine(_engine_cfg(), _child_cmd(listing, "diverge"),
                             tmp_path, dict(os.environ),
                             tmp_path / "engine.log", listing)
    assert rc is None
    assert "consecutive ITERATION DIVERGE" in reason
    assert time.monotonic() - t0 < 30.0      # killed in seconds, not at timeout


def test_run_engine_soft_timeout_flags_nonconvergence(tmp_path, fast_poll):
    listing = tmp_path / "demo_0001.out"
    rc, reason = _run_engine(
        _engine_cfg(engine_soft_timeout_s=0.5, diverge_max_cycles=0),
        _child_cmd(listing, "sleep"), tmp_path, dict(os.environ),
        tmp_path / "engine.log", listing)
    assert rc is None
    assert "engine_soft_timeout_s" in reason


def test_run_engine_hard_timeout_still_raises(tmp_path, fast_poll):
    listing = tmp_path / "demo_0001.out"
    with pytest.raises(subprocess.TimeoutExpired):
        _run_engine(_engine_cfg(engine_timeout_s=0.5, diverge_max_cycles=0),
                    _child_cmd(listing, "sleep"), tmp_path, dict(os.environ),
                    tmp_path / "engine.log", listing)


def test_run_engine_normal_exit_and_stale_listing_ignored(tmp_path, fast_poll):
    """A clean exit returns (rc, None) -- and a stale listing full of diverge
    cycles from a previous solve in the same dir is dropped up front, so it can
    never trip the monitor before the engine recreates the file."""
    listing = tmp_path / "demo_0001.out"
    listing.write_text(DIVERGE_CYCLE * 30, encoding="utf-8")
    rc, reason = _run_engine(_engine_cfg(), _child_cmd(listing, "ok"),
                             tmp_path, dict(os.environ),
                             tmp_path / "engine.log", listing)
    assert reason is None
    assert rc == 0
    assert "NORMAL TERMINATION" in listing.read_text(encoding="utf-8")


# ---- loop behaviour (hermetic, run_solver/extract stubbed) --------------------
DIVERGED = RunResult(
    False, "engine",
    "engine did not converge: 12 consecutive ITERATION DIVERGE / "
    "timestep-decrease cycles without an accepted step "
    "(run.diverge_max_cycles=12)", diverged=True)


def _cfg(case_dir: Path, out_dir: Path, load_cases, max_iter=1) -> Config:
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


def _results(sigma: float, disp: float, energy) -> Results:
    return Results(element_ids=ELEM_IDS.copy(),
                   energy=np.asarray(energy, dtype=float),
                   vonmises=np.full(ELEM_IDS.size, sigma, dtype=float),
                   sigma_max=sigma, disp=disp, disp_node_id=60000001,
                   disps={60000001: disp})


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


def _stub_diverging_solver(monkeypatch, diverge_when, solver_calls,
                           extract_calls):
    """Stub run_solver/extract; ``diverge_when(stem, nth_call_for_stem)``
    decides whether a solve diverges. Healthy solves are feasible (within the
    250 / 1.0 limits)."""
    seen: dict = {}

    def fake_run_solver(cfg, run_dir, stem=None):
        n = seen[stem] = seen.get(stem, -1) + 1
        solver_calls.append((stem, n))
        return DIVERGED if diverge_when(stem, n) else \
            RunResult(True, "ok", "NORMAL TERMINATION")

    def fake_extract(cfg, run_dir, keep_vtk=False, stem=None, disp_node_id=None,
                     disp_node_ids=None, exclude_element_ids=None):
        extract_calls.append(stem)
        return _results(100.0, 0.1, energy=[2.0, 1.0])

    monkeypatch.setattr(loop_mod, "run_solver", fake_run_solver)
    monkeypatch.setattr(loop_mod, "extract", fake_extract)


def _spy_gate(monkeypatch, gate_calls):
    """Record every (feasible, violation) the loop hands to next_target_vf."""
    real_build = loop_mod.build_optimizer

    def spy_build(*args, **kwargs):
        opt = real_build(*args, **kwargs)
        real_next = opt.next_target_vf

        def spy_next(current_vf, feasible, violation=None):
            gate_calls.append((feasible, violation))
            return real_next(current_vf, feasible, violation)

        opt.next_target_vf = spy_next
        return opt

    monkeypatch.setattr(loop_mod, "build_optimizer", spy_build)


def test_diverged_iteration_backs_off_and_run_continues(case_env, monkeypatch):
    """One diverged iteration in the middle of a run: the loop marks it
    infeasible (worst violation -> the gate backs off), never parses its
    results, and continues to the next iteration instead of failing."""
    case_dir, out = case_env(("lc_a",))
    cfg = _cfg(case_dir, out, [LoadCase(name="a", stem="lc_a")], max_iter=3)
    solver_calls, extract_calls, gate_calls = [], [], []
    _stub_diverging_solver(monkeypatch, lambda stem, n: n == 1,
                           solver_calls, extract_calls)
    _spy_gate(monkeypatch, gate_calls)

    status = loop_mod.run_optimization(cfg, log=lambda *_: None)

    assert status.state != "failed"
    assert len(solver_calls) == 3                  # every iteration still solved
    assert extract_calls == ["lc_a", "lc_a"]       # diverged results never parsed
    # gate calls: iter0 feasible, iter1 diverged (worst violation), iter2 feasible
    assert gate_calls[0][0] is True
    assert gate_calls[1] == (False, float("inf"))
    assert gate_calls[2][0] is True
    # the diverged iteration is on record as infeasible with the OR message
    rows = st.read_history(cfg.work())
    assert len(rows) == 3
    assert rows[1]["feasible"] == "False"
    assert "did not converge" in rows[1]["or_termination"]
    assert rows[0]["feasible"] == "True" and rows[2]["feasible"] == "True"


def test_diverge_on_first_iteration_keeps_mask_and_continues(case_env, monkeypatch):
    """Iteration 0 diverging leaves no previous sensitivity to re-grow with:
    the alive mask stays put (no gate/update call) and the run carries on."""
    case_dir, out = case_env(("lc_a",))
    cfg = _cfg(case_dir, out, [LoadCase(name="a", stem="lc_a")], max_iter=2)
    solver_calls, extract_calls, gate_calls = [], [], []
    _stub_diverging_solver(monkeypatch, lambda stem, n: n == 0,
                           solver_calls, extract_calls)
    _spy_gate(monkeypatch, gate_calls)

    status = loop_mod.run_optimization(cfg, log=lambda *_: None)

    assert status.state != "failed"
    assert len(solver_calls) == 2
    assert len(gate_calls) == 1 and gate_calls[0][0] is True   # only iter1 gated
    rows = st.read_history(cfg.work())
    # full mini deck (2 elements) still alive through the diverged iteration
    assert rows[0]["elements_alive"] == "2"


def test_run_fails_after_n_consecutive_diverged_iterations(case_env, monkeypatch):
    case_dir, out = case_env(("lc_a",))
    cfg = _cfg(case_dir, out, [LoadCase(name="a", stem="lc_a")], max_iter=10)
    cfg.run.diverge_fail_after = 3
    solver_calls, extract_calls = [], []
    _stub_diverging_solver(monkeypatch, lambda stem, n: True,
                           solver_calls, extract_calls)

    status = loop_mod.run_optimization(cfg, log=lambda *_: None)

    assert status.state == "failed"
    assert "3 consecutive" in status.message
    assert "did not converge" in status.message
    assert len(solver_calls) == 3          # stopped at N, not at max_iter
    assert extract_calls == []


def test_successful_iteration_resets_consecutive_counter(case_env, monkeypatch):
    """Alternating diverge/ok never accumulates to diverge_fail_after=2: the
    counter counts *consecutive* diverged iterations only."""
    case_dir, out = case_env(("lc_a",))
    cfg = _cfg(case_dir, out, [LoadCase(name="a", stem="lc_a")], max_iter=6)
    cfg.run.diverge_fail_after = 2
    solver_calls, extract_calls = [], []
    _stub_diverging_solver(monkeypatch, lambda stem, n: n % 2 == 0,
                           solver_calls, extract_calls)

    status = loop_mod.run_optimization(cfg, log=lambda *_: None)

    assert status.state != "failed"
    assert len(solver_calls) == 6


def test_diverged_case_named_in_status_message(case_env, monkeypatch):
    """Multi-case: the status message names the case that did not converge,
    e.g. "case 'push': engine did not converge -- treated as infeasible"."""
    case_dir, out = case_env(("lc_pull", "lc_push"))
    cfg = _cfg(case_dir, out, [
        LoadCase(name="pull", stem="lc_pull", weight=1.0),
        LoadCase(name="push", stem="lc_push", weight=1.0)], max_iter=1)
    solver_calls, extract_calls = [], []
    _stub_diverging_solver(monkeypatch, lambda stem, n: stem == "lc_push",
                           solver_calls, extract_calls)

    status = loop_mod.run_optimization(cfg, log=lambda *_: None)

    assert status.state != "failed"        # 1 diverged < diverge_fail_after 3
    assert status.message == \
        "case 'push': engine did not converge -- treated as infeasible"
    assert status.feasible is False
    assert "did not converge" in status.or_termination
    # the healthy case was solved, the diverged one attempted, nothing parsed
    # beyond the healthy case
    assert [s for s, _ in solver_calls] == ["lc_pull", "lc_push"]
    assert extract_calls == ["lc_pull"]
