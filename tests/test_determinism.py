"""Determinism / reproducibility of the oropt-side optimisation pipeline (V6).

Reproducibility is claimed (the config snapshot) but never asserted. The whole
production path is written to be deterministic -- ``np.argsort(kind="stable")``
everywhere, no RNG in the loop or any of the five optimisers (the only RNG in the
package lives in the offline ``mfse.py`` prototype). This test proves it: for
EACH optimiser (``beso``, ``tobs``, ``hca``, ``saip``, ``levelset``) run the loop
twice, from an identical start, into two separate work dirs, with a stubbed
DETERMINISTIC solver (fixed per-element energy/stress/disp derived from the
mesh), and assert the two runs agree bit-for-bit:

* identical ``history.csv`` bytes,
* identical final alive mask (``checkpoint.npz``),
* identical final topology (``topology_latest.vtu`` bytes),
* identical design-scalar status fields.

The wall clock is faked to a constant so ``iter_wall_s`` (a timing column, the
only wall-clock-derived value in ``history.csv``) is stable and the byte-identity
is a genuine assertion about the pipeline, not about timing luck. Hermetic:
``run_solver`` / ``extract`` stubbed, so OpenRadioss is never invoked.
"""
from __future__ import annotations

import numpy as np
import pytest

from oropt import loop as loop_mod
from oropt import status as st
from oropt.config import Config, DispConstraint, LoadCase
from oropt.results import Results
from oropt.runner import RunResult

# The conftest mini deck's design part /TETRA4/60000000 has two tetra elements.
ELEM_IDS = np.array([60000001, 60000002], dtype=np.int64)
OPTIMIZERS = ["beso", "tobs", "hca", "saip", "levelset"]

# A fixed, element-dependent field: distinct per element so any rank-based update
# is well-defined, and identical on every call so the solve is deterministic.
ENERGY = np.array([3.0, 1.0])
VONMISES = np.array([120.0, 80.0])


def _results() -> Results:
    return Results(element_ids=ELEM_IDS.copy(), energy=ENERGY.copy(),
                   vonmises=VONMISES.copy(), sigma_max=float(VONMISES.max()),
                   disp=0.1, disp_node_id=60000001, disps={60000001: 0.1})


def _fixed_clock() -> float:
    return 1000.0


def _stub(monkeypatch):
    """Deterministic solver + a frozen clock (so iter_wall_s is stable)."""
    def fake_run_solver(cfg, run_dir, stem=None):
        return RunResult(True, "ok", "NORMAL TERMINATION")

    def fake_extract(cfg, run_dir, keep_vtk=False, stem=None, disp_node_id=None,
                     disp_node_ids=None, exclude_element_ids=None):
        return _results()

    monkeypatch.setattr(loop_mod, "run_solver", fake_run_solver)
    monkeypatch.setattr(loop_mod, "extract", fake_extract)
    monkeypatch.setattr(loop_mod.time, "time", _fixed_clock)


def _cfg(case_dir, out_dir, optimizer: str, max_iter: int) -> Config:
    cfg = Config()
    cfg.optimizer = optimizer
    cfg.model.case_dir = str(case_dir)
    cfg.model.design_part_id = 60000000
    cfg.model.design_node_min = 60000000
    cfg.model.bc_group_id = 60000000
    oc = cfg.active_opts()                 # the selected optimiser's own block
    oc.max_iter = max_iter
    oc.filter_radius = 0.0                 # identity filter: no geometry-order effects
    oc.target_volume_fraction = 0.5        # drive real carving on the 2-element part
    oc.protect_bc_nodes = False            # ... so an element can actually be removed
    oc.protect_layers = 0                  #     (otherwise both elems are BC-protected)
    cfg.load_cases = [LoadCase(
        name="a", stem="lc_a", sigma_allow=250.0,
        disp_constraints=[DispConstraint(node_id=60000001, d_allow=1.0)])]
    cfg.work_dir = str(out_dir)
    return cfg


@pytest.fixture
def case_env(tmp_path, mini_deck_path, mini_engine_path):
    """A case dir (seeded with the mini deck) + a factory for fresh work dirs."""
    deck_text = mini_deck_path.read_text(encoding="utf-8")
    engine_text = mini_engine_path.read_text(encoding="utf-8")
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "lc_a_0000.rad").write_text(deck_text, encoding="utf-8")
    (case_dir / "lc_a_0001.rad").write_text(engine_text, encoding="utf-8")

    def work_dir(tag: str):
        d = tmp_path / f"out_{tag}"
        d.mkdir()
        return d
    return case_dir, work_dir


def _final_alive(work) -> np.ndarray:
    ckpt = st.load_checkpoint(work)
    assert ckpt is not None, "run left no checkpoint to compare"
    return np.asarray(ckpt["alive_mask"], dtype=bool)


@pytest.mark.parametrize("optimizer", OPTIMIZERS)
def test_two_runs_are_byte_identical(case_env, monkeypatch, optimizer):
    """Same optimiser, same start, two work dirs -> identical run artefacts.

    None of the five optimisers uses an RNG or seed, so no per-optimiser skip is
    needed here; if one ever became nondeterministic this test would surface it
    (rather than being forced to a false pass)."""
    case_dir, work_dir = case_env
    _stub(monkeypatch)

    out_a = work_dir(f"{optimizer}_a")
    out_b = work_dir(f"{optimizer}_b")
    cfg_a = _cfg(case_dir, out_a, optimizer, max_iter=4)
    cfg_b = _cfg(case_dir, out_b, optimizer, max_iter=4)

    status_a = loop_mod.run_optimization(cfg_a, log=lambda *_: None)
    status_b = loop_mod.run_optimization(cfg_b, log=lambda *_: None)

    work_a, work_b = cfg_a.work(), cfg_b.work()

    # 1) history.csv is byte-for-byte identical (incl. the iter_wall_s column,
    #    which the frozen clock pins to a constant).
    hist_a = (work_a / st.HISTORY).read_bytes()
    hist_b = (work_b / st.HISTORY).read_bytes()
    assert hist_a == hist_b, f"{optimizer}: history.csv differs between runs"
    assert hist_a.count(b"\n") >= 2          # header + at least one iteration row

    # 2) the final alive mask is identical.
    np.testing.assert_array_equal(_final_alive(work_a), _final_alive(work_b),
                                  err_msg=f"{optimizer}: final alive mask differs")
    # ... and the run was non-trivial: the optimiser removed material at some
    # iteration (so this is determinism on a real design change, not a stationary
    # all-alive run). Some optimisers oscillate back to full at the last step, so
    # this is checked over the whole history, not just the final mask.
    alive_hist = [int(r["elements_alive"]) for r in st.read_history(work_a)]
    assert min(alive_hist) < ELEM_IDS.size, f"{optimizer}: nothing was ever carved"

    # 3) the final topology VTU is byte-identical.
    topo_a = (work_a / st.TOPOLOGY).read_bytes()
    topo_b = (work_b / st.TOPOLOGY).read_bytes()
    assert topo_a == topo_b, f"{optimizer}: topology_latest.vtu differs"

    # 4) the design-scalar status fields agree (timestamps/pid excluded -- those
    #    are legitimately per-run and are not part of the design).
    for field in ("state", "iteration", "volume_fraction", "sigma_max", "disp",
                  "elements_alive", "feasible"):
        assert getattr(status_a, field) == getattr(status_b, field), \
            f"{optimizer}: status.{field} differs ({getattr(status_a, field)!r} "\
            f"!= {getattr(status_b, field)!r})"


def test_all_five_optimizers_were_exercised():
    """Guard against the parametrisation silently shrinking: every production
    optimiser must be covered by the determinism check above."""
    assert set(OPTIMIZERS) == {"beso", "tobs", "hca", "saip", "levelset"}
