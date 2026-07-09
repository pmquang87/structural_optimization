"""The converged exit keeps the checkpoint's ``it + 1`` invariant.

Iteration ``it`` fully completed (solved, published, archived, history row
written) before the convergence check fires, but the converged ``break`` used
to skip the end-of-iteration ``save_checkpoint`` -- so an extend-max_iter
resume re-ran the identical, already-solved converged design (a full solve
wasted) and appended a duplicate history row. Hermetic: solver/extract stubbed.
"""
from __future__ import annotations

import numpy as np

from oropt import loop as loop_mod
from oropt import status as st
from oropt.config import Config, DispConstraint, LoadCase
from oropt.results import Results
from oropt.runner import RunResult

ELEM_IDS = np.array([60000001, 60000002], dtype=np.int64)


def test_converged_run_checkpoints_past_the_converged_iteration(
        tmp_path, mini_deck_path, mini_engine_path, monkeypatch):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "lc_a_0000.rad").write_text(
        mini_deck_path.read_text(encoding="utf-8"), encoding="utf-8")
    (case_dir / "lc_a_0001.rad").write_text(
        mini_engine_path.read_text(encoding="utf-8"), encoding="utf-8")

    cfg = Config()
    cfg.model.case_dir = str(case_dir)
    cfg.model.design_part_id = 60000000
    cfg.model.design_node_min = 60000000
    cfg.model.bc_group_id = 60000000
    cfg.load_cases = [LoadCase(
        name="a", stem="lc_a", sigma_allow=250.0,
        disp_constraints=[DispConstraint(node_id=60000001, d_allow=1.0)])]
    cfg.work_dir = str(tmp_path / "out")
    # Converge trivially: target vf 1.0 (no removal asked), a 2-iteration
    # window, and feasible stub results -> converged at iteration 1.
    cfg.beso.max_iter = 10
    cfg.beso.filter_radius = 0.0
    cfg.beso.target_volume_fraction = 1.0
    cfg.beso.convergence_window = 2

    monkeypatch.setattr(loop_mod, "run_solver",
                        lambda cfg, run_dir, stem=None:
                        RunResult(True, "ok", "NORMAL TERMINATION"))
    monkeypatch.setattr(
        loop_mod, "extract",
        lambda *a, **k: Results(element_ids=ELEM_IDS.copy(),
                                energy=np.array([2.0, 1.0]),
                                vonmises=np.array([100.0, 90.0]),
                                sigma_max=100.0, disp=0.1,
                                disp_node_id=60000001, disps={60000001: 0.1}))

    status = loop_mod.run_optimization(cfg, log=lambda *_: None)

    assert status.state == "converged"
    it = status.iteration
    rows = st.read_history(cfg.work())
    assert len(rows) == it + 1                    # one row per completed iteration
    # the invariant: a resume starts AFTER the converged iteration
    assert st.checkpoint_iteration(cfg.work()) == it + 1
