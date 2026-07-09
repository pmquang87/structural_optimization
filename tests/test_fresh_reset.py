"""Fresh (non-resume) runs clear a previous run's accumulating outputs.

Re-running fresh into a used folder must not prepend the old run's history rows or
evolution snapshots (a killed dead run showing up before the real one). The
``iter_NNNN/`` archives (reuse seed) and ``checkpoint.npz`` are preserved. Mirrors
``oropt.run._tee_log`` truncating ``run.log`` on a fresh start.
"""
from __future__ import annotations

import numpy as np

from oropt import loop as loop_mod
from oropt import status as st
from oropt.config import Config, DispConstraint, LoadCase
from oropt.results import Results
from oropt.runner import RunResult

ELEM_IDS = np.array([60000001, 60000002], dtype=np.int64)


# ---- unit: _reset_stale_outputs ---------------------------------------------
def test_reset_clears_history_and_snapshots_keeps_archives(tmp_path):
    (tmp_path / st.HISTORY).write_text("iteration,vf\n0,1.0\n", encoding="utf-8")
    (tmp_path / "topology_iter0000.vtu").write_text("x", encoding="utf-8")
    (tmp_path / "topology_iter0007.vtu").write_text("x", encoding="utf-8")
    (tmp_path / "topology_smoothed_iter0003.stl").write_text("x", encoding="utf-8")
    (tmp_path / "checkpoint.npz").write_text("keep", encoding="utf-8")
    (tmp_path / "topology_latest.vtu").write_text("keep", encoding="utf-8")
    seed = tmp_path / "iter_0000"
    seed.mkdir()
    (seed / "deck_0000.rad").write_text("seed", encoding="utf-8")

    loop_mod._reset_stale_outputs(tmp_path, log=lambda *_: None)

    # cleared: accumulating history + per-iteration snapshots
    assert not (tmp_path / st.HISTORY).exists()
    assert not (tmp_path / "topology_iter0000.vtu").exists()
    assert not (tmp_path / "topology_iter0007.vtu").exists()
    assert not (tmp_path / "topology_smoothed_iter0003.stl").exists()
    # preserved: checkpoint, wholesale-overwritten end product, reuse-seed archive
    assert (tmp_path / "checkpoint.npz").exists()
    assert (tmp_path / "topology_latest.vtu").exists()
    assert (seed / "deck_0000.rad").exists()


def test_reset_noop_on_clean_folder(tmp_path):
    loop_mod._reset_stale_outputs(tmp_path, log=lambda *_: None)   # nothing to clear
    assert list(tmp_path.iterdir()) == []


# ---- integration: a fresh loop run truncates a prior run's history ----------
def _cfg(case_dir, out_dir, load_cases, max_iter=1) -> Config:
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


def _results(sigma, disp, energy) -> Results:
    return Results(element_ids=ELEM_IDS.copy(), energy=np.asarray(energy, float),
                   vonmises=np.full(ELEM_IDS.size, sigma, float),
                   sigma_max=sigma, disp=disp, disp_node_id=60000001,
                   disps={60000001: disp})


def test_fresh_run_truncates_prior_history(tmp_path, mini_deck_path,
                                           mini_engine_path, monkeypatch):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "lc_a_0000.rad").write_text(
        mini_deck_path.read_text(encoding="utf-8"), encoding="utf-8")
    (case_dir / "lc_a_0001.rad").write_text(
        mini_engine_path.read_text(encoding="utf-8"), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    # a previous (dead) run left 3 history rows behind
    (out / st.HISTORY).write_text(
        "iteration,volume_fraction,sigma_max,disp,elements_alive,feasible,"
        "iter_wall_s,or_termination,optimizer\n"
        "0,1.0,0.0,0.0,2,True,1.0,NORMAL,beso\n"
        "1,0.9,0.0,0.0,2,True,1.0,NORMAL,beso\n"
        "2,0.8,0.0,0.0,2,True,1.0,NORMAL,beso\n", encoding="utf-8")

    monkeypatch.setattr(loop_mod, "run_solver",
                        lambda cfg, run_dir, stem=None:
                        RunResult(True, "ok", "NORMAL TERMINATION"))
    monkeypatch.setattr(loop_mod, "extract",
                        lambda *a, **k: _results(100.0, 0.1, [2.0, 1.0]))

    cfg = _cfg(case_dir, out, [LoadCase(name="a", stem="lc_a")], max_iter=1)
    loop_mod.run_optimization(cfg, resume=False, log=lambda *_: None)

    rows = st.read_history(cfg.work())
    # only the new run's single (real) iteration remains -- the 3 dead rows are gone
    assert len(rows) == 1
    assert rows[0]["iteration"] == "0"
    assert float(rows[0]["sigma_max"]) == 100.0


def test_resume_without_checkpoint_falls_back_to_fresh(tmp_path, mini_deck_path,
                                                       mini_engine_path,
                                                       monkeypatch):
    """--resume with no checkpoint.npz (e.g. the prior run crashed during
    iteration 0, before the first save) used to start from scratch silently
    while KEEPING the old history/snapshots -- duplicate iteration numbers, and
    the post-run last-feasible selection could pick the old run's design. It
    must downgrade to a fresh run, stale outputs cleared."""
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    (case_dir / "lc_a_0000.rad").write_text(
        mini_deck_path.read_text(encoding="utf-8"), encoding="utf-8")
    (case_dir / "lc_a_0001.rad").write_text(
        mini_engine_path.read_text(encoding="utf-8"), encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    (out / st.HISTORY).write_text(
        "iteration,volume_fraction,sigma_max,disp,elements_alive,feasible,"
        "iter_wall_s,or_termination,optimizer\n"
        "0,1.0,50.0,0.0,2,True,1.0,NORMAL,beso\n", encoding="utf-8")
    assert not (out / st.CHECKPOINT).exists()

    monkeypatch.setattr(loop_mod, "run_solver",
                        lambda cfg, run_dir, stem=None:
                        RunResult(True, "ok", "NORMAL TERMINATION"))
    monkeypatch.setattr(loop_mod, "extract",
                        lambda *a, **k: _results(100.0, 0.1, [2.0, 1.0]))

    cfg = _cfg(case_dir, out, [LoadCase(name="a", stem="lc_a")], max_iter=1)
    logs: list[str] = []
    loop_mod.run_optimization(cfg, resume=True, log=logs.append)

    rows = st.read_history(cfg.work())
    assert len(rows) == 1                        # old row cleared, not appended-to
    assert float(rows[0]["sigma_max"]) == 100.0  # ... and it is the NEW run's row
    assert any("starting FRESH" in m for m in logs)
