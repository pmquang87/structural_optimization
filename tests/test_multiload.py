"""Multiple load cases: weighted-sum sensitivity aggregation, worst-case
feasibility, single-case reproduction of the classic path, and config roundtrip.

Hermetic: ``run_solver`` and ``extract`` are stubbed to return synthetic per-case
``Results`` so OpenRadioss is never invoked. The real deck rewrite, mesh build,
BESO update and status/topology writes all run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from oropt import loop as loop_mod
from oropt.beso import combine_sensitivity
from oropt.config import Config, LoadCase
from oropt.results import Results
from oropt.runner import RunResult

# The mini deck (conftest.MINI_DECK) has design part /TETRA4/60000000 with two
# tetra elements, ids 60000001 and 60000002.
ELEM_IDS = np.array([60000001, 60000002], dtype=np.int64)


# ---- helpers ---------------------------------------------------------------
def _make_case_dir(case_dir: Path, deck_text: str, engine_text: str, stems) -> None:
    """Write a starter+engine deck pair for every stem (all share one mesh)."""
    case_dir.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (case_dir / f"{stem}_0000.rad").write_text(deck_text, encoding="utf-8")
        (case_dir / f"{stem}_0001.rad").write_text(engine_text, encoding="utf-8")


def _cfg(case_dir: Path, out_dir: Path, stem: str, load_cases, max_iter=1) -> Config:
    cfg = Config()
    cfg.model.case_dir = str(case_dir)
    cfg.model.stem = stem
    cfg.model.design_part_id = 60000000
    cfg.model.design_node_min = 60000000
    cfg.model.bc_group_id = 60000000
    cfg.model.disp_node_id = 60000001
    cfg.constraints.sigma_allow = 250.0
    cfg.constraints.d_allow = 1.0
    cfg.beso.max_iter = max_iter
    cfg.beso.filter_radius = 0.0      # identity filter: keep the test reasoning simple
    cfg.load_cases = list(load_cases)
    cfg.work_dir = str(out_dir)
    return cfg


def _results(sigma: float, disp: float, energy) -> Results:
    energy = np.asarray(energy, dtype=float)
    return Results(element_ids=ELEM_IDS.copy(), energy=energy,
                   vonmises=np.full(ELEM_IDS.size, sigma, dtype=float),
                   sigma_max=sigma, disp=disp, disp_node_id=60000001)


def _stub_solver(monkeypatch, results_by_stem, calls, fail_stems=()):
    def fake_run_solver(cfg, run_dir):
        calls.append((cfg.model.stem, Path(run_dir)))
        if cfg.model.stem in fail_stems:
            return RunResult(False, "engine", "ERROR TERMINATION")
        return RunResult(True, "ok", "NORMAL TERMINATION")

    def fake_extract(cfg, run_dir, keep_vtk=False):
        return results_by_stem[cfg.model.stem]

    monkeypatch.setattr(loop_mod, "run_solver", fake_run_solver)
    monkeypatch.setattr(loop_mod, "extract", fake_extract)


@pytest.fixture
def case_env(tmp_path, mini_deck_path, mini_engine_path):
    """A case dir factory + an output dir, seeded from the conftest mini deck."""
    deck_text = mini_deck_path.read_text(encoding="utf-8")
    engine_text = mini_engine_path.read_text(encoding="utf-8")
    case_dir = tmp_path / "case"
    out_dir = tmp_path / "out"

    def make(stems):
        _make_case_dir(case_dir, deck_text, engine_text, stems)
        return case_dir, out_dir
    return make


# ---- (a) weighted-sum sensitivity aggregation ------------------------------
def test_combine_sensitivity_weighted_sum():
    raws = [np.array([0.0, 2.0, 4.0]),    # max 4 -> [0, .5, 1]
            np.array([3.0, 0.0, 1.0])]    # max 3 -> [1, 0, 1/3]
    out = combine_sensitivity(raws, [1.0, 2.0])
    assert np.allclose(out, [0.0 + 2.0, 0.5 + 0.0, 1.0 + 2.0 / 3.0])


def test_combine_sensitivity_single_case_is_identity():
    raw = np.array([1.0, 5.0, 0.0])
    out = combine_sensitivity([raw], [3.0])   # weight ignored for one case
    assert out is raw                         # byte-identical: same array object


def test_combine_sensitivity_all_zero_case_no_div_by_zero():
    raws = [np.zeros(3), np.array([0.0, 2.0, 0.0])]
    out = combine_sensitivity(raws, [1.0, 1.0])
    assert np.allclose(out, [0.0, 1.0, 0.0])   # zero case contributes 0, no NaN


def test_loop_feeds_per_case_raws_and_weights_to_combine(case_env, monkeypatch):
    case_dir, out = case_env(("lc_a", "lc_b"))
    cfg = _cfg(case_dir, out, "lc_a", [
        LoadCase(name="a", stem="lc_a", weight=1.0),
        LoadCase(name="b", stem="lc_b", weight=3.0),
    ])
    _stub_solver(monkeypatch, {
        "lc_a": _results(100.0, 0.1, energy=[2.0, 0.0]),
        "lc_b": _results(120.0, 0.2, energy=[0.0, 5.0]),
    }, calls=[])

    captured = {}
    real_combine = loop_mod.combine_sensitivity

    def spy(raws, weights):
        captured["raws"] = [np.asarray(r).copy() for r in raws]
        captured["weights"] = list(weights)
        return real_combine(raws, weights)

    monkeypatch.setattr(loop_mod, "combine_sensitivity", spy)
    loop_mod.run_optimization(cfg, log=lambda *_: None)

    assert captured["weights"] == [1.0, 3.0]
    # per-case raw sensitivity = energy mapped onto the deck element order
    assert np.allclose(captured["raws"][0], [2.0, 0.0])
    assert np.allclose(captured["raws"][1], [0.0, 5.0])


# ---- (b) worst-case feasibility (infeasible if ANY case violates) ----------
def test_feasible_when_every_case_within_limits(case_env, monkeypatch):
    case_dir, out = case_env(("lc_a", "lc_b"))
    cfg = _cfg(case_dir, out, "lc_a", [
        LoadCase(name="a", stem="lc_a", weight=1.0),
        LoadCase(name="b", stem="lc_b", weight=1.0),
    ])
    _stub_solver(monkeypatch, {
        "lc_a": _results(100.0, 0.10, energy=[1.0, 0.0]),
        "lc_b": _results(120.0, 0.20, energy=[0.0, 1.0]),
    }, calls=[])
    status = loop_mod.run_optimization(cfg, log=lambda *_: None)
    assert status.feasible is True
    assert status.sigma_max == 120.0     # worst (max) over cases
    assert status.disp == 0.20


def test_infeasible_if_any_case_violates(case_env, monkeypatch):
    case_dir, out = case_env(("lc_a", "lc_b"))
    cfg = _cfg(case_dir, out, "lc_a", [
        LoadCase(name="a", stem="lc_a", weight=1.0),
        LoadCase(name="b", stem="lc_b", weight=1.0),
    ])
    _stub_solver(monkeypatch, {
        "lc_a": _results(100.0, 0.10, energy=[1.0, 0.0]),   # within limits
        "lc_b": _results(400.0, 0.20, energy=[0.0, 1.0]),   # sigma 400 > 250 allow
    }, calls=[])
    status = loop_mod.run_optimization(cfg, log=lambda *_: None)
    assert status.feasible is False
    assert status.sigma_max == 400.0     # worst case surfaced
    assert status.disp == 0.20


def test_infeasible_if_displacement_case_violates(case_env, monkeypatch):
    case_dir, out = case_env(("lc_a", "lc_b"))
    cfg = _cfg(case_dir, out, "lc_a", [
        LoadCase(name="a", stem="lc_a", weight=1.0),
        LoadCase(name="b", stem="lc_b", weight=1.0),
    ])
    _stub_solver(monkeypatch, {
        "lc_a": _results(100.0, 0.10, energy=[1.0, 0.0]),
        "lc_b": _results(120.0, 5.00, energy=[0.0, 1.0]),   # disp 5.0 > 1.0 allow
    }, calls=[])
    status = loop_mod.run_optimization(cfg, log=lambda *_: None)
    assert status.feasible is False
    assert status.disp == 5.0


def test_per_case_limit_override_relaxes_gate(case_env, monkeypatch):
    case_dir, out = case_env(("lc_a", "lc_b"))
    cfg = _cfg(case_dir, out, "lc_a", [
        LoadCase(name="a", stem="lc_a", weight=1.0),
        # case b is allowed a higher stress than the global 250
        LoadCase(name="b", stem="lc_b", weight=1.0, sigma_allow=500.0),
    ])
    _stub_solver(monkeypatch, {
        "lc_a": _results(100.0, 0.10, energy=[1.0, 0.0]),
        "lc_b": _results(400.0, 0.20, energy=[0.0, 1.0]),   # 400 <= its own 500
    }, calls=[])
    status = loop_mod.run_optimization(cfg, log=lambda *_: None)
    assert status.feasible is True
    assert status.sigma_max == 400.0


def test_solve_failure_in_one_case_fails_iteration(case_env, monkeypatch):
    case_dir, out = case_env(("lc_a", "lc_b"))
    cfg = _cfg(case_dir, out, "lc_a", [
        LoadCase(name="a", stem="lc_a", weight=1.0),
        LoadCase(name="b", stem="lc_b", weight=1.0),
    ])
    calls = []
    _stub_solver(monkeypatch, {
        "lc_a": _results(100.0, 0.10, energy=[1.0, 0.0]),
        "lc_b": _results(120.0, 0.20, energy=[0.0, 1.0]),
    }, calls=calls, fail_stems=("lc_b",))
    status = loop_mod.run_optimization(cfg, log=lambda *_: None)
    assert status.state == "failed"
    assert "lc_b" in status.message or "'b'" in status.message
    # case a solved, case b attempted (then aborted) - both got a solve call
    assert [s for s, _ in calls] == ["lc_a", "lc_b"]


# ---- (c) a single-case config reproduces the classic single-solve path -----
def test_single_case_uses_plain_solve_dir_and_one_solve_per_iter(case_env, monkeypatch):
    case_dir, out = case_env(("implicit_demo",))
    cfg = _cfg(case_dir, out, "implicit_demo", load_cases=[], max_iter=2)
    calls = []
    _stub_solver(monkeypatch,
                 {"implicit_demo": _results(100.0, 0.1, energy=[2.0, 1.0])},
                 calls=calls)
    status = loop_mod.run_optimization(cfg, log=lambda *_: None)

    work = cfg.work()
    assert len(calls) == 2                       # exactly one solve per iteration
    for stem, run_dir in calls:
        assert stem == "implicit_demo"
        assert run_dir == work / "solve"         # plain solve/, never solve/case_0
    assert (work / "solve").is_dir()
    assert not list((work / "solve").glob("case_*"))   # no per-case sub-dirs
    assert status.feasible is True
    assert status.sigma_max == 100.0


def test_multi_case_uses_per_case_subdirs(case_env, monkeypatch):
    case_dir, out = case_env(("lc_a", "lc_b"))
    cfg = _cfg(case_dir, out, "lc_a", [
        LoadCase(name="a", stem="lc_a", weight=1.0),
        LoadCase(name="b", stem="lc_b", weight=1.0),
    ])
    calls = []
    _stub_solver(monkeypatch, {
        "lc_a": _results(100.0, 0.1, energy=[1.0, 0.0]),
        "lc_b": _results(120.0, 0.2, energy=[0.0, 1.0]),
    }, calls=calls)
    loop_mod.run_optimization(cfg, log=lambda *_: None)

    work = cfg.work()
    assert [s for s, _ in calls] == ["lc_a", "lc_b"]
    assert calls[0][1] == work / "solve" / "case_0"
    assert calls[1][1] == work / "solve" / "case_1"


def test_mismatched_mesh_across_cases_is_rejected(case_env, monkeypatch):
    case_dir, out = case_env(("lc_a",))
    # second case deck has a different design-element set (drop one element)
    bad = (case_dir / "lc_a_0000.rad").read_text(encoding="utf-8").replace(
        "  60000002  60000002  60000003  60000004  60000005\n", "")
    (case_dir / "lc_b_0000.rad").write_text(bad, encoding="utf-8")
    (case_dir / "lc_b_0001.rad").write_text(
        (case_dir / "lc_a_0001.rad").read_text(encoding="utf-8"), encoding="utf-8")
    cfg = _cfg(case_dir, out, "lc_a", [
        LoadCase(name="a", stem="lc_a", weight=1.0),
        LoadCase(name="b", stem="lc_b", weight=1.0),
    ])
    _stub_solver(monkeypatch, {}, calls=[])
    with pytest.raises(ValueError, match="same mesh"):
        loop_mod.run_optimization(cfg, log=lambda *_: None)


# ---- (e) per-case post-processing: archive + d3plot for EVERY case ---------
def test_multi_case_archives_every_case_each_iteration(case_env, monkeypatch):
    """With archive_iterations on, each iteration's folder holds the curated
    outputs of *every* load case side by side (here just the mutated deck the
    stub solver leaves), not only the primary case's."""
    case_dir, out = case_env(("lc_a", "lc_b"))
    cfg = _cfg(case_dir, out, "lc_a", [
        LoadCase(name="a", stem="lc_a", weight=1.0),
        LoadCase(name="b", stem="lc_b", weight=1.0),
    ])
    cfg.beso.archive_iterations = True
    _stub_solver(monkeypatch, {
        "lc_a": _results(100.0, 0.1, energy=[1.0, 0.0]),
        "lc_b": _results(120.0, 0.2, energy=[0.0, 1.0]),
    }, calls=[])
    loop_mod.run_optimization(cfg, log=lambda *_: None)

    it0 = cfg.work() / "iter_0000"
    assert it0.is_dir()
    names = {p.name for p in it0.iterdir()}
    # both cases archived into the one iteration folder (distinct stems, no clash)
    assert "lc_a_0000.rad" in names
    assert "lc_b_0000.rad" in names


def test_multi_case_converts_d3plot_for_every_case(case_env, monkeypatch):
    """Post-run d3plot conversion runs once per load case, each keyed to its own
    stem and its own solve sub-dir."""
    case_dir, out = case_env(("lc_a", "lc_b"))
    cfg = _cfg(case_dir, out, "lc_a", [
        LoadCase(name="a", stem="lc_a", weight=1.0),
        LoadCase(name="b", stem="lc_b", weight=1.0),
    ])
    _stub_solver(monkeypatch, {
        "lc_a": _results(100.0, 0.1, energy=[1.0, 0.0]),
        "lc_b": _results(120.0, 0.2, energy=[0.0, 1.0]),
    }, calls=[])
    conv_calls = []
    monkeypatch.setattr(loop_mod, "convert_final",
                        lambda c, sd, w, log: conv_calls.append((c.model.stem, Path(sd))))
    loop_mod.run_optimization(cfg, log=lambda *_: None)

    work = cfg.work()
    assert conv_calls == [
        ("lc_a", work / "solve" / "case_0"),
        ("lc_b", work / "solve" / "case_1"),
    ]


def test_single_case_converts_d3plot_once_with_plain_solve_dir(case_env, monkeypatch):
    """A single (classic) case still converts exactly once, against plain
    ``solve/`` — byte-identical to the pre-multi-case behaviour."""
    case_dir, out = case_env(("implicit_demo",))
    cfg = _cfg(case_dir, out, "implicit_demo", load_cases=[])
    _stub_solver(monkeypatch,
                 {"implicit_demo": _results(100.0, 0.1, energy=[2.0, 1.0])},
                 calls=[])
    conv_calls = []
    monkeypatch.setattr(loop_mod, "convert_final",
                        lambda c, sd, w, log: conv_calls.append((c.model.stem, Path(sd))))
    loop_mod.run_optimization(cfg, log=lambda *_: None)

    assert conv_calls == [("implicit_demo", cfg.work() / "solve")]


# ---- (d) config roundtrip --------------------------------------------------
def test_load_cases_default_empty_and_single_resolution():
    cfg = Config()
    assert cfg.load_cases == []                  # opt-in; classic run by default
    cfg.model.stem = "demo"
    cfg.model.disp_node_id = 42
    cfg.constraints.sigma_allow = 200.0
    cfg.constraints.d_allow = 3.0
    cases = cfg.load_case_list()
    assert len(cases) == 1
    c = cases[0]
    assert c.name == "default" and c.stem == "demo" and c.weight == 1.0
    assert c.disp_node_id == 42
    assert c.sigma_allow == 200.0 and c.d_allow == 3.0
    assert c.starter.name == "demo_0000.rad" and c.engine.name == "demo_0001.rad"


def test_load_case_fallbacks_fill_from_model_and_constraints():
    cfg = Config()
    cfg.model.stem = "base"
    cfg.model.disp_node_id = 7
    cfg.constraints.sigma_allow = 250.0
    cfg.constraints.d_allow = 1.0
    cfg.load_cases = [
        LoadCase(name="x", stem="lc_x", weight=2.0),               # inherit all limits
        LoadCase(name="y", stem="", disp_node_id=9, sigma_allow=99.0),  # blank stem
    ]
    cases = cfg.load_case_list()
    assert cases[0].stem == "lc_x" and cases[0].disp_node_id == 7
    assert cases[0].sigma_allow == 250.0 and cases[0].d_allow == 1.0
    assert cases[1].stem == "base"          # blank stem -> model.stem
    assert cases[1].disp_node_id == 9
    assert cases[1].sigma_allow == 99.0 and cases[1].d_allow == 1.0   # global d_allow


def test_load_cases_yaml_roundtrip(tmp_path):
    cfg = Config()
    cfg.load_cases = [
        LoadCase(name="pull_x", stem="lc_x", weight=1.0, disp_node_id=111,
                 sigma_allow=300.0, d_allow=2.0),
        LoadCase(name="pull_y", stem="lc_y", weight=0.5),
    ]
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)
    assert len(back.load_cases) == 2
    assert back.load_cases[0] == cfg.load_cases[0]      # dataclass field equality
    assert back.load_cases[1] == cfg.load_cases[1]
    # resolves the same way after a roundtrip
    assert [c.stem for c in back.load_case_list()] == ["lc_x", "lc_y"]


def test_empty_load_cases_roundtrip(tmp_path):
    cfg = Config()
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    assert Config.from_yaml(p).load_cases == []
