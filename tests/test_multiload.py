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
    cfg.model.design_part_id = 60000000
    cfg.model.design_node_min = 60000000
    cfg.model.bc_group_id = 60000000
    cfg.beso.max_iter = max_iter
    cfg.beso.filter_radius = 0.0      # identity filter: keep the test reasoning simple
    # Load cases are the single source of truth; an empty list means the classic
    # single case (stem given). Fill in the per-case defaults these tests assume so
    # a case only has to spell out what it overrides.
    specs = list(load_cases) or [LoadCase(name="default", stem=stem)]
    for lc in specs:
        lc.stem = lc.stem or stem
        if lc.disp_node_id is None:
            lc.disp_node_id = 60000001
        if lc.sigma_allow is None:
            lc.sigma_allow = 250.0
        if lc.d_allow is None:
            lc.d_allow = 1.0
    cfg.load_cases = specs
    cfg.work_dir = str(out_dir)
    return cfg


def _results(sigma: float, disp: float, energy) -> Results:
    energy = np.asarray(energy, dtype=float)
    return Results(element_ids=ELEM_IDS.copy(), energy=energy,
                   vonmises=np.full(ELEM_IDS.size, sigma, dtype=float),
                   sigma_max=sigma, disp=disp, disp_node_id=60000001)


def _stub_solver(monkeypatch, results_by_stem, calls, fail_stems=()):
    def fake_run_solver(cfg, run_dir, stem=None):
        calls.append((stem, Path(run_dir)))
        if stem in fail_stems:
            return RunResult(False, "engine", "ERROR TERMINATION")
        return RunResult(True, "ok", "NORMAL TERMINATION")

    def fake_extract(cfg, run_dir, keep_vtk=False, stem=None, disp_node_id=None,
                     exclude_element_ids=None):
        return results_by_stem[stem]

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


def test_status_reports_per_case_limits_not_global(case_env, monkeypatch):
    """The Monitor reads status.sigma_allow/d_allow + status.cases; these must
    carry each case's OWN limit so the displayed limit matches what gated
    feasibility (the worst-stress / worst-disp case's limit at the headline)."""
    case_dir, out = case_env(("lc_a", "lc_b"))
    cfg = _cfg(case_dir, out, "lc_a", [
        LoadCase(name="a", stem="lc_a", weight=1.0),                 # -> global 250
        LoadCase(name="b", stem="lc_b", weight=1.0, sigma_allow=500.0),
    ])
    _stub_solver(monkeypatch, {
        "lc_a": _results(100.0, 0.30, energy=[1.0, 0.0]),   # worst disp
        "lc_b": _results(400.0, 0.20, energy=[0.0, 1.0]),   # worst stress, own limit 500
    }, calls=[])
    status = loop_mod.run_optimization(cfg, log=lambda *_: None)

    # headline σ_max is case b's (worst); its limit is b's own 500, NOT the global 250
    assert status.sigma_max == 400.0 and status.sigma_allow == 500.0
    # headline disp is case a's (worst); its limit is the global 1.0 fallback
    assert status.disp == 0.30 and status.d_allow == 1.0
    # full per-case breakdown, each gated against its own limit
    by_name = {c["name"]: c for c in status.cases}
    assert by_name["a"]["sigma_allow"] == 250.0 and by_name["b"]["sigma_allow"] == 500.0
    assert by_name["a"]["feasible"] and by_name["b"]["feasible"]


def test_blank_limits_leave_case_unconstrained(case_env, monkeypatch):
    """A load case with blank (None) sigma_allow/d_allow is unconstrained on those
    quantities -> always feasible, however high the stress/displacement, and the
    published limits are NaN / None ('no limit')."""
    case_dir, out = case_env(("lc_a",))
    cfg = _cfg(case_dir, out, "lc_a", [])
    cfg.load_cases[0].sigma_allow = None
    cfg.load_cases[0].d_allow = None
    _stub_solver(monkeypatch,
                 {"lc_a": _results(9999.0, 9999.0, energy=[1.0, 0.0])}, calls=[])
    status = loop_mod.run_optimization(cfg, log=lambda *_: None)

    assert status.feasible is True                       # nothing to violate
    assert status.sigma_allow != status.sigma_allow      # headline limit is NaN
    assert status.d_allow != status.d_allow
    assert status.cases[0]["sigma_allow"] is None        # per-case breakdown: no limit
    assert status.cases[0]["d_allow"] is None


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
    outputs of *every* load case (here just the mutated deck the stub solver
    leaves), each grouped under its own stem-named sub-folder."""
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
    # each case archived into its own stem-named sub-folder, not side by side
    assert (it0 / "lc_a" / "lc_a_0000.rad").is_file()
    assert (it0 / "lc_b" / "lc_b_0000.rad").is_file()
    # the iteration folder holds only the per-case sub-folders (no flat files)
    assert {p.name for p in it0.iterdir()} == {"lc_a", "lc_b"}


def test_run_writes_config_snapshot_into_run_folder(case_env, monkeypatch):
    """run_optimization drops a config_used.yaml in the run folder so every result
    set carries the exact config that produced it (reproducible from its folder)."""
    case_dir, out = case_env(("lc_a",))
    cfg = _cfg(case_dir, out, "lc_a", [])
    _stub_solver(monkeypatch,
                 {"lc_a": _results(100.0, 0.1, energy=[1.0, 0.0])}, calls=[])
    loop_mod.run_optimization(cfg, log=lambda *_: None)

    snap = cfg.work() / "config_used.yaml"
    assert snap.is_file()
    assert Config.from_yaml(snap).load_cases[0].stem == "lc_a"   # round-trips


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
                        lambda c, sd, w, stem, log: conv_calls.append((stem, Path(sd))))
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
                        lambda c, sd, w, stem, log: conv_calls.append((stem, Path(sd))))
    loop_mod.run_optimization(cfg, log=lambda *_: None)

    assert conv_calls == [("implicit_demo", cfg.work() / "solve")]


# ---- (d) load-case resolution (single source of truth, no fallbacks) -------
def test_empty_load_cases_resolve_to_nothing():
    cfg = Config()
    assert cfg.load_cases == []
    assert cfg.load_case_list() == []        # no synthesised default any more


def test_single_load_case_resolves_to_its_own_values():
    cfg = Config()
    cfg.load_cases = [LoadCase(name="pull", stem="demo", weight=2.0,
                               disp_node_id=42, sigma_allow=200.0, d_allow=3.0)]
    cases = cfg.load_case_list()
    assert len(cases) == 1
    c = cases[0]
    assert c.name == "pull" and c.stem == "demo" and c.weight == 2.0
    assert c.disp_node_id == 42
    assert c.sigma_allow == 200.0 and c.d_allow == 3.0
    assert c.starter.name == "demo_0000.rad" and c.engine.name == "demo_0001.rad"


def test_load_case_has_no_model_or_constraints_fallback():
    # Blank stem / unset limits stay as-is (validation rejects them); there is no
    # inheritance from a global model/constraints any more.
    cfg = Config()
    cfg.load_cases = [LoadCase(name="x", stem="", sigma_allow=None, d_allow=None)]
    c = cfg.load_case_list()[0]
    assert c.stem == "" and c.sigma_allow is None and c.d_allow is None


def test_legacy_config_migrates_into_one_load_case():
    # Back-compat: an old single-case YAML (model.stem + constraints, no
    # load_cases) is folded into one explicit load case on read.
    cfg = Config.from_dict({
        "model": {"stem": "base", "disp_node_id": 7},
        "constraints": {"sigma_allow": 250.0, "d_allow": 1.0},
    })
    cases = cfg.load_case_list()
    assert len(cases) == 1
    c = cases[0]
    assert c.stem == "base" and c.disp_node_id == 7
    assert c.sigma_allow == 250.0 and c.d_allow == 1.0


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
