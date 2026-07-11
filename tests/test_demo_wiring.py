"""End-to-end proof of the demo backend wiring (roadmap U2): the REAL
run_optimization loop on the bundled examples/cantilever case with
``demo.enabled`` — no solver, no monkeypatching. This is the "evaluate oropt
with zero OpenRadioss" contract."""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from oropt.config import Config
from oropt.loop import run_optimization
from oropt.runner import backend_problems
from oropt.validate import check_config, has_errors

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "cantilever"


def _demo_cfg(tmp_path) -> Config:
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    for f in EXAMPLE.glob("cantilever_000*.rad"):
        shutil.copy2(f, case_dir / f.name)
    return Config.from_dict({
        "demo": {"enabled": True},
        "model": {"case_dir": str(case_dir), "design_part_id": 60000000,
                  "design_node_min": 60000000, "bc_group_id": 60000000},
        "load_cases": [{"name": "tip", "stem": "cantilever",
                        "sigma_allow": 400.0,
                        "disp_constraints": [{"node_id": 60000699,
                                              "d_allow": 5.0}]}],
        "beso": {"target_volume_fraction": 0.85, "evolution_rate": 0.05,
                 "filter_radius": 8.0, "max_iter": 3, "protect_layers": 1,
                 "archive_iterations": False},
        # keep post-processing light for the test; smoothing/report are
        # exercised elsewhere and best-effort anyway
        "d3plot": {"enabled": False}, "smooth": {"enabled": False},
        "animate": {"enabled": False}, "report": {"enabled": False},
        "work_dir": str(tmp_path / "run"),
    })


@pytest.fixture(scope="module")
def example_present():
    if not (EXAMPLE / "cantilever_0000.rad").is_file():
        pytest.skip("bundled cantilever example missing")


def test_demo_backend_validates_with_no_solver(tmp_path, example_present):
    """demo.enabled bypasses every native-executable requirement, so a machine
    with zero OpenRadioss validates clean (the whole point of the backend)."""
    cfg = _demo_cfg(tmp_path)
    assert backend_problems(cfg) == []
    problems = check_config(cfg)
    assert not has_errors(problems), [str(p) for p in problems]


def test_demo_backend_runs_the_real_loop(tmp_path, example_present):
    """run_optimization end-to-end: iterations happen, material is removed,
    status/history/topology artefacts are written — with no solver installed."""
    cfg = _demo_cfg(tmp_path)
    status = run_optimization(cfg)
    assert status.state in ("converged", "stopped", "running", "failed")
    assert status.state != "failed", status.message
    work = Path(cfg.work_dir)
    assert (work / "status.json").is_file()
    assert (work / "history.csv").is_file()
    assert (work / "topology_latest.vtu").is_file()
    # the loop actually optimised: volume fraction moved below 1
    assert status.volume_fraction < 1.0
    assert status.iteration >= 1
    # synthetic response engaged: finite stress/disp were reported
    assert status.sigma_max > 0.0
