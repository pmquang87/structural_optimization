"""Fail-fast config validation: one test per error class, plus a clean pass.

All hermetic -- decks and solver executables are tiny tmp files, the Docker image
probe is off by default, so nothing here touches a real solver or daemon.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from oropt.config import Config, DispConstraint, LoadCase
from oropt.runner import backend_problems, run_solver
from oropt.validate import (ERROR, WARNING, Problem, check_config, has_errors,
                            validate_config)


def _good_cfg(tmp_path: Path, sub: str = "g") -> Config:
    """A fully valid single-case config: real decks + fake solver exes, no MPI.

    Everything lives under ``tmp_path/sub`` so a single test can build several
    independent good configs.
    """
    root = tmp_path / sub
    case = root / "case"
    case.mkdir(parents=True)
    (case / "demo_0000.rad").write_text("starter", encoding="utf-8")
    (case / "demo_0001.rad").write_text("engine", encoding="utf-8")
    exec_dir = root / "or" / "exec"
    exec_dir.mkdir(parents=True)
    (exec_dir / "starter_win64.exe").write_text("x", encoding="utf-8")
    (exec_dir / "engine_win64_impi.exe").write_text("x", encoding="utf-8")

    cfg = Config()
    cfg.model.case_dir = str(case)
    cfg.load_cases = [LoadCase(name="demo", stem="demo",
                               sigma_allow=250.0, d_allow=1.0)]
    cfg.or_paths.root = str(root / "or")
    cfg.run.use_mpi = False          # native backend, no mpiexec needed
    return cfg


def _errors(cfg: Config) -> list[str]:
    return [p.message for p in check_config(cfg) if p.severity == ERROR]


def _warnings(cfg: Config) -> list[str]:
    return [p.message for p in check_config(cfg) if p.severity == WARNING]


# ---- the happy path --------------------------------------------------------
def test_well_formed_config_is_clean(tmp_path):
    cfg = _good_cfg(tmp_path)
    assert validate_config(cfg) == []
    assert check_config(cfg) == []
    assert has_errors(check_config(cfg)) is False


@pytest.mark.parametrize("name", ["beso", "levelset", "tobs", "hca",
                                  "BESO", "  Tobs ", "HCA"])
def test_valid_optimizers_accepted(tmp_path, name):
    cfg = _good_cfg(tmp_path)
    cfg.optimizer = name
    assert validate_config(cfg) == []


# ---- unrecognised keys (raw dict passed in) --------------------------------
def test_unknown_keys_error_when_raw_passed(tmp_path):
    cfg = _good_cfg(tmp_path)
    raw = {"beso": {"evolution_ratte": 0.05}}     # typo for evolution_rate
    errs = [p.message for p in check_config(cfg, raw=raw)
            if p.severity == ERROR]
    assert any("evolution_ratte" in e and "unrecognised" in e for e in errs)
    # ...and an unrecognised key now BLOCKS the launch (a typo would otherwise
    # silently revert to the default and waste a multi-hour run)
    assert has_errors(check_config(cfg, raw=raw))


def test_unknown_keys_silent_when_raw_omitted(tmp_path):
    cfg = _good_cfg(tmp_path)
    # without the raw mapping there is nothing to compare against -> clean
    assert check_config(cfg) == []


# ---- optimiser selector ----------------------------------------------------
def test_bad_optimizer_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.optimizer = "genetic"
    errs = _errors(cfg)
    assert any("optimizer must be one of" in e and "genetic" in e for e in errs)
    assert has_errors(check_config(cfg))


# ---- decks / model directory ----------------------------------------------
def test_missing_decks_reported_by_full_path(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.load_cases[0].stem = "absent"    # no absent_0000/_0001 in the case dir
    errs = _errors(cfg)
    base = Path(cfg.model.case_dir).resolve()
    assert any(str(base / "absent_0000.rad") in e for e in errs)
    assert any(str(base / "absent_0001.rad") in e for e in errs)


def test_no_load_cases_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.load_cases = []
    assert any("no load cases defined" in e for e in _errors(cfg))


def test_blank_stem_on_load_case_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.load_cases[0].stem = ""
    assert any("deck stem is required" in e for e in _errors(cfg))


def test_missing_case_dir_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.model.case_dir = str(tmp_path / "nope")
    assert any("model.case_dir does not exist" in e for e in _errors(cfg))


def test_run_folder_under_a_file_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    blocker = tmp_path / "blocker"
    blocker.write_text("x", encoding="utf-8")        # a file, not a directory
    cfg.work_dir = str(blocker / "sub" / "work")     # an ancestor is a file
    assert any("run folder is not creatable" in e for e in _errors(cfg))


# ---- numeric sanity --------------------------------------------------------
@pytest.mark.parametrize("tvf", [0.0, -0.1, 1.5, 2.0])
def test_target_volume_fraction_out_of_range_is_error(tmp_path, tvf):
    cfg = _good_cfg(tmp_path)
    cfg.beso.target_volume_fraction = tvf
    assert any("target_volume_fraction must be in (0, 1]" in e for e in _errors(cfg))


def test_target_volume_fraction_one_is_warning_not_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.beso.target_volume_fraction = 1.0
    assert _errors(cfg) == []
    assert any("no material will be removed" in w for w in _warnings(cfg))


@pytest.mark.parametrize("er", [0.0, -0.5])
def test_nonpositive_evolution_rate_is_error(tmp_path, er):
    cfg = _good_cfg(tmp_path)
    cfg.beso.evolution_rate = er
    assert any("evolution_rate must be > 0" in e for e in _errors(cfg))


def test_negative_filter_radius_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.beso.filter_radius = -1.0
    assert any("filter_radius must be >= 0" in e for e in _errors(cfg))


def test_zero_filter_radius_is_allowed(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.beso.filter_radius = 0.0
    assert _errors(cfg) == []


def test_negative_backoff_gain_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.beso.backoff_gain = -1.0
    assert any("backoff_gain must be >= 0" in e for e in _errors(cfg))


def test_nonpositive_backoff_cap_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.beso.backoff_cap = 0.0
    assert any("backoff_cap must be > 0" in e for e in _errors(cfg))


def test_negative_backoff_floor_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.beso.backoff_floor = -0.1
    assert any("backoff_floor must be >= 0" in e for e in _errors(cfg))


def test_backoff_floor_above_cap_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.beso.backoff_floor = 5.0            # default cap is 4.0
    assert any("backoff_floor must be <= backoff_cap" in e for e in _errors(cfg))


def test_negative_nucleation_rate_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.optimizer = "levelset"
    cfg.levelset.nucleation_rate = -0.5
    assert any("nucleation_rate must be >= 0" in e for e in _errors(cfg))


@pytest.mark.parametrize("dt", [0.0, -0.2, 1.5])
def test_damping_threshold_out_of_range_is_error(tmp_path, dt):
    cfg = _good_cfg(tmp_path)
    cfg.beso.damping_threshold = dt
    assert any("damping_threshold must be in (0, 1]" in e for e in _errors(cfg))


def test_backoff_controller_valid_knobs_are_clean(tmp_path):
    cfg = _good_cfg(tmp_path)             # the good config sets both limits
    cfg.beso.backoff_gain = 10.0
    cfg.beso.backoff_cap = 2.0
    cfg.beso.damping_threshold = 0.95
    assert validate_config(cfg) == []


def test_backoff_controller_without_limits_is_warning(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.load_cases = [LoadCase(name="demo", stem="demo")]   # no limits at all
    cfg.beso.backoff_gain = 10.0
    assert _errors(cfg) == []
    assert any("never engage" in w for w in _warnings(cfg))


def test_negative_addback_stress_bias_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.beso.addback_stress_bias = -0.5
    assert any("addback_stress_bias must be >= 0" in e for e in _errors(cfg))


def test_addback_stress_bias_without_stress_limit_is_warning(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.load_cases = [LoadCase(name="demo", stem="demo", d_allow=1.0)]  # no sigma
    cfg.beso.addback_stress_bias = 2.0
    assert _errors(cfg) == []
    assert any("addback_stress_bias" in w and "never engage" in w
               for w in _warnings(cfg))
    # with a stress limit somewhere it is clean
    cfg.load_cases = [LoadCase(name="demo", stem="demo", sigma_allow=250.0,
                               d_allow=1.0)]
    assert _warnings(cfg) == []


def test_numeric_sanity_follows_selected_optimizer(tmp_path):
    # the active (tobs) block is validated...
    cfg = _good_cfg(tmp_path)
    cfg.optimizer = "tobs"
    cfg.tobs.target_volume_fraction = 0.0
    assert any("target_volume_fraction" in e for e in _errors(cfg))
    # ...and the inactive (beso) block is ignored.
    cfg2 = _good_cfg(tmp_path, "h")
    cfg2.optimizer = "tobs"
    cfg2.beso.target_volume_fraction = 5.0
    assert _errors(cfg2) == []


# ---- per-case weights & feasibility limits --------------------------------
def test_negative_load_case_weight_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.load_cases = [LoadCase(name="a", stem="demo", weight=-1.0),
                      LoadCase(name="b", stem="demo", weight=2.0)]
    errs = _errors(cfg)
    assert any("weight must be >= 0" in e for e in errs)
    assert not any("all load-case weights are zero" in e for e in errs)


def test_all_zero_weights_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.load_cases = [LoadCase(name="a", stem="demo", weight=0.0),
                      LoadCase(name="b", stem="demo", weight=0.0)]
    assert any("all load-case weights are zero" in e for e in _errors(cfg))


def test_nonpositive_sigma_allow_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.load_cases[0].sigma_allow = 0.0
    assert any("sigma_allow must be > 0" in e for e in _errors(cfg))


def test_blank_per_case_limits_are_allowed(tmp_path):
    # sigma_allow / d_allow are optional: a blank limit leaves that quantity
    # unconstrained and is not an error.
    cfg = _good_cfg(tmp_path)
    cfg.load_cases[0].sigma_allow = None
    cfg.load_cases[0].disp_constraints = []
    assert _errors(cfg) == []


def test_disp_constraint_missing_node_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.load_cases[0].disp_constraints = [DispConstraint(node_id=None, d_allow=1.0)]
    assert any("node id is required" in e for e in _errors(cfg))


def test_disp_constraint_nonpositive_d_allow_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.load_cases[0].disp_constraints = [DispConstraint(node_id=42, d_allow=0.0)]
    assert any("d_allow must be > 0" in e for e in _errors(cfg))


def test_multiple_disp_constraints_are_clean(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.load_cases[0].disp_constraints = [
        DispConstraint(node_id=10021367, d_allow=1.0),
        DispConstraint(node_id=10021400, d_allow=2.0)]
    assert validate_config(cfg) == []


# ---- solver backend --------------------------------------------------------
def test_docker_enabled_but_cli_missing_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.docker.enabled = True
    cfg.docker.docker_exe = "definitely-not-a-real-docker-xyz"
    errs = _errors(cfg)
    assert any("docker CLI not found" in e for e in errs)


def test_native_missing_solver_exes_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.or_paths.root = str(tmp_path / "no_such_or")     # starter+engine vanish
    assert sum("executable not found" in e for e in _errors(cfg)) == 2


def test_native_mpiexec_missing_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.run.use_mpi = True
    cfg.or_paths.intel_mpi_root = str(tmp_path / "no_mpi")
    assert any("mpiexec not found" in e for e in _errors(cfg))


def test_native_np_not_one_is_warning(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.run.np = 2
    assert _errors(cfg) == []
    assert any("requires np=1" in w for w in _warnings(cfg))


def test_runner_setup_uses_shared_backend_problems(tmp_path):
    # The refactor: run_solver's pre-flight is exactly backend_problems(), joined.
    cfg = _good_cfg(tmp_path)
    cfg.or_paths.root = str(tmp_path / "absent")
    res = run_solver(cfg, tmp_path)
    assert res.ok is False and res.stage == "setup"
    assert res.message == "; ".join(backend_problems(cfg))


# ---- public API shape ------------------------------------------------------
def test_validate_config_returns_severity_prefixed_strings(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.optimizer = "nope"               # an error
    cfg.run.np = 2                       # a warning
    msgs = validate_config(cfg)
    assert all(isinstance(m, str) for m in msgs)
    assert any(m.startswith("error:") for m in msgs)
    assert any(m.startswith("warning:") for m in msgs)


def test_problem_str_is_severity_prefixed():
    assert str(Problem(ERROR, "boom")) == "error: boom"


def test_duplicate_stems_is_error(tmp_path):
    """The whole multi-case file layout (per-case solve dirs, iter archives,
    iter-0 reuse, d3plots) is keyed by stem: two cases sharing one silently
    overwrite each other's artefacts every iteration."""
    cfg = _good_cfg(tmp_path)
    case = Path(cfg.model.case_dir)
    (case / "pull_0000.rad").write_text("starter", encoding="utf-8")
    (case / "pull_0001.rad").write_text("engine", encoding="utf-8")
    cfg.load_cases = [
        LoadCase(name="a", stem="pull", sigma_allow=250.0, d_allow=1.0),
        LoadCase(name="b", stem="pull", sigma_allow=250.0, d_allow=1.0)]
    errs = _errors(cfg)
    assert any("distinct deck stems" in e and "'pull'" in e for e in errs)


def test_negative_wall_budget_is_error(tmp_path):
    cfg = _good_cfg(tmp_path)
    cfg.run.max_wall_hours = -2.0
    assert any("max_wall_hours" in e for e in _errors(cfg))
