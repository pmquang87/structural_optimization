"""Config (de)serialisation and the run/output-folder fallback behaviour."""
from __future__ import annotations

from pathlib import Path

from oropt.config import Config, unknown_keys


def test_run_folder_defaults_to_case_dir():
    cfg = Config()
    case = r"E:\decks\my_case"
    cfg.model.case_dir = case
    cfg.work_dir = ""
    assert Path(cfg.run_folder()) == Path(case)


def test_run_folder_uses_explicit_work_dir():
    cfg = Config()
    cfg.model.case_dir = r"E:\decks\my_case"
    cfg.work_dir = "runs/run01"
    assert cfg.run_folder() == "runs/run01"


def test_run_folder_blank_variants_fall_back_to_case_dir():
    cfg = Config()
    case = "/data/case"
    cfg.model.case_dir = case
    for blank in ("", "   ", None):
        cfg.work_dir = blank  # type: ignore[assignment]
        assert Path(cfg.run_folder()) == Path(case)


def test_from_yaml_blank_work_dir_defaults_to_case_dir(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text('model:\n  case_dir: /data/case\n  stem: demo\nwork_dir: ""\n',
                 encoding="utf-8")
    cfg = Config.from_yaml(p)
    assert cfg.work_dir == ""
    assert Path(cfg.run_folder()) == Path("/data/case")


def test_from_yaml_missing_work_dir_defaults_to_case_dir(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("model:\n  case_dir: /data/case\n", encoding="utf-8")
    cfg = Config.from_yaml(p)
    assert cfg.work_dir == ""
    assert Path(cfg.run_folder()) == Path("/data/case")


def test_work_creates_explicit_run_folder(tmp_path):
    cfg = Config()
    target = tmp_path / "outdir"
    cfg.work_dir = str(target)
    p = cfg.work()
    assert p == target.resolve()
    assert p.is_dir()


def test_work_falls_back_to_case_dir_and_creates_it(tmp_path):
    cfg = Config()
    case = tmp_path / "case_as_output"
    cfg.model.case_dir = str(case)
    cfg.work_dir = ""
    p = cfg.work()
    assert p == case.resolve()
    assert p.is_dir()


def test_archive_iterations_flag_default_on_and_roundtrips(tmp_path):
    cfg = Config()
    assert cfg.beso.archive_iterations is True          # on by default
    assert cfg.beso.archive_restart is False            # ~345 MB/iter -> opt-in
    cfg.beso.archive_iterations = False
    cfg.beso.archive_restart = True
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)
    assert back.beso.archive_iterations is False
    assert back.beso.archive_restart is True            # roundtrips when opted in


def test_archive_restart_default_off_for_every_optimiser():
    cfg = Config()
    assert cfg.beso.archive_restart is False
    assert cfg.levelset.archive_restart is False
    assert cfg.tobs.archive_restart is False


def test_unknown_keys_flags_typos_and_misplaced_knobs():
    data = {
        "optimizer": "beso",
        "bogus_top": 1,                       # unknown top-level key
        "beso": {"evolution_rate": 0.02, "evolution_ratte": 0.05},  # typo
        "model": {"stem": "demo"},            # all-known section -> nothing
        "load_cases": [{"name": "a", "stem": "lc_a", "whoops": 1}],
        "animate": {"custom_views": [{"name": "v", "nope": 2}]},
    }
    keys = set(unknown_keys(data))
    assert keys == {"bogus_top", "beso.evolution_ratte",
                    "load_cases[0].whoops", "animate.custom_views[0].nope"}


def test_unknown_keys_empty_for_serialised_config():
    # everything Config.to_yaml writes must be a recognised key (no false positives)
    from dataclasses import asdict
    assert unknown_keys(asdict(Config())) == []


def test_protect_bc_and_smooth_defaults_and_roundtrip(tmp_path):
    cfg = Config()
    assert cfg.beso.protect_bc_nodes is True            # default: BC frozen
    assert cfg.smooth.enabled is True                   # on by default
    cfg.beso.protect_bc_nodes = False
    cfg.smooth.enabled = False
    cfg.smooth.method = "laplacian"
    cfg.smooth.output_format = "both"
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)
    assert back.beso.protect_bc_nodes is False
    assert back.smooth.enabled is False
    assert back.smooth.method == "laplacian"
    assert back.smooth.output_format == "both"
