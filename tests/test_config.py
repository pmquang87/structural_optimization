"""Config (de)serialisation and the run/output-folder fallback behaviour."""
from __future__ import annotations

from oropt.config import Config


def test_run_folder_defaults_to_case_dir():
    cfg = Config()
    cfg.model.case_dir = r"E:\decks\my_case"
    cfg.work_dir = ""
    assert cfg.run_folder() == r"E:\decks\my_case"


def test_run_folder_uses_explicit_work_dir():
    cfg = Config()
    cfg.model.case_dir = r"E:\decks\my_case"
    cfg.work_dir = "runs/run01"
    assert cfg.run_folder() == "runs/run01"


def test_run_folder_blank_variants_fall_back_to_case_dir():
    cfg = Config()
    cfg.model.case_dir = "/data/case"
    for blank in ("", "   ", None):
        cfg.work_dir = blank  # type: ignore[assignment]
        assert cfg.run_folder() == "/data/case"


def test_from_yaml_blank_work_dir_defaults_to_input(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text('model:\n  case_dir: /data/case\n  stem: demo\nwork_dir: ""\n',
                 encoding="utf-8")
    cfg = Config.from_yaml(p)
    assert cfg.work_dir == ""
    assert cfg.run_folder() == "/data/case"


def test_from_yaml_missing_work_dir_defaults_to_input(tmp_path):
    p = tmp_path / "cfg.yaml"
    p.write_text("model:\n  case_dir: /data/case\n", encoding="utf-8")
    cfg = Config.from_yaml(p)
    assert cfg.work_dir == ""
    assert cfg.run_folder() == "/data/case"


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


def test_archive_iterations_flag_default_off_and_roundtrips(tmp_path):
    cfg = Config()
    assert cfg.beso.archive_iterations is False        # opt-in
    cfg.beso.archive_iterations = True
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    assert Config.from_yaml(p).beso.archive_iterations is True
