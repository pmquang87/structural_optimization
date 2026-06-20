"""GUI load-case table <-> LoadCase conversion (Streamlit-free, hermetic) plus a
smoke test that the dashboard script renders with the load-case editor wired in.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import oropt
from oropt.config import Config, LoadCase
from oropt.gui.cases import (CASE_COLUMNS, load_cases_from_records,
                             records_from_load_cases)


# ---- pure record <-> LoadCase conversion -----------------------------------
def test_roundtrip_records_load_cases():
    cases = [
        LoadCase(name="pull_z", stem="lc_z", weight=1.0, disp_node_id=111,
                 sigma_allow=300.0, d_allow=2.0),
        LoadCase(name="pull_x", stem="lc_x", weight=0.5),
    ]
    recs = records_from_load_cases(cases)
    assert [r["name"] for r in recs] == ["pull_z", "pull_x"]
    assert set(recs[0]) == set(CASE_COLUMNS)
    assert load_cases_from_records(recs) == cases     # dataclass field equality


def test_blank_optional_cells_become_none():
    # NaN (how pandas blanks a numeric cell) and "" both mean "inherit default"
    recs = [{"name": "x", "stem": "lc_x", "weight": 2.0,
             "disp_node_id": float("nan"), "sigma_allow": "", "d_allow": None}]
    [lc] = load_cases_from_records(recs)
    assert lc.disp_node_id is None
    assert lc.sigma_allow is None
    assert lc.d_allow is None
    assert lc.weight == 2.0


def test_blank_weight_defaults_to_one_but_zero_preserved():
    recs = [{"name": "a", "stem": "lc_a", "weight": None},
            {"name": "b", "stem": "lc_b", "weight": 0.0}]
    a, b = load_cases_from_records(recs)
    assert a.weight == 1.0
    assert b.weight == 0.0


def test_fully_empty_rows_are_dropped():
    recs = [{"name": "", "stem": "", "weight": float("nan")},   # trailing blank row
            {"name": None, "stem": None},
            {"name": "keep", "stem": "lc_k", "weight": 1.0}]
    out = load_cases_from_records(recs)
    assert [lc.name for lc in out] == ["keep"]


def test_blank_name_defaults_but_stem_kept():
    [lc] = load_cases_from_records([{"name": "", "stem": "lc_only", "weight": 1.0}])
    assert lc.name == "case"
    assert lc.stem == "lc_only"


def test_numeric_strings_are_coerced():
    recs = [{"name": "x", "stem": "lc_x", "weight": "2",
             "disp_node_id": "10021367", "sigma_allow": "480", "d_allow": "1.5"}]
    [lc] = load_cases_from_records(recs)
    assert lc.weight == 2.0
    assert lc.disp_node_id == 10021367
    assert lc.sigma_allow == 480.0
    assert lc.d_allow == 1.5


def test_editor_output_resolves_through_config(tmp_path):
    cfg = Config()
    cfg.model.stem = "base"
    cfg.model.disp_node_id = 7
    cfg.constraints.sigma_allow = 250.0
    cfg.constraints.d_allow = 1.0
    cfg.load_cases = load_cases_from_records([
        {"name": "z", "stem": "lc_z", "weight": 1.0},          # inherit limits
        {"name": "uses_model_deck", "stem": "", "weight": 0.5},  # blank stem
    ])
    resolved = cfg.load_case_list()
    assert [c.stem for c in resolved] == ["lc_z", "base"]       # blank -> model.stem
    assert resolved[0].disp_node_id == 7                        # inherited
    assert resolved[0].sigma_allow == 250.0 and resolved[0].d_allow == 1.0
    # survives a YAML roundtrip (Save config -> reload)
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    assert Config.from_yaml(p).load_cases == cfg.load_cases


# ---- dashboard smoke test --------------------------------------------------
def test_app_renders_with_load_case_editor(tmp_path):
    """The Streamlit script runs end-to-end (sidebar + all tabs incl. the new
    load-case editor) without raising, against a minimal on-disk config."""
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

    cfg = Config()
    cfg.model.case_dir = str(tmp_path)
    cfg.work_dir = str(tmp_path / "work")
    cfg.load_cases = [LoadCase(name="a", stem="lc_a", weight=1.0),
                      LoadCase(name="b", stem="lc_b", weight=0.5, sigma_allow=480.0)]
    cfg_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(cfg_path)

    app_file = Path(oropt.__file__).resolve().parent / "gui" / "app.py"
    # Cold process import of streamlit/pyvista/pandas is slow on first run.
    at = AppTest.from_file(str(app_file), default_timeout=30)
    at.run()                                            # default config first
    assert not at.exception
    # repoint at our multi-case config and rerun
    at.sidebar.text_input[0].set_value(str(cfg_path)).run()
    assert not at.exception
