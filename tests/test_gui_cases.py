"""GUI load-case table <-> LoadCase conversion (Streamlit-free, hermetic) plus a
smoke test that the dashboard script renders with the load-case editor wired in.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import oropt
from oropt.config import Config, DispConstraint, LoadCase
from oropt.gui.cases import (CASE_COLUMNS, format_disp_constraints,
                             load_cases_from_records, parse_disp_constraints,
                             records_from_load_cases)

pytestmark = pytest.mark.gui   # Streamlit AppTest / pandas-pyarrow: excluded on non-Windows CI


# ---- pure record <-> LoadCase conversion -----------------------------------
def test_roundtrip_records_load_cases():
    cases = [
        LoadCase(name="pull_z", stem="lc_z", weight=1.0, sigma_allow=300.0,
                 disp_constraints=[DispConstraint(node_id=111, d_allow=2.0),
                                   DispConstraint(node_id=222, d_allow=1.5)]),
        LoadCase(name="pull_x", stem="lc_x", weight=0.5),
    ]
    recs = records_from_load_cases(cases)
    assert [r["name"] for r in recs] == ["pull_z", "pull_x"]
    assert set(recs[0]) == set(CASE_COLUMNS)
    # the per-node constraints round-trip through the semicolon-separated column
    assert recs[0]["disp_constraints"] == "111:2; 222:1.5"
    assert load_cases_from_records(recs) == cases     # dataclass field equality


def test_disp_constraints_column_bare_node_is_unconstrained():
    # a bare node (no :limit) tracks the node but leaves it unconstrained
    assert parse_disp_constraints("111") == [DispConstraint(node_id=111,
                                                            d_allow=None)]
    assert format_disp_constraints([DispConstraint(node_id=111)]) == "111"


def test_disp_constraints_column_tolerant_of_separators_and_bad_tokens():
    # ';', ',' and newlines all separate; unparseable tokens are skipped
    parsed = parse_disp_constraints("111:1.0, 222:2\n333 ; oops:5 ; :7")
    assert parsed == [DispConstraint(node_id=111, d_allow=1.0),
                      DispConstraint(node_id=222, d_allow=2.0),
                      DispConstraint(node_id=333, d_allow=None)]
    assert parse_disp_constraints("") == []
    assert parse_disp_constraints(None) == []


def test_blank_optional_cells_become_none():
    # NaN (how pandas blanks a numeric cell) and "" both mean "inherit default"
    recs = [{"name": "x", "stem": "lc_x", "weight": 2.0,
             "disp_constraints": "", "sigma_allow": float("nan")}]
    [lc] = load_cases_from_records(recs)
    assert lc.disp_constraints == []
    assert lc.sigma_allow is None
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
             "disp_constraints": "10021367:1.5; 10021400:2", "sigma_allow": "480"}]
    [lc] = load_cases_from_records(recs)
    assert lc.weight == 2.0
    assert lc.sigma_allow == 480.0
    assert lc.disp_constraints == [DispConstraint(node_id=10021367, d_allow=1.5),
                                   DispConstraint(node_id=10021400, d_allow=2.0)]


def test_editor_output_resolves_through_config(tmp_path):
    # Each load case is the single source of truth for its stem/disp/limits;
    # there is no inheritance from a global model/constraints.
    cfg = Config()
    cfg.load_cases = load_cases_from_records([
        {"name": "z", "stem": "lc_z", "weight": 1.0,
         "disp_constraints": "7:1.0", "sigma_allow": 250.0},
        {"name": "push", "stem": "lc_push", "weight": 0.5,
         "disp_constraints": "9:2.0; 10:3.0", "sigma_allow": 300.0},
    ])
    resolved = cfg.load_case_list()
    assert [c.stem for c in resolved] == ["lc_z", "lc_push"]
    assert resolved[0].disp_constraints == [DispConstraint(node_id=7, d_allow=1.0)]
    assert resolved[0].sigma_allow == 250.0
    assert resolved[1].disp_constraints == [DispConstraint(node_id=9, d_allow=2.0),
                                            DispConstraint(node_id=10, d_allow=3.0)]
    assert resolved[1].sigma_allow == 300.0
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
    # the load-case editor is its own tab (compare in-memory; labels carry emoji)
    assert any("Load cases" in t.label for t in at.tabs)
    # repoint at our multi-case config and rerun
    at.sidebar.text_input[0].set_value(str(cfg_path)).run()
    assert not at.exception


# ---- fast-mode checkbox column ---------------------------------------------
def test_fast_mode_column_present_and_defaults_off():
    assert "fast_mode" in CASE_COLUMNS
    [rec] = records_from_load_cases([LoadCase(name="a", stem="lc_a")])
    assert rec["fast_mode"] is False                     # unchecked by default


def test_fast_mode_roundtrips_through_records():
    cases = [LoadCase(name="fast", stem="lc_f", sigma_allow=254.0, fast_mode=True),
             LoadCase(name="slow", stem="lc_s", sigma_allow=300.0)]
    recs = records_from_load_cases(cases)
    assert [r["fast_mode"] for r in recs] == [True, False]
    assert load_cases_from_records(recs) == cases


def test_fast_mode_blank_or_missing_cell_is_false():
    # a checkbox cell may arrive missing (older row), None or NaN -> unchecked
    for cell in ({}, {"fast_mode": None}, {"fast_mode": float("nan")}):
        row = {"name": "x", "stem": "lc_x", "weight": 1.0, **cell}
        [lc] = load_cases_from_records([row])
        assert lc.fast_mode is False


def test_fast_mode_truthy_text_is_checked():
    # a hand-edited CSV/text cell reads leniently
    for val, want in (("true", True), ("1", True), ("false", False),
                      ("", False), (True, True), (False, False)):
        [lc] = load_cases_from_records([{"name": "x", "stem": "lc_x",
                                         "fast_mode": val}])
        assert lc.fast_mode is want, (val, want)
