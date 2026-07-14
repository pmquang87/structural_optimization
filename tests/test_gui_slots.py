"""Per-slot CPU allocation (run.solver_slots): the Streamlit-free row<->SolverSlot
helpers, and the run-settings editor rendering without error."""
from __future__ import annotations

from pathlib import Path

from oropt.config import Config, LoadCase, SolverSlot
from oropt.gui.slots import records_from_slots, slots_from_records


def test_records_roundtrip():
    slots = [SolverSlot(np=1, nt=8), SolverSlot(np=2, nt=4)]
    rows = records_from_slots(slots)
    assert rows == [{"np": 1, "nt": 8}, {"np": 2, "nt": 4}]
    back = slots_from_records(rows)
    assert [(s.np, s.nt) for s in back] == [(1, 8), (2, 4)]


def test_blank_trailing_row_dropped():
    rows = [{"np": 1, "nt": 8}, {"np": None, "nt": None}]
    assert [(s.np, s.nt) for s in slots_from_records(rows)] == [(1, 8)]


def test_partial_row_defaults_missing_field():
    # nt filled / np blank -> np=1; np filled / nt blank -> nt=12
    out = slots_from_records([{"np": None, "nt": 6}, {"np": 2, "nt": None}])
    assert [(s.np, s.nt) for s in out] == [(1, 6), (2, 12)]


def test_nan_treated_as_blank():
    out = slots_from_records([{"np": float("nan"), "nt": float("nan")},
                             {"np": 1, "nt": 8}])
    assert [(s.np, s.nt) for s in out] == [(1, 8)]


def test_empty_records_yield_no_slots():
    assert slots_from_records([]) == []
    assert records_from_slots([]) == []


def test_gui_solver_slots_editor_renders(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import oropt
    import oropt.gui.runstate as runstate

    monkeypatch.setattr(runstate, "find_active_run", lambda *a, **k: None)
    cfg = Config()
    cfg.work_dir = str(tmp_path / "work")
    Path(cfg.work_dir).mkdir()
    cfg.load_cases = [LoadCase(name="a", stem="g", sigma_allow=1.0),
                      LoadCase(name="b", stem="h", sigma_allow=1.0)]
    cfg.run.solver_slots = [SolverSlot(np=1, nt=8), SolverSlot(np=1, nt=4)]
    cfg_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(cfg_path)
    app = Path(oropt.__file__).resolve().parent / "gui" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(cfg_path)).run()
    assert not at.exception            # the per-slot editor wired up without error
