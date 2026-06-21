"""GUI custom-camera-angle table <-> CustomView conversion (Streamlit-free)."""
from __future__ import annotations

from oropt.config import CustomView
from oropt.gui.views import (VIEW_COLUMNS, custom_views_from_records,
                             records_from_custom_views)


def test_roundtrip_records_custom_views():
    views = [CustomView("tq", "front", 30.0, 10.0),
             CustomView("plan", "top", 0.0, 0.0)]
    recs = records_from_custom_views(views)
    assert [r["name"] for r in recs] == ["tq", "plan"]
    assert set(recs[0]) == set(VIEW_COLUMNS)
    assert custom_views_from_records(recs) == views      # dataclass field equality


def test_unnamed_rows_are_dropped():
    recs = [{"name": "", "base": "iso"},                 # trailing blank editor row
            {"name": None, "base": "top"},
            {"name": "keep", "base": "front", "azimuth": 20.0}]
    out = custom_views_from_records(recs)
    assert [cv.name for cv in out] == ["keep"]


def test_blank_base_and_offsets_default():
    recs = [{"name": "x", "base": "", "azimuth": float("nan"), "elevation": None}]
    [cv] = custom_views_from_records(recs)
    assert cv.base == "iso"                              # blank base -> iso
    assert cv.azimuth == 0.0 and cv.elevation == 0.0


def test_numeric_strings_and_base_case_coerced():
    recs = [{"name": "tq", "base": "FRONT", "azimuth": "30", "elevation": "-12"}]
    [cv] = custom_views_from_records(recs)
    assert cv.base == "front"                            # lowercased
    assert cv.azimuth == 30.0 and cv.elevation == -12.0
