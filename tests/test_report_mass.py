"""Absolute mass/cost in the report summary when material density is set
(roadmap item U7). Exercises oropt.report._rows / _summarise directly."""
from __future__ import annotations

from oropt.report import Summary, _rows


def _summary(**over):
    base = dict(
        optimizer="beso", iterations=10, state="converged", message="",
        feasible=True, start_vf=1.0, final_vf=0.5, mass_removed_pct=50.0,
        sigma_max=200.0, sigma_allow=250.0, disp=1.0, d_allow=2.0,
        total_wall_s=100.0, n_cases=1,
    )
    base.update(over)
    return Summary(**base)


def test_no_mass_row_without_density():
    rows = dict(_rows(_summary()))
    assert "Mass" not in rows            # density unset -> volume/% only, unchanged


def test_mass_row_present_with_density():
    s = _summary(design_volume=1000.0, material_density=2.0)   # V0=1000, rho=2
    rows = dict(_rows(s))
    # start mass = 1.0*1000*2 = 2000 ; final = 0.5*1000*2 = 1000
    assert "Mass" in rows
    assert "2000" in rows["Mass"] and "1000" in rows["Mass"]
    assert "Cost" not in rows            # no cost configured


def test_cost_row_present_with_cost():
    s = _summary(design_volume=1000.0, material_density=2.0,
                 material_cost_per_mass=3.0)
    rows = dict(_rows(s))
    assert "Cost" in rows
    # start cost = 2000*3 = 6000
    assert "6000" in rows["Cost"]


def test_no_mass_row_without_design_volume():
    # density set but V0 unknown (pre-upgrade status) -> no mass row, no crash
    rows = dict(_rows(_summary(material_density=2.0, design_volume=0.0)))
    assert "Mass" not in rows
