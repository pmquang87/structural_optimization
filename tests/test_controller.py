"""Multipoint feasibility back-off controller: boundary fit, gate fallback,
clamping, floor-on-violation, state round-trip, config selection and the
checkpoint plumbing.

Hermetic: config dataclasses + numpy arrays only, never touches OpenRadioss.
"""
from __future__ import annotations

import numpy as np
import pytest

from oropt.beso import gate_target_vf
from oropt.config import Beso as BesoCfg, Config
from oropt.controller import MultipointBackoff, build_backoff_controller
from oropt.status import load_checkpoint, save_checkpoint


def _cfg(**kw) -> BesoCfg:
    base = dict(evolution_rate=0.02, target_volume_fraction=0.3,
                backoff_mode="multipoint", multipoint_window=5,
                utilization_target=1.0)
    base.update(kw)
    return BesoCfg(**base)


# ---- selection ---------------------------------------------------------------
def test_build_backoff_controller_selects_by_mode():
    assert build_backoff_controller(BesoCfg()) is None                 # default gate
    assert isinstance(build_backoff_controller(_cfg()), MultipointBackoff)
    for block in ("beso", "levelset", "tobs", "hca", "saip"):
        cfg = Config()
        assert build_backoff_controller(getattr(cfg, block)) is None   # all default off
        getattr(cfg, block).backoff_mode = "multipoint"
        assert build_backoff_controller(getattr(cfg, block)) is not None


# ---- gate fallback while the fit is unusable -----------------------------------
def test_falls_back_to_gate_until_two_points():
    c = _cfg()
    ctrl = MultipointBackoff(c)
    for feasible, violation in ((True, 0.8), (False, 1.2), (True, None)):
        assert ctrl.next_target_vf(0.7, feasible, violation) \
            == gate_target_vf(c, 0.7, feasible, violation)
    ctrl.record(0.7, 0.8)                             # one point: still no slope
    assert ctrl.next_target_vf(0.7, True, 0.8) == gate_target_vf(c, 0.7, True, 0.8)


def test_falls_back_on_degenerate_or_wrong_sign_fit():
    c = _cfg()
    # no volume spread: identical vfs cannot support a slope
    ctrl = MultipointBackoff(c)
    ctrl.record(0.7, 0.8); ctrl.record(0.7, 0.9)
    assert ctrl.next_target_vf(0.7, True, 0.9) == gate_target_vf(c, 0.7, True, 0.9)
    # non-negative slope (violation FALLING as material is removed = noise)
    ctrl = MultipointBackoff(c)
    ctrl.record(0.8, 0.7); ctrl.record(0.7, 0.6)
    assert ctrl.next_target_vf(0.7, True, 0.6) == gate_target_vf(c, 0.7, True, 0.6)
    # a violation-free run (no limits: worst_violation = 0 everywhere) never fits
    ctrl = MultipointBackoff(c)
    ctrl.record(0.9, 0.0); ctrl.record(0.8, 0.0); ctrl.record(0.7, 0.0)
    assert ctrl.next_target_vf(0.7, True, 0.0) == gate_target_vf(c, 0.7, True, 0.0)


def test_non_finite_violations_are_not_recorded():
    ctrl = MultipointBackoff(_cfg())
    ctrl.record(0.9, float("inf"))                     # diverged solve
    ctrl.record(0.8, float("nan"))
    ctrl.record(0.7, None)
    assert ctrl.state().shape == (0, 2)


# ---- the fit steers to the predicted boundary ----------------------------------
def test_steps_toward_predicted_boundary_and_glides():
    """violation = 2 - 2*vf: the boundary (v = 1) sits exactly at vf = 0.5.
    From far above, removal runs at the full ER step; near the boundary the
    step shrinks to land on it instead of overshooting — the glide the classic
    gate lacks."""
    c = _cfg(evolution_rate=0.05)
    ctrl = MultipointBackoff(c)
    vf = 0.9
    for _ in range(30):
        violation = 2.0 - 2.0 * vf
        ctrl.record(vf, violation)
        vf = ctrl.next_target_vf(vf, feasible=violation <= 1.0,
                                 violation=violation)
    assert vf == pytest.approx(0.5, abs=0.01)          # parked on the boundary
    # and it got there monotonically once the fit engaged: never below boundary
    assert vf >= 0.5 - 1e-9


def test_step_clamped_to_gate_authority():
    """However far away the fitted boundary is, one step never shrinks faster
    than ER or grows faster than ER*backoff_cap."""
    c = _cfg(evolution_rate=0.02, backoff_cap=4.0)
    ctrl = MultipointBackoff(c)
    # boundary far BELOW: fit v = 2 - 2*vf around vf ~ 0.9 -> boundary 0.5
    ctrl.record(0.92, 0.16); ctrl.record(0.90, 0.20)
    t = ctrl.next_target_vf(0.90, True, 0.20)
    assert t == pytest.approx(0.90 * (1 - 0.02))       # exactly one ER shrink
    # boundary far ABOVE: fit predicts massive material need
    ctrl = MultipointBackoff(c)
    ctrl.record(0.50, 3.0); ctrl.record(0.52, 2.9)
    t = ctrl.next_target_vf(0.52, False, 2.9)
    assert t == pytest.approx(0.52 * (1 + 0.02 * 4.0))  # capped growth
    assert t <= 1.0


def test_infeasible_always_backs_off_at_least_the_floor():
    """When the design is measured infeasible the target must grow, even if
    the (noisy) fit claims the boundary is below the current volume."""
    c = _cfg(evolution_rate=0.02, backoff_floor=0.25)
    ctrl = MultipointBackoff(c)
    # fit says boundary ~0.6, but the run sits at 0.7 and is VIOLATED
    ctrl.record(0.75, 0.85); ctrl.record(0.70, 1.05)
    t = ctrl.next_target_vf(0.70, feasible=False, violation=1.05)
    assert t >= 0.70 * (1 + 0.02 * 0.25) - 1e-12       # grew by >= floor*ER


def test_never_below_target_volume_fraction_or_above_one():
    c = _cfg(evolution_rate=0.5, target_volume_fraction=0.45)
    ctrl = MultipointBackoff(c)
    ctrl.record(0.50, 0.10); ctrl.record(0.48, 0.12)   # boundary far below tvf
    assert ctrl.next_target_vf(0.48, True, 0.12) >= 0.45
    c2 = _cfg(evolution_rate=0.5, backoff_cap=10.0)
    ctrl2 = MultipointBackoff(c2)
    ctrl2.record(0.90, 5.0); ctrl2.record(0.95, 4.0)
    assert ctrl2.next_target_vf(0.95, False, 4.0) <= 1.0


def test_utilization_target_leaves_a_margin():
    """utilization_target = 0.9 parks the design where the fit predicts 90%
    utilisation — above the v = 1 crossing, i.e. more material, a margin."""
    c_tight = _cfg(evolution_rate=0.05, utilization_target=1.0)
    c_safe = _cfg(evolution_rate=0.05, utilization_target=0.9)

    def converge(c):
        ctrl = MultipointBackoff(c)
        vf = 0.9
        for _ in range(30):
            violation = 2.0 - 2.0 * vf
            ctrl.record(vf, violation)
            vf = ctrl.next_target_vf(vf, violation <= 1.0, violation)
        return vf

    assert converge(c_safe) > converge(c_tight)
    assert converge(c_safe) == pytest.approx(0.55, abs=0.01)   # v=0.9 at vf=0.55


# ---- fit windowing -------------------------------------------------------------
def test_fit_uses_only_the_last_window_points():
    """Old history from a different regime must not pollute the local fit: the
    first points lie on a shifted line; only the last `window` points (the
    true local model) decide the boundary."""
    c = _cfg(evolution_rate=0.5, multipoint_window=3, backoff_cap=1.0)
    ctrl = MultipointBackoff(c)
    for vf in (0.95, 0.9, 0.85):                       # stale regime: v = 4 - 4*vf
        ctrl.record(vf, 4.0 - 4.0 * vf)
    for vf in (0.8, 0.7, 0.6):                         # current regime: v = 2 - 2*vf
        ctrl.record(vf, 2.0 - 2.0 * vf)
    # boundary of the current regime is 0.5; with the stale points mixed in the
    # fit would land elsewhere. Generous clamps so the prediction shows through.
    t = ctrl.next_target_vf(0.6, True, 0.8)
    assert t == pytest.approx(0.5, abs=1e-6)


# ---- state round-trip (resume) --------------------------------------------------
def test_state_roundtrip_via_checkpoint(tmp_path):
    ctrl = MultipointBackoff(_cfg())
    ctrl.record(0.9, 1.2)
    ctrl.record(0.8, 1.1)
    save_checkpoint(tmp_path, 7, np.ones(5, bool), None, ctrl=ctrl.state())

    ckpt = load_checkpoint(tmp_path)
    assert ckpt["iteration"] == 7
    restored = MultipointBackoff(_cfg())
    restored.restore(ckpt["ctrl"])
    assert restored.vfs == ctrl.vfs
    assert restored.violations == ctrl.violations


def test_checkpoint_without_ctrl_stays_compatible(tmp_path):
    save_checkpoint(tmp_path, 3, np.ones(4, bool), None)   # no ctrl passed
    ckpt = load_checkpoint(tmp_path)
    assert ckpt["ctrl"] is None
    ctrl = MultipointBackoff(_cfg())
    ctrl.restore(ckpt["ctrl"])                             # a no-op, not a crash
    assert ctrl.state().shape == (0, 2)


def test_restore_ignores_malformed_state():
    ctrl = MultipointBackoff(_cfg())
    ctrl.restore(np.ones(5))                               # wrong shape
    ctrl.restore(np.ones((2, 3)))                          # wrong width
    assert ctrl.state().shape == (0, 2)
