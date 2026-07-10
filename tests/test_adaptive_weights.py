"""Adaptive per-load-case weights (oropt.controller.WeightController + the
run.adaptive_weights wiring). See docs/roadmap_2026.md item A1."""
from __future__ import annotations

import numpy as np

from oropt.controller import WeightController, build_weight_controller


def test_violated_case_gains_weight():
    """The case furthest over its limit should gain weight relative to the case
    with the most slack, and the total weight is preserved (combined-sensitivity
    scale unchanged)."""
    wc = WeightController([1.0, 1.0], gain=0.5, bound=4.0)
    w = wc.weights
    for _ in range(3):
        w = wc.update([1.6, 0.4])           # case 0 violated, case 1 slack
    assert w[0] > 1.0 > w[1]
    # scale ≈ preserved by renormalisation (the final safety clamp doesn't fire
    # here, so the sum stays on target)
    assert abs(sum(w) - 2.0) < 0.25


def test_equal_utilisation_is_a_fixed_point():
    """When both cases sit at the same utilisation the weights don't move."""
    wc = WeightController([1.0, 1.0], gain=0.5)
    w = wc.update([0.9, 0.9])
    assert w == [1.0, 1.0]


def test_noop_without_two_usable_signals():
    """Fewer than two finite/positive ratios -> weights untouched (a case with no
    configured limit reports None and sits out)."""
    wc = WeightController([1.0, 2.0], gain=0.5)
    assert wc.update([None, 0.5]) == [1.0, 2.0]
    assert wc.update([np.inf, 0.5]) == [1.0, 2.0]
    assert wc.update([0.0, 0.5]) == [1.0, 2.0]   # non-positive is not usable


def test_bound_clamps_runaway():
    """No weight escapes [base/bound, base*bound] no matter how lopsided the
    violations are."""
    wc = WeightController([1.0, 1.0], gain=1.0, bound=2.0)
    for _ in range(50):
        wc.update([5.0, 0.1])
    assert wc.weights[0] <= 2.0 + 1e-9        # base*bound
    assert wc.weights[1] >= 0.5 - 1e-9        # base/bound


def test_state_restore_round_trip():
    wc = WeightController([1.0, 1.0], gain=0.5)
    for _ in range(4):
        wc.update([1.4, 0.6])
    s = wc.state()
    wc2 = WeightController([1.0, 1.0], gain=0.5)
    wc2.restore(s)
    assert wc2.weights == wc.weights
    # a length mismatch is ignored -> stays at base
    wc3 = WeightController([1.0, 1.0, 1.0])
    wc3.restore(s)
    assert wc3.weights == [1.0, 1.0, 1.0]


def test_builder_gated_on_flag_and_case_count():
    class _Run:
        adaptive_weights = False
        adaptive_weight_gain = 0.5
        adaptive_weight_bound = 4.0

    class _Cfg:
        run = _Run()

    cfg = _Cfg()
    assert build_weight_controller(cfg, [1.0, 1.0]) is None        # flag off
    cfg.run.adaptive_weights = True
    assert build_weight_controller(cfg, [1.0]) is None             # single case
    wc = build_weight_controller(cfg, [1.0, 1.0])
    assert isinstance(wc, WeightController)
    assert wc.gain == 0.5 and wc.bound == 4.0
