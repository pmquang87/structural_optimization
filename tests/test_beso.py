"""BESO sensitivity mapping, filtering, and the volume-target update."""
import numpy as np
import pytest

from oropt.beso import Beso
from oropt.config import Beso as BesoCfg
from oropt.mesh import Mesh
from oropt.results import Results


def _chain_mesh(n=5):
    conn = np.array([[i, i + 1, i + 2, i + 3] for i in range(n)])
    return Mesh(centroids=np.zeros((n, 3)), volumes=np.ones(n),
               conn_rows=conn, n_nodes=int(conn.max()) + 1, design_node_min=0)


def _beso(target_vf=0.6, sensitivity="energy"):
    m = _chain_mesh(5)
    cfg = BesoCfg(filter_radius=0.0, target_volume_fraction=target_vf,
                  evolution_rate=0.2, max_add_ratio=0.0, sensitivity=sensitivity)
    protected = np.array([True, False, False, False, False])   # element 0 = seed
    return Beso(m, cfg, protected)


def test_raw_sensitivity_maps_by_id():
    b = _beso()
    elem_ids = np.array([1, 2, 3, 4, 5])
    res = Results(element_ids=np.array([1, 3, 5]),
                  energy=np.array([10.0, 30.0, 50.0]),
                  vonmises=np.array([1.0, 3.0, 5.0]),
                  sigma_max=5.0, disp=0.1, disp_node_id=None)
    raw = b.raw_sensitivity(res, elem_ids, np.ones(5, bool))
    assert raw.tolist() == [10, 0, 30, 0, 50]      # dead/absent elements -> 0


def test_filter_identity_and_history():
    b = _beso()
    raw = np.array([1.0, 2, 3, 4, 5])
    assert np.allclose(b.filter_history(raw, None), raw)         # radius 0 -> identity
    prev = np.zeros(5)
    out = b.filter_history(raw, prev)                            # history_weight 0.5
    assert np.allclose(out, 0.5 * raw)


def test_update_removes_lowest_keeps_protected():
    b = _beso(target_vf=0.6)
    sens = np.array([10.0, 20, 30, 40, 50])   # element 1 is lowest unprotected
    new = b.update(np.ones(5, bool), sens, target_vf=0.6)
    assert new[0]            # protected element kept despite low sensitivity
    assert not new[1]        # lowest-sensitivity unprotected element deleted
    assert new[4]            # highest-sensitivity kept
    assert new.sum() < 5     # something was removed


def test_update_progresses_when_lowest_sensitivity_is_protected():
    """Regression: if the lowest-sensitivity elements are all protected (e.g. a
    low-stress keep-out region), removal must still proceed by deleting the
    lowest-ranked *removable* elements. The old 'rank all, then force protected
    back on' logic deleted only protected elements and restored them, stalling
    the run at the start volume (vf stuck at 1.0 forever)."""
    m = _chain_mesh(5)                                  # vols all 1.0, V0 = 5
    cfg = BesoCfg(filter_radius=0.0, target_volume_fraction=0.6,
                  evolution_rate=0.2, max_add_ratio=0.0)
    protected = np.array([True, True, False, False, False])
    b = Beso(m, cfg, protected)
    sens = np.array([1.0, 2.0, 100.0, 200.0, 300.0])    # protected elems rank LAST

    new = b.update(np.ones(5, bool), sens, target_vf=0.6)

    assert new.sum() < 5                                 # progress: not everything restored
    assert new[0] and new[1]                             # protected always kept
    assert new[4]                                        # best removable kept
    assert not new[2]                                    # lowest removable deleted
    assert abs(b.volume_fraction(new) - 0.6) < 1e-9      # hits the total-volume target


def test_next_target_vf_gate():
    b = _beso()
    assert b.next_target_vf(0.8, feasible=True) < 0.8     # shrink while feasible
    assert b.next_target_vf(0.8, feasible=False) > 0.8    # back off when infeasible
    # never drops below the configured floor
    assert b.next_target_vf(0.61, feasible=True) == b.cfg.target_volume_fraction


def test_next_target_vf_defaults_byte_identical_with_violation():
    """Default knobs (backoff_gain 0, damping_threshold 1.0): passing the
    violation ratio must reproduce the classic binary gate exactly — existing
    runs are unaffected by the new loop plumbing."""
    b = _beso()
    for vf in (0.95, 0.8, 0.61):
        for v in (0.0, 0.5, 0.97, 1.0):     # any feasible utilisation
            assert b.next_target_vf(vf, feasible=True, violation=v) \
                == b.next_target_vf(vf, feasible=True)
        for v in (1.001, 1.3, 50.0):        # any violation magnitude
            assert b.next_target_vf(vf, feasible=False, violation=v) \
                == b.next_target_vf(vf, feasible=False)


def test_next_target_vf_without_violation_keeps_binary_gate():
    """A caller on the pre-existing two-argument interface gets the classic
    gate even with the controller knobs set (violation defaults to None)."""
    b = _beso()                              # er = 0.2
    b.cfg.backoff_gain = 10.0
    b.cfg.damping_threshold = 0.9
    assert b.next_target_vf(0.8, feasible=True) == 0.8 * (1 - 0.2)
    assert b.next_target_vf(0.8, feasible=False) == 0.8 * (1 + 0.2)


def test_next_target_vf_proportional_backoff_scales_with_violation():
    b = _beso()                              # er = 0.2
    b.cfg.backoff_gain = 5.0
    b.cfg.backoff_cap = 4.0
    fixed = b.next_target_vf(0.8, feasible=False)         # classic +ER step
    just_over = b.next_target_vf(0.8, feasible=False, violation=1.01)
    way_over = b.next_target_vf(0.8, feasible=False, violation=1.2)
    # a hair over the limit -> a nudge, not the full fixed step
    assert 0.8 < just_over < fixed
    assert just_over == pytest.approx(0.8 * (1 + 0.2 * 5.0 * 0.01))
    # the growth step scales with the violation magnitude, up to the cap
    assert way_over > just_over
    assert way_over == pytest.approx(0.8 * (1 + 0.2 * 1.0))   # gain*(v-1) = 1.0
    # saturation: er_eff <= cap * ER, and the target never exceeds 1.0
    assert b.next_target_vf(0.5, feasible=False, violation=100.0) \
        == pytest.approx(0.5 * (1 + 0.2 * 4.0))               # capped at 4x ER
    assert b.next_target_vf(0.9, feasible=False, violation=100.0) == 1.0
    # cap of 1x ER reproduces the classic step even for a huge violation
    b.cfg.backoff_cap = 1.0
    assert b.next_target_vf(0.8, feasible=False, violation=100.0) == fixed


def test_next_target_vf_damps_removal_near_the_limit():
    b = _beso()                              # er = 0.2, floor 0.6
    b.cfg.damping_threshold = 0.9
    # at/below the threshold: the full removal rate
    full = b.next_target_vf(0.8, feasible=True, violation=0.9)
    assert full == b.next_target_vf(0.8, feasible=True)
    # halfway between threshold and the limit: half the removal rate
    half = b.next_target_vf(0.8, feasible=True, violation=0.95)
    assert half == pytest.approx(0.8 * (1 - 0.2 * 0.5))
    # exactly on the limit: hold (glide into the constraint, don't overshoot)
    at_limit = b.next_target_vf(0.8, feasible=True, violation=1.0)
    assert at_limit == 0.8
    assert full < half < at_limit            # closer to the limit -> less removal
