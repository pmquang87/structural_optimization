"""HCA (hybrid cellular automata) optimiser: setpoint bisection hitting the
volume target, move-limited density decay, growth of void candidates, protected
elements, density persistence/thresholding, connectivity, and config/loop
selection.

Hermetic: synthetic meshes + sensitivity arrays only, never touches OpenRadioss.
"""
from __future__ import annotations

import numpy as np
import pytest

from oropt.beso import Beso
from oropt.config import Config, HcaOpts as HcaCfg
from oropt.hca import _ALIVE, _X_MIN, Hca
from oropt.loop import build_optimizer
from oropt.mesh import Mesh
from oropt.results import Results
from oropt.validate import VALID_OPTIMIZERS


def _fan_mesh(n=12):
    """``n`` unit-volume tets that all share node 0, so *any* alive subset stays
    connected (a single component) — isolates the update logic from island-drop."""
    conn = np.array([[0, i + 1, i + 2, i + 3] for i in range(n)])
    return Mesh(centroids=np.zeros((n, 3)), volumes=np.ones(n), conn_rows=conn,
                n_nodes=int(conn.max()) + 1, design_node_min=0)


def _chain_with_island(n_chain=5):
    """A connected chain of tets plus one disconnected island tet (last index)."""
    conn = [[i, i + 1, i + 2, i + 3] for i in range(n_chain)]
    nmax = n_chain + 3
    conn.append([nmax + 10, nmax + 11, nmax + 12, nmax + 13])   # shares no node
    conn = np.array(conn)
    n = len(conn)
    return Mesh(centroids=np.zeros((n, 3)), volumes=np.ones(n), conn_rows=conn,
                n_nodes=int(conn.max()) + 1, design_node_min=0)


def _hca(mesh, protected, target_vf=0.5, kp=1.0, move_limit=1.0,
         field_history_weight=1.0, evolution_rate=0.2):
    cfg = HcaCfg(filter_radius=0.0, target_volume_fraction=target_vf,
                 evolution_rate=evolution_rate, kp=kp, move_limit=move_limit,
                 field_history_weight=field_history_weight)
    return Hca(mesh, cfg, protected)


# ---- (a) the setpoint bisection hits the volume target ------------------------
def test_bisection_largest_volume_not_exceeding_budget():
    mesh = _fan_mesh(10)
    protected = np.zeros(10, bool); protected[0] = True
    hca = _hca(mesh, protected)
    x = np.ones(10)
    field = np.linspace(1.0, 0.1, 10)                 # distinct energies, no ties

    # kept removable volume is non-increasing in the setpoint (bisection is valid)
    stars = np.geomspace(1e-3, 1e3, 41)
    vols = [hca._removable_vol_at(x, field, s) for s in stars]
    assert all(vols[i] >= vols[i + 1] - 1e-12 for i in range(len(vols) - 1))

    budget = 4.0
    s_star = hca._solve_setpoint(x, field, budget)
    v = hca._removable_vol_at(x, field, s_star)
    assert v <= budget + 1e-9                          # never exceeds the budget
    # ... and it's the *largest* such: a slightly smaller setpoint overshoots it
    assert hca._removable_vol_at(x, field, s_star * (1 - 1e-3)) > budget
    assert v >= budget - 1.0 - 1e-9                    # within one (unit) element


def test_step_drives_volume_down_to_target_removing_lowest_energy():
    mesh = _fan_mesh(12)
    protected = np.zeros(12, bool); protected[0] = True
    hca = _hca(mesh, protected)
    sens = np.linspace(1.0, 0.1, 12)                   # element 11 least energetic

    alive = np.ones(12, bool)
    prev = hca.volume_fraction(alive)
    assert prev == 1.0
    for target in (0.75, 0.5, 0.25):
        alive = hca.update(alive, sens, target)        # successively shrink
        vf = hca.volume_fraction(alive)
        assert vf <= prev + 1e-9                       # monotone (never grows past)
        assert vf <= target + 1e-9                     # bisection keeps vol <= target
        assert vf >= target - 1.0 / 12 - 1e-9          # ... within one element of it
        prev = vf

    # the removed elements are exactly the lowest-energy ones (controller ranks
    # by S_e/S*), the highest-energy removable stays, the protected stays
    assert not alive[11] and not alive[10]
    assert alive[1]
    assert alive[0]


def test_move_limit_damps_removal_until_densities_decay():
    """With a small move limit no element can cross the 0.5 alive threshold in
    one step from full density — the design lags the volume target while the
    virtual densities decay, then removal lands selectively (the classic damped
    HCA behaviour; why the default move_limit is 1.0)."""
    mesh = _fan_mesh(12)
    protected = np.zeros(12, bool); protected[0] = True
    hca = _hca(mesh, protected, move_limit=0.2)
    sens = np.linspace(1.0, 0.1, 12)

    alive = np.ones(12, bool)
    alive = hca.update(alive, sens, target_vf=0.5)     # x: 1.0 -> 0.8, none dead
    assert alive.all()
    alive = hca.update(alive, sens, target_vf=0.5)     # x: 0.8 -> 0.6, none dead
    assert alive.all()
    alive = hca.update(alive, sens, target_vf=0.5)     # now the threshold is in reach
    assert hca.volume_fraction(alive) == pytest.approx(0.5)
    assert not alive[11]                                # ... cut from the bottom
    assert alive[1] and alive[0]


# ---- (b) growth: a void candidate with high energy is materialised ------------
def _chain_mesh(n=5):
    conn = np.array([[i, i + 1, i + 2, i + 3] for i in range(n)])
    return Mesh(centroids=np.zeros((n, 3)), volumes=np.ones(n),
                conn_rows=conn, n_nodes=int(conn.max()) + 1, design_node_min=0)


_PROTECTED = np.array([True, False, False, False, False])   # element 0 = seed
_ALIVE0 = np.array([True, True, True, True, False])         # element 4 = candidate
_SENS = np.array([1.0, 1.0, 1.0, 1.0, 5.0])                 # candidate ranks best


def test_hca_grows_void_candidate():
    new = _hca(_chain_mesh(), _PROTECTED, target_vf=1.0).update(
        _ALIVE0, _SENS, target_vf=1.0)
    assert new[4]                        # void candidate added (grown)
    assert new.all()


# ---- (c) protected elements are never removed ----------------------------------
def test_protected_elements_stay_alive_even_at_tiny_target():
    mesh = _fan_mesh(12)
    protected = np.zeros(12, bool); protected[0] = protected[3] = True
    hca = _hca(mesh, protected)
    sens = np.linspace(1.0, 0.1, 12)
    sens[0] = sens[3] = 0.0                            # protected rank LAST by energy

    alive = hca.update(np.ones(12, bool), sens, target_vf=0.1)  # below protected floor

    assert alive[0] and alive[3]                       # forced alive regardless
    assert hca.vol[alive].sum() >= hca.vol[protected].sum() - 1e-9
    assert hca.x[0] == 1.0 and hca.x[3] == 1.0         # ... pinned at full density


# ---- (d) the result stays connected to the anchor ------------------------------
def test_update_drops_island_not_connected_to_anchor():
    mesh = _chain_with_island(n_chain=5)               # 6 elements, last = island
    n = mesh.n_elements
    protected = np.zeros(n, bool); protected[0] = True   # anchor = chain element 0
    hca = _hca(mesh, protected, target_vf=0.9)
    sens = np.full(n, 0.1)
    sens[-1] = 100.0                                   # island has the HIGHEST energy

    alive = hca.update(np.ones(n, bool), sens, target_vf=0.9)

    assert not alive[-1]            # island dropped despite high energy (disconnected)
    assert alive[0]                 # protected/anchor kept
    assert alive[:n - 1].any()      # the connected chain survives


# ---- (e) the density field persists and matches the mask -----------------------
def test_density_field_persists_and_thresholding_self_consistent():
    mesh = _fan_mesh(12)
    protected = np.zeros(12, bool); protected[0] = True
    hca = _hca(mesh, protected)
    sens = np.linspace(1.0, 0.1, 12)

    alive = hca.update(np.ones(12, bool), sens, target_vf=0.75)

    assert hca.x is not None and hca.x.shape == (12,)
    assert (hca.x >= _X_MIN - 1e-12).all() and (hca.x <= 1.0 + 1e-12).all()
    # re-thresholding the stored field reproduces the returned mask (fan mesh
    # drops no islands): alive == {x >= 0.5} | protected
    assert np.array_equal((hca.x >= _ALIVE) | protected, alive)


def test_zero_energy_field_is_a_no_op():
    """No usable signal (failed extraction) -> the design must not erode."""
    mesh = _fan_mesh(8)
    protected = np.zeros(8, bool); protected[0] = True
    hca = _hca(mesh, protected)
    alive0 = np.ones(8, bool); alive0[5] = False
    alive = hca.update(alive0, np.zeros(8), target_vf=0.5)
    assert np.array_equal(alive, alive0)


def test_field_history_blends_previous_iterations():
    mesh = _fan_mesh(6)
    protected = np.zeros(6, bool); protected[0] = True
    hca = _hca(mesh, protected, field_history_weight=0.5)
    s1 = np.linspace(1.0, 0.5, 6)
    s2 = np.linspace(0.2, 2.0, 6)

    hca.update(np.ones(6, bool), s1, target_vf=1.0)
    assert np.allclose(hca._field_prev, s1)            # first iteration: no blend
    hca.update(np.ones(6, bool), s2, target_vf=1.0)
    assert np.allclose(hca._field_prev, 0.5 * s2 + 0.5 * s1)   # LS-TaSC-style EMA


# ---- sensitivity delegation (shares the BESO helpers) --------------------------
def test_raw_sensitivity_and_filter_match_beso():
    mesh = _fan_mesh(5)
    protected = np.zeros(5, bool); protected[0] = True
    hca = _hca(mesh, protected)
    elem_ids = np.array([1, 2, 3, 4, 5])
    res = Results(element_ids=np.array([1, 3, 5]),
                  energy=np.array([10.0, 30.0, 50.0]),
                  vonmises=np.array([1.0, 3.0, 5.0]),
                  sigma_max=5.0, disp=0.1, disp_node_id=None)
    raw = hca.raw_sensitivity(res, elem_ids, np.ones(5, bool))
    assert raw.tolist() == [10, 0, 30, 0, 50]
    assert np.allclose(hca.filter_history(raw, None), raw)          # radius 0 -> identity
    assert np.allclose(hca.filter_history(raw, np.zeros(5)), 0.5 * raw)  # history 0.5


def test_next_target_vf_gate_matches_beso():
    hca = _hca(_fan_mesh(6), np.zeros(6, bool), target_vf=0.6, evolution_rate=0.2)
    assert hca.next_target_vf(0.8, feasible=True) < 0.8     # shrink while feasible
    assert hca.next_target_vf(0.8, feasible=False) > 0.8    # back off when infeasible
    assert hca.next_target_vf(0.61, feasible=True) == 0.6   # never below the floor
    # default knobs: the violation ratio changes nothing (classic binary gate)
    assert hca.next_target_vf(0.8, feasible=False, violation=3.0) \
        == hca.next_target_vf(0.8, feasible=False)
    # violation-aware controller: the same shared gate as BESO
    hca.cfg.backoff_gain = 5.0
    hca.cfg.damping_threshold = 0.9
    assert hca.next_target_vf(0.8, feasible=False, violation=1.1) \
        == pytest.approx(0.8 * (1 + 0.2 * 0.5))      # ER*gain*(v-1) = 0.2*5*0.1
    assert hca.next_target_vf(0.8, feasible=True, violation=0.95) \
        == pytest.approx(0.8 * (1 - 0.2 * 0.5))      # half-damped removal


# ---- (f) config roundtrip + loop selection by optimizer name -------------------
def test_config_roundtrip_and_active_opts(tmp_path):
    cfg = Config()
    assert cfg.optimizer == "beso"                     # default unchanged
    assert cfg.active_opts() is cfg.beso

    cfg.optimizer = "hca"
    cfg.hca.kp = 0.7
    cfg.hca.move_limit = 0.3
    cfg.hca.field_history_weight = 0.6
    cfg.hca.target_volume_fraction = 0.4
    cfg.hca.max_iter = 77
    assert cfg.active_opts() is cfg.hca

    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)
    assert back.optimizer == "hca"
    assert back.active_opts() is back.hca
    assert back.hca.kp == 0.7
    assert back.hca.move_limit == 0.3
    assert back.hca.field_history_weight == 0.6
    assert back.hca.target_volume_fraction == 0.4
    assert back.hca.max_iter == 77


def test_hca_defaults():
    cfg = Config()
    assert cfg.hca.kp == 1.0
    assert cfg.hca.move_limit == 1.0
    assert cfg.hca.field_history_weight == 1.0
    assert "hca" in VALID_OPTIMIZERS


def test_build_optimizer_selects_hca_by_name():
    mesh = _fan_mesh(6)
    protected = np.zeros(6, bool); protected[0] = True

    cfg = Config()
    assert isinstance(build_optimizer(cfg, mesh, protected), Beso)   # default

    cfg.optimizer = "hca"
    assert isinstance(build_optimizer(cfg, mesh, protected), Hca)

    cfg.optimizer = "HCA"                               # case-insensitive
    assert isinstance(build_optimizer(cfg, mesh, protected), Hca)

    cfg.optimizer = "bogus"
    with pytest.raises(ValueError):
        build_optimizer(cfg, mesh, protected)


# ---- external prunes are reconciled (the level-set "phantom volume" leak) -----
def test_grow_request_after_prune_does_not_lose_volume():
    """Ten grow requests in a row after an external prune each iteration must
    not ratchet the volume DOWN (the leak signature: vf falls against a grow
    target while the phantom is re-charged every bisection)."""
    n = 20
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = True
    hca = _hca(mesh, protected, kp=1.0, move_limit=1.0)
    sens = np.linspace(1.0, 0.5, n)

    alive = np.ones(n, bool)
    vf = hca.volume_fraction(alive)
    for _ in range(10):
        target = min(1.0, vf + 0.02)                  # the gate asks to GROW
        alive = hca.update(alive, sens, target)
        # external removal-only post-pass: always shave one alive, unprotected
        # fringe element. Use the HIGHEST-energy one (a real prune shaves the
        # fringe next to hot material, whose filtered energy keeps its density
        # alive) so a stale x cannot self-heal via the bisection ranking.
        removable = np.flatnonzero(alive & ~protected)
        alive = alive.copy(); alive[removable[0]] = False
        vf = hca.volume_fraction(alive)
    # Volume-neutral over each update+prune pair: the run may pay for the
    # first prune, but must not ratchet 1 element down per iteration (the
    # pre-fix trace ends at vf=0.50 here -- half the part gone against ten
    # consecutive GROW requests).
    assert vf >= 1.0 - 2.0 / n - 1e-9


# ---- MHCA variable neighbourhood (Afrousheh et al. 2019) ----------------------
def _line_mesh(n=12, spacing=1.0):
    """``n`` unit-volume tets in a line, centroids ``spacing`` apart, all sharing
    node 0 so connectivity never interferes -- gives the filter radius real
    geometry to act on (the fan mesh's all-zero centroids see every radius as
    all-inclusive)."""
    conn = np.array([[0, i + 1, i + 2, i + 3] for i in range(n)])
    cent = np.zeros((n, 3)); cent[:, 0] = np.arange(n) * spacing
    return Mesh(centroids=cent, volumes=np.ones(n), conn_rows=conn,
                n_nodes=int(conn.max()) + 1, design_node_min=0)


def test_mhca_off_by_default_and_when_not_above_filter_radius():
    mesh = _line_mesh()
    protected = np.zeros(mesh.n_elements, bool); protected[0] = True
    hca = _hca(mesh, protected)                       # radius_start = 0 -> off
    assert hca.cfg.radius_start == 0.0
    for it in (0, 5, 50):
        hca.set_iteration(it)
        assert hca._scheduled_radius() == hca.cfg.filter_radius
    # a start radius at/below the filter radius could only grow -> also off
    cfg = HcaCfg(filter_radius=2.0, radius_start=1.5)
    hca2 = Hca(mesh, cfg, protected)
    hca2.set_iteration(0)
    assert hca2._scheduled_radius() == 2.0


def test_mhca_radius_decays_and_quantises():
    mesh = _line_mesh()
    protected = np.zeros(mesh.n_elements, bool)
    cfg = HcaCfg(filter_radius=1.0, radius_start=4.0, radius_iters=12,
                 radius_steps=4)                      # levels: 4, 3, 2, 1
    hca = Hca(mesh, cfg, protected)

    radii = []
    for it in range(0, 20):
        hca.set_iteration(it)
        radii.append(hca._scheduled_radius())
    assert radii[0] == 4.0                            # starts wide
    assert radii[-1] == 1.0                           # ends at filter_radius
    assert all(radii[i] >= radii[i + 1] for i in range(len(radii) - 1))
    assert set(radii) <= {4.0, 3.0, 2.0, 1.0}         # quantised to the levels
    # past the decay window the radius stays pinned at filter_radius
    hca.set_iteration(100)
    assert hca._scheduled_radius() == 1.0


def test_mhca_filter_widens_early_and_matches_classic_late():
    """Early iterations average over a wider neighbourhood than the base filter;
    once the schedule has decayed the filtering is identical to classic HCA."""
    mesh = _line_mesh(n=12, spacing=1.0)
    protected = np.zeros(mesh.n_elements, bool)
    cfg = HcaCfg(filter_radius=1.0, radius_start=6.0, radius_iters=4,
                 radius_steps=2, history_weight=1.0)  # levels: 6, 1
    hca = Hca(mesh, cfg, protected)
    classic = Hca(mesh, HcaCfg(filter_radius=1.0, history_weight=1.0), protected)

    raw = np.zeros(mesh.n_elements); raw[0] = 1.0     # a point source to smear

    hca.set_iteration(0)                              # radius 6: wide smear
    wide = hca.filter_history(raw, None)
    hca.set_iteration(10)                             # decayed: radius 1
    narrow = hca.filter_history(raw, None)

    assert np.count_nonzero(wide) > np.count_nonzero(narrow)   # wider support early
    assert np.allclose(narrow, classic.filter_history(raw, None))  # late == classic


def test_mhca_filter_matrices_are_cached_per_radius():
    mesh = _line_mesh()
    protected = np.zeros(mesh.n_elements, bool)
    cfg = HcaCfg(filter_radius=1.0, radius_start=3.0, radius_iters=4,
                 radius_steps=3)                      # levels: 3, 2, 1
    hca = Hca(mesh, cfg, protected)
    raw = np.ones(mesh.n_elements)

    for it in range(8):                               # walks all three levels
        hca.set_iteration(it)
        hca.filter_history(raw, None)
    assert set(hca._W_cache.keys()) == {3.0, 2.0, 1.0}
    # a revisited radius reuses the cached matrix (identity, not a rebuild)
    W = hca._W_cache[1.0]
    hca.set_iteration(50)
    hca.filter_history(raw, None)
    assert hca._W_cache[1.0] is W


def test_mhca_defaults_roundtrip():
    cfg = Config()
    assert cfg.hca.radius_start == 0.0                # off by default
    assert cfg.hca.radius_iters == 20
    assert cfg.hca.radius_steps == 4
