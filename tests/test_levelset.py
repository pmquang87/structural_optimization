"""Discrete nodal level-set optimiser: bisected volume targeting, protected
elements, phi->alive self-consistency, connectivity, and config/loop selection.

Hermetic: synthetic meshes only, never touches OpenRadioss.
"""
from __future__ import annotations

import numpy as np
import pytest

from oropt.beso import Beso
from oropt.config import Config, LevelSet as LevelSetCfg
from oropt.levelset import LevelSet
from oropt.loop import build_optimizer
from oropt.mesh import Mesh
from oropt.results import Results


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


def _ls(mesh, protected, target_vf=0.5, smoothing_passes=1, dt=1.0,
        band_width=3.0, evolution_rate=0.2):
    cfg = LevelSetCfg(filter_radius=0.0, target_volume_fraction=target_vf,
                      evolution_rate=evolution_rate, smoothing_passes=smoothing_passes,
                      dt=dt, band_width=band_width)
    return LevelSet(mesh, cfg, protected)


# ---- (a) volume moves monotonically toward the target via the bisection -------
def test_bisection_largest_volume_not_exceeding_budget():
    mesh = _fan_mesh(10)
    protected = np.zeros(10, bool); protected[0] = True
    ls = _ls(mesh, protected, band_width=10.0)
    phi_base = np.linspace(2.0, -2.0, mesh.n_nodes)     # arbitrary nodal field

    # kept removable volume is non-increasing in the shift tau (bisection is valid)
    taus = np.linspace(-3.0, 3.0, 25)
    vols = [ls._removable_vol_at(phi_base, t) for t in taus]
    assert all(vols[i] >= vols[i + 1] - 1e-12 for i in range(len(vols) - 1))

    budget = 4.0
    tau = ls._solve_tau(phi_base, budget)
    v = ls._removable_vol_at(phi_base, tau)
    assert v <= budget + 1e-9                            # never exceeds the budget
    # ... and it's the *largest* such: a slightly smaller shift overshoots it
    assert ls._removable_vol_at(phi_base, tau - 1e-2) > budget
    assert v >= budget - 1.0 - 1e-9                      # within one (unit) element


def test_step_drives_volume_down_to_target():
    mesh = _fan_mesh(12)
    protected = np.zeros(12, bool); protected[0] = True
    ls = _ls(mesh, protected)
    sens = np.linspace(1.0, 0.1, 12)                 # spatially varying energy

    alive = np.ones(12, bool)
    vf = ls.volume_fraction(alive)
    assert vf == 1.0
    prev = vf
    for target in (0.75, 0.5, 0.25):
        alive = ls.update(alive, sens, target)         # successively shrink the design
        vf = ls.volume_fraction(alive)
        assert vf <= prev + 1e-9                      # monotone (never grows past)
        assert vf <= target + 1e-9                    # bisection keeps volume <= target
        assert vf >= target - 1.0 / 12 - 1e-9         # ... and within one element of it
        prev = vf


# ---- (b) protected elements always stay alive ---------------------------------
def test_protected_elements_stay_alive_even_at_tiny_target():
    mesh = _fan_mesh(12)
    protected = np.zeros(12, bool); protected[0] = protected[3] = True
    ls = _ls(mesh, protected)
    sens = np.linspace(1.0, 0.1, 12)
    sens[0] = sens[3] = 0.0                            # protected rank LAST by energy

    alive = ls.update(np.ones(12, bool), sens, target_vf=0.1)   # below protected floor

    assert alive[0] and alive[3]                       # forced alive regardless
    assert ls.vol[alive].sum() >= ls.vol[protected].sum() - 1e-9


# ---- (c) phi -> alive thresholding is self-consistent -------------------------
def test_phi_alive_thresholding_self_consistent():
    mesh = _fan_mesh(12)
    protected = np.zeros(12, bool); protected[0] = True
    ls = _ls(mesh, protected, band_width=10.0)        # large band -> no clip in 1 step
    sens = np.linspace(1.0, 0.1, 12)

    alive = ls.update(np.ones(12, bool), sens, target_vf=0.5)

    # re-thresholding the stored field reproduces the returned mask (fan mesh drops
    # no islands), i.e. alive == {elem mean phi >= 0} | protected
    assert np.array_equal(ls.elements_alive(ls.phi), alive)
    assert ls.phi is not None and ls.phi.shape == (mesh.n_nodes,)


# ---- (d) the result stays connected to the anchor -----------------------------
def test_update_drops_island_not_connected_to_anchor():
    mesh = _chain_with_island(n_chain=5)              # 6 elements, last is the island
    n = mesh.n_elements
    protected = np.zeros(n, bool); protected[0] = True  # anchor = chain element 0
    ls = _ls(mesh, protected, target_vf=0.9)
    sens = np.full(n, 0.1)
    sens[-1] = 100.0                                   # island has the HIGHEST energy

    alive = ls.update(np.ones(n, bool), sens, target_vf=0.9)

    assert not alive[-1]            # island dropped despite high energy (disconnected)
    assert alive[0]                 # protected/anchor kept
    assert alive[:n - 1].any()      # the connected chain survives


# ---- hole nucleation (regression: elevator-linkage run, 2026-07-06) -----------
def _bar_mesh(n=24):
    """A 1-D chain of ``n`` unit-volume tets along x (element i shares 3 nodes
    with element i+1): free surfaces everywhere, so where material is removed is
    entirely up to the optimiser — not to island-dropping."""
    conn = np.array([[i, i + 1, i + 2, i + 3] for i in range(n)])
    cent = np.zeros((n, 3))
    cent[:, 0] = np.arange(n, dtype=float)
    return Mesh(centroids=cent, volumes=np.ones(n), conn_rows=conn,
                n_nodes=int(conn.max()) + 1, design_node_min=0)


def test_solid_bar_carves_low_energy_interior_without_prevoid():
    """On a fully solid bar with NO pre-existing void, the low-energy mid-span
    must be carved within a few updates (hole nucleation on a free surface).
    The old binary phi init gave every element the same clamp value, so away
    from a void fringe the bisection had nothing to discriminate on."""
    n = 24
    mesh = _bar_mesh(n)
    protected = np.zeros(n, bool)
    protected[0] = protected[n - 1] = True          # anchors at both ends
    ls = _ls(mesh, protected)
    # realistic skewed energy: load paths at both ends, slack mid-span
    x = np.arange(n, dtype=float)
    sens = (1e-3 + np.exp(-0.5 * ((x - 4.0) / 3.0) ** 2)
            + np.exp(-0.5 * ((x - (n - 5.0)) / 3.0) ** 2))

    alive = np.ones(n, bool)
    for target in (0.9, 0.8, 0.7):
        alive = ls.update(alive, sens, target)

    removed = np.flatnonzero(~alive)
    assert removed.size >= 4                        # material actually went away
    assert removed.min() >= 8 and removed.max() <= n - 9   # ... from the slack mid-span
    assert alive[:5].all() and alive[-5:].all()     # the loaded ends are intact


def test_carving_is_not_pinned_to_existing_void_interface():
    """The live-run defect (elevator linkage, 2026-07-06): with a binary phi
    init only the smoothing fringe next to an existing void (the growth boxes)
    had any phi variation, so the bisection removed the elements AT the void
    interface — even the hot ones carrying load — while a far low-energy pocket
    was never touched (29,105 of 29,769 removals inside a box over 6
    iterations). On this bar the old init removed the hot interface elements
    [8, 9]; the energy-rank init must carve the slack pocket instead."""
    n = 40
    mesh = _bar_mesh(n)
    protected = np.zeros(n, bool)
    protected[16] = protected[n - 1] = True         # anchors either side of the pocket
    ls = _ls(mesh, protected, smoothing_passes=3)   # production default fringe
    alive = np.ones(n, bool)
    alive[:8] = False                               # pre-void "growth box" at the left
    sens = np.full(n, 0.01)                         # skewed bulk: ~1 % of the peak
    sens[:8] = 0.0                                  # dead elements report no energy
    sens[8:12] = 1.0                                # the void interface carries the load
    sens[24:28] = 0.001                             # slack pocket far from the void

    vf = ls.volume_fraction(alive)                  # 32/40
    new = ls.update(alive, sens, vf - 2.0 / n)      # one update, budget ~2 elements

    removed = set(np.flatnonzero(alive & ~new))
    assert removed                                  # something was removed
    assert removed <= set(range(24, 28))            # ... and only from the slack pocket
    assert new[8:12].all()                          # the hot interface elements survive


def test_constant_volume_swap_resurrects_hot_void_and_carves_pocket():
    """Bi-directional exchange at a *constant* volume target: a high-energy void
    region rises above the threshold while the low-energy pocket sinks below it
    (the nucleation reaction term keeps slack material sinking even when the
    volume budget alone asks for no net removal)."""
    n = 24
    mesh = _bar_mesh(n)
    protected = np.zeros(n, bool)
    protected[6] = protected[n - 1] = True          # anchors either side of the pocket
    ls = _ls(mesh, protected)
    alive = np.ones(n, bool)
    alive[:4] = False                               # hot void: resurrection candidate
    sens = np.full(n, 1.0)
    sens[:4] = 5.0                                  # void region would carry load
    sens[10:14] = 1e-3                              # slack pocket

    vf = ls.volume_fraction(alive)                  # constant target: 20/24
    for _ in range(4):
        alive = ls.update(alive, sens, vf)

    assert alive[:4].all()          # hot void resurrected
    assert not alive[10:14].any()   # slack pocket carved in exchange


# ---- velocity normalisation (regression: H2 squash, elevator-linkage run) -----
def test_speed_scale_ignores_protected_artefact_peak():
    """The unit speed must come from material the optimiser can act on: on the
    elevator-linkage run the global sens argmax sat in the stress-excluded,
    PROTECTED load-introduction region (max/median 417x), so max-normalisation
    left 73% of alive elements moving at < 1% of dt and handed the evolution to
    tau + smoothing instead of the mechanics (docs/levelset_stuck_analysis.md,
    H2)."""
    n = 24
    mesh = _bar_mesh(n)
    protected = np.zeros(n, bool); protected[0] = True
    ls = _ls(mesh, protected)
    alive = np.ones(n, bool)
    sens = np.linspace(2.0, 1.0, n)
    sens[0] = 500.0                                  # protected artefact peak

    scale = ls._speed_scale(alive, sens)
    assert scale <= 2.0 + 1e-9                       # the artefact does not own it
    assert scale == pytest.approx(
        np.percentile(np.abs(sens[alive & ~protected]), 99.0))

    # a dead element's (stale) energy doesn't set the scale either
    alive[5] = False
    sens[5] = 300.0
    assert ls._speed_scale(alive, sens) <= 2.0 + 1e-9


def test_speed_scale_degenerate_fallbacks():
    """p99 of a pool that is >= 99% zeros is 0 -- the scale falls back to the
    pool max, then to the global max, and is 0.0 only for an all-zero field
    (update() then skips normalising instead of dividing by zero)."""
    protected = np.zeros(200, bool); protected[0] = True
    ls = _ls(_fan_mesh(200), protected)
    alive = np.ones(200, bool)

    sens = np.zeros(200); sens[4] = 7.0             # 198/199 actionable are zero
    assert ls._speed_scale(alive, sens) == 7.0      # pool max, not p99 (= 0)

    sens = np.zeros(200); sens[0] = 9.0             # actionable all-zero
    assert ls._speed_scale(alive, sens) == 9.0      # global max keeps dt bounded

    ls_all = _ls(_fan_mesh(200), np.ones(200, bool))  # everything protected
    sens = np.linspace(1.0, 2.0, 200)
    assert ls_all._speed_scale(alive, sens) == pytest.approx(
        np.percentile(sens, 99.0))                  # pool = every element

    assert ls._speed_scale(alive, np.zeros(200)) == 0.0
    out = ls.update(alive, np.zeros(200), target_vf=0.9)   # no div-by-zero
    assert out.any()


def test_artefact_peak_does_not_squash_the_swap():
    """The H2 regression end to end: the constant-volume swap above, but with a
    protected element carrying a 500x load-introduction artefact. Under
    max-normalisation the artefact rescaled every real velocity to ~1e-3 of dt,
    so the hot void could not climb the ~1.5 phi gap to the threshold in any
    realistic number of updates; the robust scale keeps the swap timeline the
    same as without the artefact."""
    n = 24
    mesh = _bar_mesh(n)
    protected = np.zeros(n, bool)
    protected[6] = protected[n - 1] = True          # anchors either side of the pocket
    ls = _ls(mesh, protected)
    alive = np.ones(n, bool)
    alive[:4] = False                               # hot void: resurrection candidate
    sens = np.full(n, 1.0)
    sens[:4] = 5.0                                  # void region would carry load
    sens[10:14] = 1e-3                              # slack pocket
    sens[n - 1] = 500.0                             # protected artefact peak

    vf = ls.volume_fraction(alive)                  # constant target
    for _ in range(4):
        alive = ls.update(alive, sens, vf)

    assert alive[:4].all()          # hot void resurrected despite the artefact
    assert not alive[10:14].any()   # slack pocket carved in exchange


def test_update_invariant_to_protected_artefact_magnitude():
    """Seam contract: how hot the protected artefact is must not steer the
    evolution. Its incident nodes clip to speed 1 either way and the scale is
    computed without it, so 100x vs 10000x gives bit-identical masks and phi.
    (Pre-fix the scale WAS the artefact, so every velocity in the part shrank
    in proportion to it.)"""
    n = 40
    mesh = _bar_mesh(n)
    protected = np.zeros(n, bool)
    protected[0] = protected[n - 1] = True
    base = 1e-3 + np.linspace(1.0, 0.2, n)          # generic decreasing energy
    masks, phis = [], []
    for peak in (100.0, 10000.0):
        ls = _ls(mesh, protected)
        sens = base.copy()
        sens[0] = peak                              # protected artefact
        alive = np.ones(n, bool)
        for target in (0.9, 0.8):
            alive = ls.update(alive, sens, target)
        masks.append(alive)
        phis.append(ls.phi)
    assert np.array_equal(masks[0], masks[1])
    assert np.array_equal(phis[0], phis[1])


# ---- sensitivity delegation (shares the BESO helpers) -------------------------
def test_raw_sensitivity_and_filter_match_beso():
    mesh = _fan_mesh(5)
    protected = np.zeros(5, bool); protected[0] = True
    ls = _ls(mesh, protected)
    elem_ids = np.array([1, 2, 3, 4, 5])
    res = Results(element_ids=np.array([1, 3, 5]),
                  energy=np.array([10.0, 30.0, 50.0]),
                  vonmises=np.array([1.0, 3.0, 5.0]),
                  sigma_max=5.0, disp=0.1, disp_node_id=None)
    raw = ls.raw_sensitivity(res, elem_ids, np.ones(5, bool))
    assert raw.tolist() == [10, 0, 30, 0, 50]
    assert np.allclose(ls.filter_history(raw, None), raw)          # radius 0 -> identity
    assert np.allclose(ls.filter_history(raw, np.zeros(5)), 0.5 * raw)  # history 0.5


def test_next_target_vf_gate_matches_beso():
    ls = _ls(_fan_mesh(6), np.zeros(6, bool), target_vf=0.6, evolution_rate=0.2)
    assert ls.next_target_vf(0.8, feasible=True) < 0.8       # shrink while feasible
    assert ls.next_target_vf(0.8, feasible=False) > 0.8      # back off when infeasible
    assert ls.next_target_vf(0.61, feasible=True) == 0.6     # never below the floor
    # default knobs: the violation ratio changes nothing (classic binary gate)
    assert ls.next_target_vf(0.8, feasible=False, violation=3.0) \
        == ls.next_target_vf(0.8, feasible=False)
    # violation-aware controller: the same shared gate as BESO
    ls.cfg.backoff_gain = 5.0
    ls.cfg.damping_threshold = 0.9
    assert ls.next_target_vf(0.8, feasible=False, violation=1.1) \
        == pytest.approx(0.8 * (1 + 0.2 * 0.5))      # ER*gain*(v-1) = 0.2*5*0.1
    assert ls.next_target_vf(0.8, feasible=True, violation=0.95) \
        == pytest.approx(0.8 * (1 - 0.2 * 0.5))      # half-damped removal


# ---- (e) config roundtrip + loop selection by optimizer name ------------------
def test_config_roundtrip_and_active_opts(tmp_path):
    cfg = Config()
    assert cfg.optimizer == "beso"                     # default unchanged
    assert cfg.active_opts() is cfg.beso

    cfg.optimizer = "levelset"
    cfg.levelset.dt = 0.5
    cfg.levelset.smoothing_passes = 7
    cfg.levelset.band_width = 4.0
    cfg.levelset.target_volume_fraction = 0.3
    assert cfg.active_opts() is cfg.levelset

    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)
    assert back.optimizer == "levelset"
    assert back.active_opts() is back.levelset
    assert back.levelset.dt == 0.5
    assert back.levelset.smoothing_passes == 7
    assert back.levelset.band_width == 4.0
    assert back.levelset.target_volume_fraction == 0.3


def test_build_optimizer_selects_by_name():
    mesh = _fan_mesh(6)
    protected = np.zeros(6, bool); protected[0] = True

    cfg = Config()
    assert isinstance(build_optimizer(cfg, mesh, protected), Beso)

    cfg.optimizer = "levelset"
    assert isinstance(build_optimizer(cfg, mesh, protected), LevelSet)

    cfg.optimizer = "LevelSet"                          # case-insensitive
    assert isinstance(build_optimizer(cfg, mesh, protected), LevelSet)

    cfg.optimizer = "bogus"
    with pytest.raises(ValueError):
        build_optimizer(cfg, mesh, protected)
