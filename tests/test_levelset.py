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

    vf = ls.volume_fraction(np.ones(12, bool))
    assert vf == 1.0
    prev = vf
    for target in (0.75, 0.5, 0.25):
        alive = ls.update(np.ones(12, bool) if prev == 1.0 else alive, sens, target)
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
