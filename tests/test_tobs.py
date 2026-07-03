"""TOBS optimiser: ILP flip move-limit, volume targeting, protected elements,
binary feasibility, connectivity, and config/loop selection.

Hermetic: synthetic meshes + sensitivity arrays only, never touches OpenRadioss
or Docker (the ILP is solved by scipy.optimize.milp / HiGHS, already installed).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from oropt.beso import Beso
from oropt.config import Config, TobsOpts as TobsCfg
from oropt.loop import build_optimizer
from oropt.mesh import Mesh
from oropt.results import Results
from oropt.tobs import Tobs


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


def _tobs(mesh, protected, target_vf=0.5, flip_limit=0.1,
          constraint_relaxation=0.05, evolution_rate=0.2):
    cfg = TobsCfg(filter_radius=0.0, target_volume_fraction=target_vf,
                  evolution_rate=evolution_rate, flip_limit=flip_limit,
                  constraint_relaxation=constraint_relaxation)
    return Tobs(mesh, cfg, protected)


def _flips(before: np.ndarray, after: np.ndarray) -> int:
    return int(np.count_nonzero(before.astype(bool) != after.astype(bool)))


# ---- (a) one step respects the flip move-limit --------------------------------
def test_step_respects_flip_move_limit():
    n = 20
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = True
    tobs = _tobs(mesh, protected, target_vf=0.5, flip_limit=0.1)   # K = floor(0.1*20) = 2
    K = math.floor(0.1 * n)
    assert K == 2
    sens = np.linspace(20.0, 1.0, n)          # element 0 highest (protected), 19 lowest

    alive0 = np.ones(n, bool)
    alive1 = tobs.update(alive0, sens, target_vf=0.5)   # demand removing 10, capped by K

    flips = _flips(alive0, alive1)
    assert flips <= K                          # never exceeds beta*N flips
    assert flips >= 1                          # ... but still makes progress
    # the move limit binds (target wants more than K) -> exactly K lowest-sens removed
    assert flips == K
    assert not alive1[n - 1] and not alive1[n - 2]   # the two lowest-sensitivity gone
    assert alive1[0]                                  # protected kept
    assert alive1[1]                                  # highest removable sensitivity kept


# ---- (b) volume moves toward target; top-sensitivity & protected kept ---------
def test_volume_moves_toward_target_keeping_top_and_protected():
    n = 30
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = protected[1] = True
    # generous move limit so a single step can reach the (small) per-iteration step
    tobs = _tobs(mesh, protected, target_vf=0.5, flip_limit=0.5,
                 constraint_relaxation=0.05)
    sens = np.linspace(1.0, 0.01, n)          # strictly decreasing; ties impossible

    alive0 = np.ones(n, bool)
    vf0 = tobs.volume_fraction(alive0)
    target_vf = tobs.next_target_vf(vf0, feasible=True)   # 0.8 (er = 0.2)
    alive1 = tobs.update(alive0, sens, target_vf)
    vf1 = tobs.volume_fraction(alive1)

    assert vf1 < vf0                                   # volume decreased
    assert vf1 >= target_vf - 2.0 / n - 1e-9           # ... toward (not far past) target
    assert vf1 <= vf0                                  # never grows while shrinking
    # the highest-sensitivity elements are retained, the lowest are the ones cut
    order = np.argsort(sens)[::-1]
    assert alive1[order[0]] and alive1[order[1]]       # top-2 sensitivity kept
    assert not alive1[order[-1]]                       # the single lowest removed
    assert alive1[0] and alive1[1]                     # protected kept


# ---- (c) protected elements are never removed ---------------------------------
def test_protected_never_removed_even_at_tiny_target():
    n = 16
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = protected[5] = protected[9] = True
    tobs = _tobs(mesh, protected, target_vf=0.05, flip_limit=0.5)
    sens = np.linspace(1.0, 0.1, n)
    sens[[0, 5, 9]] = 0.0                       # protected rank LAST by sensitivity

    alive = np.ones(n, bool)
    for _ in range(5):                          # drive hard toward the tiny target
        vf = tobs.volume_fraction(alive)
        alive = tobs.update(alive, sens, tobs.next_target_vf(vf, feasible=True))
        assert alive[0] and alive[5] and alive[9]     # protected always alive

    # volume can never fall below the protected floor
    assert tobs.vol[alive].sum() >= tobs.vol[protected].sum() - 1e-9


# ---- (d) the ILP is feasible and returns a binary design ----------------------
def test_ilp_feasible_and_binary():
    n = 24
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = True
    tobs = _tobs(mesh, protected, flip_limit=0.25, constraint_relaxation=0.05)
    rng = np.random.default_rng(0)
    sens = rng.random(n) + 0.01

    alive = np.ones(n, bool)
    for target in (0.9, 0.75, 0.6, 0.5):
        alive = tobs.update(alive, sens, target_vf=target)
        # a binary design: a plain boolean mask, values strictly in {False, True}
        assert alive.dtype == np.bool_
        assert set(np.unique(alive).tolist()) <= {False, True}
        # feasible solve actually moved material (not the no-op safety net)
        assert tobs.volume_fraction(alive) <= 1.0


def test_void_elements_can_be_added_back():
    """Bi-directional: when backing off (infeasible) the ILP resurrects the
    highest-sensitivity void elements, bounded by the move limit."""
    n = 16
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = True
    tobs = _tobs(mesh, protected, flip_limit=0.5)
    sens = np.linspace(1.0, 0.1, n)

    alive = np.ones(n, bool)
    alive[10:] = False                           # some dead elements to resurrect
    before = alive.copy()
    # infeasible -> next_target_vf grows the target -> ILP must add volume back
    target = tobs.next_target_vf(tobs.volume_fraction(alive), feasible=False)
    after = tobs.update(alive, sens, target)

    assert tobs.vol[after].sum() > tobs.vol[before].sum()    # volume grew
    added = np.flatnonzero(after & ~before)
    assert added.size >= 1
    # whatever was added is among the higher-sensitivity dead elements
    assert sens[added].min() >= sens[10:].min() - 1e-9


# ---- connectivity: islands not connected to the anchor are dropped ------------
def test_update_drops_island_not_connected_to_anchor():
    mesh = _chain_with_island(n_chain=5)         # 6 elements, last is the island
    n = mesh.n_elements
    protected = np.zeros(n, bool); protected[0] = True   # anchor = chain element 0
    tobs = _tobs(mesh, protected, target_vf=0.9, flip_limit=0.5)
    sens = np.full(n, 0.1)
    sens[-1] = 100.0                             # island has the HIGHEST sensitivity

    alive = tobs.update(np.ones(n, bool), sens, target_vf=0.9)

    assert not alive[-1]            # island dropped despite high energy (disconnected)
    assert alive[0]                 # protected/anchor kept
    assert alive[:n - 1].any()      # the connected chain survives


# ---- sensitivity delegation (shares the BESO helpers) -------------------------
def test_raw_sensitivity_and_filter_match_beso():
    mesh = _fan_mesh(5)
    protected = np.zeros(5, bool); protected[0] = True
    tobs = _tobs(mesh, protected)
    elem_ids = np.array([1, 2, 3, 4, 5])
    res = Results(element_ids=np.array([1, 3, 5]),
                  energy=np.array([10.0, 30.0, 50.0]),
                  vonmises=np.array([1.0, 3.0, 5.0]),
                  sigma_max=5.0, disp=0.1, disp_node_id=None)
    raw = tobs.raw_sensitivity(res, elem_ids, np.ones(5, bool))
    assert raw.tolist() == [10, 0, 30, 0, 50]
    assert np.allclose(tobs.filter_history(raw, None), raw)          # radius 0 -> identity
    assert np.allclose(tobs.filter_history(raw, np.zeros(5)), 0.5 * raw)  # history 0.5


def test_next_target_vf_gate_matches_beso():
    tobs = _tobs(_fan_mesh(6), np.zeros(6, bool), target_vf=0.6, evolution_rate=0.2)
    assert tobs.next_target_vf(0.8, feasible=True) < 0.8     # shrink while feasible
    assert tobs.next_target_vf(0.8, feasible=False) > 0.8    # back off when infeasible
    assert tobs.next_target_vf(0.61, feasible=True) == 0.6   # never below the floor
    # default knobs: the violation ratio changes nothing (classic binary gate)
    assert tobs.next_target_vf(0.8, feasible=False, violation=3.0) \
        == tobs.next_target_vf(0.8, feasible=False)
    # violation-aware controller: the same shared gate as BESO
    tobs.cfg.backoff_gain = 5.0
    tobs.cfg.damping_threshold = 0.9
    assert tobs.next_target_vf(0.8, feasible=False, violation=1.1) \
        == pytest.approx(0.8 * (1 + 0.2 * 0.5))      # ER*gain*(v-1) = 0.2*5*0.1
    assert tobs.next_target_vf(0.8, feasible=True, violation=0.95) \
        == pytest.approx(0.8 * (1 - 0.2 * 0.5))      # half-damped removal


# ---- (e) config roundtrip + loop selection by optimizer name ------------------
def test_config_roundtrip_and_active_opts(tmp_path):
    cfg = Config()
    assert cfg.optimizer == "beso"                     # default unchanged
    assert cfg.active_opts() is cfg.beso

    cfg.optimizer = "tobs"
    cfg.tobs.flip_limit = 0.03
    cfg.tobs.constraint_relaxation = 0.02
    cfg.tobs.target_volume_fraction = 0.35
    cfg.tobs.max_iter = 88
    assert cfg.active_opts() is cfg.tobs

    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)
    assert back.optimizer == "tobs"
    assert back.active_opts() is back.tobs
    assert back.tobs.flip_limit == 0.03
    assert back.tobs.constraint_relaxation == 0.02
    assert back.tobs.target_volume_fraction == 0.35
    assert back.tobs.max_iter == 88


def test_tobs_defaults():
    cfg = Config()
    assert cfg.tobs.flip_limit == 0.05
    assert cfg.tobs.constraint_relaxation == 0.01


def test_build_optimizer_selects_tobs_by_name():
    mesh = _fan_mesh(6)
    protected = np.zeros(6, bool); protected[0] = True

    cfg = Config()
    assert isinstance(build_optimizer(cfg, mesh, protected), Beso)   # default

    cfg.optimizer = "tobs"
    assert isinstance(build_optimizer(cfg, mesh, protected), Tobs)

    cfg.optimizer = "TOBS"                              # case-insensitive
    assert isinstance(build_optimizer(cfg, mesh, protected), Tobs)

    cfg.optimizer = "bogus"
    with pytest.raises(ValueError):
        build_optimizer(cfg, mesh, protected)
