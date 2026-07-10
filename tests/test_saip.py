"""SAIP optimiser: canonical-relaxation flips, move limit, volume targeting,
protected elements, oscillation damping, connectivity, config/loop selection.

Hermetic: synthetic meshes + sensitivity arrays only, never touches OpenRadioss
or Docker (the subproblem is solved analytically — no MILP solver involved).
"""
from __future__ import annotations

import math

import numpy as np

from oropt.beso import Beso
from oropt.config import Config, SaipOpts as SaipCfg
from oropt.loop import build_optimizer
from oropt.mesh import Mesh
from oropt.results import Results
from oropt.saip import Saip


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


def _saip(mesh, protected, target_vf=0.5, flip_limit=0.1,
          oscillation_damping=1.0, evolution_rate=0.2):
    cfg = SaipCfg(filter_radius=0.0, target_volume_fraction=target_vf,
                  evolution_rate=evolution_rate, flip_limit=flip_limit,
                  oscillation_damping=oscillation_damping)
    return Saip(mesh, cfg, protected)


def _flips(before: np.ndarray, after: np.ndarray) -> int:
    return int(np.count_nonzero(before.astype(bool) != after.astype(bool)))


# ---- (a) one step respects the flip move-limit --------------------------------
def test_step_respects_flip_move_limit():
    n = 20
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = True
    saip = _saip(mesh, protected, target_vf=0.5, flip_limit=0.1)   # K = floor(0.1*20) = 2
    K = math.floor(0.1 * n)
    assert K == 2
    sens = np.linspace(20.0, 1.0, n)          # element 0 highest (protected), 19 lowest

    alive0 = np.ones(n, bool)
    alive1 = saip.update(alive0, sens, target_vf=0.5)   # demand removing 10, capped by K

    flips = _flips(alive0, alive1)
    assert flips <= K                          # never exceeds the move limit
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
    saip = _saip(mesh, protected, target_vf=0.5, flip_limit=0.5)
    sens = np.linspace(1.0, 0.01, n)          # strictly decreasing; ties impossible

    alive0 = np.ones(n, bool)
    vf0 = saip.volume_fraction(alive0)
    target_vf = saip.next_target_vf(vf0, feasible=True)   # 0.8 (er = 0.2)
    alive1 = saip.update(alive0, sens, target_vf)
    vf1 = saip.volume_fraction(alive1)

    assert vf1 < vf0                                   # volume decreased
    assert vf1 >= target_vf - 2.0 / n - 1e-9           # ... toward (not far past) target
    assert vf1 <= target_vf + 1e-9                     # bisection lands at/below target
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
    saip = _saip(mesh, protected, target_vf=0.05, flip_limit=0.5)
    sens = np.linspace(1.0, 0.1, n)
    sens[[0, 5, 9]] = 0.0                       # protected rank LAST by sensitivity

    alive = np.ones(n, bool)
    for _ in range(5):                          # drive hard toward the tiny target
        vf = saip.volume_fraction(alive)
        alive = saip.update(alive, sens, saip.next_target_vf(vf, feasible=True))
        assert alive[0] and alive[5] and alive[9]     # protected always alive

    # volume can never fall below the protected floor
    assert saip.vol[alive].sum() >= saip.vol[protected].sum() - 1e-9


# ---- (d) the update returns a binary design and stays within the budget -------
def test_update_binary_and_within_budget():
    n = 24
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = True
    saip = _saip(mesh, protected, flip_limit=0.25)
    rng = np.random.default_rng(0)
    sens = rng.random(n) + 0.01

    alive = np.ones(n, bool)
    for target in (0.9, 0.75, 0.6, 0.5):
        alive = saip.update(alive, sens, target_vf=target)
        assert alive.dtype == np.bool_
        assert set(np.unique(alive).tolist()) <= {False, True}
        # the bisection never overshoots the removable budget
        removable_V = saip.vol[alive & ~protected].sum()
        budget = target * saip.V0 - saip.vol[protected].sum()
        assert removable_V <= budget + 1e-9


def test_void_elements_can_be_added_back():
    """Bi-directional: when backing off (infeasible) the relaxation resurrects
    the highest-value-density void elements, bounded by the move limit."""
    n = 16
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = True
    saip = _saip(mesh, protected, flip_limit=0.5)
    sens = np.linspace(1.0, 0.1, n)

    alive = np.ones(n, bool)
    alive[10:] = False                           # some dead elements to resurrect
    before = alive.copy()
    # infeasible -> next_target_vf grows the target -> flips must add volume back
    target = saip.next_target_vf(saip.volume_fraction(alive), feasible=False)
    after = saip.update(alive, sens, target)

    assert saip.vol[after].sum() > saip.vol[before].sum()    # volume grew
    added = np.flatnonzero(after & ~before)
    assert added.size >= 1
    # whatever was added is among the higher-sensitivity dead elements
    assert sens[added].min() >= sens[10:].min() - 1e-9


# ---- value density: cheap-volume elements win ties in sensitivity -------------
def test_ranking_is_value_density_not_raw_sensitivity():
    """Two elements with equal sensitivity but different volume: the smaller
    element has the higher value density s/vol, so the *larger* one is removed
    first — the canonical relaxation prices volume, BESO's raw threshold does
    not."""
    n = 6
    conn = np.array([[0, i + 1, i + 2, i + 3] for i in range(n)])
    vols = np.ones(n); vols[3] = 3.0             # element 3: same s, 3x the volume
    mesh = Mesh(centroids=np.zeros((n, 3)), volumes=vols, conn_rows=conn,
                n_nodes=int(conn.max()) + 1, design_node_min=0)
    protected = np.zeros(n, bool); protected[0] = True
    saip = _saip(mesh, protected, flip_limit=0.5)
    sens = np.array([9.0, 5.0, 5.0, 1.0, 1.0, 9.0])   # 3 and 4 tie on sensitivity

    # budget forces removing ~3 volume units: taking element 3 alone (vol 3,
    # value density 1/3) beats taking both unit elements 4 (density 1) plus
    # another — the dual threshold removes the lowest s/vol first.
    alive = saip.update(np.ones(n, bool), sens, target_vf=(vols.sum() - 3.0) / vols.sum())
    assert not alive[3]                          # the low-density big element goes
    assert alive[4]                              # the equal-s small element survives


# ---- oscillation damping: recently flipped elements rank behind ----------------
def test_oscillation_damping_reranks_recent_flips():
    n = 20
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = True
    sens = np.linspace(20.0, 1.0, n)
    K = 2                                        # flip_limit 0.1 * 20

    def two_steps(damping):
        saip = _saip(mesh, protected, flip_limit=0.1,
                     oscillation_damping=damping)
        a0 = np.ones(n, bool)
        a1 = saip.update(a0, sens, target_vf=0.5)     # removes the K lowest
        # back off: the target grows, so the relaxation wants to re-add the
        # strongest voids — which are exactly the elements just removed
        a2 = saip.update(a1, sens, target_vf=1.0)
        return a1, a2

    # undamped: the just-removed elements are the only voids, so they flip back
    a1, a2 = two_steps(damping=1.0)
    assert _flips(a1, a2) == K                   # full ping-pong
    # damped: candidacy is unchanged (they are still the only positive-gain
    # flips), so the *set* cannot change here — the damping only re-ranks
    a1d, a2d = two_steps(damping=0.5)
    assert (a1d == a1).all() and (a2d == a2).all()
    # the memory records both sides of the last update
    saip = _saip(mesh, protected, flip_limit=0.1, oscillation_damping=0.5)
    a0 = np.ones(n, bool)
    a1 = saip.update(a0, sens, target_vf=0.5)
    assert (saip._prev_in == a0).all() and (saip._prev_out == a1).all()


def test_oscillation_damping_prefers_fresh_candidates():
    """When a just-flipped void competes with a fresh void of slightly lower
    gain for the last move-limit slot, damping hands the slot to the fresh one."""
    n = 10
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = True
    sens = np.linspace(10.0, 1.0, n)

    damped = _saip(mesh, protected, flip_limit=0.1, oscillation_damping=0.1)
    a0 = np.ones(n, bool); a0[8] = a0[9] = False   # two voids from the start
    # first update: K=1 -> re-adds the stronger void (element 8)
    a1 = damped.update(a0, sens, target_vf=1.0)
    assert a1[8] and not a1[9]
    # loop prunes element 8 again (simulates a post-pass) -> it re-voids
    a1[8] = False
    # element 8 just flipped twice (memory: a1 vs previous input a0) -> damped;
    # element 9 (gain slightly lower, never flipped) now outranks it
    a2 = damped.update(a1, sens, target_vf=1.0)
    assert a2[9] and not a2[8]

    # control: without damping the higher-gain element 8 wins the slot again
    undamped = _saip(mesh, protected, flip_limit=0.1, oscillation_damping=1.0)
    undamped.update(a0, sens, target_vf=1.0)
    a2u = undamped.update(a1, sens, target_vf=1.0)
    assert a2u[8] and not a2u[9]


# ---- SCIP conservative subproblem (moving asymptotes, opt-in) ------------------
def _saip_scip(mesh, protected, target_vf=0.5, flip_limit=0.1,
               oscillation_damping=1.0, **scip_knobs):
    """A Saip whose cfg carries the (lead-owned, getattr-read) SCIP knobs, set
    as plain attributes exactly as the future SaipOpts fields will appear."""
    cfg = SaipCfg(filter_radius=0.0, target_volume_fraction=target_vf,
                  evolution_rate=0.2, flip_limit=flip_limit,
                  oscillation_damping=oscillation_damping)
    for name, value in scip_knobs.items():
        setattr(cfg, name, value)
    return Saip(mesh, cfg, protected)


def _run_bouncing(saip, n, iters, seed=3):
    """Drive a fixed pseudo-random scenario with a bouncing volume target;
    returns the list of masks after each update (byte-identity fixture)."""
    rng = np.random.default_rng(seed)
    alive = np.ones(n, bool)
    masks = []
    for i in range(iters):
        sens = rng.random(n) + 0.01
        target = 0.5 if i % 2 == 0 else 1.0
        alive = saip.update(alive, sens, target_vf=target)
        masks.append(alive.copy())
    return masks


def test_scip_default_off_is_byte_identical():
    """With scip_asymptotes absent or false the update path is unchanged: a cfg
    predating the knobs and a cfg with the flag explicitly off both reproduce
    the legacy masks exactly, iteration by iteration."""
    n = 24
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = True

    legacy = _saip(mesh, protected, flip_limit=0.25)              # no scip attrs at all
    off = _saip_scip(mesh, protected, flip_limit=0.25,
                     scip_asymptotes=False,
                     scip_gamma_tight=0.5, scip_gamma_relax=1.5)  # knobs present, flag off
    for m_leg, m_off in zip(_run_bouncing(legacy, n, 8), _run_bouncing(off, n, 8)):
        assert (m_leg == m_off).all()
    assert off._scip_t is None                                    # state never allocated


def _bounce_flip_counts(saip, n, sens, iters):
    """Alternate the volume target so the plain linear subproblem ping-pongs;
    return per-element flip counts over *iters* updates."""
    alive = np.ones(n, bool)
    counts = np.zeros(n, int)
    for i in range(iters):
        target = 0.5 if i % 2 == 0 else 1.0
        new = saip.update(alive, sens, target_vf=target)
        counts += (new != alive)
        alive = new
    return counts


def test_scip_asymptotes_decay_ping_pong():
    """Near-flat sensitivities + K=1 + a bouncing target: the linear subproblem
    hammers the same lowest-sensitivity element in and out every iteration.
    With asymptotes on, its tightened gain hands the churn to fresh elements —
    strictly fewer flips; with no relaxation it stops oscillating entirely."""
    n = 10
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = True
    sens = np.linspace(10.0, 9.1, n)             # e9 lowest; near-flat so t bites
    iters = 12

    plain = _bounce_flip_counts(_saip(mesh, protected, flip_limit=0.1),
                                n, sens, iters)
    assert plain[n - 1] == iters                 # legacy: flips every iteration

    scip = _bounce_flip_counts(
        _saip_scip(mesh, protected, flip_limit=0.1, scip_asymptotes=True),
        n, sens, iters)
    assert scip[n - 1] < plain[n - 1]            # ping-pong decays (default knobs)

    # gamma_relax = 1.0: conservatism never resets -> the oscillation of the
    # hammered element stops for good after the first detected reversal
    saip = _saip_scip(mesh, protected, flip_limit=0.1, scip_asymptotes=True,
                      scip_gamma_relax=1.0)
    alive = np.ones(n, bool)
    flip_iters = []
    for i in range(iters):
        new = saip.update(alive, sens, target_vf=0.5 if i % 2 == 0 else 1.0)
        if new[n - 1] != alive[n - 1]:
            flip_iters.append(i)
        alive = new
    assert flip_iters == [0, 1]                  # removed once, re-added once, then held


def test_scip_conservatism_bounds_and_protected():
    """t_e never leaves [t_min, 1] under aggressive gammas and sustained
    oscillation pressure; protected elements never flip regardless."""
    n = 12
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = protected[3] = True
    sens = np.linspace(1.0, 0.9, n)
    sens[[0, 3]] = 0.0                           # protected rank last -> max pressure
    t_min = 0.2
    saip = _saip_scip(mesh, protected, flip_limit=0.2, scip_asymptotes=True,
                      scip_gamma_tight=0.5, scip_gamma_relax=1.3,
                      scip_t_min=t_min)
    alive = np.ones(n, bool)
    for i in range(20):
        alive = saip.update(alive, sens, target_vf=0.4 if i % 2 == 0 else 1.0)
        assert alive[0] and alive[3]             # protected always alive
        assert (saip._scip_t >= t_min - 1e-12).all()
        assert (saip._scip_t <= 1.0 + 1e-12).all()


def test_scip_monotone_moves_match_plain_mode():
    """A clean monotonic removal sequence (steadily shrinking target, no
    reversals) never tightens an asymptote, so the mode changes nothing: the
    masks are identical on and off — the conservatism only bites on
    oscillation."""
    n = 20
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = True
    sens = np.linspace(2.0, 0.1, n)

    plain = _saip(mesh, protected, flip_limit=0.2)
    scip = _saip_scip(mesh, protected, flip_limit=0.2, scip_asymptotes=True)
    a_plain = np.ones(n, bool)
    a_scip = np.ones(n, bool)
    for target in (0.95, 0.85, 0.75, 0.65, 0.55, 0.5):
        a_plain = plain.update(a_plain, sens, target_vf=target)
        a_scip = scip.update(a_scip, sens, target_vf=target)
        assert (a_scip == a_plain).all()
    assert (scip._scip_t == 1.0).all()           # asymptotes stayed fully relaxed


# ---- all-zero sensitivity: the no-flip safety net ------------------------------
def test_zero_sensitivity_keeps_design_unchanged():
    n = 12
    mesh = _fan_mesh(n)
    protected = np.zeros(n, bool); protected[0] = True
    saip = _saip(mesh, protected, flip_limit=0.5)
    alive = np.ones(n, bool); alive[6:] = False

    out = saip.update(alive, np.zeros(n), target_vf=0.3)
    assert (out == (alive | protected)).all()


# ---- connectivity: islands not connected to the anchor are dropped ------------
def test_update_drops_island_not_connected_to_anchor():
    mesh = _chain_with_island(n_chain=5)         # 6 elements, last is the island
    n = mesh.n_elements
    protected = np.zeros(n, bool); protected[0] = True   # anchor = chain element 0
    saip = _saip(mesh, protected, target_vf=0.9, flip_limit=0.5)
    sens = np.full(n, 0.1)
    sens[-1] = 100.0                             # island has the HIGHEST sensitivity

    alive = saip.update(np.ones(n, bool), sens, target_vf=0.9)

    assert not alive[-1]            # island dropped despite high energy (disconnected)
    assert alive[0]                 # protected/anchor kept
    assert alive[:n - 1].any()      # the connected chain survives


# ---- sensitivity delegation (shares the BESO helpers) -------------------------
def test_raw_sensitivity_and_filter_match_beso():
    mesh = _fan_mesh(5)
    protected = np.zeros(5, bool); protected[0] = True
    saip = _saip(mesh, protected)
    elem_ids = np.array([1, 2, 3, 4, 5])
    res = Results(element_ids=np.array([1, 3, 5]),
                  energy=np.array([10.0, 30.0, 50.0]),
                  vonmises=np.array([1.0, 3.0, 5.0]),
                  sigma_max=5.0, disp=0.1, disp_node_id=None)
    raw = saip.raw_sensitivity(res, elem_ids, np.ones(5, bool))
    assert raw.tolist() == [10, 0, 30, 0, 50]
    assert np.allclose(saip.filter_history(raw, None), raw)          # radius 0 -> identity
    assert np.allclose(saip.filter_history(raw, np.zeros(5)), 0.5 * raw)  # history 0.5


def test_next_target_vf_gate_matches_beso():
    saip = _saip(_fan_mesh(6), np.zeros(6, bool), target_vf=0.6, evolution_rate=0.2)
    assert saip.next_target_vf(0.8, feasible=True) < 0.8     # shrink while feasible
    assert saip.next_target_vf(0.8, feasible=False) > 0.8    # back off when infeasible
    assert saip.next_target_vf(0.61, feasible=True) == 0.6   # never below the floor
    # default knobs: the violation ratio changes nothing (classic binary gate)
    assert saip.next_target_vf(0.8, feasible=False, violation=3.0) \
        == saip.next_target_vf(0.8, feasible=False)


# ---- (e) config roundtrip + loop selection by optimizer name ------------------
def test_config_roundtrip_and_active_opts(tmp_path):
    cfg = Config()
    assert cfg.optimizer == "beso"                     # default unchanged
    assert cfg.active_opts() is cfg.beso

    cfg.optimizer = "saip"
    cfg.saip.flip_limit = 0.03
    cfg.saip.oscillation_damping = 0.7
    cfg.saip.target_volume_fraction = 0.35
    cfg.saip.max_iter = 88
    assert cfg.active_opts() is cfg.saip

    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)
    assert back.optimizer == "saip"
    assert back.active_opts() is back.saip
    assert back.saip.flip_limit == 0.03
    assert back.saip.oscillation_damping == 0.7
    assert back.saip.target_volume_fraction == 0.35
    assert back.saip.max_iter == 88


def test_saip_defaults():
    cfg = Config()
    assert cfg.saip.flip_limit == 0.05
    assert cfg.saip.oscillation_damping == 0.5


def test_build_optimizer_selects_saip_by_name():
    mesh = _fan_mesh(6)
    protected = np.zeros(6, bool); protected[0] = True

    cfg = Config()
    assert isinstance(build_optimizer(cfg, mesh, protected), Beso)   # default

    cfg.optimizer = "saip"
    assert isinstance(build_optimizer(cfg, mesh, protected), Saip)

    cfg.optimizer = "SAIP"                              # case-insensitive
    assert isinstance(build_optimizer(cfg, mesh, protected), Saip)
