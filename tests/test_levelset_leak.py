"""Level-set volume-control leak regression (docs/levelset_stuck_analysis.md).

The 2026-07-05 elevator-linkage run: ``LevelSet.update`` bisects tau so the
thresholded phi keeps exactly the target volume, then removal-only post-passes
(keep_connected inside update; the loop's ``_morph_open`` + keep_connected)
prune the mask -- and phi was never re-synced, so every subsequent bisection
was charged for the pruned ("phantom") volume and eroded real interface
material to pay for it: a permanent ~0.0015*V0/iteration leak that the
proportional back-off could not outpace, even while the gate asked to GROW.
Fixed by reconciling phi with the incoming mask at ``update()`` entry and
refunding the pruned volume to the bisection budget. These tests drive the
exact production chain (``gate_target_vf -> update -> apply_manufacturing ->
keep_connected``, adapted from docs/levelset_stuck_repro.py) and assert the
leak is closed, plus the phi checkpoint and the loop's stall guards.

Hermetic: synthetic meshes only, never touches OpenRadioss.
"""
from __future__ import annotations

import numpy as np

import oropt.status as st
from oropt.beso import gate_target_vf
from oropt.config import LevelSet as LevelSetCfg, ManufacturingOpts
from oropt.levelset import LevelSet
from oropt.loop import GROW_STALL_ITERS, _grow_stall, _removal_spike
from oropt.manufacturing import apply_manufacturing
from oropt.mesh import Mesh


# ---- synthetic case: bar + void growth slab (docs/levelset_stuck_repro.py) ----
NX, NY, NZ = 24, 9, 6
PART_Y = 6.0                        # part below, growth slab (starts void) above


def kuhn_mesh(nx: int, ny: int, nz: int) -> Mesh:
    """Regular grid of unit cubes, each split into the 6 Kuhn tets around the
    (0,0,0)-(1,1,1) diagonal -- conformal across cube faces."""
    xs, ys, zs = np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(float)

    def nid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    kuhn = [(0, 1, 3, 7), (0, 3, 2, 7), (0, 2, 6, 7),
            (0, 6, 4, 7), (0, 4, 5, 7), (0, 5, 1, 7)]
    conn = []
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                c = [nid(i + (b >> 2 & 1), j + (b >> 1 & 1), k + (b & 1))
                     for b in range(8)]
                for t in kuhn:
                    conn.append([c[t[0]], c[t[1]], c[t[2]], c[t[3]]])
    conn = np.asarray(conn, dtype=np.int64)
    pts = xyz[conn]
    a, b, c, d = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    volumes = np.abs(np.einsum("ij,ij->i", a - d,
                               np.cross(b - d, c - d))) / 6.0
    return Mesh(centroids=pts.mean(axis=1), volumes=volumes, conn_rows=conn,
                n_nodes=xyz.shape[0], design_node_min=0)


def _case():
    """Mesh + masks + config of the synthetic elevator-linkage stand-in: the
    run's knobs (ER 0.015, proportional back-off, min_member_layers 1) on a
    bar with a void growth slab on top."""
    mesh = kuhn_mesh(NX, NY, NZ)
    candidate = mesh.centroids[:, 1] > PART_Y          # growth slab starts void
    part = ~candidate
    protected = part & ((mesh.centroids[:, 0] < 2.0)
                        | (mesh.centroids[:, 0] > NX - 2.0))
    cfg = LevelSetCfg(evolution_rate=0.015, filter_radius=1.2,
                      target_volume_fraction=0.3, backoff_gain=1.0,
                      backoff_cap=4.0, damping_threshold=0.9,
                      dt=1.0, smoothing_passes=3, band_width=3.0)
    opt = LevelSet(mesh, cfg, protected, anchor=protected)
    # load-introduction energy peak ~50x the bulk + mild noise (mimics the run)
    c = mesh.centroids
    peak = np.array([float(NX) - 1.0, PART_Y / 2.0, NZ / 2.0])
    field = (0.02 + np.exp(-np.sum((c - peak) ** 2, axis=1) / 8.0)) \
        * (1.0 + 0.05 * np.random.default_rng(42).standard_normal(c.shape[0]))
    return mesh, opt, candidate, protected, field


def _drive(mesh, opt, candidate, protected, field, n_iter=12, n_feasible=2,
           min_member_layers=1):
    """The production chain per iteration, exactly as oropt.loop runs it:
    gate_target_vf -> update -> apply_manufacturing -> keep_connected.

    Instruments the repro's volume decomposition (all of V0): ``phantom`` =
    volume alive in phi but dead in the incoming mask at update() entry (what
    the previous iteration's post-passes pruned), ``vol_update`` = what
    update() returned, ``pruned`` = what this iteration's post-passes then
    removed, ``leak`` = achieved vf minus the gate's target.
    """
    mfg = ManufacturingOpts(min_member_layers=min_member_layers)
    W = mesh.filter_matrix(opt.cfg.filter_radius)
    vol, V0 = mesh.volumes, opt.V0
    alive = ~candidate
    rows = []
    for it in range(n_iter):
        feasible = it < n_feasible
        violation = 0.955 if feasible else 1.002      # pinned barely infeasible
        vf = opt.volume_fraction(alive)
        target_vf = gate_target_vf(opt.cfg, vf, feasible, violation)
        sens = W @ (field * alive)
        phantom = 0.0 if opt.phi is None else float(
            vol[(opt._elem_mean(opt.phi) >= 0.0) & ~alive & ~protected].sum())
        alive = opt.update(alive, sens, target_vf)
        vol_update = float(vol[alive].sum())
        if min_member_layers:
            alive = apply_manufacturing(alive, mesh, mfg, protected,
                                        sensitivity=sens)
            alive = mesh.keep_connected(alive, protected)
        vf_new = opt.volume_fraction(alive)
        rows.append(dict(it=it, feasible=feasible, vf_in=vf,
                         target_vf=target_vf, vf=vf_new,
                         leak=vf_new - target_vf,
                         phantom=phantom / V0,
                         vol_update=vol_update / V0,
                         pruned=vol_update / V0 - vf_new))
    return rows


# ---- the leak is closed --------------------------------------------------------
def test_prune_leak_closed_under_grow_targets():
    """Pre-fix, every pinned-infeasible iteration lost ~0.011*V0 (this mesh)
    AGAINST a grow target: the bisection was charged for the phantom volume
    (phi-alive, mask-dead; phantom(it) == dOpen(it-1)) and eroded real
    interface material to pay for it, permanently. Post-fix the controller
    sees what the iteration keeps: the phantom is measured, re-synced out of
    phi and refunded to the budget, so the update meets target+refund within
    the bisection floor and the prune is volume-neutral over the iteration
    pair. The residual is the *difference of consecutive* open bites --
    zero-mean projection noise, not a drift."""
    mesh, opt, candidate, protected, field = _case()
    rows = _drive(mesh, opt, candidate, protected, field)
    infe = [r for r in rows if not r["feasible"]]
    assert len(infe) >= 8

    # granularity the bisection can actually hit: one (largest) element
    elem_gran = float(mesh.volumes.max()) / opt.V0

    for prev, r in zip(rows, rows[1:]):
        # (1) accounting visibility: what enters update() as phantom is what
        # the previous iteration's post-passes pruned (modulo islands the
        # update's own keep_connected dropped -- measured 0 in the analysis)
        assert abs(r["phantom"] - prev["pruned"]) <= 2 * elem_gran
        # (2) the volume controller hits target + refund within the bisection
        # floor: the pruned volume is charged to the PRUNE, no longer to live
        # interface material (pre-fix: vol_update == target, phantom unpaid)
        assert r["target_vf"] + r["phantom"] - 2 * elem_gran \
            <= r["vol_update"] <= r["target_vf"] + r["phantom"] + 1e-9

    # (3) the ratchet is gone: the gate's grow requests actually grow the
    # design over the pinned stretch (pre-fix: -0.011*V0 EVERY iteration,
    # -0.13 of V0 over this window, against the same grow targets)
    assert infe[-1]["vf"] > infe[0]["vf"]

    # (4) per-iteration tracking is bounded noise, not the open's bite: the
    # worst pre-fix leak was -0.0109..-0.0125 every iteration; post-fix the
    # residual |dOpen(t) - dOpen(t-1)| stays well under half that, and it is
    # zero-mean instead of one-signed
    leaks = [r["leak"] for r in infe[1:]]
    assert max(abs(l) for l in leaks) <= 0.006
    assert abs(float(np.mean(leaks))) <= 0.0015
    assert sum(l > 0 for l in leaks) >= len(leaks) // 2   # not one-signed


def test_prune_leak_closed_without_manufacturing_unchanged():
    """Sanity: with no post-pass pruning there is nothing to refund -- the
    tracking is as tight as before the fix (bisection floor only)."""
    mesh, opt, candidate, protected, field = _case()
    rows = _drive(mesh, opt, candidate, protected, field,
                  min_member_layers=0)
    elem_gran = float(mesh.volumes.max()) / opt.V0
    for r in rows:
        assert -elem_gran <= r["leak"] <= 1e-9


# ---- phi re-sync contract -------------------------------------------------------
def test_resync_phi_kills_pruned_elements_and_returns_their_volume():
    mesh, opt, candidate, protected, field = _case()
    sens = mesh.filter_matrix(opt.cfg.filter_radius) @ (field * ~candidate)
    alive = opt.update(~candidate, sens, 0.95)

    pruned = alive.copy()
    victims = np.flatnonzero(alive & ~protected)[::7][:40]   # scattered prune
    pruned[victims] = False

    got = opt._resync_phi(pruned)
    assert got == float(mesh.volumes[victims].sum())
    # nothing outside the pruned mask (modulo protected) is phi-alive any more
    assert not (opt.elements_alive(opt.phi) & ~pruned & ~protected).any()


def test_external_prune_is_volume_neutral_at_the_next_update():
    """An externally pruned mask must not shrink the NEXT update's outcome:
    the pruned volume is refunded to the budget, so the returned mask still
    meets the same target instead of target-minus-prune (the old behaviour
    charged tau for the phantom volume and eroded the interface)."""
    mesh, opt, candidate, protected, field = _case()
    sens = mesh.filter_matrix(opt.cfg.filter_radius) @ (field * ~candidate)
    alive = opt.update(~candidate, sens, 0.95)

    pruned = alive.copy()
    victims = np.flatnonzero(alive & ~protected)[::7][:40]
    pruned[victims] = False
    target = opt.volume_fraction(alive)                # same volume as before

    new = opt.update(pruned, sens, target)
    elem_gran = float(mesh.volumes.max()) / opt.V0
    assert opt.volume_fraction(new) >= target - elem_gran   # not target - prune


# ---- phi checkpointing ----------------------------------------------------------
def test_resume_with_checkpointed_phi_is_bit_identical(tmp_path):
    """A stopped/resumed run must continue exactly the run it interrupted:
    without phi in the checkpoint the field re-initialises from the mask,
    which both flips elements (~5k measured on the real mesh) and re-orders
    the whole field by the current sensitivity rank."""
    mesh, opt, candidate, protected, field = _case()
    W = mesh.filter_matrix(opt.cfg.filter_radius)

    alive = ~candidate
    targets = [0.96, 0.94, 0.93, 0.925, 0.92]
    for t in targets[:3]:
        sens = W @ (field * alive)
        alive = opt.update(alive, sens, t)
    st.save_checkpoint(tmp_path, 3, alive, sens, phi=opt.phi)

    cont = []
    for t in targets[3:]:
        sens = W @ (field * alive)
        alive = opt.update(alive, sens, t)
        cont.append(alive)

    ckpt = st.load_checkpoint(tmp_path)
    opt2 = LevelSet(mesh, opt.cfg, protected, anchor=protected)
    opt2.phi = ckpt["phi"]
    alive2 = ckpt["alive_mask"]
    for t, expect in zip(targets[3:], cont):
        sens2 = W @ (field * alive2)
        alive2 = opt2.update(alive2, sens2, t)
        assert np.array_equal(alive2, expect)
    assert np.array_equal(opt2.phi, opt.phi)


# ---- loop stall guards ----------------------------------------------------------
def test_grow_stall_counts_only_grow_requests_that_lost_volume():
    n = _grow_stall(0, prev_target_vf=None, prev_vf=None, vf=0.9)
    assert n == 0                                       # no previous request
    n = _grow_stall(0, prev_target_vf=0.9403, prev_vf=0.94, vf=0.9385)
    assert n == 1                                       # grow asked, vf fell
    n = _grow_stall(n, prev_target_vf=0.9388, prev_vf=0.9385, vf=0.937)
    assert n == 2
    assert GROW_STALL_ITERS == 3
    n = _grow_stall(n, prev_target_vf=0.9373, prev_vf=0.937, vf=0.9355)
    assert n == GROW_STALL_ITERS                        # would warn here
    # a shrink request losing volume is the controller working -- reset
    assert _grow_stall(n, prev_target_vf=0.92, prev_vf=0.9355, vf=0.93) == 0
    # a grow request that grew is healthy -- reset
    assert _grow_stall(2, prev_target_vf=0.9403, prev_vf=0.94, vf=0.9402) == 0


def test_removal_spike_needs_history_and_a_real_spike():
    assert not _removal_spike(38175, [])                # first nucleation carve
    assert not _removal_spike(38175, [1300, 1500])      # too little history
    assert _removal_spike(38175, [1300, 1500, 1400])    # the web collapse
    assert not _removal_spike(1600, [1300, 1500, 1400]) # steady fringe rate
    assert not _removal_spike(40, [0, 0, 0])            # quiet phase noise
    assert _removal_spike(500, [0, 0, 0, 2])            # quiet phase, real jump
