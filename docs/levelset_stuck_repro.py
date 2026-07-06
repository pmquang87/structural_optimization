"""Synthetic reproduction of the elevator-linkage level-set stall (2026-07-05/06).

The real run (levelset, ER 0.015, backoff_gain 1.0, min_member_layers 1, two
growth boxes, code at 2d3c1ce = pre-PR#57) pinned sigma_max ~0.5-1.7 MPa above
the 292 MPa limit from iteration 2 on. The proportional back-off then asked
for a *grow* step (target_vf = vf * (1 + ER*(v-1)) with v-1 ~ 0.002 -> +3e-5),
yet the alive volume fraction kept FALLING ~0.0014-0.0016 of V0 every
iteration. This script reproduces that on a small synthetic case and
decomposes the per-iteration volume budget to show where the loss happens,
using the real oropt code paths end to end:

    gate_target_vf -> LevelSet.update -> apply_manufacturing -> keep_connected

Mechanism being demonstrated (see docs/levelset_stuck_analysis.md):

1. ``LevelSet.update`` bisects tau so the thresholded phi keeps exactly the
   target volume, *then* the mask is pruned by removal-only post-passes
   (keep_connected inside update; the loop's _morph_open + keep_connected).
2. ``self.phi`` is never re-synced to the pruned mask, so next iteration the
   bisection budget is charged for "phantom" volume (phi-alive but pruned),
   and tau erodes real interface material to pay for it, where the prune then
   shaves a fresh one-element fringe again -> a permanent leak.
3. The run-era back-off (no floor) requested growth ~3e-5, far below the
   leak, so the run ratcheted deeper into infeasibility.

Era handling: PR #57 (merged 2026-07-06) added the energy-rank phi init, the
``nucleation_rate`` reaction term and the ``backoff_floor``. The run-era
behaviour is reproduced *bit-identically on current code* by assigning the
old binary-indicator phi through the public ``opt.phi`` (exactly how a resume
seeds state) and setting ``nucleation_rate=0`` / ``backoff_floor=0``; with
``nucleation_rate=0`` the velocity term reduces to the pre-#57 one. A
"post-#57 defaults" case (rank init, rate 0.5, floor 0.25) shows what the fix
changed and that the prune leak itself is still present.

It also instruments the two falsifiable side hypotheses on the run-era code:

* H1 (no nucleation): the run-era binary init plateau sits at +/-1 (the
  +/-band_width clamp is a no-op on the scatter of a +/-1 field). Away from a
  void interface the only downward forces on phi are the uniform -tau shift
  (which equilibrates against dt*Vn at the interface) and Laplacian smoothing
  diffusing the interface deficit inward at ~sqrt(passes*iters) layers, so
  hole "nucleation" is diffusion-limited interface creep plus a slow uniform
  plateau drift -- never a targeted interior hole.
* H2 (normalisation squash): Vn is normalised by its global max; with an
  energy peak ~50x the bulk (load introduction), the bulk velocity is ~0.02
  and the evolution is dominated by tau + smoothing, not the sensitivity.

Run:  python docs/levelset_stuck_repro.py
"""
from __future__ import annotations

import numpy as np

from oropt.beso import gate_target_vf
from oropt.config import LevelSet as LevelSetCfg, ManufacturingOpts
from oropt.levelset import LevelSet
from oropt.manufacturing import apply_manufacturing
from oropt.mesh import Mesh

RNG = np.random.default_rng(42)

# Grid: part = 40x8x8 unit cubes, growth slab on top = 40x4x8 (starts void),
# each cube split into 6 Kuhn tets sharing the main diagonal (node-conformal).
NX, NY, NZ = 40, 12, 8
PART_Y = 8.0                       # part below, growth slab above

# Iterations: 2 feasible (damped, like real iters 0-1), then pinned barely
# infeasible (like real iters 2-7, sigma ~292.6/292 -> violation ~1.002).
N_FEASIBLE, N_TOTAL = 2, 14
V_FEASIBLE, V_INFEASIBLE = 0.955, 1.002


def kuhn_mesh(nx: int, ny: int, nz: int) -> Mesh:
    """Regular grid of unit cubes, each split into the 6 Kuhn tets around the
    (0,0,0)-(1,1,1) diagonal -- conformal across cube faces."""
    xs, ys, zs = np.arange(nx + 1), np.arange(ny + 1), np.arange(nz + 1)
    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    xyz = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1).astype(float)

    def nid(i, j, k):
        return (i * (ny + 1) + j) * (nz + 1) + k

    # cube corners indexed by bits (x, y, z); 6 tets all containing edge 0-7
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
    pts = xyz[conn]                                    # (N,4,3)
    centroids = pts.mean(axis=1)
    a, b, c, d = pts[:, 0], pts[:, 1], pts[:, 2], pts[:, 3]
    volumes = np.abs(np.einsum("ij,ij->i", a - d,
                               np.cross(b - d, c - d))) / 6.0
    return Mesh(centroids=centroids, volumes=volumes, conn_rows=conn,
                n_nodes=xyz.shape[0], design_node_min=0)


def energy_field(mesh: Mesh) -> np.ndarray:
    """Static strain-energy-density stand-in: a load-introduction peak ~50x the
    bulk (mimics the run, where the global max sits in the stress-excluded
    load region and squashes the normalised bulk velocity), plus mild noise
    (real filtered fields are not smooth at element granularity)."""
    c = mesh.centroids
    peak = np.array([float(NX) - 1.0, PART_Y / 2.0, NZ / 2.0])
    d2 = np.sum((c - peak) ** 2, axis=1)
    base = 0.02 + np.exp(-d2 / 8.0)
    return base * (1.0 + 0.05 * RNG.standard_normal(c.shape[0]))


def legacy_init_phi(opt: LevelSet, alive: np.ndarray) -> np.ndarray:
    """The pre-PR#57 ``_init_phi``: smoothed signed indicator of the alive set
    (+1 inside / -1 outside). Assigned through the public ``opt.phi`` exactly
    like a resume seeds state -- no production code is patched."""
    ind = np.where(np.asarray(alive, dtype=bool), 1.0, -1.0)
    phi = opt._scatter(ind)
    phi = opt._smooth(phi, opt.cfg.smoothing_passes)
    return np.clip(phi, -opt.cfg.band_width, opt.cfg.band_width)


def run_case(min_member_layers: int, resync_phi: bool = False,
             legacy: bool = True, n_iter: int = N_TOTAL,
             verbose: bool = True) -> dict:
    """One driven optimisation.

    legacy=True  -> run-era semantics on current code: binary-indicator phi
                    init, nucleation_rate=0, backoff_floor=0 (bit-equivalent
                    to the 2d3c1ce code the run executed).
    legacy=False -> post-PR#57 defaults: energy-rank init (via update's own
                    lazy _init_phi), nucleation_rate=0.5, backoff_floor=0.25.
    """
    mesh = kuhn_mesh(NX, NY, NZ)
    candidate = mesh.centroids[:, 1] > PART_Y          # growth slab starts void
    part = ~candidate
    protected = part & ((mesh.centroids[:, 0] < 2.0)
                        | (mesh.centroids[:, 0] > NX - 2.0))
    anchor = protected

    cfg = LevelSetCfg(evolution_rate=0.015, filter_radius=1.2,
                      target_volume_fraction=0.3, backoff_gain=1.0,
                      backoff_cap=4.0, damping_threshold=0.9,
                      dt=1.0, smoothing_passes=3, band_width=3.0,
                      nucleation_rate=(0.0 if legacy else 0.5),
                      backoff_floor=(0.0 if legacy else 0.25))
    mfg = ManufacturingOpts(min_member_layers=min_member_layers)
    opt = LevelSet(mesh, cfg, protected, anchor=anchor)
    W = mesh.filter_matrix(cfg.filter_radius)
    field = energy_field(mesh)

    alive = ~candidate
    vol = mesh.volumes
    V0 = opt.V0
    bulk = part & (mesh.centroids[:, 0] > 10) & (mesh.centroids[:, 0] < 30) \
        & (mesh.centroids[:, 1] < 5.0)                 # >=3 layers from the slab

    # distance from the initial void interface (the slab at y = PART_Y), in
    # element layers -- for the diffusion-creep (H1) and removal-location
    # measurements. Depth 0 = the part layer touching the slab.
    depth = np.maximum(PART_Y - mesh.centroids[:, 1], 0.0)
    bands = [(0, 2), (2, 4), (4, 6), (6, 8)]
    removed_by_band = np.zeros(len(bands) + 1, dtype=int)  # +1: in-slab removals

    rows = []
    if verbose:
        print(f"\n=== case: min_member_layers={min_member_layers}"
              f" resync_phi={resync_phi} "
              f"{'run-era (legacy init, rate 0, floor 0)' if legacy else 'post-#57 defaults (rank init, rate 0.5, floor 0.25)'} ===")
        print(f"{'it':>3} {'target_vf':>10} {'vf_actual':>10} {'leak':>9} "
              f"{'tau':>8} {'gapTau':>8} {'dThresh':>8} {'dConnU':>8} "
              f"{'dOpen':>8} {'dConn2':>8} {'phantom':>8} {'bulk_em':>8}")

    for it in range(n_iter):
        feasible = it < N_FEASIBLE
        violation = V_FEASIBLE if feasible else V_INFEASIBLE
        vf = opt.volume_fraction(alive)
        target_vf = gate_target_vf(cfg, vf, feasible, violation)

        # the loop's sensitivity: dead elements contribute 0, then filter
        sens = W @ (field * alive)

        if legacy and (opt.phi is None or resync_phi):
            opt.phi = legacy_init_phi(opt, alive)      # run-era init / re-sync
        elif resync_phi:
            opt.phi = None                             # rank re-init from mask

        # ---- replicate update() internals for instrumentation --------------
        phi = opt._init_phi(alive, sens) if opt.phi is None else opt.phi
        Vn = opt._scatter(sens)
        scale = float(np.abs(Vn).max())
        if scale > 0:
            Vn = Vn / scale
        vel = Vn - cfg.nucleation_rate * (1.0 - Vn)
        phi_base = opt._smooth(phi + cfg.dt * vel, cfg.smoothing_passes)
        target_V = target_vf * V0
        protected_V = float(vol[protected].sum())
        budget = target_V - protected_V
        tau = opt._solve_tau(phi_base, budget)
        achieved = opt._removable_vol_at(phi_base, tau)
        phi_new = np.clip(phi_base - tau, -cfg.band_width, cfg.band_width)
        m_thresh = opt.elements_alive(phi_new)
        m_conn = mesh.keep_connected(m_thresh, opt.anchor)

        # phantom volume the *previous* prune left in phi (alive in phi, dead
        # in the actual mask) -- what the bisection budget gets charged for
        phi_alive_prev = opt.elements_alive(phi) if opt.phi is not None \
            else alive | protected
        phantom_prev = float(vol[phi_alive_prev & ~alive & ~protected].sum())

        # ---- the real code path, asserted identical ------------------------
        ret = opt.update(alive, sens, target_vf)
        assert np.array_equal(ret, m_conn), "instrumented replica diverged"
        assert np.allclose(opt.phi, phi_new), "instrumented phi diverged"

        m_open = apply_manufacturing(ret, mesh, mfg, protected,
                                     sensitivity=sens)
        m_final = mesh.keep_connected(m_open, anchor)

        vf_new = opt.volume_fraction(m_final)
        removed_mask = alive & ~m_final
        removed_by_band[-1] += int((removed_mask & candidate).sum())
        for bi, (lo_b, hi_b) in enumerate(bands):
            removed_by_band[bi] += int(
                (removed_mask & part & (depth >= lo_b) & (depth < hi_b)).sum())
        em_now = opt._elem_mean(opt.phi)
        band_min = [float(em_now[part & (depth >= lo_b) & (depth < hi_b)].min())
                    for lo_b, hi_b in bands]
        row = dict(
            it=it, feasible=feasible, target_vf=target_vf, vf=vf_new,
            leak=vf_new - target_vf, tau=tau,
            gap_tau=(budget - achieved) / V0,          # _solve_tau floor gap
            d_thresh=(vol[m_thresh].sum() - target_V) / V0,
            d_conn_upd=(vol[m_thresh].sum() - vol[m_conn].sum()) / V0,
            d_open=(vol[m_conn].sum() - vol[m_open].sum()) / V0,
            d_conn2=(vol[m_open].sum() - vol[m_final].sum()) / V0,
            phantom=phantom_prev / V0,
            grown=int((m_final & candidate).sum()),
            removed=int(removed_mask.sum()),
            bulk_em_min=float(em_now[bulk].min()),
            band_min=band_min,
        )
        rows.append(row)
        if verbose:
            print(f"{it:>3} {target_vf:>10.6f} {vf_new:>10.6f} "
                  f"{row['leak']:>9.5f} {tau:>8.4f} {row['gap_tau']:>8.5f} "
                  f"{row['d_thresh']:>8.5f} {row['d_conn_upd']:>8.5f} "
                  f"{row['d_open']:>8.5f} {row['d_conn2']:>8.5f} "
                  f"{row['phantom']:>8.5f} {row['bulk_em_min']:>8.4f}")
        alive = m_final

    # H1 (run-era): the initial plateau is binary at +/-1, NOT at +/-band_width
    opt2 = LevelSet(mesh, cfg, protected, anchor=anchor)
    phi0 = legacy_init_phi(opt2, ~candidate)
    em0 = opt2._elem_mean(phi0)
    interior0 = float(np.mean(np.abs(em0[bulk] - 1.0) < 1e-12))
    at_clamp0 = float(np.mean(np.abs(phi0) >= cfg.band_width - 1e-12))

    return dict(rows=rows, interior_plateau_at_1=interior0,
                init_at_clamp=at_clamp0, removed_by_band=removed_by_band,
                bands=bands, phi=opt.phi, opt=opt, mesh=mesh)


def main() -> None:
    full = run_case(min_member_layers=1)
    no_mfg = run_case(min_member_layers=0, n_iter=50, verbose=False)
    resync = run_case(min_member_layers=1, resync_phi=True)
    post57 = run_case(min_member_layers=1, legacy=False)

    infe = [r for r in full["rows"] if not r["feasible"]]
    print("\n--- H3 leak decomposition over the pinned-infeasible iterations "
          "(run-era semantics) ---")
    leak = np.mean([r["leak"] for r in infe])
    d_open = np.mean([r["d_open"] for r in infe])
    d_conn = np.mean([r["d_conn_upd"] + r["d_conn2"] for r in infe])
    gap = np.mean([r["gap_tau"] for r in infe])
    print(f"full chain     : mean leak/iter = {leak:+.5f} of V0 "
          f"(open {d_open:.5f}, keep_connected {d_conn:.5f}, "
          f"tau floor {gap:.5f})")
    infe_b = [r for r in no_mfg["rows"] if not r["feasible"]][:len(infe)]
    print(f"no morph_open  : mean leak/iter = "
          f"{np.mean([r['leak'] for r in infe_b]):+.5f} of V0 "
          "(the _solve_tau floor gap only)")
    infe_c = [r for r in resync["rows"] if not r["feasible"]]
    print(f"phi re-synced  : mean leak/iter = "
          f"{np.mean([r['leak'] for r in infe_c]):+.5f} of V0, decaying "
          f"{infe_c[0]['leak']:+.5f} -> {infe_c[-1]['leak']:+.5f} "
          "(without the desync ratchet the fringe anneals)")
    print("desync ratchet : phantom(it) == dOpen(it-1) -- the bisection budget "
          "is charged for pruned-but-phi-alive volume every iteration")

    infe_d = [r for r in post57["rows"] if not r["feasible"]]
    d_open_d = np.mean([r["d_open"] for r in infe_d])
    net_d = post57["rows"][-1]["vf"] - post57["rows"][N_FEASIBLE - 1]["vf"]
    print(f"\n--- post-PR#57 defaults (rank init, nucleation 0.5, backoff "
          f"floor 0.25) ---")
    print(f"morph_open still leaks {d_open_d:.5f} of V0 per infeasible "
          f"iteration; floored back-off requests "
          f"{0.015 * 0.25:.5f}*vf; net vf over the infeasible stretch: "
          f"{net_d:+.5f} (the leak persists; on this synthetic case its "
          "fringe is proportionally larger than the floor step)")

    print("\n--- removal location (run-era full chain), distance from the "
          "slab interface in element layers ---")
    rb, bands = full["removed_by_band"], full["bands"]
    print(f"in-slab (grown-then-shaved candidates): {rb[-1]}")
    for (lo, hi), n in zip(bands, rb[:-1]):
        print(f"part, {lo}-{hi} layers below the interface: {n}")

    print(f"\n--- H1 (run-era): plateau + diffusion-limited creep (no-mfg "
          f"case, {len(no_mfg['rows'])} iterations) ---")
    print(f"initial nodal phi at the +/-band_width clamp: "
          f"{full['init_at_clamp']:.3f} of nodes; bulk elements with "
          f"mean-phi == +1.0 exactly: {full['interior_plateau_at_1']:.3f} "
          "(the plateau is binary at +/-1, the clamp in the run-era init is "
          "a no-op)")
    for bi, (lo, hi) in enumerate(no_mfg["bands"]):
        mins = [r["band_min"][bi] for r in no_mfg["rows"]]
        crossed = next((i for i, v in enumerate(mins) if v < 0.0), None)
        state = (f"first element crosses phi=0 at iteration {crossed}"
                 if crossed is not None else "never crosses phi=0")
        print(f"part {lo}-{hi} layers below the interface: min mean-phi "
              f"{mins[0]:+.3f} -> {mins[-1]:+.3f}, {state}")

    # ---- the assertions that make this a demonstration, not a plot ---------
    for r in infe:
        assert r["target_vf"] >= r["vf"] + r["leak"] - 1e-12  # gate asked to grow
    assert all(r["leak"] < 0 for r in infe), \
        "expected the mask to shrink every pinned-infeasible iteration"
    assert abs(np.mean([r["d_open"] + r["d_conn_upd"] + r["d_conn2"]
                        for r in infe]) + leak) < 5e-4, \
        "leak should be explained by the removal-only post-passes"
    assert abs(np.mean([r['leak'] for r in infe_b])) < abs(leak) / 20, \
        "without morph_open the leak should essentially vanish"
    for r in infe[1:]:
        assert abs(r["phantom"]) > abs(r["gap_tau"]), \
            "phi should carry phantom volume from the previous prune"
    deep = [r["band_min"][-1] for r in no_mfg["rows"]]
    assert min(deep) > 0.0, \
        ">=6 layers from the interface nothing nucleates within 50 iterations"
    assert d_open_d > abs(gap) * 10, \
        "post-#57 the morph_open prune leak must still be present"
    print("\nall mechanism assertions hold")


if __name__ == "__main__":
    main()
