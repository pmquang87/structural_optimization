"""Real-scale verification of the level-set stall on the salvaged
elevator-linkage run artifacts (2026-07-05/06).

Complements docs/levelset_stuck_repro.py (small synthetic case, exact
mechanism decomposition) by replaying ONE optimiser iteration on the actual
2,272,868-element design mesh from the salvaged checkpoint, and by measuring
the sensitivity-normalisation squash (H2) on the run's own filtered
sensitivity field.

Needs the salvage folder copied out of the run dir before the follow-up BESO
run overwrote it (checkpoint.npz, queue_configs/elevator_linkage_dispfix.yaml,
iter_0007/, and implicit_elevator-linkage_pull_0000.rad.pre-brick-fix.bak):

    python docs/levelset_stuck_replay_real.py <salvage_dir>

Notes on fidelity:

* The run-era root starter was OVERWRITTEN at 2026-07-06 09:09 by a fresh
  growth-mesh PREPARE for the follow-up BESO run (2,285,778 design elements
  vs the run's 2,272,868 -- same part, regenerated expansion mesh). The
  run-era design mesh is recovered from the .pre-brick-fix.bak instead: the
  brick fix only deleted degenerate non-design /BRICK cards, so the design
  part (/TETRA4/60000000), nodes and groups are the run's own.
* checkpoint.npz stores only (iteration, alive_mask, sens_prev) -- the nodal
  phi field is NOT checkpointed, so the replay initialises phi from the alive
  mask exactly like a resumed run would. The replayed iteration is therefore
  literally "what iteration 8 would have done after a resume", not a
  bit-perfect continuation; the leak mechanism it demonstrates is the same,
  and the phi loss on resume is itself one of the findings.
* The run executed pre-PR#57 code (2d3c1ce). On current code the run-era
  semantics are reproduced bit-identically by seeding the old binary
  indicator phi through the public ``opt.phi`` (exactly how a resume seeds
  state) and zeroing ``nucleation_rate`` / ``backoff_floor``. A second leg
  replays the same iteration with the post-#57 defaults (energy-rank init,
  nucleation 0.5, floor 0.25) to measure what the fix changed on the real
  mesh -- and that the morph_open prune leak is still present.
* The LevelSet is built with filter_radius=0: the spatial filter matrix W is
  only used by filter_history (we feed the checkpointed, already-filtered
  sens_prev), and skipping it avoids a pointless multi-minute KD-tree pass.
"""
from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from oropt.beso import gate_target_vf
from oropt.config import Config
from oropt.deck import Deck
from oropt.levelset import LevelSet
from oropt.loop import (collect_protect_nodes, growth_candidate_mask,
                        resolve_growth_boxes, stress_exclude_mask)
from oropt.manufacturing import apply_manufacturing
from oropt.mesh import Mesh

# From history.csv iteration 7 (the last level-set iteration that solved):
# sigma_max 293.666 vs sigma_allow 292.0 -> the violation the gate reacts to.
SIGMA_MAX_IT7, SIGMA_ALLOW = 293.666, 292.0


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pct(x: np.ndarray, q: float) -> float:
    return float(np.percentile(x, q))


def removed_components(mesh: Mesh, removed: np.ndarray) -> list[int]:
    """Sizes of the connected components of the removed element set (via
    shared nodes), largest first -- a single dominant component means one
    contiguous region was amputated, not a diffuse fringe shave."""
    from scipy.sparse.csgraph import connected_components
    idx = np.flatnonzero(removed)
    if idx.size == 0:
        return []
    inc = mesh._incidence(idx)
    _, labels = connected_components(inc @ inc.T, directed=False)
    return sorted(np.bincount(labels).tolist(), reverse=True)


def main(salvage: Path) -> None:
    cfg = Config.from_yaml(salvage / "queue_configs" /
                           "elevator_linkage_dispfix.yaml")
    m = cfg.model
    oc = cfg.active_opts()
    assert cfg.optimizer_name() == "levelset"

    log("loading run-era deck (.pre-brick-fix.bak, 174 MB) ...")
    deck = Deck.load(
        salvage / "implicit_elevator-linkage_pull_0000.rad.pre-brick-fix.bak",
        m.design_part_id, m.design_node_min)
    mesh = Mesh.from_deck(deck)
    log(f"deck: {deck.n_design_elements} design elements, "
        f"{mesh.n_nodes} nodes")
    assert deck.n_design_elements == 2272868, \
        "not the run-era design mesh (expected 2,272,868 design elements)"

    # ---- reconstruct the run's masks exactly as loop.run_optimization does --
    frozen = collect_protect_nodes(deck, m, include_bc=oc.protect_bc_nodes)
    protected = mesh.protected_mask(deck, frozen,
                                    contact_dist=oc.contact_protect_dist,
                                    layers=oc.protect_layers)
    anchor_nodes = collect_protect_nodes(deck, m, include_bc=True)
    anchor = mesh.protected_mask(deck, anchor_nodes,
                                 contact_dist=oc.contact_protect_dist,
                                 layers=oc.protect_layers)
    candidate = growth_candidate_mask(deck, mesh, m, log=log)
    protected = protected & ~candidate
    excluded = stress_exclude_mask(deck, mesh, m)
    log(f"candidates: {int(candidate.sum())}  protected: "
        f"{int(protected.sum())} ({100 * protected.mean():.1f}%)  "
        f"stress-excluded: {int(excluded.sum())}")

    ck = np.load(salvage / "checkpoint.npz")
    alive = ck["alive_mask"].astype(bool)
    sens = ck["sens_prev"].astype(float)
    log(f"checkpoint: iteration={int(ck['iteration'])}, "
        f"alive={int(alive.sum())}")
    assert alive.size == deck.n_design_elements

    in_box = mesh.in_boxes_mask(resolve_growth_boxes(deck, m.growth_boxes))
    box_tree = cKDTree(mesh.centroids[in_box])

    vol = mesh.volumes
    V0 = float(vol.sum())

    def classify_removed(before: np.ndarray, after: np.ndarray,
                         label: str) -> None:
        removed = before & ~after
        out_removed = removed & ~in_box
        n_near = n_far = 0
        dmax = 0.0
        if out_removed.any():
            d, _ = box_tree.query(mesh.centroids[out_removed], k=1)
            n_near = int((d <= 1.0).sum())
            n_far = int((d > 1.0).sum())
            dmax = float(d.max())
        comp = removed_components(mesh, removed)
        print(f"{label}: removed {int(removed.sum())} "
              f"(volume {float(vol[removed].sum()) / V0:.5f} of V0, mean elem "
              f"vol {float(vol[removed].mean()):.4f} vs design-space mean "
              f"{float(vol.mean()):.4f}; "
              f"{int((removed & in_box).sum())} inside a growth region, "
              f"{n_near} within 1 mm, {n_far} beyond, max {dmax:.2f} mm); "
              f"removed & protected: {int((removed & protected).sum())}; "
              f"components (largest 5): {comp[:5]}")
        # Energy ranking + contact with the survivors. A keep_connected island
        # shares no node with the surviving alive set, so a removed element
        # touching a survivor died by the threshold (or the open), not as an
        # island; low percentiles = the lowest-energy material went first.
        s_before = np.sort(sens[before])
        pctl = np.searchsorted(s_before, sens[removed]) / max(s_before.size, 1)
        node_alive = np.zeros(mesh.n_nodes, dtype=bool)
        node_alive[np.unique(mesh.conn_rows[after])] = True
        touches = node_alive[mesh.conn_rows[removed]].any(axis=1)
        print(f"    removed-element sens percentile: median "
              f"{np.median(pctl) * 100:.1f}, in bottom 25% of energy: "
              f"{float(np.mean(pctl < 0.25)) * 100:.0f}%; sharing a node with "
              f"a surviving element (=> not an island): "
              f"{float(touches.mean()) * 100:.1f}%")
        idx = np.flatnonzero(removed)
        from scipy.sparse.csgraph import connected_components
        inc = mesh._incidence(idx)
        _, labels = connected_components(inc @ inc.T, directed=False)
        sizes = np.bincount(labels)
        for lbl in np.argsort(sizes)[::-1][:3]:
            sel = idx[labels == lbl]
            c = mesh.centroids[sel]
            d, _ = box_tree.query(c, k=1)
            sp = np.searchsorted(s_before, sens[sel]) / max(s_before.size, 1)
            print(f"    component {sizes[lbl]:>6} elems: extent "
                  f"{np.round(c.max(axis=0) - c.min(axis=0), 1)} mm, "
                  f"dist-to-box {d.min():.1f}-{d.max():.1f} mm, "
                  f"median sens pct {np.median(sp) * 100:.1f}")

    # ---- the run's OWN final update: diff iter-7 deck vs checkpoint mask ---
    # history.csv row 7 solved a_7 (2,036,666 alive); the checkpoint holds the
    # a_8 the update+prune chain produced from it (never solved -- the run was
    # stopped). The archived iter_0007 deck lists exactly the a_7 design
    # elements, so the diff reconstructs what the last update actually did.
    log("loading archived iter_0007 deck for the a_7 alive set ...")
    deck7 = Deck.load(salvage / "iter_0007" / "implicit_elevator-linkage_pull"
                      / "implicit_elevator-linkage_pull_0000.rad",
                      m.design_part_id, m.design_node_min)
    a7 = np.isin(deck.elem_ids, deck7.elem_ids, assume_unique=True)
    print(f"\n--- the run's own last update (a_7 -> a_8, from the archives) ---")
    print(f"a_7 alive {int(a7.sum())} (history row 7: 2036666), "
          f"a_8 alive {int(alive.sum())}, "
          f"net {int(alive.sum()) - int(a7.sum()):+d}, "
          f"grown {int((alive & ~a7).sum())}")
    print(f"vf(a_7) = {float(vol[a7].sum()) / V0:.6f} -> "
          f"vf(a_8) = {float(vol[alive].sum()) / V0:.6f}")
    classify_removed(a7, alive, "a_7 -> a_8")

    # ---- H2: normalisation squash on the run's own sensitivity -------------
    s_alive = sens[alive]
    smax = float(s_alive.max())
    imax = int(np.flatnonzero(alive)[int(np.argmax(s_alive))])
    exc_tree = cKDTree(mesh.centroids[excluded])
    d_exc, _ = exc_tree.query(mesh.centroids[imax], k=1)
    top = sens >= 0.999 * smax
    print("\n--- H2: sensitivity distribution over alive elements "
          "(checkpointed sens_prev, filtered+history-blended) ---")
    print(f"max {smax:.6g} | p99.9 {pct(s_alive, 99.9):.6g} | "
          f"p99 {pct(s_alive, 99):.6g} | p95 {pct(s_alive, 95):.6g} | "
          f"median {pct(s_alive, 50):.6g}")
    print(f"max / p99 = {smax / pct(s_alive, 99):.1f}x ; "
          f"max / median = {smax / pct(s_alive, 50):.1f}x")
    print(f"argmax element centroid {mesh.centroids[imax].round(2)}, "
          f"distance to nearest stress-excluded element "
          f"{float(d_exc):.3f} mm; excluded? {bool(excluded[imax])}")
    print(f"elements within 0.1% of max: {int(top.sum())}, of which "
          f"stress-excluded: {int((top & excluded).sum())}, "
          f"protected: {int((top & protected).sum())}")
    frac01 = float(np.mean(s_alive < 0.01 * smax))
    frac05 = float(np.mean(s_alive < 0.05 * smax))
    print(f"alive elements with sens < 1% of max: {100 * frac01:.1f}% ; "
          f"< 5% of max: {100 * frac05:.1f}%")

    # ---- one replayed iteration (resume + run-era semantics) ----------------
    # nucleation_rate=0 / backoff_floor=0 reproduce the pre-PR#57 code the run
    # executed; the binary-indicator phi below reproduces its _init_phi.
    lcfg = dataclasses.replace(cfg.levelset, filter_radius=0.0,
                               nucleation_rate=0.0, backoff_floor=0.0)
    log("building LevelSet (node graph + smoothing operator) ...")
    opt = LevelSet(mesh, lcfg, protected, anchor=anchor)
    assert abs(opt.V0 - V0) < 1e-6 * V0
    vf = opt.volume_fraction(alive)
    violation = SIGMA_MAX_IT7 / SIGMA_ALLOW
    target_vf = gate_target_vf(lcfg, vf, False, violation)
    log(f"V0={V0:.1f}  vf={vf:.6f}  violation={violation:.6f}  "
        f"target_vf={target_vf:.6f} (grow by {(target_vf / vf - 1):.2e})")

    log("phi = smoothed binary indicator of the alive mask -- run-era resume "
        "semantics ...")
    ind = np.where(alive, 1.0, -1.0)
    phi = np.clip(opt._smooth(opt._scatter(ind), lcfg.smoothing_passes),
                  -lcfg.band_width, lcfg.band_width)
    em0 = opt._elem_mean(phi)
    implied = (em0 >= 0.0) | protected
    resume_flip = implied ^ (alive | protected)
    print(f"\n--- resume perturbation: elements_alive(run-era init phi) vs "
          f"alive: {int(resume_flip.sum())} elements flip "
          f"({float(vol[resume_flip].sum()) / V0:.5f} of V0) ---")

    # nodal velocity + squash, as update() computes it
    Vn = opt._scatter(sens)
    scale = float(np.abs(Vn).max())
    Vn_n = Vn / scale if scale > 0 else Vn
    print(f"normalised nodal velocity Vn: p50 {pct(Vn_n, 50):.4g}, "
          f"p99 {pct(Vn_n, 99):.4g}, max 1.0 -> bulk phi push per iteration "
          f"~{pct(Vn_n, 50):.1e} of the [-3, 3] band")

    log("evolving phi + bisection ...")
    opt.phi = phi
    vel = Vn_n - lcfg.nucleation_rate * (1.0 - Vn_n)   # rate 0 -> run-era Vn
    phi_base = opt._smooth(phi + lcfg.dt * vel, lcfg.smoothing_passes)
    target_V = target_vf * V0
    protected_V = float(vol[protected].sum())
    budget = target_V - protected_V
    tau = opt._solve_tau(phi_base, budget)
    achieved = opt._removable_vol_at(phi_base, tau)
    phi_new = np.clip(phi_base - tau, -lcfg.band_width, lcfg.band_width)
    m_thresh = opt.elements_alive(phi_new)
    log("keep_connected (inside update) ...")
    m_conn = mesh.keep_connected(m_thresh, opt.anchor)

    # the real code path, asserted identical to the instrumented replica
    opt.phi = phi
    ret = opt.update(alive, sens, target_vf)
    assert np.array_equal(ret, m_conn), "instrumented replica diverged"

    log("apply_manufacturing (min_member_layers=1) ...")
    m_open = apply_manufacturing(ret, mesh, cfg.manufacturing, protected,
                                 sensitivity=sens)
    log("final keep_connected ...")
    m_final = mesh.keep_connected(m_open, anchor)

    vf_new = float(vol[m_final].sum()) / V0
    print("\n--- replayed iteration 8 (resume semantics) volume budget, "
          "as fractions of V0 ---")
    print(f"tau = {tau:.5f}  bisection floor gap (budget - achieved) = "
          f"{(budget - achieved) / V0:.2e}")
    print(f"thresholded phi mask:            {float(vol[m_thresh].sum()) / V0:.6f}")
    print(f"after update keep_connected:     {float(vol[m_conn].sum()) / V0:.6f} "
          f"(drop {float(vol[m_thresh].sum() - vol[m_conn].sum()) / V0:.6f})")
    print(f"after _morph_open:               {float(vol[m_open].sum()) / V0:.6f} "
          f"(drop {float(vol[m_conn].sum() - vol[m_open].sum()) / V0:.6f})")
    print(f"after final keep_connected:      {vf_new:.6f} "
          f"(drop {float(vol[m_open].sum() - vol[m_final].sum()) / V0:.6f})")
    print(f"target_vf {target_vf:.6f} -> actual {vf_new:.6f} "
          f"(leak {vf_new - target_vf:+.6f}); real run leaked -0.0014 to "
          "-0.0016 of V0 per iteration over iters 2-7")

    grown = m_final & ~alive
    print(f"grown {int(grown.sum())}")
    classify_removed(alive, m_final, "replayed a_8 -> a_9")
    print("(run-measured over iters 0-6: 29105 inside, 628 within 1 mm, "
          "36 beyond (max 1.6 mm); 659 grown)")

    # ---- the same iteration under post-PR#57 defaults ------------------------
    # Energy-rank init (update's own lazy _init_phi), nucleation_rate 0.5,
    # backoff_floor 0.25 -- what a fresh run on current master would do from
    # this state. Measures what #57 changed on the real mesh, and that the
    # morph_open prune leak is still present.
    lcfg57 = dataclasses.replace(cfg.levelset, filter_radius=0.0)
    assert lcfg57.nucleation_rate > 0.0 and lcfg57.backoff_floor > 0.0
    target57 = gate_target_vf(lcfg57, vf, False, violation)
    log(f"post-#57 leg: target_vf={target57:.6f} "
        f"(floored back-off: grow by {(target57 / vf - 1):.2e}) ...")
    opt57 = LevelSet(mesh, lcfg57, protected, anchor=anchor)
    ret57 = opt57.update(alive, sens, target57)     # lazy rank init inside
    log("post-#57 apply_manufacturing + keep_connected ...")
    m57 = mesh.keep_connected(
        apply_manufacturing(ret57, mesh, cfg.manufacturing, protected,
                            sensitivity=sens), anchor)
    vf57 = float(vol[m57].sum()) / V0
    print(f"\n--- the same iteration, post-PR#57 defaults (rank init, "
          f"nucleation 0.5, floor 0.25) ---")
    print(f"target_vf {target57:.6f} -> actual {vf57:.6f} "
          f"(prune leak {vf57 - float(vol[ret57].sum()) / V0:+.6f}; net "
          f"{vf57 - vf:+.6f} vs run-era net {vf_new - vf:+.6f})")
    print(f"grown {int((m57 & ~alive).sum())}")
    classify_removed(alive, m57, "post-#57 a_8 -> a_9")


if __name__ == "__main__":
    default = (r"E:\foxcore_data\_MITEB\openradioss\implicit_elevator-linkage"
               r"\implicit_6kN_elevator-linkage_neutral-pull__BC-A_"
               r"Erpro-Wie-Gebaut_foxcore-rund\opti_run1"
               r"\_levelset_salvage_20260706")
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(default))
