"""SAIP — Sequential Approximate Integer Programming (a config-selectable
alternative to BESO/TOBS that picks each iteration's element flips by solving
the linearised binary subproblem *analytically* with the family's canonical
relaxation, instead of BESO's heuristic threshold or TOBS's general ILP solver).

Liang & Cheng, "Topology optimization via sequential integer programming and
canonical relaxation algorithm", *CMAME* 348:1005-1036 (2019); the 128-line
code, *SMO* 61:411-431 (2020); consolidating review, *Engineering Optimization*
(2025). The conservative SCIP variant: Sun, Cheng, Zhang & Liang, *Acta
Mechanica Sinica* 40 (2024). Survey and fit assessment for oropt:
``docs/topology_sota_2026.md``.

The design variables are the same binary alive/void element flags as BESO
(``x_e in {0,1}``, 1 == alive). OpenRadioss exposes no design gradients; the
basic SAIP sensitivity for compliance-type objectives is the element strain
energy ``u_e^T k_e u_e`` — exactly the per-element ``/ANIM/ELEM/ENER`` field,
already spatially filtered and history-averaged by :func:`filter_history`
(shared with BESO). Each iteration solves the linearised subproblem

* **objective** — maximise ``sum_e s_e * x_e`` (keep high-, drop
  low-sensitivity material);
* **volume budget** — ``sum_e vol_e * x_e <= target_vf * V0`` (the same
  ``evolution_rate``/``target_volume_fraction`` gate as BESO);
* **move limit** — at most ``K = flip_limit * N`` elements flip per iteration
  (the trust region that keeps successive linearisations valid);

by *canonical relaxation*: dualise the volume constraint with a single
multiplier ``lambda``; each element's optimal flip then separates into a
per-element sign test on the reduced gain ``g_e = s_e - lambda * vol_e``
(a solid flips void when ``g_e < 0``, a void flips solid when ``g_e > 0``),
the move limit is enforced by keeping only the ``K`` largest reduced gains,
and ``lambda`` is bisected so the resulting kept volume meets the budget —
closed-form per element, no MILP, microseconds at any mesh size. Under a
*binding* move limit the removal priority ``lambda*vol - s`` is volume-
weighted, i.e. the capped step removes the most volume per flip.

``oscillation_damping`` adds a lightweight conservatism in the *spirit* of
SCIP's conservative (moving-asymptote) subproblems — it is NOT the paper's
reciprocal-variable formulation: an element whose state changed anywhere in
the last loop iteration (flipped by the previous update, or pruned back by
the loop's post-passes) has its flip *priority* scaled down, so it ranks
behind fresh candidates for immediately flipping back and the add/remove
ping-pong that plain successive linearisation is prone to decays instead of
cycling. Candidacy (the sign test) is unchanged — only the ranking within the
flip cap. The flip memory is one iteration deep and intentionally not
checkpointed: a resume merely loses the damping for its first update, exactly
like the level-set/HCA fields re-initialising on an optimiser switch.

Protected elements are pinned (never flipped; forced alive) and disconnected
islands are dropped via :meth:`Mesh.keep_connected` — exactly like BESO.

Mirrors the :class:`~oropt.beso.Beso` interface so the loop can drive either.
"""
from __future__ import annotations

import math

import numpy as np

from .beso import blend_history, gate_target_vf, map_sensitivity
from .config import SaipOpts as SaipCfg
from .mesh import Mesh
from .results import Results


class Saip:
    def __init__(self, mesh: Mesh, cfg: SaipCfg, protected_mask: np.ndarray,
                 anchor: np.ndarray | None = None):
        self.mesh = mesh
        self.cfg = cfg
        self.protected = np.asarray(protected_mask, dtype=bool)
        # Elements that anchor connectivity for island-dropping (defaults to the
        # protected set, decoupled so the BC/load region keeps anchoring even when
        # its elements are deletable). Mirrors Beso/LevelSet/Tobs/Hca.
        self.anchor = (np.asarray(anchor, dtype=bool)
                       if anchor is not None else self.protected)
        self.vol = mesh.volumes
        self.V0 = float(self.vol.sum())
        self._W = mesh.filter_matrix(cfg.filter_radius)
        # One-iteration flip memory driving the oscillation damping: the last
        # update's input and returned masks. An element counts as "just
        # flipped" when its state changed anywhere in the last loop iteration
        # -- by this optimiser's own update (prev_in != prev_out) or by the
        # loop's post-passes pruning the returned mask (prev_out != next
        # input). Comparing successive inputs alone would miss the classic
        # add -> prune ping-pong entirely: both inputs show the element void.
        self._prev_in: np.ndarray | None = None
        self._prev_out: np.ndarray | None = None

    # ---- volume bookkeeping (mirrors Beso) --------------------------------
    def volume_fraction(self, alive_mask: np.ndarray) -> float:
        return float(self.vol[alive_mask].sum() / self.V0)

    # ---- sensitivity (delegates to the shared BESO helpers) ---------------
    def raw_sensitivity(self, results: Results, elem_ids: np.ndarray,
                        alive_mask: np.ndarray) -> np.ndarray:
        return map_sensitivity(results, elem_ids, self.cfg.sensitivity,
                               self.cfg.blend_weight)

    def filter_history(self, raw: np.ndarray,
                       sens_prev: np.ndarray | None) -> np.ndarray:
        return blend_history(self._W, raw, sens_prev, self.cfg.history_weight)

    # ---- target volume & constraint gate (shared with Beso) ---------------
    def next_target_vf(self, current_vf: float, feasible: bool,
                       violation: float | None = None) -> float:
        return gate_target_vf(self.cfg, current_vf, feasible, violation)

    # ---- canonical relaxation ----------------------------------------------
    def _flips_at(self, x: np.ndarray, sens: np.ndarray, damp: np.ndarray,
                  K: int, lam: float) -> np.ndarray:
        """The flip set the dualised subproblem picks at multiplier *lam*.

        Reduced gain of flipping element e in its only allowed direction:
        ``lam*vol - sens`` for a solid (remove), ``sens - lam*vol`` for a void
        (add); positive gain = the flip improves the dual objective. The move
        limit keeps the ``K`` largest *damped* gains (damping re-ranks, never
        re-admits: a non-candidate stays a non-candidate).
        """
        g = sens - lam * self.vol
        gain = np.where(x, -g, g)
        gain[self.protected] = -1.0            # protected never flip
        cand = np.flatnonzero(gain > 0.0)
        flip = np.zeros(x.size, dtype=bool)
        if cand.size == 0:
            return flip
        if cand.size > K:
            pri = gain[cand] * damp[cand]
            cand = cand[np.argpartition(pri, cand.size - K)[cand.size - K:]]
        flip[cand] = True
        return flip

    def _removable_vol_at(self, x: np.ndarray, sens: np.ndarray,
                          damp: np.ndarray, K: int, lam: float) -> float:
        """Volume of *removable* (non-protected) elements alive after applying
        the flips at *lam*. Uses exactly the flip set :meth:`update` applies, so
        the bisection target and the final mask never disagree. Non-increasing
        in *lam*: removals gain and additions lose reduced gain as *lam* grows,
        both as candidates and in the capped ranking."""
        new = x ^ self._flips_at(x, sens, damp, K, lam)
        return float(self.vol[new & ~self.protected].sum())

    def _solve_lambda(self, x: np.ndarray, sens: np.ndarray, damp: np.ndarray,
                      K: int, budget: float) -> float:
        """Smallest multiplier ``lambda`` whose removable kept-volume is
        <= *budget* (i.e. the *largest* volume not exceeding the budget).
        Protected elements are excluded here and forced alive by :meth:`update`.

        ``lambda`` prices volume in sensitivity-per-volume units, so the walk
        spans ``[0, ~2x the max value density]``: at 0 nothing is removed and
        every energised void is added (maximum volume); above the max ``s/vol``
        every removable solid wants out and no void wants in (minimum volume).
        When even that cannot reach the budget the move limit binds — the high
        end is returned and the loop's next target catches up later, exactly
        like TOBS's flip cap.
        """
        rem = ~self.protected
        vd = sens[rem] / np.maximum(self.vol[rem], 1e-300)
        hi = 2.0 * float(vd.max()) + 1.0 if vd.size else 1.0
        lo = 0.0
        if budget <= 0.0:
            return hi                            # target below the protected floor
        if self._removable_vol_at(x, sens, damp, K, lo) <= budget:
            return lo                            # target above reach: grow everything
        if self._removable_vol_at(x, sens, damp, K, hi) > budget:
            return hi                            # move limit binds: max removal
        # invariant: vol(lo) > budget >= vol(hi); keep the hi side so vol <= budget
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if self._removable_vol_at(x, sens, damp, K, mid) > budget:
                lo = mid
            else:
                hi = mid
        return hi

    # ---- alive-set update --------------------------------------------------
    def update(self, alive_mask: np.ndarray, sens: np.ndarray,
               target_vf: float) -> np.ndarray:
        """One SAIP step: bisect the volume multiplier, apply the capped flip
        set of the canonical relaxation, force protected alive, drop islands.

        ``sens`` is the filtered/history-averaged per-element energy. Returns
        the new alive mask, exactly like BESO.
        """
        x = np.asarray(alive_mask, dtype=bool)
        sens = np.asarray(sens, dtype=float)

        # one-iteration flip memory -> SCIP-inspired ranking damp
        if (self._prev_in is not None and self._prev_in.shape == x.shape
                and self._prev_out is not None):
            flipped = (self._prev_in != self._prev_out) | (self._prev_out != x)
        else:
            flipped = np.zeros(x.size, dtype=bool)
        self._prev_in = x.copy()
        damp = np.where(flipped, self.cfg.oscillation_damping, 1.0)

        if float(sens.max() if sens.size else 0.0) <= 0.0:
            # no usable energy signal (failed extraction / all-zero field):
            # every reduced gain would be pure -lam*vol noise, so keep the
            # design unchanged (TOBS's no-flip safety net).
            alive = self.mesh.keep_connected(x | self.protected, self.anchor)
            self._prev_out = alive.copy()
            return alive

        K = max(1, math.floor(self.cfg.flip_limit * x.size))
        budget = target_vf * self.V0 - float(self.vol[self.protected].sum())
        lam = self._solve_lambda(x, sens, damp, K, budget)

        new_alive = x ^ self._flips_at(x, sens, damp, K, lam)
        new_alive |= self.protected              # protected always kept
        alive = self.mesh.keep_connected(new_alive, self.anchor)
        self._prev_out = alive.copy()
        return alive
