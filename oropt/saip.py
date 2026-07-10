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

Two conservatism mechanisms temper the raw successive linearisation:

``oscillation_damping`` — a lightweight *ranking* conservatism: an element
whose state changed anywhere in the last loop iteration (flipped by the
previous update, or pruned back by the loop's post-passes) has its flip
*priority* scaled down, so it ranks behind fresh candidates for immediately
flipping back. Candidacy (the sign test) is unchanged — only the ranking
within the flip cap.

``scip_asymptotes`` (opt-in, default off) — the SCIP conservative subproblem
(Sun, Cheng, Zhang & Liang, *Acta Mech. Sin.* 40, 2024) adapted to the binary
flip. SCIP replaces the linear objective approximation with reciprocal
intervening variables whose conservativeness is set by MMA-style moving
asymptotes. For a binary variable the MMA reciprocal form with lower asymptote
``L_e < 0`` evaluated at its only two points reduces *exactly* (no further
approximation) to one per-element factor ``t_e = -L_e/(1-L_e) in (0, 1]``:
removing a solid is predicted to cost ``s_e/t_e`` (inflated) and adding a void
to gain ``t_e*s_e`` (deflated). The canonical relaxation then runs unchanged on
these conservative coefficients, so — unlike the damping — the *sign test
itself* becomes conservative: a marginal flip of an oscillating element stops
being a candidate at all, and the add/remove ping-pong decays geometrically
instead of cycling. ``t_e`` adapts by the standard MMA asymptote rule mapped to
flip history: tighten (``t_e *= scip_gamma_tight``) when the element's last two
flips reversed direction, relax (``t_e *= scip_gamma_relax``, capped at 1 =
plain linear gain) when it stayed put or moved monotonically, floored at
``scip_t_min``. "Reversed direction" means opposite flips in *consecutive*
iterations (MMA's ``(dx_k)(dx_{k-1}) < 0`` test) — a re-add long after a
removal is a legitimate back-off, not an oscillation. Honest deviations from
the paper: the adaptation trigger is the classic MMA flip-history test, not
the paper's specific update formula; a
single ``t_e`` replaces the full asymptote pair (lossless at the two binary
endpoints, but the paper's continuous inner subproblem iterations are not
reproduced); adaptation happens across outer iterations only — consistent with
the paper's "no additional structural analyses" property, but without its
inner conservativeness check. Both mechanisms can coexist (damping re-ranks,
asymptotes change the gains); prefer one — set ``oscillation_damping: 1.0``
when enabling ``scip_asymptotes``.

The flip memory and the asymptote factors are a few iterations deep and
intentionally not checkpointed: a resume re-initialises ``t_e = 1`` and merely
loses the conservatism for its first update, exactly like the level-set/HCA
fields re-initialising on an optimiser switch.

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
        # SCIP conservative subproblem (opt-in; see module docstring). Knobs are
        # read via getattr so a config predating them keeps today's behaviour.
        self._scip_on = bool(getattr(cfg, "scip_asymptotes", False))
        self._scip_gamma_tight = float(getattr(cfg, "scip_gamma_tight", 0.7))
        self._scip_gamma_relax = float(getattr(cfg, "scip_gamma_relax", 1.2))
        self._scip_t_min = float(getattr(cfg, "scip_t_min", 0.1))
        # Per-element conservatism factor t_e in [t_min, 1] (1 = plain linear
        # gain) and the direction of each element's most recent flip
        # (-1 removed, +1 added, 0 never flipped). Like the flip memory above,
        # intentionally not checkpointed: a resume restarts at t_e = 1.
        self._scip_t: np.ndarray | None = None
        self._scip_dir: np.ndarray | None = None

    # ---- volume bookkeeping (mirrors Beso) --------------------------------
    def volume_fraction(self, alive_mask: np.ndarray) -> float:
        return float(self.vol[alive_mask].sum() / self.V0)

    # ---- sensitivity (delegates to the shared BESO helpers) ---------------
    def raw_sensitivity(self, results: Results, elem_ids: np.ndarray,
                        alive_mask: np.ndarray) -> np.ndarray:
        return map_sensitivity(results, elem_ids, self.cfg.sensitivity,
                               self.cfg.blend_weight,
                               tdsa_nu=getattr(self.cfg, "tdsa_nu", 0.33))

    def filter_history(self, raw: np.ndarray,
                       sens_prev: np.ndarray | None) -> np.ndarray:
        return blend_history(self._W, raw, sens_prev, self.cfg.history_weight)

    # ---- target volume & constraint gate (shared with Beso) ---------------
    def next_target_vf(self, current_vf: float, feasible: bool,
                       violation: float | None = None) -> float:
        return gate_target_vf(self.cfg, current_vf, feasible, violation)

    # ---- SCIP moving-asymptote state (opt-in) ------------------------------
    def _advance_asymptotes(self, x: np.ndarray) -> None:
        """MMA-style adaptation of the per-element conservatism ``t_e`` from the
        last loop iteration's flip events. Two events can have happened since
        the previous ``update``: this optimiser's own flip (``prev_in ->
        prev_out``) and the loop's post-pass pruning (``prev_out -> x``); with
        binary states two nonzero events in one iteration are necessarily
        opposite (an immediate within-iteration ping-pong). An element whose
        newest event reverses its previously recorded direction oscillated ->
        tighten; one that stayed put or moved monotonically -> relax toward the
        plain linear gain (t_e = 1). Must run *before* the flip memory is
        overwritten with the current input."""
        n = x.size
        if self._scip_t is None or self._scip_t.size != n:
            self._scip_t = np.ones(n)
            self._scip_dir = np.zeros(n, dtype=np.int8)
        if (self._prev_in is None or self._prev_in.shape != x.shape
                or self._prev_out is None):
            return                               # no history yet: stay all-relaxed
        d1 = self._prev_out.astype(np.int8) - self._prev_in.astype(np.int8)
        d2 = x.astype(np.int8) - self._prev_out.astype(np.int8)
        last = self._scip_dir
        osc = (((d1 != 0) & (d2 != 0))                        # add->prune ping-pong
               | ((d1 != 0) & (last == -d1))                  # update reversed trend
               | ((d1 == 0) & (d2 != 0) & (last == -d2)))     # post-pass reversed it
        t = self._scip_t
        t[osc] *= self._scip_gamma_tight
        t[~osc] *= self._scip_gamma_relax
        np.clip(t, self._scip_t_min, 1.0, out=t)
        # newest nonzero flip direction wins; an element that did not flip at
        # all this iteration resets to 0, so only *consecutive*-iteration
        # reversals tighten — exactly MMA's (dx_k)(dx_{k-1}) < 0 test (a re-add
        # long after a removal is a legitimate back-off, not an oscillation).
        self._scip_dir = np.where(d2 != 0, d2, d1).astype(np.int8)

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
        if self._scip_on:                        # consumes _prev_in/_prev_out:
            self._advance_asymptotes(x)          # must precede the memory update
        self._prev_in = x.copy()
        damp = np.where(flipped, self.cfg.oscillation_damping, 1.0)

        if float(sens.max() if sens.size else 0.0) <= 0.0:
            # no usable energy signal (failed extraction / all-zero field):
            # every reduced gain would be pure -lam*vol noise, so keep the
            # design unchanged (TOBS's no-flip safety net).
            alive = self.mesh.keep_connected(x | self.protected, self.anchor)
            self._prev_out = alive.copy()
            return alive

        if self._scip_on:
            # SCIP conservative gains: the MMA reciprocal approximation at the
            # two binary points is exactly a per-element factor t_e — removing
            # a solid costs s/t (inflated), adding a void gains t*s (deflated).
            # t_e = 1 everywhere reproduces the linear subproblem bit-for-bit;
            # sign is preserved (t > 0), so the zero-signal guard above and the
            # bisection's monotonicity argument are unaffected.
            t = self._scip_t
            sens = np.where(x, sens / t, sens * t)

        K = max(1, math.floor(self.cfg.flip_limit * x.size))
        budget = target_vf * self.V0 - float(self.vol[self.protected].sum())
        lam = self._solve_lambda(x, sens, damp, K, budget)

        new_alive = x ^ self._flips_at(x, sens, damp, K, lam)
        new_alive |= self.protected              # protected always kept
        alive = self.mesh.keep_connected(new_alive, self.anchor)
        self._prev_out = alive.copy()
        return alive
