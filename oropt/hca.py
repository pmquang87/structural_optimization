"""HCA — Hybrid Cellular Automata (a config-selectable alternative to BESO that
regulates each element's *virtual density* with a local controller instead of
ranking against a threshold; the method behind LS-TaSC, built specifically for
nonlinear/contact problems with no design gradients — exactly this regime).

Tovar, Patel, Niebur, Sen & Renaud, "Topology Optimization Using a Hybrid
Cellular Automaton Method With Local Control Rules", *J. Mech. Des.* 128(6),
2006; Patel, Kang, Renaud & Tovar (crashworthiness HCA); LS-TaSC theory manual.

Every element keeps a continuous virtual density ``x_e in [x_min, 1]`` that
persists between iterations (the way the level-set persists its ``phi`` field).
The only structural response available is the per-element internal-energy
density from ``/ANIM/ELEM/ENER``, already spatially filtered (the neighbourhood
averaging of the cellular automaton — the same filter matrix as BESO) and
history-blended by :func:`filter_history` (LS-TaSC's weighted sum over the
previous iterations). Each iteration:

* a **proportional controller** drives every element's density toward a
  *uniform* energy-density setpoint ``S*``:
  ``x_e <- clip(x_e + clip(Kp * (S_e - S*)/S*, -move_limit, +move_limit))`` —
  overworked elements (``S_e > S*``) gain material, underworked ones lose it;
* the **setpoint ``S*`` is found by bisection** so that the *thresholded*
  design hits the per-iteration volume target (the same
  ``evolution_rate``/``target_volume_fraction`` gate as BESO). The controller
  step is monotone non-increasing in ``S*``, so the bisection is well-posed;
* the density field is **thresholded to the binary alive mask** (oropt is
  hard-kill element deletion): alive iff ``x_e >= 0.5``. Protected elements are
  pinned at full density and forced alive, and disconnected islands are dropped
  via :meth:`Mesh.keep_connected` — exactly like BESO.

With the fixed 0.5 threshold an element can only be *removed in a single
iteration* when ``min(Kp, move_limit) > 0.5`` (the defaults allow it); smaller
values make densities decay over several iterations — removal then lags the
per-iteration volume target and catches up, which is the classic damped HCA
behaviour, at the cost of extra (expensive) solves.

Mirrors the :class:`~oropt.beso.Beso` interface so the loop can drive either.
"""
from __future__ import annotations

import numpy as np

from .beso import blend_history, gate_target_vf, map_sensitivity
from .config import HcaOpts as HcaCfg
from .mesh import Mesh
from .results import Results

# Density floor: a dead element keeps a small virtual density (never exactly 0)
# so the controller's relative error stays finite and the element can regrow.
_X_MIN = 0.01
# Hard-kill threshold: an element is alive iff its virtual density is >= this.
# Fixed (not a config knob) to keep the HCA config minimal; the midpoint of
# [x_min, 1] leaves symmetric headroom for growth and decay.
_ALIVE = 0.5


class Hca:
    def __init__(self, mesh: Mesh, cfg: HcaCfg, protected_mask: np.ndarray,
                 anchor: np.ndarray | None = None):
        self.mesh = mesh
        self.cfg = cfg
        self.protected = np.asarray(protected_mask, dtype=bool)
        # Elements that anchor connectivity for island-dropping (defaults to the
        # protected set, decoupled so the BC/load region keeps anchoring even when
        # its elements are deletable). Mirrors Beso/LevelSet/Tobs.
        self.anchor = (np.asarray(anchor, dtype=bool)
                       if anchor is not None else self.protected)
        self.vol = mesh.volumes
        self.V0 = float(self.vol.sum())
        self._W = mesh.filter_matrix(cfg.filter_radius)

        # Per-element virtual density; initialised lazily from the first alive
        # mask seen (so it matches the current / resumed geometry).
        self.x: np.ndarray | None = None
        # HCA-internal running average of the controller's input field (the
        # LS-TaSC multi-iteration weighted sum); None until the first update.
        self._field_prev: np.ndarray | None = None

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

    # ---- controller step + setpoint bisection ------------------------------
    def _step(self, x: np.ndarray, field: np.ndarray,
              s_star: float) -> np.ndarray:
        """One proportional-controller update of the density field toward the
        energy-density setpoint *s_star*, move-limited and clipped to
        ``[x_min, 1]``. Monotone non-increasing in *s_star* element-wise (the
        relative error ``S_e/S* - 1`` is), which makes the bisection valid."""
        ml = self.cfg.move_limit
        delta = np.clip(self.cfg.kp * (field - s_star) / s_star, -ml, ml)
        return np.clip(x + delta, _X_MIN, 1.0)

    def _removable_vol_at(self, x: np.ndarray, field: np.ndarray,
                          s_star: float) -> float:
        """Volume of *removable* (non-protected) elements that would be alive
        after a controller step at setpoint *s_star*.

        Uses exactly the step that :meth:`update` applies and the threshold it
        masks with, so the bisection target and the final mask never disagree at
        the boundary. Non-increasing in *s_star* (see :meth:`_step`)."""
        keep = (self._step(x, field, s_star) >= _ALIVE) & ~self.protected
        return float(self.vol[keep].sum())

    def _solve_setpoint(self, x: np.ndarray, field: np.ndarray,
                        budget: float) -> float:
        """Smallest setpoint ``S*`` whose removable kept-volume is <= *budget*
        (i.e. the *largest* volume not exceeding the budget). Protected elements
        are excluded here and forced alive by :meth:`update`.

        ``S*`` is a positive energy scale, so the bisection walks the geometric
        mean over ``[~0, ~inf) * max(field)``: at the low end every element with
        any energy gains the full move limit (grow everything), at the high end
        every element loses ``min(Kp, move_limit)`` (maximum removal). When even
        that maximum step cannot reach the budget — the move limit binds, like
        TOBS's flip cap — the high end is returned and the loop's next target
        catches up on later iterations.
        """
        smax = float(field.max())
        lo = 1e-9 * smax
        hi = 1e9 * smax
        if budget <= 0.0:
            return hi                            # target below the protected floor
        if self._removable_vol_at(x, field, lo) <= budget:
            return lo                            # target above reach: grow everything
        if self._removable_vol_at(x, field, hi) > budget:
            return hi                            # move limit binds: max removal
        # invariant: vol(lo) > budget >= vol(hi); keep the hi side so vol <= budget
        for _ in range(64):
            mid = float(np.sqrt(lo * hi))
            if self._removable_vol_at(x, field, mid) > budget:
                lo = mid
            else:
                hi = mid
        return hi

    # ---- alive-set update --------------------------------------------------
    def update(self, alive_mask: np.ndarray, sens: np.ndarray,
               target_vf: float) -> np.ndarray:
        """One HCA iteration: blend the field, bisect the setpoint, step the
        densities, threshold to the new alive mask.

        ``sens`` is the filtered/history-averaged per-element energy. Returns
        the new alive mask; ``self.x`` is updated to stay consistent with it
        (protected elements pinned at full density; island-dropping excepted,
        like the level-set's ``phi``).
        """
        alive_mask = np.asarray(alive_mask, dtype=bool)
        if self.x is None:
            self.x = np.where(alive_mask, 1.0, _X_MIN)

        # HCA-internal running blend with the previous iterations' field (the
        # LS-TaSC multi-iteration weighted sum). At the default weight 1.0 this
        # is a no-op: the shared filter_history blend already covers it.
        field = np.asarray(sens, dtype=float)
        w = self.cfg.field_history_weight
        if self._field_prev is not None and w < 1.0:
            field = w * field + (1.0 - w) * self._field_prev
        self._field_prev = field

        if float(field.max() if field.size else 0.0) <= 0.0:
            # no usable energy signal (failed extraction / all-zero field):
            # taking a controller step would erode everything uniformly, so
            # keep the design unchanged (TOBS's no-flip safety net).
            alive = alive_mask | self.protected
            return self.mesh.keep_connected(alive, self.anchor)

        target_V = target_vf * self.V0
        protected_V = float(self.vol[self.protected].sum())
        s_star = self._solve_setpoint(self.x, field, target_V - protected_V)

        self.x = self._step(self.x, field, s_star)
        self.x[self.protected] = 1.0             # protected pinned at full density
        alive = (self.x >= _ALIVE) | self.protected
        return self.mesh.keep_connected(alive, self.anchor)
