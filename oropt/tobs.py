"""TOBS — Topology Optimisation of Binary Structures (a config-selectable
alternative to BESO that picks each iteration's element flips with an integer
linear program instead of a sensitivity threshold).

Sivapuram & Picelli, "Topology optimization of binary structures using Integer
Linear Programming", *Finite Elements in Analysis and Design* 139:49-61 (2018).

The design variables are the same binary alive/void element flags as BESO
(``x_e in {0,1}``, 1 == alive). OpenRadioss exposes no design gradients, so the
only usable sensitivity is the per-element strain-energy density, already
spatially filtered and history-averaged by :func:`filter_history` (shared with
BESO). Each iteration TOBS solves a small 0/1 ILP for the *flips*
``dx_e in {-1,0,+1}`` with :func:`scipy.optimize.milp` (HiGHS):

* **objective** — maximise ``sum_e s_e * dx_e`` i.e. minimise ``-s . dx`` (so the
  step keeps high- and discards low-sensitivity material);
* **move limit** — ``sum_e |dx_e| <= beta * N`` (``beta`` = ``flip_limit``): at
  most a fraction of the elements flip per iteration, which is what makes the
  successive linearisations valid;
* **volume target** — the element-volume-weighted volume is stepped toward the
  per-iteration target (the same ``evolution_rate``/``target_volume_fraction``
  gate as BESO) as a linearised constraint. The target change is clamped to the
  volume the move limit can actually deliver and relaxed by a band of width
  ``constraint_relaxation * V0`` (the paper's ``epsilon``-relaxation) so the
  binary subproblem is *always feasible* and a discrete solution lands in-band.

For ``dx_e`` the standard TOBS variable bounds apply: a currently-solid element
may only stay or be removed (``dx_e in {-1,0}``), a currently-void element may
only stay or be added (``dx_e in {0,+1}``). Protected elements are pinned
(``dx_e = 0``) and forced alive afterwards, and disconnected islands are dropped
via :meth:`Mesh.keep_connected` — exactly like BESO.

Mirrors the :class:`~oropt.beso.Beso` interface so the loop can drive either.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .beso import blend_history, gate_target_vf, map_sensitivity
from .config import TobsOpts as TobsCfg
from .mesh import Mesh
from .results import Results


class Tobs:
    def __init__(self, mesh: Mesh, cfg: TobsCfg, protected_mask: np.ndarray,
                 anchor: np.ndarray | None = None):
        self.mesh = mesh
        self.cfg = cfg
        self.protected = np.asarray(protected_mask, dtype=bool)
        # Elements that anchor connectivity for island-dropping (defaults to the
        # protected set, decoupled so the BC/load region keeps anchoring even when
        # its elements are deletable). Mirrors Beso/LevelSet.
        self.anchor = (np.asarray(anchor, dtype=bool)
                       if anchor is not None else self.protected)
        self.vol = mesh.volumes
        self.V0 = float(self.vol.sum())
        self._W = mesh.filter_matrix(cfg.filter_radius)

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

    # ---- volume-target band ------------------------------------------------
    def _volume_band(self, x: np.ndarray, sens: np.ndarray, K: int,
                     dV: float) -> tuple[float, float]:
        """``(lo, hi)`` bounds for the linearised volume constraint
        ``lo <= sum_e vol_e * dx_e <= hi``.

        The signed desired change ``dV`` (from :meth:`next_target_vf`) is clamped
        to the volume the move limit can actually deliver — the ``K`` lowest-
        sensitivity solids when shrinking, the ``K`` highest-sensitivity voids when
        backing off — so a feasible flip set always exists within the move limit.
        The band is widened by ``epsilon`` (``constraint_relaxation * V0``), but
        never by less than one element volume, so a *discrete* flip set is
        guaranteed to land inside it (the paper's constraint relaxation).
        """
        prot, vol = self.protected, self.vol
        eps = self.cfg.constraint_relaxation * self.V0
        if dV < 0.0:                                   # shrink: remove low-sens solids
            rem = np.flatnonzero(x & ~prot)
            if rem.size == 0:
                return -eps, 0.0                       # nothing removable
            capK = rem[np.argsort(sens[rem])][:K]      # K lowest-sensitivity removable
            reach = float(vol[capK].sum())             # most volume removable in K flips
            slack = max(eps, float(vol[capK].max()))
            tgt = -min(-dV, reach)
            return tgt - slack, tgt
        if dV > 0.0:                                   # back off: add high-sens voids
            add = np.flatnonzero(~x & ~prot)
            if add.size == 0:
                return 0.0, eps                        # nothing to add
            capK = add[np.argsort(sens[add])[::-1]][:K]   # K highest-sensitivity voids
            reach = float(vol[capK].sum())
            slack = max(eps, float(vol[capK].max()))
            tgt = min(dV, reach)
            return tgt, tgt + slack
        return -eps, eps                               # already at target

    # ---- alive-set update --------------------------------------------------
    def update(self, alive_mask: np.ndarray, sens: np.ndarray,
               target_vf: float) -> np.ndarray:
        """Solve the per-iteration TOBS ILP for the element flips and apply them.

        ``sens`` is the filtered/history-averaged per-element energy. Returns the
        new alive mask: protected elements are forced alive and islands not
        connected to the anchor are dropped, exactly like BESO.
        """
        x = np.asarray(alive_mask, dtype=bool)
        N = x.size
        sens = np.asarray(sens, dtype=float)
        prot, vol = self.protected, self.vol

        # per-variable flip bounds: solid -> {-1,0}, void -> {0,+1};
        # protected pinned to 0 (milp never flips them; forced alive below).
        lb = np.where(x, -1.0, 0.0)
        ub = np.where(x, 0.0, 1.0)
        lb[prot] = 0.0
        ub[prot] = 0.0

        # objective: maximise sum(s * dx) == minimise sum(-s * dx)
        c = -sens

        # move limit: sum|dx| <= K. Under the bounds above |dx_e| = a_e * dx_e with
        # a_e = -1 for solids (dx<=0) and +1 for voids (dx>=0).
        K = max(1, math.floor(self.cfg.flip_limit * N))
        a = np.where(x, -1.0, 1.0)
        move = LinearConstraint(a.reshape(1, -1), -np.inf, float(K))

        # volume target (clamped to move-limit capacity + epsilon-relaxed band)
        dV = target_vf * self.V0 - float(vol[x].sum())
        lo, hi = self._volume_band(x, sens, K, dV)
        vcon = LinearConstraint(vol.reshape(1, -1), lo, hi)

        res = milp(c, constraints=[move, vcon], integrality=np.ones(N),
                   bounds=Bounds(lb, ub))
        if res.success and res.x is not None:
            dx = np.rint(res.x).astype(int)
            new_alive = (x.astype(np.int8) + dx) != 0
        else:                                          # safety net: no flips this step
            new_alive = x.copy()
        new_alive |= prot                              # protected always kept
        return self.mesh.keep_connected(new_alive, self.anchor)
