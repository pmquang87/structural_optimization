"""BESO core: rank, filter, threshold, add-back, connectivity, constraint gate.

The sensitivity number is the per-element internal-energy density read from
``/ANIM/ELEM/ENER`` (optionally von-Mises, or a blend). It is spatially filtered
and averaged with the previous iteration (Huang-Xie) for mesh-independence, then
a volume threshold keeps the highest-ranked elements: low-ranked solids are
deleted, high-ranked voids may be re-added (bi-directional), protected elements
are forced to stay, and disconnected islands are dropped. The loop decides the
*target* volume fraction each iteration from whether OpenRadioss reports the
current design feasible.
"""
from __future__ import annotations

import numpy as np

from .config import Beso as BesoCfg
from .mesh import Mesh
from .results import Results


class Beso:
    def __init__(self, mesh: Mesh, cfg: BesoCfg, protected_mask: np.ndarray,
                 anchor: np.ndarray | None = None):
        self.mesh = mesh
        self.cfg = cfg
        self.protected = np.asarray(protected_mask, dtype=bool)
        # Elements that anchor connectivity for island-dropping. Defaults to the
        # frozen (protected) set, but is decoupled so the BC/load region can keep
        # anchoring even when its elements are allowed to be deleted (i.e. when
        # they are no longer in ``protected``).
        self.anchor = (np.asarray(anchor, dtype=bool)
                       if anchor is not None else self.protected)
        self.vol = mesh.volumes
        self.V0 = float(self.vol.sum())
        self._W = mesh.filter_matrix(cfg.filter_radius)

    # ---- volume bookkeeping ------------------------------------------------
    def volume_fraction(self, alive_mask: np.ndarray) -> float:
        return float(self.vol[alive_mask].sum() / self.V0)

    # ---- sensitivity -------------------------------------------------------
    def raw_sensitivity(self, results: Results, elem_ids: np.ndarray,
                        alive_mask: np.ndarray) -> np.ndarray:
        """Map per-element OpenRadioss fields onto a full (N,) sensitivity array.

        ``elem_ids`` is the deck's full design element-id list (mesh order, sorted
        ascending). Dead elements get 0 and are pulled up only by the filter, which
        is what makes them eligible for bi-directional add-back.
        """
        n = elem_ids.size
        raw = np.zeros(n, dtype=float)
        pos = np.searchsorted(elem_ids, results.element_ids)
        valid = (pos < n) & (elem_ids[np.clip(pos, 0, n - 1)] == results.element_ids)
        pos = pos[valid]

        crit = self.cfg.sensitivity
        if crit == "vonmises":
            val = results.vonmises[valid]
        elif crit == "blend":
            en = results.energy[valid]; vm = results.vonmises[valid]
            en = en / en.max() if en.max() > 0 else en
            vm = vm / vm.max() if vm.max() > 0 else vm
            val = self.cfg.blend_weight * vm + (1 - self.cfg.blend_weight) * en
        else:  # "energy" (default): internal-energy density
            val = results.energy[valid]
        raw[pos] = val
        return raw

    def filter_history(self, raw: np.ndarray,
                       sens_prev: np.ndarray | None) -> np.ndarray:
        """Spatially filter, then average with the previous iteration."""
        filt = self._W @ raw
        if sens_prev is None or sens_prev.shape != filt.shape:
            return filt
        h = self.cfg.history_weight
        return h * filt + (1.0 - h) * sens_prev

    # ---- target volume & constraint gate -----------------------------------
    def next_target_vf(self, current_vf: float, feasible: bool) -> float:
        """Mass objective: shrink by the evolution rate while feasible; if the
        last design violated a limit, add material back to recover."""
        er = self.cfg.evolution_rate
        if feasible:
            return max(self.cfg.target_volume_fraction, current_vf * (1.0 - er))
        return min(1.0, current_vf * (1.0 + er))    # back off toward feasibility

    # ---- alive-set update --------------------------------------------------
    def update(self, alive_mask: np.ndarray, sens: np.ndarray,
               target_vf: float) -> np.ndarray:
        """New alive mask: protected elements are always kept; among the
        *removable* (non-protected) elements keep the highest-ranked until the
        total volume reaches *target_vf*, cap bi-directional add-back, drop islands.

        Only the removable elements are ranked against the volume budget — never
        all elements with protected then forced back on. Otherwise, when the
        lowest-sensitivity elements happen to be protected (e.g. a low-stress
        keep-out region), the global volume threshold would mark only protected
        elements for deletion and ``|= protected`` would immediately restore them,
        stalling the optimisation at the start volume (no element ever removed).
        """
        alive_mask = np.asarray(alive_mask, dtype=bool)
        target_V = target_vf * self.V0

        cand = self.protected.copy()
        removable = np.flatnonzero(~self.protected)
        if removable.size:
            # protected volume is always kept; spend what's left on the best removables
            budget = max(0.0, target_V - float(self.vol[self.protected].sum()))
            order = removable[np.argsort(sens[removable])[::-1]]   # high sensitivity first
            cum = np.cumsum(self.vol[order])
            cand[order[cum <= budget]] = True

        # cap re-added (dead -> alive) volume at max_add_ratio * V0
        newly = cand & ~alive_mask
        cap = self.cfg.max_add_ratio * self.V0
        if self.vol[newly].sum() > cap and cap >= 0:
            nidx = np.flatnonzero(newly)
            nord = nidx[np.argsort(sens[nidx])[::-1]]
            keep_add = nord[np.cumsum(self.vol[nord]) <= cap]
            cand[newly] = False
            cand[keep_add] = True
            cand |= self.protected

        # drop anything no longer connected to an anchor (BC/load) seed
        cand = self.mesh.keep_connected(cand, self.anchor)
        return cand
