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


# ---- shared sensitivity helpers (reused by other optimisers, e.g. level-set) --
def map_sensitivity(results: Results, elem_ids: np.ndarray,
                    sensitivity: str = "energy",
                    blend_weight: float = 0.5) -> np.ndarray:
    """Map per-element OpenRadioss fields onto a full (N,) sensitivity array.

    ``elem_ids`` is the deck's full design element-id list (mesh order, sorted
    ascending). Elements with no result (dead/absent) get 0 and are pulled up only
    by the spatial filter. ``sensitivity`` is ``"energy"`` | ``"vonmises"`` |
    ``"blend"`` (``blend_weight`` weights von-Mises in the blend).
    """
    n = elem_ids.size
    raw = np.zeros(n, dtype=float)
    pos = np.searchsorted(elem_ids, results.element_ids)
    valid = (pos < n) & (elem_ids[np.clip(pos, 0, n - 1)] == results.element_ids)
    pos = pos[valid]
    if sensitivity == "vonmises":
        val = results.vonmises[valid]
    elif sensitivity == "blend":
        en = results.energy[valid]; vm = results.vonmises[valid]
        en = en / en.max() if en.max() > 0 else en
        vm = vm / vm.max() if vm.max() > 0 else vm
        val = blend_weight * vm + (1 - blend_weight) * en
    else:  # "energy" (default): internal-energy density
        val = results.energy[valid]
    raw[pos] = val
    return raw


def blend_history(W, raw: np.ndarray, sens_prev: np.ndarray | None,
                  history_weight: float) -> np.ndarray:
    """Spatially filter (``W @ raw``), then average with the previous iteration."""
    filt = W @ raw
    if sens_prev is None or sens_prev.shape != filt.shape:
        return filt
    return history_weight * filt + (1.0 - history_weight) * sens_prev


def gate_target_vf(cfg, current_vf: float, feasible: bool,
                   violation: float | None = None) -> float:
    """Per-iteration volume target shared by every optimiser (the constraint
    gate behind each ``next_target_vf``): shrink by the evolution rate while
    feasible; if the last design violated a limit, add material back to recover.

    *violation* is the worst constraint-utilisation ratio ``value/limit`` across
    load cases and both limit types (``v <= 1`` = feasible; see
    ``loop.worst_violation``). ``None`` — or the default knobs — reproduces the
    classic binary gate (a fixed ±ER step from the feasible flag alone) exactly.
    With the knobs set the step becomes proportional to the constraint values,
    the way TOSCA's controller mode / LS-TaSC's constrained scaling react to the
    stress level instead of an on/off flag:

    * ``backoff_gain > 0`` — infeasible growth is
      ``ER * max(floor, min(gain*(v-1), cap))`` instead of a full ER step, so a
      1 % violation triggers a nudge and a 50 % violation a (capped) surge,
      rather than the same fixed step for both. ``backoff_floor`` bounds the
      nudge from below: without it a persistent hair-above-the-limit violation
      (say 0.3 %) yields ``ER*gain*0.003`` ~ nothing, and the run sits in a
      limit cycle pinned just above the allowable;
    * ``damping_threshold < 1`` — while feasible with ``v`` above the threshold,
      removal slows by ``(1-v)/(1-threshold)``, gliding the design into the
      limit instead of overshooting and ping-ponging feasible/infeasible.
    """
    er = cfg.evolution_rate
    if not feasible:
        if violation is not None and cfg.backoff_gain > 0.0:
            er *= max(cfg.backoff_floor,
                      min(cfg.backoff_gain * (violation - 1.0),
                          cfg.backoff_cap))
        return min(1.0, current_vf * (1.0 + er))    # back off toward feasibility
    if violation is not None and cfg.damping_threshold < 1.0 \
            and violation > cfg.damping_threshold:
        er *= max(0.0, 1.0 - violation) / (1.0 - cfg.damping_threshold)
    return max(cfg.target_volume_fraction, current_vf * (1.0 - er))


def combine_sensitivity(raws: list[np.ndarray],
                        weights: list[float]) -> np.ndarray:
    """Weighted-sum sensitivity over several load cases (optimiser-agnostic).

    ``s_e = sum_i weight_i * (raw_i / max(raw_i))`` — each case's raw per-element
    sensitivity (from :func:`map_sensitivity`) is normalised by its own peak
    before weighting, so the weights express *relative* importance regardless of
    how the cases' absolute strain-energy scales differ. Done on the raw
    (pre-filter) sensitivity so the existing spatial filter + Huang-Xie history
    blend then apply once to the combined field.

    A single case is returned untouched (no normalisation, weight ignored) so the
    classic single-load path stays byte-identical: per-iteration max normalisation
    would otherwise rescale the sensitivity by a quantity that varies each
    iteration and perturb the history blend.
    """
    if len(raws) == 1:
        return raws[0]
    total = np.zeros_like(raws[0], dtype=float)
    for raw, w in zip(raws, weights):
        mx = float(raw.max()) if raw.size else 0.0
        total += w * (raw / mx if mx > 0.0 else raw)
    return total


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

        Dead elements get 0 and are pulled up only by the filter, which is what
        makes them eligible for bi-directional add-back.
        """
        return map_sensitivity(results, elem_ids, self.cfg.sensitivity,
                               self.cfg.blend_weight)

    def filter_history(self, raw: np.ndarray,
                       sens_prev: np.ndarray | None) -> np.ndarray:
        """Spatially filter, then average with the previous iteration."""
        return blend_history(self._W, raw, sens_prev, self.cfg.history_weight)

    # ---- target volume & constraint gate -----------------------------------
    def next_target_vf(self, current_vf: float, feasible: bool,
                       violation: float | None = None) -> float:
        """Mass objective: shrink by the evolution rate while feasible; if the
        last design violated a limit, add material back to recover. The step is
        proportional to *violation* when the back-off knobs are set (see
        :func:`gate_target_vf`)."""
        return gate_target_vf(self.cfg, current_vf, feasible, violation)

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
