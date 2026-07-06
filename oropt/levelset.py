"""Discrete nodal level-set topology optimiser (a config-selectable alternative
to BESO that produces smoother boundaries than ragged element removal).

The structure is represented by a nodal level-set field ``phi`` (one value per
mesh node). An element is *alive* iff the mean of ``phi`` over its nodes is
non-negative. There are no gradients/adjoint available — the only shape
sensitivity is the per-element strain-energy density, already spatially filtered
and history-averaged by :func:`filter_history` (shared with BESO). Each iteration:

* scatter the filtered per-element energy onto nodes (volume-weighted average over
  incident elements) -> a nodal "velocity" ``Vn`` (high energy -> grow, low ->
  erode), minus a nucleation reaction term ``nucleation_rate * (1 - Vn)`` (a
  crude topological derivative) so low-energy regions carry a *negative* speed
  and can sink below the threshold anywhere — not only at an existing void
  interface;
* evolve ``phi <- phi + dt*(Vn - lambda)`` and apply a few Laplacian/Jacobi
  smoothing passes as reaction-diffusion-style regularisation (cheap and stable
  on an unstructured tet mesh — no full Hamilton-Jacobi solve);
* choose ``lambda`` by bisection so the thresholded alive volume hits the
  per-iteration target volume (reusing evolution_rate / target_volume_fraction);
* threshold ``phi`` -> new alive mask, force protected elements alive, and drop
  islands not connected to the anchor.

The smoothing operator is row-stochastic, so it preserves constants; a uniform
``-dt*lambda`` shift therefore commutes with smoothing, which keeps the
thresholded volume *exactly monotone* in ``lambda`` and makes the bisection
well-posed. ``phi`` is clamped to +/-``band_width`` each step to stay bounded.

Mirrors the :class:`~oropt.beso.Beso` interface so the loop can drive either.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.stats import rankdata

from .beso import blend_history, gate_target_vf, map_sensitivity
from .config import LevelSet as LevelSetCfg
from .mesh import Mesh
from .results import Results

# Under-relaxation factor for one smoothing pass: phi <- (1-w)*phi + w*(S @ phi).
# Fixed (not a config knob) to keep the level-set config minimal; 0.5 is a stable
# Jacobi/Laplacian relaxation.
_SMOOTH_RELAX = 0.5


class LevelSet:
    def __init__(self, mesh: Mesh, cfg: LevelSetCfg, protected_mask: np.ndarray,
                 anchor: np.ndarray | None = None):
        self.mesh = mesh
        self.cfg = cfg
        self.protected = np.asarray(protected_mask, dtype=bool)
        # Elements that anchor connectivity for island-dropping (defaults to the
        # protected set, decoupled so the BC/load region keeps anchoring even when
        # its elements are deletable). Mirrors Beso.
        self.anchor = (np.asarray(anchor, dtype=bool)
                       if anchor is not None else self.protected)
        self.vol = mesh.volumes
        self.V0 = float(self.vol.sum())
        self._W = mesh.filter_matrix(cfg.filter_radius)

        # Element-node incidence B (ne x n_nodes): used to scatter element fields
        # onto nodes and to build the node graph for smoothing.
        ne = mesh.n_elements
        rows = np.repeat(np.arange(ne), 4)
        cols = mesh.conn_rows.ravel()
        B = sparse.csr_matrix((np.ones(rows.size), (rows, cols)),
                              shape=(ne, mesh.n_nodes))
        self._B = B
        self._node_vol = np.asarray(B.T @ self.vol).ravel()   # sum of incident vols / node
        self._smooth_op = self._build_smooth_op(B)

        # Nodal level-set field; initialised lazily from the first alive mask and
        # sensitivity seen (so it matches the current / resumed geometry).
        self.phi: np.ndarray | None = None

    # ---- node graph / smoothing -------------------------------------------
    @staticmethod
    def _build_smooth_op(B: sparse.csr_matrix) -> sparse.csr_matrix:
        """Row-stochastic neighbour-averaging operator S over the node graph.

        Two nodes are neighbours if they share an element (``C = B.T @ B`` minus
        its diagonal). Each row is normalised to sum to 1 over its neighbours;
        nodes with no neighbours get a self-loop (identity) so the whole operator
        preserves constants — essential for the bisection's monotonicity.
        """
        C = (B.T @ B).tocsr()
        C.setdiag(0)
        C.eliminate_zeros()
        deg = np.asarray(C.sum(axis=1)).ravel()
        inv = np.zeros_like(deg, dtype=float)
        nz = deg > 0
        inv[nz] = 1.0 / deg[nz]
        S = sparse.diags(inv) @ C
        iso = ~nz
        if iso.any():
            S = S + sparse.diags(iso.astype(float))
        return S.tocsr()

    def _smooth(self, phi: np.ndarray, passes: int) -> np.ndarray:
        out = np.asarray(phi, dtype=float)
        for _ in range(max(0, int(passes))):
            out = (1.0 - _SMOOTH_RELAX) * out + _SMOOTH_RELAX * (self._smooth_op @ out)
        return out

    # ---- scatter element -> node ------------------------------------------
    def _scatter(self, elem_vals: np.ndarray) -> np.ndarray:
        """Volume-weighted average of a per-element field onto nodes."""
        num = np.asarray(self._B.T @ (self.vol * elem_vals)).ravel()
        out = np.zeros_like(num)
        nz = self._node_vol > 0
        out[nz] = num[nz] / self._node_vol[nz]
        return out

    def _elem_mean(self, phi: np.ndarray) -> np.ndarray:
        """Mean of a nodal field over each element's 4 nodes."""
        return phi[self.mesh.conn_rows].mean(axis=1)

    def _init_phi(self, alive_mask: np.ndarray, sens: np.ndarray) -> np.ndarray:
        """Signed energy-rank spread: alive elements over (0, +band_width] by
        their filtered-energy rank, void elements over [-band_width, 0).

        A binary indicator (+1 alive / -1 void) is NOT used: it gives every
        element away from an alive/void interface the same value, so the tau
        bisection can only ever pick elements in the smoothing fringe next to
        existing voids — low-energy material in the free interior is never
        carved (no hole nucleation; observed live on the elevator-linkage run).
        Rank-spreading each set instead orders the whole part from step 0 while
        keeping the sign contract (alive <=> phi >= 0) so a resumed geometry
        still thresholds back to the mask it was initialised from.
        """
        alive = np.asarray(alive_mask, dtype=bool)
        s = np.asarray(sens, dtype=float)
        bw = self.cfg.band_width
        phi_e = np.zeros(alive.size)
        for mask, lo, hi in ((alive, 0.0, bw), (~alive, -bw, 0.0)):
            k = int(mask.sum())
            if k:
                r = rankdata(s[mask], method="average")        # ties stay tied
                phi_e[mask] = lo + (hi - lo) * r / (k + 1.0)   # open interval
        phi = self._scatter(phi_e)
        phi = self._smooth(phi, self.cfg.smoothing_passes)
        return np.clip(phi, -bw, bw)

    # ---- public: thresholding ---------------------------------------------
    def elements_alive(self, phi: np.ndarray) -> np.ndarray:
        """Elements whose mean phi >= 0, with protected elements forced alive.

        Single source of truth for phi -> alive, so the stored field stays
        self-consistent with the returned mask (modulo island-dropping)."""
        return (self._elem_mean(phi) >= 0.0) | self.protected

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

    # ---- lambda bisection --------------------------------------------------
    def _removable_vol_at(self, phi_base: np.ndarray, tau: float) -> float:
        """Volume of *removable* (non-protected) elements that would be alive if
        the field were shifted by ``-tau`` (i.e. ``phi = clip(phi_base - tau)``).

        Uses exactly the field that :meth:`update` stores and :meth:`elements_alive`
        thresholds, so the bisection target and the final mask never disagree by a
        rounding ULP at the boundary. Non-increasing in ``tau`` (clip is monotone),
        so the bisection is well-posed."""
        bw = self.cfg.band_width
        em = self._elem_mean(np.clip(phi_base - tau, -bw, bw))
        keep = (em >= 0.0) & ~self.protected
        return float(self.vol[keep].sum())

    def _solve_tau(self, phi_base: np.ndarray, budget: float) -> float:
        """Smallest shift ``tau`` whose removable kept-volume is <= ``budget``
        (i.e. the *largest* volume not exceeding the budget). Protected elements
        are excluded here and forced alive by :meth:`elements_alive`.

        ``tau`` plays the role of ``dt*lambda``: because the smoothing preserves
        constants, subtracting ``tau`` shifts every node uniformly, which is the
        Lagrange-multiplier step that meets the per-iteration volume target.
        """
        if not (~self.protected).any():
            return float(phi_base.max()) + 1.0 if phi_base.size else 0.0
        bw = self.cfg.band_width
        lo = float(phi_base.min()) - bw - 1.0   # all removable clipped to +bw -> all kept
        hi = float(phi_base.max()) + bw + 1.0   # all removable clipped to -bw -> none kept
        if budget <= 0.0:
            return hi                            # target below the protected floor
        if self._removable_vol_at(phi_base, lo) <= budget:
            return lo                            # target above full volume: keep all
        # invariant: vol(lo) > budget >= vol(hi); keep the hi side so vol <= budget
        for _ in range(64):
            mid = 0.5 * (lo + hi)
            if self._removable_vol_at(phi_base, mid) > budget:
                lo = mid
            else:
                hi = mid
        return hi

    # ---- alive-set update --------------------------------------------------
    def update(self, alive_mask: np.ndarray, sens: np.ndarray,
               target_vf: float) -> np.ndarray:
        """Evolve phi one pseudo-time step toward *target_vf* and threshold it.

        ``sens`` is the filtered/history-averaged per-element energy. Returns the
        new alive mask; ``self.phi`` is updated to stay consistent with it.
        """
        alive_mask = np.asarray(alive_mask, dtype=bool)
        if self.phi is None:
            self.phi = self._init_phi(alive_mask, sens)

        # nodal velocity from the shape sensitivity (normalised so dt is meaningful)
        Vn = self._scatter(np.asarray(sens, dtype=float))
        scale = float(np.abs(Vn).max())
        if scale > 0:
            Vn = Vn / scale

        # nucleation reaction term (crude topological derivative): low-energy
        # nodes get a negative speed instead of the >= 0 pure-energy velocity.
        # Its uniform part is absorbed by tau; what remains lets slack material
        # sink below the threshold anywhere — and un-pins nodes parked at the
        # +band_width clamp, where the pure-energy velocity would hold them.
        vel = Vn - self.cfg.nucleation_rate * (1.0 - Vn)

        # evolve + regularise (smoothing preserves constants -> see _solve_tau)
        phi_base = self._smooth(self.phi + self.cfg.dt * vel, self.cfg.smoothing_passes)

        # bisect the threshold so the kept volume meets the per-iteration target
        target_V = target_vf * self.V0
        protected_V = float(self.vol[self.protected].sum())
        tau = self._solve_tau(phi_base, target_V - protected_V)

        bw = self.cfg.band_width
        self.phi = np.clip(phi_base - tau, -bw, bw)
        alive = self.elements_alive(self.phi)
        alive = self.mesh.keep_connected(alive, self.anchor)
        return alive
