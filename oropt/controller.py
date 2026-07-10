"""Multipoint feasibility back-off: the volume target from a *predicted*
constraint boundary instead of a reaction to last iteration's feasible flag.

The classic gate (:func:`oropt.beso.gate_target_vf`) reacts: shrink one
evolution-rate step while feasible, grow when violated — an on/off rule that
ping-pongs across the limit, and the proportional knobs (``backoff_gain`` /
``damping_threshold``) only shape the *reaction*. LS-TaSC's multipoint method
(Roux, "The LS-TaSC multipoint method for constrained topology optimization",
LS-DYNA conf.; survey in ``docs/topology_sota_2026.md``) instead treats the
few *global* variables — here the mass/volume target — as a small constrained
optimisation driven by **numerical derivatives over points the run has
already visited**: response data every iteration produces anyway, so the
constraint boundary is *predicted* at zero extra solves.

This module is that idea reduced to oropt's one global variable. Each
iteration the loop records the pair ``(volume fraction, worst
constraint-utilisation ratio v)`` (see :func:`oropt.loop.worst_violation`;
``v <= 1`` = feasible). :meth:`MultipointBackoff.next_target_vf` then fits a
local linear model ``v(vf)`` over the last ``multipoint_window`` points and
steps the volume target straight toward the vf where the model crosses
``utilization_target`` — the predicted constraint boundary — instead of
walking blind ER steps into it and bouncing back:

* the step is clamped to the gate's own authority: never shrinking faster
  than one ``evolution_rate`` step, never growing faster than
  ``ER * backoff_cap``, never below ``target_volume_fraction``;
* while *violated* the target always grows by at least
  ``ER * backoff_floor`` (whatever the fit says: the measured violation
  outranks the model);
* whenever the fit is unusable — fewer than two recorded points, no volume
  spread, a non-negative slope (removing material should *increase* the
  utilisation; anything else means the local model is noise), or no finite
  violation signal at all (no limits configured, a diverged solve) — it
  falls back to the classic gate verbatim, so ``backoff_mode: multipoint``
  can never be worse-defined than the default.

The recorded points are the controller's only state; the loop checkpoints
them (``ctrl`` in ``checkpoint.npz``) so a ``--resume`` keeps the fit instead
of re-learning the boundary. Selected per optimiser block via
``backoff_mode: multipoint`` — the optimiser itself is untouched, this wraps
only its ``next_target_vf``.

Adapting the *load-case weights* the same way (LS-TaSC's second global
variable set) is future work; today the per-case weights stay as configured.
"""
from __future__ import annotations

import math

import numpy as np

from .beso import gate_target_vf


class MultipointBackoff:
    """Volume-target controller fitted to the run's own (vf, violation) history.

    Wraps the classic gate of one optimiser config block *cfg* (any of the
    per-optimiser dataclasses — they all carry the shared back-off knobs).
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.vfs: list[float] = []
        self.violations: list[float] = []

    # ---- history -----------------------------------------------------------
    def record(self, vf: float, violation: float | None) -> None:
        """Add this iteration's measured point. Non-finite violations (a
        diverged solve reported as inf) and blank ones are not usable data."""
        if violation is None or not math.isfinite(violation):
            return
        self.vfs.append(float(vf))
        self.violations.append(float(violation))

    def state(self) -> np.ndarray:
        """The recorded points as an (n, 2) array for the checkpoint."""
        if not self.vfs:
            return np.empty((0, 2))
        return np.column_stack([self.vfs, self.violations])

    def restore(self, state: np.ndarray | None) -> None:
        """Reload points saved by :meth:`state` (a resume). Anything malformed
        is ignored — the controller then re-learns from the gate fallback."""
        if state is None:
            return
        arr = np.asarray(state, dtype=float)
        if arr.ndim != 2 or arr.shape[1] != 2:
            return
        self.vfs = arr[:, 0].tolist()
        self.violations = arr[:, 1].tolist()

    # ---- the fit -----------------------------------------------------------
    def _boundary_vf(self) -> float | None:
        """The vf where the local linear fit ``v(vf)`` crosses the utilisation
        target — the predicted constraint boundary. ``None`` when the last
        ``multipoint_window`` points cannot support a fit (too few, no volume
        spread, or a non-negative slope: more material must mean *less*
        utilisation, anything else is local noise)."""
        w = max(2, int(self.cfg.multipoint_window))
        vfs = np.asarray(self.vfs[-w:], dtype=float)
        vio = np.asarray(self.violations[-w:], dtype=float)
        if vfs.size < 2 or float(vfs.max() - vfs.min()) < 1e-9:
            return None
        slope, intercept = np.polyfit(vfs, vio, 1)
        if not (math.isfinite(slope) and math.isfinite(intercept)) \
                or slope >= 0.0:
            return None
        return float((self.cfg.utilization_target - intercept) / slope)

    # ---- the gate replacement ----------------------------------------------
    def next_target_vf(self, current_vf: float, feasible: bool,
                       violation: float | None = None) -> float:
        """Per-iteration volume target: step toward the predicted boundary,
        clamped to the gate's authority; classic gate when the fit is unusable.

        Mirrors the signature of every optimiser's ``next_target_vf`` so the
        loop can call either interchangeably.
        """
        boundary = self._boundary_vf()
        if boundary is None:
            return gate_target_vf(self.cfg, current_vf, feasible, violation)
        er = self.cfg.evolution_rate
        lo = current_vf * (1.0 - er)                       # max shrink: one ER step
        hi = current_vf * (1.0 + er * self.cfg.backoff_cap)  # max growth: the gate's cap
        target = min(max(boundary, lo), hi)
        if not feasible:
            # the measured violation outranks the model: always back off
            target = max(target, current_vf * (1.0 + er * self.cfg.backoff_floor))
        return max(self.cfg.target_volume_fraction, min(1.0, target))


def build_backoff_controller(opt_cfg) -> MultipointBackoff | None:
    """The controller for one optimiser config block, or ``None`` for the
    classic gate (``backoff_mode: gate`` — the default — and any unknown
    value, which validation flags before a run starts)."""
    if getattr(opt_cfg, "backoff_mode", "gate") == "multipoint":
        return MultipointBackoff(opt_cfg)
    return None
