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

The *load-case weights* — LS-TaSC's second global variable set — are handled
by :class:`WeightController` below (opt-in via ``run.adaptive_weights``); the
per-case weights stay as configured unless it is enabled.
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


class WeightController:
    """Adaptive per-load-case weights — LS-TaSC's second global variable set.

    In a multi-load run the objective is a weighted sum of the per-case
    sensitivities (:func:`oropt.beso.combine_sensitivity`); with *fixed* weights
    the case that is easiest to satisfy dominates the material budget and a
    harder case's load path can starve and collapse (the documented multi-load
    failure mode). LS-TaSC instead treats the weights as *global* variables
    driven to satisfy the constraints from data the run already produces
    (Roux, "The LS-TaSC multipoint method…"; survey in
    ``docs/topology_sota_2026.md`` §3).

    This is that idea reduced to a robust proportional rule: each iteration take
    every case's worst constraint-utilisation ratio ``v_i = value/limit`` (the
    same per-case number the feasibility check computes, ``v_i <= 1`` feasible)
    and nudge the weights toward *equal utilisation* — the case furthest over
    its limit gains weight (keep more of the material serving its load path),
    the case with the most slack loses it::

        w_i <- w_i * (v_i / v_mean) ** gain      (multiplicative, gain small)

    then the weights are renormalised to preserve the original weight sum (so
    the *combined* sensitivity scale — and thus the history blend — is
    essentially unchanged; only the split between cases moves) and finally
    clamped to ``[base_i / bound, base_i * bound]`` so no case is ever starved
    or runs away (the clamp is the last step — a hard bound — so it can leave
    the sum slightly off the target when it actually fires).

    The update runs only when it has a usable signal — at least two cases and
    at least two finite, positive ``v_i`` (cases with no configured limit, or a
    diverged/blank solve, are held at their current weight and excluded from
    the mean). Whenever the signal is unusable the weights are returned
    untouched, so enabling the controller can never be worse-defined than the
    fixed weights. The weights are the controller's only state; the loop
    checkpoints them (``weights`` in ``checkpoint.npz``) so a ``--resume``
    continues the adaptation instead of resetting to the configured split.
    """

    def __init__(self, base_weights, gain: float = 0.5, bound: float = 4.0):
        self.base = [float(w) for w in base_weights]
        self.weights = list(self.base)
        self.gain = float(gain)
        self.bound = max(1.0, float(bound))
        self._sum = sum(self.base) or float(len(self.base))

    def update(self, violations) -> list[float]:
        """Step the weights from this iteration's per-case violation ratios and
        return the new weight list. ``violations[i]`` is ``None``/non-finite for
        a case with no usable signal; such cases keep their weight and sit out
        the mean. A no-op (returns the current weights) when fewer than two
        cases carry a usable, positive ratio."""
        v = [float(x) if (x is not None and math.isfinite(x) and x > 0.0)
             else None for x in violations]
        usable = [x for x in v if x is not None]
        if len(usable) < 2:
            return list(self.weights)
        v_mean = sum(usable) / len(usable)
        if v_mean <= 0.0:
            return list(self.weights)
        new = list(self.weights)
        for i, vi in enumerate(v):
            if vi is None:
                continue
            new[i] = self.weights[i] * (vi / v_mean) ** self.gain
        # renormalise to preserve the original total weight (combined-sensitivity
        # scale essentially unchanged; only the per-case split moves)...
        s = sum(new)
        if s > 0.0:
            new = [w * self._sum / s for w in new]
        # ...then a final hard clamp so no case is ever starved or runs away.
        new = [min(self.base[i] * self.bound, max(self.base[i] / self.bound, w))
               for i, w in enumerate(new)]
        self.weights = new
        return list(self.weights)

    def state(self) -> np.ndarray:
        """The current weights as a 1-D array for the checkpoint."""
        return np.asarray(self.weights, dtype=float)

    def restore(self, state) -> None:
        """Reload weights saved by :meth:`state` (a resume). A length mismatch
        or malformed array is ignored — adaptation then restarts from base."""
        if state is None:
            return
        arr = np.asarray(state, dtype=float).ravel()
        if arr.size != len(self.base) or not np.all(np.isfinite(arr)):
            return
        self.weights = arr.tolist()


def build_weight_controller(cfg, base_weights) -> WeightController | None:
    """A :class:`WeightController` when ``run.adaptive_weights`` is on and the
    run has at least two load cases; ``None`` otherwise (the fixed-weight path,
    byte-identical to before)."""
    if not getattr(cfg.run, "adaptive_weights", False):
        return None
    if len(list(base_weights)) < 2:
        return None
    return WeightController(base_weights,
                            gain=getattr(cfg.run, "adaptive_weight_gain", 0.5),
                            bound=getattr(cfg.run, "adaptive_weight_bound", 4.0))
