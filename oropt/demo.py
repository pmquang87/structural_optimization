"""Synthetic demo physics: run the whole pipeline with zero OpenRadioss.

This is **not a solver**. :func:`demo_solve` fabricates a plausible response
(per-element strain-energy / von-Mises fields plus the scalar constraint
values) directly from the deck geometry, so the entire optimisation pipeline —
loop, feasibility gate, monitor, report, smoothing, GIF — can be exercised,
demonstrated and benchmarked on any machine with nothing but
``pip install -e .`` (roadmap items U2/P2).

Determinism
-----------
The output is a pure function of ``(deck geometry, alive mask, case constraint
nodes, opts)``: plain numpy arithmetic, no random numbers, no time, no
environment. Two calls with the same inputs return bit-identical arrays and
scalars, so demo runs are exactly reproducible (and usable as regression
fixtures).

Response model
--------------
* **Energy field** (the BESO-family sensitivity): per-alive-element centroids
  are scored by proximity to a synthetic *load path*. The load anchor is the
  position of the case's displacement-constraint node(s) when they resolve in
  the deck (else one end of the longest bounding-box axis); a second pseudo-BC
  anchor sits at the opposite end of the bounding box, and the segment between
  them is the load path. The field is a sum of smooth exponential falloffs —
  ``exp(-d/L)`` with ``L`` a fraction of the design bounding-box diagonal —
  from the load anchor, the BC anchor and the path segment, plus a small
  positive floor, so every alive element carries strictly positive energy
  (``Results.is_null_solve`` is False for any nonempty alive set) and ranking
  by it removes material far from the load path first.
* **Constraint response**: removing material stiffens nothing — responses grow
  as ``scale = (V0 / V_alive) ** hardening``, where ``V0`` / ``V_alive`` are the
  total and surviving design volumes computed from the real tetrahedron
  geometry (not an element-count proxy). Then ``sigma_max = sigma0 * scale``
  and every constrained node's displacement is ``disp0 * scale``, so the
  feasibility gate engages exactly as with a real solve: keep removing and the
  design eventually violates its limits.
* **Von-Mises field**: the energy-shaped field rescaled so its maximum equals
  ``sigma_max`` (mirroring a real run, where the reported peak is the max of
  the per-element field).

Degenerate inputs never raise: an empty alive set returns empty arrays with
``nan`` scalars, and a constraint node id absent from the deck reports ``nan``
in ``Results.disps`` (mirroring :func:`oropt.results.parse_vtk`).
"""
from __future__ import annotations

import numpy as np

from .results import Results
from .runner import RunResult

#: energy-field floor: keeps every alive element strictly positive so the
#: null-solve guard never trips on a live demo design.
_ENERGY_FLOOR = 1e-6


def _constraint_node_ids(case) -> list[int]:
    """The case's displacement-constraint node ids, in declared order."""
    return [int(dc.node_id) for dc in getattr(case, "disp_constraints", [])
            if getattr(dc, "node_id", None) is not None]


def _empty_results(want_nodes: list[int]) -> Results:
    """A valid Results for a design with no alive elements (nothing solved)."""
    disps = {nid: float("nan") for nid in want_nodes}
    first = want_nodes[0] if want_nodes else None
    return Results(element_ids=np.empty(0, dtype=np.int64),
                   energy=np.empty(0, dtype=float),
                   vonmises=np.empty(0, dtype=float),
                   sigma_max=float("nan"), disp=float("nan"),
                   disp_node_id=first, disps=disps)


def demo_solve(deck, alive, case, opts) -> tuple[RunResult, Results]:
    """Fabricate one iteration's solver response from the deck geometry.

    Parameters mirror one real solve of one load case: *deck* is the parsed
    :class:`oropt.deck.Deck`, *alive* the (N,) bool mask over its design
    elements, *case* the :class:`oropt.config.ResolvedCase` being "solved"
    (only its ``disp_constraints`` node ids matter here), and *opts* a
    duck-typed options object read via ``getattr`` with defaults —
    ``sigma0`` (peak von-Mises at full volume, 100.0), ``disp0``
    (constrained-node displacement at full volume, 0.5) and ``hardening``
    (response-growth exponent, 1.5) — so a plain namespace works.

    Returns ``(RunResult, Results)`` shaped exactly like a real
    solve+extract: ``Results.element_ids`` holds only the ALIVE design
    elements (a real animation only contains surviving elements) with
    ``energy`` / ``vonmises`` aligned to it. See the module docstring for the
    response model and the determinism guarantee.
    """
    alive = np.asarray(alive, dtype=bool)
    if alive.shape != deck.elem_ids.shape:
        raise ValueError("alive mask must align with deck.elem_ids")
    sigma0 = float(getattr(opts, "sigma0", 100.0))
    disp0 = float(getattr(opts, "disp0", 0.5))
    hardening = float(getattr(opts, "hardening", 1.5))

    run = RunResult(ok=True, stage="ok", message="demo backend (synthetic physics)")
    want_nodes = _constraint_node_ids(case)
    if alive.size == 0 or not alive.any():
        return run, _empty_results(want_nodes)

    # ---- design-part geometry (full mesh, so anchors/scales are stable
    # across iterations regardless of which elements are currently alive) ----
    order = np.argsort(deck.node_ids)
    sorted_ids = deck.node_ids[order]
    rows = order[np.searchsorted(sorted_ids, deck.elem_conn)]   # (N,4) node rows
    xyz = deck.node_xyz[rows]                                   # (N,4,3)
    centroids = xyz.mean(axis=1)
    a, b, c, d = xyz[:, 0], xyz[:, 1], xyz[:, 2], xyz[:, 3]
    volumes = np.abs(np.einsum("ij,ij->i", a - d, np.cross(b - d, c - d))) / 6.0

    pts = deck.node_xyz[np.unique(rows)]        # only nodes some design element uses
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    L = 0.25 * diag if diag > 0.0 else 1.0      # falloff length ~ quarter diagonal

    # ---- anchors: load = constraint node(s) if resolvable, else one end of
    # the longest bbox axis; pseudo-BC = the opposite bbox end --------------
    load_pts = []
    for nid in want_nodes:
        k = int(np.searchsorted(sorted_ids, nid))
        if k < sorted_ids.size and sorted_ids[k] == nid:
            load_pts.append(deck.node_xyz[order[k]])
    axis = int(np.argmax(hi - lo))
    mid = (lo + hi) / 2.0
    end_lo, end_hi = mid.copy(), mid.copy()
    end_lo[axis], end_hi[axis] = lo[axis], hi[axis]
    load_xyz = (np.asarray(load_pts, dtype=float) if load_pts
                else end_hi[None, :])
    load_c = load_xyz.mean(axis=0)
    bc_pt = (end_lo if np.linalg.norm(end_lo - load_c)
             >= np.linalg.norm(end_hi - load_c) else end_hi)

    # ---- energy field over the alive elements ------------------------------
    ca = centroids[alive]
    d_load = np.linalg.norm(ca[:, None, :] - load_xyz[None, :, :], axis=2).min(axis=1)
    d_bc = np.linalg.norm(ca - bc_pt, axis=1)
    seg = load_c - bc_pt
    denom = float(seg @ seg)
    t = (np.clip((ca - bc_pt) @ seg / denom, 0.0, 1.0) if denom > 0.0
         else np.zeros(ca.shape[0]))
    d_path = np.linalg.norm(ca - (bc_pt + t[:, None] * seg), axis=1)
    energy = (np.exp(-d_load / L)
              + 0.35 * np.exp(-d_bc / (0.6 * L))
              + 0.50 * np.exp(-d_path / (0.5 * L))
              + _ENERGY_FLOOR)

    # ---- constraint response: removal RAISES stress & displacement ---------
    v0 = float(volumes.sum())
    v_alive = float(volumes[alive].sum())
    if v0 > 0.0 and v_alive > 0.0:
        scale = (v0 / v_alive) ** hardening
    else:   # degenerate zero-volume tets: fall back to the element-count proxy
        scale = (alive.size / float(alive.sum())) ** hardening
    sigma_max = sigma0 * scale
    vonmises = energy * (sigma_max / float(energy.max()))

    deck_ids = frozenset(int(v) for v in deck.node_ids)
    disps = {nid: (disp0 * scale if nid in deck_ids else float("nan"))
             for nid in want_nodes}
    first = want_nodes[0] if want_nodes else None
    disp = disps.get(first, float("nan")) if first is not None else float("nan")

    return run, Results(element_ids=deck.elem_ids[alive], energy=energy,
                        vonmises=vonmises, sigma_max=sigma_max, disp=disp,
                        disp_node_id=first, disps=disps)
