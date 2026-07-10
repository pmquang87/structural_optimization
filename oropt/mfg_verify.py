"""Independent geometric audit of manufacturing constraints on the FINAL design.

This module is an **independent verifier**, not a re-run of the enforcement code
in :mod:`oropt.manufacturing`. :func:`oropt.manufacturing.apply_manufacturing`
*enforces* the printability constraints each iteration, and its tests check that
the enforcement functions run and mutate the mask the way they intend. Nothing,
however, independently *measures* that the geometry that actually ships satisfies
the configured constraints. Two gaps make that dangerous:

* a bug in the enforcement geometry (``_self_support``'s support cone,
  ``_draw``'s undercut walk, ``_extrude``'s prism vote) would be invisible — the
  enforcement tests would still pass because they assert the code's own behaviour,
  not an external ground truth; and
* the optimisation loop re-applies :meth:`oropt.mesh.Mesh.keep_connected`
  *after* manufacturing, which only ever removes elements but can nonetheless
  reintroduce a violation (e.g. drop the base of a column, leaving an
  now-unsupported overhang or an undercut).

So every measurement here is **reimplemented from raw geometry** (centroids,
volumes, shared-node element adjacency). It deliberately does NOT call
``_self_support`` / ``_draw`` / ``_extrude`` to check their own output — that
would only confirm the enforcement is self-consistent, which is exactly what we
cannot trust. The measurement definitions below are written to mirror the
*intent* of each constraint independently:

Measurement definitions
------------------------
* **Overhang / self-support** — for each alive element (not on the build plate)
  measure the smallest angle-from-the-build-direction to any alive supporting
  element strictly below it within a ``3 * spacing`` neighbourhood. That angle is
  the element's overhang angle (0 = supporter directly beneath, 90 = fully
  horizontal / no supporter). An element is *unsupported* when its best supporter
  exceeds the allowed cone half-angle, or when it has no supporter below at all.
  Reported as ``(worst_angle_deg, n_unsupported)``.
* **Minimum member size** — a granulometry: the largest structuring-element hop
  count ``L`` for which a morphological *open* (``L`` erosions then ``L``
  dilations over shared-node adjacency) still keeps an element is that element's
  member size. The minimum over all alive elements is the thinnest member's
  thickness, in adjacency hops (0 = a sliver that does not survive a single-hop
  open). Directly comparable to ``min_member_layers``.
* **Maximum member size** (MAXDIM) — the graph-hop distance from each alive
  element to the nearest void; the maximum over the alive set is the design's
  thickest member. A fully solid body (no void) reports ``n_elements`` (unbounded).
  Comparable to ``max_member_layers``.
* **Casting / draw** — bin elements into columns along the draw axis; a column is
  undercut-free when its alive elements form a single bottom-anchored contiguous
  run (single-sided) or a single contiguous run anywhere (two-sided). Reported as
  the number of columns that contain an undercut.
* **Extrusion** — bin elements into prisms by footprint; a prism is uniform when
  it is all-alive or all-void. Reported as the count and fraction of prisms that
  mix alive and void along the axis.
* **Symmetry** — mirror each element's centroid across the plane, pair it with its
  mutually-nearest reflected partner (within half a median element spacing) and
  report the fraction of paired elements whose partner has a different alive state.

Everything is numpy/scipy only and deterministic (stable sorts, no RNG).

The public surface is the five measurement functions, the bonus
:func:`max_member_thickness`, and :func:`verify`, which reads a duck-typed
manufacturing-settings object (mirroring :class:`oropt.config.ManufacturingOpts`)
or a plain dict and returns an :class:`MfgReport` with per-check pass/fail,
measured-vs-limit, a ``warnings`` list and an ``ok`` flag.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

_AXIS = {"x": 0, "y": 1, "z": 2}


# --------------------------------------------------------------------------- #
# independent geometry helpers (reimplemented, NOT imported from manufacturing)
# --------------------------------------------------------------------------- #
def _median_spacing(centroids: np.ndarray) -> float:
    """Characteristic element spacing: median nearest-neighbour centroid distance.

    Reimplemented locally (rather than importing ``manufacturing._spacing``) so
    the audit shares no code with the thing it audits."""
    c = np.asarray(centroids, dtype=float)
    if c.shape[0] < 2:
        return 1.0
    d, _ = cKDTree(c).query(c, k=2)
    nn = d[:, 1]
    nn = nn[nn > 0]
    s = float(np.median(nn)) if nn.size else 1.0
    return s if s > 0 else 1.0


def _adjacency(mesh) -> sparse.csr_matrix:
    """Symmetric, reflexive shared-node element adjacency (``A[i, j] > 0`` iff
    elements *i* and *j* share a node). Built directly from ``mesh.conn_rows`` so
    it does not depend on ``manufacturing._element_adjacency``."""
    n = mesh.n_elements
    rows = np.repeat(np.arange(n), mesh.conn_rows.shape[1])
    cols = np.asarray(mesh.conn_rows).ravel()
    data = np.ones(rows.size, dtype=np.float64)
    inc = sparse.coo_matrix((data, (rows, cols)),
                            shape=(n, int(mesh.n_nodes))).tocsr()
    return (inc @ inc.T).tocsr()


def _erode(a: np.ndarray, A: sparse.csr_matrix) -> np.ndarray:
    """One erosion: drop any alive element that has a void (non-alive) neighbour."""
    dead = ~a
    return a & ~(A.dot(dead.astype(np.float64)) > 0.0)


def _dilate(a: np.ndarray, A: sparse.csr_matrix) -> np.ndarray:
    """One dilation: turn on any element with an alive neighbour."""
    return A.dot(a.astype(np.float64)) > 0.0


def _open(alive: np.ndarray, A: sparse.csr_matrix, layers: int) -> np.ndarray:
    """Morphological open: ``layers`` erosions then ``layers`` dilations, clipped
    back to the original alive set (anti-extensive)."""
    a = alive.copy()
    for _ in range(layers):
        a = _erode(a, A)
    for _ in range(layers):
        a = _dilate(a, A)
    return a & alive


def _plane_basis(d: np.ndarray):
    """Two orthonormal vectors spanning the plane perpendicular to unit axis *d*."""
    ref = (np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.9
           else np.array([0.0, 1.0, 0.0]))
    e1 = np.cross(d, ref)
    e1 = e1 / (np.linalg.norm(e1) + 1e-12)
    e2 = np.cross(d, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-12)
    return e1, e2


def _columns(centroids: np.ndarray, axis: np.ndarray,
             spacing: float) -> list[np.ndarray]:
    """Bin elements into columns/prisms along *axis*.

    Centroids are projected onto the plane perpendicular to *axis*, quantised to a
    ``spacing`` grid; elements in one grid cell form a column, returned as
    element-index arrays sorted ascending by height along *axis*. Reimplemented
    from ``manufacturing._columns`` so casting/extrusion are audited independently.
    """
    c = np.asarray(centroids, dtype=float)
    d = np.asarray(axis, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-12)
    e1, e2 = _plane_basis(d)
    h = c @ d
    key = np.stack([np.round((c @ e1) / spacing).astype(np.int64),
                    np.round((c @ e2) / spacing).astype(np.int64)], axis=1)
    order = np.lexsort((key[:, 1], key[:, 0]))
    ks = key[order]
    cols: list[np.ndarray] = []
    start = 0
    for i in range(1, len(order) + 1):
        if i == len(order) or bool((ks[i] != ks[start]).any()):
            grp = order[start:i]
            cols.append(grp[np.argsort(h[grp], kind="stable")])
            start = i
    return cols


def _plane_axis_offset(plane) -> tuple[int, float]:
    """``(axis_index, offset)`` from a symmetry-plane dict/object."""
    if hasattr(plane, "get"):
        axis = plane.get("axis")
        offset = plane.get("offset", 0.0)
    else:
        axis = plane["axis"] if isinstance(plane, dict) else getattr(plane, "axis")
        offset = (plane.get("offset", 0.0) if isinstance(plane, dict)
                  else getattr(plane, "offset", 0.0))
    return _AXIS[str(axis).strip().lower()], float(offset if offset is not None else 0.0)


# --------------------------------------------------------------------------- #
# 1. overhang / self-support
# --------------------------------------------------------------------------- #
def max_overhang_violation(mesh, alive: np.ndarray, build_dir,
                           max_angle_deg: float,
                           protected: np.ndarray | None = None
                           ) -> tuple[float, int]:
    """Measure overhang self-support along ``build_dir``.

    For each alive, non-protected element above the build plate, find the alive
    element strictly below it (within ``3 * spacing``) that subtends the smallest
    angle from the build direction — that is the element's overhang angle. The
    element is *unsupported* when that best angle exceeds ``max_angle_deg`` or when
    no supporter exists below it (angle reported as 90). Protected and on-plate
    elements are treated as supported anchors (angle 0) and never counted as
    violations, but a protected/on-plate alive element can still *support* others.

    Returns ``(worst_angle_deg, n_unsupported)`` — the worst overhang angle present
    anywhere in the alive set, and the number of unsupported elements. A design
    that satisfies the constraint has ``n_unsupported == 0`` and
    ``worst_angle_deg <= max_angle_deg``.
    """
    alive = np.asarray(alive, dtype=bool)
    n = alive.size
    protected = (np.zeros(n, dtype=bool) if protected is None
                 else np.asarray(protected, dtype=bool))
    c = np.asarray(mesh.centroids, dtype=float)
    if not alive.any():
        return 0.0, 0

    u = np.asarray(build_dir, dtype=float)
    u = u / (np.linalg.norm(u) + 1e-12)
    h = c @ u
    spacing = _median_spacing(c)
    radius = 3.0 * spacing
    plate_tol = spacing
    plate_h = float(h[alive].min())
    cos_thresh = float(np.cos(np.radians(max_angle_deg)))
    tree = cKDTree(c)

    worst = 0.0
    n_unsupported = 0
    for i in np.flatnonzero(alive):
        if protected[i]:
            continue
        if h[i] - plate_h <= plate_tol:          # rests on the build plate
            continue
        best_cos = -1.0                          # cos(angle from vertical); larger = better
        for j in tree.query_ball_point(c[i], radius):
            if j == i or not alive[j]:
                continue
            dh = h[i] - h[j]                     # >0 iff j is below i
            if dh <= 0.0:
                continue
            dist = float(np.linalg.norm(c[j] - c[i]))
            if dist <= 1e-12:
                continue
            cosang = dh / dist
            if cosang > best_cos:
                best_cos = cosang
        if best_cos < -0.5:                      # nothing below -> floating overhang
            angle = 90.0
            n_unsupported += 1
        else:
            best_cos = min(1.0, best_cos)
            angle = float(np.degrees(np.arccos(best_cos)))
            if best_cos + 1e-9 < cos_thresh:     # outside the allowed cone
                n_unsupported += 1
        if angle > worst:
            worst = angle
    return worst, n_unsupported


# --------------------------------------------------------------------------- #
# 2. minimum member size (granulometry)
# --------------------------------------------------------------------------- #
def min_member_thickness(mesh, alive: np.ndarray) -> int:
    """Thinnest member thickness, in shared-node adjacency hops (granulometry).

    Each alive element's *member size* is the largest ``L`` for which an
    ``L``-layer morphological open keeps it alive; the return value is the minimum
    over all alive elements — the thinnest feature. ``0`` means a sliver that a
    single-hop open removes. A fully solid isolated body (no void to erode into)
    saturates at ``n_elements``. Compare with ``min_member_layers``: a measured
    value below the configured layer count means a too-thin member is present.
    """
    alive = np.asarray(alive, dtype=bool)
    if not alive.any():
        return 0
    A = _adjacency(mesh)
    cap = int(alive.size)
    size = np.zeros(alive.size, dtype=np.int64)
    L = 1
    while L <= cap:
        o = _open(alive, A, L)
        if not o.any():
            break
        size[o] = L
        L += 1
    return int(size[alive].min())


# --------------------------------------------------------------------------- #
# bonus: maximum member size (MAXDIM)
# --------------------------------------------------------------------------- #
def max_member_thickness(mesh, alive: np.ndarray) -> int:
    """Thickest member thickness = the max graph-hop distance from an alive
    element to the nearest void. A fully solid body (no void) returns
    ``n_elements`` (unbounded). Compare with ``max_member_layers``: a measured
    value above the configured limit means a lump too far from any void remains."""
    alive = np.asarray(alive, dtype=bool)
    if not alive.any():
        return 0
    A = _adjacency(mesh)
    reach = ~alive
    if not reach.any():                          # no void anywhere -> unbounded
        return int(alive.size)
    dist = np.zeros(alive.size, dtype=np.int64)
    remaining = alive & ~reach
    k = 0
    cap = int(alive.size)
    while remaining.any() and k <= cap:
        k += 1
        new_reach = _dilate(reach, A)
        newly = new_reach & ~reach & alive
        dist[newly] = k
        reach = new_reach
        remaining = alive & ~reach
    if remaining.any():                          # alive island with no void path
        dist[remaining] = cap
    return int(dist[alive].max())


# --------------------------------------------------------------------------- #
# 3. casting / draw direction
# --------------------------------------------------------------------------- #
def draw_undercut_violation(mesh, alive: np.ndarray, draw_dir,
                            two_sided: bool) -> int:
    """Count columns along ``draw_dir`` that contain a casting undercut.

    Single-sided: the alive elements of a column must be a contiguous run anchored
    at the base (a bottom prefix) — any alive element above a void is an undercut.
    Two-sided: the alive elements must form a single contiguous run anywhere
    (one solid band around a parting surface) — a second, separated run is an
    undercut. Returns the number of columns that are not undercut-free.
    """
    alive = np.asarray(alive, dtype=bool)
    c = np.asarray(mesh.centroids, dtype=float)
    spacing = _median_spacing(c)
    axis = np.asarray(draw_dir, dtype=float)
    bad = 0
    for col in _columns(c, axis, spacing):
        a = alive[col]
        if not a.any():
            continue
        n_runs = int(np.count_nonzero(np.diff(a.astype(np.int8)) == 1)) + int(a[0])
        if two_sided:
            if n_runs > 1:                       # more than one contiguous band
                bad += 1
        else:
            first_void = int(np.argmax(~a)) if (~a).any() else a.size
            if a[first_void:].any():             # solid sitting above the first void
                bad += 1
    return bad


# --------------------------------------------------------------------------- #
# 4. extrusion
# --------------------------------------------------------------------------- #
def extrusion_nonuniformity(mesh, alive: np.ndarray,
                            extrude_axis) -> tuple[int, float]:
    """Count prisms that are NOT uniform along ``extrude_axis``.

    Elements are binned into prisms by their footprint perpendicular to the axis;
    a prism is uniform when all of its elements are alive or all are void. Returns
    ``(n_nonuniform, fraction_nonuniform)`` over the prisms present.
    """
    alive = np.asarray(alive, dtype=bool)
    c = np.asarray(mesh.centroids, dtype=float)
    spacing = _median_spacing(c)
    axis = np.asarray(extrude_axis, dtype=float)
    cols = _columns(c, axis, spacing)
    n_prisms = len(cols)
    n_bad = 0
    for col in cols:
        a = alive[col]
        if a.any() and not a.all():
            n_bad += 1
    frac = (n_bad / n_prisms) if n_prisms else 0.0
    return n_bad, float(frac)


# --------------------------------------------------------------------------- #
# 5. symmetry
# --------------------------------------------------------------------------- #
def symmetry_residual(mesh, alive: np.ndarray, plane) -> float:
    """Fraction of paired elements whose mirror partner has a different alive state.

    Each element is reflected across *plane* and paired with the element nearest
    its reflection; only mutually-nearest pairs within half a median spacing are
    counted (an asymmetric region is simply ignored, matching how the enforcement
    couples pairs). ``0.0`` = perfectly symmetric over the paired region; a
    positive value measures the residual asymmetry the design still carries.
    """
    alive = np.asarray(alive, dtype=bool)
    c = np.asarray(mesh.centroids, dtype=float)
    axis, offset = _plane_axis_offset(plane)
    reflected = c.copy()
    reflected[:, axis] = 2.0 * offset - c[:, axis]
    spacing = _median_spacing(c)
    dist, idx = cKDTree(c).query(reflected, k=1)
    ar = np.arange(c.shape[0])
    mutual = (idx[idx] == ar) & (dist <= 0.5 * spacing + 1e-9)
    n_pairs = int(mutual.sum())
    if n_pairs == 0:
        return 0.0
    mismatch = mutual & (alive != alive[idx])
    return float(int(mismatch.sum()) / n_pairs)


# --------------------------------------------------------------------------- #
# top-level report
# --------------------------------------------------------------------------- #
@dataclass
class CheckResult:
    """One manufacturing constraint's audit outcome."""
    name: str                 # constraint id, e.g. "overhang", "symmetry[x@0.0]"
    active: bool              # was the constraint configured (non-null/non-zero)?
    passed: bool              # did the final geometry satisfy it?
    measured: float           # the measured quantity (see `detail` for units)
    limit: float              # the configured limit it is compared against
    detail: str = ""          # human-readable measured-vs-limit description


@dataclass
class MfgReport:
    """Result of :func:`verify`: per-check outcomes, warnings and an overall flag."""
    checks: list[CheckResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    ok: bool = True

    def failed(self) -> list[CheckResult]:
        """The active checks that did not pass."""
        return [c for c in self.checks if c.active and not c.passed]


def _get(opts, name, default=None):
    """Duck-typed field access: works for a dict or any object with attributes."""
    if opts is None:
        return default
    if isinstance(opts, dict):
        return opts.get(name, default)
    return getattr(opts, name, default)


def verify(mesh, alive: np.ndarray, opts_like,
           protected: np.ndarray | None = None) -> MfgReport:
    """Independently audit the FINAL *alive* mask against the manufacturing
    settings in *opts_like*.

    *opts_like* mirrors :class:`oropt.config.ManufacturingOpts` — pass
    ``cfg.manufacturing`` directly, or an equivalent plain dict; both are read by
    duck-typing (``min_member_layers``, ``max_member_layers``, ``symmetry_planes``,
    ``draw_direction``, ``draw_two_sided``, ``extrusion_axis``, ``build_direction``,
    ``max_overhang_angle``). Only the constraints that are actually enabled
    (non-null / non-zero) are measured. Returns an :class:`MfgReport`; ``ok`` is
    True iff every active check passed.
    """
    alive = np.asarray(alive, dtype=bool)
    report = MfgReport()

    # 1. minimum member size
    min_layers = int(_get(opts_like, "min_member_layers", 0) or 0)
    if min_layers > 0:
        measured = min_member_thickness(mesh, alive)
        passed = measured >= min_layers
        report.checks.append(CheckResult(
            "min_member", True, passed, float(measured), float(min_layers),
            f"thinnest member {measured} hops (>= {min_layers} required)"))

    # 2. maximum member size
    max_layers = int(_get(opts_like, "max_member_layers", 0) or 0)
    if max_layers > 0:
        measured = max_member_thickness(mesh, alive)
        passed = measured <= max_layers
        report.checks.append(CheckResult(
            "max_member", True, passed, float(measured), float(max_layers),
            f"thickest member {measured} hops-to-void (<= {max_layers} allowed)"))

    # 3. symmetry planes
    for plane in (_get(opts_like, "symmetry_planes", None) or []):
        axis, offset = _plane_axis_offset(plane)
        residual = symmetry_residual(mesh, alive, plane)
        passed = residual <= 1e-9
        name = f"symmetry[{'xyz'[axis]}@{offset:g}]"
        report.checks.append(CheckResult(
            name, True, passed, residual, 0.0,
            f"asymmetry residual {residual:.4f} (0 required)"))

    # 4. casting / draw direction
    draw = _get(opts_like, "draw_direction", None)
    if draw is not None:
        two_sided = bool(_get(opts_like, "draw_two_sided", False))
        n_bad = draw_undercut_violation(mesh, alive, np.asarray(draw, dtype=float),
                                        two_sided)
        passed = n_bad == 0
        report.checks.append(CheckResult(
            "draw", True, passed, float(n_bad), 0.0,
            f"{n_bad} column(s) with an undercut "
            f"({'two-sided' if two_sided else 'single-sided'})"))

    # 5. extrusion
    ext = _get(opts_like, "extrusion_axis", None)
    if ext is not None:
        n_bad, frac = extrusion_nonuniformity(mesh, alive,
                                              np.asarray(ext, dtype=float))
        passed = n_bad == 0
        report.checks.append(CheckResult(
            "extrusion", True, passed, float(n_bad), 0.0,
            f"{n_bad} non-uniform prism(s) ({frac:.1%} of prisms)"))

    # 6. overhang / self-support
    bd = _get(opts_like, "build_direction", None)
    ang = float(_get(opts_like, "max_overhang_angle", 0.0) or 0.0)
    if bd is not None and ang > 0.0:
        worst, n_unsup = max_overhang_violation(
            mesh, alive, np.asarray(bd, dtype=float), ang, protected)
        passed = n_unsup == 0
        report.checks.append(CheckResult(
            "overhang", True, passed, worst, ang,
            f"{n_unsup} unsupported element(s); worst overhang "
            f"{worst:.1f} deg (<= {ang:g} deg allowed)"))

    report.ok = all(c.passed for c in report.checks if c.active)
    report.warnings = [f"manufacturing check '{c.name}' FAILED: {c.detail}"
                       for c in report.failed()]
    return report
