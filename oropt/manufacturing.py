"""Manufacturing constraints on the alive element set.

The target part is powder-bed-fusion printed (e.g. AlSi10Mg) but may also be
cast or extruded, so the evolving BESO / level-set / TOBS / HCA topology must
stay manufacturable. :func:`apply_manufacturing` post-processes the freshly
computed alive mask each iteration, enforcing (in order, each independently
switchable and OFF by default):

1. **Minimum member size** — a morphological *open* (erode then dilate over
   element adjacency defined by shared nodes) removes thin features and
   single-element slivers narrower than the structuring element. Anti-extensive:
   it only ever removes material (never grows beyond the original alive set).
2. **Maximum member size** (OptiStruct MAXDIM) — carve bulky solid lumps so
   every alive element lies within ``max_member_layers`` adjacency hops of a
   void, punching distributed voids into thick regions (deepest/least-useful
   material first) while leaving walls of the allowed thickness.
3. **Symmetry planes** — mirror the alive set across each plane so the design is
   symmetric. Rule: *either alive ⇒ both alive* (keep an element if it or its
   mirror is alive), which enforces symmetry without over-removing; the volume
   target catches up over iterations.
4. **Casting / draw direction** — along the draw direction each column of
   elements must be free of undercuts so a die can slide out: single-sided keeps
   a solid bottom prefix (no solid above a void); two-sided keeps one contiguous
   run around a parting surface.
5. **Extrusion** — constant cross-section along an axis: elements are binned into
   prisms by their projected footprint and each prism is made uniform by a
   majority vote.
6. **Overhang / self-support** — along the build direction, forbid an alive
   element that lacks solid support within a downward cone (a simple
   self-supporting filter); the lowest layer rests on the build plate. Applied
   last, so support is judged on the near-final mask.

The ordering is deliberate (size filters, then symmetry, then the directional
demold/extrusion/overhang gates last) and is mirrored in
:class:`oropt.config.ManufacturingOpts`. Protected elements (BC/load/keep-out)
are always kept alive. The function does *not* drop disconnected islands — a
constraint can split the structure, so the caller (the loop) re-applies
:meth:`oropt.mesh.Mesh.keep_connected` with the connectivity anchor afterwards.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .mesh import Mesh

_AXIS = {"x": 0, "y": 1, "z": 2}


def manufacturing_active(opts) -> bool:
    """True iff *opts* enables at least one manufacturing constraint (else a no-op)."""
    if opts is None:
        return False
    if int(getattr(opts, "min_member_layers", 0) or 0) > 0:
        return True
    if int(getattr(opts, "max_member_layers", 0) or 0) > 0:
        return True
    if getattr(opts, "symmetry_planes", None):
        return True
    if getattr(opts, "draw_direction", None) is not None:
        return True
    if getattr(opts, "extrusion_axis", None) is not None:
        return True
    bd = getattr(opts, "build_direction", None)
    ang = float(getattr(opts, "max_overhang_angle", 0.0) or 0.0)
    if bd is not None and ang > 0.0:
        return True
    return False


def apply_manufacturing(alive: np.ndarray, mesh: Mesh, opts,
                        protected: np.ndarray | None = None,
                        sensitivity: np.ndarray | None = None) -> np.ndarray:
    """Enforce the configured manufacturing constraints on *alive*; return the
    new mask.

    ``sensitivity`` (per-element, higher = more load-bearing) is optional and used
    only by the maximum-member-size carve to remove the least-useful material
    first; when ``None`` that carve falls back to a deterministic geometric order.

    With every constraint OFF (the default) the mask is returned unchanged.
    Disconnected islands are **not** dropped here — the caller re-applies
    ``mesh.keep_connected`` with the connectivity anchor.
    """
    alive = np.asarray(alive, dtype=bool).copy()
    if not manufacturing_active(opts):
        return alive
    n = alive.size
    protected = (np.zeros(n, dtype=bool) if protected is None
                 else np.asarray(protected, dtype=bool))

    # 1. minimum member size: morphological open over shared-node adjacency
    min_layers = int(getattr(opts, "min_member_layers", 0) or 0)
    # 2. maximum member size: carve interiors deeper than the hop limit
    max_layers = int(getattr(opts, "max_member_layers", 0) or 0)
    if min_layers > 0 or max_layers > 0:
        A = _element_adjacency(mesh)
        if min_layers > 0:
            alive = _morph_open(alive, A, min_layers, protected)
        if max_layers > 0:
            alive = _max_member(alive, A, max_layers, protected, sensitivity)

    # 3. symmetry planes: enforce "either alive => both alive" across each plane
    planes = getattr(opts, "symmetry_planes", None) or []
    if planes:
        spacing = _spacing(mesh.centroids)
        for plane in planes:
            alive = _symmetrize(alive, mesh, plane, spacing)

    # 4. casting / draw direction: no undercuts per column along the draw axis
    draw = getattr(opts, "draw_direction", None)
    if draw is not None:
        alive = _draw(alive, mesh, np.asarray(draw, dtype=float),
                      bool(getattr(opts, "draw_two_sided", False)),
                      _spacing(mesh.centroids))

    # 5. extrusion: constant cross-section along the extrusion axis
    ext = getattr(opts, "extrusion_axis", None)
    if ext is not None:
        alive = _extrude(alive, mesh, np.asarray(ext, dtype=float),
                         _spacing(mesh.centroids))

    # 6. overhang / self-support along the build direction (judged last)
    bd = getattr(opts, "build_direction", None)
    ang = float(getattr(opts, "max_overhang_angle", 0.0) or 0.0)
    if bd is not None and ang > 0.0:
        alive = _self_support(alive, mesh, np.asarray(bd, dtype=float),
                              ang, protected)

    # protected elements must always survive every constraint
    alive |= protected
    return alive


# ---- minimum member size ---------------------------------------------------
def _element_adjacency(mesh: Mesh):
    """Symmetric, reflexive element-element adjacency: ``A[i, j] > 0`` iff
    elements *i* and *j* share a node (the diagonal is each element's 4 nodes).

    Built (in float, to avoid int8 overflow in the product) from the same
    element-node incidence the rest of the mesh code uses, so "neighbour" means
    exactly what ``keep_connected`` / ``protected_mask`` mean by it."""
    inc = mesh._incidence(np.arange(mesh.n_elements)).astype(np.float64)
    return (inc @ inc.T).tocsr()


def _morph_open(alive: np.ndarray, A, layers: int,
                protected: np.ndarray) -> np.ndarray:
    """Morphological open = ``layers`` erosions then ``layers`` dilations.

    Erosion turns off any alive element with a void neighbour; dilation turns a
    void element on if it has an alive neighbour. Together they delete features
    thinner than ``layers`` hops while leaving thicker bulk intact. Protected
    elements never erode (and shield their neighbours), and the result is clipped
    to the original alive set so the open only removes material.
    """
    a = alive.copy()
    for _ in range(layers):
        dead = (~a) & (~protected)
        has_dead_neighbour = A.dot(dead.astype(np.float64)) > 0.0
        a = (a & ~has_dead_neighbour) | protected
    for _ in range(layers):
        has_alive_neighbour = A.dot(a.astype(np.float64)) > 0.0
        a = a | has_alive_neighbour
    return (a & alive) | protected


# ---- maximum member size ---------------------------------------------------
def _max_member(alive: np.ndarray, A, max_layers: int, protected: np.ndarray,
                sensitivity: np.ndarray | None) -> np.ndarray:
    """Carve bulky lumps so every alive element is within ``max_layers`` hops of
    a void (OptiStruct MAXDIM).

    A void's neighbourhood is grown ``max_layers`` hops with the same matvec the
    open uses; anything still solid outside that reach is *over-limit* — too deep
    inside a lump. Each pass carves a spread (adjacency-independent) set of the
    over-limit elements — lowest ``sensitivity`` first when supplied, else in a
    fixed geometric order — so holes stay distributed and walls of the allowed
    thickness survive rather than the whole core being hollowed. Blocking a
    carved element's neighbours within a pass keeps the punched voids apart;
    recomputing between passes lets the reach spread into what was just carved,
    so it converges (each pass strictly removes material, bounded below by the
    void count) to a mask with no member thicker than the limit. Protected
    elements are never over-limit candidates, so a fully-protected lump is left
    intact (the caller re-adds protected regardless).
    """
    a = np.asarray(alive, dtype=bool).copy()
    prot = np.asarray(protected, dtype=bool)
    key = np.asarray(sensitivity, dtype=float) if sensitivity is not None else None
    indptr, indices = A.indptr, A.indices
    for _ in range(int(a.sum()) + 1):
        reached = ~a                                    # voids and their max_layers-hop reach
        for _ in range(max_layers):
            reached = reached | (A.dot(reached.astype(np.float64)) > 0.0)
        over = a & ~prot & ~reached
        if not over.any():
            break
        idx = np.flatnonzero(over)
        if key is not None:
            idx = idx[np.argsort(key[idx], kind="stable")]   # least-useful first
        blocked = np.zeros(a.size, dtype=bool)
        for e in idx:
            if blocked[e]:
                continue
            a[e] = False
            blocked[indices[indptr[e]:indptr[e + 1]]] = True  # keep holes spread out
    return a


# ---- symmetry planes -------------------------------------------------------
def _spacing(centroids: np.ndarray) -> float:
    """Characteristic element spacing: the median nearest-neighbour centroid
    distance (used as the symmetry match tolerance and overhang search scale)."""
    c = np.asarray(centroids, dtype=float)
    if c.shape[0] < 2:
        return 1.0
    tree = cKDTree(c)
    d, _ = tree.query(c, k=2)
    nn = d[:, 1]
    nn = nn[nn > 0]
    s = float(np.median(nn)) if nn.size else 1.0
    return s if s > 0 else 1.0


def _symmetrize(alive: np.ndarray, mesh: Mesh, plane,
                spacing: float) -> np.ndarray:
    """Mirror *alive* across one plane: an element is kept if it or its mirror
    is alive. Each element is paired with the element nearest its reflected
    centroid; only mutually-nearest pairs within tolerance are coupled, so an
    asymmetric mesh region is left untouched rather than mis-paired.
    """
    axis = _AXIS[str(plane["axis"]).strip().lower()]
    offset = float(plane.get("offset", 0.0) if hasattr(plane, "get")
                   else plane["offset"])
    c = mesh.centroids
    reflected = c.copy()
    reflected[:, axis] = 2.0 * offset - c[:, axis]
    dist, idx = cKDTree(c).query(reflected, k=1)
    ar = np.arange(c.shape[0])
    mutual = (idx[idx] == ar) & (dist <= 0.5 * spacing + 1e-9)
    out = np.asarray(alive, dtype=bool).copy()
    out[mutual] = out[mutual] | out[idx[mutual]]
    return out


# ---- casting / extrusion column machinery ----------------------------------
def _plane_basis(d: np.ndarray):
    """Two orthonormal vectors spanning the plane perpendicular to unit axis *d*
    (used to project centroids to a 2-D footprint for column/prism binning)."""
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

    Centroids are projected onto the plane perpendicular to *axis* and quantised
    to a ``spacing`` grid; elements sharing a grid cell form one column, returned
    as element-index arrays sorted by height along *axis* (ascending). Shared by
    casting (no-undercut-per-column) and extrusion (uniform-per-prism).
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


def _largest_run(a: np.ndarray) -> np.ndarray:
    """Boolean mask of the longest contiguous ``True`` run in 1-D *a* (ties: the
    first / lowest run). All-``False`` in returns all-``False``."""
    n = a.size
    best_start, best_len = 0, 0
    i = 0
    while i < n:
        if a[i]:
            j = i
            while j < n and a[j]:
                j += 1
            if j - i > best_len:
                best_start, best_len = i, j - i
            i = j
        else:
            i += 1
    mask = np.zeros(n, dtype=bool)
    mask[best_start:best_start + best_len] = True
    return mask


def _draw(alive: np.ndarray, mesh: Mesh, axis: np.ndarray,
          two_sided: bool, spacing: float) -> np.ndarray:
    """Remove casting undercuts along the draw *axis* (removal only).

    Single-sided: the solid must be a bottom prefix of each column — walking up,
    any alive element above the first void is an undercut and is removed. Two-
    sided: the solid must be one contiguous run around a parting surface, so all
    but the longest alive run in each column is removed.
    """
    out = np.asarray(alive, dtype=bool).copy()
    for col in _columns(mesh.centroids, axis, spacing):
        a = out[col]
        if not a.any():
            continue
        if two_sided:
            out[col[a & ~_largest_run(a)]] = False
        else:
            void_above = np.logical_or.accumulate(~a)      # True from the first void up
            out[col[a & void_above]] = False               # solid sitting above a void
    return out


def _extrude(alive: np.ndarray, mesh: Mesh, axis: np.ndarray,
             spacing: float) -> np.ndarray:
    """Force a constant cross-section along the extrusion *axis*: each prism
    (column of elements sharing a footprint) is made uniform by a majority vote
    — solid iff at least half its elements are alive (ties kept alive)."""
    out = np.asarray(alive, dtype=bool).copy()
    for col in _columns(mesh.centroids, axis, spacing):
        out[col] = out[col].mean() >= 0.5
    return out


# ---- overhang / self-support ----------------------------------------------
def _self_support(alive: np.ndarray, mesh: Mesh, build_dir: np.ndarray,
                  max_overhang_angle: float, protected: np.ndarray) -> np.ndarray:
    """Self-supporting filter along ``build_dir``.

    Sweeping elements bottom-up, an alive element is kept if it sits on the build
    plate (within one layer of the lowest height) or has an already-supported
    alive element below it within a downward cone of half-angle
    ``max_overhang_angle`` (measured from the build direction). Otherwise it is a
    floating / unsupported overhang and is forbidden. Support chains layer by
    layer, so a column standing on the plate stays; an island in mid-air does not.
    Protected elements are treated as supported anchors.
    """
    c = mesh.centroids
    u = build_dir / (np.linalg.norm(build_dir) + 1e-12)
    h = c @ u                                   # height along the build direction
    spacing = _spacing(c)
    radius = 3.0 * spacing                      # downward neighbourhood to search
    plate_tol = spacing                         # lowest layer rests on the plate
    cos_thresh = float(np.cos(np.radians(max_overhang_angle)))

    out = np.asarray(alive, dtype=bool).copy()
    supported = np.asarray(protected, dtype=bool).copy()
    plate_h = float(h[out].min()) if out.any() else float(h.min())
    tree = cKDTree(c)

    for i in np.argsort(h, kind="stable"):
        if not out[i] or supported[i]:
            continue
        if h[i] - plate_h <= plate_tol:
            supported[i] = True
            continue
        ok = False
        for j in tree.query_ball_point(c[i], radius):
            if j == i or not out[j] or not supported[j]:
                continue
            dh = h[j] - h[i]
            if dh >= 0.0:                        # supporter must be strictly below
                continue
            dist = float(np.linalg.norm(c[j] - c[i]))
            if dist <= 1e-12:
                continue
            if (-dh) / dist >= cos_thresh:       # within the downward cone
                ok = True
                break
        if ok:
            supported[i] = True
        else:
            out[i] = False
    return out
