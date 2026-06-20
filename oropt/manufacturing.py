"""Additive-manufacturing (AM) printability constraints on the alive element set.

The target part is powder-bed-fusion printed (e.g. AlSi10Mg), so the evolving
BESO / level-set topology must stay manufacturable. :func:`apply_manufacturing`
post-processes the freshly computed alive mask each iteration, enforcing (in
order, each independently switchable and OFF by default):

1. **Minimum member size** — a morphological *open* (erode then dilate over
   element adjacency defined by shared nodes) removes thin features and
   single-element slivers narrower than the structuring element. Anti-extensive:
   it only ever removes material (never grows beyond the original alive set).
2. **Symmetry planes** — mirror the alive set across each plane so the design is
   symmetric. Rule: *either alive ⇒ both alive* (keep an element if it or its
   mirror is alive), which enforces symmetry without over-removing; the volume
   target catches up over iterations.
3. **Overhang / self-support** — along the build direction, forbid an alive
   element that lacks solid support within a downward cone (a simple
   self-supporting filter); the lowest layer rests on the build plate.

Protected elements (BC/load/keep-out) are always kept alive. The function does
*not* drop disconnected islands — a constraint can split the structure, so the
caller (the loop) re-applies :meth:`oropt.mesh.Mesh.keep_connected` with the
connectivity anchor afterwards.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .mesh import Mesh

_AXIS = {"x": 0, "y": 1, "z": 2}


def manufacturing_active(opts) -> bool:
    """True iff *opts* enables at least one AM constraint (else a no-op)."""
    if opts is None:
        return False
    if int(getattr(opts, "min_member_layers", 0) or 0) > 0:
        return True
    if getattr(opts, "symmetry_planes", None):
        return True
    bd = getattr(opts, "build_direction", None)
    ang = float(getattr(opts, "max_overhang_angle", 0.0) or 0.0)
    if bd is not None and ang > 0.0:
        return True
    return False


def apply_manufacturing(alive: np.ndarray, mesh: Mesh, opts,
                        protected: np.ndarray | None = None) -> np.ndarray:
    """Enforce the configured AM constraints on *alive*; return the new mask.

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
    layers = int(getattr(opts, "min_member_layers", 0) or 0)
    if layers > 0:
        alive = _morph_open(alive, _element_adjacency(mesh), layers, protected)

    # 2. symmetry planes: enforce "either alive => both alive" across each plane
    planes = getattr(opts, "symmetry_planes", None) or []
    if planes:
        spacing = _spacing(mesh.centroids)
        for plane in planes:
            alive = _symmetrize(alive, mesh, plane, spacing)

    # 3. overhang / self-support along the build direction
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
