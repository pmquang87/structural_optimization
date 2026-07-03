"""Geometry and graph operations on the design-part mesh.

Built from a :class:`~oropt.deck.Deck`, this provides everything the BESO core
needs that is purely geometric/topological: element centroids and volumes, the
spatial sensitivity-filter matrix, element connectivity (to drop floating
islands), and the set of protected elements that must never be deleted
(boundary-condition, contact and load regions).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .deck import Deck


# --------------------------------------------------------------------------- #
# growth-region geometry (centroid-in-primitive tests + 3D-overlay outlines)
#
# Pure, numpy-only helpers shared by the candidate selection (Mesh.in_boxes_mask
# -> the loop) and the Monitor/report 3D overlay. They duck-type a
# :class:`~oropt.config.GrowthBox` (its shape / bounds / centre / radius / axis
# fields), so this module never has to import config.
# --------------------------------------------------------------------------- #

# The 12 edges of the unit box: two of the 8 corners share an edge iff their
# indices (x_bit<<2 | y_bit<<1 | z_bit, matching the corner order in box_corners)
# differ in exactly one bit. Frame-invariant, so it labels an oriented box too.
_BOX_EDGES: tuple[tuple[int, int], ...] = tuple(
    (i, j) for i in range(8) for j in range(i + 1, 8)
    if bin(i ^ j).count("1") == 1)


def local_frame_basis(box):
    """``(origin(3,), R(3,3))`` for a box's oriented local frame, or ``None``.

    ``R``'s rows are the orthonormal basis ``e1, e2, e3`` from Gram-Schmidt of the
    box's ``x_axis`` (``e1``) and ``xy_axis`` (spanning ``e1``-``e2``), with
    ``e3 = e1 x e2`` (LS-DYNA ``*DEFINE_BOX_LOCAL`` skew system). Local coordinates
    of a point ``c`` are ``(c - origin) @ R.T``. Returns ``None`` when the box has
    no frame or the two axes are zero-length/parallel (a degenerate frame the
    caller treats as world-aligned)."""
    if not box.has_local_frame():
        return None
    e1 = np.asarray(box.x_axis, dtype=float)
    b = np.asarray(box.xy_axis, dtype=float)
    n1 = float(np.linalg.norm(e1))
    if n1 == 0.0:
        return None
    e1 = e1 / n1
    b = b - np.dot(b, e1) * e1
    n2 = float(np.linalg.norm(b))
    if n2 == 0.0:                       # xy_axis parallel to x_axis -> degenerate
        return None
    e2 = b / n2
    e3 = np.cross(e1, e2)
    origin = (np.asarray(box.origin, dtype=float) if box.origin is not None
              else np.zeros(3))
    return origin, np.vstack([e1, e2, e3])


def _box_member(c: np.ndarray, box) -> np.ndarray:
    frame = local_frame_basis(box)
    if frame is not None:
        origin, R = frame
        c = (c - origin) @ R.T          # centroids in the box's local frame
    return ((c[:, 0] >= box.x_min) & (c[:, 0] <= box.x_max)
            & (c[:, 1] >= box.y_min) & (c[:, 1] <= box.y_max)
            & (c[:, 2] >= box.z_min) & (c[:, 2] <= box.z_max))


def _sphere_member(c: np.ndarray, box) -> np.ndarray:
    d = c - np.array([box.cx, box.cy, box.cz], dtype=float)
    return np.einsum("ij,ij->i", d, d) <= float(box.radius) ** 2


def _cylinder_member(c: np.ndarray, box) -> np.ndarray:
    p1 = np.array([box.x1, box.y1, box.z1], dtype=float)
    axis = np.array([box.x2, box.y2, box.z2], dtype=float) - p1
    dd = float(axis @ axis)
    if dd == 0.0:                       # zero-length axis -> empty (degenerate)
        return np.zeros(c.shape[0], dtype=bool)
    v = c - p1
    t = (v @ axis) / dd                 # projection along the axis, 0..1 = inside
    radial = v - np.outer(t, axis)      # perpendicular offset from the axis
    radial2 = np.einsum("ij,ij->i", radial, radial)
    return (t >= 0.0) & (t <= 1.0) & (radial2 <= float(box.radius) ** 2)


def primitive_member(centroids: np.ndarray, box) -> np.ndarray:
    """Boolean mask of *centroids* inside a single growth region *box*.

    Dispatches on ``box.shape_kind()``: a rectangular box (axis-aligned, or in the
    box's local frame when one is set), a sphere (centre + radius) or a finite
    cylinder (two axis end-points + radius). Bounds are inclusive."""
    kind = box.shape_kind()
    if kind == "sphere":
        return _sphere_member(centroids, box)
    if kind == "cylinder":
        return _cylinder_member(centroids, box)
    return _box_member(centroids, box)


def box_corners(box) -> np.ndarray:
    """The 8 world-space corners of a rectangular growth *box* (``(8, 3)``).

    Corner ``i`` takes ``x_max`` iff bit 2 of ``i`` is set, ``y_max`` iff bit 1,
    ``z_max`` iff bit 0 — the ordering :data:`_BOX_EDGES` is built against. An
    oriented box (local frame) has its corners transformed back into world space."""
    xs = (box.x_min, box.x_max)
    ys = (box.y_min, box.y_max)
    zs = (box.z_min, box.z_max)
    local = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=float)
    frame = local_frame_basis(box)
    if frame is not None:
        origin, R = frame
        return origin + local @ R       # inverse of (c - origin) @ R.T
    return local


def overlay_primitives(boxes) -> list[dict]:
    """JSON-serialisable wireframe-outline descriptors for the growth regions.

    One dict per region, consumed identically by the GUI Monitor's in-process
    pyvista overlay and the report's isolated render subprocess:

    * box      -> ``{"kind": "box", "name", "corners": [[x,y,z]*8], "edges": [[i,j]*12]}``
    * sphere   -> ``{"kind": "sphere", "name", "center": [x,y,z], "radius": r}``
    * cylinder -> ``{"kind": "cylinder", "name", "p1", "p2", "radius": r}``

    Regions with no drawable geometry are skipped: a zero-radius sphere/cylinder,
    a zero-length cylinder axis, a zero-size box, and a box still referencing a
    ``deck_box_id`` (its corners are only known once the deck is loaded)."""
    out: list[dict] = []
    for b in boxes or []:
        kind = b.shape_kind()
        if kind == "sphere":
            if float(b.radius) > 0.0:
                out.append({"kind": "sphere", "name": b.name,
                            "center": [b.cx, b.cy, b.cz], "radius": float(b.radius)})
        elif kind == "cylinder":
            p1 = [b.x1, b.y1, b.z1]
            p2 = [b.x2, b.y2, b.z2]
            if float(b.radius) > 0.0 and p1 != p2:
                out.append({"kind": "cylinder", "name": b.name,
                            "p1": p1, "p2": p2, "radius": float(b.radius)})
        else:                                            # box
            if getattr(b, "deck_box_id", None) is not None:
                continue                                 # corners not yet resolved
            corners = box_corners(b)
            if float(np.ptp(corners, axis=0).max()) <= 0.0:
                continue                                 # zero-size box
            out.append({"kind": "box", "name": b.name,
                        "corners": corners.tolist(),
                        "edges": [list(e) for e in _BOX_EDGES]})
    return out


@dataclass
class Mesh:
    centroids: np.ndarray     # (N,3) element centroids
    volumes: np.ndarray       # (N,) tetra volumes
    conn_rows: np.ndarray     # (N,4) node ROW indices into node_xyz (0-based)
    n_nodes: int
    design_node_min: int

    @classmethod
    def from_deck(cls, deck: Deck) -> "Mesh":
        # map every connectivity node id to its row in deck.node_xyz (vectorised)
        order = np.argsort(deck.node_ids)
        sorted_ids = deck.node_ids[order]
        pos = np.searchsorted(sorted_ids, deck.elem_conn)
        rows = order[pos]                                  # (N,4)
        xyz = deck.node_xyz[rows]                          # (N,4,3)
        centroids = xyz.mean(axis=1)
        # tetra volume = |(a-d) . ((b-d) x (c-d))| / 6
        a, b, c, d = xyz[:, 0], xyz[:, 1], xyz[:, 2], xyz[:, 3]
        volumes = np.abs(np.einsum("ij,ij->i", a - d,
                                   np.cross(b - d, c - d))) / 6.0
        return cls(centroids=centroids, volumes=volumes, conn_rows=rows,
                   n_nodes=deck.node_ids.size, design_node_min=deck.design_node_min)

    @property
    def n_elements(self) -> int:
        return int(self.volumes.size)

    # ---- sensitivity filter ------------------------------------------------
    def filter_matrix(self, radius: float) -> sparse.csr_matrix:
        """Row-normalised linear ("hat") filter weights over element centroids.

        ``filtered = W @ raw``. ``radius <= 0`` returns the identity (no filter).
        """
        n = self.n_elements
        if radius <= 0:
            return sparse.identity(n, format="csr")
        tree = cKDTree(self.centroids)
        dmat = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
        w = np.maximum(0.0, 1.0 - dmat.data / radius)
        W = sparse.coo_matrix((w, (dmat.row, dmat.col)), shape=(n, n)).tocsr()
        W.setdiag(1.0)                                     # ensure self-weight
        rowsum = np.asarray(W.sum(axis=1)).ravel()
        rowsum[rowsum == 0] = 1.0
        return sparse.diags(1.0 / rowsum) @ W

    # ---- growth boxes --------------------------------------------------------
    def in_boxes_mask(self, boxes) -> np.ndarray:
        """Elements whose centroid lies inside any of *boxes* — the growth-region
        candidate set. Union over the regions (inclusive bounds); an empty/None
        list returns all-False. Each region is a
        :class:`~oropt.config.GrowthBox`; its ``shape`` ("box" / "sphere" /
        "cylinder", plus an optional local frame for a box) selects the
        centroid-in-primitive test (see :func:`primitive_member`).
        """
        mask = np.zeros(self.n_elements, dtype=bool)
        for b in boxes or []:
            mask |= primitive_member(self.centroids, b)
        return mask

    # ---- connectivity ------------------------------------------------------
    def _incidence(self, elem_subset: np.ndarray) -> sparse.csr_matrix:
        """Bipartite element-node incidence for the given element indices."""
        ne = elem_subset.size
        rows = np.repeat(np.arange(ne), 4)
        cols = self.conn_rows[elem_subset].ravel()
        data = np.ones(rows.size, dtype=np.int8)
        return sparse.coo_matrix((data, (rows, cols)),
                                 shape=(ne, self.n_nodes)).tocsr()

    def keep_connected(self, alive_mask: np.ndarray,
                       seed_mask: np.ndarray) -> np.ndarray:
        """Restrict *alive_mask* to elements connected (via shared nodes) to a
        seed (boundary/load) element. Floating islands are dropped."""
        alive_mask = np.asarray(alive_mask, dtype=bool)
        alive_idx = np.flatnonzero(alive_mask)
        if alive_idx.size == 0:
            return alive_mask.copy()
        inc = self._incidence(alive_idx)                   # (na, M)
        # element-element adjacency via shared nodes: A = inc @ inc.T
        A = (inc @ inc.T)
        n_comp, labels = connected_components(A, directed=False)
        seed_local = seed_mask[alive_idx]
        keep_labels = set(np.unique(labels[seed_local]).tolist())
        local_keep = np.array([lbl in keep_labels for lbl in labels]) \
            if keep_labels else np.zeros(alive_idx.size, dtype=bool)
        out = np.zeros_like(alive_mask)
        out[alive_idx[local_keep]] = True
        return out

    # ---- protected elements ------------------------------------------------
    def protected_mask(self, deck: Deck, protect_node_ids: np.ndarray,
                       contact_dist: float = 0.0, layers: int = 2) -> np.ndarray:
        """Elements that must never be deleted: those touching a protected node
        (BC/symmetry + user keep-out sets) or lying within *contact_dist* of a
        rigid (cylinder) node, dilated *layers* hops.

        Returns a boolean mask aligned with the element arrays.
        """
        node_ids = deck.node_ids
        seed_nodes = set(int(v) for v in protect_node_ids)

        if contact_dist > 0:
            is_rigid = node_ids < self.design_node_min
            if is_rigid.any():
                rigid_tree = cKDTree(deck.node_xyz[is_rigid])
                design_idx = np.flatnonzero(~is_rigid)
                dist, _ = rigid_tree.query(deck.node_xyz[design_idx], k=1)
                near = design_idx[dist <= contact_dist]
                seed_nodes.update(int(node_ids[i]) for i in near)

        # rows (into node_xyz) of seed nodes
        id_to_row = {int(v): i for i, v in enumerate(node_ids)}
        seed_rows = np.array([id_to_row[v] for v in seed_nodes
                              if v in id_to_row], dtype=np.int64)

        # elements touching a seed node, then dilate by `layers` via shared nodes
        node_flag = np.zeros(self.n_nodes, dtype=bool)
        node_flag[seed_rows] = True
        mask = node_flag[self.conn_rows].any(axis=1)
        for _ in range(max(0, layers)):
            node_flag = np.zeros(self.n_nodes, dtype=bool)
            node_flag[np.unique(self.conn_rows[mask])] = True
            mask = node_flag[self.conn_rows].any(axis=1)
        return mask
