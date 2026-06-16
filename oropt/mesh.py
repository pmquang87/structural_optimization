"""Geometry and graph operations on the design-part mesh.

Built from a :class:`~oropt.deck.Deck`, this provides everything the BESO core
needs that is purely geometric/topological: element centroids and volumes, the
spatial sensitivity-filter matrix, element connectivity (to drop floating
islands), and the set of protected elements that must never be deleted
(boundary-condition, contact and load regions).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from .deck import Deck


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
