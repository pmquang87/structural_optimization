"""Growth keep-out: neighbour parts whose occupied volume forbids material.

A keep-out deck (``model.growth_keepout_rad``) describes nearby parts that are
**never solved** -- only their geometry matters, as forbidden growth space. Where
a growth box overlaps that space, the overlapping candidates are *held void* every
iteration (they start void like any candidate, but are never grown), so the
optimiser can never place material inside the neighbour parts. The auto-mesh
PREPARE step honours the same test by simply not generating candidate tets there.

The occupied space is the union of the neighbour parts' solid elements (their
actual mesh, not a bounding box), tetrahedralised so the membership test is exact
point-in-tetrahedron. An optional clearance keeps a gap around the neighbour
parts: a candidate whose centroid is within ``clearance`` of a neighbour node is
forbidden too (nearest-node distance ~ surface distance for a finely meshed
neighbour -- a small, conservative-enough band around a densely-noded surface).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.spatial import cKDTree

from .deck import read_solid_geometry
from .mesh import points_in_tets


@dataclass
class KeepOut:
    """Neighbour-part geometry defining forbidden growth space."""
    tet_xyz: np.ndarray      # (V,4,3) occupied volume, tetrahedralised
    node_xyz: np.ndarray     # (P,3) referenced neighbour nodes (clearance test)
    part_ids: list           # solid part ids read from the deck
    clearance: float         # extra forbidden band around the volume [model units]
    source: str              # deck path, for run-log / preview / validation messages

    def block_mask(self, centroids: np.ndarray) -> np.ndarray:
        """Boolean mask: which *centroids* fall inside the neighbour volume, or
        within :attr:`clearance` of it -- the growth candidates to forbid."""
        centroids = np.asarray(centroids, dtype=float)
        inside = points_in_tets(centroids, self.tet_xyz)
        if self.clearance > 0.0 and len(self.node_xyz):
            dist, _ = cKDTree(self.node_xyz).query(centroids)
            inside = inside | (dist <= self.clearance)
        return inside


def resolve_keepout(model, case_dir=None) -> Optional[KeepOut]:
    """Build the :class:`KeepOut` for ``model.growth_keepout_rad``, or ``None``
    when the feature is unconfigured.

    The deck path is resolved relative to *case_dir* (defaulting to
    ``model.case_dir``) like the load-case decks. ``model.growth_keepout_part_ids``
    selects which parts form the keep-out (empty = all solid parts) and
    ``model.growth_keepout_clearance_mm`` sets the clearance band. Raises
    ``ValueError`` for a missing or unparsable deck (surfaced at run start, before
    any solve)."""
    rad = getattr(model, "growth_keepout_rad", None)
    if not rad:
        return None
    base = Path(case_dir if case_dir is not None
                else getattr(model, "case_dir", "."))
    path = Path(rad)
    if not path.is_absolute():
        path = base / rad
    if not path.exists():
        raise ValueError(f"growth keep-out deck not found: {path}")
    part_ids = list(getattr(model, "growth_keepout_part_ids", []) or [])
    tet_xyz, node_xyz, found = read_solid_geometry(path, part_ids or None)
    clearance = float(getattr(model, "growth_keepout_clearance_mm", 0.0) or 0.0)
    return KeepOut(tet_xyz=tet_xyz, node_xyz=node_xyz, part_ids=found,
                   clearance=clearance, source=str(path))
