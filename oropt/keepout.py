"""Growth keep-out: neighbour parts whose occupied volume forbids material.

A keep-out deck (``model.growth_keepout_rad``) describes nearby parts that are
**never solved** -- only their geometry matters, as forbidden growth space. Where
a growth box overlaps that space, the overlapping candidates are *held void* every
iteration (they start void like any candidate, but are never grown), so the
optimiser can never place material inside the neighbour parts. The auto-mesh
PREPARE step honours the same test by simply not generating candidate tets there.

The occupied space is the union of the neighbour parts' solid elements (their
actual mesh, not a bounding box), tetrahedralised so the membership test is exact
point-in-tetrahedron. The clearance shifts the forbidden boundary:

* ``clearance > 0`` keeps a **gap** around the neighbour parts: a candidate whose
  centroid is within ``clearance`` of a neighbour node is forbidden too
  (nearest-node distance ~ surface distance for a finely meshed neighbour -- a
  small, conservative-enough band around a densely-noded surface).
* ``clearance < 0`` allows a **deliberate penetration** of up to ``|clearance|``
  into the neighbour volume (an interference/overlap band -- e.g. a weld/bond
  allowance, or compensating a neighbour envelope meshed oversize): only
  candidates DEEPER than ``|clearance|`` below the neighbour *surface* stay
  forbidden. Depth is measured to the nearest surface node (interior nodes are
  excluded -- their small distances would masquerade as "near the surface" deep
  inside the volume), which over-estimates the true depth by up to the
  neighbour's surface-facet size, so the band errs on the side of LESS
  penetration than asked. Mesh the neighbour finer than ``|clearance|`` for a
  tight band.
"""
from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from functools import cached_property, lru_cache
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
    surf_xyz: np.ndarray     # (S,3) neighbour SURFACE nodes (penetration depth)
    part_ids: list           # solid part ids read from the deck
    clearance: float         # >0 extra gap around the volume; <0 allowed
    #                          penetration depth into it [model units]
    source: str              # deck path, for run-log / preview / validation messages

    # The trees are built once per KeepOut and reused: block_mask runs several
    # times per preview / run start (candidate mask, blocked mask, preview rows)
    # over the same geometry, and a production neighbour cloud is large.
    @cached_property
    def _node_tree(self) -> cKDTree:
        return cKDTree(self.node_xyz)

    @cached_property
    def _surf_tree(self) -> cKDTree:
        return cKDTree(self.surf_xyz)

    def block_mask(self, centroids: np.ndarray) -> np.ndarray:
        """Boolean mask: which *centroids* are forbidden growth space.

        Inside the neighbour volume, widened by a positive :attr:`clearance`
        (anything within it of a neighbour node is forbidden too) or shrunk by
        a negative one (a centroid within ``|clearance|`` of the neighbour
        *surface* is allowed -- the sanctioned interference band)."""
        centroids = np.asarray(centroids, dtype=float)
        inside = points_in_tets(centroids, self.tet_xyz)
        if self.clearance > 0.0 and len(self.node_xyz):
            dist, _ = self._node_tree.query(centroids)
            inside = inside | (dist <= self.clearance)
        elif self.clearance < 0.0 and inside.any() and len(self.surf_xyz):
            idx = np.flatnonzero(inside)
            depth, _ = self._surf_tree.query(centroids[idx])
            inside[idx[depth <= -self.clearance]] = False
        return inside


# One-slot memo for the resolved keep-out: a single preview/run-start resolves
# the same deck several times (candidate mask, blocked mask, preview), and a
# production neighbour deck takes seconds to parse. Keyed on the deck's stat
# signature + selection + clearance, so any change on disk (or of the knobs)
# rebuilds; a repeat with identical inputs returns the SAME KeepOut, keeping its
# cached KD-trees too.
_RESOLVE_MEMO: dict = {}


def resolve_keepout(model, case_dir=None) -> Optional[KeepOut]:
    """Build the :class:`KeepOut` for ``model.growth_keepout_rad``, or ``None``
    when the feature is unconfigured.

    The deck path is resolved relative to *case_dir* (defaulting to
    ``model.case_dir``) like the load-case decks. ``model.growth_keepout_part_ids``
    selects which parts form the keep-out (empty = all solid parts) and
    ``model.growth_keepout_clearance_mm`` sets the clearance band (negative =
    allowed penetration depth; see the module docstring). Raises ``ValueError``
    for a missing or unparsable deck, or a non-finite clearance (surfaced at run
    start, before any solve)."""
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
    clearance = float(getattr(model, "growth_keepout_clearance_mm", 0.0) or 0.0)
    if not math.isfinite(clearance):
        # NaN would silently disable both clearance branches (every comparison
        # is False) -- reject it loudly instead of quietly acting as 0.
        raise ValueError("growth_keepout_clearance_mm must be a finite number: "
                         f"got {clearance!r}")

    st = path.stat()
    key = (str(path.resolve()), st.st_mtime_ns, st.st_size,
           tuple(sorted(part_ids)), clearance)
    hit = _RESOLVE_MEMO.get("keepout")
    if hit is not None and hit[0] == key:
        return hit[1]
    tet_xyz, node_xyz, surf_xyz, found = read_solid_geometry(
        path, part_ids or None)
    ko = KeepOut(tet_xyz=tet_xyz, node_xyz=node_xyz, surf_xyz=surf_xyz,
                 part_ids=found, clearance=clearance, source=str(path))
    _RESOLVE_MEMO["keepout"] = (key, ko)
    return ko


# --------------------------------------------------------------------------- #
# positive deck growth region (GrowthBox shape="deck") -- the mirror of KeepOut
#
# A growth box may take its shape from a separate deck's parts (``region_rad``)
# instead of a primitive: material may be ADDED where those parts sit (or, with
# ``forbid=True``, forbidden there). The geometry read is identical to the
# keep-out's, so it lives here; it is attached to the box for the pure membership
# test (:func:`oropt.mesh._deck_member`) and the 3D overlay
# (:func:`oropt.mesh.overlay_primitives`). Kept here (not in loop.py) so the
# post-run overlay builders (report / animate) can resolve it without importing
# loop (which imports them -- a cycle).
# --------------------------------------------------------------------------- #

@lru_cache(maxsize=32)
def _load_region_solid(path_str: str, part_ids_key: tuple, _mtime: int):
    """Solid geometry ``(tet_xyz, node_xyz)`` of the parts in a ``shape="deck"``
    region's deck, memoised so the same deck isn't re-parsed for each of the
    candidate / blocked / preview / overlay passes. Keyed on the resolved path,
    the selected part ids and the file mtime, so an edited deck is re-read."""
    tet_xyz, node_xyz, _surf, _found = read_solid_geometry(
        Path(path_str), list(part_ids_key) or None)
    return tet_xyz, node_xyz


def resolve_deck_region(box, case_dir=None, label: Optional[str] = None):
    """A ``shape="deck"`` :class:`~oropt.config.GrowthBox` with its parts' solid
    geometry loaded and attached -- the runtime attributes ``_region_tets``
    (V,4,3) / ``_region_nodes`` (P,3) / ``_region_clearance`` consumed by
    :func:`oropt.mesh._deck_member` (membership) and
    :func:`oropt.mesh.overlay_primitives` (outline). The deck path (``region_rad``)
    resolves relative to *case_dir* like the load-case decks. Raises ``ValueError``
    for a blank/missing/unparsable deck (surfaced at run start / in the preview,
    before any solve). The positive mirror of :func:`resolve_keepout`."""
    label = label or box.name or "deck region"
    rad = (getattr(box, "region_rad", None) or "").strip()
    if not rad:
        raise ValueError(
            f"growth box {label!r}: shape 'deck' needs region_rad -- a Radioss "
            "deck whose parts' geometry defines the growth region")
    path = Path(rad)
    if not path.is_absolute():
        path = Path(case_dir if case_dir is not None else ".") / rad
    if not path.exists():
        raise ValueError(f"growth box {label!r}: region deck not found: {path}")
    part_ids = tuple(int(p) for p in (getattr(box, "region_part_ids", None) or []))
    tet_xyz, node_xyz = _load_region_solid(
        str(path.resolve()), part_ids, path.stat().st_mtime_ns)
    rb = dataclasses.replace(box)
    rb._region_tets = tet_xyz
    rb._region_nodes = node_xyz
    rb._region_clearance = float(getattr(box, "region_clearance_mm", 0.0) or 0.0)
    return rb


def resolve_overlay_boxes(boxes, case_dir=None) -> list:
    """Best-effort attach deck-region geometry so the 3D overlay can outline each
    ``shape="deck"`` region. A region whose deck is blank/missing/unparsable is
    returned unchanged (the overlay then skips it) -- the overlay is cosmetic and
    must never raise. Non-deck regions pass through unchanged."""
    out = []
    for b in (boxes or []):
        if getattr(b, "shape_kind", lambda: "")() == "deck":
            try:
                out.append(resolve_deck_region(b, case_dir))
                continue
            except (ValueError, OSError):
                pass
        out.append(b)
    return out
