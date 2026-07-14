"""Growth-mesh PREPARE step (phase 2): auto-generate the candidate mesh inside
growth regions by direct node/element creation, so the region volume no longer
has to be pre-meshed in a pre-processor.

An explicit, user-triggered step — ``python -m oropt.growthmesh --config …`` or
the GUI's "Generate growth mesh" button — never something hidden inside run
start. It reads the primary load case's starter deck, tetrahedralises the space
inside every growth region (:attr:`~oropt.config.Model.growth_boxes`) and writes
EXTENDED starter decks the user can inspect and diff before running:

* **Output contract** — a full deck set under ``<case_dir>/growth_mesh/``: for
  every load case the extended ``<stem>_0000.rad`` (the same file, with the new
  ``/NODE`` and ``/TETRA4/<design_part_id>`` cards appended inside their blocks
  and everything else byte-identical) plus a verbatim copy of the case's
  ``<stem>_0001.rad`` engine deck, **and** a copy of every auxiliary geometry
  deck the config references relative to ``case_dir`` — each ``shape="deck"``
  growth region's ``region_rad`` and the ``growth_keepout_rad`` deck — so the
  folder is a self-contained case dir (those decks are re-read at run time
  relative to ``case_dir``, which now points here). Point ``model.case_dir`` at that folder (the
  CLI prints the line to change; the GUI offers a button) and run — the run
  itself stays byte-identical phase-1 behaviour: the generated elements start
  void because their centroids lie inside the regions, and the existing
  run-start guards (empty region, ``design_node_min``, reachability) double as
  automatic self-checks of the generated mesh. They are re-run here, on the
  extended deck, *before* anything is written.

* **Meshing backend** — TetGen constrained tetrahedralisation via the optional
  ``tetgen`` package (``pip install "oropt[growthmesh]"``; note TetGen itself is
  AGPL-licensed). The design part's watertight exterior surface (the boundary
  faces of its TET4 mesh) is embedded as internal facets of a PLC whose outer
  boundary is the slightly inflated AABB of part ∪ regions (walls
  pre-subdivided near the element scale where regions are close, coarse
  elsewhere — see :func:`box_shell`), and
  tetrahedralised with TetGen's ``-Y`` switch: the input facets and vertices
  are preserved exactly, Steiner points appear only in the interior. Every output tetrahedron
  touching the part therefore reuses the part's own surface **nodes** — exact
  node conformity, which is what lets all the phase-1 machinery
  (``keep_connected``, free-node pinning, ``/SURF/PART/EXT`` regeneration, the
  verbatim per-iteration rewrite) work untouched, with no tied interface. The
  output tets are then classified by centroid: those inside the part duplicate
  existing elements and are dropped, those outside every region are scaffolding
  and are dropped; the survivors are the new candidate elements.

* **Element sizing** — a background sizing mesh (TetGen ``-m``, see
  :class:`BgSizing`), NOT a global ``-a`` volume bound: the target-edge field
  equals the part-derived element size inside every growth region (plus a
  small halo) and coarsens with distance, capped at a fraction of the domain
  diagonal. A global ``-a`` would mesh the scaffolding between the part and
  the domain walls as finely as the candidates — for a large part with small
  regions that is tens of millions of throw-away tets (memory runaway); with
  the sizing field the scaffold cost scales with the *region* volume budget
  plus a thin boundary layer along the part surface.

* **Ids** — new node ids are allocated above ``max(existing)`` and >=
  ``design_node_min`` (so the free-node guard can pin them while void); new
  element ids above ``max(existing design elements)``.

Placement remains the user's responsibility exactly as in phase 1: the
generator only sees the design part, so keep regions clear of *other* parts
(rigid bodies, shells) or grown material may interpenetrate them.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import shutil
import sys
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import yaml
from scipy.spatial import cKDTree

from .config import Config
from .deck import Deck
from .keepout import resolve_keepout
from .loop import growth_candidate_mask, resolve_growth_boxes
from .mesh import Mesh, box_corners, points_in_tets, primitive_member

#: sub-folder of ``model.case_dir`` receiving the extended deck set
GROWTH_MESH_DIRNAME = "growth_mesh"

#: the four faces of a TET4, as local vertex indices
_TET_FACES = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])

#: backend signature: (points (P,3), tris (F,3) indices into points,
#: target tet volume inside the regions, -q radius-edge ratio, background
#: sizing field or None for a single global -a bound)
#: -> (out points (Q,3), out tets (M,4))
MeshBackend = Callable[[np.ndarray, np.ndarray, float, float,
                        Optional["BgSizing"]],
                       tuple[np.ndarray, np.ndarray]]

#: background sizing-mesh resolution near the regions, x target_edge
_BG_CELL_FACTOR = 4.0

#: fine-sizing halo around each region AABB, x target_edge. Two background
#: cells: every bg tet overlapping a region then has ALL its nodes at the
#: target value, so the size interpolated anywhere inside the region is
#: exactly target_edge (gradation cannot clip the region boundary).
_SIZING_PAD_FACTOR = 2.0 * _BG_CELL_FACTOR


# --------------------------------------------------------------------------- #
# geometry core (hermetic: numpy/scipy only, no tetgen)
# --------------------------------------------------------------------------- #

def exterior_faces(elem_conn: np.ndarray) -> np.ndarray:
    """The design part's exterior surface triangles as ``(F, 3)`` node ids.

    A face of a conforming TET4 mesh is exterior iff exactly one element uses
    it; the union of those faces is the part's watertight boundary surface(s) —
    including internal cavity shells and each shell of a multi-body part, all
    of which belong in the PLC as internal facets."""
    faces = elem_conn[:, _TET_FACES].reshape(-1, 3)
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True,
                                   return_counts=True)
    return faces[counts[inverse] == 1]


def mean_edge_length(points: np.ndarray, tris: np.ndarray) -> float:
    """Mean edge length of a triangle surface (*tris* index into *points*) —
    the part-derived element-size reference the ``size_factor`` knob scales."""
    p = points[tris]
    e = np.concatenate([p[:, 1] - p[:, 0], p[:, 2] - p[:, 1], p[:, 0] - p[:, 2]])
    return float(np.linalg.norm(e, axis=1).mean())


def box_shell(lo: np.ndarray, hi: np.ndarray, cell: float,
              grids: Optional[list] = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """Triangulated axis-aligned box shell over ``[lo, hi]`` with every wall
    subdivided into ~*cell*-sized quads (each split into two triangles).

    The subdivision matters: under ``-Y`` TetGen may not split boundary facets,
    and it *rejects* interior refinement points that encroach one — a wall left
    as two giant triangles would silently veto the ``-a``/``-m`` element sizing
    across most of the domain (verified against tetgen 0.8.4). Pre-subdivided
    wall vertices are ordinary input points, so refinement proceeds.

    *grids* (three 1-D arrays of wall grid lines, one per axis) overrides the
    uniform subdivision: :func:`build_plc` passes :func:`graded_axis` grids so
    the walls are fine only near the growth regions — under ``-m`` sizing the
    far walls carry coarse tets and giant fine-wall grids would just burn
    memory (all wall vertices are preserved input points)."""
    lo = np.asarray(lo, dtype=float)
    hi = np.asarray(hi, dtype=float)
    if grids is None:
        ndiv = np.ceil((hi - lo) / max(float(cell), 1e-12)).astype(int)
        ndiv = np.clip(ndiv, 1, 256)      # cap the wall triangle count
        grids = [np.linspace(lo[a], hi[a], ndiv[a] + 1) for a in range(3)]
    pts_all, tri_all, off = [], [], 0
    for ax in range(3):                                  # wall-normal axis
        u, v = (ax + 1) % 3, (ax + 2) % 3
        uu, vv = np.meshgrid(grids[u], grids[v], indexing="ij")
        nu, nv = uu.shape
        idx = np.arange(nu * nv).reshape(nu, nv)
        q00 = idx[:-1, :-1].ravel(); q10 = idx[1:, :-1].ravel()
        q01 = idx[:-1, 1:].ravel(); q11 = idx[1:, 1:].ravel()
        tris = np.concatenate([np.stack([q00, q10, q11], axis=1),
                               np.stack([q00, q11, q01], axis=1)])
        for w in (lo[ax], hi[ax]):
            p = np.empty((nu, nv, 3))
            p[..., ax] = w; p[..., u] = uu; p[..., v] = vv
            pts_all.append(p.reshape(-1, 3))
            tri_all.append(tris + off)
            off += nu * nv
        # (identical linspace values per axis -> shared wall edges/corners are
        # bitwise-equal points, merged exactly below)
    pts = np.vstack(pts_all)
    tris = np.vstack(tri_all)
    _, first, inverse = np.unique(pts, axis=0, return_index=True,
                                  return_inverse=True)
    return pts[first], inverse[tris]


def build_plc(surf_points: np.ndarray, surf_tris: np.ndarray,
              lo: np.ndarray, hi: np.ndarray, wall_cell: float,
              fine_intervals: Optional[list] = None,
              coarse_cell: Optional[float] = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the PLC: the part surface plus the subdivided outer box shell
    over ``[lo, hi]``. Returns ``(points, tris)`` with the surface points FIRST
    and un-reindexed, so index ``i < len(surf_points)`` still refers to surface
    point ``i`` (the node-id mapping relies on this).

    With *fine_intervals* (per-axis lists of ``(lo, hi)`` bands — the padded
    region projections) and *coarse_cell*, the walls are subdivided at
    *wall_cell* only inside the bands and coarsen geometrically to
    *coarse_cell* away from them, matching the ``-m`` sizing field: a wall
    facet only vetoes refinement it is actually near (see :func:`box_shell`)."""
    if fine_intervals is not None:
        cc = wall_cell if coarse_cell is None else max(coarse_cell, wall_cell)
        grids = [graded_axis(lo[a], hi[a], fine_intervals[a], wall_cell, cc)
                 for a in range(3)]
        shell_pts, shell_tris = box_shell(lo, hi, wall_cell, grids=grids)
    else:
        shell_pts, shell_tris = box_shell(lo, hi, wall_cell)
    points = np.vstack([surf_points, shell_pts])
    tris = np.vstack([surf_tris, shell_tris + len(surf_points)])
    return points, tris


def region_aabb(box) -> tuple[np.ndarray, np.ndarray]:
    """Loose AABB ``(lo, hi)`` of ONE resolved growth region (all shapes;
    oriented boxes via their world-space corners, polyhedra via their explicit
    node set, deck regions via the referenced parts' node bounds)."""
    kind = box.shape_kind()
    if kind == "sphere":
        c = np.array([box.cx, box.cy, box.cz], dtype=float)
        return c - box.radius, c + box.radius
    if kind == "cylinder":
        p = np.array([[box.x1, box.y1, box.z1],
                      [box.x2, box.y2, box.z2]], dtype=float)
        return p.min(axis=0) - box.radius, p.max(axis=0) + box.radius
    if kind == "polyhedron":
        p = np.asarray(box.points, dtype=float)
        return p.min(axis=0), p.max(axis=0)
    if kind == "deck":
        # the region's part geometry was attached by resolve_growth_boxes; the
        # clearance band grows the AABB outward the same way _deck_member does.
        nodes = np.asarray(getattr(box, "_region_nodes", np.empty((0, 3))),
                           dtype=float)
        if not len(nodes):
            tets = np.asarray(getattr(box, "_region_tets", np.empty((0, 4, 3))),
                              dtype=float)
            nodes = tets.reshape(-1, 3) if len(tets) else nodes
        clr = float(getattr(box, "_region_clearance", 0.0) or 0.0)
        return nodes.min(axis=0) - clr, nodes.max(axis=0) + clr
    c = box_corners(box)
    return c.min(axis=0), c.max(axis=0)


def region_bounds(boxes) -> tuple[np.ndarray, np.ndarray]:
    """Loose AABB ``(lo, hi)`` over all resolved growth regions
    (:func:`region_aabb` per region)."""
    aabbs = [region_aabb(b) for b in boxes]
    return (np.min([lo for lo, _ in aabbs], axis=0),
            np.max([hi for _, hi in aabbs], axis=0))


def signed_volumes(tet_xyz: np.ndarray) -> np.ndarray:
    """Signed volume of each tet ``(M, 4, 3)``:
    ``det(n2-n1, n3-n1, n4-n1) / 6`` — the node-ordering convention check."""
    a, b, c, d = tet_xyz[:, 0], tet_xyz[:, 1], tet_xyz[:, 2], tet_xyz[:, 3]
    return np.einsum("ij,ij->i", b - a, np.cross(c - a, d - a)) / 6.0


def orient_tets(tets: np.ndarray, points: np.ndarray, sign: float) -> np.ndarray:
    """Return *tets* with the last two nodes swapped wherever the signed volume
    disagrees with *sign* — new elements follow the deck's own node-ordering
    convention (the majority sign of the existing part elements) so the solver
    sees consistently oriented TETRA4 cards."""
    out = np.array(tets, dtype=np.int64, copy=True)
    flip = signed_volumes(points[out]) * sign < 0
    out[flip, 2], out[flip, 3] = tets[flip, 3], tets[flip, 2]
    return out


def tet_quality(tet_xyz: np.ndarray) -> np.ndarray:
    """Per-tet shape quality ``6*sqrt(2)*V / L_rms^3`` (regular tet = 1,
    degenerate = 0) — the number reported in the generated-mesh stats."""
    v = np.abs(signed_volumes(tet_xyz))
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    e2 = np.stack([((tet_xyz[:, i] - tet_xyz[:, j]) ** 2).sum(axis=1)
                   for i, j in pairs])
    l_rms = np.sqrt(e2.mean(axis=0))
    with np.errstate(divide="ignore", invalid="ignore"):
        q = np.where(l_rms > 0, 6.0 * np.sqrt(2.0) * v / l_rms ** 3, 0.0)
    return q


def map_to_original_nodes(out_points: np.ndarray, surf_xyz: np.ndarray,
                          surf_ids: np.ndarray, tol: float) -> np.ndarray:
    """Original node id for every output vertex that coincides (within *tol*)
    with a part surface node, ``-1`` elsewhere — how TetGen's preserved input
    vertices are mapped back to the deck's node ids for exact conformity."""
    out = np.full(len(out_points), -1, dtype=np.int64)
    if len(out_points) == 0 or len(surf_xyz) == 0:
        return out
    dist, idx = cKDTree(surf_xyz).query(out_points, k=1)
    hit = dist <= tol
    out[hit] = surf_ids[idx[hit]]
    return out


# --------------------------------------------------------------------------- #
# background sizing mesh (TetGen -m; hermetic construction, numpy only)
# --------------------------------------------------------------------------- #

@dataclasses.dataclass(frozen=True)
class BgSizing:
    """A coarse background tet mesh carrying a per-node target-edge field —
    TetGen's ``-m`` metric input. This is what keeps the PREPARE step's memory
    proportional to the *region* volume: the field is ``target_edge`` inside
    every growth region (plus the :data:`_SIZING_PAD_FACTOR` halo) and grows
    with distance, so only the candidates are meshed finely while the
    scaffolding coarsens toward the cap."""
    points: np.ndarray          # (N, 3) node coordinates
    tets: np.ndarray            # (M, 4) indices into points
    mtr: np.ndarray             # (N,) target edge length per node


def graded_axis(lo: float, hi: float, intervals, fine: float, coarse: float,
                growth: float = 1.5) -> np.ndarray:
    """1-D grid over ``[lo, hi]``: ~*fine* spacing inside the ``(lo, hi)``
    *intervals*, steps growing geometrically (factor *growth*) with distance
    from the nearest interval, capped at *coarse*. The wall grids and the
    background mesh both use this, so fine resolution exists exactly in the
    bands the regions project onto and nowhere else."""
    lo, hi = float(lo), float(hi)
    fine = max(float(fine), 1e-12)
    coarse = max(float(coarse), fine)
    if not hi - lo > 0.0:
        return np.array([lo, hi])
    pts = [lo]
    x = lo
    while True:
        d = min((max(a - x, x - b, 0.0) for a, b in intervals),
                default=np.inf)
        h = min(coarse, fine + (growth - 1.0) * d)
        if np.isfinite(d) and d > 0.0:
            h = min(h, max(d, fine))     # approach an interval, don't leap it
        if x + 1.5 * h >= hi:            # absorb the tail into the last cell
            pts.append(hi)
            return np.asarray(pts)
        x += h
        pts.append(x)


def box_grid_tets(gx: np.ndarray, gy: np.ndarray, gz: np.ndarray
                  ) -> tuple[np.ndarray, np.ndarray]:
    """Tensor-product grid over the three axis-line arrays, each cell split
    into 6 Kuhn tets around its main diagonal — conforming across cells (all
    face diagonals run min-corner to max-corner) and positively oriented
    (TetGen walks the background mesh and assumes consistent orientation)."""
    nx, ny, nz = len(gx), len(gy), len(gz)
    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
    pts = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    idx = np.arange(nx * ny * nz).reshape(nx, ny, nz)
    c = {(dx, dy, dz): idx[dx:nx - 1 + dx, dy:ny - 1 + dy,
                           dz:nz - 1 + dz].ravel()
         for dx in (0, 1) for dy in (0, 1) for dz in (0, 1)}
    # the 6 monotone corner-to-corner paths 000 -> a -> b -> 111
    paths = [((1, 0, 0), (1, 1, 0)), ((1, 0, 0), (1, 0, 1)),
             ((0, 1, 0), (1, 1, 0)), ((0, 1, 0), (0, 1, 1)),
             ((0, 0, 1), (1, 0, 1)), ((0, 0, 1), (0, 1, 1))]
    tets = np.concatenate([np.stack([c[0, 0, 0], c[a], c[b], c[1, 1, 1]],
                                    axis=1) for a, b in paths])
    return pts, orient_tets(tets, pts, 1.0)


def sizing_field(points: np.ndarray, aabbs, target: float, pad: float,
                 cap: float, slope: float = 1.0) -> np.ndarray:
    """Target edge length at each of *points*: *target* within *pad* of any
    region AABB, growing at *slope* per unit distance beyond that, capped at
    *cap*. Distance to the AABB (not the exact shape) errs on the fine side
    for non-box regions — a little over-refinement around their corners, never
    an under-resolved region."""
    d = np.full(len(points), np.inf)
    for lo, hi in aabbs:
        v = np.maximum(np.maximum(lo - points, points - hi), 0.0)
        d = np.minimum(d, np.linalg.norm(v, axis=1))
    return np.clip(target + slope * np.maximum(d - pad, 0.0), target, cap)


def build_sizing(lo: np.ndarray, hi: np.ndarray, boxes, target_edge: float,
                 cap: float) -> BgSizing:
    """The background sizing mesh for the (margin-inflated) domain box
    ``[lo, hi]``: a graded tensor grid, fine (:data:`_BG_CELL_FACTOR` x
    target) near the region AABBs, coarsening to *cap* in the far field. The
    bg box is inflated by one *cap* cell so every PLC point lies strictly
    inside it (a point-location miss segfaults tetgen, pyvista/tetgen#65),
    and every axis keeps >= 4 grid lines (a 1-cell-thin background mesh
    crashes TetGen's reconstruction, verified with tetgen 0.8.4)."""
    aabbs = [region_aabb(b) for b in boxes]
    h_bg = _BG_CELL_FACTOR * target_edge
    pad = _SIZING_PAD_FACTOR * target_edge
    grids = []
    for a in range(3):
        iv = [(alo[a] - pad, ahi[a] + pad) for alo, ahi in aabbs]
        g = graded_axis(lo[a] - cap, hi[a] + cap, iv, h_bg, cap)
        if len(g) < 4:
            g = np.linspace(lo[a] - cap, hi[a] + cap, 4)
        grids.append(g)
    pts, tets = box_grid_tets(*grids)
    return BgSizing(pts, tets, sizing_field(pts, aabbs, target_edge, pad, cap))


# --------------------------------------------------------------------------- #
# deck card formatting (matches the converter's fixed 10/20/20/20 columns)
# --------------------------------------------------------------------------- #

def format_node_lines(ids: np.ndarray, xyz: np.ndarray) -> list[str]:
    """``/NODE`` cards for the new nodes: id in 10 columns, coordinates in 20
    each — parseable both by fixed columns and by whitespace tokens."""
    return [f"{int(i):10d}{x:>20.12g}{y:>20.12g}{z:>20.12g}"
            for i, (x, y, z) in zip(ids, xyz)]


def format_elem_lines(ids: np.ndarray, conn: np.ndarray) -> list[str]:
    """``/TETRA4`` cards for the new elements: five ids in 10 columns each."""
    return [f"{int(i):10d}{int(a):10d}{int(b):10d}{int(c):10d}{int(d):10d}"
            for i, (a, b, c, d) in zip(ids, conn)]


def allocate_ids(existing_max: int, count: int, minimum: int = 0) -> np.ndarray:
    """*count* consecutive new ids starting above *existing_max* and never
    below *minimum* (``design_node_min`` for nodes, so the free-node guard can
    pin them while their elements are void)."""
    start = max(int(existing_max) + 1, int(minimum))
    return np.arange(start, start + count, dtype=np.int64)


# --------------------------------------------------------------------------- #
# TetGen backend (the only tetgen-touching code; optional dependency)
# --------------------------------------------------------------------------- #

def tetgen_backend(points: np.ndarray, tris: np.ndarray,
                   max_volume: float, min_ratio: float,
                   sizing: Optional[BgSizing] = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Constrained tetrahedralisation of the PLC via the ``tetgen`` package.

    ``nobisect`` is TetGen's ``-Y``: the input facets/vertices (the part
    surface) are preserved exactly and Steiner points are inserted only in the
    interior — the property the whole node-conformity contract rests on.
    ``min_ratio`` (``-q``) is the radius-edge quality bound.

    With *sizing* the element sizing comes from the background-mesh metric
    (``-m``): fine only near the regions, so the scaffold outside them stops
    consuming the fine element budget. Without it, ``max_volume`` (``-a``)
    bounds every tet in the domain — the old behaviour, kept for callers
    without a sizing field.

    The ``-m`` run is normalised to a unit-scale box: the wrapper's
    background-mesh path segfaults for domains more than a few absolute units
    across regardless of offset (pyvista/tetgen#65, reproduced with 0.8.4 on
    win/amd64); sizing is scale-invariant and the output is mapped back, the
    preserved surface vertices landing within the coordinate-match tolerance
    of :func:`map_to_original_nodes`."""
    try:
        import pyvista as pv
        import tetgen
    except ImportError as exc:
        raise RuntimeError(
            "growth-mesh generation needs the optional 'tetgen' package "
            "(the pyvista-maintained TetGen wrapper). Install it with:  "
            "pip install \"oropt[growthmesh]\"  — note TetGen itself is "
            "AGPL-licensed (fine to use; evaluate before redistributing)."
        ) from exc
    pts = np.asarray(points, dtype=float)
    kwargs = dict(plc=True, nobisect=True, quality=True,
                  minratio=float(min_ratio), steinerleft=-1, verbose=0)
    offset, scale = np.zeros(3), 1.0
    if sizing is None:
        kwargs.update(fixedvolume=True, maxvolume=float(max_volume))
    else:
        try:
            from tetgen.pytetgen import MTR_POINTDATA_KEY
        except ImportError:                            # pragma: no cover
            MTR_POINTDATA_KEY = "target_size"
        bg_pts = np.asarray(sizing.points, dtype=float)
        offset = bg_pts.min(axis=0)
        scale = float((bg_pts.max(axis=0) - offset).max()) or 1.0
        bg_tets = np.asarray(sizing.tets, dtype=np.int64)
        cells = np.hstack([np.full((len(bg_tets), 1), 4, dtype=np.int64),
                           bg_tets]).ravel()
        celltypes = np.full(len(bg_tets), pv.CellType.TETRA, dtype=np.uint8)
        grid = pv.UnstructuredGrid(cells, celltypes, (bg_pts - offset) / scale)
        grid.point_data[MTR_POINTDATA_KEY] = \
            np.asarray(sizing.mtr, dtype=float) / scale
        kwargs["bgmesh"] = grid
        pts = (pts - offset) / scale
    faces = np.hstack([np.full((len(tris), 1), 3, dtype=np.int64),
                       np.asarray(tris, dtype=np.int64)]).ravel()
    tg = tetgen.TetGen(pv.PolyData(pts, faces))
    out = tg.tetrahedralize(**kwargs)
    return (np.asarray(out[0], dtype=float) * scale + offset,
            np.asarray(out[1], dtype=np.int64))


# --------------------------------------------------------------------------- #
# the PREPARE step
# --------------------------------------------------------------------------- #

@dataclasses.dataclass
class GrowthMeshReport:
    """What one PREPARE run generated and where it wrote it."""
    out_dir: str
    starters: list              # extended starter decks written (str paths)
    engines: list               # engine decks copied verbatim (str paths)
    n_new_nodes: int
    n_new_elems: int
    n_generated: int            # backend output tets before classification
    per_region: list            # (region label, new candidate elements) rows
    target_edge: float          # element sizing used (model units)
    max_volume: float           # target tet volume inside the regions
    quality_min: float          # min/median shape quality of the new elements
    quality_median: float
    node_id_range: tuple        # (first, last) new node ids
    elem_id_range: tuple        # (first, last) new element ids
    total_candidates: int       # extended-deck candidate elements (guard-checked)
    written: bool               # False on --dry-run
    # Highest ORIGINAL design element id -- the original/expansion id boundary
    # carve-off regions need (model.growth_original_elem_max); recorded into the
    # config by point_config_at alongside case_dir.
    original_elem_max: int = 0
    # Auxiliary geometry decks copied into out_dir so the folder is a
    # self-contained case dir: each shape="deck" region's region_rad and the
    # keep-out deck (both resolved relative to case_dir at run time). Str paths.
    region_decks: list = dataclasses.field(default_factory=list)


def report_from_dict(data: dict) -> GrowthMeshReport:
    """Rebuild a :class:`GrowthMeshReport` from its ``dataclasses.asdict``/JSON
    form — the inverse of the CLI's ``--json`` file, which is how the GUI's
    isolated PREPARE subprocess hands its result back. JSON turns the tuples
    into lists (coerced back here); unknown keys (a newer writer) are dropped
    the same way :meth:`oropt.config.Config.from_dict` drops them."""
    known = {f.name for f in dataclasses.fields(GrowthMeshReport)}
    d = {k: v for k, v in data.items() if k in known}
    d["per_region"] = [tuple(r) for r in d.get("per_region", [])]
    for k in ("node_id_range", "elem_id_range"):
        if k in d:
            d[k] = tuple(d[k])
    return GrowthMeshReport(**d)


def _load_case_decks(cfg: Config) -> tuple[list, list[Deck]]:
    """Load every case's starter deck and enforce the shared-mesh contract
    (same design-part element ids), exactly like run start does."""
    cases = cfg.load_case_list()
    if not cases:
        raise ValueError("no load cases defined -- nothing to extend")
    m = cfg.model
    decks = []
    for case in cases:
        if not Path(case.starter).is_file():
            raise ValueError(f"starter deck not found: {case.starter}")
        decks.append(Deck.load(case.starter, m.design_part_id, m.design_node_min))
    for case, deck in zip(cases[1:], decks[1:]):
        if not np.array_equal(deck.elem_ids, decks[0].elem_ids):
            raise ValueError(
                f"load case {case.name!r} (stem {case.stem!r}) has a different "
                f"design-part element set than the primary case; all load cases "
                "must share the same mesh")
    return cases, decks


def prepare_growth_mesh(cfg: Config, size_factor: float = 1.0,
                        min_ratio: float = 1.5,
                        out_dir: Optional[str | Path] = None,
                        write: bool = True,
                        backend: Optional[MeshBackend] = None,
                        log: Callable[[str], None] = print) -> GrowthMeshReport:
    """Generate the candidate mesh for ``cfg.model.growth_boxes`` and write the
    extended deck set (see the module docstring for the full contract).

    *size_factor* scales the target element edge relative to the part's mean
    surface edge length; *min_ratio* is TetGen's ``-q`` radius-edge bound.
    *backend* defaults to :func:`tetgen_backend` (tests inject a fake one).
    ``write=False`` runs everything — generation, classification, splicing and
    the phase-1 guards — but writes nothing (the CLI's ``--dry-run``).

    Raises ``ValueError`` for setup problems (no regions, nothing to add, a
    guard failure) and ``RuntimeError`` when the tetgen package is missing.
    Nothing is written unless every check passed."""
    backend = backend or tetgen_backend
    m = cfg.model
    if not (m.growth_boxes or []):
        raise ValueError("no growth regions configured (model.growth_boxes) -- "
                         "nothing to generate")
    cases, decks = _load_case_decks(cfg)
    primary = decks[0]
    resolved = resolve_growth_boxes(primary, m.growth_boxes, m.case_dir)
    # POSITIVE (add-material) regions define where to generate candidate mesh;
    # NEGATIVE (forbid=True) regions are forbidden space -- an inline keep-out
    # that only drops generated tets (never meshed themselves, see classify).
    boxes = [b for b in resolved if not getattr(b, "forbid", False)]
    forbid_boxes = [b for b in resolved if getattr(b, "forbid", False)]
    if not boxes:
        raise ValueError("no POSITIVE growth regions configured "
                         "(model.growth_boxes) -- every configured region is "
                         "negative (forbid=true), so there is nothing to mesh")
    labels = [b.name or f"#{i + 1}" for i, b in enumerate(boxes)]
    # Resolve the keep-out geometry (if any) BEFORE the expensive TetGen run so a
    # missing/unparsable deck fails fast; its inside test drops generated
    # candidate tets that land in the neighbour parts (see the classify step).
    keepout = resolve_keepout(m, m.case_dir)

    # ---- part exterior surface + sizing -------------------------------------
    faces = exterior_faces(primary.elem_conn)
    surf_ids = np.unique(faces)
    id_to_row = {int(v): i for i, v in enumerate(primary.node_ids)}
    try:
        surf_xyz = primary.node_xyz[[id_to_row[int(v)] for v in surf_ids]]
    except KeyError as exc:
        raise ValueError(f"design element references node id {exc} absent from "
                         "the /NODE block") from exc
    if len(np.unique(np.round(surf_xyz, 9), axis=0)) != len(surf_xyz):
        raise ValueError(
            "the design part's surface has coincident nodes -- the generated "
            "mesh could not be mapped back to unique node ids. Merge/"
            "equivalence the duplicate nodes first")
    surf_tris = np.searchsorted(surf_ids, faces)       # surf_ids is sorted (unique)
    edge = mean_edge_length(surf_xyz, surf_tris)
    target_edge = float(size_factor) * edge
    max_volume = target_edge ** 3 / (6.0 * np.sqrt(2.0))   # regular-tet volume

    # ---- PLC domain: AABB(part surface ∪ regions), slightly inflated --------
    rlo, rhi = region_bounds(boxes)
    lo = np.minimum(surf_xyz.min(axis=0), rlo)
    hi = np.maximum(surf_xyz.max(axis=0), rhi)
    diag = float(np.linalg.norm(hi - lo))
    # Far-field element scale: the -m sizing cap and the coarse wall cell.
    cap_edge = max(diag / 16.0, 2.0 * target_edge)
    # Walls subdivided at ~2x the target edge near the regions, coarsening to
    # ~cap_edge away from them; the margin keeps the regions clear of the fine
    # wall facets' encroachment zones and the part surface clear of the coarse
    # ones (see box_shell).
    margin = max(4.0 * target_edge, 0.6 * cap_edge, 1e-3 * diag)
    dlo, dhi = lo - margin, hi + margin
    pad = _SIZING_PAD_FACTOR * target_edge
    aabbs = [region_aabb(b) for b in boxes]
    fine_iv = [[(alo[a] - pad, ahi[a] + pad) for alo, ahi in aabbs]
               for a in range(3)]
    points, tris = build_plc(surf_xyz, surf_tris, dlo, dhi,
                             wall_cell=2.0 * target_edge,
                             fine_intervals=fine_iv, coarse_cell=cap_edge)
    sizing = build_sizing(dlo, dhi, boxes, target_edge, cap_edge)

    log(f"[oropt] growth-mesh: part surface {len(surf_tris)} triangles / "
        f"{len(surf_ids)} nodes; mean edge {edge:.4g}, target edge "
        f"{target_edge:.4g} (size_factor {size_factor}), target tet volume "
        f"{max_volume:.4g}")
    log(f"[oropt] growth-mesh: sizing field {target_edge:.4g} inside the "
        f"regions (+{pad:.3g} halo) growing to {cap_edge:.4g} far away; "
        f"background mesh {len(sizing.points)} nodes / {len(sizing.tets)} "
        f"tets")
    log(f"[oropt] growth-mesh: tetrahedralising the PLC "
        f"({len(points)} points) ...")
    out_points, out_tets = backend(points, tris, max_volume, min_ratio, sizing)
    log(f"[oropt] growth-mesh: backend returned {len(out_tets)} tets / "
        f"{len(out_points)} points")

    # ---- classify: keep centroids inside a region and outside the part ------
    cent = out_points[out_tets].mean(axis=1)
    in_region = np.zeros(len(cent), dtype=bool)
    for b in boxes:
        in_region |= primitive_member(cent, b)
    part_xyz = primary.node_xyz[Mesh.from_deck(primary).conn_rows]
    inside_part = np.zeros(len(cent), dtype=bool)
    inside_part[in_region] = points_in_tets(cent[in_region], part_xyz)
    keep = in_region & ~inside_part
    # Keep-out: never generate candidate tets inside the neighbour parts, so the
    # extended deck has no growable material there (the run-time
    # growth_candidate_mask holds any that survive void, but the clean fix is to
    # not create them at all). ``keepout`` was resolved up front.
    if keepout is not None:
        blocked = keep & keepout.block_mask(cent)
        n_blocked = int(blocked.sum())
        if n_blocked:
            keep = keep & ~blocked
            log(f"[oropt] growth-mesh: keep-out ({Path(keepout.source).name}) "
                f"removed {n_blocked} generated candidate tet(s) inside the "
                "neighbour parts")
    # Negative (forbid=true) regions: forbidden growth space, so no candidate
    # tets are generated inside them either -- same treatment as the keep-out.
    if forbid_boxes:
        forb = np.zeros(len(cent), dtype=bool)
        for b in forbid_boxes:
            forb |= primitive_member(cent, b)
        blocked = keep & forb
        n_blocked = int(blocked.sum())
        if n_blocked:
            keep = keep & ~blocked
            log(f"[oropt] growth-mesh: negative (forbidden) region(s) removed "
                f"{n_blocked} generated candidate tet(s)")
    if not keep.any():
        raise ValueError(
            "the generated mesh adds no candidate elements: every tet inside "
            "a growth region duplicates existing part elements (the regions "
            "appear fully pre-meshed already), lands inside the keep-out or a "
            "negative (forbidden) region, or no tet landed in a region")
    kept = out_tets[keep]
    per_region = [(lbl, int(primitive_member(cent[keep], b).sum()))
                  for lbl, b in zip(labels, boxes)]

    # ---- map preserved surface vertices back to original node ids -----------
    tol = max(1e-9, 1e-9 * diag)
    vert_map = map_to_original_nodes(out_points, surf_xyz, surf_ids, tol)
    used = np.unique(kept)
    new_vert = used[vert_map[used] < 0]

    # Over ALL /NODE blocks, not deck.node_ids (the first block only): converter
    # output carries the other includes' nodes in later blocks, and a new id
    # colliding with one of those is a starter-fatal duplicate declaration.
    node_max = max(d.max_node_id() for d in decks)
    elem_max = max(int(d.elem_ids.max()) for d in decks)
    new_node_ids = allocate_ids(node_max, len(new_vert), m.design_node_min)
    vert_map = vert_map.copy()
    vert_map[new_vert] = new_node_ids
    new_elem_ids = allocate_ids(elem_max, len(kept))

    # new elements in the deck's own node-ordering convention
    part_sign = float(np.sign(np.median(signed_volumes(part_xyz))) or 1.0)
    kept = orient_tets(kept, out_points, part_sign)
    conn = vert_map[kept]

    node_lines = format_node_lines(new_node_ids, out_points[new_vert])
    elem_lines = format_elem_lines(new_elem_ids, conn)

    # ---- phase-1 guards on the extended primary deck, BEFORE any write ------
    ext_lines = primary.extended_lines(node_lines, elem_lines)
    ext_deck = Deck(list(ext_lines), primary.newline, m.design_part_id,
                    m.design_node_min)
    if ext_deck.n_design_elements != primary.n_design_elements + len(kept):
        raise ValueError("internal error: extended deck element count mismatch")
    # The re-check runs with the just-computed original/expansion id boundary
    # filled in, so a carve-off region is judged exactly as the eventual run
    # will judge it (once point_config_at records the boundary in the config).
    m_run = dataclasses.replace(m, growth_original_elem_max=elem_max)
    try:
        total = int(growth_candidate_mask(
            ext_deck, Mesh.from_deck(ext_deck), m_run, log=lambda _s: None).sum())
    except ValueError as exc:
        raise ValueError(
            f"the generated mesh failed the phase-1 run-start guards: {exc}"
        ) from exc

    q = tet_quality(out_points[kept])
    report = GrowthMeshReport(
        out_dir="", starters=[], engines=[],
        n_new_nodes=len(new_vert), n_new_elems=len(kept),
        n_generated=len(out_tets), per_region=per_region,
        target_edge=target_edge, max_volume=max_volume,
        quality_min=float(q.min()), quality_median=float(np.median(q)),
        node_id_range=((int(new_node_ids[0]), int(new_node_ids[-1]))
                       if len(new_node_ids) else (0, 0)),
        elem_id_range=(int(new_elem_ids[0]), int(new_elem_ids[-1])),
        total_candidates=total, written=bool(write),
        original_elem_max=elem_max)
    log(f"[oropt] growth-mesh: {report.n_new_elems} new candidate elements, "
        f"{report.n_new_nodes} new nodes "
        f"(quality min {report.quality_min:.3f} / median "
        f"{report.quality_median:.3f}); guards passed "
        f"({total} candidate elements on the extended deck)")
    for lbl, n in per_region:
        log(f"[oropt] growth-mesh: region {lbl!r}: {n} new elements")

    # ---- write the extended deck set ----------------------------------------
    dest = Path(out_dir) if out_dir else Path(m.case_dir) / GROWTH_MESH_DIRNAME
    report.out_dir = str(dest)
    if not write:
        log("[oropt] growth-mesh: dry run -- nothing written")
        return report
    dest.mkdir(parents=True, exist_ok=True)
    for case, deck in zip(cases, decks):
        starter_out = dest / f"{case.stem}_0000.rad"
        if starter_out.resolve() == Path(case.starter).resolve():
            raise ValueError(f"output {starter_out} would overwrite the source "
                             "starter deck -- choose another --out-dir")
        text = deck.newline.join(deck.extended_lines(node_lines, elem_lines))
        Path(starter_out).write_text(text + deck.newline, encoding="utf-8",
                                     newline="")
        report.starters.append(str(starter_out))
        engine_out = dest / f"{case.stem}_0001.rad"
        if Path(case.engine).is_file():
            shutil.copy2(case.engine, engine_out)
            report.engines.append(str(engine_out))
        else:
            log(f"[oropt] growth-mesh: engine deck missing, not copied: "
                f"{case.engine}")
    # Copy the auxiliary geometry decks the config references RELATIVE to case_dir
    # -- every shape="deck" growth region's region_rad and the keep-out deck --
    # into the output folder too, so pointing model.case_dir here yields a
    # SELF-CONTAINED, runnable deck set. Without this, the run (whose case_dir is
    # now this folder) resolves those relative paths under it and fails with
    # "region deck not found" / "keep-out deck not found". Absolute paths already
    # resolve regardless of case_dir, so they are left untouched.
    aux_rel: list[str] = []
    for b in resolved:
        if b.shape_kind() == "deck" and (getattr(b, "region_rad", "") or "").strip():
            aux_rel.append(b.region_rad.strip())
    if getattr(m, "growth_keepout_rad", None):
        aux_rel.append(str(m.growth_keepout_rad).strip())
    seen: set = set()
    for rel in aux_rel:
        if rel in seen or Path(rel).is_absolute():
            continue                                 # dup, or resolves without a copy
        seen.add(rel)
        src = Path(m.case_dir) / rel
        out = dest / rel
        if not src.is_file():
            log(f"[oropt] growth-mesh: referenced deck missing, not copied: {src}")
            continue
        if src.resolve() == out.resolve():
            continue                                 # in-place run: same file
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out)
        report.region_decks.append(str(out))
        log(f"[oropt] growth-mesh: copied referenced deck {rel!r} -> {out}")
    log(f"[oropt] growth-mesh: extended deck set written to {dest} -- point "
        f"model.case_dir there to use it")
    return report


def point_config_at(cfg_path: str | Path, out_dir: str | Path,
                    original_elem_max: Optional[int] = None) -> None:
    """Rewrite ``model.case_dir`` in the YAML at *cfg_path* to *out_dir* — the
    one-line config change that makes a run use the extended deck set — and,
    when *original_elem_max* is given (:attr:`GrowthMeshReport.
    original_elem_max`), record it as ``model.growth_original_elem_max`` so
    carve-off regions can tell the original part from the generated expansion
    elements. Only those keys are touched; everything else round-trips through
    the same ``safe_load``/``safe_dump`` pair the app's own Save uses."""
    raw = Config.read_yaml_dict(cfg_path)
    raw.setdefault("model", {})["case_dir"] = str(out_dir)
    if original_elem_max is not None:
        raw["model"]["growth_original_elem_max"] = int(original_elem_max)
    Path(cfg_path).write_text(
        yaml.safe_dump(raw, sort_keys=False, default_flow_style=False),
        encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="oropt-growthmesh",
        description="Generate the growth-region candidate mesh (TetGen) and "
                    "write extended starter decks for every load case")
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--size-factor", type=float, default=1.0,
                    help="target element edge as a multiple of the part's mean "
                         "surface edge length (default 1.0)")
    ap.add_argument("--min-ratio", type=float, default=1.5,
                    help="TetGen -q radius-edge quality bound (default 1.5)")
    ap.add_argument("--out-dir", default=None,
                    help="output folder for the extended deck set (default "
                         f"<case_dir>/{GROWTH_MESH_DIRNAME})")
    ap.add_argument("--dry-run", action="store_true",
                    help="generate, classify and guard-check but write nothing")
    ap.add_argument("--json", default=None, metavar="PATH",
                    help="on success, also write the report as JSON to PATH — "
                         "how the GUI's isolated PREPARE subprocess hands the "
                         "report back (see oropt.gui.growthprep)")
    args = ap.parse_args(argv)

    cfg = Config.from_yaml(args.config)
    try:
        rep = prepare_growth_mesh(
            cfg, size_factor=args.size_factor, min_ratio=args.min_ratio,
            out_dir=args.out_dir, write=not args.dry_run,
            log=lambda s: print(s, flush=True))
    except (ValueError, RuntimeError) as exc:
        print(f"[oropt] growth-mesh: ERROR: {exc}", flush=True)
        return 2
    if args.json:
        Path(args.json).write_text(json.dumps(dataclasses.asdict(rep)),
                                   encoding="utf-8")
    if rep.written:
        print(f"[oropt] growth-mesh: done. To run on the extended decks set\n"
              f"    model.case_dir: {rep.out_dir}\n"
              f"    model.growth_original_elem_max: {rep.original_elem_max}\n"
              f"in {args.config} (or use the GUI button). The second key is the "
              "original/expansion element-id boundary carve-off regions need.",
              flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
