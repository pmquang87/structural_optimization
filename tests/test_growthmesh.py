"""Growth-mesh PREPARE step (phase 2): auto-generating candidate elements
inside growth regions by direct node/element creation (oropt.growthmesh).

The core is hermetic — surface extraction, PLC assembly, classification, id
allocation, card formatting, deck splicing and the guard integration are all
exercised with a *fabricated* meshing backend injected in place of TetGen, so
the whole pipeline runs without the optional dependency. The TetGen-backed
end-to-end tests sit behind ``pytest.importorskip("tetgen")`` so the suite
passes either way.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from oropt.config import Config, GrowthBox, LoadCase
from oropt.deck import Deck, _parse_elem, _parse_node
from oropt.growthmesh import (GROWTH_MESH_DIRNAME, GrowthMeshReport,
                              allocate_ids, box_grid_tets, box_shell,
                              build_plc, build_sizing, exterior_faces,
                              format_elem_lines, format_node_lines,
                              graded_axis, main, map_to_original_nodes,
                              mean_edge_length, orient_tets, point_config_at,
                              points_in_tets, prepare_growth_mesh, region_aabb,
                              region_bounds, report_from_dict, signed_volumes,
                              sizing_field, tet_quality, tetgen_backend)
from oropt.loop import growth_candidate_mask
from oropt.mesh import Mesh

from conftest import MINI_DECK

# Region attached to the part's x=0 face (nodes 60000001/3/4 of MINI_DECK):
# generated tets there share those surface nodes -> reachable.
WING = GrowthBox(name="wing", x_min=-1.0, x_max=-0.001, y_min=-0.5, y_max=1.5,
                 z_min=-0.5, z_max=1.5)
# Region carving the part's second tet (centroid (.5,.5,.5)): carving must now
# be opted into explicitly (carve defaults to off / part kept intact).
CARVE = GrowthBox(name="carve", shape="sphere", cx=0.5, cy=0.5, cz=0.5,
                  radius=0.11, carve=True)

_TWO_TETS = np.array([[60000001, 60000002, 60000003, 60000004],
                      [60000002, 60000003, 60000004, 60000005]])


def _silent(_msg):
    pass


def _mini_cfg(tmp_path, boxes, stem="mini") -> Config:
    cfg = Config()
    cfg.model.case_dir = str(tmp_path)
    cfg.model.growth_boxes = list(boxes)
    cfg.load_cases = [LoadCase(name="c", stem=stem, sigma_allow=1.0)]
    return cfg


def _write_mini(tmp_path, stem="mini", extra_node: str = "") -> Path:
    text = MINI_DECK
    if extra_node:
        text = text.replace("/PART/60000000", extra_node + "\n/PART/60000000")
    p = tmp_path / f"{stem}_0000.rad"
    p.write_text(text, encoding="utf-8")
    return p


def _fake_backend(points, tris, max_volume, min_ratio, sizing=None):
    """Fabricated 'TetGen': two good candidates on the x=0 face, one duplicate
    of an existing part element, one tet far outside every region."""
    surf = points[:5]                       # build_plc keeps surface points first
    extra = np.array([[-0.5, 0.5, 0.25],    # 5: q0, in WING
                      [-0.5, 0.25, 0.75],   # 6: q1, in WING
                      [5.0, 5.0, 5.0],      # 7: far away
                      [5.0, 6.0, 5.0]])     # 8: far away
    pts = np.vstack([surf, extra])
    tets = np.array([
        [0, 2, 3, 5],       # n1,n3,n4 + q0 -> kept (WING, conformal)
        [0, 2, 5, 6],       # n1,n3 + q0,q1 -> kept (WING, chained)
        [1, 2, 3, 4],       # n2,n3,n4,n5 = existing e2 -> inside part, dropped
        [1, 4, 7, 8],       # outside every region -> dropped
    ])
    return pts, tets


# ---- exterior surface extraction ---------------------------------------------
def test_exterior_faces_two_tets_drop_shared_face():
    faces = exterior_faces(_TWO_TETS)
    assert len(faces) == 6                          # 8 faces - the shared pair
    keys = {tuple(sorted(f)) for f in faces}
    assert (60000002, 60000003, 60000004) not in keys   # the internal face
    assert (60000001, 60000002, 60000003) in keys


def test_exterior_faces_single_tet_all_four():
    faces = exterior_faces(_TWO_TETS[:1])
    assert len(faces) == 4
    assert {tuple(sorted(f)) for f in faces} == {
        (60000001, 60000002, 60000003), (60000001, 60000002, 60000004),
        (60000001, 60000003, 60000004), (60000002, 60000003, 60000004)}


def test_mean_edge_length_unit_face():
    pts = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    tri = np.array([[0, 1, 2]])
    assert mean_edge_length(pts, tri) == pytest.approx((1 + 1 + np.sqrt(2)) / 3)


# ---- PLC assembly -------------------------------------------------------------
def test_box_shell_coarse_is_plain_box():
    pts, tris = box_shell(np.zeros(3), np.ones(3), cell=10.0)
    assert len(pts) == 8 and len(tris) == 12
    assert len(np.unique(pts, axis=0)) == 8         # corners merged, not repeated


def test_box_shell_subdivided_watertight():
    pts, tris = box_shell(np.zeros(3), np.array([2.0, 1.0, 1.0]), cell=0.5)
    assert len(np.unique(pts, axis=0)) == len(pts)  # no duplicate points
    assert tris.min() >= 0 and tris.max() < len(pts)
    # closed surface: every edge is shared by exactly two triangles
    edges = np.sort(tris[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert set(counts.tolist()) == {2}
    # subdivision actually happened at ~cell size
    assert len(tris) > 12


def test_build_plc_surface_points_first():
    surf = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0]])
    tris = np.array([[0, 1, 2]])
    pts, out = build_plc(surf, tris, np.full(3, -2.0), np.full(3, 3.0), 10.0)
    assert np.array_equal(pts[:3], surf)            # indices preserved
    assert np.array_equal(out[0], tris[0])
    assert len(pts) == 3 + 8 and len(out) == 1 + 12


def test_region_bounds_all_shapes():
    lo, hi = region_bounds([
        GrowthBox(shape="sphere", cx=1.0, cy=1.0, cz=1.0, radius=0.5),
        GrowthBox(shape="cylinder", x1=2.0, y1=0.0, z1=0.0, x2=3.0, y2=0.0,
                  z2=0.0, radius=0.25),
        GrowthBox(x_min=-1.0, x_max=0.0, y_min=0.0, y_max=1.0, z_min=0.0,
                  z_max=1.0)])
    assert np.allclose(lo, [-1.0, -0.25, -0.25])
    assert np.allclose(hi, [3.25, 1.5, 1.5])


def test_region_bounds_polyhedron_points_min_max():
    lo, hi = region_bounds([GrowthBox(shape="polyhedron", points=[
        [-0.5, 0.0, 0.25], [1.0, -2.0, 0.0], [0.0, 3.0, 0.5], [0.5, 0.5, 4.0]])])
    assert np.allclose(lo, [-0.5, -2.0, 0.0])
    assert np.allclose(hi, [1.0, 3.0, 4.0])


# ---- background sizing mesh (the -m field replacing the global -a bound) -------
def test_graded_axis_fine_in_band_coarse_far():
    g = graded_axis(0.0, 100.0, [(10.0, 14.0)], fine=1.0, coarse=10.0)
    assert g[0] == 0.0 and g[-1] == 100.0
    steps = np.diff(g)
    assert (steps > 0).all()
    # ~fine spacing across the band, coarse cap far away from it
    mids = (g[:-1] + g[1:]) / 2
    in_band = (mids > 10.0) & (mids < 14.0)
    assert in_band.any() and steps[in_band].max() <= 1.0 + 1e-9
    far = mids > 60.0
    assert steps[far].min() > 5.0
    # and vastly fewer lines than a uniformly fine grid would need
    assert len(g) < 40


def test_graded_axis_no_intervals_all_coarse():
    g = graded_axis(0.0, 100.0, [], fine=1.0, coarse=10.0)
    assert g[0] == 0.0 and g[-1] == 100.0
    assert np.diff(g).min() > 5.0


def test_box_grid_tets_fill_conform_orient():
    gx = np.array([0.0, 1.0, 3.0])
    gy = np.array([0.0, 2.0])
    gz = np.array([0.0, 1.0, 2.0])
    pts, tets = box_grid_tets(gx, gy, gz)
    assert len(pts) == 3 * 2 * 3 and len(tets) == 6 * (2 * 1 * 2)
    vols = signed_volumes(pts[tets])
    assert (vols > 0).all()                          # consistent orientation
    assert vols.sum() == pytest.approx(3.0 * 2.0 * 2.0)   # fills the box
    # conforming: every face is shared by exactly two tets or lies on the hull
    faces = np.sort(tets[:, [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]]]
                    .reshape(-1, 3), axis=1)
    _, counts = np.unique(faces, axis=0, return_counts=True)
    assert set(counts.tolist()) <= {1, 2}


def test_sizing_field_target_pad_slope_cap():
    aabbs = [(np.zeros(3), np.ones(3))]
    pts = np.array([[0.5, 0.5, 0.5],     # inside -> target
                    [1.5, 0.5, 0.5],     # within pad -> target
                    [3.0, 0.5, 0.5],     # d=2: target + slope*(2-pad)
                    [90.0, 0.5, 0.5]])   # far -> cap
    mtr = sizing_field(pts, aabbs, target=0.5, pad=1.0, cap=4.0, slope=1.0)
    assert mtr[0] == 0.5 and mtr[1] == 0.5
    assert mtr[2] == pytest.approx(0.5 + (2.0 - 1.0))
    assert mtr[3] == 4.0


def test_build_sizing_covers_domain_fine_only_near_region():
    region = GrowthBox(name="r", x_min=0.0, x_max=2.0, y_min=0.0, y_max=2.0,
                       z_min=0.0, z_max=2.0)
    lo, hi = np.full(3, -1.0), np.full(3, 40.0)
    s = build_sizing(lo, hi, [region], target_edge=0.25, cap=5.0)
    assert (s.points.min(axis=0) <= lo).all()        # bg box encloses domain
    assert (s.points.max(axis=0) >= hi).all()
    assert len(s.mtr) == len(s.points)
    assert s.mtr.min() == pytest.approx(0.25)        # target reached...
    inside = ((s.points >= 0.0) & (s.points <= 2.0)).all(axis=1)
    assert inside.any() and np.allclose(s.mtr[inside], 0.25)
    far = (s.points > 20.0).all(axis=1)
    assert far.any() and (s.mtr[far] == 5.0).all()   # ...and capped far away
    # graded: far fewer nodes than a uniformly fine grid (41/0.25=164^3)
    assert len(s.points) < 30_000


def test_build_sizing_tiny_domain_keeps_min_grid():
    region = GrowthBox(name="r", x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
                       z_min=0.0, z_max=1.0)
    s = build_sizing(np.zeros(3), np.ones(3), [region], target_edge=5.0,
                     cap=10.0)
    # >= 4 grid lines per axis even when one coarse cell would span the box
    # (a 1-cell background mesh crashes TetGen's reconstruction)
    for a in range(3):
        assert len(np.unique(s.points[:, a])) >= 4


def test_region_aabb_matches_region_bounds_per_shape():
    shapes = [GrowthBox(shape="sphere", cx=1.0, cy=1.0, cz=1.0, radius=0.5),
              GrowthBox(shape="cylinder", x1=2.0, y1=0.0, z1=0.0, x2=3.0,
                        y2=0.0, z2=0.0, radius=0.25),
              GrowthBox(x_min=-1.0, x_max=0.0, y_min=0.0, y_max=1.0,
                        z_min=0.0, z_max=1.0)]
    for b in shapes:
        lo, hi = region_aabb(b)
        blo, bhi = region_bounds([b])
        assert np.allclose(lo, blo) and np.allclose(hi, bhi)


def test_build_plc_graded_walls_fewer_points_than_uniform():
    surf = np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0]])
    tris = np.array([[0, 1, 2]])
    lo, hi = np.full(3, -50.0), np.full(3, 50.0)
    fine_iv = [[(-2.0, 2.0)]] * 3
    p_uni, _ = build_plc(surf, tris, lo, hi, wall_cell=1.0)
    p_grad, t_grad = build_plc(surf, tris, lo, hi, wall_cell=1.0,
                               fine_intervals=fine_iv, coarse_cell=10.0)
    assert len(p_grad) < len(p_uni) / 4
    # the graded shell is still watertight
    shell = t_grad[len(tris):]
    edges = np.sort(shell[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert set(counts.tolist()) == {2}


def test_prepare_passes_sizing_field_to_backend(tmp_path, mini_engine_path):
    """prepare wires a BgSizing into the backend: mtr equals the target edge
    at background nodes inside the region and never exceeds the far cap."""
    _write_mini(tmp_path)
    seen = {}

    def spy(points, tris, max_volume, min_ratio, sizing=None):
        seen["sizing"] = sizing
        return _fake_backend(points, tris, max_volume, min_ratio)

    rep = prepare_growth_mesh(_mini_cfg(tmp_path, [WING]), backend=spy,
                              log=_silent)
    s = seen["sizing"]
    assert s is not None and len(s.mtr) == len(s.points)
    assert s.mtr.min() == pytest.approx(rep.target_edge)
    # every bg node within the sizing halo of the region AABB carries the
    # target edge (bg cells are bigger than this tiny region, so testing
    # nodes strictly inside it would test nothing)
    lo, hi = region_aabb(WING)
    d = np.linalg.norm(np.maximum(np.maximum(lo - s.points,
                                             s.points - hi), 0.0), axis=1)
    near = d <= 8.0 * rep.target_edge
    assert near.any() and np.allclose(s.mtr[near], rep.target_edge)


# ---- classification geometry ---------------------------------------------------
def test_points_in_tets_inside_boundary_outside():
    tet = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]]])
    pts = np.array([[0.1, 0.1, 0.1],     # inside
                    [0.0, 0.0, 0.0],     # vertex (on boundary counts as inside)
                    [1.0, 1.0, 1.0],     # outside
                    [-0.01, 0.1, 0.1]])  # just outside
    assert points_in_tets(pts, tet).tolist() == [True, True, False, False]


def test_points_in_tets_empty_inputs():
    tet = np.zeros((0, 4, 3))
    assert points_in_tets(np.zeros((0, 3)), tet).size == 0
    assert not points_in_tets(np.array([[0., 0, 0]]), tet).any()


def test_points_in_tets_skips_degenerate():
    flat = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]])  # coplanar
    assert not points_in_tets(np.array([[0.3, 0.3, 0.0]]), flat).any()


def test_orient_tets_matches_requested_sign():
    pts = np.array([[0., 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    good = np.array([[0, 1, 2, 3]])                 # signed volume +1/6
    bad = np.array([[0, 2, 1, 3]])                  # flipped
    both = np.vstack([good, bad])
    out = orient_tets(both, pts, sign=1.0)
    assert (signed_volumes(pts[out]) > 0).all()
    out = orient_tets(both, pts, sign=-1.0)
    assert (signed_volumes(pts[out]) < 0).all()


def test_tet_quality_regular_and_degenerate():
    a = 1.0 / np.sqrt(2.0)                          # regular tet, edge 1
    reg = np.array([[[a, 0, -a / np.sqrt(2)], [-a, 0, -a / np.sqrt(2)],
                     [0, a, a / np.sqrt(2)], [0, -a, a / np.sqrt(2)]]])
    flat = np.array([[[0., 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]]])
    q = tet_quality(np.vstack([reg, flat]))
    assert q[0] == pytest.approx(1.0, abs=1e-6)
    assert q[1] == 0.0


def test_map_to_original_nodes_exact_and_new():
    surf_xyz = np.array([[0., 0, 0], [1, 0, 0]])
    surf_ids = np.array([60000001, 60000002])
    out_pts = np.array([[1., 0, 0], [-0.5, 0.5, 0.5], [0., 0, 0]])
    got = map_to_original_nodes(out_pts, surf_xyz, surf_ids, tol=1e-9)
    assert got.tolist() == [60000002, -1, 60000001]


# ---- id allocation + card formatting -------------------------------------------
def test_allocate_ids_above_max_and_clamped_to_node_min():
    assert allocate_ids(60000005, 3, 60000000).tolist() == [
        60000006, 60000007, 60000008]
    # rigid-range max below design_node_min -> clamp up
    assert allocate_ids(10000001, 2, 60000000).tolist() == [60000000, 60000001]
    assert allocate_ids(5, 0, 0).size == 0


def test_format_node_lines_parse_back_both_ways():
    [line] = format_node_lines(np.array([60000042]),
                               np.array([[-1.5, 2.25, -1234.56789]]))
    nid, x, y, z = _parse_node(line)                 # token path
    assert (nid, x, y, z) == (60000042, -1.5, 2.25, -1234.56789)
    # fixed 10/20/20/20 columns hold too
    assert int(line[0:10]) == 60000042
    assert float(line[10:30]) == -1.5
    assert float(line[30:50]) == 2.25
    assert float(line[50:70]) == -1234.56789


def test_format_elem_lines_parse_back_both_ways():
    [line] = format_elem_lines(np.array([60000003]),
                               np.array([[60000001, 60000003, 60000004,
                                          60000006]]))
    assert _parse_elem(line) == (60000003, 60000001, 60000003, 60000004,
                                 60000006)
    assert [int(line[i:i + 10]) for i in range(0, 50, 10)] == [
        60000003, 60000001, 60000003, 60000004, 60000006]


# ---- deck splicing --------------------------------------------------------------
def test_extended_lines_splice_and_reload(mini_deck_path):
    deck = Deck.load(mini_deck_path, 60000000, 60000000)
    node_lines = format_node_lines(np.array([60000006]),
                                   np.array([[-0.5, 0.5, 0.25]]))
    elem_lines = format_elem_lines(np.array([60000003]),
                                   np.array([[60000001, 60000003, 60000004,
                                              60000006]]))
    ext = Deck(deck.extended_lines(node_lines, elem_lines), deck.newline,
               60000000, 60000000)
    assert deck.n_design_elements == 2              # original untouched
    assert ext.n_design_elements == 3
    assert 60000006 in ext.node_ids and 60000003 in ext.elem_ids
    # everything outside the two blocks is verbatim
    assert ext.lines[-1].strip() == "/END"
    assert sum(1 for ln in ext.lines if ln.strip() == "/PROP/SOLID/3") == 1
    # the spliced deck still write()-round-trips
    out = mini_deck_path.parent / "ext_0000.rad"
    ext.write(out, np.ones(3, dtype=bool))
    again = Deck.load(out, 60000000, 60000000)
    assert again.n_design_elements == 3
    assert np.array_equal(again.elem_ids, ext.elem_ids)


# ---- the full PREPARE pipeline with a fabricated backend ------------------------
def test_prepare_pipeline_hermetic(tmp_path, mini_engine_path):
    _write_mini(tmp_path)
    cfg = _mini_cfg(tmp_path, [WING, CARVE])
    rep = prepare_growth_mesh(cfg, backend=_fake_backend, log=_silent)

    assert rep.n_new_elems == 2 and rep.n_new_nodes == 2
    assert rep.n_generated == 4
    assert rep.per_region == [("wing", 2), ("carve", 0)]
    assert rep.node_id_range == (60000006, 60000007)
    assert rep.elem_id_range == (60000003, 60000004)
    assert rep.written and rep.out_dir == str(tmp_path / GROWTH_MESH_DIRNAME)

    starter = tmp_path / GROWTH_MESH_DIRNAME / "mini_0000.rad"
    engine = tmp_path / GROWTH_MESH_DIRNAME / "mini_0001.rad"
    assert starter.is_file()
    assert engine.is_file()                          # copied verbatim
    assert engine.read_text(encoding="utf-8") == \
        mini_engine_path.read_text(encoding="utf-8")

    ext = Deck.load(starter, 60000000, 60000000)
    assert ext.n_design_elements == 4                # 2 old + 2 new
    mask = growth_candidate_mask(ext, Mesh.from_deck(ext), cfg.model,
                                 log=_silent)
    # candidates: carved e2 + the two new elements; e1 stays alive
    assert mask.sum() == rep.total_candidates == 3
    assert not mask[ext.elem_ids.tolist().index(60000001)]
    # conformity: the new elements reference original surface node ids
    new_rows = np.isin(ext.elem_ids, [60000003, 60000004])
    assert 60000001 in ext.elem_conn[new_rows]
    # node-ordering convention: new elements share the part's positive sign
    xyz = ext.node_xyz[Mesh.from_deck(ext).conn_rows]
    assert (signed_volumes(xyz[new_rows]) > 0).all()


def test_prepare_pipeline_hermetic_polyhedron_region(tmp_path, mini_engine_path):
    """The PREPARE step works unchanged with a polyhedron region: the WING box
    volume expressed as an explicit 8-node point set selects the same fabricated
    candidates (region_bounds feeds the PLC AABB, primitive_member classifies)."""
    _write_mini(tmp_path)
    poly_wing = GrowthBox(name="pwing", shape="polyhedron", points=[
        [x, y, z] for x in (-1.0, -0.001) for y in (-0.5, 1.5)
        for z in (-0.5, 1.5)])
    cfg = _mini_cfg(tmp_path, [poly_wing])
    rep = prepare_growth_mesh(cfg, backend=_fake_backend, log=_silent)
    assert rep.n_new_elems == 2 and rep.n_new_nodes == 2
    assert rep.per_region == [("pwing", 2)]
    ext = Deck.load(tmp_path / GROWTH_MESH_DIRNAME / "mini_0000.rad",
                    60000000, 60000000)
    mask = growth_candidate_mask(ext, Mesh.from_deck(ext), cfg.model,
                                 log=_silent)
    assert int(mask.sum()) == rep.total_candidates == 2


def test_prepare_overlapping_region_carve_off(tmp_path, mini_engine_path):
    """A region overlapping the part is fine either way; by default (carve off)
    the overlapped original element stays alive on the extended deck -- only
    the generated expansion elements are candidates. carve: true opts into the
    carve-and-regrow variant. The report carries the original/expansion id
    boundary carve-off regions need at run time."""
    _write_mini(tmp_path)
    # covers the fake backend's two new tets AND e1's centroid (.25,.25,.25)
    overlap = GrowthBox(name="olap", x_min=-1.0, x_max=0.4, y_min=-0.5,
                        y_max=1.5, z_min=-0.5, z_max=1.5)
    rep = prepare_growth_mesh(_mini_cfg(tmp_path, [overlap]),
                              backend=_fake_backend, log=_silent)
    assert rep.n_new_elems == 2
    assert rep.total_candidates == 2             # e1 stays alive (default)
    assert rep.original_elem_max == 60000002

    carving = prepare_growth_mesh(
        _mini_cfg(tmp_path, [dataclasses.replace(overlap, carve=True)]),
        backend=_fake_backend, log=_silent)
    assert carving.n_new_elems == 2
    assert carving.total_candidates == 3         # 2 new + carved original e1
    assert carving.original_elem_max == 60000002

    # the run agrees once the config carries the recorded boundary
    ext = Deck.load(tmp_path / GROWTH_MESH_DIRNAME / "mini_0000.rad",
                    60000000, 60000000)
    cfg = _mini_cfg(tmp_path, [overlap])
    cfg.model.growth_original_elem_max = rep.original_elem_max
    mask = growth_candidate_mask(ext, Mesh.from_deck(ext), cfg.model,
                                 log=_silent)
    assert int(mask.sum()) == 2
    assert not mask[ext.elem_ids.tolist().index(60000001)]   # original kept


def test_prepare_dry_run_writes_nothing(tmp_path):
    _write_mini(tmp_path)
    cfg = _mini_cfg(tmp_path, [WING, CARVE])
    rep = prepare_growth_mesh(cfg, backend=_fake_backend, write=False,
                              log=_silent)
    assert rep.n_new_elems == 2 and not rep.written
    assert not (tmp_path / GROWTH_MESH_DIRNAME).exists()


def test_prepare_multi_case_consistent_splice(tmp_path, mini_engine_path):
    _write_mini(tmp_path, stem="ma")
    # case B carries an extra unreferenced node with a HIGHER id: allocation
    # must clear the max over ALL case decks, not just the primary's
    _write_mini(tmp_path, stem="mb",
                extra_node="  60000009                 9.0                 "
                           "9.0                 9.0")
    cfg = _mini_cfg(tmp_path, [WING, CARVE], stem="ma")
    cfg.load_cases.append(LoadCase(name="d", stem="mb", sigma_allow=1.0))
    rep = prepare_growth_mesh(cfg, backend=_fake_backend, log=_silent)
    assert rep.node_id_range[0] == 60000010          # above case B's 60000009
    da = Deck.load(tmp_path / GROWTH_MESH_DIRNAME / "ma_0000.rad",
                   60000000, 60000000)
    db = Deck.load(tmp_path / GROWTH_MESH_DIRNAME / "mb_0000.rad",
                   60000000, 60000000)
    assert np.array_equal(da.elem_ids, db.elem_ids)  # identical element splice
    assert np.array_equal(da.elem_conn, db.elem_conn)
    assert np.isin(rep.node_id_range, da.node_ids).all()
    assert np.isin(rep.node_id_range, db.node_ids).all()


def test_prepare_no_regions_raises(tmp_path):
    _write_mini(tmp_path)
    with pytest.raises(ValueError, match="no growth regions"):
        prepare_growth_mesh(_mini_cfg(tmp_path, []), backend=_fake_backend,
                            log=_silent)


def test_prepare_nothing_to_add_raises(tmp_path):
    _write_mini(tmp_path)

    def only_junk(points, tris, max_volume, min_ratio, sizing=None):
        pts = np.vstack([points[:5], [[5.0, 5.0, 5.0], [5.0, 6.0, 5.0]]])
        return pts, np.array([[1, 4, 5, 6]])         # outside every region
    with pytest.raises(ValueError, match="adds no candidate elements"):
        prepare_growth_mesh(_mini_cfg(tmp_path, [WING]), backend=only_junk,
                            log=_silent)
    assert not (tmp_path / GROWTH_MESH_DIRNAME).exists()


def test_prepare_guard_failure_writes_nothing(tmp_path):
    _write_mini(tmp_path)

    def island(points, tris, max_volume, min_ratio, sizing=None):
        pts = np.vstack([points[:5],
                         [[-0.9, 0.0, 0.0], [-0.8, 0.0, 0.0],
                          [-0.9, 0.1, 0.0], [-0.9, 0.0, 0.1]]])
        return pts, np.array([[5, 6, 7, 8]])          # in WING, no shared nodes
    with pytest.raises(ValueError, match="run-start guards"):
        prepare_growth_mesh(_mini_cfg(tmp_path, [WING]), backend=island,
                            log=_silent)
    assert not (tmp_path / GROWTH_MESH_DIRNAME).exists()


def test_prepare_mismatched_case_mesh_raises(tmp_path):
    _write_mini(tmp_path, stem="ma")
    p = tmp_path / "mb_0000.rad"
    p.write_text(MINI_DECK.replace(
        "  60000002  60000002  60000003  60000004  60000005\n", ""),
        encoding="utf-8")
    cfg = _mini_cfg(tmp_path, [WING], stem="ma")
    cfg.load_cases.append(LoadCase(name="d", stem="mb", sigma_allow=1.0))
    with pytest.raises(ValueError, match="different"):
        prepare_growth_mesh(cfg, backend=_fake_backend, log=_silent)


def test_tetgen_backend_missing_package_message(monkeypatch):
    monkeypatch.setitem(sys.modules, "tetgen", None)  # import -> ImportError
    with pytest.raises(RuntimeError, match=r"oropt\[growthmesh\]"):
        tetgen_backend(np.zeros((3, 3)), np.array([[0, 1, 2]]), 1.0, 1.5)


# ---- config pointing + CLI -------------------------------------------------------
def test_point_config_at_touches_only_case_dir(tmp_path):
    cfg = Config()
    cfg.model.case_dir = str(tmp_path)
    cfg.optimizer = "tobs"
    cfg.load_cases = [LoadCase(name="c", stem="mini", sigma_allow=7.0)]
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    point_config_at(p, tmp_path / GROWTH_MESH_DIRNAME)
    back = Config.from_yaml(p)
    assert back.model.case_dir == str(tmp_path / GROWTH_MESH_DIRNAME)
    assert back.optimizer == "tobs"                  # everything else intact
    assert back.load_cases[0].sigma_allow == 7.0
    assert back.model.growth_original_elem_max is None   # not given -> untouched


def test_point_config_at_records_original_elem_max(tmp_path):
    cfg = Config()
    cfg.model.case_dir = str(tmp_path)
    cfg.load_cases = [LoadCase(name="c", stem="mini", sigma_allow=7.0)]
    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    point_config_at(p, tmp_path / GROWTH_MESH_DIRNAME,
                    original_elem_max=60000002)
    back = Config.from_yaml(p)
    assert back.model.case_dir == str(tmp_path / GROWTH_MESH_DIRNAME)
    assert back.model.growth_original_elem_max == 60000002


def test_cli_error_exit_without_regions(tmp_path, capsys):
    _write_mini(tmp_path)
    p = tmp_path / "cfg.yaml"
    _mini_cfg(tmp_path, []).to_yaml(p)
    assert main(["--config", str(p)]) == 2
    assert "no growth regions" in capsys.readouterr().out


def test_report_from_dict_round_trips_asdict_json():
    rep = GrowthMeshReport(
        out_dir="gm", starters=["a_0000.rad"], engines=[], n_new_nodes=2,
        n_new_elems=2, n_generated=4, per_region=[("wing", 2)],
        target_edge=1.0, max_volume=0.1, quality_min=0.5, quality_median=0.8,
        node_id_range=(60000006, 60000007),
        elem_id_range=(60000003, 60000004), total_candidates=2, written=True,
        original_elem_max=60000002)
    d = json.loads(json.dumps(dataclasses.asdict(rep)))   # tuples -> lists
    d["some_future_key"] = 1                              # unknown keys dropped
    assert report_from_dict(d) == rep


def test_cli_json_error_exit_writes_nothing(tmp_path):
    _write_mini(tmp_path)
    p = tmp_path / "cfg.yaml"
    _mini_cfg(tmp_path, []).to_yaml(p)
    jpath = tmp_path / "report.json"
    assert main(["--config", str(p), "--json", str(jpath)]) == 2
    assert not jpath.exists()


def test_cli_json_report_hermetic(tmp_path, monkeypatch, mini_engine_path):
    """--json hands the report across the process boundary — the GUI's
    isolated PREPARE launch (oropt.gui.growthprep) parses it back into the
    same GrowthMeshReport the in-process call used to return."""
    import oropt.growthmesh as gm
    monkeypatch.setattr(gm, "tetgen_backend", _fake_backend)
    _write_mini(tmp_path)
    p = tmp_path / "cfg.yaml"
    _mini_cfg(tmp_path, [WING]).to_yaml(p)
    jpath = tmp_path / "report.json"
    assert main(["--config", str(p), "--json", str(jpath)]) == 0
    rep = report_from_dict(json.loads(jpath.read_text(encoding="utf-8")))
    assert rep.written and rep.n_new_elems == 2
    assert rep.out_dir == str(tmp_path / GROWTH_MESH_DIRNAME)
    assert rep.per_region == [("wing", 2)]
    assert rep.original_elem_max == 60000002


# =============================================================================
# TetGen-backed end-to-end (skipped when the optional package is missing)
# =============================================================================
def test_tetgen_prepare_end_to_end(tmp_path, mini_engine_path):
    pytest.importorskip("tetgen")
    _write_mini(tmp_path)
    cfg = _mini_cfg(tmp_path, [WING])
    rep = prepare_growth_mesh(cfg, log=_silent)
    assert rep.n_new_elems > 0
    assert rep.quality_min > 0.0
    ext = Deck.load(tmp_path / GROWTH_MESH_DIRNAME / "mini_0000.rad",
                    60000000, 60000000)
    mask = growth_candidate_mask(ext, Mesh.from_deck(ext), cfg.model,
                                 log=_silent)
    assert int(mask.sum()) == rep.n_new_elems == rep.total_candidates
    # exact conformity: some new element reuses the part's x=0 face nodes
    new_rows = ext.elem_ids > 60000002
    assert np.isin(ext.elem_conn[new_rows],
                   [60000001, 60000003, 60000004]).any()
    # new node ids start right above max(existing) and are pinnable design ids
    assert rep.node_id_range[0] == 60000006
    assert rep.node_id_range[0] >= 60000000          # >= design_node_min


def test_tetgen_cli_end_to_end(tmp_path, mini_engine_path):
    pytest.importorskip("tetgen")
    _write_mini(tmp_path)
    p = tmp_path / "cfg.yaml"
    _mini_cfg(tmp_path, [WING]).to_yaml(p)
    assert main(["--config", str(p), "--dry-run"]) == 0
    assert not (tmp_path / GROWTH_MESH_DIRNAME).exists()
    jpath = tmp_path / "report.json"
    assert main(["--config", str(p), "--json", str(jpath)]) == 0
    assert (tmp_path / GROWTH_MESH_DIRNAME / "mini_0000.rad").is_file()
    assert (tmp_path / GROWTH_MESH_DIRNAME / "mini_0001.rad").is_file()
    rep = report_from_dict(json.loads(jpath.read_text(encoding="utf-8")))
    assert rep.n_new_elems > 0
    assert rep.starters == [str(tmp_path / GROWTH_MESH_DIRNAME / "mini_0000.rad")]


def test_tetgen_size_factor_scales_element_count(tmp_path):
    pytest.importorskip("tetgen")
    _write_mini(tmp_path)
    cfg = _mini_cfg(tmp_path, [WING])
    fine = prepare_growth_mesh(cfg, size_factor=0.7, write=False, log=_silent)
    coarse = prepare_growth_mesh(cfg, size_factor=2.0, write=False, log=_silent)
    assert fine.n_new_elems > coarse.n_new_elems


def _write_bar(tmp_path, nx=24, stem="bar") -> Path:
    """A nx x 1 x 1 bar of unit cubes (Kuhn tets) as a starter deck: a big
    domain relative to a small end region, the memory-scaling scenario."""
    pts, tets = box_grid_tets(np.arange(nx + 1.0), np.arange(2.0),
                              np.arange(2.0))
    node_ids = 60000001 + np.arange(len(pts))
    elem_ids = 60000001 + np.arange(len(tets))
    lines = (["#- bar", "/MAT/LAW1/3", "mat", "/NODE"]
             + format_node_lines(node_ids, pts)
             + ["/PART/60000000", "bar", "         3         3         0",
                "/TETRA4/60000000"]
             + format_elem_lines(elem_ids, node_ids[tets])
             + ["/PROP/SOLID/3", "prop", "/END"])
    p = tmp_path / f"{stem}_0000.rad"
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def test_tetgen_scaffold_scales_with_region_not_domain(tmp_path):
    """THE memory-scaling regression this module's -m sizing exists for: a
    24x1x1 bar with a small region at one end. Under the old global -a bound
    the whole margin-inflated domain (~3800 model-volume units) was meshed at
    part sizing — measured ~26k tets on this exact case (and >=77M / >50 GB on
    a real 2M-element model). With the background sizing field the scaffold
    coarsens away from the region: measured ~6k tets. The bound asserts the
    scaffold stays on the region-volume side of that gap with ~2x headroom.
    The domain diagonal (~26 units) also exceeds the wrapper's ~4-unit -m
    segfault threshold (pyvista/tetgen#65), so this test doubles as coverage
    of the unit-scale normalisation in tetgen_backend."""
    pytest.importorskip("tetgen")
    nx = 24
    _write_bar(tmp_path, nx=nx)
    region = GrowthBox(name="end", x_min=nx - 0.001, x_max=nx + 1.5,
                       y_min=-0.25, y_max=1.25, z_min=-0.25, z_max=1.25)
    cfg = _mini_cfg(tmp_path, [region], stem="bar")
    rep = prepare_growth_mesh(cfg, log=_silent)

    assert rep.n_generated < 15_000          # old global -a produced ~26k
    # the kept candidates stay near the region-volume budget ...
    region_budget = 1.5 ** 3 / rep.max_volume
    assert 0 < rep.n_new_elems < 30 * region_budget
    # ... and are actually sized near the target edge, not the far-field cap
    ext = Deck.load(tmp_path / GROWTH_MESH_DIRNAME / "bar_0000.rad",
                    60000000, 60000000)
    new_rows = ext.elem_ids > int(60000000 + len(box_grid_tets(
        np.arange(nx + 1.0), np.arange(2.0), np.arange(2.0))[1]))
    xyz = ext.node_xyz[Mesh.from_deck(ext).conn_rows]
    mean_vol = float(np.abs(signed_volumes(xyz[new_rows])).mean())
    assert mean_vol < 3.0 * rep.max_volume
    # conformity survives the backend's unit-scale round trip: new elements
    # reuse the bar's end-face node ids
    end_face_ids = 60000001 + np.flatnonzero(
        box_grid_tets(np.arange(nx + 1.0), np.arange(2.0),
                      np.arange(2.0))[0][:, 0] == nx)
    assert np.isin(end_face_ids, ext.elem_conn[new_rows]).any()


# ---- GUI wiring -------------------------------------------------------------
def _growth_gui_cfg(tmp_path) -> Path:
    """A mini case + config with one region and its own work folder — the
    shared fixture wiring of every growth-mesh AppTest below."""
    _write_mini(tmp_path)
    cfg = _mini_cfg(tmp_path, [WING])
    cfg.work_dir = str(tmp_path / "work")
    cfg_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(cfg_path)
    return cfg_path


def _growth_apptest(cfg_path: Path):
    """The app loaded on *cfg_path* (same AppTest wiring as the preview-button
    test; cold-import AppTests need a generous timeout)."""
    import oropt
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
    app_file = Path(oropt.__file__).resolve().parent / "gui" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(cfg_path)).run()
    return at


def test_app_growth_mesh_button_renders(tmp_path):
    """The 'Generate growth mesh' expander/button renders when regions are
    configured (same AppTest wiring as the preview-button test)."""
    at = _growth_apptest(_growth_gui_cfg(tmp_path))
    assert not at.exception
    keys = {b.key for b in at.button}
    assert "growth_mesh_generate" in keys


def test_app_growth_mesh_generate_launches_subprocess(tmp_path, monkeypatch):
    """Clicking ⚙️ Generate launches the PREPARE subprocess (growthprep.start,
    faked here) instead of running TetGen in-process, and the panel flips to
    the running view (cancel button) in the same script run."""
    from oropt.gui import growthprep
    calls = {}

    def fake_start(cfg, prep, size_factor, min_ratio, cwd):
        calls["params"] = (float(size_factor), float(min_ratio))
        prep.mkdir(parents=True, exist_ok=True)
        (prep / growthprep.PIDFILE).write_text(str(os.getpid()),
                                               encoding="utf-8")
        return os.getpid()

    monkeypatch.setattr(growthprep, "start", fake_start)
    at = _growth_apptest(_growth_gui_cfg(tmp_path))
    next(b for b in at.button if b.key == "growth_mesh_generate").click().run()
    assert not at.exception
    assert calls["params"] == (1.0, 1.5)
    assert any(b.key == "growth_mesh_cancel" for b in at.button)


def test_app_growth_mesh_running_panel_reattaches(tmp_path):
    """A live PREPARE pid file renders the running view (cancel button) even
    in a fresh session — the panel re-attaches from files alone, so a page
    reload or GUI restart never orphans a generation in flight."""
    from oropt.gui import growthprep
    cfg_path = _growth_gui_cfg(tmp_path)
    prep = tmp_path / "work" / growthprep.PREPARE_DIRNAME
    prep.mkdir(parents=True)
    (prep / growthprep.PIDFILE).write_text(str(os.getpid()), encoding="utf-8")
    (prep / growthprep.LOG_NAME).write_text(
        "[oropt] growth-mesh: tetrahedralising the PLC (9 points) ...\n",
        encoding="utf-8")
    at = _growth_apptest(cfg_path)
    assert not at.exception
    assert any(b.key == "growth_mesh_cancel" for b in at.button)


def test_app_growth_mesh_done_panel_and_point_button(tmp_path):
    """A finished PREPARE's report.json renders the success panel from disk,
    and 📁 use-decks still rewrites the config exactly like the old in-process
    flow (case_dir + the original/expansion element-id boundary)."""
    from oropt.gui import growthprep
    cfg_path = _growth_gui_cfg(tmp_path)
    prep = tmp_path / "work" / growthprep.PREPARE_DIRNAME
    prep.mkdir(parents=True)
    out_dir = tmp_path / GROWTH_MESH_DIRNAME
    rep = GrowthMeshReport(
        out_dir=str(out_dir), starters=[], engines=[], n_new_nodes=2,
        n_new_elems=2, n_generated=4, per_region=[("wing", 2)],
        target_edge=1.0, max_volume=0.1, quality_min=0.5, quality_median=0.8,
        node_id_range=(60000006, 60000007),
        elem_id_range=(60000003, 60000004), total_candidates=2, written=True,
        original_elem_max=60000002)
    (prep / growthprep.REPORT_NAME).write_text(
        json.dumps(dataclasses.asdict(rep)), encoding="utf-8")
    at = _growth_apptest(cfg_path)
    assert not at.exception
    next(b for b in at.button if b.key == "growth_mesh_use").click().run()
    assert not at.exception
    back = Config.from_yaml(cfg_path)
    assert back.model.case_dir == str(out_dir)
    assert back.model.growth_original_elem_max == 60000002
