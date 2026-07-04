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
import sys
from pathlib import Path

import numpy as np
import pytest

from oropt.config import Config, GrowthBox, LoadCase
from oropt.deck import Deck, _parse_elem, _parse_node
from oropt.growthmesh import (GROWTH_MESH_DIRNAME, allocate_ids, box_shell,
                              build_plc, exterior_faces, format_elem_lines,
                              format_node_lines, main, map_to_original_nodes,
                              mean_edge_length, orient_tets, point_config_at,
                              points_in_tets, prepare_growth_mesh,
                              region_bounds, signed_volumes, tet_quality,
                              tetgen_backend)
from oropt.loop import growth_candidate_mask
from oropt.mesh import Mesh

from conftest import MINI_DECK

# Region attached to the part's x=0 face (nodes 60000001/3/4 of MINI_DECK):
# generated tets there share those surface nodes -> reachable.
WING = GrowthBox(name="wing", x_min=-1.0, x_max=-0.001, y_min=-0.5, y_max=1.5,
                 z_min=-0.5, z_max=1.5)
# Region carving the part's second tet (centroid (.5,.5,.5)): carve-and-regrow.
CARVE = GrowthBox(name="carve", shape="sphere", cx=0.5, cy=0.5, cz=0.5,
                  radius=0.11)

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


def _fake_backend(points, tris, max_volume, min_ratio):
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
    """A region overlapping the part is fine either way; with carve off the
    overlapped original element stays alive on the extended deck -- only the
    generated expansion elements are candidates. The report carries the
    original/expansion id boundary carve-off regions need at run time."""
    _write_mini(tmp_path)
    # covers the fake backend's two new tets AND e1's centroid (.25,.25,.25)
    overlap = GrowthBox(name="olap", x_min=-1.0, x_max=0.4, y_min=-0.5,
                        y_max=1.5, z_min=-0.5, z_max=1.5)
    carving = prepare_growth_mesh(_mini_cfg(tmp_path, [overlap]),
                                  backend=_fake_backend, log=_silent)
    assert carving.n_new_elems == 2
    assert carving.total_candidates == 3         # 2 new + carved original e1
    assert carving.original_elem_max == 60000002

    no_carve = dataclasses.replace(overlap, carve=False)
    rep = prepare_growth_mesh(_mini_cfg(tmp_path, [no_carve]),
                              backend=_fake_backend, log=_silent)
    assert rep.n_new_elems == 2
    assert rep.total_candidates == 2             # e1 stays alive
    assert rep.original_elem_max == 60000002

    # the run agrees once the config carries the recorded boundary
    ext = Deck.load(tmp_path / GROWTH_MESH_DIRNAME / "mini_0000.rad",
                    60000000, 60000000)
    cfg = _mini_cfg(tmp_path, [no_carve])
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

    def only_junk(points, tris, max_volume, min_ratio):
        pts = np.vstack([points[:5], [[5.0, 5.0, 5.0], [5.0, 6.0, 5.0]]])
        return pts, np.array([[1, 4, 5, 6]])         # outside every region
    with pytest.raises(ValueError, match="adds no candidate elements"):
        prepare_growth_mesh(_mini_cfg(tmp_path, [WING]), backend=only_junk,
                            log=_silent)
    assert not (tmp_path / GROWTH_MESH_DIRNAME).exists()


def test_prepare_guard_failure_writes_nothing(tmp_path):
    _write_mini(tmp_path)

    def island(points, tris, max_volume, min_ratio):
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
    assert main(["--config", str(p)]) == 0
    assert (tmp_path / GROWTH_MESH_DIRNAME / "mini_0000.rad").is_file()
    assert (tmp_path / GROWTH_MESH_DIRNAME / "mini_0001.rad").is_file()


def test_tetgen_size_factor_scales_element_count(tmp_path):
    pytest.importorskip("tetgen")
    _write_mini(tmp_path)
    cfg = _mini_cfg(tmp_path, [WING])
    fine = prepare_growth_mesh(cfg, size_factor=0.7, write=False, log=_silent)
    coarse = prepare_growth_mesh(cfg, size_factor=2.0, write=False, log=_silent)
    assert fine.n_new_elems > coarse.n_new_elems


# ---- GUI wiring -------------------------------------------------------------
def test_app_growth_mesh_button_renders(tmp_path):
    """The 'Generate growth mesh' expander/button renders when regions are
    configured (same AppTest wiring as the preview-button test)."""
    import oropt
    AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

    _write_mini(tmp_path)
    cfg = _mini_cfg(tmp_path, [WING])
    cfg.work_dir = str(tmp_path / "work")
    cfg_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(cfg_path)

    app_file = Path(oropt.__file__).resolve().parent / "gui" / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(cfg_path)).run()
    assert not at.exception
    keys = {b.key for b in at.button}
    assert "growth_mesh_generate" in keys
