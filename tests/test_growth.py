"""Growth boxes (add-material regions): config schema, candidate selection,
run-start guards, and material growing into the boxes with every optimiser.

A growth box marks pre-meshed candidate elements that start the run VOID; the
optimisers' existing bi-directional updates may then add them. These tests cover
the seams: GrowthBox YAML/dict round-trips, centroid-in-box selection (union,
inclusive bounds), the three run-start guards in
:func:`oropt.loop.growth_candidate_mask` (empty box, node ids below
design_node_min, unreachable candidates), growth through each optimiser's
update, the validation checks, and the GUI row helpers.
"""
from __future__ import annotations

import numpy as np
import pytest

from oropt.beso import Beso
from oropt.config import (Beso as BesoCfg, Config, GrowthBox,
                          LevelSet as LevelSetCfg, LoadCase, Model,
                          TobsOpts as TobsCfg, unknown_keys)
from oropt.deck import Deck
from oropt.gui.boxes import growth_boxes_from_records, records_from_growth_boxes
from oropt.levelset import LevelSet
from oropt.loop import growth_candidate_mask, resolve_growth_boxes
from oropt.mesh import Mesh, local_frame_basis, overlay_primitives
from oropt.tobs import Tobs
from oropt.validate import check_config

# Four tets: e1-e2 share a face (the "part"), e3 touches ONLY e2 via node 5 (a
# chain candidate), e4 is a disjoint island far away. Centroids: e1 (.25,.25,.25),
# e2 (.5,.5,.5), e3 (1.5,1.75,1.5), e4 (5.25,5.25,5.25).
GROWTH_DECK = """\
/NODE
  60000001   0.0   0.0   0.0
  60000002   1.0   0.0   0.0
  60000003   0.0   1.0   0.0
  60000004   0.0   0.0   1.0
  60000005   1.0   1.0   1.0
  60000006   2.0   2.0   2.0
  60000007   5.0   5.0   5.0
  60000008   6.0   5.0   5.0
  60000009   5.0   6.0   5.0
  60000010   5.0   5.0   6.0
  60000011   2.0   2.0   1.0
  60000012   1.0   2.0   2.0
/PART/60000000
linkage
         3         3         0
/TETRA4/60000000
  60000001  60000001  60000002  60000003  60000004
  60000002  60000002  60000003  60000004  60000005
  60000003  60000005  60000006  60000011  60000012
  60000004  60000007  60000008  60000009  60000010
#-  PROPERTIES:
/PROP/SOLID/3
prop
/GRNOD/NODE/60000000
sym
  60000001
/END
"""

BOX_E2 = GrowthBox(name="b2", x_min=0.4, x_max=0.6, y_min=0.4, y_max=0.6,
                   z_min=0.4, z_max=0.6)
BOX_E3 = GrowthBox(name="b3", x_min=1.4, x_max=1.6, y_min=1.7, y_max=1.8,
                   z_min=1.4, z_max=1.6)
BOX_ISLAND = GrowthBox(name="island", x_min=5.0, x_max=6.0, y_min=5.0,
                       y_max=6.0, z_min=5.0, z_max=6.0)
BOX_EMPTY = GrowthBox(name="offside", x_min=10.0, x_max=11.0, y_min=10.0,
                      y_max=11.0, z_min=10.0, z_max=11.0)


def _load(tmp_path, design_node_min=60000000):
    p = tmp_path / "g_0000.rad"
    p.write_text(GROWTH_DECK, encoding="utf-8")
    deck = Deck.load(p, design_part_id=60000000, design_node_min=design_node_min)
    return deck, Mesh.from_deck(deck)


def _silent(_msg):
    pass


# ---- config schema ----------------------------------------------------------
def test_growth_box_dict_coercion_and_yaml_roundtrip(tmp_path):
    cfg = Config.from_dict({"model": {"growth_boxes": [
        {"name": "rib", "x_min": 0.0, "x_max": 1.0, "y_min": 0.0, "y_max": 2.0,
         "z_min": -1.0, "z_max": 1.0}]}})
    assert isinstance(cfg.model.growth_boxes[0], GrowthBox)
    assert cfg.model.growth_boxes[0].y_max == 2.0
    p = tmp_path / "c.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)
    assert back.model.growth_boxes == cfg.model.growth_boxes


def test_growth_boxes_default_empty():
    assert Config().model.growth_boxes == []
    assert Model().growth_boxes == []


def test_unknown_keys_flags_growth_box_typos():
    data = {"model": {"growth_boxes": [
        {"name": "b", "x_mim": 0.0, "x_max": 1.0}]}}
    bad = unknown_keys(data)
    assert "model.growth_boxes[0].x_mim" in bad
    assert "model.growth_boxes[0].x_max" not in bad
    assert "model.growth_boxes" not in bad          # the field itself is known


# ---- centroid-in-box selection ----------------------------------------------
def test_in_boxes_mask_membership_union_and_inclusive_bounds():
    m = Mesh(centroids=np.array([[0., 0, 0], [1, 1, 1], [2, 2, 2]]),
             volumes=np.ones(3), conn_rows=np.array([[0, 1, 2, 3]] * 3),
             n_nodes=4, design_node_min=0)
    near = GrowthBox(x_min=-.5, x_max=.5, y_min=-.5, y_max=.5,
                     z_min=-.5, z_max=.5)
    exact = GrowthBox(x_min=2, x_max=2, y_min=2, y_max=2, z_min=2, z_max=2)
    assert m.in_boxes_mask([near]).tolist() == [True, False, False]
    assert m.in_boxes_mask([exact]).tolist() == [False, False, True]   # inclusive
    assert m.in_boxes_mask([near, exact]).tolist() == [True, False, True]
    assert m.in_boxes_mask([]).tolist() == [False] * 3
    assert m.in_boxes_mask(None).tolist() == [False] * 3


# ---- run-start guards --------------------------------------------------------
def test_candidate_mask_all_false_without_boxes(tmp_path):
    deck, mesh = _load(tmp_path)
    mask = growth_candidate_mask(deck, mesh, Model(), log=_silent)
    assert not mask.any()


def test_candidate_mask_selects_box_elements_and_logs(tmp_path):
    deck, mesh = _load(tmp_path)
    lines: list[str] = []
    mask = growth_candidate_mask(deck, mesh, Model(growth_boxes=[BOX_E2]),
                                 log=lines.append)
    assert mask.tolist() == [False, True, False, False]
    assert any("'b2'" in ln and "1 candidate" in ln for ln in lines)


def test_candidate_reachable_through_another_candidate(tmp_path):
    """e3 touches the structure only via e2 (node 5); with both candidates the
    shared-node path alive->e2->e3 makes e3 growable -- no error."""
    deck, mesh = _load(tmp_path)
    mask = growth_candidate_mask(
        deck, mesh, Model(growth_boxes=[BOX_E2, BOX_E3]), log=_silent)
    assert mask.tolist() == [False, True, True, False]


def test_empty_box_raises(tmp_path):
    deck, mesh = _load(tmp_path)
    with pytest.raises(ValueError, match="'offside'.*no design elements"):
        growth_candidate_mask(deck, mesh, Model(growth_boxes=[BOX_EMPTY]),
                              log=_silent)


def test_unreachable_candidate_raises(tmp_path):
    deck, mesh = _load(tmp_path)
    with pytest.raises(ValueError, match="'island'.*share no nodes"):
        growth_candidate_mask(deck, mesh, Model(growth_boxes=[BOX_ISLAND]),
                              log=_silent)


def test_candidate_nodes_below_design_node_min_raise(tmp_path):
    deck, mesh = _load(tmp_path, design_node_min=60000099)
    with pytest.raises(ValueError, match="design_node_min"):
        growth_candidate_mask(deck, mesh, Model(growth_boxes=[BOX_E2]),
                              log=_silent)


# ---- growth through each optimiser's update ----------------------------------
def _chain_mesh(n=5):
    conn = np.array([[i, i + 1, i + 2, i + 3] for i in range(n)])
    return Mesh(centroids=np.zeros((n, 3)), volumes=np.ones(n),
                conn_rows=conn, n_nodes=int(conn.max()) + 1, design_node_min=0)


_PROTECTED = np.array([True, False, False, False, False])   # element 0 = seed
_ALIVE0 = np.array([True, True, True, True, False])         # element 4 = candidate
_SENS = np.array([1.0, 1.0, 1.0, 1.0, 5.0])                 # candidate ranks best


def test_beso_grows_void_candidate():
    cfg = BesoCfg(filter_radius=0.0, target_volume_fraction=1.0,
                  evolution_rate=0.2, max_add_ratio=1.0)
    new = Beso(_chain_mesh(), cfg, _PROTECTED).update(_ALIVE0, _SENS,
                                                      target_vf=1.0)
    assert new[4]                       # void candidate added (grown)
    assert new.all()


def test_beso_growth_paced_by_max_add_ratio():
    """max_add_ratio caps how fast material grows into the boxes -- with a zero
    cap the candidate stays void even when the volume target asks for it (why
    validation warns when it is below evolution_rate)."""
    cfg = BesoCfg(filter_radius=0.0, target_volume_fraction=1.0,
                  evolution_rate=0.2, max_add_ratio=0.0)
    new = Beso(_chain_mesh(), cfg, _PROTECTED).update(_ALIVE0, _SENS,
                                                      target_vf=1.0)
    assert not new[4]


def test_tobs_grows_void_candidate():
    cfg = TobsCfg(filter_radius=0.0, flip_limit=1.0, constraint_relaxation=0.1)
    new = Tobs(_chain_mesh(), cfg, _PROTECTED).update(_ALIVE0, _SENS,
                                                      target_vf=1.0)
    assert new[4]


def test_levelset_grows_void_candidate():
    cfg = LevelSetCfg(filter_radius=0.0, smoothing_passes=0, dt=1.0,
                      band_width=3.0)
    new = LevelSet(_chain_mesh(), cfg, _PROTECTED).update(_ALIVE0, _SENS,
                                                          target_vf=1.0)
    assert new[4]


# ---- validation ---------------------------------------------------------------
def _cfg(boxes, optimizer="beso"):
    cfg = Config()
    cfg.optimizer = optimizer
    cfg.model.growth_boxes = boxes
    cfg.load_cases = [LoadCase(name="c", stem="s", sigma_allow=1.0, d_allow=1.0)]
    return cfg


def _growth_problems(cfg):
    return [str(p) for p in check_config(cfg)
            if "growth box" in str(p) or "max_add_ratio" in str(p)]


def test_validate_flags_inverted_and_degenerate_bounds():
    bad = GrowthBox(name="bad", x_min=1.0, x_max=0.0, y_min=0.0, y_max=0.0,
                    z_min=0.0, z_max=1.0)
    probs = _growth_problems(_cfg([bad]))
    assert any(p.startswith("error") and "x_min" in p for p in probs)
    assert any(p.startswith("warning") and "degenerate" in p for p in probs)


def test_validate_warns_add_ratio_below_evolution_rate_for_beso_only():
    box = GrowthBox(name="ok", x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
                    z_min=0.0, z_max=1.0)
    cfg = _cfg([box])                     # beso defaults: 0.01 < 0.02
    assert any("max_add_ratio" in p for p in _growth_problems(cfg))
    cfg.beso.max_add_ratio = cfg.beso.evolution_rate
    assert not any("max_add_ratio" in p for p in _growth_problems(cfg))
    assert not any("max_add_ratio" in p
                   for p in _growth_problems(_cfg([box], optimizer="tobs")))


def test_validate_silent_without_boxes():
    assert _growth_problems(_cfg([])) == []


# ---- GUI row helpers -----------------------------------------------------------
def test_gui_records_roundtrip():
    boxes = [GrowthBox(name="a", x_min=0.0, x_max=1.0, y_min=2.0, y_max=3.0,
                       z_min=4.0, z_max=5.0)]
    assert growth_boxes_from_records(records_from_growth_boxes(boxes)) == boxes


def test_gui_blank_and_partial_rows_dropped():
    rows = [
        {"name": None, "x_min": None, "x_max": None, "y_min": None,
         "y_max": None, "z_min": None, "z_max": None},         # editor's blank row
        {"name": "partial", "x_min": 0.0, "x_max": float("nan"), "y_min": 0.0,
         "y_max": 1.0, "z_min": 0.0, "z_max": 1.0},            # missing a bound
    ]
    assert growth_boxes_from_records(rows) == []


# =====================================================================
# Phase 1.5: shapes, oriented boxes, 3D overlay, deck /BOX/RECTA input
# =====================================================================

# ---- config: shape + local frame + deck reference ---------------------------
def test_growth_box_shape_and_frame_yaml_roundtrip(tmp_path):
    cfg = Config.from_dict({"model": {"growth_boxes": [
        {"name": "orb", "shape": "box", "x_min": 0.0, "x_max": 1.0, "y_min": 0.0,
         "y_max": 1.0, "z_min": 0.0, "z_max": 1.0, "origin": [1.0, 2.0, 3.0],
         "x_axis": [1.0, 1.0, 0.0], "xy_axis": [-1.0, 1.0, 0.0]},
        {"name": "ball", "shape": "sphere", "cx": 1.0, "cy": 2.0, "cz": 3.0,
         "radius": 0.5},
        {"name": "rod", "shape": "cylinder", "x1": 0.0, "y1": 0.0, "z1": 0.0,
         "x2": 2.0, "y2": 0.0, "z2": 0.0, "radius": 0.3}]}})
    b0 = cfg.model.growth_boxes[0]
    assert b0.shape_kind() == "box" and b0.has_local_frame()
    assert b0.x_axis == [1.0, 1.0, 0.0]
    assert cfg.model.growth_boxes[2].shape_kind() == "cylinder"
    p = tmp_path / "c.yaml"
    cfg.to_yaml(p)
    assert Config.from_yaml(p).model.growth_boxes == cfg.model.growth_boxes


def test_unknown_keys_flags_new_growth_fields():
    data = {"model": {"growth_boxes": [
        {"name": "b", "shape": "sphere", "radiuss": 1.0}]}}
    bad = unknown_keys(data)
    assert "model.growth_boxes[0].radiuss" in bad
    assert "model.growth_boxes[0].shape" not in bad          # a known field
    assert "model.growth_boxes[0].radius" not in bad


# ---- sphere / cylinder centroid membership ----------------------------------
def _mesh_at(points):
    pts = np.asarray(points, dtype=float)
    return Mesh(centroids=pts, volumes=np.ones(len(pts)),
                conn_rows=np.array([[0, 1, 2, 3]] * len(pts)),
                n_nodes=4, design_node_min=0)


def test_in_boxes_mask_sphere_inclusive_surface():
    m = _mesh_at([[0, 0, 0], [1, 0, 0], [3, 0, 0]])
    s = GrowthBox(shape="sphere", cx=0.0, cy=0.0, cz=0.0, radius=1.0)
    # centroid exactly on the surface (dist == radius) is inside (inclusive)
    assert m.in_boxes_mask([s]).tolist() == [True, True, False]


def test_in_boxes_mask_cylinder_finite_caps_and_radius():
    m = _mesh_at([[1, 0, 0],       # on axis, mid-length -> inside
                  [1, 0.9, 0],     # radial 0.9 < 1 -> inside
                  [1, 1.5, 0],     # radial 1.5 > 1 -> outside
                  [3, 0, 0],       # beyond the far cap (t > 1) -> outside
                  [-0.5, 0, 0],    # before the near cap (t < 0) -> outside
                  [0, 0, 0]])      # on the near cap, on axis -> inside
    c = GrowthBox(shape="cylinder", x1=0.0, y1=0.0, z1=0.0,
                  x2=2.0, y2=0.0, z2=0.0, radius=1.0)
    assert m.in_boxes_mask([c]).tolist() == [True, True, False, False, False, True]


def test_in_boxes_mask_shape_union():
    m = _mesh_at([[0, 0, 0], [5, 0, 0], [9, 9, 9]])
    s = GrowthBox(shape="sphere", cx=0.0, cy=0.0, cz=0.0, radius=1.0)
    b = GrowthBox(shape="box", x_min=4.5, x_max=5.5, y_min=-0.5, y_max=0.5,
                  z_min=-0.5, z_max=0.5)
    assert m.in_boxes_mask([s, b]).tolist() == [True, True, False]


# ---- oriented (local-frame) boxes -------------------------------------------
def test_local_frame_basis_orthonormal_gram_schmidt():
    b = GrowthBox(shape="box", x_axis=[2.0, 0.0, 0.0], xy_axis=[1.0, 1.0, 0.0])
    origin, R = local_frame_basis(b)
    assert np.allclose(origin, [0, 0, 0])
    # rows orthonormal; the non-orthogonal xy_axis is Gram-Schmidt-projected
    assert np.allclose(R @ R.T, np.eye(3))
    assert np.allclose(R[0], [1, 0, 0])          # e1 = normalised x_axis
    assert np.allclose(R[1], [0, 1, 0])          # e2 after removing the e1 part


def test_local_frame_basis_none_when_absent_or_degenerate():
    assert local_frame_basis(GrowthBox(shape="box")) is None       # no frame
    # parallel axes cannot define a plane -> degenerate -> None (world-aligned)
    assert local_frame_basis(GrowthBox(
        shape="box", x_axis=[1.0, 0.0, 0.0], xy_axis=[2.0, 0.0, 0.0])) is None


def test_in_boxes_mask_oriented_box_rotated_45deg():
    # a thin box whose local +x runs along world (1,1,0): a point 1 unit along that
    # diagonal is inside, the same distance along world +x falls outside the width
    b = GrowthBox(shape="box", x_min=0.0, x_max=2.0, y_min=-0.5, y_max=0.5,
                  z_min=-0.5, z_max=0.5, origin=[0.0, 0.0, 0.0],
                  x_axis=[1.0, 1.0, 0.0], xy_axis=[-1.0, 1.0, 0.0])
    m = _mesh_at([[0.7071, 0.7071, 0.0], [1.0, 0.0, 0.0]])
    assert m.in_boxes_mask([b]).tolist() == [True, False]


# ---- 3D-overlay primitives ---------------------------------------------------
def test_overlay_primitives_all_shapes():
    boxes = [
        GrowthBox(name="b", shape="box", x_min=0.0, x_max=1.0, y_min=0.0,
                  y_max=1.0, z_min=0.0, z_max=1.0),
        GrowthBox(name="s", shape="sphere", cx=1.0, cy=2.0, cz=3.0, radius=0.5),
        GrowthBox(name="c", shape="cylinder", x1=0.0, y1=0.0, z1=0.0, x2=1.0,
                  y2=0.0, z2=0.0, radius=0.3)]
    prims = overlay_primitives(boxes)
    assert [p["kind"] for p in prims] == ["box", "sphere", "cylinder"]
    assert len(prims[0]["corners"]) == 8 and len(prims[0]["edges"]) == 12
    assert prims[1]["center"] == [1.0, 2.0, 3.0] and prims[1]["radius"] == 0.5
    assert prims[2]["p1"] == [0.0, 0.0, 0.0] and prims[2]["p2"] == [1.0, 0.0, 0.0]


def test_overlay_primitives_skips_degenerate_and_deck_ref():
    boxes = [
        GrowthBox(shape="sphere", radius=0.0),                    # zero radius
        GrowthBox(shape="cylinder", x1=0.0, y1=0.0, z1=0.0,       # zero-length axis
                  x2=0.0, y2=0.0, z2=0.0, radius=1.0),
        GrowthBox(shape="box"),                                   # zero-size box
        GrowthBox(shape="box", deck_box_id=7)]                    # corners unresolved
    assert overlay_primitives(boxes) == []


def test_overlay_primitives_oriented_box_corners_in_world():
    b = GrowthBox(shape="box", x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
                  z_min=0.0, z_max=1.0, origin=[10.0, 0.0, 0.0],
                  x_axis=[0.0, 1.0, 0.0], xy_axis=[-1.0, 0.0, 0.0])
    corners = np.asarray(overlay_primitives([b])[0]["corners"])
    # the local (x_min,y_min,z_min) corner maps back to the frame origin
    assert np.any(np.all(np.isclose(corners, [10.0, 0.0, 0.0]), axis=1))
    assert corners.shape == (8, 3)


# ---- deck /BOX/RECTA reference resolution -----------------------------------
_DECK_WITH_BOX = GROWTH_DECK.replace("/END", (
    "/BOX/RECTA/7001\n"
    "e2_box\n"
    "        0.6    0.6    0.6\n"          # reversed corners on purpose
    "        0.4    0.4    0.4\n"
    "/END"))


def _load_with_box(tmp_path):
    p = tmp_path / "gb_0000.rad"
    p.write_text(_DECK_WITH_BOX, encoding="utf-8")
    deck = Deck.load(p, design_part_id=60000000, design_node_min=60000000)
    return deck, Mesh.from_deck(deck)


def test_resolve_growth_boxes_fills_coords_from_deck(tmp_path):
    deck, _ = _load_with_box(tmp_path)
    [resolved] = resolve_growth_boxes(deck, [GrowthBox(name="ref", deck_box_id=7001)])
    assert resolved.deck_box_id is None and resolved.shape_kind() == "box"
    assert (resolved.x_min, resolved.x_max) == (0.4, 0.6)         # normalised
    assert (resolved.z_min, resolved.z_max) == (0.4, 0.6)


def test_resolve_growth_boxes_passthrough(tmp_path):
    deck, _ = _load(tmp_path)
    assert resolve_growth_boxes(deck, [BOX_E2]) == [BOX_E2]       # no deck_box_id


def test_resolve_growth_boxes_missing_card_raises(tmp_path):
    deck, _ = _load(tmp_path)
    with pytest.raises(ValueError, match="/BOX/RECTA/424242"):
        resolve_growth_boxes(deck, [GrowthBox(name="x", deck_box_id=424242)])


def test_candidate_mask_via_deck_box_id(tmp_path):
    deck, mesh = _load_with_box(tmp_path)
    mask = growth_candidate_mask(
        deck, mesh, Model(growth_boxes=[GrowthBox(deck_box_id=7001)]), log=_silent)
    assert mask.tolist() == [False, True, False, False]          # selects e2


def test_candidate_mask_sphere_shape(tmp_path):
    deck, mesh = _load(tmp_path)
    sph = GrowthBox(name="sph", shape="sphere", cx=0.5, cy=0.5, cz=0.5, radius=0.1)
    mask = growth_candidate_mask(deck, mesh, Model(growth_boxes=[sph]), log=_silent)
    assert mask.tolist() == [False, True, False, False]


# ---- per-shape / frame validation -------------------------------------------
def test_validate_sphere_and_cylinder_radius():
    bad_s = _cfg([GrowthBox(name="s", shape="sphere", radius=0.0)])
    assert any(p.startswith("error") and "sphere radius" in p
               for p in _growth_problems(bad_s))
    bad_c = _cfg([GrowthBox(name="c", shape="cylinder", x1=0.0, y1=0.0, z1=0.0,
                            x2=0.0, y2=0.0, z2=0.0, radius=1.0)])
    assert any("zero-length axis" in p for p in _growth_problems(bad_c))


def test_validate_unknown_shape_errors():
    bad = _cfg([GrowthBox(name="x", shape="pyramid")])
    assert any(p.startswith("error") and "unknown shape" in p
               for p in _growth_problems(bad))


def test_validate_local_frame_partial_warns_and_parallel_errors():
    partial = GrowthBox(name="p", shape="box", x_min=0.0, x_max=1.0, y_min=0.0,
                        y_max=1.0, z_min=0.0, z_max=1.0, x_axis=[1.0, 0.0, 0.0])
    assert any(p.startswith("warning") and "local frame" in p
               for p in _growth_problems(_cfg([partial])))
    par = GrowthBox(name="q", shape="box", x_min=0.0, x_max=1.0, y_min=0.0,
                    y_max=1.0, z_min=0.0, z_max=1.0, x_axis=[1.0, 0.0, 0.0],
                    xy_axis=[2.0, 0.0, 0.0])
    assert any(p.startswith("error") and "parallel" in p
               for p in _growth_problems(_cfg([par])))


def test_validate_valid_oriented_box_silent():
    ok = GrowthBox(name="ok", shape="box", x_min=0.0, x_max=1.0, y_min=0.0,
                   y_max=1.0, z_min=0.0, z_max=1.0, origin=[0.0, 0.0, 0.0],
                   x_axis=[1.0, 1.0, 0.0], xy_axis=[-1.0, 1.0, 0.0])
    # tobs avoids beso's unrelated max_add_ratio warning
    assert _growth_problems(_cfg([ok], optimizer="tobs")) == []


# ---- GUI row helpers across shapes ------------------------------------------
def test_gui_records_roundtrip_all_shapes():
    boxes = [
        GrowthBox(name="b", shape="box", x_min=0.0, x_max=1.0, y_min=2.0,
                  y_max=3.0, z_min=4.0, z_max=5.0),
        GrowthBox(name="s", shape="sphere", cx=1.0, cy=2.0, cz=3.0, radius=0.5),
        GrowthBox(name="c", shape="cylinder", x1=0.0, y1=0.0, z1=0.0, x2=1.0,
                  y2=1.0, z2=1.0, radius=0.4)]
    assert growth_boxes_from_records(records_from_growth_boxes(boxes)) == boxes


def test_gui_partial_and_unknown_shape_rows_dropped():
    rows = [
        {"name": "s", "shape": "sphere", "cx": 0.0, "cy": 0.0, "cz": None,
         "radius": 1.0},                                          # missing cz
        {"name": "u", "shape": "blob", "cx": 0.0, "cy": 0.0, "cz": 0.0,
         "radius": 1.0},                                          # unknown shape
        {"name": "c", "shape": "cylinder", "x1": 0.0, "y1": 0.0, "z1": 0.0,
         "x2": 1.0, "y2": 0.0, "z2": 0.0, "radius": 0.5}]         # complete
    out = growth_boxes_from_records(rows)
    assert [b.shape_kind() for b in out] == ["cylinder"]
