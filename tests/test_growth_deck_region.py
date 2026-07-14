"""Positive deck growth regions (``GrowthBox`` ``shape="deck"``).

The mirror of the keep-out deck (:mod:`tests.test_keepout`): a growth box whose
shape is the occupied volume of parts in a *separate* Radioss deck
(``region_rad``), read only for its geometry (never solved). Covers the seams:
deck resolution + geometry attachment (:func:`oropt.loop.resolve_growth_boxes` ->
:func:`oropt.loop._resolve_deck_region`), the centroid-in-part membership test
(:func:`oropt.mesh._deck_member`) incl. the clearance band, candidate selection
through the full pipeline (:func:`oropt.loop.growth_candidate_mask` / the preview),
the growth-mesh AABB (:func:`oropt.growthmesh.region_aabb`), the overlay skip, the
config round-trip, validation, and the GUI row round-trip.
"""
from __future__ import annotations

import numpy as np
import pytest

from oropt.config import Config, GrowthBox, LoadCase, Model
from oropt.growthmesh import region_aabb
from oropt.loop import (growth_blocked_mask, growth_candidate_mask,
                        preview_growth_boxes, resolve_growth_boxes)
from oropt.mesh import Mesh, overlay_primitives
from oropt.validate import check_config

# Same design geometry as tests/test_growth.py / tests/test_keepout.py:
# e1(60000001) centroid (0.25,0.25,0.25), e2(60000002) centroid (0.5,0.5,0.5),
# e3(60000003) centroid (1.5,1.75,1.5), e4(60000004) island.
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
/PROP/SOLID/3
prop
/GRNOD/NODE/60000000
sym
  60000001
/END
"""

# A region part: one tet enclosing e2's centroid (0.5,0.5,0.5) but not e1's
# (0.25,0.25,0.25), e3's or e4's -> selects exactly design element e2.
REGION_TET_DECK = """\
/NODE
  71000001   0.45   0.45   0.45
  71000002   1.50   0.45   0.45
  71000003   0.45   1.50   0.45
  71000004   0.45   0.45   1.50
/PART/71000000
region
         3         3         0
/TETRA4/71000000
  71000001  71000001  71000002  71000003  71000004
/END
"""

# A tiny region part sitting just ABOVE e2 (all nodes near z~0.9): its volume
# contains NO design centroid, but a node is ~0.4 from e2's centroid -> only a
# clearance band picks e2 up.
REGION_FAR_DECK = """\
/NODE
  72000001   0.50   0.50   0.90
  72000002   0.60   0.50   0.90
  72000003   0.50   0.60   0.90
  72000004   0.50   0.50   1.00
/PART/72000000
region_far
         3         3         0
/TETRA4/72000000
  72000001  72000001  72000002  72000003  72000004
/END
"""


def _load(tmp_path, design_node_min=60000000):
    from oropt.deck import Deck
    p = tmp_path / "g_0000.rad"
    p.write_text(GROWTH_DECK, encoding="utf-8")
    deck = Deck.load(p, design_part_id=60000000, design_node_min=design_node_min)
    return deck, Mesh.from_deck(deck)


def _write_region(tmp_path, text=REGION_TET_DECK, name="region_0000.rad"):
    (tmp_path / name).write_text(text, encoding="utf-8")
    return name


def _model(tmp_path, boxes, **kw):
    return Model(case_dir=str(tmp_path), growth_boxes=list(boxes), **kw)


def _silent(_msg):
    pass


def _deck_box(name="reg", **kw):
    return GrowthBox(name=name, shape="deck", region_rad="region_0000.rad", **kw)


# ---- resolution + membership -----------------------------------------------
def test_resolve_deck_region_attaches_geometry(tmp_path):
    deck, _mesh = _load(tmp_path)
    _write_region(tmp_path)
    [rb] = resolve_growth_boxes(deck, [_deck_box()], str(tmp_path))
    assert rb.shape_kind() == "deck"
    assert rb._region_tets.shape == (1, 4, 3)          # one tet
    assert len(rb._region_nodes) == 4
    assert rb._region_clearance == 0.0


def test_deck_region_selects_only_e2(tmp_path):
    deck, mesh = _load(tmp_path)
    _write_region(tmp_path)
    [rb] = resolve_growth_boxes(deck, [_deck_box()], str(tmp_path))
    # e1(0.25^3) out, e2(0.5^3) in, e3/e4 out
    assert mesh.in_boxes_mask([rb]).tolist() == [False, True, False, False]


def test_growth_candidate_mask_deck_region(tmp_path):
    deck, mesh = _load(tmp_path)
    _write_region(tmp_path)
    m = _model(tmp_path, [_deck_box()])
    cand = growth_candidate_mask(deck, mesh, m, log=_silent)
    assert cand.tolist() == [False, True, False, False]  # e2 starts void


def test_deck_region_missing_deck_raises(tmp_path):
    deck, mesh = _load(tmp_path)                          # no region file written
    m = _model(tmp_path, [_deck_box()])
    with pytest.raises(ValueError, match="region deck not found"):
        growth_candidate_mask(deck, mesh, m, log=_silent)


def test_deck_region_blank_region_rad_raises(tmp_path):
    deck, mesh = _load(tmp_path)
    m = _model(tmp_path, [GrowthBox(name="r", shape="deck")])
    with pytest.raises(ValueError, match="needs region_rad"):
        growth_candidate_mask(deck, mesh, m, log=_silent)


def test_deck_region_clearance_broadens_selection(tmp_path):
    deck, mesh = _load(tmp_path)
    _write_region(tmp_path, REGION_FAR_DECK)
    box0 = GrowthBox(name="far", shape="deck", region_rad="region_0000.rad")
    [r0] = resolve_growth_boxes(deck, [box0], str(tmp_path))
    assert not mesh.in_boxes_mask([r0]).any()             # volume holds no centroid
    box1 = GrowthBox(name="far", shape="deck", region_rad="region_0000.rad",
                     region_clearance_mm=0.5)
    [r1] = resolve_growth_boxes(deck, [box1], str(tmp_path))
    # e2 (nearest node ~0.4 away) picked up by clearance; e1 (~0.74) still out
    assert mesh.in_boxes_mask([r1]).tolist() == [False, True, False, False]


def test_deck_region_part_id_filter(tmp_path):
    """An explicit, present part id selects that part's volume (here e2)."""
    deck, mesh = _load(tmp_path)
    _write_region(tmp_path)
    m = _model(tmp_path, [_deck_box(region_part_ids=[71000000])])
    assert growth_candidate_mask(deck, mesh, m, log=_silent).tolist() == \
        [False, True, False, False]


# ---- preview ----------------------------------------------------------------
def test_preview_reports_deck_region_count(tmp_path):
    deck, mesh = _load(tmp_path)
    _write_region(tmp_path)
    m = _model(tmp_path, [_deck_box()])
    prev = preview_growth_boxes(deck, mesh, m)
    assert prev.rows[0].shape == "deck" and prev.rows[0].count == 1
    assert prev.total_candidates == 1 and prev.guard == ""


def test_preview_deck_region_missing_deck_shows_note(tmp_path):
    deck, mesh = _load(tmp_path)                          # no region file
    m = _model(tmp_path, [_deck_box()])
    prev = preview_growth_boxes(deck, mesh, m)
    assert "not found" in prev.rows[0].note


# ---- growth-mesh AABB -------------------------------------------------------
def test_region_aabb_deck(tmp_path):
    deck, _mesh = _load(tmp_path)
    _write_region(tmp_path)
    [rb] = resolve_growth_boxes(deck, [_deck_box(region_clearance_mm=0.1)],
                                str(tmp_path))
    lo, hi = region_aabb(rb)
    assert np.allclose(lo, [0.45 - 0.1, 0.45 - 0.1, 0.45 - 0.1])
    assert np.allclose(hi, [1.50 + 0.1, 1.50 + 0.1, 1.50 + 0.1])


# ---- overlay ----------------------------------------------------------------
def test_overlay_skips_unresolved_deck_region():
    # an unresolved deck region (no geometry attached) has no drawable outline
    assert overlay_primitives([GrowthBox(name="r", shape="deck",
                                         region_rad="x.rad")]) == []


def test_overlay_outlines_resolved_deck_region(tmp_path):
    import json
    deck, _mesh = _load(tmp_path)
    _write_region(tmp_path)                          # a single tet region
    [rb] = resolve_growth_boxes(deck, [_deck_box()], str(tmp_path))
    [pr] = overlay_primitives([rb])
    assert pr["kind"] == "polyhedron" and pr["name"] == "reg"
    # tetrahedron hull: 4 extreme vertices, 6 edges
    assert len(pr["corners"]) == 4 and len(pr["edges"]) == 6
    json.dumps(pr)                                   # JSON-serialisable (subprocess)


def test_resolve_overlay_boxes_best_effort(tmp_path):
    from oropt.keepout import resolve_overlay_boxes
    _load(tmp_path)
    _write_region(tmp_path)
    good = _deck_box(name="ok")
    missing = GrowthBox(name="gone", shape="deck", region_rad="nope.rad")
    plain = GrowthBox(name="box", x_min=0.0, x_max=1.0, y_min=0.0, y_max=1.0,
                      z_min=0.0, z_max=1.0)
    resolved = resolve_overlay_boxes([good, missing, plain], str(tmp_path))
    assert getattr(resolved[0], "_region_nodes", None) is not None   # attached
    assert getattr(resolved[1], "_region_nodes", None) is None       # missing deck
    assert resolved[2] is plain                                      # non-deck passthrough
    # overlay draws the resolved deck region and the plain box, not the missing one
    assert sorted(p["name"] for p in overlay_primitives(resolved)) == ["box", "ok"]


def test_hull_wireframe_reduces_to_extreme_vertices():
    from oropt.mesh import hull_wireframe
    import numpy as np
    # a cube's 8 corners + an interior point -> the interior point is dropped, so
    # the outline has 8 corners (edges are the triangulated hull's, incl. face
    # diagonals, like the polyhedron overlay). All edge indices are valid corners.
    cube = np.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)]
                    + [[0.5, 0.5, 0.5]], dtype=float)
    corners, edges = hull_wireframe(cube)
    assert len(corners) == 8                                 # interior point dropped
    assert edges and all(0 <= i < 8 and 0 <= j < 8 for i, j in edges)
    assert hull_wireframe(np.zeros((3, 3))) is None          # < 4 points
    assert hull_wireframe(np.zeros((5, 3))) is None          # degenerate hull


# ---- config round-trip ------------------------------------------------------
def test_deck_region_yaml_roundtrip(tmp_path):
    cfg = Config.from_dict({"model": {"growth_boxes": [{
        "name": "reg", "shape": "deck", "region_rad": "region_0000.rad",
        "region_part_ids": [71000000], "region_clearance_mm": 0.25}]}})
    b = cfg.model.growth_boxes[0]
    assert b.shape_kind() == "deck" and b.region_rad == "region_0000.rad"
    p = tmp_path / "c.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p).model.growth_boxes[0]
    assert back.region_rad == "region_0000.rad"
    assert back.region_part_ids == [71000000]
    assert back.region_clearance_mm == 0.25


def test_deck_region_defaults():
    b = GrowthBox()
    assert b.region_rad is None and b.region_part_ids == []
    assert b.region_clearance_mm == 0.0


# ---- validation -------------------------------------------------------------
def _cfg(tmp_path, boxes, **model_kw):
    (tmp_path / "g_0000.rad").write_text(GROWTH_DECK, encoding="utf-8")
    (tmp_path / "g_0001.rad").write_text("/RUN\n/END\n", encoding="utf-8")
    cfg = Config()
    cfg.model.case_dir = str(tmp_path)
    cfg.model.growth_boxes = list(boxes)
    for k, v in model_kw.items():
        setattr(cfg.model, k, v)
    cfg.load_cases = [LoadCase(name="c", stem="g", sigma_allow=1.0)]
    return cfg


def _problems(cfg):
    return [str(p) for p in check_config(cfg)]


def test_validate_valid_deck_region_clean(tmp_path):
    _write_region(tmp_path)
    cfg = _cfg(tmp_path, [_deck_box()])
    assert not [p for p in _problems(cfg)
                if p.startswith("error") and "growth box" in p]


def test_validate_missing_deck_region_errors(tmp_path):
    cfg = _cfg(tmp_path, [_deck_box()])                  # no region file
    assert any(p.startswith("error") and "region deck not found" in p
               for p in _problems(cfg))


def test_validate_blank_region_rad_errors(tmp_path):
    cfg = _cfg(tmp_path, [GrowthBox(name="r", shape="deck")])
    assert any(p.startswith("error") and "needs region_rad" in p
               for p in _problems(cfg))


def test_validate_negative_clearance_errors(tmp_path):
    _write_region(tmp_path)
    cfg = _cfg(tmp_path, [_deck_box(region_clearance_mm=-1.0)])
    assert any(p.startswith("error") and "region_clearance_mm" in p
               for p in _problems(cfg))


def test_validate_absent_part_id_warns(tmp_path):
    _write_region(tmp_path)
    cfg = _cfg(tmp_path, [_deck_box(region_part_ids=[71000000, 99999999])])
    assert any(p.startswith("warning") and "no solid elements for part" in p
               and "99999999" in p for p in _problems(cfg))


# ---- composes with the forbid (positive/negative) polarity ------------------
_POS_E2 = GrowthBox(name="pos", x_min=0.4, x_max=0.6, y_min=0.4, y_max=0.6,
                    z_min=0.4, z_max=0.6)          # a positive box covering e2


def test_deck_region_as_negative_holds_candidate_void(tmp_path):
    """A deck region with forbid=True is an inline keep-out shaped by a deck:
    a positive box candidate inside it starts void AND is held void."""
    deck, mesh = _load(tmp_path)
    _write_region(tmp_path)                          # REGION_TET covers e2
    m = _model(tmp_path, [_POS_E2, _deck_box(name="neg", forbid=True)])
    cand = growth_candidate_mask(deck, mesh, m, log=_silent)
    assert cand.tolist() == [False, True, False, False]   # e2 starts void
    blocked = growth_blocked_mask(deck, mesh, m)
    assert blocked.tolist() == [False, True, False, False]  # e2 held void


def test_preview_deck_negative_region_row(tmp_path):
    deck, mesh = _load(tmp_path)
    _write_region(tmp_path)
    m = _model(tmp_path, [_POS_E2, _deck_box(name="neg", forbid=True)])
    prev = preview_growth_boxes(deck, mesh, m)
    neg_row = next(r for r in prev.rows if r.name == "neg")
    assert "forbidden" in neg_row.note.lower()


def test_gui_deck_region_forbid_roundtrips():
    from oropt.gui.boxes import (growth_boxes_from_records,
                                 records_from_growth_boxes)
    src = [GrowthBox(name="neg", shape="deck", region_rad="r.rad", forbid=True)]
    [row] = records_from_growth_boxes(src)
    assert row["forbid"] is True
    [back] = growth_boxes_from_records([row])
    assert back.shape_kind() == "deck" and back.forbid is True


# ---- GUI row round-trip -----------------------------------------------------
def test_gui_deck_region_records_roundtrip():
    from oropt.gui.boxes import (growth_boxes_from_records,
                                 records_from_growth_boxes)
    src = [GrowthBox(name="reg", shape="deck", region_rad="region_0000.rad",
                     region_part_ids=[71000000, 71000001],
                     region_clearance_mm=0.5)]
    rows = records_from_growth_boxes(src)
    assert rows[0]["shape"] == "deck"
    assert rows[0]["region_rad"] == "region_0000.rad"
    assert rows[0]["region_part_ids"] == "71000000,71000001"
    [back] = growth_boxes_from_records(rows)
    assert back.shape_kind() == "deck"
    assert back.region_rad == "region_0000.rad"
    assert back.region_part_ids == [71000000, 71000001]
    assert back.region_clearance_mm == 0.5


def test_gui_deck_row_blank_deck_dropped():
    from oropt.gui.boxes import growth_boxes_from_records
    # a deck row with no region_rad is incomplete -> dropped
    assert growth_boxes_from_records(
        [{"name": "r", "shape": "deck", "region_rad": None}]) == []
