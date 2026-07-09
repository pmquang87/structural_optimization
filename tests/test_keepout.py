"""Growth keep-out: an additional Radioss deck of nearby parts (never solved)
whose occupied volume forbids growth. Where a growth box overlaps that space the
candidates are held void every iteration, so no material grows into the neighbour
parts.

Covers the seams: neighbour-deck parsing (``deck.read_solid_geometry``, incl.
multiple /NODE blocks, part filtering and /BRICK decomposition), the point-in-
volume + clearance test (:class:`oropt.keepout.KeepOut`), keep-out subtraction in
:func:`oropt.loop.growth_candidate_mask` / :func:`~oropt.loop.growth_blocked_mask`
(and the reachability guard exempting held-void candidates), the preview and the
run-loop enforcement, the config round-trip, validation, and the auto-mesh PREPARE
exclusion.
"""
from __future__ import annotations

import numpy as np
import pytest

from oropt.beso import Beso
from oropt.config import (Beso as BesoCfg, Config, GrowthBox, LoadCase, Model)
from oropt.deck import Deck, read_solid_geometry
from oropt.keepout import KeepOut, resolve_keepout
from oropt.loop import (growth_blocked_mask, growth_candidate_mask,
                        preview_growth_boxes)
from oropt.mesh import Mesh
from oropt.validate import check_config

# ---- design deck (same geometry as tests/test_growth.py) --------------------
# e1(nodes 1-4) = anchored part; e2(2-5) centroid (.5,.5,.5); e3(5,6,11,12)
# centroid (1.5,1.75,1.5), touches the structure only via e2 (node 5); e4 island.
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

# A neighbour part: one tet covering the origin corner -> contains e2's centroid
# (.5,.5,.5) but not e3's (1.5,1.75,1.5). Node ids in the same coordinate frame.
NEIGHBOUR_TET_DECK = """\
/NODE
  70000001   0.0   0.0   0.0
  70000002   2.0   0.0   0.0
  70000003   0.0   2.0   0.0
  70000004   0.0   0.0   2.0
/PART/70000000
neighbour
         3         3         0
/TETRA4/70000000
  70000001  70000001  70000002  70000003  70000004
/END
"""


def _load(tmp_path, design_node_min=60000000):
    p = tmp_path / "g_0000.rad"
    p.write_text(GROWTH_DECK, encoding="utf-8")
    deck = Deck.load(p, design_part_id=60000000, design_node_min=design_node_min)
    return deck, Mesh.from_deck(deck)


def _write_neighbour(tmp_path, text=NEIGHBOUR_TET_DECK, name="neighbour_0000.rad"):
    (tmp_path / name).write_text(text, encoding="utf-8")
    return name


def _model(tmp_path, boxes, **kw):
    return Model(case_dir=str(tmp_path), growth_boxes=list(boxes), **kw)


def _silent(_msg):
    pass


# ---- neighbour-deck parsing (read_solid_geometry) ---------------------------
def test_read_solid_geometry_tets(tmp_path):
    _write_neighbour(tmp_path)
    tet_xyz, node_xyz, surf_xyz, parts = read_solid_geometry(
        tmp_path / "neighbour_0000.rad")
    assert tet_xyz.shape == (1, 4, 3)
    assert parts == [70000000]
    assert len(node_xyz) == 4
    # the tet is exactly the four declared nodes
    assert np.allclose(sorted(tet_xyz[0].tolist()),
                       sorted([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]]))


def test_read_solid_geometry_multiple_node_blocks(tmp_path):
    """A converted deck emits one /NODE block per include; all are read."""
    text = """\
/NODE
  70000001   0.0   0.0   0.0
  70000002   2.0   0.0   0.0
/PART/70000000
p
/NODE
  70000003   0.0   2.0   0.0
  70000004   0.0   0.0   2.0
/TETRA4/70000000
  70000001  70000001  70000002  70000003  70000004
/END
"""
    (tmp_path / "n.rad").write_text(text, encoding="utf-8")
    tet_xyz, node_xyz, surf_xyz, parts = read_solid_geometry(tmp_path / "n.rad")
    assert tet_xyz.shape == (1, 4, 3)          # nodes from both blocks resolved
    assert len(node_xyz) == 4


def test_read_solid_geometry_part_filter(tmp_path):
    text = """\
/NODE
  70000001   0.0   0.0   0.0
  70000002   2.0   0.0   0.0
  70000003   0.0   2.0   0.0
  70000004   0.0   0.0   2.0
  70000005   9.0   9.0   9.0
  70000006  11.0   9.0   9.0
  70000007   9.0  11.0   9.0
  70000008   9.0   9.0  11.0
/TETRA4/70000000
  70000001  70000001  70000002  70000003  70000004
/TETRA4/70000001
  70000002  70000005  70000006  70000007  70000008
/END
"""
    (tmp_path / "n.rad").write_text(text, encoding="utf-8")
    _, _, _, parts_all = read_solid_geometry(tmp_path / "n.rad")
    assert parts_all == [70000000, 70000001]
    tets, _, _, parts = read_solid_geometry(tmp_path / "n.rad", part_ids=[70000001])
    assert parts == [70000001]
    # only the far tet survived the filter
    assert np.allclose(tets.mean(axis=1)[0], [9.5, 9.5, 9.5], atol=0.6)


def test_read_solid_geometry_brick_decomposition(tmp_path):
    """An 8-node /BRICK is split into 6 tets covering the hex volume exactly."""
    text = """\
/NODE
  70000001   0.0   0.0   0.0
  70000002   1.0   0.0   0.0
  70000003   1.0   1.0   0.0
  70000004   0.0   1.0   0.0
  70000005   0.0   0.0   1.0
  70000006   1.0   0.0   1.0
  70000007   1.0   1.0   1.0
  70000008   0.0   1.0   1.0
/BRICK/70000000
  70000001  70000001  70000002  70000003  70000004  70000005  70000006  70000007  70000008
/END
"""
    (tmp_path / "b.rad").write_text(text, encoding="utf-8")
    tet_xyz, node_xyz, surf_xyz, parts = read_solid_geometry(tmp_path / "b.rad")
    assert tet_xyz.shape == (6, 4, 3)          # 1 brick -> 6 tets
    assert len(node_xyz) == 8
    ko = KeepOut(tet_xyz, node_xyz, surf_xyz, parts, 0.0, "b.rad")
    probe = np.array([[.5, .5, .5], [.1, .9, .1], [2., 2., 2.]])
    assert ko.block_mask(probe).tolist() == [True, True, False]


def test_read_solid_geometry_no_solids_raises(tmp_path):
    (tmp_path / "empty.rad").write_text("/NODE\n  1  0. 0. 0.\n/END\n",
                                        encoding="utf-8")
    with pytest.raises(ValueError, match="no solid"):
        read_solid_geometry(tmp_path / "empty.rad")


def test_read_solid_geometry_missing_node_raises(tmp_path):
    text = ("/NODE\n  70000001  0. 0. 0.\n/TETRA4/70000000\n"
            "  1  70000001  70000002  70000003  70000004\n/END\n")
    (tmp_path / "n.rad").write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="absent"):
        read_solid_geometry(tmp_path / "n.rad")


# ---- KeepOut.block_mask (inside + clearance) --------------------------------
def test_keepout_block_mask_inside_and_clearance(tmp_path):
    tet_xyz, node_xyz, surf_xyz, parts = read_solid_geometry(
        tmp_path / _write_neighbour(tmp_path))
    pts = np.array([[.5, .5, .5],             # inside the tet
                    [1.5, 1.75, 1.5]])         # outside (e3 centroid, ~2.14 away)
    assert KeepOut(tet_xyz, node_xyz, surf_xyz, parts, 0.0, "x").block_mask(pts).tolist() \
        == [True, False]
    # a clearance band wide enough reaches the second point
    assert KeepOut(tet_xyz, node_xyz, surf_xyz, parts, 2.2, "x").block_mask(pts).tolist() \
        == [True, True]
    assert KeepOut(tet_xyz, node_xyz, surf_xyz, parts, 2.0, "x").block_mask(pts).tolist() \
        == [True, False]


# ---- resolve_keepout --------------------------------------------------------
def test_resolve_keepout_none_when_unconfigured(tmp_path):
    assert resolve_keepout(_model(tmp_path, [])) is None


def test_resolve_keepout_relative_path(tmp_path):
    name = _write_neighbour(tmp_path)
    ko = resolve_keepout(_model(tmp_path, [BOX_E2], growth_keepout_rad=name))
    assert ko is not None and ko.part_ids == [70000000]


def test_resolve_keepout_missing_deck_raises(tmp_path):
    m = _model(tmp_path, [BOX_E2], growth_keepout_rad="nope.rad")
    with pytest.raises(ValueError, match="not found"):
        resolve_keepout(m)


# ---- growth_blocked_mask / growth_candidate_mask ----------------------------
def test_blocked_mask_all_false_without_keepout(tmp_path):
    deck, mesh = _load(tmp_path)
    assert not growth_blocked_mask(deck, mesh, _model(tmp_path, [BOX_E2])).any()


def test_blocked_mask_selects_overlapping_candidate(tmp_path):
    deck, mesh = _load(tmp_path)
    m = _model(tmp_path, [BOX_E2], growth_keepout_rad=_write_neighbour(tmp_path))
    # e2 (index 1) is inside the neighbour tet -> blocked; nothing else
    assert growth_blocked_mask(deck, mesh, m).tolist() == [False, True, False, False]


def test_candidate_mask_keeps_blocked_void_and_logs(tmp_path):
    """A fully-blocked box: e2 still starts void (returned in the mask) but the
    log flags that the region can grow nothing; the growable guards do not trip."""
    deck, mesh = _load(tmp_path)
    m = _model(tmp_path, [BOX_E2], growth_keepout_rad=_write_neighbour(tmp_path))
    lines: list[str] = []
    mask = growth_candidate_mask(deck, mesh, m, log=lines.append)
    assert mask.tolist() == [False, True, False, False]         # void-start intact
    assert growth_blocked_mask(deck, mesh, m).tolist() == [False, True, False, False]
    assert any("held void" in ln for ln in lines)
    assert any("can grow nothing" in ln for ln in lines)


def test_candidate_mask_clearance_blocks_far_candidate(tmp_path):
    deck, mesh = _load(tmp_path)
    name = _write_neighbour(tmp_path)
    m = _model(tmp_path, [BOX_E2, BOX_E3], growth_keepout_rad=name,
               growth_keepout_clearance_mm=2.2)
    # both candidates now inside the forbidden band -> both held void
    assert growth_blocked_mask(deck, mesh, m).tolist() == [False, True, True, False]


def test_keepout_stranding_growable_candidate_raises(tmp_path):
    """e3 reaches the structure only through e2; holding e2 void strands e3, so
    the reachability guard (through non-blocked paths) aborts at run start."""
    deck, mesh = _load(tmp_path)
    m = _model(tmp_path, [BOX_E2, BOX_E3],
               growth_keepout_rad=_write_neighbour(tmp_path))
    with pytest.raises(ValueError, match="'b3'.*share no nodes"):
        growth_candidate_mask(deck, mesh, m, log=_silent)


def test_keepout_no_overlap_is_noop(tmp_path):
    """A neighbour far from every box removes nothing (a no-op keep-out)."""
    deck, mesh = _load(tmp_path)
    far = """\
/NODE
  70000001  50.0  50.0  50.0
  70000002  52.0  50.0  50.0
  70000003  50.0  52.0  50.0
  70000004  50.0  50.0  52.0
/TETRA4/70000000
  70000001  70000001  70000002  70000003  70000004
/END
"""
    name = _write_neighbour(tmp_path, text=far, name="far_0000.rad")
    m = _model(tmp_path, [BOX_E2, BOX_E3], growth_keepout_rad=name)
    assert not growth_blocked_mask(deck, mesh, m).any()
    assert growth_candidate_mask(deck, mesh, m, log=_silent).tolist() \
        == [False, True, True, False]


# ---- run-loop enforcement (held void even when the update would grow it) ----
def test_hold_void_masks_a_grown_candidate():
    """BESO's bi-directional add-back would grow a void candidate; the loop holds
    the keep-out subset void with ``alive & ~blocked`` -- this documents why that
    per-iteration mask is required."""
    conn = np.array([[i, i + 1, i + 2, i + 3] for i in range(5)])
    mesh = Mesh(centroids=np.zeros((5, 3)), volumes=np.ones(5),
                conn_rows=conn, n_nodes=int(conn.max()) + 1, design_node_min=0)
    protected = np.array([True, False, False, False, False])
    alive0 = np.array([True, True, True, True, False])         # elem 4 = candidate
    sens = np.array([1.0, 1.0, 1.0, 1.0, 5.0])                 # ranks it best
    cfg = BesoCfg(filter_radius=0.0, target_volume_fraction=1.0,
                  evolution_rate=0.2, max_add_ratio=1.0)
    grown = Beso(mesh, cfg, protected).update(alive0, sens, target_vf=1.0)
    assert grown[4]                                            # update grows it
    blocked = np.array([False, False, False, False, True])     # keep-out forbids it
    assert not (grown & ~blocked)[4]                           # loop holds it void


# ---- preview ----------------------------------------------------------------
def test_preview_reports_keepout_summary_and_row_note(tmp_path):
    deck, mesh = _load(tmp_path)
    m = _model(tmp_path, [BOX_E2, BOX_E3], growth_keepout_rad=_write_neighbour(tmp_path))
    prev = preview_growth_boxes(deck, mesh, m)
    assert "held void" in prev.keepout
    assert "1 candidate" in prev.keepout                       # only e2 blocked
    b2 = next(r for r in prev.rows if r.name == "b2")
    assert "held void by keep-out" in b2.note


def test_preview_noop_keepout_note(tmp_path):
    deck, mesh = _load(tmp_path)
    far = NEIGHBOUR_TET_DECK.replace("0.0   0.0   0.0", "50.0  50.0  50.0")
    name = _write_neighbour(tmp_path, text=far, name="far_0000.rad")
    m = _model(tmp_path, [BOX_E2], growth_keepout_rad=name)
    prev = preview_growth_boxes(deck, mesh, m)
    assert "no-op" in prev.keepout


def test_preview_broken_keepout_never_raises(tmp_path):
    deck, mesh = _load(tmp_path)
    m = _model(tmp_path, [BOX_E2], growth_keepout_rad="missing.rad")
    prev = preview_growth_boxes(deck, mesh, m)          # must not raise
    assert "error" in prev.keepout.lower()


# ---- config round-trip ------------------------------------------------------
def test_keepout_fields_yaml_roundtrip(tmp_path):
    cfg = Config.from_dict({"model": {
        "growth_keepout_rad": "neighbour_0000.rad",
        "growth_keepout_part_ids": [70000000, 70000001],
        "growth_keepout_clearance_mm": 0.5}})
    assert cfg.model.growth_keepout_rad == "neighbour_0000.rad"
    p = tmp_path / "c.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)
    assert back.model.growth_keepout_rad == "neighbour_0000.rad"
    assert back.model.growth_keepout_part_ids == [70000000, 70000001]
    assert back.model.growth_keepout_clearance_mm == 0.5


def test_keepout_defaults_off():
    assert Model().growth_keepout_rad is None
    assert Model().growth_keepout_part_ids == []
    assert Model().growth_keepout_clearance_mm == 0.0


# ---- validation -------------------------------------------------------------
def _cfg(tmp_path, boxes, **model_kw):
    # Write valid design decks so the only problems are keep-out ones (the deck
    # -existence errors otherwise carry the tmp path, which may contain "keepout").
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


def test_validate_missing_keepout_deck_errors(tmp_path):
    cfg = _cfg(tmp_path, [BOX_E2], growth_keepout_rad="nope.rad")
    assert any(p.startswith("error") and "keep-out deck not found" in p
               for p in _problems(cfg))


def test_validate_keepout_without_boxes_warns(tmp_path):
    name = _write_neighbour(tmp_path)
    cfg = _cfg(tmp_path, [], growth_keepout_rad=name)
    assert any(p.startswith("warning") and "no-op" in p for p in _problems(cfg))


def test_validate_keepout_partially_absent_part_id_warns(tmp_path):
    """A present + an absent requested part -> the run still builds a keep-out
    from the present part, so validation warns (not errors) about the absent id."""
    name = _write_neighbour(tmp_path)
    cfg = _cfg(tmp_path, [BOX_E2], growth_keepout_rad=name,
               growth_keepout_part_ids=[70000000, 99999999])
    assert any(p.startswith("warning") and "no solid elements for part"
               in p and "99999999" in p for p in _problems(cfg))


def test_validate_negative_clearance_warns_not_errors(tmp_path):
    """Negative clearance = allowed penetration depth (a real feature now):
    validation flags it as a WARNING (guarding against a sign typo) but must
    not block the launch."""
    name = _write_neighbour(tmp_path)
    cfg = _cfg(tmp_path, [BOX_E2], growth_keepout_rad=name,
               growth_keepout_clearance_mm=-1.0)
    probs = _problems(cfg)
    assert any(p.startswith("warning") and "penetrate" in p for p in probs)
    assert not any(p.startswith("error") and "clearance" in p for p in probs)


def test_validate_nan_clearance_is_error(tmp_path):
    """NaN slipped through the old `< 0` check (NaN comparisons are all False)
    and silently disabled the clearance band downstream."""
    name = _write_neighbour(tmp_path)
    cfg = _cfg(tmp_path, [BOX_E2], growth_keepout_rad=name,
               growth_keepout_clearance_mm=float("nan"))
    assert any(p.startswith("error") and "finite" in p for p in _problems(cfg))


def test_validate_valid_keepout_clean(tmp_path):
    name = _write_neighbour(tmp_path)
    cfg = _cfg(tmp_path, [BOX_E2], growth_keepout_rad=name)
    assert not [p for p in _problems(cfg)
                if p.startswith("error") and "keep-out" in p]


# ---- GUI (the keep-out expander round-trips onto cfg) -----------------------
def test_gui_keepout_inputs_roundtrip(tmp_path, monkeypatch):
    from pathlib import Path

    from streamlit.testing.v1 import AppTest

    import oropt
    import oropt.gui.runstate as runstate

    monkeypatch.setattr(runstate, "find_active_run", lambda *a, **k: None)
    cfg = Config()
    cfg.work_dir = str(tmp_path / "work")
    Path(cfg.work_dir).mkdir()
    cfg.load_cases = [LoadCase(name="a", stem="g", sigma_allow=1.0)]
    cfg.model.growth_boxes = [BOX_E2]                 # so the keep-out UI renders
    cfg.model.growth_keepout_rad = "neighbour_0000.rad"
    cfg.model.growth_keepout_part_ids = [70000000]
    cfg.model.growth_keepout_clearance_mm = 0.5
    cfg_path = tmp_path / "cfg.yaml"
    cfg.to_yaml(cfg_path)
    app = Path(oropt.__file__).resolve().parent / "gui" / "app.py"
    at = AppTest.from_file(str(app), default_timeout=60)
    at.run()
    at.sidebar.text_input[0].set_value(str(cfg_path)).run()
    assert not at.exception
    # the keep-out deck path input shows the configured value
    ko = [t for t in at.text_input if t.value == "neighbour_0000.rad"]
    assert ko, "keep-out deck text input not rendered with its configured value"


# ---- negative clearance: allowed penetration band ---------------------------
def _brick_grid_deck(n=3):
    """An n x n x n /BRICK grid over [0,n]^3 -- has ((n-1)^3) interior nodes, so
    surface extraction is actually exercised (a tiny all-surface mesh can't
    tell the surface cloud from the full node cloud)."""
    def nid(i, j, k):
        return 70000001 + i + j * (n + 1) + k * (n + 1) ** 2
    lines = ["/NODE"]
    for k in range(n + 1):
        for j in range(n + 1):
            for i in range(n + 1):
                lines.append(f"  {nid(i, j, k)}   {i}.0   {j}.0   {k}.0")
    lines.append("/BRICK/70000000")
    eid = 1
    for k in range(n):
        for j in range(n):
            for i in range(n):
                c = [nid(i, j, k), nid(i + 1, j, k), nid(i + 1, j + 1, k),
                     nid(i, j + 1, k), nid(i, j, k + 1), nid(i + 1, j, k + 1),
                     nid(i + 1, j + 1, k + 1), nid(i, j + 1, k + 1)]
                lines.append("  " + "  ".join(str(v) for v in [eid] + c))
                eid += 1
    lines.append("/END")
    return "\n".join(lines) + "\n"


def test_surface_nodes_exclude_interior_and_brick_interfaces(tmp_path):
    """The 3x3x3 brick grid has 64 nodes, 8 of them interior. The surface set
    must hold exactly the 56 shell nodes: interior nodes excluded, and the
    interior brick-brick interfaces (whose 6-tet split diagonals need not
    match across elements) must not read as surface."""
    (tmp_path / "grid.rad").write_text(_brick_grid_deck(3), encoding="utf-8")
    tet_xyz, node_xyz, surf_xyz, parts = read_solid_geometry(tmp_path / "grid.rad")
    assert tet_xyz.shape == (27 * 6, 4, 3)
    assert len(node_xyz) == 64
    assert len(surf_xyz) == 56                       # 64 - 8 interior
    # every surface node really is on the shell of [0,3]^3
    on_shell = ((surf_xyz == 0.0) | (surf_xyz == 3.0)).any(axis=1)
    assert on_shell.all()


def test_negative_clearance_allows_shallow_penetration(tmp_path):
    """clearance < 0 = allowed penetration depth: only candidates DEEPER than
    |clearance| below the neighbour surface stay blocked."""
    (tmp_path / "grid.rad").write_text(_brick_grid_deck(3), encoding="utf-8")
    tet_xyz, node_xyz, surf_xyz, parts = read_solid_geometry(tmp_path / "grid.rad")
    pts = np.array([[1.5, 1.5, 0.2],     # 0.2 below the bottom face (shallow)
                    [1.5, 1.5, 1.5],     # dead centre (deep)
                    [1.5, 1.5, -0.5]])   # outside
    ko0 = KeepOut(tet_xyz, node_xyz, surf_xyz, parts, 0.0, "x")
    assert ko0.block_mask(pts).tolist() == [True, True, False]
    ko = KeepOut(tet_xyz, node_xyz, surf_xyz, parts, -1.0, "x")
    # nearest surface NODE: 0.735 for the shallow point (allowed), 1.658 for
    # the centre (still blocked). The centre also proves interior nodes are
    # not in the surface cloud: (1,1,1) sits only 0.866 away and would
    # otherwise wrongly free it.
    assert ko.block_mask(pts).tolist() == [False, True, False]
    # a tighter band keeps both blocked
    tight = KeepOut(tet_xyz, node_xyz, surf_xyz, parts, -0.5, "x")
    assert tight.block_mask(pts).tolist() == [True, True, False]


def test_negative_clearance_through_blocked_mask(tmp_path):
    """End-to-end: e2's candidate sits 0.866 deep in the neighbour tet -- an
    allowed penetration of 1.0 frees it, 0.5 does not."""
    deck, mesh = _load(tmp_path)
    name = _write_neighbour(tmp_path)
    deep = _model(tmp_path, [BOX_E2], growth_keepout_rad=name,
                  growth_keepout_clearance_mm=-1.0)
    assert not growth_blocked_mask(deck, mesh, deep).any()
    shallow = _model(tmp_path, [BOX_E2], growth_keepout_rad=name,
                     growth_keepout_clearance_mm=-0.5)
    assert growth_blocked_mask(deck, mesh, shallow).tolist() \
        == [False, True, False, False]


def test_preview_notes_allowed_penetration(tmp_path):
    deck, mesh = _load(tmp_path)
    m = _model(tmp_path, [BOX_E2], growth_keepout_rad=_write_neighbour(tmp_path),
               growth_keepout_clearance_mm=-0.5)
    prev = preview_growth_boxes(deck, mesh, m)
    assert "allowed penetration 0.5" in prev.keepout


def test_resolve_keepout_nan_clearance_raises(tmp_path):
    m = _model(tmp_path, [BOX_E2], growth_keepout_rad=_write_neighbour(tmp_path),
               growth_keepout_clearance_mm=float("nan"))
    with pytest.raises(ValueError, match="finite"):
        resolve_keepout(m)


# ---- resolve_keepout memo ----------------------------------------------------
def test_resolve_keepout_memo_reuses_and_invalidates(tmp_path):
    name = _write_neighbour(tmp_path)
    m = _model(tmp_path, [BOX_E2], growth_keepout_rad=name)
    a = resolve_keepout(m)
    b = resolve_keepout(m)
    assert b is a                                    # same inputs -> same object
    # a different clearance is a different keep-out
    m2 = _model(tmp_path, [BOX_E2], growth_keepout_rad=name,
                growth_keepout_clearance_mm=-1.0)
    c = resolve_keepout(m2)
    assert c is not a and c.clearance == -1.0
    # a changed deck on disk invalidates (content/size/mtime signature)
    (tmp_path / name).write_text(
        NEIGHBOUR_TET_DECK.replace("2.0   0.0   0.0", "3.0   0.0   0.0"),
        encoding="utf-8")
    d = resolve_keepout(m)
    assert d is not a
    assert float(d.node_xyz[:, 0].max()) == 3.0      # the NEW geometry was read
