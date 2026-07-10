"""Deck parsing + filtered-write (element deletion, free-node pinning, verbatim)."""
import numpy as np

from oropt.deck import Deck, prepare_engine


def test_parse(mini_deck_path):
    d = Deck.load(mini_deck_path, 60000000, 60000000)
    assert d.n_design_elements == 2
    assert d.elem_ids.tolist() == [60000001, 60000002]
    assert set(d.node_ids.tolist()) == {60000001, 60000002, 60000003,
                                        60000004, 60000005, 10000001}
    assert d.elem_conn.shape == (2, 4)
    # /GRNOD/NODE/60000000 lists design nodes 60000001, 60000002
    assert d.protected_nodes == frozenset({60000001, 60000002})
    assert set(d.group_nodes(60000000).tolist()) == {60000001, 60000002}


def test_write_all_alive_roundtrip(mini_deck_path, tmp_path):
    d = Deck.load(mini_deck_path, 60000000, 60000000)
    out = tmp_path / "all_0000.rad"
    summ = d.write(out, np.ones(2, bool))
    assert summ["elements_alive"] == 2 and summ["free_nodes_pinned"] == 0
    text = out.read_text()
    # non-element cards preserved verbatim
    assert "/MAT/LAW1/3" in text and "/PROP/SOLID/3" in text and "/BCS/90009" in text
    d2 = Deck.load(out, 60000000, 60000000)
    assert d2.n_design_elements == 2


def test_delete_pins_free_node(mini_deck_path, tmp_path):
    d = Deck.load(mini_deck_path, 60000000, 60000000)
    out = tmp_path / "del_0000.rad"
    # delete the 2nd tet -> node 60000005 (only in that tet) becomes free
    summ = d.write(out, np.array([True, False]), no_pin={60000001, 60000002})
    assert summ["elements_alive"] == 1
    assert summ["free_nodes_pinned"] == 1
    text = out.read_text()
    assert "/GRNOD/NODE/91000001" in text and "/BCS/91000002" in text
    assert "60000005" in text.split("/GRNOD/NODE/91000001")[1]  # pinned node listed
    d2 = Deck.load(out, 60000000, 60000000)
    assert d2.elem_ids.tolist() == [60000001]


def test_no_pin_excludes_constrained(mini_deck_path, tmp_path):
    d = Deck.load(mini_deck_path, 60000000, 60000000)
    out = tmp_path / "del2_0000.rad"
    # delete both tets; every design node becomes free, but no_pin excludes the
    # two symmetry nodes -> only the other three get pinned
    summ = d.write(out, np.array([False, False]), no_pin={60000001, 60000002})
    assert summ["free_nodes_pinned"] == 3


# A design-range node in no element (the converter's --rigid-cog-master synthesises
# such element-free /RBODY master nodes). It is structural, not orphaned by deletion.
_MASTER_DECK = """\
/NODE
  60000001   0.0   0.0   0.0
  60000002   1.0   0.0   0.0
  60000003   0.0   1.0   0.0
  60000004   0.0   0.0   1.0
  60099999  50.0  50.0  50.0
/TETRA4/60000000
  60000001  60000001  60000002  60000003  60000004
/GRNOD/NODE/60000000
sym
  60000001
/BCS/90009
bc
#  Tra rot   skew_ID  grnod_ID
   111 111         0     60000000
/END
"""


def test_element_free_design_node_never_pinned(tmp_path):
    """A design-range node in no element (a synthesised rigid-body master) is
    structural, not orphaned by deletion -- it must never be pinned, even at full
    volume. Pinning one with a full-fix /BCS double-constrains the rigid master
    (locks a loaded master's free DOFs -> zero external work) and OpenRadioss flags
    it as an incompatible kinematic condition -> AUTOSPC -> dead solve."""
    p = tmp_path / "master_0000.rad"
    p.write_text(_MASTER_DECK, encoding="utf-8")
    d = Deck.load(p, 60000000, 60000000)
    assert 60099999 in d.node_ids.tolist()             # present, design-range
    out = tmp_path / "m_0000.rad"
    summ = d.write(out, np.ones(1, bool))              # full volume, nothing deleted
    assert summ["free_nodes_pinned"] == 0              # element-free node not pinned
    assert "/GRNOD/NODE/91000001" not in out.read_text()   # no free-node block at all


def test_prepare_engine_anim_dt(mini_engine_path, tmp_path):
    out = tmp_path / "eng_0001.rad"
    prepare_engine(mini_engine_path, out, anim_dt=1.0)
    lines = out.read_text().splitlines()
    i = lines.index("/ANIM/DT")
    assert lines[i + 1].split() == ["0.", "1.0"]   # frequency rewritten
    assert "/IMPL/NONLIN/1" in lines                # implicit controls untouched


def test_prepare_engine_default_writes_no_stress_tensor(mini_engine_path, tmp_path):
    """anim_stress_tensor defaults OFF: output byte-identical to before the knob."""
    out = tmp_path / "eng_0001.rad"
    prepare_engine(mini_engine_path, out)           # no options at all
    assert out.read_text() == mini_engine_path.read_text()
    assert "/ANIM/BRICK/TENS/STRESS" not in out.read_text()


def test_prepare_engine_injects_stress_tensor_once(mini_engine_path, tmp_path):
    out = tmp_path / "eng_0001.rad"
    prepare_engine(mini_engine_path, out, anim_dt=1.0, anim_stress_tensor=True)
    lines = out.read_text().splitlines()
    assert lines.count("/ANIM/BRICK/TENS/STRESS") == 1
    # placed with the other /ANIM cards: right after the /ANIM/DT value line,
    # which itself is intact (never split from its header)
    i = lines.index("/ANIM/DT")
    assert lines[i + 1].split() == ["0.", "1.0"]
    assert lines[i + 2] == "/ANIM/BRICK/TENS/STRESS"
    assert "/IMPL/NONLIN/1" in lines                # implicit controls untouched


def test_prepare_engine_stress_tensor_idempotent(mini_engine_path, tmp_path):
    """Re-preparing a deck that already requests the tensor adds nothing."""
    once = tmp_path / "eng1_0001.rad"
    prepare_engine(mini_engine_path, once, anim_stress_tensor=True)
    twice = tmp_path / "eng2_0001.rad"
    prepare_engine(once, twice, anim_stress_tensor=True)
    assert twice.read_text() == once.read_text()
    assert twice.read_text().splitlines().count("/ANIM/BRICK/TENS/STRESS") == 1


def test_prepare_engine_stress_tensor_no_anim_cards(tmp_path):
    """A deck with no /ANIM card at all still gets the tensor request (appended)."""
    src = tmp_path / "bare_0001.rad"
    src.write_text("/RUN/x/1\n1\n/IMPL/NONLIN/1\n", encoding="utf-8")
    out = tmp_path / "bare_out_0001.rad"
    prepare_engine(src, out, anim_stress_tensor=True)
    lines = out.read_text().splitlines()
    assert lines.count("/ANIM/BRICK/TENS/STRESS") == 1
    assert lines[-1] == "/ANIM/BRICK/TENS/STRESS"


# ---- /BOX/RECTA parsing (growth-box deck references) ------------------------
# Two /BOX/RECTA cards: 7000001 has a leading skew line + reversed corners (so the
# parser must skip the skew line and normalise min<=max); 7000002 carries a
# trailing /unit_ID on the header and no skew line.
_BOX_DECK = """\
/NODE
  60000001   0.0   0.0   0.0
  60000002   1.0   0.0   0.0
  60000003   0.0   1.0   0.0
  60000004   0.0   0.0   1.0
/TETRA4/60000000
  60000001  60000001  60000002  60000003  60000004
/BOX/RECTA/7000001
growth_rib
                    0
        40.0    5.0   25.0
        10.0   -5.0    0.0
/BOX/RECTA/7000002/13
gusset
        -20.0   -5.0    0.0
          0.0    5.0   12.0
/BOX/SPHER/7000003
ball
                    0                 4.0
          1.0    2.0    3.0
/BOX/CYLIN/7000004
rod
                    0                 6.0
          0.0    0.0    0.0
         10.0    0.0    0.0
/BOX/RECTA/7000005
oriented
                    9                 0.0
          0.0    0.0    0.0
          2.0    1.0    1.0
/SKEW/FIX/9
frame9
          5.0    0.0    0.0
          0.0    1.0    0.0
          0.0    0.0    1.0
/END
"""


def _box_deck(tmp_path):
    p = tmp_path / "boxes_0000.rad"
    p.write_text(_BOX_DECK, encoding="utf-8")
    return Deck.load(p, design_part_id=60000000, design_node_min=60000000)


def test_box_recta_normalises_corners_and_skips_skew(tmp_path):
    d = _box_deck(tmp_path)
    # reversed corners + a skew line before them -> normalised (min, max) per axis
    assert d.box_recta(7000001) == (10.0, 40.0, -5.0, 5.0, 0.0, 25.0)


def test_box_recta_header_with_unit_id(tmp_path):
    d = _box_deck(tmp_path)
    assert d.box_recta(7000002) == (-20.0, 0.0, -5.0, 5.0, 0.0, 12.0)


def test_box_recta_absent_returns_none(tmp_path):
    d = _box_deck(tmp_path)
    assert d.box_recta(9999999) is None


def test_box_sphere_card(tmp_path):
    d = _box_deck(tmp_path)
    spec = d.box(7000003)
    assert spec == {"shape": "sphere", "cx": 1.0, "cy": 2.0, "cz": 3.0,
                    "radius": 2.0}                 # Diam 4.0 -> radius 2.0
    assert d.box_recta(7000003) is None            # not a rectangular box


def test_box_cylinder_card(tmp_path):
    d = _box_deck(tmp_path)
    spec = d.box(7000004)
    assert spec == {"shape": "cylinder", "x1": 0.0, "y1": 0.0, "z1": 0.0,
                    "x2": 10.0, "y2": 0.0, "z2": 0.0, "radius": 3.0}


def test_box_recta_with_skew_is_oriented(tmp_path):
    d = _box_deck(tmp_path)
    spec = d.box(7000005)
    assert spec["shape"] == "box"
    assert (spec["x_min"], spec["x_max"]) == (0.0, 2.0)
    # skew_ID 9 -> /SKEW/FIX/9 frame attached
    assert spec["origin"] == [5.0, 0.0, 0.0]
    assert spec["x_axis"] == [0.0, 1.0, 0.0]
    assert spec["xy_axis"] == [0.0, 0.0, 1.0]


def test_skew_fix_reads_frame(tmp_path):
    d = _box_deck(tmp_path)
    assert d.skew_fix(9) == [[5.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert d.skew_fix(404) is None


# ---- title lines that look like data -----------------------------------------
def test_group_nodes_digit_leading_title(tmp_path):
    """A /GRNOD/NODE title starting with a digit (e.g. '2nd_symmetry_set') used
    to be classified as a node-id line and crash int() with a raw ValueError at
    every run start."""
    deck = _MASTER_DECK.replace("/GRNOD/NODE/60000000\nsym\n",
                                "/GRNOD/NODE/60000000\n2nd_symmetry_set\n")
    p = tmp_path / "t_0000.rad"
    p.write_text(deck, encoding="utf-8")
    d = Deck.load(p, 60000000, 60000000)
    assert d.group_nodes(60000000).tolist() == [60000001]   # title skipped, data kept


def test_box_numeric_title_not_swallowed(tmp_path):
    """A purely numeric /BOX title (e.g. '1234') used to be consumed as the
    skew_ID [Diam] line, dropping the real one -- box() then reported the card
    missing (or attached an unrelated /SKEW frame)."""
    deck = _BOX_DECK.replace("/BOX/SPHER/7000003\nball\n",
                             "/BOX/SPHER/7000003\n1234\n")
    p = tmp_path / "nb_0000.rad"
    p.write_text(deck, encoding="utf-8")
    d = Deck.load(p, 60000000, 60000000)
    b = d.box(7000003)
    assert b is not None
    assert b["shape"] == "sphere" and b["radius"] == 2.0    # real Diam line read
