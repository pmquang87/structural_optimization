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


def test_prepare_engine_anim_dt(mini_engine_path, tmp_path):
    out = tmp_path / "eng_0001.rad"
    prepare_engine(mini_engine_path, out, anim_dt=1.0)
    lines = out.read_text().splitlines()
    i = lines.index("/ANIM/DT")
    assert lines[i + 1].split() == ["0.", "1.0"]   # frequency rewritten
    assert "/IMPL/NONLIN/1" in lines                # implicit controls untouched
