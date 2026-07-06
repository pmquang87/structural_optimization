"""Fast-mode tied-linear deck building: topology discovery, tie sets, grnod
injection and the /IMPL/LINEAR engine writer, all on a tiny synthetic deck.

The synthetic deck mirrors the elevator-linkage load path in miniature: a loading
rigid body whose master node carries a /CLOAD, a fully-fixed rigid body, a
fixed-support /INTER/TYPE7 whose master is a /SURF/GRSHEL owned by the fixed body,
and six design nodes placed so exactly two seat against the loading cylinder and
two against the fixed support surface. (The real-deck reproduction — grnod
90007/90010 with 245/841 ties matching make_tied2 — is exercised by the offline
validation, not in the hermetic unit suite.)
"""
from __future__ import annotations

import numpy as np
import pytest

from oropt.deck import Deck
from oropt.fastmode import (FastModeError, build_fast_case, discover_tie,
                            index_deck, write_linear_engine)

DESIGN_PART = 60000000
DESIGN_MIN = 60000000

# Loading cyl nodes 1000/1001 seat against design 60000001/60000002 (0.1 mm);
# fixed support shell node 2001 seats against design 60000003/60000004 (< 0.3 mm),
# while 60000005 (1 mm) and 60000006 (far) stay free.
_DECK = """\
/NODE
  60000001 0.1 0.0 0.0
  60000002 1.1 0.0 0.0
  60000003 10.05 10.0 10.0
  60000004 10.0 10.0 10.2
  60000005 11.0 10.0 10.0
  60000006 5.0 5.0 5.0
/NODE
  1000 0.0 0.0 0.0
  1001 1.0 0.0 0.0
  2000 20.0 20.0 20.0
  2001 10.0 10.0 10.0
/TETRA4/60000000
  1 60000001 60000002 60000003 60000004
  2 60000003 60000004 60000005 60000006
/GRNOD/NODE/90007
loading_rb_nodes
  1000 1001
/GRNOD/NODE/90008
loading_master
  1000
/GRNOD/NODE/90010
fixed_rb_nodes
  2000 2001
/GRNOD/NODE/90011
fixed_master
  2000
/GRSHEL/SHEL/90017
fix_shell
  2001
/SURF/GRSHEL/90016
fix_surf
  90017
/RBODY/1000
loading
#  node_ID sens skew Ispher Mass grnd_ID ...
  1000 0 0 0 0 90007 0 0 0
/RBODY/2000
fixed
  2000 0 0 0 0 90010 0 0 0
/BCS/1
bc_load
#  Tra rot skew grnod
  100 111 0 90008
/BCS/2
bc_fix
  111 111 0 90011
/CLOAD/1
load
#funct Dir ...
  1 Y 0 0 90008 1 -1.0
/INTER/TYPE7/90001
self_contact
  90013 90014 4 0 0
/INTER/TYPE7/90002
support_contact
  90015 90016 4 0 0
/END
"""

_ENGINE = """\
/RUN/test_case/1
1
#
/TFILE
0.0001
#
/IMPL/NONLIN/1
/MON/ON
#
"""


def _deck(text: str = _DECK) -> Deck:
    return Deck(text.splitlines(), "\n", DESIGN_PART, DESIGN_MIN)


# ---- deck indexing ---------------------------------------------------------
def test_index_deck_parses_load_path_topology():
    idx = index_deck(_DECK.splitlines())
    assert idx.cloads == [90008]                       # /CLOAD -> its grnod
    assert idx.rbody == {1000: {"master": 1000, "grnd": 90007},
                         2000: {"master": 2000, "grnd": 90010}}
    assert {"tra": "111", "rot": "111", "grnod": 90011} in idx.bcs
    assert idx.surf_grshel[90016] == [90017]
    assert idx.grshel[90017] == [2001]
    assert [it["mast"] for it in idx.inters7] == [90014, 90016]
    assert idx.coords[1000] == (0.0, 0.0, 0.0)         # all node blocks, not just design


# ---- discovery -------------------------------------------------------------
def test_discover_tie_finds_groups_and_seated_nodes():
    tie = discover_tie(_deck(), DESIGN_MIN)
    assert tie.load_grnod == 90007                     # loading rbody grnod
    assert tie.fix_grnod == 90010                      # fixed rbody grnod
    # nearest design node to each loading-cyl node
    assert set(tie.load_tie.tolist()) == {60000001, 60000002}
    # design nodes within 0.3 mm of the fixed support surface (not the 1 mm one)
    assert set(tie.support_tie.tolist()) == {60000003, 60000004}


def test_discover_tie_support_thresh_widens_the_patch():
    # a 2 mm threshold also grabs 60000005 (1 mm away)
    tie = discover_tie(_deck(), DESIGN_MIN, support_thresh=2.0)
    assert set(tie.support_tie.tolist()) == {60000003, 60000004, 60000005}


def test_discover_tie_raises_without_cload():
    text = _DECK.replace("/CLOAD/1\nload\n#funct Dir ...\n  1 Y 0 0 90008 1 -1.0\n", "")
    with pytest.raises(FastModeError, match="no /CLOAD"):
        discover_tie(_deck(text), DESIGN_MIN)


def test_discover_tie_raises_without_fixed_support_surface():
    # drop the fixed-support TYPE7 -> no owned SURF/GRSHEL master remains
    text = _DECK.replace(
        "/INTER/TYPE7/90002\nsupport_contact\n  90015 90016 4 0 0\n", "")
    with pytest.raises(FastModeError, match="fixed-support contact-master"):
        discover_tie(_deck(text), DESIGN_MIN)


# ---- engine writer ---------------------------------------------------------
def test_write_linear_engine_is_plain_impl_linear(tmp_path):
    src = tmp_path / "src_0001.rad"
    src.write_text(_ENGINE, encoding="utf-8")
    dst = tmp_path / "dst_0001.rad"
    write_linear_engine(src, dst, anim_dt=1.0)
    text = dst.read_text()
    lines = text.splitlines()
    assert lines[0] == "/RUN/test_case/1"              # /RUN header reused
    assert lines[1] == "1"                             # Tstop reused
    assert "/IMPL/LINEAR" in text
    assert "/IMPL/PRINT/STIF" not in text              # not needed for the screen
    assert "/IMPL/NONLIN" not in text                  # nonlinear controls dropped
    for card in ("/ANIM/VECT/DISP", "/ANIM/BRICK/TENS/STRESS",
                 "/ANIM/ELEM/VONM", "/ANIM/ELEM/ENER"):
        assert card in text                            # stress + energy for extract()
    assert "0. 1.0" in text                            # /ANIM/DT value


# ---- grnod injection -------------------------------------------------------
def _grnod(path, gid):
    from pathlib import Path
    return set(index_deck(Path(path).read_text().splitlines()).grnod.get(gid, []))


def test_build_fast_case_full_alive_ties_all_seated_nodes(tmp_path):
    deck = _deck()
    tie = discover_tie(deck, DESIGN_MIN)
    alive = np.ones(deck.n_design_elements, bool)
    starter = tmp_path / "case_0000.rad"
    engine = tmp_path / "case_0001.rad"
    src = tmp_path / "src_0001.rad"
    src.write_text(_ENGINE, encoding="utf-8")
    deck.write(starter, alive, no_pin=set())
    info = build_fast_case(deck, alive, starter, src, engine, tie)
    assert info == {"load_tied": 2, "support_tied": 2}
    # the seated design nodes were folded into the two rigid-body node groups
    assert _grnod(starter, 90007) == {1000, 1001, 60000001, 60000002}
    assert _grnod(starter, 90010) == {2000, 2001, 60000003, 60000004}
    assert "/IMPL/LINEAR" in engine.read_text()


def test_build_fast_case_intersects_with_alive_mesh(tmp_path):
    # Killing element 1 drops nodes 60000001/2 from the alive mesh (Deck.write pins
    # them as free nodes); the load tie must NOT also make them rigid slaves.
    deck = _deck()
    tie = discover_tie(deck, DESIGN_MIN)
    alive = np.array([False, True])                    # keep only element 2
    starter = tmp_path / "case_0000.rad"
    engine = tmp_path / "case_0001.rad"
    src = tmp_path / "src_0001.rad"
    src.write_text(_ENGINE, encoding="utf-8")
    deck.write(starter, alive, no_pin=set())
    info = build_fast_case(deck, alive, starter, src, engine, tie)
    assert info == {"load_tied": 0, "support_tied": 2}
    assert _grnod(starter, 90007) == {1000, 1001}       # no dead nodes tied in
    assert _grnod(starter, 90010) == {2000, 2001, 60000003, 60000004}


# ---- live monitor ----------------------------------------------------------
def test_solve_activity_names_the_mode():
    from oropt.config import Config, LoadCase
    from oropt.loop import _solve_activity
    cfg = Config()
    cfg.load_cases = [LoadCase(name="pull", stem="s0", fast_mode=True),
                      LoadCase(name="push", stem="s1")]
    fast, slow = cfg.load_case_list()
    assert _solve_activity(3, fast, 0, 2) == \
        "iter 3: solving 'pull' — fast linear (tied) [case 1/2]"
    assert "full nonlinear" in _solve_activity(3, slow, 1, 2)
    # a single-case run omits the [case i/n] suffix
    assert _solve_activity(0, fast, 0, 1) == \
        "iter 0: solving 'pull' — fast linear (tied)"


def test_status_activity_roundtrips(tmp_path):
    from oropt.status import Status, read_status, write_status
    act = "iter 5: solving 'pull' — fast linear (tied)"
    write_status(tmp_path, Status(state="running", activity=act))
    assert read_status(tmp_path).activity == act
