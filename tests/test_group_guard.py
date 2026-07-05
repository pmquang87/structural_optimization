"""Run-start guard for configured /GRNOD/NODE group ids that select zero nodes.

The live failure: stress_exclude_group_ids said 999999998 but the deck's group
is /GRNOD/NODE/99999998 (one digit fewer). Deck.group_nodes silently returned
empty, the exclusion did nothing, and a multi-hour run chased a spurious
INFEASIBLE signal from the load-introduction region the config meant to ignore.
:func:`oropt.loop.validate_group_ids` now fails at run start instead, naming
the deck's real group ids; :func:`oropt.loop.preview_growth_boxes` surfaces the
same message (``group_guard``) for the GUI's preview panel.
"""
from __future__ import annotations

import pytest

from oropt import loop as loop_mod
from oropt.config import Config, DispConstraint, LoadCase, Model
from oropt.deck import Deck
from oropt.loop import preview_growth_boxes, validate_group_ids
from oropt.mesh import Mesh

# Two tets + three /GRNOD/NODE groups: 60000000 (the BC set), 99999998 (a real
# exclusion set -- the id the live failure mistyped as 999999998), and 77
# (present in the deck but listing no nodes).
GUARD_DECK = """\
/NODE
  60000001   0.0   0.0   0.0
  60000002   1.0   0.0   0.0
  60000003   0.0   1.0   0.0
  60000004   0.0   0.0   1.0
  60000005   1.0   1.0   1.0
/PART/60000000
linkage
         3         3         0
/TETRA4/60000000
  60000001  60000001  60000002  60000003  60000004
  60000002  60000002  60000003  60000004  60000005
#-  PROPERTIES:
/PROP/SOLID/3
prop
/GRNOD/NODE/60000000
sym
  60000001  60000002
/GRNOD/NODE/99999998
exclude_load_introduction
  60000005
/GRNOD/NODE/77
empty_set
/END
"""


def _load(tmp_path):
    p = tmp_path / "guard_0000.rad"
    p.write_text(GUARD_DECK, encoding="utf-8")
    return Deck.load(p, design_part_id=60000000, design_node_min=60000000)


# ---- Deck.group_ids ----------------------------------------------------------
def test_group_ids_lists_every_grnod_block(tmp_path):
    deck = _load(tmp_path)
    assert deck.group_ids() == [77, 60000000, 99999998]


def test_group_nodes_still_empty_for_missing_id(tmp_path):
    deck = _load(tmp_path)                    # the silent behavior being guarded
    assert deck.group_nodes(999999998).size == 0


# ---- validate_group_ids ------------------------------------------------------
def test_clean_config_passes(tmp_path):
    deck = _load(tmp_path)
    model = Model(bc_group_id=60000000, stress_exclude_group_ids=[99999998])
    validate_group_ids(deck, model)           # no raise


def test_typo_stress_exclude_id_raises_with_closest_match(tmp_path):
    """The live failure: 999999998 configured, 99999998 in the deck."""
    deck = _load(tmp_path)
    model = Model(bc_group_id=60000000, stress_exclude_group_ids=[999999998])
    with pytest.raises(ValueError, match=r"stress_exclude_group_ids: no "
                                         r"/GRNOD/NODE/999999998 in the deck "
                                         r"\(closest in the deck: 99999998\)"):
        validate_group_ids(deck, model)


def test_message_lists_deck_groups(tmp_path):
    deck = _load(tmp_path)
    model = Model(bc_group_id=60000000, freeze_group_ids=[123456789])
    with pytest.raises(ValueError) as exc:
        validate_group_ids(deck, model)
    assert "/GRNOD/NODE groups in the deck: 77, 60000000, 99999998" in str(exc.value)


def test_bad_bc_group_id_raises(tmp_path):
    deck = _load(tmp_path)
    with pytest.raises(ValueError, match=r"bc_group_id: no /GRNOD/NODE/42"):
        validate_group_ids(deck, Model(bc_group_id=42))


def test_existing_but_empty_group_distinguished(tmp_path):
    deck = _load(tmp_path)
    model = Model(bc_group_id=60000000, freeze_group_ids=[77])
    with pytest.raises(ValueError, match=r"freeze_group_ids: /GRNOD/NODE/77 "
                                         r"exists in the deck but lists no "
                                         r"nodes") as exc:
        validate_group_ids(deck, model)
    assert "closest" not in str(exc.value)    # not the missing-id (typo) wording


def test_all_offenders_reported_together(tmp_path):
    deck = _load(tmp_path)
    model = Model(bc_group_id=60000000, freeze_group_ids=[77],
                  stress_exclude_group_ids=[999999998])
    with pytest.raises(ValueError) as exc:
        validate_group_ids(deck, model)
    msg = str(exc.value)
    assert "freeze_group_ids" in msg and "stress_exclude_group_ids" in msg


# ---- GUI preview surfacing ----------------------------------------------------
def test_preview_surfaces_group_guard_without_growth_boxes(tmp_path):
    deck = _load(tmp_path)
    mesh = Mesh.from_deck(deck)
    pv = preview_growth_boxes(deck, mesh, Model(
        bc_group_id=60000000, stress_exclude_group_ids=[999999998]))
    assert "999999998" in pv.group_guard
    assert "closest in the deck: 99999998" in pv.group_guard
    assert pv.guard == ""                     # the growth guard is independent


def test_preview_group_guard_empty_when_clean(tmp_path):
    deck = _load(tmp_path)
    mesh = Mesh.from_deck(deck)
    pv = preview_growth_boxes(deck, mesh, Model(
        bc_group_id=60000000, stress_exclude_group_ids=[99999998]))
    assert pv.group_guard == ""


# ---- run_optimization fails before any solve -----------------------------------
def test_run_start_aborts_before_solver(tmp_path, mini_deck_path,
                                        mini_engine_path, monkeypatch):
    """A typo'd group id stops run_optimization at deck load -- the solver is
    never invoked (the whole point: fail before the ~13-min solve)."""
    cfg = Config()
    cfg.model.case_dir = str(tmp_path)        # holds mini_0000/mini_0001.rad
    cfg.model.design_part_id = 60000000
    cfg.model.design_node_min = 60000000
    cfg.model.bc_group_id = 60000000          # exists in MINI_DECK
    cfg.model.stress_exclude_group_ids = [999999998]   # MINI_DECK has no such group
    cfg.load_cases = [LoadCase(
        name="only", stem="mini", sigma_allow=250.0,
        disp_constraints=[DispConstraint(node_id=60000001, d_allow=1.0)])]
    cfg.work_dir = str(tmp_path / "out")

    called: list = []
    monkeypatch.setattr(loop_mod, "run_solver",
                        lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match="stress_exclude_group_ids"):
        loop_mod.run_optimization(cfg, log=lambda *_: None)
    assert not called
