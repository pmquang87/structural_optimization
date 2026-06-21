"""Shared fixtures: a tiny synthetic starter deck and mesh for fast unit tests."""
from __future__ import annotations

from pathlib import Path

import pytest

# 5 design nodes (two tets sharing a face) + 1 rigid node; node 60000001 sits in
# /GRNOD (protected), node 60000005 belongs only to the 2nd tet (becomes free if
# that tet is deleted -> must be pinned).
MINI_DECK = """\
#- header
/MAT/LAW1/3
mat
#- nodes follow
/NODE
#  Node ID               X               Y               Z
  60000001                 0.0                 0.0                 0.0
  60000002                 1.0                 0.0                 0.0
  60000003                 0.0                 1.0                 0.0
  60000004                 0.0                 0.0                 1.0
  60000005                 1.0                 1.0                 1.0
  10000001                 5.0                 5.0                 5.0
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
sym_set
  60000001  60000002
/BCS/90009
bc
#  Tra rot   skew_ID  grnod_ID
   111 111         0     60000000
/END
"""


@pytest.fixture
def mini_deck_path(tmp_path: Path) -> Path:
    p = tmp_path / "mini_0000.rad"
    p.write_text(MINI_DECK, encoding="utf-8")
    return p


@pytest.fixture
def mini_engine_path(tmp_path: Path) -> Path:
    p = tmp_path / "mini_0001.rad"
    p.write_text("/RUN/x/1\n1\n/ANIM/DT\n0. 0.1\n/IMPL/NONLIN/1\n", encoding="utf-8")
    return p
