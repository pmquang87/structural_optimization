"""protect_bc_nodes: frozen-set selection + connectivity-anchor decoupling.

The option lets the optimiser delete elements touching the BC node-group while
the BC/load region still anchors connectivity (so islands are still dropped).
"""
from __future__ import annotations

import numpy as np

from oropt.beso import Beso
from oropt.config import Beso as BesoCfg
from oropt.loop import collect_protect_nodes
from oropt.mesh import Mesh


class _FakeDeck:
    """Duck-typed deck: group_nodes(id) -> node ids declared in that group."""
    def __init__(self, groups):
        self._g = {int(k): np.asarray(v, dtype=np.int64) for k, v in groups.items()}

    def group_nodes(self, gid):
        return self._g.get(int(gid), np.empty(0, dtype=np.int64))


class _FakeModel:
    def __init__(self, bc_group_id, freeze_group_ids=(), freeze_node_ids=()):
        self.bc_group_id = bc_group_id
        self.freeze_group_ids = list(freeze_group_ids)
        self.freeze_node_ids = list(freeze_node_ids)


def test_collect_protect_nodes_includes_bc_by_default():
    deck = _FakeDeck({100: [1, 2, 3], 999: [7, 8]})
    model = _FakeModel(bc_group_id=100, freeze_group_ids=[999])
    nodes = collect_protect_nodes(deck, model)            # include_bc default True
    assert set(nodes.tolist()) == {1, 2, 3, 7, 8}


def test_collect_protect_nodes_can_exclude_bc():
    deck = _FakeDeck({100: [1, 2, 3], 999: [7, 8]})
    model = _FakeModel(bc_group_id=100, freeze_group_ids=[999])
    nodes = collect_protect_nodes(deck, model, include_bc=False)
    assert set(nodes.tolist()) == {7, 8}                  # BC group dropped


def _two_elem_mesh():
    return Mesh(centroids=np.array([[0., 0, 0], [10., 0, 0]]),
                volumes=np.array([1.0, 1.0]),
                conn_rows=np.array([[0, 1, 2, 3], [4, 5, 6, 7]]),
                n_nodes=8, design_node_min=0)


def test_beso_anchor_defaults_to_protected():
    beso = Beso(_two_elem_mesh(), BesoCfg(), np.array([True, False]))
    assert np.array_equal(beso.anchor, [True, False])


def test_beso_anchor_decoupled_when_provided():
    protected = np.array([False, False])     # BC deletable -> nothing frozen
    anchor = np.array([True, False])         # BC element still anchors connectivity
    beso = Beso(_two_elem_mesh(), BesoCfg(), protected, anchor=anchor)
    assert np.array_equal(beso.protected, protected)
    assert np.array_equal(beso.anchor, anchor)
