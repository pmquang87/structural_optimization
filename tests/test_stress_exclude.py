"""Stress-exclusion region: a node set whose design elements have their
von-Mises ignored — dropped from sigma_max / feasibility / Monitor / report.

Covers the three seams: collecting the excluded nodes from the config, mapping
them to the touching design elements, and dropping those elements from the peak
stress in :func:`oropt.results.parse_vtk`.
"""
from __future__ import annotations

import numpy as np

from oropt.loop import collect_stress_exclude_nodes, stress_exclude_mask
from oropt.mesh import Mesh
from oropt.results import parse_vtk


class _FakeDeck:
    """Duck-typed deck for the node/element collectors."""
    def __init__(self, groups, node_ids, node_xyz, elem_ids):
        self._g = {int(k): np.asarray(v, dtype=np.int64) for k, v in groups.items()}
        self.node_ids = np.asarray(node_ids, dtype=np.int64)
        self.node_xyz = np.asarray(node_xyz, dtype=float)
        self.elem_ids = np.asarray(elem_ids, dtype=np.int64)

    @property
    def n_design_elements(self):
        return int(self.elem_ids.size)

    def group_nodes(self, gid):
        return self._g.get(int(gid), np.empty(0, dtype=np.int64))


class _FakeModel:
    def __init__(self, group_ids=(), node_ids=()):
        self.stress_exclude_group_ids = list(group_ids)
        self.stress_exclude_node_ids = list(node_ids)


def _two_tet_setup(model):
    """Two tets sharing nodes 2,3,4; node 5 belongs only to the second tet."""
    deck = _FakeDeck(
        groups={999999998: [5]},
        node_ids=[1, 2, 3, 4, 5],
        node_xyz=np.zeros((5, 3)),
        elem_ids=[60000001, 60000002])
    mesh = Mesh(centroids=np.zeros((2, 3)), volumes=np.ones(2),
                conn_rows=np.array([[0, 1, 2, 3], [1, 2, 3, 4]]),
                n_nodes=5, design_node_min=0)
    return deck, mesh


def test_collect_nodes_from_group_and_explicit():
    deck, _ = _two_tet_setup(None)
    model = _FakeModel(group_ids=[999999998], node_ids=[2])
    nodes = collect_stress_exclude_nodes(deck, model)
    assert set(nodes.tolist()) == {2, 5}          # group 999999998 -> {5}, plus explicit 2


def test_collect_nodes_empty_by_default():
    deck, _ = _two_tet_setup(None)
    nodes = collect_stress_exclude_nodes(deck, _FakeModel())
    assert nodes.size == 0


def test_mask_selects_touching_elements():
    model = _FakeModel(group_ids=[999999998])     # group -> node 5 -> only 2nd tet
    deck, mesh = _two_tet_setup(model)
    mask = stress_exclude_mask(deck, mesh, model)
    assert mask.tolist() == [False, True]


def test_mask_all_false_when_unconfigured():
    deck, mesh = _two_tet_setup(None)
    mask = stress_exclude_mask(deck, mesh, _FakeModel())
    assert mask.tolist() == [False, False]


def _make_vtk(path):
    """Synthetic anim_to_vtk-style file: 2 design tets (vm 100, 250) + rigid tri."""
    import pyvista as pv
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                       [1, 1, 1], [2, 2, 2]], dtype=float)
    cells = np.array([4, 0, 1, 2, 3, 4, 1, 2, 3, 4, 3, 0, 1, 5])
    ctypes = np.array([pv.CellType.TETRA, pv.CellType.TETRA,
                       pv.CellType.TRIANGLE], np.uint8)
    g = pv.UnstructuredGrid(cells, ctypes, points)
    g.cell_data["ELEMENT_ID"] = np.array([60000001, 60000002, 10000001])
    g.cell_data["PART_ID"] = np.array([60000000, 60000000, 10000000])
    g.cell_data["EROSION_STATUS"] = np.array([1, 1, 1])
    g.cell_data["3DELEM_Specific_Energy"] = np.array([11.0, 22.0, 0.0])
    g.cell_data["3DELEM_Von_Mises"] = np.array([100.0, 250.0, 0.0])
    g.point_data["NODE_ID"] = np.array([1, 2, 3, 4, 5, 6])
    g.point_data["Displacement"] = np.zeros((6, 3))
    g.save(str(path))


def test_parse_vtk_excludes_peak_stress(tmp_path):
    vtk = tmp_path / "a.vtk"
    _make_vtk(vtk)
    # The 250-MPa element (60000002) is the hot-spot we ignore.
    r = parse_vtk(vtk, design_part_id=60000000, disp_node_id=5,
                  exclude_element_ids=np.array([60000002]))
    assert r.sigma_max == 100.0                   # peak drops to the other element
    # the per-element arrays stay full so the sensitivity still sees every element
    assert r.element_ids.tolist() == [60000001, 60000002]
    assert r.vonmises.tolist() == [100.0, 250.0]


def test_parse_vtk_no_exclusion_unchanged(tmp_path):
    vtk = tmp_path / "b.vtk"
    _make_vtk(vtk)
    r = parse_vtk(vtk, design_part_id=60000000, disp_node_id=5)
    assert r.sigma_max == 250.0                   # byte-identical to before the feature
