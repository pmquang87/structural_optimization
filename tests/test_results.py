"""parse_vtk: filter design part + active elements, read fields and displacement."""
import numpy as np

from oropt.results import parse_vtk


def _make_vtk(path, erosion=(1, 1, 1)):
    """Synthetic anim_to_vtk-style file: 2 design tets + 1 rigid triangle."""
    import pyvista as pv
    points = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1],
                       [1, 1, 1], [2, 2, 2]], dtype=float)
    cells = np.array([4, 0, 1, 2, 3, 4, 1, 2, 3, 4, 3, 0, 1, 5])
    ctypes = np.array([pv.CellType.TETRA, pv.CellType.TETRA,
                       pv.CellType.TRIANGLE], np.uint8)
    g = pv.UnstructuredGrid(cells, ctypes, points)
    g.cell_data["ELEMENT_ID"] = np.array([60000001, 60000002, 10000001])
    g.cell_data["PART_ID"] = np.array([60000000, 60000000, 10000000])
    g.cell_data["EROSION_STATUS"] = np.array(erosion)
    g.cell_data["3DELEM_Specific_Energy"] = np.array([11.0, 22.0, 0.0])
    g.cell_data["3DELEM_Von_Mises"] = np.array([100.0, 250.0, 0.0])
    g.point_data["NODE_ID"] = np.array([1, 2, 3, 4, 5, 6])
    disp = np.zeros((6, 3)); disp[4] = [0.3, 0.4, 0.0]      # NODE_ID 5 -> |d|=0.5
    g.point_data["Displacement"] = disp
    g.save(str(path))


def test_parse_design_fields(tmp_path):
    vtk = tmp_path / "a.vtk"
    _make_vtk(vtk)
    r = parse_vtk(vtk, design_part_id=60000000, disp_node_id=5)
    assert r.element_ids.tolist() == [60000001, 60000002]    # rigid triangle excluded
    assert r.energy.tolist() == [11.0, 22.0]
    assert r.sigma_max == 250.0
    assert np.isclose(r.disp, 0.5)


def test_erosion_excluded(tmp_path):
    vtk = tmp_path / "b.vtk"
    _make_vtk(vtk, erosion=(1, 0, 1))                        # 2nd tet eroded
    r = parse_vtk(vtk, design_part_id=60000000, disp_node_id=5)
    assert r.element_ids.tolist() == [60000001]
    assert r.sigma_max == 100.0
