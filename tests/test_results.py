"""parse_vtk: filter design part + active elements, read fields and displacement."""
import numpy as np

from oropt.results import parse_vtk


#: per-cell 6-component stress tensor used by the tensor-parsing tests
#: (rows: design tet 1, design tet 2, rigid triangle)
_STRESS_3 = np.array([[10.0, 1.0, 2.0, 3.0, 4.0, 5.0],
                      [20.0, 6.0, 7.0, 8.0, 9.0, 1.5],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])


def _make_vtk(path, erosion=(1, 1, 1), stress_name=None):
    """Synthetic anim_to_vtk-style file: 2 design tets + 1 rigid triangle.

    *stress_name* (optional) adds ``_STRESS_3`` as a 6-component cell array
    under that name — the exact name the real converter emits for
    /ANIM/BRICK/TENS/STRESS is unverified, so tests exercise several.
    """
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
    if stress_name is not None:
        g.cell_data[stress_name] = _STRESS_3.copy()
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


def test_parse_stress_tensor_absent_is_none(tmp_path):
    vtk = tmp_path / "s0.vtk"
    _make_vtk(vtk)                                           # no tensor array
    r = parse_vtk(vtk, design_part_id=60000000)
    assert r.stress is None


def test_parse_stress_tensor_candidate_name(tmp_path):
    """A 6-component cell array under a known candidate name lands in
    Results.stress, aligned with element_ids and part-filtered."""
    vtk = tmp_path / "s1.vtk"
    _make_vtk(vtk, stress_name="3DELEM_Stress_tensor")
    r = parse_vtk(vtk, design_part_id=60000000)
    assert r.element_ids.tolist() == [60000001, 60000002]
    assert r.stress is not None and r.stress.shape == (2, 6)
    assert np.allclose(r.stress, _STRESS_3[:2])              # rigid triangle dropped


def test_parse_stress_tensor_fallback_name_scan(tmp_path):
    """An unanticipated converter spelling still matches via the case-insensitive
    'tens'/'stress' 6-component scan."""
    vtk = tmp_path / "s2.vtk"
    _make_vtk(vtk, stress_name="3DELEM_TENS/STRESS_weird")
    r = parse_vtk(vtk, design_part_id=60000000)
    assert r.stress is not None
    assert np.allclose(r.stress, _STRESS_3[:2])


def test_parse_stress_tensor_erosion_filtered(tmp_path):
    """The tensor gets the same keep-filtering as energy/vonmises."""
    vtk = tmp_path / "s3.vtk"
    _make_vtk(vtk, erosion=(1, 0, 1), stress_name="3DELEM_Stress_tensor")
    r = parse_vtk(vtk, design_part_id=60000000)
    assert r.element_ids.tolist() == [60000001]
    assert r.stress.shape == (1, 6)
    assert np.allclose(r.stress[0], _STRESS_3[0])


def test_parse_multiple_disp_nodes(tmp_path):
    vtk = tmp_path / "c.vtk"
    _make_vtk(vtk)
    r = parse_vtk(vtk, design_part_id=60000000, disp_node_ids=[5, 2, 999])
    assert np.isclose(r.disps[5], 0.5)       # NODE_ID 5 -> |d| = 0.5
    assert r.disps[2] == 0.0                  # NODE_ID 2 -> no displacement
    assert np.isnan(r.disps[999])             # node absent from the animation
    # the scalar convenience fields track the FIRST requested node
    assert r.disp_node_id == 5 and np.isclose(r.disp, 0.5)


def test_parse_no_disp_nodes_leaves_disps_empty(tmp_path):
    vtk = tmp_path / "d.vtk"
    _make_vtk(vtk)
    r = parse_vtk(vtk, design_part_id=60000000)   # no node requested
    assert r.disps == {} and r.disp_node_id is None
    assert np.isnan(r.disp)


def test_run_anim_to_vtk_accepts_small_valid_output(tmp_path, monkeypatch):
    """A clean conversion of a SMALL validation mesh is well under the old
    1024-byte floor and was reported as 'anim_to_vtk failed'. Success is now a
    clean exit + a VTK header, size-independent."""
    import subprocess
    from oropt.config import Config
    from oropt.results import run_anim_to_vtk

    exe = tmp_path / "or" / "exec" / "anim_to_vtk_win64.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("x", encoding="utf-8")
    cfg = Config()
    cfg.or_paths.root = str(tmp_path / "or")

    def fake_run(cmd, stdout=None, stderr=None, env=None, check=False):
        stdout.write("# vtk DataFile Version 2.0\ntiny but valid\n")
        return subprocess.CompletedProcess(cmd, 0, stderr=b"")
    monkeypatch.setattr(subprocess, "run", fake_run)

    out = tmp_path / "out.vtk"
    got = run_anim_to_vtk(cfg, tmp_path / "mA001", out)
    assert got == out and out.stat().st_size < 1024      # small yet accepted


def test_run_anim_to_vtk_rejects_non_vtk_output(tmp_path, monkeypatch):
    import subprocess

    import pytest

    from oropt.config import Config
    from oropt.results import run_anim_to_vtk

    exe = tmp_path / "or" / "exec" / "anim_to_vtk_win64.exe"
    exe.parent.mkdir(parents=True)
    exe.write_text("x", encoding="utf-8")
    cfg = Config()
    cfg.or_paths.root = str(tmp_path / "or")

    def fake_run(cmd, stdout=None, stderr=None, env=None, check=False):
        stdout.write("ERROR: could not read animation\n")
        return subprocess.CompletedProcess(cmd, 0, stderr=b"bad anim")
    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="anim_to_vtk failed"):
        run_anim_to_vtk(cfg, tmp_path / "mA001", tmp_path / "out.vtk")
