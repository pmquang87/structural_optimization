"""Surface smoothing of the final geometry: guards + real STL/VTP output."""
from __future__ import annotations

import numpy as np

from oropt import status as st
from oropt.config import Config
from oropt.smoothing import _formats, smooth_final


def _write_cube_topology(work):
    """Write a small topology_latest.vtu: a unit cube split into 5 tetrahedra."""
    node_xyz = np.array([
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
    conn = np.array([
        [0, 1, 3, 4], [1, 2, 3, 6], [1, 3, 4, 6], [1, 4, 5, 6], [3, 4, 6, 7]])
    st.write_topology(work, node_xyz, conn, np.ones(conn.shape[0], dtype=bool))


def test_formats_mapping():
    assert _formats("stl") == ["stl"]
    assert _formats("vtp") == ["vtp"]
    assert _formats("both") == ["stl", "vtp"]
    assert _formats("garbage") == ["stl"]          # safe fallback


def test_smooth_final_disabled_is_noop(tmp_path):
    cfg = Config()
    cfg.smooth.enabled = False
    assert smooth_final(cfg, tmp_path, lambda *_: None) is None


def test_smooth_final_no_topology_returns_none(tmp_path):
    cfg = Config()
    cfg.smooth.enabled = True
    logs: list[str] = []
    assert smooth_final(cfg, tmp_path, logs.append) is None
    assert any("no topology" in m.lower() for m in logs)


def test_smooth_final_writes_stl(tmp_path):
    _write_cube_topology(tmp_path)
    cfg = Config()
    cfg.smooth.enabled = True
    cfg.smooth.iterations = 5
    cfg.smooth.output_format = "stl"
    out = smooth_final(cfg, tmp_path, lambda *_: None)
    assert out is not None and out.suffix == ".stl" and out.is_file()


def test_smooth_final_both_formats_laplacian(tmp_path):
    _write_cube_topology(tmp_path)
    cfg = Config()
    cfg.smooth.enabled = True
    cfg.smooth.iterations = 3
    cfg.smooth.method = "laplacian"
    cfg.smooth.output_format = "both"
    smooth_final(cfg, tmp_path, lambda *_: None)
    assert (tmp_path / "topology_smoothed.stl").is_file()
    assert (tmp_path / "topology_smoothed.vtp").is_file()
