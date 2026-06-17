"""Status/history/checkpoint round-trips and topology export."""
import os
import subprocess
import sys

import numpy as np

from oropt import status as st


def test_status_roundtrip(tmp_path):
    s = st.Status(state="running", iteration=3, volume_fraction=0.72,
                  sigma_max=310.5, disp=1.4, feasible=False)
    st.write_status(tmp_path, s)
    r = st.read_status(tmp_path)
    assert r is not None
    assert r.state == "running" and r.iteration == 3
    assert abs(r.volume_fraction - 0.72) < 1e-9 and r.feasible is False
    assert r.updated                      # timestamp stamped on write


def test_history_append(tmp_path):
    for it in range(3):
        st.append_history(tmp_path, {"iteration": it, "volume_fraction": 1 - 0.1 * it,
                                     "sigma_max": 300 + it, "disp": 1.2,
                                     "elements_alive": 100 - it, "feasible": True,
                                     "iter_wall_s": 12.0, "or_termination": "ok"})
    rows = st.read_history(tmp_path)
    assert len(rows) == 3 and rows[0]["iteration"] == "0"
    assert rows[2]["sigma_max"] == "302"


def test_checkpoint_roundtrip(tmp_path):
    alive = np.array([True, False, True, True])
    sens = np.array([1.0, 2, 3, 4])
    st.save_checkpoint(tmp_path, 7, alive, sens)
    c = st.load_checkpoint(tmp_path)
    assert c["iteration"] == 7
    assert np.array_equal(c["alive_mask"], alive)
    assert np.allclose(c["sens_prev"], sens)


def test_write_topology(tmp_path):
    node_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                        dtype=float)
    conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    alive = np.array([True, False])
    st.write_topology(tmp_path, node_xyz, conn, alive, fields={"sens": np.array([9.0, 8.0])})
    import pyvista as pv
    grid = pv.read(str(tmp_path / st.TOPOLOGY))
    assert grid.n_cells == 1                       # only the alive tet
    assert np.isclose(grid.cell_data["sens"][0], 9.0)


def test_topology_snapshot_name():
    assert st.topology_snapshot_name(0) == "topology_iter0000.vtu"
    assert st.topology_snapshot_name(7) == "topology_iter0007.vtu"
    assert st.topology_snapshot_name(1234) == "topology_iter1234.vtu"


def test_write_topology_snapshot(tmp_path):
    node_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 1]],
                        dtype=float)
    conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    alive = np.array([True, True])
    st.write_topology(tmp_path, node_xyz, conn, alive,
                      fields={"sens": np.array([9.0, 8.0])}, iteration=5)
    import pyvista as pv
    # the latest file is always (over)written ...
    latest = pv.read(str(tmp_path / st.TOPOLOGY))
    assert latest.n_cells == 2
    # ... and an immutable per-iteration snapshot is kept alongside it
    snap_path = tmp_path / "topology_iter0005.vtu"
    assert snap_path.exists()
    snap = pv.read(str(snap_path))
    assert snap.n_cells == 2
    assert np.isclose(snap.cell_data["sens"][0], 9.0)


def test_write_topology_no_snapshot_by_default(tmp_path):
    node_xyz = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    conn = np.array([[0, 1, 2, 3]])
    alive = np.array([True])
    st.write_topology(tmp_path, node_xyz, conn, alive)   # no iteration -> latest only
    assert (tmp_path / st.TOPOLOGY).exists()
    assert not list(tmp_path.glob("topology_iter*.vtu"))


def test_pid_alive():
    assert st.pid_alive(os.getpid())
    assert not st.pid_alive(2_000_000_000)
    assert not st.pid_alive(None)


def test_pid_alive_exited_child():
    # subprocess keeps the child's process handle open after wait(), which on
    # Windows keeps the kernel object around — the lingering-handle case that
    # made pid_alive() report an exited run as still running.
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    assert not st.pid_alive(child.pid)
