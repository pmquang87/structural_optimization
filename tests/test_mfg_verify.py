"""Independent manufacturing-constraint verifier (oropt.mfg_verify).

Hermetic: small fabricated meshes with KNOWN manufacturability, never touches
OpenRadioss. Each defect (a floating overhang past the angle, a one-element
casting undercut, a non-uniform extrusion prism, a broken symmetry, a thin
member) is built on purpose and the measured violation asserted; a clean design
passes every active check. The verifier reimplements the geometry, so these
tests are an external ground truth on the final mask -- independent of
oropt.manufacturing's enforcement.
"""
from __future__ import annotations

import numpy as np

from oropt.config import ManufacturingOpts
from oropt.mesh import Mesh
from oropt import mfg_verify as mv


# --------------------------------------------------------------------------- #
# mesh fixtures (built the way tests/test_mesh.py / test_manufacturing.py do)
# --------------------------------------------------------------------------- #
def _single_column_mesh(z_heights):
    """One vertical column of tets stacked along +z (shared footprint at x=y=0)."""
    z = np.asarray(z_heights, dtype=float)
    centroids = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    conn = np.array([[4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
                     for i in range(len(z))])
    return Mesh(centroids=centroids, volumes=np.ones(len(z)), conn_rows=conn,
                n_nodes=int(conn.max()) + 1, design_node_min=0)


def _two_column_mesh():
    """Two vertical prisms along z, footprints at x=0 and x=5, three elements each."""
    centroids = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0],   # column A (x=0)
        [5.0, 0.0, 0.0], [5.0, 0.0, 1.0], [5.0, 0.0, 2.0],   # column B (x=5)
    ])
    conn = np.array([[4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3] for i in range(6)])
    return Mesh(centroids=centroids, volumes=np.ones(6), conn_rows=conn,
                n_nodes=int(conn.max()) + 1, design_node_min=0)


def _column_with_floater_mesh():
    """A 4-tet column rising along +z on the plate plus one floating tet above."""
    centroids = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0],
        [10.0, 10.0, 100.0],                                 # floating overhang
    ])
    conn = np.array([[i, i + 1, i + 2, i + 3] for i in range(5)])
    return Mesh(centroids=centroids, volumes=np.ones(5), conn_rows=conn,
                n_nodes=int(conn.max()) + 1, design_node_min=0)


def _mirror_pairs_mesh():
    """Four tets at x = -2, -1, +1, +2 (mirror-symmetric about x = 0)."""
    centroids = np.array([[-2.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    conn = np.array([[i, i + 1, i + 2, i + 3] for i in range(4)])
    return Mesh(centroids=centroids, volumes=np.ones(4), conn_rows=conn,
                n_nodes=int(conn.max()) + 1, design_node_min=0)


def _block_and_sliver_mesh():
    """5-tet solid block (all share node 100) + a thin sliver whose only neighbour
    is a void tet (mirrors tests/test_manufacturing.py)."""
    conn = np.array([
        [100, 0, 1, 2], [100, 3, 4, 5], [100, 6, 7, 8],
        [100, 9, 10, 11], [100, 12, 13, 14],
        [200, 201, 202, 203],       # sliver: shares node 203 with the void only
        [203, 204, 205, 206],       # void neighbour of the sliver
    ])
    return Mesh(centroids=np.zeros((len(conn), 3)), volumes=np.ones(len(conn)),
                conn_rows=conn, n_nodes=int(conn.max()) + 1, design_node_min=0)


# --------------------------------------------------------------------------- #
# 1. overhang / self-support
# --------------------------------------------------------------------------- #
def test_overhang_flags_floater():
    mesh = _column_with_floater_mesh()
    alive = np.ones(5, dtype=bool)                 # floater is unsupported
    worst, n_unsup = mv.max_overhang_violation(mesh, alive, [0, 0, 1.0], 45.0)
    assert n_unsup == 1                            # only the floater
    assert worst == 90.0                           # nothing beneath it


def test_overhang_clean_column_passes():
    mesh = _column_with_floater_mesh()
    alive = np.array([True, True, True, True, False])   # drop the floater
    worst, n_unsup = mv.max_overhang_violation(mesh, alive, [0, 0, 1.0], 45.0)
    assert n_unsup == 0
    assert worst <= 45.0 + 1e-9                    # every element cone-supported


def test_overhang_shallow_angle_violates():
    """A supported-but-too-shallow diagonal: an element whose only supporter within
    reach sits laterally offset, so its overhang angle exceeds the allowed cone.
    The column (spacing 1) keeps the plate tolerance small so the offset element is
    judged above the plate."""
    col_z = np.arange(6, dtype=float)
    centroids = np.vstack([
        np.column_stack([np.zeros_like(col_z), np.zeros_like(col_z), col_z]),
        np.array([[2.5, 0.0, 5.0]]),               # offset element, dx=2.5, dz=1
    ])
    conn = np.array([[4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
                     for i in range(len(centroids))])
    mesh = Mesh(centroids=centroids, volumes=np.ones(len(centroids)),
                conn_rows=conn, n_nodes=int(conn.max()) + 1, design_node_min=0)
    alive = np.ones(len(centroids), dtype=bool)
    # only supporter in reach is (0,0,4): angle-from-vertical = atan(2.5/1) ~ 68 deg
    worst, n_unsup = mv.max_overhang_violation(mesh, alive, [0, 0, 1.0], 45.0)
    assert n_unsup == 1
    assert 65.0 < worst < 72.0


def test_overhang_protected_not_counted():
    mesh = _column_with_floater_mesh()
    alive = np.ones(5, dtype=bool)
    protected = np.zeros(5, dtype=bool)
    protected[4] = True                            # the floater is a keep-out
    _worst, n_unsup = mv.max_overhang_violation(mesh, alive, [0, 0, 1.0], 45.0,
                                                protected)
    assert n_unsup == 0


# --------------------------------------------------------------------------- #
# 2. minimum member size
# --------------------------------------------------------------------------- #
def test_min_member_detects_sliver():
    mesh = _block_and_sliver_mesh()
    alive = np.array([True, True, True, True, True, True, False])  # block + sliver
    # the sliver does not survive a single-hop open -> thinnest member 0
    assert mv.min_member_thickness(mesh, alive) == 0


def test_min_member_solid_block_is_thick():
    mesh = _block_and_sliver_mesh()
    alive = np.array([True, True, True, True, True, False, False])  # block only
    # a solid mutually-adjacent block has no thin member -> well above 1
    assert mv.min_member_thickness(mesh, alive) >= 1


# --------------------------------------------------------------------------- #
# bonus: maximum member size
# --------------------------------------------------------------------------- #
def _chain_mesh(n):
    """A chain of tets where element i shares one node with element i+1."""
    conn = np.array([[3 * i, 3 * i + 1, 3 * i + 2, 3 * i + 3] for i in range(n)])
    centroids = np.column_stack([np.arange(n, dtype=float),
                                 np.zeros(n), np.zeros(n)])
    return Mesh(centroids=centroids, volumes=np.ones(n), conn_rows=conn,
                n_nodes=int(conn.max()) + 1, design_node_min=0)


def test_max_member_distance_to_void():
    mesh = _chain_mesh(4)
    alive = np.array([True, True, True, False])    # the end element is the void
    # deepest alive element (index 0) is 3 hops from the void (chain 0-1-2-3)
    assert mv.max_member_thickness(mesh, alive) == 3


# --------------------------------------------------------------------------- #
# 3. casting / draw direction
# --------------------------------------------------------------------------- #
def test_draw_single_sided_undercut():
    mesh = _single_column_mesh([0.0, 1.0, 2.0, 3.0, 4.0])
    alive = np.array([True, True, True, False, True])   # a shelf floats above void
    assert mv.draw_undercut_violation(mesh, alive, [0, 0, 1.0], two_sided=False) == 1


def test_draw_single_sided_clean():
    mesh = _single_column_mesh([0.0, 1.0, 2.0, 3.0, 4.0])
    alive = np.array([True, True, True, False, False])  # solid is a bottom prefix
    assert mv.draw_undercut_violation(mesh, alive, [0, 0, 1.0], two_sided=False) == 0


def test_draw_two_sided_two_runs_is_undercut():
    mesh = _single_column_mesh([0.0, 1.0, 2.0, 3.0, 4.0])
    alive = np.array([True, False, True, True, True])   # a lone base + a 3-run
    assert mv.draw_undercut_violation(mesh, alive, [0, 0, 1.0], two_sided=True) == 1


def test_draw_two_sided_single_run_clean():
    mesh = _single_column_mesh([0.0, 1.0, 2.0, 3.0, 4.0])
    alive = np.array([False, False, True, True, True])  # one contiguous band
    assert mv.draw_undercut_violation(mesh, alive, [0, 0, 1.0], two_sided=True) == 0


# --------------------------------------------------------------------------- #
# 4. extrusion
# --------------------------------------------------------------------------- #
def test_extrusion_nonuniform_prism():
    mesh = _two_column_mesh()
    alive = np.array([True, True, True, True, False, False])  # B mixes alive/void
    n_bad, frac = mv.extrusion_nonuniformity(mesh, alive, [0, 0, 1.0])
    assert n_bad == 1
    assert abs(frac - 0.5) < 1e-9


def test_extrusion_uniform_passes():
    mesh = _two_column_mesh()
    alive = np.array([True, True, True, False, False, False])  # both prisms uniform
    n_bad, frac = mv.extrusion_nonuniformity(mesh, alive, [0, 0, 1.0])
    assert n_bad == 0
    assert frac == 0.0


# --------------------------------------------------------------------------- #
# 5. symmetry
# --------------------------------------------------------------------------- #
def test_symmetry_residual_broken():
    mesh = _mirror_pairs_mesh()
    alive = np.array([True, False, False, False])   # -2 alive, its +2 mirror dead
    plane = {"axis": "x", "offset": 0.0}
    # pairs (0,3) and (1,2): one mismatched pair -> 2 of 4 paired elements differ
    assert abs(mv.symmetry_residual(mesh, alive, plane) - 0.5) < 1e-9


def test_symmetry_residual_clean():
    mesh = _mirror_pairs_mesh()
    alive = np.array([True, False, False, True])    # symmetric about x = 0
    plane = {"axis": "x", "offset": 0.0}
    assert mv.symmetry_residual(mesh, alive, plane) == 0.0


# --------------------------------------------------------------------------- #
# verify() -- report assembly
# --------------------------------------------------------------------------- #
def test_verify_flags_overhang():
    mesh = _column_with_floater_mesh()
    alive = np.ones(5, dtype=bool)
    opts = ManufacturingOpts(build_direction=[0, 0, 1.0], max_overhang_angle=45.0)
    report = mv.verify(mesh, alive, opts)
    assert not report.ok
    assert [c.name for c in report.failed()] == ["overhang"]
    assert report.warnings and "overhang" in report.warnings[0]


def test_verify_flags_draw_undercut():
    mesh = _single_column_mesh([0.0, 1.0, 2.0, 3.0, 4.0])
    alive = np.array([True, True, True, False, True])
    opts = ManufacturingOpts(draw_direction=[0, 0, 1.0])
    report = mv.verify(mesh, alive, opts)
    assert not report.ok
    assert [c.name for c in report.failed()] == ["draw"]


def test_verify_flags_extrusion():
    mesh = _two_column_mesh()
    alive = np.array([True, True, True, True, False, False])
    opts = ManufacturingOpts(extrusion_axis=[0, 0, 1.0])
    report = mv.verify(mesh, alive, opts)
    assert not report.ok
    assert [c.name for c in report.failed()] == ["extrusion"]


def test_verify_flags_symmetry():
    mesh = _mirror_pairs_mesh()
    alive = np.array([True, False, False, False])
    opts = ManufacturingOpts(symmetry_planes=[{"axis": "x", "offset": 0.0}])
    report = mv.verify(mesh, alive, opts)
    assert not report.ok
    assert report.failed()[0].name.startswith("symmetry")


def test_verify_accepts_plain_dict():
    mesh = _two_column_mesh()
    alive = np.array([True, True, True, True, False, False])
    report = mv.verify(mesh, alive, {"extrusion_axis": [0, 0, 1.0]})
    assert not report.ok
    assert [c.name for c in report.failed()] == ["extrusion"]


def test_verify_inactive_when_all_off():
    mesh = _two_column_mesh()
    alive = np.array([True, True, False, True, False, False])
    report = mv.verify(mesh, alive, ManufacturingOpts())   # every field off
    assert report.ok
    assert report.checks == []
    assert report.warnings == []


def test_verify_clean_design_passes_all_active_checks():
    """A tidy vertical column: on-plate & cone-supported (overhang), a bottom
    prefix (draw), a uniform prism (extrusion), self-mirroring about x=0
    (symmetry). Every active check passes."""
    mesh = _single_column_mesh([0.0, 1.0, 2.0, 3.0])
    alive = np.ones(4, dtype=bool)
    opts = ManufacturingOpts(
        build_direction=[0, 0, 1.0], max_overhang_angle=45.0,
        draw_direction=[0, 0, 1.0],
        extrusion_axis=[0, 0, 1.0],
        symmetry_planes=[{"axis": "x", "offset": 0.0}],
    )
    report = mv.verify(mesh, alive, opts)
    assert report.ok
    assert report.warnings == []
    assert {c.name for c in report.checks} == {
        "overhang", "draw", "extrusion", "symmetry[x@0]"}
    assert all(c.passed for c in report.checks)
