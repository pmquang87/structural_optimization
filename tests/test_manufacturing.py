"""Additive-manufacturing printability constraints (oropt.manufacturing).

Hermetic: synthetic meshes only, never touches OpenRadioss. Covers symmetry,
minimum member size (morphological open), overhang/self-support, the
all-constraints-off no-op, and the config roundtrip.
"""
from __future__ import annotations

import numpy as np

from oropt.config import Config, ManufacturingOpts
from oropt.manufacturing import apply_manufacturing, manufacturing_active
from oropt.mesh import Mesh


# ---- minimum member size (morphological open) ---------------------------------
def _block_and_sliver_mesh():
    """A solid *block* of 5 tets that all share a common node (mutually adjacent),
    plus a thin alive sliver tet whose only topological neighbour is a void tet.

    Element layout (alive in the test): [block0..block4, sliver, void].
    The block has interior support (every member's neighbours are alive) so it
    survives an open; the sliver has a void neighbour so erosion removes it and
    nothing regrows it.
    """
    conn = [
        [100, 0, 1, 2],      # block elements 0..4 all contain shared node 100
        [100, 3, 4, 5],
        [100, 6, 7, 8],
        [100, 9, 10, 11],
        [100, 12, 13, 14],
        [200, 201, 202, 203],   # sliver: shares node 203 with the void tet only
        [203, 204, 205, 206],   # void neighbour of the sliver
    ]
    conn = np.array(conn)
    return Mesh(centroids=np.zeros((len(conn), 3)), volumes=np.ones(len(conn)),
                conn_rows=conn, n_nodes=int(conn.max()) + 1, design_node_min=0)


def test_min_member_open_removes_sliver_keeps_block():
    mesh = _block_and_sliver_mesh()
    alive = np.array([True, True, True, True, True, True, False])  # block + sliver
    protected = np.zeros(7, dtype=bool)
    opts = ManufacturingOpts(min_member_layers=1)

    out = apply_manufacturing(alive, mesh, opts, protected)

    assert out[:5].all()        # thick block survives the open
    assert not out[5]           # isolated thin sliver is removed
    assert not out[6]           # void stays void (open never adds material)


def test_min_member_open_keeps_protected_sliver():
    """A thin element that is protected must survive the open regardless."""
    mesh = _block_and_sliver_mesh()
    alive = np.array([True, True, True, True, True, True, False])
    protected = np.zeros(7, dtype=bool); protected[5] = True   # protect the sliver
    opts = ManufacturingOpts(min_member_layers=1)

    out = apply_manufacturing(alive, mesh, opts, protected)

    assert out[5]               # protected sliver kept despite being thin
    assert out[:5].all()


# ---- symmetry -----------------------------------------------------------------
def _mirror_pairs_mesh():
    """Four tets at x = -2, -1, +1, +2 (mirror-symmetric about the x = 0 plane)."""
    centroids = np.array([[-2.0, 0, 0], [-1.0, 0, 0], [1.0, 0, 0], [2.0, 0, 0]])
    conn = np.array([[i, i + 1, i + 2, i + 3] for i in range(4)])
    return Mesh(centroids=centroids, volumes=np.ones(4), conn_rows=conn,
                n_nodes=int(conn.max()) + 1, design_node_min=0)


def test_symmetry_makes_mask_mirror_symmetric():
    mesh = _mirror_pairs_mesh()
    alive = np.array([True, False, False, False])      # deliberately asymmetric
    opts = ManufacturingOpts(symmetry_planes=[{"axis": "x", "offset": 0.0}])

    out = apply_manufacturing(alive, mesh, opts)

    # "either alive => both alive": element 0 (x=-2) pulls its mirror 3 (x=+2) on
    assert out.tolist() == [True, False, False, True]
    # the pair (1, 2) was both dead -> stays dead (no over-removal, no over-add)
    assert out[1] == out[2] == False


def test_symmetry_offset_plane():
    """Symmetry about a non-zero plane (x = 1.5) pairs x=1 with x=2, x=-1 with x=4."""
    centroids = np.array([[1.0, 0, 0], [2.0, 0, 0]])   # mirror partners about x=1.5
    conn = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    mesh = Mesh(centroids=centroids, volumes=np.ones(2), conn_rows=conn,
                n_nodes=5, design_node_min=0)
    alive = np.array([False, True])
    opts = ManufacturingOpts(symmetry_planes=[{"axis": "x", "offset": 1.5}])

    out = apply_manufacturing(alive, mesh, opts)

    assert out.tolist() == [True, True]                # symmetric about x = 1.5


# ---- overhang / self-support --------------------------------------------------
def _column_with_floater_mesh():
    """A 4-tet column rising along +z (rests on the plate) plus one tet floating
    far above with nothing beneath it."""
    centroids = np.array([
        [0.0, 0.0, 0.0],   # 0 column base (on the plate)
        [0.0, 0.0, 1.0],   # 1
        [0.0, 0.0, 2.0],   # 2
        [0.0, 0.0, 3.0],   # 3 column top (supported via the cone, not the plate)
        [10.0, 10.0, 100.0],  # 4 floating overhang, unsupported
    ])
    conn = np.array([[i, i + 1, i + 2, i + 3] for i in range(5)])
    return Mesh(centroids=centroids, volumes=np.ones(5), conn_rows=conn,
                n_nodes=int(conn.max()) + 1, design_node_min=0)


def test_overhang_forbids_floating_keeps_supported():
    mesh = _column_with_floater_mesh()
    alive = np.ones(5, dtype=bool)
    opts = ManufacturingOpts(build_direction=[0.0, 0.0, 1.0], max_overhang_angle=45.0)

    out = apply_manufacturing(alive, mesh, opts)

    assert not out[4]           # floating element with nothing beneath -> forbidden
    assert out[0]               # base rests on the build plate -> kept
    assert out[3]               # column top is cone-supported by the layer below
    assert out[:4].all()        # the whole self-supporting column survives


def test_overhang_off_when_no_build_direction():
    mesh = _column_with_floater_mesh()
    alive = np.ones(5, dtype=bool)
    # angle set but no build direction -> overhang inactive, nothing removed
    opts = ManufacturingOpts(build_direction=None, max_overhang_angle=45.0)

    out = apply_manufacturing(alive, mesh, opts)

    assert out.all()


# ---- all constraints off ------------------------------------------------------
def test_all_constraints_off_is_identity():
    mesh = _column_with_floater_mesh()
    alive = np.array([True, False, True, False, True])
    opts = ManufacturingOpts()                          # every field default/off

    assert not manufacturing_active(opts)
    out = apply_manufacturing(alive, mesh, opts)
    assert np.array_equal(out, alive)
    assert out is not alive                              # returns a copy, not the input


def test_default_opts_inactive_and_passthrough():
    assert manufacturing_active(None) is False
    assert manufacturing_active(ManufacturingOpts()) is False
    assert manufacturing_active(ManufacturingOpts(min_member_layers=2)) is True
    assert manufacturing_active(
        ManufacturingOpts(symmetry_planes=[{"axis": "y", "offset": 0.0}])) is True
    assert manufacturing_active(
        ManufacturingOpts(build_direction=[0, 0, 1], max_overhang_angle=40.0)) is True
    # build direction without a positive angle is still off
    assert manufacturing_active(
        ManufacturingOpts(build_direction=[0, 0, 1], max_overhang_angle=0.0)) is False


# ---- config roundtrip ---------------------------------------------------------
def test_manufacturing_config_defaults_and_roundtrip(tmp_path):
    cfg = Config()
    assert cfg.manufacturing.min_member_layers == 0
    assert cfg.manufacturing.symmetry_planes == []
    assert cfg.manufacturing.build_direction is None
    assert cfg.manufacturing.max_overhang_angle == 0.0

    cfg.manufacturing.min_member_layers = 2
    cfg.manufacturing.symmetry_planes = [{"axis": "x", "offset": 0.0},
                                         {"axis": "z", "offset": 5.0}]
    cfg.manufacturing.build_direction = [0.0, 0.0, 1.0]
    cfg.manufacturing.max_overhang_angle = 45.0

    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)

    assert back.manufacturing.min_member_layers == 2
    assert back.manufacturing.symmetry_planes == [{"axis": "x", "offset": 0.0},
                                                  {"axis": "z", "offset": 5.0}]
    assert back.manufacturing.build_direction == [0.0, 0.0, 1.0]
    assert back.manufacturing.max_overhang_angle == 45.0
