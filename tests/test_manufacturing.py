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


# ---- maximum member size ------------------------------------------------------
def _cored_lump_mesh():
    """A central element ``Ec`` (index 0) surrounded by four arm elements
    E1..E4 (indices 1-4), each of which also touches its own void tet V1..V4
    (indices 5-8). So the arms sit one hop from a void (dist 1) while the core is
    two hops from any void (dist 2) — a lump too thick for a 1-hop max member."""
    conn = np.array([
        [0, 1, 2, 3],        # 0 Ec: shares one node with each arm
        [1, 10, 11, 12],     # 1 E1 (node 1 <-> Ec, node 12 <-> V1)
        [2, 20, 21, 22],     # 2 E2
        [3, 30, 31, 32],     # 3 E3
        [0, 40, 41, 42],     # 4 E4 (node 0 <-> Ec, node 42 <-> V4)
        [12, 13, 14, 15],    # 5 V1
        [22, 23, 24, 25],    # 6 V2
        [32, 33, 34, 35],    # 7 V3
        [42, 43, 44, 45],    # 8 V4
    ])
    return Mesh(centroids=np.zeros((len(conn), 3)), volumes=np.ones(len(conn)),
                conn_rows=conn, n_nodes=int(conn.max()) + 1, design_node_min=0)


def test_max_member_carves_core_keeps_walls():
    mesh = _cored_lump_mesh()
    alive = np.array([True, True, True, True, True, False, False, False, False])
    opts = ManufacturingOpts(max_member_layers=1)

    out = apply_manufacturing(alive, mesh, opts)

    assert not out[0]           # core is 2 hops from a void -> carved
    assert out[1:5].all()       # arms are within 1 hop of a void -> kept
    assert not out[5:].any()    # voids stay void (carve never adds material)


def test_max_member_limit_not_exceeded_is_identity():
    mesh = _cored_lump_mesh()
    alive = np.array([True, True, True, True, True, False, False, False, False])
    opts = ManufacturingOpts(max_member_layers=3)   # core (dist 2) is within limit

    out = apply_manufacturing(alive, mesh, opts)

    assert np.array_equal(out, alive)               # nothing over-limit -> no carve


def test_max_member_never_carves_protected_core():
    mesh = _cored_lump_mesh()
    alive = np.array([True, True, True, True, True, False, False, False, False])
    protected = np.zeros(9, dtype=bool); protected[0] = True   # protect the core
    opts = ManufacturingOpts(max_member_layers=1)

    out = apply_manufacturing(alive, mesh, opts, protected)

    assert out[0]               # protected core survives despite being over-limit
    assert out[1:5].all()


def _bar_mesh():
    """A chain V - A1 - A2 - A3 - A4 - V (indices 0..5) where each element shares
    exactly one node with the next. The two interior elements A2, A3 (indices 2,3)
    are each two hops from the nearest void and mutually adjacent, so carving
    either one brings the other within one hop — one carve satisfies a 1-hop max
    member, and which one goes is decided by sensitivity."""
    conn = np.array([[3 * i, 3 * i + 1, 3 * i + 2, 3 * i + 3] for i in range(6)])
    return Mesh(centroids=np.arange(6).reshape(-1, 1) * np.array([[1.0, 0, 0]]),
                volumes=np.ones(6), conn_rows=conn,
                n_nodes=int(conn.max()) + 1, design_node_min=0)


def test_max_member_carves_lowest_sensitivity_first():
    mesh = _bar_mesh()
    alive = np.array([False, True, True, True, True, False])
    opts = ManufacturingOpts(max_member_layers=1)

    # A2 (index 2) is the least useful -> it is the one carved; A3 kept
    sens = np.array([9.0, 9.0, 0.1, 0.5, 9.0, 9.0])
    out = apply_manufacturing(alive, mesh, opts, sensitivity=sens)
    assert not out[2] and out[3]

    # flip the ranking -> A3 (index 3) is carved instead, A2 kept
    sens = np.array([9.0, 9.0, 0.5, 0.1, 9.0, 9.0])
    out = apply_manufacturing(alive, mesh, opts, sensitivity=sens)
    assert not out[3] and out[2]


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
    assert not out[1] and not out[2]


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


# ---- casting / draw direction -------------------------------------------------
def _single_column_mesh(z_heights):
    """One vertical column of tets stacked along +z at the given heights (shared
    footprint at x = y = 0), each with its own distinct nodes. Casting/extrusion
    read only centroids, so the connectivity is arbitrary but valid."""
    z = np.asarray(z_heights, dtype=float)
    centroids = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    conn = np.array([[4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3]
                     for i in range(len(z))])
    return Mesh(centroids=centroids, volumes=np.ones(len(z)), conn_rows=conn,
                n_nodes=int(conn.max()) + 1, design_node_min=0)


def test_casting_single_sided_removes_solid_above_void():
    mesh = _single_column_mesh([0.0, 1.0, 2.0, 3.0, 4.0])
    alive = np.array([True, True, True, False, True])   # a shelf floats above a void
    opts = ManufacturingOpts(draw_direction=[0.0, 0.0, 1.0])

    out = apply_manufacturing(alive, mesh, opts)

    # solid must be a bottom prefix: the z=4 shelf sitting above the z=3 void goes
    assert out.tolist() == [True, True, True, False, False]


def test_casting_two_sided_keeps_largest_contiguous_run():
    mesh = _single_column_mesh([0.0, 1.0, 2.0, 3.0, 4.0])
    alive = np.array([True, False, True, True, True])   # a lone base + a 3-run
    opts = ManufacturingOpts(draw_direction=[0.0, 0.0, 1.0], draw_two_sided=True)

    out = apply_manufacturing(alive, mesh, opts)

    # one contiguous block around a parting surface: the longest run survives
    assert out.tolist() == [False, False, True, True, True]


def test_casting_off_when_no_draw_direction():
    mesh = _single_column_mesh([0.0, 1.0, 2.0, 3.0, 4.0])
    alive = np.array([True, True, True, False, True])
    opts = ManufacturingOpts(draw_direction=None, draw_two_sided=True)

    out = apply_manufacturing(alive, mesh, opts)

    assert np.array_equal(out, alive)


# ---- extrusion ----------------------------------------------------------------
def _two_column_mesh():
    """Two vertical prisms along z, footprints at x = 0 and x = 5, three elements
    each (six total, column A = indices 0-2, column B = 3-5)."""
    centroids = np.array([
        [0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0],   # column A (x=0)
        [5.0, 0.0, 0.0], [5.0, 0.0, 1.0], [5.0, 0.0, 2.0],   # column B (x=5)
    ])
    conn = np.array([[4 * i, 4 * i + 1, 4 * i + 2, 4 * i + 3] for i in range(6)])
    return Mesh(centroids=centroids, volumes=np.ones(6), conn_rows=conn,
                n_nodes=int(conn.max()) + 1, design_node_min=0)


def test_extrusion_majority_vote_makes_prisms_uniform():
    mesh = _two_column_mesh()
    # column A: 2/3 alive -> whole prism solid; column B: 1/3 alive -> whole void
    alive = np.array([True, True, False, True, False, False])
    opts = ManufacturingOpts(extrusion_axis=[0.0, 0.0, 1.0])

    out = apply_manufacturing(alive, mesh, opts)

    assert out.tolist() == [True, True, True, False, False, False]


def test_extrusion_off_when_no_axis():
    mesh = _two_column_mesh()
    alive = np.array([True, True, False, True, False, False])
    opts = ManufacturingOpts(extrusion_axis=None)

    out = apply_manufacturing(alive, mesh, opts)

    assert np.array_equal(out, alive)


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
    assert manufacturing_active(ManufacturingOpts(max_member_layers=3)) is True
    assert manufacturing_active(
        ManufacturingOpts(symmetry_planes=[{"axis": "y", "offset": 0.0}])) is True
    assert manufacturing_active(
        ManufacturingOpts(draw_direction=[0, 0, 1])) is True
    assert manufacturing_active(
        ManufacturingOpts(draw_direction=[0, 0, 1], draw_two_sided=True)) is True
    assert manufacturing_active(
        ManufacturingOpts(extrusion_axis=[1, 0, 0])) is True
    assert manufacturing_active(
        ManufacturingOpts(build_direction=[0, 0, 1], max_overhang_angle=40.0)) is True
    # build direction without a positive angle is still off
    assert manufacturing_active(
        ManufacturingOpts(build_direction=[0, 0, 1], max_overhang_angle=0.0)) is False
    # a two-sided flag alone (no draw direction) is off; max member of 0 is off
    assert manufacturing_active(ManufacturingOpts(draw_two_sided=True)) is False
    assert manufacturing_active(ManufacturingOpts(max_member_layers=0)) is False


# ---- config roundtrip ---------------------------------------------------------
def test_manufacturing_config_defaults_and_roundtrip(tmp_path):
    cfg = Config()
    assert cfg.manufacturing.min_member_layers == 0
    assert cfg.manufacturing.max_member_layers == 0
    assert cfg.manufacturing.symmetry_planes == []
    assert cfg.manufacturing.draw_direction is None
    assert cfg.manufacturing.draw_two_sided is False
    assert cfg.manufacturing.extrusion_axis is None
    assert cfg.manufacturing.build_direction is None
    assert cfg.manufacturing.max_overhang_angle == 0.0

    cfg.manufacturing.min_member_layers = 2
    cfg.manufacturing.max_member_layers = 4
    cfg.manufacturing.symmetry_planes = [{"axis": "x", "offset": 0.0},
                                         {"axis": "z", "offset": 5.0}]
    cfg.manufacturing.draw_direction = [0.0, 1.0, 0.0]
    cfg.manufacturing.draw_two_sided = True
    cfg.manufacturing.extrusion_axis = [1.0, 0.0, 0.0]
    cfg.manufacturing.build_direction = [0.0, 0.0, 1.0]
    cfg.manufacturing.max_overhang_angle = 45.0

    p = tmp_path / "cfg.yaml"
    cfg.to_yaml(p)
    back = Config.from_yaml(p)

    assert back.manufacturing.min_member_layers == 2
    assert back.manufacturing.max_member_layers == 4
    assert back.manufacturing.symmetry_planes == [{"axis": "x", "offset": 0.0},
                                                  {"axis": "z", "offset": 5.0}]
    assert back.manufacturing.draw_direction == [0.0, 1.0, 0.0]
    assert back.manufacturing.draw_two_sided is True
    assert back.manufacturing.extrusion_axis == [1.0, 0.0, 0.0]
    assert back.manufacturing.build_direction == [0.0, 0.0, 1.0]
    assert back.manufacturing.max_overhang_angle == 45.0
