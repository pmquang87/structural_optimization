"""Hermetic geometric sanity checks (``oropt.sanity``) on fabricated meshes.

Every mesh here is built by hand with a *known* shared-node adjacency graph (each
graph edge realised by one shared node id, the rest of a tet's four slots are
unique private nodes) or with explicit tet connectivity for the face checks, so
the expected counts are exact and computed by inspection.
"""
import numpy as np

from oropt.mesh import Mesh
from oropt.sanity import (
    SanityReport,
    audit,
    disconnected_elements,
    newly_exposed_faces,
    thin_members,
)


def _mesh(tets) -> Mesh:
    """A :class:`Mesh` whose connectivity (and hence shared-node adjacency and
    tet faces) is exactly *tets* ``(N, 4)``; geometry fields are dummies."""
    conn = np.asarray(tets, dtype=np.int64)
    n = len(conn)
    return Mesh(centroids=np.zeros((n, 3)), volumes=np.ones(n),
                conn_rows=conn, n_nodes=int(conn.max()) + 1, design_node_min=0)


# --------------------------------------------------------------------------- #
# a mesh with a deliberate island: a 3-tet chain + 1 disconnected tet
# --------------------------------------------------------------------------- #
def _island_mesh() -> Mesh:
    # chain c0-c1-c2 (shared junction nodes 2 and 3), island isl with disjoint nodes
    return _mesh([
        [1, 2, 500, 501],     # c0  (shares node 2 with c1)
        [2, 3, 502, 503],     # c1  (shares node 3 with c2)
        [3, 4, 504, 505],     # c2
        [700, 701, 702, 703],  # isl -- shares nothing
    ])


def test_disconnected_flags_the_island():
    m = _island_mesh()
    alive = np.ones(4, bool)
    anchor = np.array([True, False, False, False])   # anchored at c0
    orphan, n = disconnected_elements(m, alive, anchor)
    assert n == 1
    assert list(orphan) == [False, False, False, True]


def test_disconnected_none_when_all_reachable():
    m = _island_mesh()
    alive = np.array([True, True, True, False])       # island not alive
    anchor = np.array([True, False, False, False])
    orphan, n = disconnected_elements(m, alive, anchor)
    assert n == 0
    assert not orphan.any()


def test_disconnected_no_live_anchor_flags_all_alive():
    m = _island_mesh()
    alive = np.array([True, True, False, False])
    anchor = np.array([False, False, False, True])    # anchor element is dead
    _, n = disconnected_elements(m, alive, anchor)
    assert n == 2


# --------------------------------------------------------------------------- #
# a mesh with a one-element-thick wall next to void, plus a robust blob
# --------------------------------------------------------------------------- #
def _thin_mesh() -> Mesh:
    # elements 0..4 = star blob (b0 centre adjacent to b1..b4)
    # elements 5..7 = wall w0-w1-w2, each backed by a (dead) void element 8..10
    return _mesh([
        [100, 101, 102, 103],   # 0 b0  (adjacent to b1,b2,b3,b4)
        [100, 900, 901, 902],   # 1 b1
        [101, 903, 904, 905],   # 2 b2
        [102, 906, 907, 908],   # 3 b3
        [103, 909, 910, 911],   # 4 b4
        [200, 202, 920, 921],   # 5 w0  (adjacent to w1 via 200, d0 via 202)
        [200, 201, 203, 922],   # 6 w1  (adjacent to w0,w2 and d1)
        [201, 204, 923, 924],   # 7 w2  (adjacent to w1 via 201, d2 via 204)
        [202, 930, 931, 932],   # 8 d0  (void neighbour of w0)
        [203, 933, 934, 935],   # 9 d1  (void neighbour of w1)
        [204, 936, 937, 938],   # 10 d2 (void neighbour of w2)
    ])


def test_thin_members_flags_only_the_wall():
    m = _thin_mesh()
    alive = np.ones(11, bool)
    alive[8:] = False                                  # d0,d1,d2 are void
    thin, n = thin_members(m, alive, min_layers=1)
    assert n == 3
    assert set(np.flatnonzero(thin)) == {5, 6, 7}      # exactly the wall


def test_thin_members_excludes_protected():
    m = _thin_mesh()
    alive = np.ones(11, bool)
    alive[8:] = False
    protected = np.zeros(11, bool)
    protected[5] = True                                # w0 is an intentional patch
    thin, n = thin_members(m, alive, min_layers=1, protected=protected)
    assert n == 2
    assert set(np.flatnonzero(thin)) == {6, 7}


def test_thin_members_off_when_min_layers_zero():
    m = _thin_mesh()
    alive = np.ones(11, bool)
    alive[8:] = False
    thin, n = thin_members(m, alive, min_layers=0)
    assert n == 0
    assert not thin.any()


# --------------------------------------------------------------------------- #
# tet meshes with known face sharing
# --------------------------------------------------------------------------- #
def _two_tet_mesh() -> Mesh:
    # T0 and T1 share exactly the face (1,2,3)
    return _mesh([[0, 1, 2, 3], [1, 2, 3, 4]])


def _cavity_mesh() -> Mesh:
    # central T0 fully surrounded: one neighbour on each of its 4 faces
    return _mesh([
        [0, 1, 2, 3],   # T0 central
        [1, 2, 3, 4],   # shares face (1,2,3)
        [0, 2, 3, 5],   # shares face (0,2,3)
        [0, 1, 3, 6],   # shares face (0,1,3)
        [0, 1, 2, 7],   # shares face (0,1,2)
    ])


def test_newly_exposed_single_face():
    m = _two_tet_mesh()
    prev = np.array([True, True])
    alive = np.array([True, False])                    # T1 deleted this step
    n_exp, n_cav = newly_exposed_faces(m, alive, prev)
    assert n_exp == 1
    assert n_cav == 0                                  # T1 has boundary faces -> open


def test_newly_exposed_none_without_deletion():
    m = _two_tet_mesh()
    a = np.array([True, True])
    assert newly_exposed_faces(m, a, a) == (0, 0)


def test_newly_exposed_sealed_cavity():
    m = _cavity_mesh()
    prev = np.ones(5, bool)
    alive = prev.copy()
    alive[0] = False                                   # carve out the centre
    n_exp, n_cav = newly_exposed_faces(m, alive, prev)
    assert n_exp == 4                                  # all 4 faces newly bared
    assert n_cav == 4                                  # a sealed single-element pocket


def test_newly_exposed_accepts_explicit_tets():
    m = _two_tet_mesh()
    tets = np.array([[0, 1, 2, 3], [1, 2, 3, 4]])
    prev = np.array([True, True])
    alive = np.array([True, False])
    assert newly_exposed_faces(m, alive, prev, tets=tets) == (1, 0)


# --------------------------------------------------------------------------- #
# audit aggregation
# --------------------------------------------------------------------------- #
def test_audit_clean_design_passes():
    m = _two_tet_mesh()
    a = np.array([True, True])
    anchor = np.array([True, True])
    rep = audit(m, a, a, anchor, min_layers=1)
    assert isinstance(rep, SanityReport)
    assert rep.ok
    assert rep.warnings == []
    assert (rep.n_disconnected, rep.n_thin,
            rep.n_newly_exposed_faces, rep.n_cavity_faces) == (0, 0, 0, 0)


def test_audit_disconnected_is_severe():
    m = _island_mesh()
    alive = np.ones(4, bool)
    anchor = np.array([True, False, False, False])
    rep = audit(m, alive, None, anchor)
    assert not rep.ok
    assert rep.n_disconnected == 1
    assert any("SEVERE" in w for w in rep.warnings)


def test_audit_disconnected_advisory_when_not_severe():
    m = _island_mesh()
    alive = np.ones(4, bool)
    anchor = np.array([True, False, False, False])
    rep = audit(m, alive, None, anchor, severe_disconnected=False)
    assert rep.ok                                      # advisory only
    assert rep.n_disconnected == 1
    assert rep.warnings and "SEVERE" not in rep.warnings[0]


def test_audit_thin_and_cavity_are_advisory():
    m = _thin_mesh()
    prev = np.ones(11, bool)
    alive = prev.copy()
    alive[8:] = False                                  # delete the three void backers
    anchor = np.ones(11, bool)                         # everything anchored -> no island
    rep = audit(m, alive, prev, anchor, min_layers=1)
    assert rep.ok                                      # thin/exposure never clears ok
    assert rep.n_thin == 3
    assert any("thin" in w for w in rep.warnings)


def test_audit_exposed_face_threshold_warning():
    m = _cavity_mesh()
    prev = np.ones(5, bool)
    alive = prev.copy()
    alive[0] = False
    anchor = np.ones(5, bool)
    rep = audit(m, alive, prev, anchor, max_exposed_faces=2)
    assert rep.ok
    assert rep.n_newly_exposed_faces == 4
    assert any("self-contact" in w for w in rep.warnings)
