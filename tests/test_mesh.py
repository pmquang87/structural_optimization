"""Mesh geometry, filter, connectivity, protected regions."""
import numpy as np

from oropt.deck import Deck
from oropt.mesh import Mesh


def test_centroids_and_volumes(mini_deck_path):
    d = Deck.load(mini_deck_path, 60000000, 60000000)
    m = Mesh.from_deck(d)
    # tet1 = unit corner tet -> volume 1/6, centroid (0.25,0.25,0.25)
    assert np.isclose(m.volumes[0], 1.0 / 6.0)
    assert np.allclose(m.centroids[0], [0.25, 0.25, 0.25])
    assert m.n_elements == 2


def test_filter_identity_when_radius_zero(mini_deck_path):
    d = Deck.load(mini_deck_path, 60000000, 60000000)
    m = Mesh.from_deck(d)
    W = m.filter_matrix(0.0)
    raw = np.array([1.0, 5.0])
    assert np.allclose(W @ raw, raw)


def _chain_mesh(n_connected=4, with_island=True):
    """A chain of tets sharing nodes, optionally plus one disconnected tet."""
    conn = [[i, i + 1, i + 2, i + 3] for i in range(n_connected)]
    nmax = n_connected + 3
    if with_island:
        conn.append([nmax + 10, nmax + 11, nmax + 12, nmax + 13])
    conn = np.array(conn)
    n = conn.size and int(conn.max()) + 1
    return Mesh(centroids=np.zeros((len(conn), 3)),
               volumes=np.ones(len(conn)), conn_rows=conn,
               n_nodes=n, design_node_min=0)


def test_keep_connected_drops_island():
    m = _chain_mesh(4, with_island=True)
    alive = np.ones(5, bool)
    seed = np.array([True, False, False, False, False])   # seed in the chain
    kept = m.keep_connected(alive, seed)
    assert kept[:4].all()        # chain kept
    assert not kept[4]           # disconnected island dropped


def test_keep_connected_no_seed():
    m = _chain_mesh(4, with_island=False)
    kept = m.keep_connected(np.ones(4, bool), np.zeros(4, bool))
    assert not kept.any()


def test_protected_mask_grows_from_bc(mini_deck_path):
    d = Deck.load(mini_deck_path, 60000000, 60000000)
    m = Mesh.from_deck(d)
    # BC nodes 60000001/2 touch both tets -> both protected
    prot = m.protected_mask(d, np.array([60000001]), contact_dist=0.0, layers=0)
    assert prot[0]              # tet1 contains node 60000001
    assert not prot[1]          # tet2 does not (layers=0, no dilation)
    prot2 = m.protected_mask(d, np.array([60000001]), contact_dist=0.0, layers=1)
    assert prot2.all()          # one hop reaches tet2 via shared nodes
