"""Property-based tests (Hypothesis) for the deck round-trip and the offline
numeric kernels (E9).

Two seams are fuzzed here:

* **Deck round-trip** -- over an arbitrary boolean alive mask on a fabricated
  small design part, ``Deck.write`` then re-read must preserve the element/node
  invariants the loop depends on: the surviving elements are exactly the alive
  ones (in order), every design mesh-node orphaned by deletion is pinned or
  already protected, and the protected node set is never lost.

* **Numeric kernels of the SIMP prototype** (``oropt/simp.py``) and the shared
  sensitivity filter (``oropt/simp.density_filter`` == ``oropt.mesh.Mesh.
  filter_matrix``): the OC density update stays in ``[rho_min, 1]``, honours the
  per-step move limit, and drives the physical volume onto the target when that
  target is move/​bound-reachable; the row-normalised filter is a partition of
  unity (each row sums to 1), non-negative, and an averaging operator (a filtered
  non-negative field is bounded by the field's own min/max and a constant field
  is preserved).

Only invariants that actually hold in the code are asserted. In particular the
forward filter ``W @ x`` is row-stochastic (an average), NOT column-stochastic,
so the *sum* of the field is NOT conserved -- that (false) invariant is
deliberately not asserted; the partition-of-unity / averaging properties are the
real conserved quantities. Hermetic and fast (``max_examples=50``).
"""
from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from oropt.deck import Deck
from oropt.mesh import Mesh
from oropt.simp import SimpParams, density_filter, oc_update, simp_modulus

# --------------------------------------------------------------------------- #
# A fabricated small design part: a chain of TET4s (each sharing a face with the
# next), two of whose nodes are protected via a /GRNOD/NODE set. Larger than the
# 2-element conftest deck so an arbitrary mask has a meaningful space to explore.
# --------------------------------------------------------------------------- #
_NODE_MIN = 60000000
_PART_ID = 60000000
_N_TETS = 6
_N_NODES = _N_TETS + 3            # a chain of k tets spans k+3 nodes
_PROTECTED = frozenset({_NODE_MIN + 1, _NODE_MIN + _N_NODES})


def _build_deck_text() -> str:
    lines = ["#- fabricated design part", "/MAT/LAW1/3", "mat",
             "/NODE",
             "#  Node ID               X               Y               Z"]
    for k in range(1, _N_NODES + 1):
        nid = _NODE_MIN + k
        x, y, z = float(k), float((k * k) % 7), float((k * 13) % 11)
        lines.append(f"  {nid:>8}  {x:>18}  {y:>18}  {z:>18}")
    lines += [f"/PART/{_PART_ID}", "part", "         3         3         0",
              f"/TETRA4/{_PART_ID}"]
    for i in range(_N_TETS):                      # tet i -> nodes i+1..i+4
        eid = _NODE_MIN + 100 + i
        n = [_NODE_MIN + i + 1 + j for j in range(4)]
        lines.append(f"  {eid:>8}  {n[0]:>8}  {n[1]:>8}  {n[2]:>8}  {n[3]:>8}")
    lines += ["#-  PROPERTIES:", "/PROP/SOLID/3", "prop",
              f"/GRNOD/NODE/{_PART_ID}", "prot_set"]
    prot = sorted(_PROTECTED)
    lines.append("".join(f"{v:>10}" for v in prot))
    lines += [f"/BCS/{_PART_ID}", "bc",
              "#  Tra rot   skew_ID  grnod_ID",
              f"   111 111         0{_PART_ID:>10}", "/END"]
    return "\n".join(lines) + "\n"


_DECK_TEXT = _build_deck_text()


def _load_deck() -> Deck:
    return Deck(_DECK_TEXT.splitlines(), "\n", _PART_ID, _NODE_MIN)


def _masks(n: int):
    return hnp.arrays(np.bool_, (n,))


# --------------------------------------------------------------------------- #
# Deck round-trip invariants
# --------------------------------------------------------------------------- #
@settings(max_examples=50, deadline=None,
          suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(mask=_masks(_N_TETS))
def test_deck_roundtrip_preserves_invariants(tmp_path, mask):
    deck = _load_deck()
    assert deck.protected_nodes == _PROTECTED       # sanity on the fixture
    no_pin = set(deck.protected_nodes)              # mirror the loop's call

    out = tmp_path / "rewritten_0000.rad"
    stats = deck.write(out, mask, no_pin=no_pin)
    deck2 = Deck.load(out, _PART_ID, _NODE_MIN)

    # (1) surviving elements are exactly the alive ones, in the same order.
    np.testing.assert_array_equal(deck2.elem_ids, deck.elem_ids[mask])
    assert stats["elements_alive"] == int(mask.sum())
    assert stats["elements_total"] == mask.size

    # (2) every design mesh-node orphaned by the deletion is pinned or protected.
    ref = (np.unique(deck.elem_conn[mask]) if mask.any()
           else np.empty(0, dtype=np.int64))
    all_mesh = np.unique(deck.elem_conn)
    design_mesh = np.intersect1d(
        deck.node_ids[deck.node_ids >= _NODE_MIN], all_mesh)
    orphaned = set(np.setdiff1d(design_mesh, ref).tolist())
    pinned = set(deck2.group_nodes(91000001).tolist())      # injected free group
    assert orphaned <= (pinned | set(_PROTECTED)), \
        f"orphaned nodes neither pinned nor protected: {orphaned - pinned - set(_PROTECTED)}"
    # the write's own count of pinned free nodes matches (orphans minus protected)
    assert stats["free_nodes_pinned"] == len(orphaned - set(_PROTECTED))

    # (3) the protected set is preserved across the rewrite.
    assert _PROTECTED <= deck2.protected_nodes

    # (4) surviving elements only reference nodes that still exist in the deck.
    if deck2.elem_ids.size:
        assert set(np.unique(deck2.elem_conn).tolist()) <= set(deck2.node_ids.tolist())


def test_deck_roundtrip_all_alive_pins_nothing(tmp_path):
    """The two boundary masks explicitly: all-alive injects no free-node block;
    all-dead pins every (unprotected) design mesh-node."""
    deck = _load_deck()
    no_pin = set(deck.protected_nodes)

    keep = deck.write(tmp_path / "keep.rad", np.ones(_N_TETS, bool), no_pin=no_pin)
    assert keep["free_nodes_pinned"] == 0

    drop = deck.write(tmp_path / "drop.rad", np.zeros(_N_TETS, bool), no_pin=no_pin)
    design_mesh = np.intersect1d(deck.node_ids[deck.node_ids >= _NODE_MIN],
                                 np.unique(deck.elem_conn))
    assert drop["free_nodes_pinned"] == len(set(design_mesh.tolist()) - _PROTECTED)


# --------------------------------------------------------------------------- #
# OC density update invariants (simp.py)
# --------------------------------------------------------------------------- #
_N_CELLS = 6
_MOVE = 0.2
_RHO_MIN = 1e-3


@settings(max_examples=50, deadline=None)
@given(
    rho=hnp.arrays(np.float64, (_N_CELLS,),
                   elements=st.floats(_RHO_MIN, 1.0, allow_nan=False,
                                      allow_infinity=False)),
    dc_pos=hnp.arrays(np.float64, (_N_CELLS,),
                      elements=st.floats(1.0, 1e3, allow_nan=False,
                                         allow_infinity=False)),
    volumes=hnp.arrays(np.float64, (_N_CELLS,),
                       elements=st.floats(0.1, 10.0, allow_nan=False,
                                          allow_infinity=False)),
    vol_frac=st.floats(0.2, 0.8),
)
def test_oc_update_bounds_move_and_volume(rho, dc_pos, volumes, vol_frac):
    # Compliance gradient is strictly < 0, so the OC scaling base is > 0 and the
    # update can actually reach anywhere in [lower, upper] for some multiplier --
    # otherwise (dc == 0) the update is pinned to the lower bound and the volume
    # target is unreachable however the multiplier is chosen.
    dc = -dc_pos
    dv = volumes                                   # dV/drho = element volume >= 0
    out = oc_update(rho, dc, dv, volumes, vol_frac,
                    move=_MOVE, rho_min=_RHO_MIN, rho_max=1.0)

    # density stays in [rho_min, 1]
    assert np.all(out >= _RHO_MIN - 1e-12)
    assert np.all(out <= 1.0 + 1e-12)
    # per-step move limit respected
    assert np.all(np.abs(out - rho) <= _MOVE + 1e-9)

    # volume constraint met when the target is reachable within move + bounds
    lower = np.maximum(_RHO_MIN, rho - _MOVE)
    upper = np.minimum(1.0, rho + _MOVE)
    target = vol_frac * volumes.sum()
    lo_V = float((volumes * lower).sum())
    hi_V = float((volumes * upper).sum())
    achieved = float((volumes * out).sum())
    if lo_V <= target <= hi_V:
        np.testing.assert_allclose(achieved, target, rtol=1e-3, atol=1e-6)
    else:                                          # unreachable -> clamps at a bound
        assert lo_V - 1e-6 <= achieved <= hi_V + 1e-6


@settings(max_examples=50, deadline=None)
@given(rho=hnp.arrays(np.float64, (_N_CELLS,),
                      elements=st.floats(0.0, 1.0, allow_nan=False,
                                         allow_infinity=False)),
       penal=st.floats(1.0, 5.0))
def test_simp_modulus_in_material_bounds(rho, penal):
    """The interpolated modulus never leaves [Emin, E0] for rho in [0,1]."""
    params = SimpParams(E0=1.0, Emin=1e-9, penal=penal)
    E = simp_modulus(rho, params)
    assert np.all(E >= params.Emin - 1e-15)
    assert np.all(E <= params.E0 + 1e-12)


# --------------------------------------------------------------------------- #
# Sensitivity filter invariants (density_filter / Mesh.filter_matrix)
# --------------------------------------------------------------------------- #
@settings(max_examples=50, deadline=None)
@given(
    centroids=hnp.arrays(np.float64, (8, 3),
                         elements=st.floats(0.0, 10.0, allow_nan=False,
                                            allow_infinity=False)),
    raw=hnp.arrays(np.float64, (8,),
                   elements=st.floats(0.0, 100.0, allow_nan=False,
                                      allow_infinity=False)),
    radius=st.floats(0.5, 6.0),
)
def test_density_filter_is_a_nonneg_partition_of_unity(centroids, raw, radius):
    W = density_filter(centroids, radius)
    rowsums = np.asarray(W.sum(axis=1)).ravel()
    # partition of unity: every row sums to 1
    np.testing.assert_allclose(rowsums, 1.0, atol=1e-9)
    # weights are non-negative
    assert W.data.min() >= -1e-15

    filtered = W @ raw
    # a filtered non-negative field is non-negative and bounded by its own range
    assert np.all(filtered >= -1e-12)
    assert np.all(filtered <= raw.max() + 1e-9)
    assert np.all(filtered >= raw.min() - 1e-9)
    # a constant field is preserved exactly (partition of unity)
    np.testing.assert_allclose(W @ np.ones(centroids.shape[0]), 1.0, atol=1e-9)


def test_mesh_filter_matrix_matches_the_filter_invariants():
    """The real ``Mesh.filter_matrix`` (built from a Deck) shares the
    partition-of-unity / non-negativity properties tested above on the prototype
    ``density_filter``."""
    mesh = Mesh.from_deck(_load_deck())
    W = mesh.filter_matrix(3.0)
    rowsums = np.asarray(W.sum(axis=1)).ravel()
    np.testing.assert_allclose(rowsums, 1.0, atol=1e-9)
    assert W.data.min() >= -1e-15
    # identity for radius <= 0
    W0 = mesh.filter_matrix(0.0)
    np.testing.assert_allclose(W0.toarray(), np.eye(mesh.n_elements), atol=1e-12)
