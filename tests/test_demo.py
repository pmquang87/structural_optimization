"""Hermetic tests for the synthetic demo backend (:mod:`oropt.demo`).

Exercised on both the conftest mini deck (two tets) and the bundled
``examples/cantilever`` deck (2850 tets) — no OpenRadioss anywhere.
"""
from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from oropt.beso import Beso
from oropt.config import Beso as BesoCfg
from oropt.deck import Deck
from oropt.demo import demo_solve
from oropt.mesh import Mesh

CANTILEVER = (Path(__file__).resolve().parents[1]
              / "examples" / "cantilever" / "cantilever_0000.rad")


def _case(stem: str, node_ids=()) -> SimpleNamespace:
    return SimpleNamespace(
        stem=stem, sigma_allow=100.0,
        disp_constraints=[SimpleNamespace(node_id=n, d_allow=1.0)
                          for n in node_ids])


def _opts(**kw) -> SimpleNamespace:
    return SimpleNamespace(enabled=True, **kw)


@pytest.fixture
def mini_deck(mini_deck_path: Path) -> Deck:
    return Deck.load(mini_deck_path, 60000000, 60000000)


@pytest.fixture(scope="module")
def cant_deck() -> Deck:
    return Deck.load(CANTILEVER, 60000000, 60000000)


def _tip_node(deck: Deck) -> int:
    """The design node nearest the centre of the max-x (loaded) face."""
    used = np.unique(deck.elem_conn)
    mask = np.isin(deck.node_ids, used)
    ids, xyz = deck.node_ids[mask], deck.node_xyz[mask]
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    target = np.array([hi[0], (lo[1] + hi[1]) / 2, (lo[2] + hi[2]) / 2])
    return int(ids[np.argmin(np.linalg.norm(xyz - target, axis=1))])


# ---- determinism & basic shape ---------------------------------------------

def test_deterministic_and_default_opts(mini_deck: Deck, cant_deck: Deck):
    for deck, node in ((mini_deck, 60000005), (cant_deck, _tip_node(cant_deck))):
        alive = np.ones(deck.n_design_elements, dtype=bool)
        alive[deck.n_design_elements // 2:] = deck is mini_deck  # nontrivial mask on cantilever
        case = _case("demo", [node])
        run1, r1 = demo_solve(deck, alive, case, _opts())
        run2, r2 = demo_solve(deck, alive, case, _opts())
        assert np.array_equal(r1.element_ids, r2.element_ids)
        assert np.array_equal(r1.energy, r2.energy)
        assert np.array_equal(r1.vonmises, r2.vonmises)
        assert r1.sigma_max == r2.sigma_max and r1.disp == r2.disp
        assert r1.disps == r2.disps
        # a bare namespace exercises the getattr defaults = explicit defaults
        _, r3 = demo_solve(deck, alive, case,
                           _opts(sigma0=100.0, disp0=0.5, hardening=1.5))
        assert np.array_equal(r1.energy, r3.energy)
        assert r1.sigma_max == r3.sigma_max
        assert run1.ok and run1.stage == "ok" and "demo" in run1.message


def test_energy_positive_not_null_and_aligned(cant_deck: Deck):
    alive = np.zeros(cant_deck.n_design_elements, dtype=bool)
    alive[::2] = True                                     # arbitrary alive subset
    _, res = demo_solve(cant_deck, alive, _case("demo", [_tip_node(cant_deck)]),
                        _opts())
    assert np.array_equal(res.element_ids, cant_deck.elem_ids[alive])
    assert res.energy.shape == res.vonmises.shape == res.element_ids.shape
    assert np.all(res.energy > 0.0)
    assert np.all(res.vonmises > 0.0)
    assert not res.is_null_solve
    assert res.sigma_max == pytest.approx(float(res.vonmises.max()))


# ---- constraint response ----------------------------------------------------

def test_removal_raises_stress_and_disp(cant_deck: Deck):
    n = cant_deck.n_design_elements
    node = _tip_node(cant_deck)
    case, opts = _case("demo", [node]), _opts(sigma0=80.0, disp0=0.4, hardening=1.5)
    prev_sig, prev_disp = -math.inf, -math.inf
    for frac in (1.0, 0.8, 0.6, 0.4):
        alive = np.zeros(n, dtype=bool)
        alive[:int(n * frac)] = True
        _, res = demo_solve(cant_deck, alive, case, opts)
        assert res.sigma_max > prev_sig
        assert res.disp > prev_disp
        assert res.disps[node] == res.disp
        prev_sig, prev_disp = res.sigma_max, res.disp
    # full volume = the configured baseline
    _, full = demo_solve(cant_deck, np.ones(n, dtype=bool), case, opts)
    assert full.sigma_max == pytest.approx(80.0)
    assert full.disp == pytest.approx(0.4)


# ---- degenerate inputs ------------------------------------------------------

def test_empty_alive_mask(mini_deck: Deck):
    alive = np.zeros(mini_deck.n_design_elements, dtype=bool)
    run, res = demo_solve(mini_deck, alive, _case("mini", [60000005]), _opts())
    assert run.ok
    assert res.element_ids.size == 0 and res.energy.size == 0
    assert res.vonmises.size == 0
    assert math.isnan(res.sigma_max) and math.isnan(res.disp)
    assert res.disp_node_id == 60000005
    assert set(res.disps) == {60000005} and math.isnan(res.disps[60000005])
    assert res.is_null_solve       # nothing alive = degenerate, like a real run


def test_missing_disp_node_gets_nan(mini_deck: Deck):
    alive = np.ones(mini_deck.n_design_elements, dtype=bool)
    _, res = demo_solve(mini_deck, alive,
                        _case("mini", [99999999, 60000005]), _opts())
    assert set(res.disps) == {99999999, 60000005}
    assert math.isnan(res.disps[99999999])
    assert math.isfinite(res.disps[60000005])
    # first constraint node drives the convenience scalars (parse_vtk semantics)
    assert res.disp_node_id == 99999999 and math.isnan(res.disp)


def test_no_disp_constraints(mini_deck: Deck):
    alive = np.ones(mini_deck.n_design_elements, dtype=bool)
    _, res = demo_solve(mini_deck, alive, _case("mini"), _opts())
    assert res.disps == {} and res.disp_node_id is None
    assert math.isnan(res.disp)
    assert math.isfinite(res.sigma_max)


# ---- field shape drives a sensible BESO update ------------------------------

def test_beso_update_removes_far_from_load_first(cant_deck: Deck):
    """One demo-driven BESO step at target_vf 0.9 strips material far from the
    load path (the removed set sits farther from the loaded tip than the kept
    set does) — the weak sanity check that the synthetic field looks like a
    beam load path, not noise."""
    deck = cant_deck
    node = _tip_node(deck)
    mesh = Mesh.from_deck(deck)
    bc_nodes = deck.group_nodes(60000000)
    protected = np.isin(deck.elem_conn, bc_nodes).any(axis=1)   # fixed-end layer
    beso = Beso(mesh, BesoCfg(filter_radius=0.0), protected)

    alive = np.ones(deck.n_design_elements, dtype=bool)
    _, res = demo_solve(deck, alive, _case("demo", [node]), _opts())
    raw = beso.raw_sensitivity(res, deck.elem_ids, alive)
    sens = beso.filter_history(raw, None)
    new_alive = beso.update(alive, sens, target_vf=0.9)

    removed = alive & ~new_alive
    assert removed.sum() > 0
    assert new_alive.sum() < alive.sum()
    # distance of each element centroid to the loaded tip node
    k = int(np.flatnonzero(deck.node_ids == node)[0])
    d_load = np.linalg.norm(mesh.centroids - deck.node_xyz[k], axis=1)
    assert d_load[removed].mean() > d_load[new_alive].mean()
