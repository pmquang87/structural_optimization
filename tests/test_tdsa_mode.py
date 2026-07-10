"""sensitivity: 'tdsa' in beso.map_sensitivity (roadmap A4, on the A2 tensor).

Hermetic: fabricated Results (no solver, no VTK). Pins that the mode returns
exactly the verified oropt.tdsa.td_compliance_3d field scattered onto the deck's
element array, that a missing tensor degrades to the energy sensitivity with a
one-time warning instead of crashing mid-run, and that fixing E=1.0 inside the
mode is legitimate (E is a pure positive prefactor of the TD -> rank-invariant).
"""
import numpy as np
import pytest

import oropt.beso as beso
from oropt.beso import TDSA_NU_DEFAULT, map_sensitivity
from oropt.results import Results
from oropt.tdsa import td_compliance_3d


def _stress(n=4, seed=0):
    rng = np.random.default_rng(seed)
    return 100.0 * rng.standard_normal((n, 6))


def _results(elem=(1, 2, 3, 4), stress="default", seed=0):
    ids = np.asarray(elem, dtype=np.int64)
    n = ids.size
    if isinstance(stress, str):
        stress = _stress(n, seed)
    return Results(element_ids=ids,
                   energy=np.linspace(1.0, 2.0, n),
                   vonmises=np.linspace(10.0, 20.0, n),
                   sigma_max=20.0, disp=0.1, disp_node_id=None, stress=stress)


def test_tdsa_matches_td_compliance_3d_directly():
    res = _results()
    got = map_sensitivity(res, np.array([1, 2, 3, 4]), "tdsa")
    want = td_compliance_3d(res.stress, E=1.0, nu=TDSA_NU_DEFAULT)
    assert np.allclose(got, want)
    # and the kwarg threads a non-default Poisson ratio through
    got_nu = map_sensitivity(res, np.array([1, 2, 3, 4]), "tdsa", tdsa_nu=0.3)
    assert np.allclose(got_nu, td_compliance_3d(res.stress, E=1.0, nu=0.3))
    assert not np.allclose(got_nu, want)          # nu genuinely changes the field


def test_tdsa_scatters_by_element_id_dead_elements_zero():
    """Same scatter semantics as the energy mode: card order (not id order),
    elements absent from the results get 0."""
    res = _results(elem=(1, 3))                    # element 2 dead / absent
    elem_ids = np.array([3, 2, 1])                 # deck card order, unsorted
    got = map_sensitivity(res, elem_ids, "tdsa")
    td = td_compliance_3d(res.stress, E=1.0, nu=TDSA_NU_DEFAULT)
    assert got[0] == pytest.approx(td[1])          # id 3
    assert got[1] == 0.0                           # id 2: dead -> 0
    assert got[2] == pytest.approx(td[0])          # id 1


def test_tdsa_stress_none_falls_back_to_energy_with_one_warning(monkeypatch):
    monkeypatch.setattr(beso, "_tdsa_fallback_warned", False)
    res = _results(stress=None)
    elem_ids = np.array([1, 2, 3, 4])
    with pytest.warns(RuntimeWarning, match="tdsa.*stress"):
        got = map_sensitivity(res, elem_ids, "tdsa")
    assert np.allclose(got, map_sensitivity(res, elem_ids, "energy"))
    # the warning is one-time: a second call stays silent (no mid-run spam)
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")
        again = map_sensitivity(res, elem_ids, "tdsa")
    assert np.allclose(again, got)


def test_tdsa_ranking_is_scale_invariant_in_E():
    """E enters td_compliance_3d only as the global 1/E prefactor, so fixing
    E=1.0 in the mode cannot change any threshold/ranking decision."""
    s = _stress(64, seed=7)
    td_1 = td_compliance_3d(s, E=1.0, nu=TDSA_NU_DEFAULT)
    td_E = td_compliance_3d(s, E=71000.0, nu=TDSA_NU_DEFAULT)
    assert np.allclose(td_E * 71000.0, td_1)               # exact 1/E scaling
    assert np.array_equal(np.argsort(td_1), np.argsort(td_E))


def test_tdsa_other_modes_unchanged_with_stress_present():
    """Carrying a tensor must not perturb the classic modes."""
    res = _results()
    elem_ids = np.array([1, 2, 3, 4])
    assert np.allclose(map_sensitivity(res, elem_ids, "energy"), res.energy)
    assert np.allclose(map_sensitivity(res, elem_ids, "vonmises"), res.vonmises)


def test_tdsa_no_overlap_returns_zeros():
    res = _results(elem=(100, 101), stress=_stress(2))
    assert map_sensitivity(res, np.array([1, 2, 3]), "tdsa").tolist() == [0, 0, 0]


def test_beso_raw_sensitivity_reads_tdsa_nu_via_getattr():
    """Beso.raw_sensitivity threads cfg.tdsa_nu (getattr default) into the mode,
    so the config dataclass needs no field until the knob is formally added."""
    from oropt.beso import Beso
    from oropt.config import Beso as BesoCfg
    from oropt.mesh import Mesh

    n = 3
    conn = np.array([[i, i + 1, i + 2, i + 3] for i in range(n)])
    mesh = Mesh(centroids=np.zeros((n, 3)), volumes=np.ones(n),
                conn_rows=conn, n_nodes=int(conn.max()) + 1, design_node_min=0)
    cfg = BesoCfg(sensitivity="tdsa", filter_radius=0.0)
    b = Beso(mesh, cfg, protected_mask=np.zeros(n, bool))
    res = _results(elem=(1, 2, 3), stress=_stress(3, seed=3))
    elem_ids = np.array([1, 2, 3])
    alive = np.ones(n, bool)
    got = b.raw_sensitivity(res, elem_ids, alive)
    assert np.allclose(got, td_compliance_3d(res.stress, E=1.0, nu=TDSA_NU_DEFAULT))
    cfg.tdsa_nu = 0.25                            # duck-typed knob via getattr
    got_25 = b.raw_sensitivity(res, elem_ids, alive)
    assert np.allclose(got_25, td_compliance_3d(res.stress, E=1.0, nu=0.25))
