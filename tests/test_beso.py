"""BESO sensitivity mapping, filtering, and the volume-target update."""
import numpy as np

from oropt.beso import Beso
from oropt.config import Beso as BesoCfg
from oropt.mesh import Mesh
from oropt.results import Results


def _chain_mesh(n=5):
    conn = np.array([[i, i + 1, i + 2, i + 3] for i in range(n)])
    return Mesh(centroids=np.zeros((n, 3)), volumes=np.ones(n),
               conn_rows=conn, n_nodes=int(conn.max()) + 1, design_node_min=0)


def _beso(target_vf=0.6, sensitivity="energy"):
    m = _chain_mesh(5)
    cfg = BesoCfg(filter_radius=0.0, target_volume_fraction=target_vf,
                  evolution_rate=0.2, max_add_ratio=0.0, sensitivity=sensitivity)
    protected = np.array([True, False, False, False, False])   # element 0 = seed
    return Beso(m, cfg, protected)


def test_raw_sensitivity_maps_by_id():
    b = _beso()
    elem_ids = np.array([1, 2, 3, 4, 5])
    res = Results(element_ids=np.array([1, 3, 5]),
                  energy=np.array([10.0, 30.0, 50.0]),
                  vonmises=np.array([1.0, 3.0, 5.0]),
                  sigma_max=5.0, disp=0.1, disp_node_id=None)
    raw = b.raw_sensitivity(res, elem_ids, np.ones(5, bool))
    assert raw.tolist() == [10, 0, 30, 0, 50]      # dead/absent elements -> 0


def test_filter_identity_and_history():
    b = _beso()
    raw = np.array([1.0, 2, 3, 4, 5])
    assert np.allclose(b.filter_history(raw, None), raw)         # radius 0 -> identity
    prev = np.zeros(5)
    out = b.filter_history(raw, prev)                            # history_weight 0.5
    assert np.allclose(out, 0.5 * raw)


def test_update_removes_lowest_keeps_protected():
    b = _beso(target_vf=0.6)
    sens = np.array([10.0, 20, 30, 40, 50])   # element 1 is lowest unprotected
    new = b.update(np.ones(5, bool), sens, target_vf=0.6)
    assert new[0]            # protected element kept despite low sensitivity
    assert not new[1]        # lowest-sensitivity unprotected element deleted
    assert new[4]            # highest-sensitivity kept
    assert new.sum() < 5     # something was removed


def test_next_target_vf_gate():
    b = _beso()
    assert b.next_target_vf(0.8, feasible=True) < 0.8     # shrink while feasible
    assert b.next_target_vf(0.8, feasible=False) > 0.8    # back off when infeasible
    # never drops below the configured floor
    assert b.next_target_vf(0.61, feasible=True) == b.cfg.target_volume_fraction
