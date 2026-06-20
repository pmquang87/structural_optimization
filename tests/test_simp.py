"""Hermetic, analytic tests for the EXPERIMENTAL SIMP/OC prototype.

No OpenRadioss, no Deck/Mesh — everything runs against the synthetic compliance
model in :mod:`oropt.simp`. These verify the maths the spike rests on: the
compliance sensitivity equals the finite-difference gradient (so the per-element
energy OpenRadioss reports really is the SIMP sensitivity), the OC bisection hits
the volume target with bounded densities, the OC step reduces the objective, the
density filter is a volume-preserving partition of unity, and the Heaviside
projection drives grayscale toward 0/1.
"""
import numpy as np

from oropt.simp import (SimpParams, SyntheticCompliance, compliance_sensitivity,
                        density_filter, grayness, heaviside_deriv,
                        heaviside_projection, oc_update, optimize, simp_modulus,
                        simp_modulus_deriv)


def _model(n=30, seed=0):
    rng = np.random.default_rng(seed)
    geom = 0.2 + rng.random(n)
    params = SimpParams(penal=3.0, Emin=1e-6)
    return SyntheticCompliance(geom=geom, load=1.0, params=params), params, rng


# ---- SIMP interpolation ----------------------------------------------------

def test_simp_modulus_bounds_and_monotonic():
    p = SimpParams(E0=2.0, Emin=1e-3, penal=3.0)
    assert np.isclose(simp_modulus(np.array([0.0]), p)[0], p.Emin)   # void -> Emin
    assert np.isclose(simp_modulus(np.array([1.0]), p)[0], p.E0)     # solid -> E0
    rho = np.linspace(0, 1, 50)
    E = simp_modulus(rho, p)
    assert np.all(np.diff(E) > 0)                                    # strictly increasing
    # analytic derivative matches finite difference
    h = 1e-6
    fd = (simp_modulus(rho + h, p) - simp_modulus(rho - h, p)) / (2 * h)
    assert np.allclose(simp_modulus_deriv(rho, p), fd, atol=1e-4)


# ---- the key result: energy IS the sensitivity -----------------------------

def test_compliance_sensitivity_matches_finite_difference():
    """dC/drho from the measured element strain energy == central-difference
    gradient of the synthetic compliance. This is what makes SIMP feasible on a
    solver (OpenRadioss) that exposes energy but no design sensitivities."""
    model, params, rng = _model(n=25, seed=1)
    rho = 0.2 + 0.6 * rng.random(25)

    _, energy = model.solve(rho)
    dc = compliance_sensitivity(rho, energy, params)

    h = 1e-7
    fd = np.empty_like(rho)
    for i in range(rho.size):
        rp = rho.copy(); rp[i] += h
        rm = rho.copy(); rm[i] -= h
        fd[i] = (model.solve(rp)[0] - model.solve(rm)[0]) / (2 * h)

    assert np.allclose(dc, fd, rtol=1e-4, atol=1e-9)
    assert np.all(dc <= 0)                       # more material never raises compliance


# ---- density filter --------------------------------------------------------

def test_density_filter_identity_when_radius_zero():
    cent = np.column_stack([np.arange(8), np.zeros(8), np.zeros(8)]).astype(float)
    W = density_filter(cent, 0.0)
    raw = np.arange(8, dtype=float)
    assert np.allclose(W @ raw, raw)


def test_density_filter_partition_of_unity_and_volume_preserving():
    """Row-normalised hat filter: each row sums to 1, so a uniform field is
    preserved exactly (=> total volume of a uniform density is preserved), and the
    mean of an arbitrary field is preserved to within boundary effects."""
    cent = np.column_stack([np.arange(40), np.zeros(40), np.zeros(40)]).astype(float)
    W = density_filter(cent, 3.0)
    ones = np.ones(40)
    assert np.allclose(W @ ones, ones)                       # partition of unity
    rng = np.random.default_rng(3)
    field = rng.random(40)
    # volume-preserving-ish: mean drifts only by boundary asymmetry
    assert abs((W @ field).mean() - field.mean()) < 0.02


# ---- OC update -------------------------------------------------------------

def test_oc_update_meets_volume_target_and_stays_in_bounds():
    model, params, rng = _model(n=40, seed=2)
    rho = 0.2 + 0.6 * rng.random(40)
    vols = 0.5 + rng.random(40)                              # non-uniform volumes
    _, energy = model.solve(rho)
    dc = compliance_sensitivity(rho, energy, params)
    vf = 0.4

    new = oc_update(rho, dc, vols, vols, vf, move=0.3, rho_min=1e-3)

    achieved = (vols * new).sum() / vols.sum()
    assert abs(achieved - vf) < 1e-6                         # bisection found lambda
    assert new.min() >= 1e-3 - 1e-12 and new.max() <= 1.0 + 1e-12


def test_oc_update_reduces_objective_under_volume_constraint():
    """From a uniform, volume-feasible start, one OC step must lower the synthetic
    compliance while keeping the volume at the target."""
    model, params, _ = _model(n=50, seed=4)
    vols = np.ones(50)
    vf = 0.4
    x = np.full(50, vf)                                      # uniform, feasible

    C0, energy = model.solve(x)
    dc = compliance_sensitivity(x, energy, params)
    x1 = oc_update(x, dc, vols, vols, vf)
    C1, _ = model.solve(x1)

    assert C1 < C0                                           # objective decreased
    assert abs((vols * x1).sum() / vols.sum() - vf) < 1e-6  # constraint held


def test_oc_update_respects_move_limit():
    model, params, rng = _model(n=20, seed=5)
    rho = np.full(20, 0.5)
    _, energy = model.solve(rho)
    dc = compliance_sensitivity(rho, energy, params)
    move = 0.1
    new = oc_update(rho, dc, np.ones(20), np.ones(20), 0.5, move=move)
    assert np.all(np.abs(new - rho) <= move + 1e-9)         # no element jumps > move


# ---- Heaviside projection --------------------------------------------------

def test_heaviside_projection_maps_unit_interval_and_fixes_eta():
    eta = 0.5
    pts = np.array([0.0, eta, 1.0])
    proj = heaviside_projection(pts, beta=10.0, eta=eta)
    assert np.allclose(proj, [0.0, eta, 1.0])               # endpoints + eta fixed
    rho = np.linspace(0, 1, 100)
    out = heaviside_projection(rho, beta=10.0, eta=eta)
    assert out.min() >= -1e-12 and out.max() <= 1.0 + 1e-12
    assert np.all(np.diff(out) >= -1e-12)                   # monotonic non-decreasing
    # analytic derivative matches finite difference
    h = 1e-6
    fd = (heaviside_projection(rho + h, 10.0, eta)
          - heaviside_projection(rho - h, 10.0, eta)) / (2 * h)
    assert np.allclose(heaviside_deriv(rho, 10.0, eta), fd, atol=1e-3)


def test_heaviside_projection_reduces_grayness_as_beta_grows():
    rng = np.random.default_rng(6)
    rho = rng.random(200)                                   # lots of intermediate gray
    g0 = grayness(rho)
    g_mid = grayness(heaviside_projection(rho, beta=4.0))
    g_hi = grayness(heaviside_projection(rho, beta=16.0))
    assert g_mid < g0                                       # projection sharpens
    assert g_hi < g_mid                                     # stronger beta -> sharper
    assert g_hi < 0.3                                       # well on the way to 0/1


# ---- end-to-end ------------------------------------------------------------

def test_optimize_converges_to_feasible_low_compliance_design():
    """The full filter -> project -> OC loop drives compliance down, lands on the
    volume target, keeps densities in [0,1], and (with Heaviside) ends near 0/1."""
    nx, ny = 16, 8
    xs, ys = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, indexing="xy")
    cent = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(nx * ny)])
    cx, cy = nx * 0.3, ny * 0.5
    r2 = (cent[:, 0] - cx) ** 2 + (cent[:, 1] - cy) ** 2
    geom = 0.1 + np.exp(-r2 / (2.0 * (0.2 * nx) ** 2))
    vols = np.ones(nx * ny)
    model = SyntheticCompliance(geom=geom, load=1.0, params=SimpParams(penal=3.0))

    res = optimize(model, cent, vols, vol_frac=0.4, n_iter=80,
                   filter_radius=1.5, beta=1.0, beta_max=8.0,
                   beta_grow_every=20, move=0.2)
    h = res.history

    assert h["compliance"][-1] < 0.5 * h["compliance"][0]   # at least halved
    assert abs(h["volume_fraction"][-1] - 0.4) < 1e-2       # on the volume target
    assert res.rho.min() >= -1e-9 and res.rho.max() <= 1.0 + 1e-9
    assert h["grayness"][-1] < 0.2                          # Heaviside drove it ~0/1
