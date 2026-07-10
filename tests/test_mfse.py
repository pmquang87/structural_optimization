"""Hermetic, analytic tests for the EXPERIMENTAL MFSE + Kriging prototype.

No OpenRadioss, no Deck/Mesh — everything runs against small synthetic point
sets and objectives in :mod:`oropt.mfse`. These verify the maths the spike
rests on: the Nystrom-approximated basis spans the same fields as the exact
eigendecomposition, the expansion produces spatially smooth (correlated)
fields, the level-set threshold hits a volume-fraction target to within one
element, the GP posterior interpolates its training data with a small nugget,
Expected Improvement behaves like an acquisition function should, and the full
sequential Kriging loop actually optimises a cheap black box in ~40
evaluations, deterministically.
"""
import numpy as np

from oropt.mfse import (MfseBasis, expected_improvement, gp_fit, gp_predict,
                        kriging_minimize, volume_threshold)


def _grid(nx, ny, spacing=1.0):
    xs, ys = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, indexing="xy")
    return np.column_stack([xs.ravel(), ys.ravel()]) * spacing


# ---- MFSE basis: Nystrom vs exact -------------------------------------------

def test_nystrom_matches_exact_eigendecomposition_fields():
    """Any field in the span of the exact leading subspace is reproduced by the
    Nystrom basis (least-squares projection) to small relative error — subspace
    agreement, which is what matters; individual eigenvectors may differ by
    sign/rotation within near-degenerate eigenvalue clusters, which is also why
    the Nystrom basis gets a few extra modes (the exact leading subspace should
    be *contained* in a slightly larger approximate one)."""
    cent = _grid(18, 16)                                 # 288 points: exact is cheap
    exact = MfseBasis.fit(cent, lc=5.0, n_modes=6, n_landmarks=None, seed=0)
    nystrom = MfseBasis.fit(cent, lc=5.0, n_modes=10, n_landmarks=120, seed=0)
    assert nystrom.landmark_idx.size == 120
    assert exact.landmark_idx.size == cent.shape[0]

    rng = np.random.default_rng(1)
    for _ in range(5):
        f = exact.field(rng.standard_normal(exact.n_modes))
        coeffs, *_ = np.linalg.lstsq(nystrom.psi, f, rcond=None)
        rel = np.linalg.norm(nystrom.field(coeffs) - f) / np.linalg.norm(f)
        assert rel < 0.08


def test_exact_basis_reduces_to_kl_expansion():
    """With landmarks == all points, Psi == V @ diag(sqrt(lambda)) — the classic
    truncated Karhunen-Loeve expansion — so Psi^T Psi == diag(lambda)."""
    cent = _grid(10, 8)
    basis = MfseBasis.fit(cent, lc=3.0, n_modes=6, n_landmarks=None, seed=0)
    gram = basis.psi.T @ basis.psi
    assert np.allclose(gram, np.diag(basis.eigvals), atol=1e-6)


def test_energy_fraction_mode_selection():
    """n_modes=None keeps the smallest mode count reaching the energy target."""
    cent = _grid(12, 10)
    basis = MfseBasis.fit(cent, lc=4.0, n_modes=None, seed=0, energy=0.99)
    assert basis.energy_captured >= 0.99
    assert basis.n_modes < cent.shape[0]                 # actually truncated
    smaller = MfseBasis.fit(cent, lc=4.0, n_modes=basis.n_modes - 1, seed=0)
    assert smaller.energy_captured < 0.99                # one fewer mode falls short


def test_field_smoothness_between_neighbours():
    """The Gaussian correlation makes nearby centroids nearly equal: adjacent
    grid neighbours (spacing 1, lc 5) differ by far less than the field's own
    spread, for random coefficient vectors."""
    nx, ny = 20, 15
    cent = _grid(nx, ny)
    basis = MfseBasis.fit(cent, lc=5.0, n_modes=12, n_landmarks=90, seed=2)
    rng = np.random.default_rng(3)
    for _ in range(5):
        f = basis.field(rng.standard_normal(basis.n_modes)).reshape(ny, nx)
        spread = f.std()
        assert spread > 0
        dx = np.abs(np.diff(f, axis=1)).mean()
        dy = np.abs(np.diff(f, axis=0)).mean()
        assert dx < 0.5 * spread and dy < 0.5 * spread


def test_mask_is_nonneg_level_set_of_field():
    cent = _grid(8, 8)
    basis = MfseBasis.fit(cent, lc=3.0, n_modes=5, seed=0)
    eta = np.random.default_rng(4).standard_normal(basis.n_modes)
    assert np.array_equal(basis.mask(eta), basis.field(eta) >= 0.0)


# ---- volume threshold --------------------------------------------------------

def test_volume_threshold_hits_target_within_one_element():
    rng = np.random.default_rng(5)
    for _ in range(4):
        field = rng.standard_normal(300)
        volumes = 0.5 + rng.random(300)
        total = volumes.sum()
        for vf in (0.1, 0.37, 0.5, 0.83):
            t = volume_threshold(field, volumes, vf)
            achieved = volumes[field >= t].sum()
            assert abs(achieved - vf * total) <= volumes.max() + 1e-12


def test_volume_threshold_extremes():
    field = np.array([3.0, 1.0, 2.0, 0.0])
    vols = np.ones(4)
    t_all = volume_threshold(field, vols, 1.0)
    assert np.count_nonzero(field >= t_all) == 4         # keep everything
    t_one = volume_threshold(field, vols, 0.25)
    assert np.count_nonzero(field >= t_one) == 1         # keep the single largest


# ---- GP regression + Expected Improvement ------------------------------------

def test_gp_posterior_interpolates_training_data():
    """With a small nugget the posterior mean passes through the data and the
    posterior std collapses there; between points it stays close to the smooth
    truth and the std is strictly positive."""
    X = np.linspace(0.0, 6.0, 9)[:, None]
    y = np.sin(X).ravel()
    model = gp_fit(X, y, sigma_n=1e-5)
    mean, std = gp_predict(model, X)
    assert np.allclose(mean, y, atol=1e-3)
    assert np.all(std < 1e-2)
    Xq = np.linspace(0.3, 5.7, 25)[:, None]
    mq, sq = gp_predict(model, Xq)
    assert np.max(np.abs(mq - np.sin(Xq).ravel())) < 0.05
    assert np.all(sq >= 0)


def test_expected_improvement_nonnegative_and_ranks_exploration():
    """EI >= 0 everywhere; ~0 at the noiseless incumbent best (no improvement
    possible there); larger at an unexplored far point (posterior uncertainty
    creates improvement probability)."""
    X = np.linspace(0.0, 4.0, 6)[:, None]
    y = (X.ravel() - 1.5) ** 2
    model = gp_fit(X, y, sigma_n=1e-5)
    y_best = float(y.min())

    grid = np.linspace(-2.0, 8.0, 200)[:, None]
    mean, std = gp_predict(model, grid)
    ei = expected_improvement(mean, std, y_best)
    assert np.all(ei >= 0)

    x_best = X[np.argmin(y)][None, :]
    x_far = np.array([[8.0]])                            # far outside the data
    ei_best = expected_improvement(*gp_predict(model, x_best), y_best)[0]
    ei_far = expected_improvement(*gp_predict(model, x_far), y_best)[0]
    assert ei_best < 1e-3                                # ~0 (nugget keeps it finite)
    assert ei_far > 10.0 * max(ei_best, 1e-12)


# ---- end-to-end: sequential Kriging over MFSE coefficients --------------------

def _target_matching_problem(seed=7):
    """Target-mask recovery: a known coefficient vector defines a topology on a
    small 2D grid; the objective is the Hamming distance of the volume-feasible
    thresholded field to that target — scalar-only, like a real oropt run."""
    nx, ny = 14, 10
    cent = _grid(nx, ny)
    basis = MfseBasis.fit(cent, lc=4.0, n_modes=8, n_landmarks=70, seed=0)
    vols = np.ones(cent.shape[0])
    eta_true = np.random.default_rng(seed).standard_normal(basis.n_modes)
    target = basis.mask(eta_true)
    vf = float(vols[target].sum() / vols.sum())

    def objective(eta):
        f = basis.field(eta)
        mask = f >= volume_threshold(f, vols, vf)
        return float(np.count_nonzero(mask != target))

    return objective, basis


def test_kriging_minimize_beats_initial_sample_within_40_evals():
    objective, basis = _target_matching_problem()
    n_init, n_iter = 10, 30
    res = kriging_minimize(objective, basis.n_modes, (-3.0, 3.0),
                           n_init=n_init, n_iter=n_iter, seed=11)

    assert res.X.shape == (n_init + n_iter, basis.n_modes)
    assert res.y.shape == (n_init + n_iter,)
    assert res.history["best_y"].shape == (n_init + n_iter,)
    assert res.y_best == res.history["best_y"][-1]
    assert np.all(np.diff(res.history["best_y"]) <= 0)   # best-so-far is monotone

    best_init = float(res.y[:n_init].min())
    assert res.y_best <= 0.6 * best_init                 # clear margin over init
    # the incumbent x really achieves the incumbent y
    assert objective(res.x_best) == res.y_best


def test_kriging_minimize_is_deterministic_given_seed():
    objective, basis = _target_matching_problem()
    r1 = kriging_minimize(objective, basis.n_modes, (-3.0, 3.0),
                          n_init=8, n_iter=10, seed=42)
    r2 = kriging_minimize(objective, basis.n_modes, (-3.0, 3.0),
                          n_init=8, n_iter=10, seed=42)
    assert np.array_equal(r1.X, r2.X)
    assert np.array_equal(r1.y, r2.y)
    assert np.array_equal(r1.x_best, r2.x_best)
    r3 = kriging_minimize(objective, basis.n_modes, (-3.0, 3.0),
                          n_init=8, n_iter=10, seed=43)
    assert not np.array_equal(r1.X, r3.X)                # seed actually matters
