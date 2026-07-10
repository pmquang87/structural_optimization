"""EXPERIMENTAL — MFSE + Kriging non-gradient topology optimization prototype.

    >>> THIS MODULE IS A RESEARCH-SPIKE PROTOTYPE. <<<
    It is deliberately NOT wired into the loop: ``loop.build_optimizer`` only
    knows ``beso`` / ``levelset`` / ``tobs`` / ``hca`` / ``saip``, and this file is not
    imported by any of them. It does NOT talk to OpenRadioss. It exercises the
    material-field series expansion (MFSE) and the sequential Kriging optimiser
    against synthetic objectives so both halves can be validated hermetically
    (see ``tests/test_mfse.py``), exactly like the SIMP spike (``oropt/simp.py``).

Why this direction matters for oropt
------------------------------------
Every optimiser currently in the loop spends one ~13-minute OpenRadioss solve
per design update and needs a per-element *field* (energy density) to rank
elements. MFSE + Kriging attacks the actual bottleneck — the number of solves —
from the other side:

* it re-parameterises the per-element 0/1 topology (575k TET4 on the real mesh)
  into **<= 50-200 correlated material-field coefficients**, mesh-independent
  in count, so the design space is small enough for a Gaussian-process
  surrogate;
* the surrogate consumes **only scalar outputs** (compliance, peak von-Mises,
  displacement) — *no field sensitivity at all*, sidestepping the entire
  "OpenRadioss exposes no design gradients" problem instead of working around
  it with energy-as-sensitivity;
* the sample appetite of the surrogate can be paid for by oropt's validated
  **35x cheaper tied-linear proxy** (``oropt.fastmode``, ~3 min/solve, ~14%
  stress bias): Kriging over MFSE coefficients evaluated on fast mode, with
  periodic full nonlinear solves as high-fidelity correction (a classic
  multi-fidelity co-Kriging setup) fits a 50-150 true-solve budget.

The calibrated expectation is *modest*: SOLO's best published compliance
benchmark needed 286 true FE evaluations — 2-6x above oropt's budget — and the
DTU critical review documents how ML-for-TO speed-ups routinely fail to
transfer out of the cheap-linear-solve regime. Hence the spike discipline:
**a validated go/no-go on a coarse proxy mesh must precede any loop
integration.** This module only proves the maths offline.

The maths, half 1: material-field series expansion (Luo et al. 2020)
--------------------------------------------------------------------
Define a bounded material field ``phi(x)`` over the design domain with an
assumed Gaussian spatial correlation between element centroids::

    C(x_i, x_j) = exp(-||x_i - x_j||^2 / lc^2)          (lc = correlation length)

Collect ``C`` over the N centroids and truncate its eigendecomposition
``C v_k = lambda_k v_k`` to the k largest eigenpairs (smallest number capturing
a chosen energy fraction, e.g. 99%). The field is then the series expansion

    phi(x_i) = sum_k eta_k * sqrt(lambda_k) * v_k(x_i)                     (1)

with the coefficient vector ``eta`` as the ONLY design variable; the topology
is the level set ``phi >= 0`` (solid) / ``phi < 0`` (void). Because C's
spectrum decays fast for smooth kernels, k stays in the tens even as N grows —
the coefficient count is set by ``lc`` relative to the domain, not by the mesh.

At N = 575k the dense N x N eigenproblem is impossible (2.6 TB), so the basis
is built with the **Nystrom approximation**: pick M landmark centroids
(deterministic seeded subsample), eigendecompose the M x M landmark correlation
``C_MM = V diag(lam) V^T``, and extend to all N points through the N x M
cross-correlation ``C_NM``. Substituting the Nystrom eigenpair estimates
(``lambda_k ~ (N/M) lam_k``, ``v_k ~ sqrt(M/N) C_NM V_k / lam_k``) into (1),
the scale factors cancel to the clean form implemented here::

    phi = C_NM @ (V[:, :k] / sqrt(lam[:k])) @ eta  =  Psi @ eta            (2)

With landmarks == all points, (2) reduces exactly to (1). Since
``diag(C) = 1``, i.i.d. standard-normal ``eta`` gives phi approximately unit
pointwise variance, so coefficient bounds of about +-3 cover the useful field
range — that is the ``bounds`` handed to the optimiser below.

The maths, half 2: sequential Kriging (EGO-style) over the coefficients
-----------------------------------------------------------------------
Plain GP regression with a simple isotropic RBF kernel
``k(a, b) = sigma_f^2 exp(-||a-b||^2 / (2 ell^2)) + sigma_n^2 delta_ab``;
``ell`` is fitted by a cheap 1-D grid search on the log marginal likelihood
(anisotropic ARD and gradient-based ML fits are deliberately out of prototype
scope). The loop is textbook Jones/EGO: a seeded Latin-hypercube initial
sample, then per iteration fit the GP, maximise Expected Improvement over
seeded random candidates plus a local refinement around the best candidate,
evaluate the true black box there, refit.

Constraint handling — out of scope, with an escape hatch
--------------------------------------------------------
The paper drives the constrained problem with TWO infill criteria (an
EI-based objective infill and a feasibility-driven constraint infill). This
prototype implements neither; instead the volume constraint is imposed
*inside the evaluator* by :func:`volume_threshold`, which finds the level-set
threshold that hits a target volume fraction exactly (to within one element),
so every evaluated design is volume-feasible by construction and the GP only
ever models the objective. Stress/displacement constraints would need the
paper's second criterion or a penalty — a go/no-go question for the spike doc,
not for this file.

References
----------
* Luo, Xing, Kang: "Topology optimization using material-field series
  expansion and Kriging-based algorithm: an effective non-gradient method",
  CMAME 364:112966 (2020) — the method this file prototypes.
* Zhang et al.: DNN-assisted MFSE variant handling ~200 coefficients, CMAME
  (2022) — the scale-up path if 50 modes prove too coarse.
* Deng et al.: "Self-directed online machine learning for topology
  optimization" (SOLO), Nature Communications 12:5199 (2021) — the
  order-of-magnitude calibration: 286 true FE solves on a compliance benchmark.
* Woldseth, Aage, Baerentzen, Sigmund: "On the use of artificial neural
  networks in topology optimisation", SMO (2022), arXiv:2208.02563 — the
  calibrating caution on ML-for-TO claims.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.linalg import cho_factor, cho_solve, eigh
from scipy.spatial.distance import cdist
from scipy.special import ndtr

# ---------------------------------------------------------------------------
# MFSE basis (Nystrom-approximated Karhunen-Loeve expansion of the correlation)
# ---------------------------------------------------------------------------


@dataclass
class MfseBasis:
    """Truncated series-expansion basis of the Gaussian correlation field.

    Built by :meth:`fit`; ``psi`` is the precomputed N x k matrix of eq. (2) in
    the module docstring, so a design is just ``field = psi @ coeffs`` and the
    topology is ``field >= 0``. ``energy_captured`` is the landmark-spectrum
    energy fraction of the retained modes (the truncation quality metric).
    """
    psi: np.ndarray                 # (N, k) expansion matrix Psi
    eigvals: np.ndarray             # (k,) retained landmark eigenvalues
    landmark_idx: np.ndarray        # (M,) centroid indices used as landmarks
    lc: float
    energy_captured: float

    @classmethod
    def fit(cls, centroids: np.ndarray, lc: float,
            n_modes: Optional[int] = None, n_landmarks: Optional[int] = None,
            seed: int = 0, energy: float = 0.99) -> "MfseBasis":
        """Build the basis over ``centroids`` (N, dim) with correlation length ``lc``.

        ``n_landmarks`` caps the dense eigenproblem (None or >= N means exact,
        no Nystrom); landmarks are a deterministic seeded subsample of the
        centroids. ``n_modes`` fixes the truncation; if None, the smallest
        count capturing ``energy`` of the landmark spectrum is kept.
        """
        centroids = np.asarray(centroids, dtype=float)
        n = centroids.shape[0]
        if lc <= 0:
            raise ValueError("correlation length lc must be > 0")
        rng = np.random.default_rng(seed)
        if n_landmarks is None or n_landmarks >= n:
            idx = np.arange(n)
        else:
            idx = np.sort(rng.permutation(n)[:n_landmarks])
        land = centroids[idx]
        m = idx.size

        c_mm = np.exp(-cdist(land, land, "sqeuclidean") / lc**2)
        c_mm[np.diag_indices(m)] += 1e-10       # jitter: keep eigh numerically PSD
        lam, vec = eigh(c_mm)
        lam = lam[::-1]; vec = vec[:, ::-1]     # descending eigenvalues
        lam = np.maximum(lam, 0.0)

        total = float(lam.sum())
        if n_modes is None:
            frac = np.cumsum(lam) / total
            k = int(np.searchsorted(frac, energy) + 1)
        else:
            k = min(int(n_modes), m)
        # drop numerically-zero modes: 1/sqrt(lam) below must stay finite
        k = min(k, int(np.count_nonzero(lam > 1e-12 * lam[0])))

        c_nm = np.exp(-cdist(centroids, land, "sqeuclidean") / lc**2)
        psi = c_nm @ (vec[:, :k] / np.sqrt(lam[:k]))
        return cls(psi=psi, eigvals=lam[:k].copy(), landmark_idx=idx, lc=float(lc),
                   energy_captured=float(lam[:k].sum() / total))

    @property
    def n_modes(self) -> int:
        return int(self.eigvals.size)

    @property
    def n_points(self) -> int:
        return int(self.psi.shape[0])

    def field(self, coeffs: np.ndarray) -> np.ndarray:
        """Material field phi at every centroid for coefficient vector ``coeffs``."""
        coeffs = np.asarray(coeffs, dtype=float)
        return self.psi @ coeffs

    def mask(self, coeffs: np.ndarray) -> np.ndarray:
        """Solid/void topology: ``phi >= 0`` per centroid (bool, shape (N,))."""
        return self.field(coeffs) >= 0.0


def volume_threshold(field: np.ndarray, volumes: np.ndarray,
                     target_vf: float) -> float:
    """Level-set threshold ``t`` so ``field >= t`` hits ``target_vf`` volume.

    Sorts elements by descending field value and cuts the cumulative volume at
    the target; the returned threshold sits midway between the last kept and
    first dropped field values, so the mask volume matches the target to within
    one element volume (exact for distinct field values, which random MFSE
    fields have almost surely). This is how an evaluator imposes volume
    feasibility *before* solving, keeping the constraint out of the surrogate.
    """
    field = np.asarray(field, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    order = np.argsort(field)[::-1]
    cumv = np.cumsum(volumes[order])
    target = float(np.clip(target_vf, 0.0, 1.0)) * cumv[-1]
    n_keep = int(np.argmin(np.abs(cumv - target))) + 1
    if n_keep >= field.size:
        return float(field[order[-1]])          # keep everything (>= is inclusive)
    return float(0.5 * (field[order[n_keep - 1]] + field[order[n_keep]]))


# ---------------------------------------------------------------------------
# Plain GP regression (numpy/scipy only — no sklearn)
# ---------------------------------------------------------------------------


@dataclass
class GpModel:
    """A fitted zero-mean RBF GP on standardised targets (see :func:`gp_fit`)."""
    X: np.ndarray                   # (n, d) training inputs
    alpha: np.ndarray               # (n,) K^-1 z (z = standardised targets)
    chol: tuple                     # cho_factor of K (for the predictive variance)
    ell: float                      # RBF length scale
    sigma_f: float                  # signal std (fixed 1.0 on standardised targets)
    sigma_n: float                  # nugget std
    y_mean: float                   # target standardisation: y = y_mean + y_scale * z
    y_scale: float


def _rbf(a: np.ndarray, b: np.ndarray, ell: float, sigma_f: float) -> np.ndarray:
    return sigma_f**2 * np.exp(-cdist(a, b, "sqeuclidean") / (2.0 * ell**2))


def gp_fit(X: np.ndarray, y: np.ndarray, ell: Optional[float] = None,
           sigma_n: float = 1e-4) -> GpModel:
    """Fit the GP; if ``ell`` is None, 1-D grid-search it by marginal likelihood.

    Targets are standardised to zero mean / unit std so ``sigma_f = 1`` is a
    sensible fixed signal amplitude and ``sigma_n`` is relative to the data
    spread. The grid spans 0.05-3x the median pairwise training distance —
    crude, but the prototype only needs the length scale to the right order.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    y = np.asarray(y, dtype=float).ravel()
    y_mean = float(y.mean())
    y_scale = float(y.std())
    if y_scale <= 0.0:
        y_scale = 1.0                           # constant targets: predict the mean
    z = (y - y_mean) / y_scale

    if ell is None:
        d = cdist(X, X)
        med = float(np.median(d[np.triu_indices_from(d, k=1)])) if X.shape[0] > 1 else 1.0
        med = med if med > 0 else 1.0
        candidates = med * np.geomspace(0.05, 3.0, 9)
    else:
        candidates = np.array([float(ell)])

    best: Optional[tuple[float, float, np.ndarray, tuple]] = None
    for ell_c in candidates:
        K = _rbf(X, X, ell_c, 1.0)
        K[np.diag_indices_from(K)] += sigma_n**2
        try:
            chol = cho_factor(K, lower=True)
        except np.linalg.LinAlgError:
            continue
        alpha = cho_solve(chol, z)
        logdet = 2.0 * float(np.sum(np.log(np.diag(chol[0]))))
        lml = -0.5 * float(z @ alpha) - 0.5 * logdet
        if best is None or lml > best[0]:
            best = (lml, float(ell_c), alpha, chol)
    if best is None:
        raise np.linalg.LinAlgError("GP covariance not positive definite for any ell")
    _, ell_fit, alpha, chol = best
    return GpModel(X=X, alpha=alpha, chol=chol, ell=ell_fit, sigma_f=1.0,
                   sigma_n=float(sigma_n), y_mean=y_mean, y_scale=y_scale)


def gp_predict(model: GpModel, Xs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Posterior mean and std at query points ``Xs`` (n_query, d), original units."""
    Xs = np.atleast_2d(np.asarray(Xs, dtype=float))
    ks = _rbf(Xs, model.X, model.ell, model.sigma_f)
    mean_z = ks @ model.alpha
    v = cho_solve(model.chol, ks.T)
    var_z = model.sigma_f**2 - np.einsum("ij,ji->i", ks, v)
    std_z = np.sqrt(np.maximum(var_z, 0.0))
    return model.y_mean + model.y_scale * mean_z, model.y_scale * std_z


def expected_improvement(mean: np.ndarray, std: np.ndarray,
                         y_best: float) -> np.ndarray:
    """EI for *minimisation*: E[max(y_best - Y, 0)] under the GP posterior.

    Where the posterior std collapses (at training points) EI degenerates to
    the deterministic improvement max(y_best - mean, 0) — zero at the incumbent.
    """
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    imp = y_best - mean
    with np.errstate(divide="ignore", invalid="ignore"):
        zscore = np.where(std > 0, imp / np.where(std > 0, std, 1.0), 0.0)
    pdf = np.exp(-0.5 * zscore**2) / np.sqrt(2.0 * np.pi)
    ei = np.where(std > 1e-12, imp * ndtr(zscore) + std * pdf, np.maximum(imp, 0.0))
    return np.maximum(ei, 0.0)


# ---------------------------------------------------------------------------
# Sequential Kriging minimisation (EGO with random-candidate EI infill)
# ---------------------------------------------------------------------------


@dataclass
class KrigingResult:
    x_best: np.ndarray              # best design found
    y_best: float                   # its objective value
    X: np.ndarray                   # (n_evals, dim) all evaluated designs
    y: np.ndarray                   # (n_evals,) their objective values
    history: dict                   # per-evaluation "best_x" (n_evals, dim), "best_y"


def _lhs(rng: np.random.Generator, n: int, dim: int) -> np.ndarray:
    """Latin-hypercube sample in [0,1]^dim: one point per stratum per dimension."""
    u = np.empty((n, dim))
    for j in range(dim):
        u[:, j] = (rng.permutation(n) + rng.random(n)) / n
    return u


def kriging_minimize(evaluator: Callable[[np.ndarray], float], dim: int,
                     bounds: tuple[float, float] = (-3.0, 3.0), *,
                     n_init: int = 10, n_iter: int = 30, seed: int = 0,
                     n_candidates: int = 2048, ell: Optional[float] = None,
                     sigma_n: float = 1e-4) -> KrigingResult:
    """Minimise a scalar black box over ``[lo, hi]^dim`` with EI infill.

    ``evaluator(x)`` is the expensive truth (13-min OpenRadioss solve or 3-min
    fastmode proxy in a real run; a synthetic objective in the tests) and is
    called exactly ``n_init + n_iter`` times. Per iteration: refit the GP on
    everything seen, score EI on ``n_candidates`` seeded-uniform candidates,
    then locally refine the EI argmax with shrinking Gaussian perturbations
    (cheap stand-in for a multistart EI optimiser), and evaluate the winner.
    Fully deterministic given ``seed`` and a deterministic evaluator.
    """
    lo = np.broadcast_to(np.asarray(bounds[0], dtype=float), (dim,)).copy()
    hi = np.broadcast_to(np.asarray(bounds[1], dtype=float), (dim,)).copy()
    rng = np.random.default_rng(seed)

    X = lo + _lhs(rng, n_init, dim) * (hi - lo)
    y = np.array([float(evaluator(x)) for x in X])
    best_x_hist = [X[int(np.argmin(y[: i + 1]))] for i in range(n_init)]
    best_y_hist = [float(np.min(y[: i + 1])) for i in range(n_init)]

    for _ in range(n_iter):
        model = gp_fit(X, y, ell=ell, sigma_n=sigma_n)
        y_best = float(y.min())

        cand = lo + rng.random((n_candidates, dim)) * (hi - lo)
        mean, std = gp_predict(model, cand)
        ei = expected_improvement(mean, std, y_best)
        x_new = cand[int(np.argmax(ei))]
        ei_new = float(ei.max())
        for scale in (0.10, 0.03, 0.01):        # local refinement of the EI winner
            local = x_new + rng.standard_normal((256, dim)) * scale * (hi - lo)
            local = np.clip(local, lo, hi)
            m_l, s_l = gp_predict(model, local)
            ei_l = expected_improvement(m_l, s_l, y_best)
            if float(ei_l.max()) > ei_new:
                x_new = local[int(np.argmax(ei_l))]
                ei_new = float(ei_l.max())

        X = np.vstack([X, x_new])
        y = np.append(y, float(evaluator(x_new)))
        i_best = int(np.argmin(y))
        best_x_hist.append(X[i_best])
        best_y_hist.append(float(y[i_best]))

    i_best = int(np.argmin(y))
    return KrigingResult(x_best=X[i_best].copy(), y_best=float(y[i_best]), X=X, y=y,
                         history={"best_x": np.array(best_x_hist),
                                  "best_y": np.array(best_y_hist)})


# ---------------------------------------------------------------------------
# Self-contained demonstration (run: python -m oropt.mfse)
# ---------------------------------------------------------------------------


def _demo() -> None:                            # pragma: no cover - demo only
    """Recover a known MFSE topology on a 2D grid from scalar mismatches only."""
    nx, ny = 20, 12
    xs, ys = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, indexing="xy")
    cent = np.column_stack([xs.ravel(), ys.ravel()])
    basis = MfseBasis.fit(cent, lc=4.0, n_modes=10, n_landmarks=80, seed=0)
    vols = np.ones(cent.shape[0])

    rng = np.random.default_rng(42)
    eta_true = rng.standard_normal(basis.n_modes)
    target = basis.mask(eta_true)
    vf = float(vols[target].sum() / vols.sum())

    def objective(eta: np.ndarray) -> float:
        f = basis.field(eta)
        mask = f >= volume_threshold(f, vols, vf)
        return float(np.count_nonzero(mask != target))

    res = kriging_minimize(objective, basis.n_modes, (-3.0, 3.0),
                           n_init=12, n_iter=48, seed=1)
    h = res.history["best_y"]
    print(f"modes / landmarks     : {basis.n_modes} / {basis.landmark_idx.size}"
          f"  (energy {basis.energy_captured:.3f})")
    print(f"target volume fraction: {vf:.3f}")
    print(f"Hamming best-of-init  : {h[11]:.0f} / {cent.shape[0]} elements")
    print(f"Hamming after infill  : {res.y_best:.0f} / {cent.shape[0]} elements")
    f_best = basis.field(res.x_best)
    final = (f_best >= volume_threshold(f_best, vols, vf)).reshape(ny, nx)
    tgt = target.reshape(ny, nx)
    for row_f, row_t in zip(final[::-1], tgt[::-1]):
        print("".join("#" if v else "." for v in row_f), "   ",
              "".join("#" if v else "." for v in row_t))


if __name__ == "__main__":                      # pragma: no cover
    _demo()
