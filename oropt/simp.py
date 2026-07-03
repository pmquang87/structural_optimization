"""EXPERIMENTAL — density-based SIMP with an Optimality-Criteria (OC) update.

    >>> THIS MODULE IS A RESEARCH-SPIKE PROTOTYPE. <<<
    It is deliberately NOT wired into the loop: ``loop.build_optimizer`` only
    knows ``beso`` / ``levelset`` / ``tobs`` / ``hca``, and this file is not imported by any
    of them. It does NOT talk to OpenRadioss. It exercises the SIMP/OC mathematics
    against a *synthetic* compliance model so the update, the bisection and the
    projection can be validated hermetically (see ``tests/test_simp.py``). The
    written findings and the go/no-go call live in ``docs/simp_spike.md``.

How SIMP relates to the existing optimiser seam
-----------------------------------------------
BESO, level-set and TOBS all implement the same *binary* seam: a class built as
``Opt(mesh, cfg, protected, anchor)`` whose ``update(alive_mask, sens, target_vf)``
maps a boolean alive/void mask to a new boolean mask, after which the loop writes
the deck by *omitting deleted element cards only* (``deck.Deck.write``). SIMP does
not fit that seam: its design variable is a continuous density ``rho_e in [0,1]``
that the solver must actually *see* as a per-element modulus ``E_e = E0*rho_e^p``.
So a real SIMP optimiser would need a wider seam (continuous state through
loop/status/checkpoint) and a brand-new deck path. This prototype only proves the
optimiser maths; the deck/conditioning questions are argued in the spike doc.

Why a prototype is even possible here
-------------------------------------
OpenRadioss exposes no design sensitivities, which is the usual death-knell for
SIMP. But the classic compliance sensitivity needs nothing OpenRadioss withholds.
For a structure whose stiffness is ``K(rho) = sum_e K_e(rho_e)`` with each element
matrix scaling linearly with its modulus, ``K_e(rho_e) = (E(rho_e)/E_ref) K_e^0``,
the exact compliance gradient is

    dC/drho_e = -U^T (dK_e/drho_e) U
              = -(E'(rho_e)/E(rho_e)) * (U^T K_e U)
              = -(E'(rho_e)/E(rho_e)) * 2 * U_e            (1)

where ``U_e`` is the *element strain energy* — exactly the per-element internal
energy OpenRadioss already writes to ``/ANIM/ELEM/ENER`` and
``oropt.results.Results.energy`` already reads. So the *sensitivity* is free, just
as it is for BESO. The hard parts are (a) emitting a per-element modulus field into
the deck and (b) conditioning of the implicit solve with soft elements — the
subjects of ``docs/simp_spike.md``.

For several load cases the sensitivity is just the weighted sum of the per-case
energies (eq. 1 is linear in ``U_e``), mirroring ``beso.combine_sensitivity``:
``dC/drho_e = -(E'/E) * 2 * sum_lc w_lc * U_e^lc``.

Note on the energy used in (1): ``U_e`` is the *penalised* (as-built, measured)
element strain energy. A common textbook form writes ``dC/drho = -p*rho^(p-1)*2*u``
which is identical but uses the *unpenalised* reference energy ``u = U_e/rho^p``
and assumes ``Emin=0, E0=1``. We keep the measured-energy form (1) because that is
what a black-box solver actually reports.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree

# ---------------------------------------------------------------------------
# Parameters
# ---------------------------------------------------------------------------


@dataclass
class SimpParams:
    """SIMP material-interpolation and projection knobs."""
    E0: float = 1.0          # solid Young's modulus
    Emin: float = 1e-9       # void modulus (>0 keeps the tangent non-singular)
    penal: float = 3.0       # SIMP penalisation exponent p
    proj_eta: float = 0.5    # Heaviside projection threshold eta


# ---------------------------------------------------------------------------
# SIMP material interpolation
# ---------------------------------------------------------------------------


def simp_modulus(rho: np.ndarray, params: SimpParams) -> np.ndarray:
    """Modified-SIMP Young's modulus  E(rho) = Emin + rho^p (E0 - Emin)."""
    rho = np.asarray(rho, dtype=float)
    return params.Emin + np.power(rho, params.penal) * (params.E0 - params.Emin)


def simp_modulus_deriv(rho: np.ndarray, params: SimpParams) -> np.ndarray:
    """dE/drho = p * rho^(p-1) * (E0 - Emin)."""
    rho = np.asarray(rho, dtype=float)
    p = params.penal
    # rho**(p-1) with a guard so p<1 never blows up at rho==0 (here p>=1 anyway)
    return p * np.power(np.maximum(rho, 0.0), p - 1.0) * (params.E0 - params.Emin)


def compliance_sensitivity(rho: np.ndarray, energy: np.ndarray,
                           params: SimpParams) -> np.ndarray:
    """Exact compliance gradient from the measured element strain energy, eq. (1).

    ``dC/drho_e = -(E'(rho_e)/E(rho_e)) * 2 * U_e``  (always <= 0: more material
    lowers compliance). ``energy`` is the per-element strain energy — in a real
    run this is ``Results.energy`` straight out of OpenRadioss (for multiple load
    cases, the weighted sum of the per-case energies); here it comes from the
    synthetic model below. This is the single function that would couple SIMP to
    OpenRadioss, and it needs no adjoint and no finite differencing.
    """
    rho = np.asarray(rho, dtype=float)
    energy = np.asarray(energy, dtype=float)
    E = simp_modulus(rho, params)
    dE = simp_modulus_deriv(rho, params)
    return -(dE / E) * 2.0 * energy


# ---------------------------------------------------------------------------
# Density filter (mirrors oropt.mesh.Mesh.filter_matrix)
# ---------------------------------------------------------------------------


def density_filter(centroids: np.ndarray, radius: float) -> sparse.csr_matrix:
    """Row-normalised linear ("hat") filter over element centroids.

    Identical construction to :meth:`oropt.mesh.Mesh.filter_matrix` — duplicated
    here only so the prototype is self-contained and runnable on a synthetic grid
    without building a full :class:`~oropt.mesh.Mesh`. A real SIMP optimiser would
    reuse ``Mesh.filter_matrix`` directly. ``radius <= 0`` is the identity (no
    filtering). ``filtered = W @ raw``; the chain rule for filtered design
    variables uses ``W.T`` (W is row-normalised, hence not symmetric).
    """
    centroids = np.asarray(centroids, dtype=float)
    n = centroids.shape[0]
    if radius <= 0:
        return sparse.identity(n, format="csr")
    tree = cKDTree(centroids)
    dmat = tree.sparse_distance_matrix(tree, radius, output_type="coo_matrix")
    w = np.maximum(0.0, 1.0 - dmat.data / radius)
    W = sparse.coo_matrix((w, (dmat.row, dmat.col)), shape=(n, n)).tocsr()
    W.setdiag(1.0)                                      # ensure self-weight
    rowsum = np.asarray(W.sum(axis=1)).ravel()
    rowsum[rowsum == 0] = 1.0
    return sparse.diags(1.0 / rowsum) @ W


# ---------------------------------------------------------------------------
# Heaviside projection (push grayscale -> 0/1 for manufacturability)
# ---------------------------------------------------------------------------


def heaviside_projection(rho: np.ndarray, beta: float, eta: float = 0.5) -> np.ndarray:
    """Smooth-Heaviside (Wang-Lazarov-Sigmund) projection of a filtered field.

    ``rho_bar = (tanh(beta*eta) + tanh(beta*(rho-eta)))
                / (tanh(beta*eta) + tanh(beta*(1-eta)))``

    ``beta -> 0`` is the identity; larger ``beta`` sharpens the step at ``eta``,
    driving intermediate densities toward 0/1. Maps [0,1] -> [0,1] monotonically
    and fixes ``rho == eta`` to ``eta``.
    """
    rho = np.asarray(rho, dtype=float)
    if beta <= 0:
        return rho.copy()
    num = np.tanh(beta * eta) + np.tanh(beta * (rho - eta))
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return num / den


def heaviside_deriv(rho: np.ndarray, beta: float, eta: float = 0.5) -> np.ndarray:
    """d(rho_bar)/d(rho) for the projection above (for the sensitivity chain)."""
    rho = np.asarray(rho, dtype=float)
    if beta <= 0:
        return np.ones_like(rho)
    den = np.tanh(beta * eta) + np.tanh(beta * (1.0 - eta))
    return beta * (1.0 - np.tanh(beta * (rho - eta)) ** 2) / den


def grayness(rho: np.ndarray) -> float:
    """Mean of ``4*rho*(1-rho)``: 0.0 for a pure 0/1 field, 1.0 for all-0.5.

    A scalar manufacturability proxy — how far the design is from black-and-white.
    """
    rho = np.asarray(rho, dtype=float)
    return float(np.mean(4.0 * rho * (1.0 - rho)))


# ---------------------------------------------------------------------------
# Optimality-Criteria update (bisection on the Lagrange multiplier)
# ---------------------------------------------------------------------------


def oc_update(rho: np.ndarray, dc: np.ndarray, dv: np.ndarray,
              volumes: np.ndarray, vol_frac: float, *,
              move: float = 0.2, eta: float = 0.5,
              rho_min: float = 1e-3, rho_max: float = 1.0,
              physical: Optional[Callable[[np.ndarray], np.ndarray]] = None,
              bisect_tol: float = 1e-8, max_bisect: int = 200) -> np.ndarray:
    """One OC step: ``rho_new = clip(rho * (-dc/(lambda*dv))^eta, move/bounds)``.

    ``lambda`` (the volume Lagrange multiplier) is found by bisection so the
    updated volume hits ``vol_frac * sum(volumes)``. ``dc`` is dC/d(design-var)
    (<= 0), ``dv`` is dV/d(design-var) (>= 0). ``physical`` maps the design
    variable to the density whose volume is constrained (filter+projection); the
    volume check uses the *actual* projected volume, so the constraint is met on
    the physical field regardless of how the sensitivities were linearised.
    Defaults to the identity (constrain the variable directly). This is the
    textbook OC of Bendsoe & Sigmund / the 88-line code, generalised to per-element
    volumes and an arbitrary physical-density map.
    """
    rho = np.asarray(rho, dtype=float)
    dc = np.asarray(dc, dtype=float)
    dv = np.asarray(dv, dtype=float)
    volumes = np.asarray(volumes, dtype=float)
    phys = physical if physical is not None else (lambda r: r)

    target_V = vol_frac * float(volumes.sum())
    lower = np.maximum(rho_min, rho - move)
    upper = np.minimum(rho_max, rho + move)
    # base of the OC scaling factor B_e = -dc/(lambda*dv); clamp >= 0 for safety
    base = np.maximum(0.0, -dc / (dv + 1e-30))

    def candidate(lmid: float) -> np.ndarray:
        scale = np.power(base / lmid, eta)
        return np.clip(rho * scale, lower, upper)

    def volume_at(lmid: float) -> float:
        return float((volumes * phys(candidate(lmid))).sum())

    # Auto-bracket: volume is monotonically decreasing in lambda.
    l1, l2 = 1e-12, 1.0
    grow = 0
    while volume_at(l2) > target_V and grow < 200:
        l2 *= 2.0
        grow += 1
    shrink = 0
    while volume_at(l1) < target_V and shrink < 200:
        l1 *= 0.5
        shrink += 1

    for _ in range(max_bisect):
        if (l2 - l1) <= bisect_tol * (l1 + l2):
            break
        lmid = 0.5 * (l1 + l2)
        if volume_at(lmid) > target_V:
            l1 = lmid                       # too much volume -> raise lambda
        else:
            l2 = lmid
    return candidate(0.5 * (l1 + l2))


# ---------------------------------------------------------------------------
# Synthetic compliance model (stands in for the OpenRadioss solve)
# ---------------------------------------------------------------------------


@dataclass
class SyntheticCompliance:
    """A hermetic, differentiable stand-in for an OpenRadioss compliance solve.

    The elements are linear springs sharing a common load ``P`` through *parallel*
    load paths (the simplest model in which removing low-energy material is cheap,
    unlike a series path where every element is critical). With per-element
    geometric factor ``g_e > 0`` and ``k_e = E(rho_e) * g_e``:

        total stiffness  K = sum_e k_e
        displacement     d = P / K
        compliance       C = P * d = P^2 / K          (external work)
        strain energy    U_e = 0.5 * k_e * d^2         (what OR reports per element)

    Everything is exact and differentiable, so :func:`compliance_sensitivity` can
    be cross-checked against finite differences (it is, in the tests). Because the
    penalised modulus ``rho^p`` is convex, under a volume cap the optimum
    concentrates material on the high-``g_e`` elements and starves the rest — the
    qualitative behaviour of real compliance topology optimisation.
    """
    geom: np.ndarray               # per-element geometric stiffness factor g_e (>0)
    load: float = 1.0              # shared applied load P
    params: SimpParams = field(default_factory=SimpParams)

    def solve(self, rho: np.ndarray) -> tuple[float, np.ndarray]:
        """Return ``(compliance, per-element strain energy)`` at density ``rho``."""
        rho = np.asarray(rho, dtype=float)
        k = simp_modulus(rho, self.params) * self.geom
        K = float(k.sum())
        d = self.load / K
        energy = 0.5 * k * d ** 2
        compliance = self.load * d
        return compliance, energy


# ---------------------------------------------------------------------------
# Driver: full filter -> project -> solve -> OC loop
# ---------------------------------------------------------------------------


@dataclass
class SimpResult:
    rho: np.ndarray              # final physical (filtered+projected) density
    x: np.ndarray                # final design variable
    history: dict                # per-iteration scalars
    iterations: int


def optimize(model: SyntheticCompliance, centroids: np.ndarray,
             volumes: np.ndarray, vol_frac: float, *,
             n_iter: int = 80, filter_radius: float = 0.0,
             beta: float = 0.0, beta_max: float = 0.0, beta_grow_every: int = 20,
             move: float = 0.2, oc_eta: float = 0.5, rho_min: float = 1e-3,
             x_init: Optional[np.ndarray] = None, tol: float = 1e-3) -> SimpResult:
    """Run the SIMP/OC loop against ``model`` and return the converged design.

    Pipeline each iteration: ``x -> rho_tilde = W@x -> rho_bar = H(rho_tilde) ->``
    solve for energies -> sensitivities (eq. 1) -> chain through the projection
    and filter (``W.T``) -> OC update. ``beta``/``beta_max`` enable optional
    Heaviside continuation (doubling ``beta`` every ``beta_grow_every`` iters up to
    ``beta_max``) to sharpen the design toward 0/1.
    """
    n = volumes.size
    W = density_filter(centroids, filter_radius)
    x = np.full(n, float(vol_frac)) if x_init is None else np.array(x_init, dtype=float)
    params = model.params
    beta_cur = float(beta)

    history: dict[str, list] = {"compliance": [], "volume_fraction": [],
                                "grayness": [], "change": [], "beta": []}
    it = 0
    for it in range(1, n_iter + 1):
        if beta_max > beta_cur and it > 1 and (it - 1) % beta_grow_every == 0:
            beta_cur = min(beta_max, beta_cur * 2.0 if beta_cur > 0 else 1.0)

        rho_tilde = W @ x
        rho_bar = heaviside_projection(rho_tilde, beta_cur, params.proj_eta)
        compliance, energy = model.solve(rho_bar)

        dc_dbar = compliance_sensitivity(rho_bar, energy, params)
        dv_dbar = volumes
        dproj = heaviside_deriv(rho_tilde, beta_cur, params.proj_eta)
        dc_dx = W.T @ (dc_dbar * dproj)
        dv_dx = W.T @ (dv_dbar * dproj)

        def phys(xx: np.ndarray) -> np.ndarray:
            return heaviside_projection(W @ xx, beta_cur, params.proj_eta)

        x_new = oc_update(x, dc_dx, dv_dx, volumes, vol_frac,
                          move=move, eta=oc_eta, rho_min=rho_min, physical=phys)
        change = float(np.max(np.abs(x_new - x)))
        x = x_new

        history["compliance"].append(compliance)
        history["volume_fraction"].append(float((volumes * rho_bar).sum() / volumes.sum()))
        history["grayness"].append(grayness(rho_bar))
        history["change"].append(change)
        history["beta"].append(beta_cur)

        if change < tol and beta_cur >= beta_max:
            break

    rho_final = heaviside_projection(W @ x, beta_cur, params.proj_eta)
    return SimpResult(rho=rho_final, x=x, history=history, iterations=it)


# ---------------------------------------------------------------------------
# Self-contained demonstration (run: python -m oropt.simp)
# ---------------------------------------------------------------------------


def _demo_problem(nx: int = 24, ny: int = 12) -> tuple[SyntheticCompliance, np.ndarray, np.ndarray]:
    """A tiny 2D grid with a smooth, localised stiffness-importance field (a bump).

    ``g_e`` peaks near the mid-left and decays away, so the optimiser should grow a
    stiff load-carrying patch there and starve the lightly-loaded corners.
    """
    xs, ys = np.meshgrid(np.arange(nx) + 0.5, np.arange(ny) + 0.5, indexing="xy")
    centroids = np.column_stack([xs.ravel(), ys.ravel(), np.zeros(nx * ny)])
    # importance bump centred at (nx*0.25, ny*0.5)
    cx, cy = nx * 0.25, ny * 0.5
    r2 = (centroids[:, 0] - cx) ** 2 + (centroids[:, 1] - cy) ** 2
    geom = 0.1 + np.exp(-r2 / (2.0 * (0.18 * nx) ** 2))
    volumes = np.ones(nx * ny)
    model = SyntheticCompliance(geom=geom, load=1.0, params=SimpParams(penal=3.0))
    return model, centroids, volumes


def _ascii_density(rho: np.ndarray, nx: int, ny: int) -> str:
    chars = " .:-=+*#%@"
    grid = rho.reshape(ny, nx)
    lines = []
    for row in grid[::-1]:                              # top row last for screen
        lines.append("".join(chars[min(len(chars) - 1, int(v * len(chars)))] for v in row))
    return "\n".join(lines)


def main() -> None:                                     # pragma: no cover - demo only
    nx, ny = 24, 12
    model, centroids, volumes = _demo_problem(nx, ny)
    res = optimize(model, centroids, volumes, vol_frac=0.4,
                   n_iter=120, filter_radius=1.5,
                   beta=1.0, beta_max=8.0, beta_grow_every=25, move=0.2)
    h = res.history
    print(f"iterations            : {res.iterations}")
    print(f"compliance  start->end: {h['compliance'][0]:.4f} -> {h['compliance'][-1]:.4f}")
    print(f"volume fraction (final): {h['volume_fraction'][-1]:.3f}  (target 0.400)")
    print(f"grayness    start->end: {h['grayness'][0]:.3f} -> {h['grayness'][-1]:.3f}")
    print(f"final beta            : {h['beta'][-1]:.1f}")
    print("\nfinal density (load bump near mid-left):\n")
    print(_ascii_density(res.rho, nx, ny))


if __name__ == "__main__":                              # pragma: no cover
    main()
