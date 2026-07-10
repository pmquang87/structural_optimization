"""EXPERIMENTAL — topological-derivative sensitivity analysis (TDSA) for 0/1 designs.

    >>> THIS MODULE IS A RESEARCH-SPIKE PROTOTYPE. <<<
    It is deliberately NOT wired into the loop: ``loop.build_optimizer`` only knows
    ``beso`` / ``levelset`` / ``tobs`` / ``hca`` / ``saip``, and nothing imports this file. It
    does NOT talk to OpenRadioss. It implements the closed-form *topological
    derivative* (TD) of compliance for linear isotropic elasticity — offline maths
    only — so the formulas and their consequences for element ranking can be
    validated hermetically (see ``tests/test_tdsa.py``).

What the topological derivative is, and why BESO's energy proxy is not it
-------------------------------------------------------------------------
BESO ranks elements by strain-energy density and deletes the least energetic.
That heuristic is the *continuum* (SIMP-style, modulus->0) sensitivity. The
question a discrete 0/1 optimiser actually asks is different: "what does the
compliance do if this element becomes a *hole*?" The exact first-order answer is
the topological derivative — the leading coefficient of the asymptotic expansion
of the objective when an infinitesimal traction-free cavity (circular in 2D,
spherical in 3D) is nucleated at a point:

    C(hole of size rho at x) = C + |B_rho| * DT(x) + o(|B_rho|)

Sun, Liang & Cheng (SMO 65:216, 2022, doi:10.1007/s00158-022-03321-x) and Sun,
Cheng & Liang (CMAME 2024, "Topological derivative based sensitivity analysis
(TDSA) for three-dimensional discrete variable topology optimization") show that
for interior elements of a discrete 0/1 design the correct 1->0 flip sensitivity
is exactly this TD — a *linear combination of quadratic forms of the stress
components* — not the raw strain-energy density. The two coincide (up to a
constant factor) for proportional/uniaxial stress fields, so BESO often gets away
with it, but they genuinely re-order multiaxial elements: at equal energy density
a pure-shear element and a hydrostatic element have different removal costs (in
2D plane stress the shear element is *more* expensive to remove by a factor
2(1-nu)/(1+nu); in 3D it is the hydrostatic element that costs more). The tests
construct exactly these counter-examples. For interface (boundary) elements the
correct sensitivity is a shape derivative instead — out of scope here, see the
SMO 2022 paper.

The closed forms implemented (and how they were verified)
---------------------------------------------------------
All TDs below are for compliance C = f.u under fixed loads (positive: removing
loaded material always increases compliance), normalised **per unit volume (3D)
/ area (2D) of removed material**, i.e. ``dC ~= td * volume_removed``.

2D, circular hole (Garreau, Guillaume & Masmoudi, "The topological asymptotic
for PDE systems: the elasticity case", SIAM J. Control Optim. 39(6):1756-1778,
2001; same form in Amstutz & Andrä, J. Comput. Phys. 216(2):573-588, 2006, whose
algorithm GetFEM's ``demo_structural_optimization`` implements verbatim). With
in-plane Lame constants (plane stress: lambda* = 2*lambda*mu/(lambda+2*mu)):

    DT = (lam + 2mu)/(2mu(lam + mu)) * [ 4mu sigma:eps + (lam - mu) tr(sigma) tr(eps) ]

which in plane stress collapses to the remarkably Poisson-free stress-only form

    DT = (4 sigma:sigma - tr(sigma)^2) / E .                                  (2D)

3D, spherical hole (same GGM/Amstutz family; E-nu form as in Novotny &
Sokolowski, *Topological Derivatives in Shape Optimization*, Springer 2013, and
the coefficient pair used by published TD codes, e.g. IbIPP's levelset88mod.m):

    DT = (lam + 2mu)/(mu(9lam + 14mu)) * (3/4) * [ 20mu sigma:eps + (3lam - 2mu) tr(sigma) tr(eps) ]
       = 3(1-nu) / (2E(7-5nu)) * [ 10(1+nu) sigma:sigma - (1+5nu) tr(sigma)^2 ]   (3D)

Verification status: the primary PDFs (GGM 2001, Amstutz & Andrä 2006) are
paywalled and were NOT directly readable from this environment; the formulas
above were instead cross-checked against four independent secondary sources that
all agree exactly: (a) GetFEM's official Amstutz-algorithm demo (2D term-by-term
match; its 3D file carries an apparent 2D-trace copy/paste slip in the second
term, which disagrees with every other source and with the micromechanics limits
below, so it was rejected); (b) the plane-stress TD polarization tensor
A = 1/(1+nu)^2 [ -(1-6nu+nu^2)E/(1-nu)^2 I(x)I + 2E II ] quoted by a
Novotny-school paper (arXiv:2008.06153), algebraically identical to (2D);
(c) the 3D coefficient pair a1 = -3(1-nu)(1-14nu+15nu^2)E/(2(1+nu)(7-5nu)(1-2nu)^2),
a2 = 15(1-nu)E/(2(1+nu)(7-5nu)) used by two independent published codes,
numerically identical to (3D); (d) classical dilute-void micromechanics limits:
(2D) reproduces the Kirsch-solution energy of a circular hole in uniaxial
tension, dU = 3 sigma^2 * (pi a^2) / (2E), and (3D) reproduces the
Mackenzie/Eshelby dilute spherical-void softening exactly (hydrostatic:
dC per void volume = 9(1-nu)p^2/(2E); uniaxial: 3(1-nu)(9+5nu)/(2(7-5nu)) *
sigma^2/E ~= 2.0045 sigma^2/E at nu = 0.3). The tests re-derive (2D)/(3D) from
the independent Lame sigma:eps representation as a further internal lock.

Why this is offline-only today
------------------------------
The TD needs the full stress tensor per element. OpenRadioss' animation output
as parsed by ``oropt.results`` currently exposes only the von-Mises scalar
(``3DELEM_Von_Mises``) plus the element energy — and the TD is *not* a function
of von Mises alone: in 2D plane stress, E*DT = 3*vm^2 + det(sigma), so two
elements with equal von Mises but different triaxiality have different TDs.
Wiring TDSA into the loop would need the deck to request the stress tensor
(/ANIM/ELEM/TENS/STRESS), ``results.py`` to parse the six components, and a
``sensitivity: tdsa`` mode in ``beso.map_sensitivity`` feeding the existing
filter/history machinery (the TD is an "importance"-oriented field exactly like
the energy density, so the seam fits). For pure compliance the payoff is the
multiaxial re-ranking measured by :func:`rank_agreement`; the bigger prize is
that the TD framework extends to non-self-adjoint objectives (stress,
displacement, frequency) where the energy heuristic has no justification at all.
"""
from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Elastic constants
# ---------------------------------------------------------------------------


def lame_parameters(E: float, nu: float) -> tuple[float, float]:
    """3D (and plane-strain in-plane) Lame constants ``(lambda, mu)``."""
    lam = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    return lam, mu


def lame_plane_stress(E: float, nu: float) -> tuple[float, float]:
    """In-plane Lame constants ``(lambda*, mu)`` for plane stress.

    ``lambda* = 2 lambda mu / (lambda + 2 mu) = E nu / (1 - nu^2)``; using it in
    any plane-strain formula converts that formula to plane stress.
    """
    return E * nu / (1.0 - nu * nu), E / (2.0 * (1.0 + nu))


def _plane_equivalent(E: float, nu: float, plane: str) -> tuple[float, float]:
    """Constants ``(E', nu')`` that make plane-STRESS formulas serve ``plane``.

    Plane strain is plane stress with ``E' = E/(1-nu^2)``, ``nu' = nu/(1-nu)``
    (the standard equivalence; it maps lambda* -> lambda and keeps mu).
    """
    if plane == "stress":
        return E, nu
    if plane == "strain":
        return E / (1.0 - nu * nu), nu / (1.0 - nu)
    raise ValueError(f"plane must be 'stress' or 'strain', got {plane!r}")


# ---------------------------------------------------------------------------
# Voigt Hooke's law (stress <-> strain), engineering shear strains
# ---------------------------------------------------------------------------
# Conventions: 2D stress rows are [s11, s22, s12]; 3D rows are
# [s11, s22, s33, s12, s23, s31]. Strain vectors use ENGINEERING shear
# (gamma = 2*eps), so the contraction sigma:eps is a plain dot product.


def strain_from_stress_2d(stress: np.ndarray, E: float, nu: float,
                          plane: str = "stress") -> np.ndarray:
    """Inverse in-plane Hooke: ``(N,3) [s11,s22,s12] -> (N,3) [e11,e22,g12]``."""
    s = np.atleast_2d(np.asarray(stress, dtype=float))
    Ee, nue = _plane_equivalent(E, nu, plane)
    out = np.empty_like(s)
    out[:, 0] = (s[:, 0] - nue * s[:, 1]) / Ee
    out[:, 1] = (s[:, 1] - nue * s[:, 0]) / Ee
    out[:, 2] = 2.0 * (1.0 + nue) * s[:, 2] / Ee            # gamma12 = 2*eps12
    return out


def stress_from_strain_2d(strain: np.ndarray, E: float, nu: float,
                          plane: str = "stress") -> np.ndarray:
    """In-plane Hooke: ``(N,3) [e11,e22,g12] -> (N,3) [s11,s22,s12]``."""
    e = np.atleast_2d(np.asarray(strain, dtype=float))
    Ee, nue = _plane_equivalent(E, nu, plane)
    c = Ee / (1.0 - nue * nue)
    out = np.empty_like(e)
    out[:, 0] = c * (e[:, 0] + nue * e[:, 1])
    out[:, 1] = c * (e[:, 1] + nue * e[:, 0])
    out[:, 2] = Ee / (2.0 * (1.0 + nue)) * e[:, 2]          # mu * gamma12
    return out


def strain_from_stress_3d(stress: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Inverse Hooke: ``(N,6) [s11..s31] -> (N,6) [e11,e22,e33,g12,g23,g31]``."""
    s = np.atleast_2d(np.asarray(stress, dtype=float))
    tr = s[:, 0] + s[:, 1] + s[:, 2]
    out = np.empty_like(s)
    out[:, :3] = ((1.0 + nu) * s[:, :3] - nu * tr[:, None]) / E
    out[:, 3:] = 2.0 * (1.0 + nu) * s[:, 3:] / E
    return out


def stress_from_strain_3d(strain: np.ndarray, E: float, nu: float) -> np.ndarray:
    """Hooke: ``(N,6) [e11,e22,e33,g12,g23,g31] -> (N,6) [s11..s31]``."""
    e = np.atleast_2d(np.asarray(strain, dtype=float))
    lam, mu = lame_parameters(E, nu)
    tr = e[:, 0] + e[:, 1] + e[:, 2]
    out = np.empty_like(e)
    out[:, :3] = lam * tr[:, None] + 2.0 * mu * e[:, :3]
    out[:, 3:] = mu * e[:, 3:]                              # sigma = mu * gamma
    return out


# ---------------------------------------------------------------------------
# Stress invariants and the classic BESO proxy (energy density)
# ---------------------------------------------------------------------------


def _invariants_2d(stress: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sigma:sigma, tr sigma) for (N,3) Voigt in-plane stress."""
    s = np.atleast_2d(np.asarray(stress, dtype=float))
    ss = s[:, 0] ** 2 + s[:, 1] ** 2 + 2.0 * s[:, 2] ** 2
    return ss, s[:, 0] + s[:, 1]


def _invariants_3d(stress: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sigma:sigma, tr sigma) for (N,6) Voigt stress."""
    s = np.atleast_2d(np.asarray(stress, dtype=float))
    ss = (s[:, 0] ** 2 + s[:, 1] ** 2 + s[:, 2] ** 2
          + 2.0 * (s[:, 3] ** 2 + s[:, 4] ** 2 + s[:, 5] ** 2))
    return ss, s[:, 0] + s[:, 1] + s[:, 2]


def energy_density_from_stress_2d(stress: np.ndarray, E: float, nu: float,
                                  plane: str = "stress") -> np.ndarray:
    """``1/2 sigma:eps`` per element — the classic BESO sensitivity proxy."""
    s = np.atleast_2d(np.asarray(stress, dtype=float))
    e = strain_from_stress_2d(s, E, nu, plane)
    return 0.5 * np.einsum("ij,ij->i", s, e)               # engineering shear: plain dot


def energy_density_from_stress_3d(stress: np.ndarray, E: float, nu: float) -> np.ndarray:
    """``1/2 sigma:eps`` per element — the classic BESO sensitivity proxy."""
    s = np.atleast_2d(np.asarray(stress, dtype=float))
    e = strain_from_stress_3d(s, E, nu)
    return 0.5 * np.einsum("ij,ij->i", s, e)


# ---------------------------------------------------------------------------
# Topological derivative of compliance (the point of the module)
# ---------------------------------------------------------------------------


def td_compliance_2d(stress: np.ndarray, E: float, nu: float,
                     plane: str = "stress") -> np.ndarray:
    """TD of compliance for a circular hole: ``(N,3) Voigt stress -> (N,)``.

    ``DT = (4 sigma:sigma - tr(sigma)^2) / E`` in plane stress (GGM 2001 /
    Amstutz & Andrä 2006 reduced to stress-only form — see module docstring for
    the derivation chain and cross-checks). Normalised per unit removed area:
    ``dC ~= DT * (pi rho^2)``. Always > 0 for nonzero stress and nu in (-1, 1):
    ``4 sigma:sigma >= 2 tr(sigma)^2`` by Cauchy-Schwarz in 2D. Plane strain via
    the E/(1-nu^2), nu/(1-nu) equivalence (the bracket is Poisson-free, so only
    the 1/E prefactor changes).
    """
    ss, tr = _invariants_2d(stress)
    Ee, _ = _plane_equivalent(E, nu, plane)
    return (4.0 * ss - tr ** 2) / Ee


def td_compliance_3d(stress: np.ndarray, E: float, nu: float) -> np.ndarray:
    """TD of compliance for a spherical hole: ``(N,6) Voigt stress -> (N,)``.

    ``DT = 3(1-nu)/(2E(7-5nu)) * [10(1+nu) sigma:sigma - (1+5nu) tr(sigma)^2]``
    (GGM 2001 Lame form reduced to stress-only; identical to the Novotny &
    Sokolowski book form and to the a1/a2 coefficient pair of published TD
    codes — module docstring). Normalised per unit removed volume:
    ``dC ~= DT * (4/3 pi rho^3)``. Positive-definite for nu in (-1, 1/2):
    ``sigma:sigma >= tr^2/3`` gives bracket ``>= (7-5nu)/3 * tr^2 > 0``.
    """
    ss, tr = _invariants_3d(stress)
    return (3.0 * (1.0 - nu) / (2.0 * E * (7.0 - 5.0 * nu))
            * (10.0 * (1.0 + nu) * ss - (1.0 + 5.0 * nu) * tr ** 2))


# ---------------------------------------------------------------------------
# Rank agreement (does TDSA actually re-order a BESO energy ranking?)
# ---------------------------------------------------------------------------


def _ranks(x: np.ndarray) -> np.ndarray:
    """Fractional ranks (ties get the mean rank), plain numpy."""
    x = np.asarray(x, dtype=float).ravel()
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=float)
    ranks[order] = np.arange(x.size, dtype=float)
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.zeros(counts.size)
    np.add.at(sums, inv, ranks)
    return sums[inv] / counts[inv]


def rank_agreement(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation between two sensitivity fields, in [-1, 1].

    1.0 means the two fields would drive identical BESO delete/keep decisions at
    every volume target; anything less means TDSA genuinely re-orders elements
    relative to the energy proxy. A constant field imposes no ordering at all,
    so it is defined to agree (returns 1.0) rather than NaN.
    """
    ra, rb = _ranks(a), _ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.sqrt((ra * ra).sum() * (rb * rb).sum()))
    if denom == 0.0:
        return 1.0
    return float((ra * rb).sum() / denom)
