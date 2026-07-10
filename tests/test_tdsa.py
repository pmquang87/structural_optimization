"""Hermetic, analytic tests for the EXPERIMENTAL TDSA prototype.

No OpenRadioss, no Deck/Mesh — pure numpy against :mod:`oropt.tdsa`. These pin
the maths the spike rests on: Hooke inversion round-trips, the stress-only TD
formulas agree with the independent Lame sigma:eps representation (GGM 2001 /
Amstutz form), hydrostatic states reduce to the hand-computable scalars, the TD
is a positive quadratic form of stress (removing loaded material always costs
compliance), uniaxial fields rank identically under TD and the BESO energy
proxy, and a multiaxial counter-example shows the two rankings genuinely differ
— which is the whole point of the module.
"""
import numpy as np

from oropt.tdsa import (energy_density_from_stress_2d, energy_density_from_stress_3d,
                        lame_parameters, lame_plane_stress, rank_agreement,
                        strain_from_stress_2d, strain_from_stress_3d,
                        stress_from_strain_2d, stress_from_strain_3d,
                        td_compliance_2d, td_compliance_3d)

E, NU = 210e3, 0.3                                        # steel-ish [MPa]


def _rand_stress(n, cols, seed=0, scale=100.0):
    rng = np.random.default_rng(seed)
    return scale * rng.standard_normal((n, cols))


# ---- Hooke inversion round-trips --------------------------------------------

def test_hooke_round_trip_2d_plane_stress_and_strain():
    s = _rand_stress(50, 3, seed=1)
    for plane in ("stress", "strain"):
        back = stress_from_strain_2d(strain_from_stress_2d(s, E, NU, plane), E, NU, plane)
        assert np.allclose(back, s, rtol=1e-12, atol=1e-9)


def test_hooke_round_trip_3d():
    s = _rand_stress(50, 6, seed=2)
    back = stress_from_strain_3d(strain_from_stress_3d(s, E, NU), E, NU)
    assert np.allclose(back, s, rtol=1e-12, atol=1e-9)


def test_lame_constants_match_textbook_identities():
    lam, mu = lame_parameters(E, NU)
    assert np.isclose(mu, E / (2 * (1 + NU)))
    assert np.isclose(E, mu * (3 * lam + 2 * mu) / (lam + mu))   # E from (lam, mu)
    lam_ps, mu_ps = lame_plane_stress(E, NU)
    assert np.isclose(mu_ps, mu)
    assert np.isclose(lam_ps, 2 * lam * mu / (lam + 2 * mu))     # lambda* identity


# ---- stress-only forms == independent Lame sigma:eps representation ---------

def test_td_2d_matches_lame_form():
    """(4 s:s - tr^2)/E == (lam+2mu)/(2mu(lam+mu)) [4mu s:e + (lam-mu) tr s tr e]
    with the in-plane Lame constants — the two published representations of the
    GGM/Amstutz circular-hole TD must be the same function of stress."""
    s = _rand_stress(40, 3, seed=3)
    for plane, (lam, mu) in (("stress", lame_plane_stress(E, NU)),
                             ("strain", lame_parameters(E, NU))):
        e = strain_from_stress_2d(s, E, NU, plane)
        se = np.einsum("ij,ij->i", s, e)                  # engineering shear: plain dot
        trs = s[:, 0] + s[:, 1]
        tre = e[:, 0] + e[:, 1]
        td_lame = (lam + 2 * mu) / (2 * mu * (lam + mu)) * (
            4 * mu * se + (lam - mu) * trs * tre)
        assert np.allclose(td_compliance_2d(s, E, NU, plane), td_lame, rtol=1e-12)


def test_td_3d_matches_lame_form():
    """Stress-only (7-5nu) form == (3/4)(lam+2mu)/(mu(9lam+14mu)) [20mu s:e +
    (3lam-2mu) tr s tr e] (GGM 2001 3D, per unit removed volume)."""
    s = _rand_stress(40, 6, seed=4)
    lam, mu = lame_parameters(E, NU)
    e = strain_from_stress_3d(s, E, NU)
    se = np.einsum("ij,ij->i", s, e)
    trs = s[:, 0] + s[:, 1] + s[:, 2]
    tre = e[:, 0] + e[:, 1] + e[:, 2]
    td_lame = 0.75 * (lam + 2 * mu) / (mu * (9 * lam + 14 * mu)) * (
        20 * mu * se + (3 * lam - 2 * mu) * trs * tre)
    assert np.allclose(td_compliance_3d(s, E, NU), td_lame, rtol=1e-12)


# ---- hydrostatic states: hand-computable scalars -----------------------------

def test_td_2d_hydrostatic_reduces_to_hand_value():
    """sigma = p*I in-plane: s:s = 2p^2, tr = 2p, so TD = (8-4)p^2/E = 4p^2/E;
    energy density = (1-nu)p^2/E. Also pins the dilute-hole biaxial limit."""
    p = 37.0
    s = np.array([[p, p, 0.0]])
    assert np.isclose(td_compliance_2d(s, E, NU)[0], 4 * p ** 2 / E)
    assert np.isclose(energy_density_from_stress_2d(s, E, NU)[0], (1 - NU) * p ** 2 / E)
    # plane strain only rescales the Poisson-free bracket by (1-nu^2)/1 via 1/E'
    assert np.isclose(td_compliance_2d(s, E, NU, "strain")[0],
                      4 * p ** 2 * (1 - NU ** 2) / E)


def test_td_3d_hydrostatic_reduces_to_hand_value():
    """sigma = p*I: bracket = [30(1+nu) - 9(1+5nu)]p^2 = 3(7-5nu)p^2, so
    TD = 9(1-nu)p^2/(2E) — the classical Mackenzie dilute-spherical-void
    compliance per unit cavity volume. Energy density = 3(1-2nu)p^2/(2E)."""
    p = 11.0
    s = np.array([[p, p, p, 0.0, 0.0, 0.0]])
    assert np.isclose(td_compliance_3d(s, E, NU)[0], 9 * (1 - NU) * p ** 2 / (2 * E))
    assert np.isclose(energy_density_from_stress_3d(s, E, NU)[0],
                      3 * (1 - 2 * NU) * p ** 2 / (2 * E))


# ---- structure of the quadratic form -----------------------------------------

def test_td_is_quadratic_in_stress():
    s2, s3 = _rand_stress(30, 3, seed=5), _rand_stress(30, 6, seed=6)
    assert np.allclose(td_compliance_2d(2 * s2, E, NU), 4 * td_compliance_2d(s2, E, NU))
    assert np.allclose(td_compliance_3d(2 * s3, E, NU), 4 * td_compliance_3d(s3, E, NU))


def test_td_positive_for_any_nonzero_stress():
    """Removing loaded material always increases compliance: the TD quadratic
    form is positive-definite for every admissible Poisson ratio."""
    s2, s3 = _rand_stress(500, 3, seed=7), _rand_stress(500, 6, seed=8)
    for nu in (-0.4, 0.0, 0.3, 0.45):
        assert np.all(td_compliance_2d(s2, E, nu) > 0)
        assert np.all(td_compliance_2d(s2, E, nu, "strain") > 0)
        assert np.all(td_compliance_3d(s3, E, nu) > 0)
    assert np.allclose(td_compliance_2d(np.zeros((1, 3)), E, NU), 0.0)
    assert np.allclose(td_compliance_3d(np.zeros((1, 6)), E, NU), 0.0)


# ---- ranking: where TDSA agrees with BESO's energy proxy, and where not ------

def test_uniaxial_field_ranks_identically_and_with_fixed_ratio():
    """For uniaxial stress both TD and energy density are proportional to
    sigma^2, so a BESO ranking is unchanged — TD/energy is a constant:
    6 in plane stress, 3(1-nu)(9+5nu)/(7-5nu) in 3D."""
    mags = np.array([10.0, -80.0, 35.0, 120.0, -5.0])
    s2 = np.zeros((5, 3)); s2[:, 0] = mags
    s3 = np.zeros((5, 6)); s3[:, 2] = mags                # axis choice is irrelevant (isotropy)

    td2, en2 = td_compliance_2d(s2, E, NU), energy_density_from_stress_2d(s2, E, NU)
    td3, en3 = td_compliance_3d(s3, E, NU), energy_density_from_stress_3d(s3, E, NU)
    assert rank_agreement(td2, en2) == 1.0
    assert rank_agreement(td3, en3) == 1.0
    assert np.allclose(td2 / en2, 6.0)
    assert np.allclose(td3 / en3, 3 * (1 - NU) * (9 + 5 * NU) / (7 - 5 * NU))


def test_multiaxial_counterexample_2d_shear_vs_hydrostatic():
    """The point of TDSA: in plane stress a pure-shear element is costlier to
    remove than a hydrostatic element of EQUAL energy density by the factor
    2(1-nu)/(1+nu) > 1. Give the hydrostatic element slightly MORE energy and
    the two sensitivities order the elements oppositely."""
    tau = 100.0
    u_shear = (1 + NU) * tau ** 2 / E                     # energy of the shear element
    p = np.sqrt(1.02 * u_shear * E / (1 - NU))            # hydro with 2% more energy
    s = np.array([[0.0, 0.0, tau],                        # pure shear
                  [p, p, 0.0],                            # hydrostatic
                  [30.0, 0.0, 0.0]])                      # weak uniaxial (lowest in both)
    en = energy_density_from_stress_2d(s, E, NU)
    td = td_compliance_2d(s, E, NU)

    assert en[1] > en[0] > en[2]                          # energy: hydro on top
    assert td[0] > td[1] > td[2]                          # TD: shear on top
    assert rank_agreement(td, en) < 1.0                   # BESO would flip a decision


def test_multiaxial_counterexample_3d_hydrostatic_vs_shear():
    """In 3D the preference reverses: per unit energy the TD of a hydrostatic
    state (3(1-nu)/(1-2nu)) exceeds a pure shear's (30(1-nu)/(7-5nu)), so a
    slightly LESS energetic hydrostatic element still outranks the shear one."""
    tau = 100.0
    u_shear = (1 + NU) * tau ** 2 / E
    p = np.sqrt((u_shear / 1.02) * 2 * E / (3 * (1 - 2 * NU)))   # 2% less energy
    s = np.array([[0.0, 0.0, 0.0, tau, 0.0, 0.0],
                  [p, p, p, 0.0, 0.0, 0.0],
                  [0.0, 30.0, 0.0, 0.0, 0.0, 0.0]])
    en = energy_density_from_stress_3d(s, E, NU)
    td = td_compliance_3d(s, E, NU)

    assert en[0] > en[1] > en[2]                          # energy: shear on top
    assert td[1] > td[0] > td[2]                          # TD: hydro on top
    assert rank_agreement(td, en) < 1.0


# ---- rank_agreement helper ---------------------------------------------------

def test_rank_agreement_bounds_ties_and_invariance():
    a = np.array([3.0, 1.0, 4.0, 1.5, 9.0])
    assert np.isclose(rank_agreement(a, a), 1.0)
    assert np.isclose(rank_agreement(a, -a), -1.0)        # exact reversal
    assert np.isclose(rank_agreement(a, np.exp(a)), 1.0)  # monotone transform invariant
    assert rank_agreement(a, np.ones(5)) == 1.0           # constant field: vacuous ordering
    b = np.array([1.0, 1.0, 2.0, 2.0, 3.0])              # ties averaged, still in [-1, 1]
    assert -1.0 <= rank_agreement(a, b) <= 1.0


# ---- vectorisation -----------------------------------------------------------

def test_vectorised_matches_per_element_loop():
    s2, s3 = _rand_stress(20, 3, seed=9), _rand_stress(20, 6, seed=10)
    loop2 = np.array([td_compliance_2d(row[None, :], E, NU)[0] for row in s2])
    loop3 = np.array([td_compliance_3d(row[None, :], E, NU)[0] for row in s3])
    assert np.allclose(td_compliance_2d(s2, E, NU), loop2)
    assert np.allclose(td_compliance_3d(s3, E, NU), loop3)
    loop_e = np.array([energy_density_from_stress_3d(row[None, :], E, NU)[0] for row in s3])
    assert np.allclose(energy_density_from_stress_3d(s3, E, NU), loop_e)
