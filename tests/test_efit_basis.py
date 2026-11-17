"""
Test efit_basis, EFIT's basis functions of p' and FF' (ppbasisfunc.f90, ffbasisfunc.f90), and the <basis> of
equilibrium.code.parameters on synthetic data, without MDSplus. The tests against DIII-D runs are in
test_equilibrium_code_parameters.py.

Ported from IDA (tests/utils/test_efit_splines.py) where the basis was recovered before it moved here.
"""
import pickle
import xml.etree.ElementTree as ET

import numpy as np
import pytest
from scipy.integrate import cumulative_trapezoid

from imas_composer import ImasComposer, NoData
from imas_composer.efit_basis import (
    EFIT_MU0,
    RECOVERY_TOLERANCE,
    _search_knots,
    basis_settings,
    efit_f_profile,
    efit_profile_basis,
    evaluate_basis,
    evaluate_f,
    evaluate_profiles,
    parse_basis,
    polynomial_basis,
    resolve_basis,
    split_coefficients,
    tension_spline_basis,
    write_basis,
)
from tests.conftest import efit_fpol

KNOT_TOLERANCE = 1e-6  # psi_n, recovered from float32 profiles
COEFFICIENT_TOLERANCE = 1e-6  # relative
# the IMAS (COCOS 11) factors of the g-file sums, +-1 / (2 pi) for p' and +-mu0 / (2 pi) for FF'
IMAS_FACTORS = {"pprime": -1 / (2 * np.pi), "ffprime": -EFIT_MU0 / (2 * np.pi)}
SHOT = 200000
CODE_PARAMETERS = "equilibrium.code.parameters"
DPRESSURE_DPSI = "equilibrium.time_slice.profiles_1d.dpressure_dpsi"
F_DF_DPSI = "equilibrium.time_slice.profiles_1d.f_df_dpsi"
SPLINE_KNOTS = np.array([0.0, 0.3, 0.55, 0.9, 1.0])
# (shape, n_knots, pptens, n_psin) of the synthetic splines, as in CAKE runs
SPLINE_CASES = [
    pytest.param(("decreasing", 7, 1.0, 257), id="mse"),
    pytest.param(("pedestal", 7, 1.0, 129), id="mse_pedestal"),
    pytest.param(("decreasing", 8, 1.0, 257), id="ffprime_8_knots"),
    pytest.param(("pedestal", 7, 3.0, 129), id="pedestal_tension_3"),
    pytest.param(("decreasing", 3, 1.0, 65), id="few_knots"),
]


def hat_values(shape: str, n_knots: int, rng: np.random.Generator) -> np.ndarray:
    """Profile values at the knots: decreasing towards the edge, or with a pedestal spike before it as in CAKE runs."""
    values = np.linspace(1.0, 0.2, n_knots) * rng.uniform(0.8, 1.2, n_knots)
    if shape == "pedestal":
        values[-2] = 3.0
    return values


def continuity_constraints(knots: np.ndarray, tension: float) -> np.ndarray:
    """(n_knots - 2, 2 n_knots) EFIT's constraints of a continuous first derivative at the interior knots
    (ppbasisfunc.f90 ppcnst, case 6), in 0-based coefficient indices: hat of knot k at 2k, bump at 2k + 1."""
    n_knots = len(knots)
    constraints = np.zeros((n_knots - 2, 2 * n_knots))
    for row, k in enumerate(range(1, n_knots - 1)):
        width_left, width_right = knots[k] - knots[k - 1], knots[k + 1] - knots[k]
        constraints[row, 2 * k - 2] = 1 / width_left
        constraints[row, 2 * k - 1] = -(-tension / np.sinh(tension * width_left) + 1 / width_left) / tension**2
        constraints[row, 2 * k] = -1 / width_right - 1 / width_left
        constraints[row, 2 * k + 1] = (-tension / np.tanh(tension * width_right) + 1 / width_right) / tension**2 - (
            tension / np.tanh(tension * width_left) - 1 / width_left
        ) / tension**2
        constraints[row, 2 * k + 2] = 1 / width_right
        constraints[row, 2 * k + 3] = (tension / np.sinh(tension * width_right) - 1 / width_right) / tension**2
    return constraints


def continuous_coefficients(hats: np.ndarray, bumps: np.ndarray, knots: np.ndarray, tension: float) -> np.ndarray:
    """Coefficients with the given hats and the bumps closest to `bumps` that satisfy `continuity_constraints`."""
    coefficients = np.empty(2 * len(knots))
    coefficients[::2], coefficients[1::2] = hats, bumps
    if len(knots) > 2:
        constraints = continuity_constraints(knots, tension)
        coefficients[1::2] -= np.linalg.pinv(constraints[:, 1::2]) @ (constraints @ coefficients)
    return coefficients


def efit_polynomial_reference(psin: np.ndarray, coefficients: np.ndarray, edge: float) -> np.ndarray:
    """Literal port of bsppel (ppbasisfunc.f90, case 0) summed over the coefficients, edge being pcurbd."""
    kppcur = len(coefficients)
    profile = np.zeros_like(psin)
    for iparm, coefficient in enumerate(coefficients, start=1):
        if iparm == 1:
            profile += coefficient * (1.0 - psin**kppcur * edge)
        else:
            profile += coefficient * (psin ** (iparm - 1) - psin**kppcur * edge)
    return profile


def spline_settings(spline: dict, knots: np.ndarray, movable_knots: bool) -> dict:
    """Settings of `basis_settings` for the synthetic spline, starting from `knots`."""
    return {
        "kind": "tension_spline",
        "n_coefficients": 2 * len(knots),
        "knots": knots,
        "tension": spline["tension"],
        "movable_knots": movable_knots,
    }


def moved_knots(knots: np.ndarray) -> np.ndarray:
    """`knots` with the interior knots moved by 3e-3 in alternating directions, as EFIT's knot optimisation does."""
    shifted = knots.copy()
    shifted[1:-1] += 3e-3 * (-1.0) ** np.arange(len(knots) - 2)
    return shifted


@pytest.fixture
def synthetic_spline(request) -> dict:
    """Coefficients and float32 profile of a random tension spline, as EFIT stores them.

    As in EFIT the first derivative is continuous at the knots. Coefficients and profile carry the IMAS factor of p'.
    """
    shape, n_knots, pptens, n_psin = request.param
    rng = np.random.default_rng(n_knots * 100 + n_psin)
    psin = np.linspace(0, 1, n_psin)
    interior = np.sort(rng.uniform(0.1, 0.95, n_knots - 2))
    while n_knots > 2 and np.diff(np.r_[0, interior, 1]).min() < 0.01:
        interior = np.sort(rng.uniform(0.1, 0.95, n_knots - 2))
    knots, tension = np.r_[0, interior, 1], pptens * (n_knots - 1)
    hats = 1e5 * hat_values(shape, n_knots, rng)
    coefficients = continuous_coefficients(hats, 1e5 * n_knots**2 * rng.normal(0, 1, n_knots), knots, tension)
    coefficients = IMAS_FACTORS["pprime"] * np.float32(coefficients).astype(np.float64)
    profile = (coefficients @ tension_spline_basis(psin, knots, tension)).astype(np.float32).astype(np.float64)
    return {"psin": psin, "knots": knots, "tension": tension, "coefficients": coefficients, "profile": profile}


@pytest.fixture(params=["polynomial", "tension_1", "tension_20", "tension_2000"])
def basis(request) -> dict:
    """A polynomial with finite boundary value, or a C1 tension spline of increasing tension."""
    if request.param == "polynomial":
        return {"kind": "polynomial", "edge": 0.0, "coefficients": np.array([1.0, -0.5, 2.0, -1.5])}
    tension = float(request.param.split("_")[1])
    rng = np.random.default_rng(int(tension))
    coefficients = continuous_coefficients(
        rng.uniform(0.5, 1, len(SPLINE_KNOTS)), rng.normal(0, 30, len(SPLINE_KNOTS)), SPLINE_KNOTS, tension
    )
    return {"kind": "tension_spline", "knots": SPLINE_KNOTS, "tension": tension, "coefficients": coefficients}


@pytest.fixture
def code_parameters() -> tuple[str, list[dict]]:
    """XML with the <basis> of two time slices, a polynomial p' and a tension spline FF', and these bases."""
    bases = []
    root = ET.Element("parameters")
    basis_element = ET.SubElement(root, "basis")
    for index in range(2):
        slice_bases = {
            "pprime": {"kind": "polynomial", "edge": 1.0, "coefficients": np.array([-1.5e5, 3.2e4]) * (index + 1)},
            "ffprime": {
                "kind": "tension_spline",
                "knots": SPLINE_KNOTS,
                "tension": 4.0,
                "coefficients": continuous_coefficients(
                    np.linspace(1.0, 0.2, 5), np.full(5, 0.1 * (index + 1)), SPLINE_KNOTS, 4.0
                ),
            },
        }
        slice_element = ET.SubElement(basis_element, "time_slice", index=str(index), source="snap")
        for profile, profile_basis in slice_bases.items():
            write_basis(slice_element, profile, profile_basis, fitted_knots=False)
        bases.append(slice_bases)
    return ET.tostring(root, encoding="unicode"), bases


@pytest.mark.parametrize("synthetic_spline", SPLINE_CASES, indirect=True)
def test_namelist_knots_are_kept(synthetic_spline):
    """Knots that reproduce the stored profile are not fitted, even where EFIT could have moved them."""
    settings = spline_settings(synthetic_spline, synthetic_spline["knots"], movable_knots=True)
    basis, rms, fitted = resolve_basis(
        settings, synthetic_spline["coefficients"], synthetic_spline["psin"], synthetic_spline["profile"]
    )
    assert rms < RECOVERY_TOLERANCE
    assert not fitted
    np.testing.assert_array_equal(basis["knots"], synthetic_spline["knots"])


@pytest.mark.parametrize("synthetic_spline", SPLINE_CASES, indirect=True)
def test_moved_knots_are_fitted(synthetic_spline):
    """Where EFIT moved the knots (kautoknt), the knots are fitted from those of the namelist."""
    settings = spline_settings(synthetic_spline, moved_knots(synthetic_spline["knots"]), movable_knots=True)
    basis, rms, fitted = resolve_basis(
        settings, synthetic_spline["coefficients"], synthetic_spline["psin"], synthetic_spline["profile"]
    )
    assert fitted
    assert rms < RECOVERY_TOLERANCE
    np.testing.assert_allclose(basis["knots"], synthetic_spline["knots"], rtol=0, atol=KNOT_TOLERANCE)
    np.testing.assert_array_equal(basis["coefficients"], synthetic_spline["coefficients"])


@pytest.mark.parametrize("synthetic_spline", SPLINE_CASES[:1], indirect=True)
def test_fixed_knots_are_not_fitted(synthetic_spline):
    """Knots EFIT kept (kautoknt = 0) are never fitted: the basis misses and the caller raises."""
    settings = spline_settings(synthetic_spline, moved_knots(synthetic_spline["knots"]), movable_knots=False)
    _, rms, fitted = resolve_basis(
        settings, synthetic_spline["coefficients"], synthetic_spline["psin"], synthetic_spline["profile"]
    )
    assert not fitted
    assert rms > RECOVERY_TOLERANCE


@pytest.mark.parametrize("synthetic_spline", SPLINE_CASES, indirect=True)
def test_search_finds_knots_far_from_the_start(synthetic_spline):
    """The candidate search starts from the points where the profile takes the hat values, not from the namelist."""
    n_knots = len(synthetic_spline["knots"])
    basis = {
        "kind": "tension_spline",
        "knots": np.linspace(0, 1, n_knots),
        "tension": synthetic_spline["tension"],
        "coefficients": synthetic_spline["coefficients"],
    }
    found = _search_knots(basis, synthetic_spline["psin"], synthetic_spline["profile"])
    np.testing.assert_allclose(found["knots"], synthetic_spline["knots"], rtol=0, atol=KNOT_TOLERANCE)


@pytest.mark.parametrize("n_knots", [2, 5, 8])
def test_hats_are_a_partition_of_unity_and_bumps_vanish_on_the_knots(n_knots):
    knots = np.sort(np.r_[0, np.random.default_rng(n_knots).uniform(0, 1, n_knots - 2), 1])
    psin = np.linspace(0, 1, 1001)
    basis = tension_spline_basis(psin, knots, tension=3.0)
    np.testing.assert_allclose(basis[::2].sum(axis=0), 1.0, atol=1e-12)
    on_knots = tension_spline_basis(knots, knots, tension=3.0)
    np.testing.assert_allclose(on_knots[::2], np.eye(n_knots), atol=1e-12)
    np.testing.assert_allclose(on_knots[1::2], 0.0, atol=1e-12)


@pytest.mark.parametrize("tension", [0.5, 6.0, 20.0])
def test_efit_constraints_give_a_continuous_slope(tension):
    """The basis satisfies EFIT's continuity constraints: coefficients obeying them give no kink at the knots."""
    rng = np.random.default_rng(3)
    coefficients = continuous_coefficients(rng.uniform(0.5, 1, 5), rng.normal(0, 30, 5), SPLINE_KNOTS, tension)
    step = 1e-6
    for knot in SPLINE_KNOTS[1:-1]:
        left, center, right = coefficients @ tension_spline_basis(
            np.array([knot - step, knot, knot + step]), SPLINE_KNOTS, tension
        )
        np.testing.assert_allclose((center - left) / step, (right - center) / step, rtol=1e-4, atol=1e-4)


def test_bump_tends_to_a_cubic_without_tension():
    """(sinh(tau t) / sinh(tau w) - t / w) / tau^2 -> t (t^2 - w^2) / (6 w) for tau -> 0."""
    knots = np.array([0.0, 0.4, 1.0])
    psin = np.linspace(0, 0.4, 101)
    bump = tension_spline_basis(psin, knots, tension=1e-3)[1]
    distance, width = 0.4 - psin, 0.4
    np.testing.assert_allclose(bump, distance * (distance**2 - width**2) / (6 * width), atol=1e-8)


@pytest.mark.parametrize("edge", [0.0, 1.0])
@pytest.mark.parametrize(
    "n_pprime, n_ffprime",
    [(2, 3), (2, 2), (3, 4)],
    ids=["efit01_203877", "efit01_118162", "cubic_pprime"],
)
def test_polynomial_profiles(n_pprime, n_ffprime, edge):
    """EFIT's polynomial basis (kppfnc = kfffnc = 0) with zero (edge 1) or finite (edge 0) profiles at the boundary,
    from float32 coefficients and profiles carrying the IMAS (COCOS 11) factors, as EFIT stores them."""
    rng = np.random.default_rng(10 * n_pprime + n_ffprime)
    psin = np.linspace(0, 1, 65)
    xrsp = np.float32(1e5 * rng.uniform(0.5, 2.0, n_pprime + n_ffprime) * rng.choice([-1, 1], n_pprime + n_ffprime))
    xrsp = xrsp.astype(np.float64)
    settings = basis_settings({"kppcur": n_pprime, "kffcur": n_ffprime, "pcurbd": edge, "fcurbd": edge})
    shares = split_coefficients(xrsp, settings)

    dense = np.linspace(0, 1, 1001)
    for key, coefficients in shares.items():
        profile = np.float32(IMAS_FACTORS[key] * efit_polynomial_reference(psin, coefficients, edge)).astype(np.float64)
        basis, rms, fitted = resolve_basis(settings[key], IMAS_FACTORS[key] * coefficients, psin, profile)
        assert basis["kind"] == "polynomial" and basis["edge"] == edge and not fitted
        assert rms < RECOVERY_TOLERANCE
        reference = IMAS_FACTORS[key] * efit_polynomial_reference(dense, coefficients, edge)
        np.testing.assert_allclose(
            evaluate_basis(dense, basis), reference, rtol=0, atol=COEFFICIENT_TOLERANCE * np.abs(reference).max()
        )


def test_default_settings_are_efits():
    """kppfnc = kfffnc = 0, pcurbd = fcurbd = 1 and kppcur = 3, kffcur = 1 of EFIT's full solution modes."""
    settings = basis_settings({})
    assert settings["pprime"] == {"kind": "polynomial", "n_coefficients": 3, "edge": 1.0}
    assert settings["ffprime"] == {"kind": "polynomial", "n_coefficients": 1, "edge": 1.0}


@pytest.mark.parametrize("icprof, n_pprime, n_ffprime, edge", [(1, 2, 2, 1.0), (2, 2, 2, 0.0), (3, 2, 3, 0.0)])
def test_icprof_overrides_the_namelist(icprof, n_pprime, n_ffprime, edge):
    settings = basis_settings({"icprof": icprof, "kppcur": 5, "kffcur": 5, "pcurbd": 0.5, "fcurbd": 0.5})
    assert settings["pprime"] == {"kind": "polynomial", "n_coefficients": n_pprime, "edge": edge}
    assert settings["ffprime"] == {"kind": "polynomial", "n_coefficients": n_ffprime, "edge": edge}


def test_spline_coefficients_follow_the_knots():
    """The snap file of 203877 EFIT02 sets kppcur 3 and kffcur 4, EFIT uses 2 coefficients per knot."""
    settings = basis_settings({
        "kppfnc": 6, "kppcur": 3, "kppknt": 2, "ppknt": [0.0, 1.0, 0.0, 0.0], "pptens": 5.0,
        "kfffnc": 6, "kffcur": 4, "kffknt": 3, "ffknt": [0.0, 0.3, 1.0], "fftens": 5.0,
    })
    assert settings["pprime"]["n_coefficients"] == 4
    assert settings["ffprime"]["n_coefficients"] == 6
    np.testing.assert_array_equal(settings["pprime"]["knots"], [0.0, 1.0])
    np.testing.assert_array_equal(settings["ffprime"]["knots"], [0.0, 0.3, 1.0])
    # pptens2 = |pptens| (kppknt - 1) / (ppknt(kppknt) - ppknt(1))
    assert settings["pprime"]["tension"] == 5.0
    assert settings["ffprime"]["tension"] == 10.0
    assert not settings["pprime"]["movable_knots"]
    assert basis_settings({"kppfnc": 6, "kppknt": 2, "ppknt": [0.0, 1.0], "kautoknt": 3})["pprime"]["movable_knots"]


@pytest.mark.parametrize("kind", [1, 2, 3, 4, 5, 7, 8])
def test_other_kinds_raise(kind):
    with pytest.raises(ValueError, match="only EFIT's polynomials"):
        basis_settings({"kfffnc": kind})


@pytest.mark.parametrize("values", [
    pytest.param({"kppfnc": 6, "kppknt": 3, "ppknt": [0.0, 1.0]}, id="fewer_knots_than_kppknt"),
    pytest.param({"kppfnc": 6}, id="no_knots"),
])
def test_spline_without_its_knots_raises(values):
    with pytest.raises(ValueError, match="kppfnc = 6 needs at least 2 knots"):
        basis_settings(values)


def test_split_needs_exactly_the_coefficients_of_both_bases():
    settings = basis_settings({"kppcur": 2, "kffcur": 3})
    shares = split_coefficients(np.arange(5.0), settings)
    np.testing.assert_array_equal(shares["pprime"], [0.0, 1.0])
    np.testing.assert_array_equal(shares["ffprime"], [2.0, 3.0, 4.0])
    with pytest.raises(ValueError, match="XRSP holds 6 coefficients"):
        split_coefficients(np.arange(6.0), settings)


def test_polynomial_basis_rejects_unknown_parts():
    with pytest.raises(ValueError, match="Unknown part"):
        polynomial_basis(np.linspace(0, 1, 3), 2, 1.0, part="second_derivative")


def test_derivative_matches_central_differences(basis):
    """Also next to the interior knots, where the spline is only C1. On a knot, where the second derivative jumps,
    the derivative of the interval to the right is continuous with the one to the left."""
    step = 1e-6
    knots = SPLINE_KNOTS[1:-1]
    psin = np.r_[np.linspace(0.01, 0.99, 37), knots - 1e-4, knots + 1e-4]
    numeric = (evaluate_basis(psin + step, basis) - evaluate_basis(psin - step, basis)) / (2 * step)
    np.testing.assert_allclose(
        evaluate_basis(psin, basis, "derivative"), numeric, rtol=1e-5, atol=1e-6 * np.abs(numeric).max()
    )
    np.testing.assert_allclose(
        evaluate_basis(knots, basis, "derivative"), evaluate_basis(knots - 1e-9, basis, "derivative"),
        rtol=1e-5, atol=1e-6 * np.abs(numeric).max(),
    )


def test_integral_matches_a_dense_trapezoid(basis):
    """The closed form of int_psin^1 against the trapezoidal rule on 200001 points."""
    dense = np.linspace(0, 1, 200001)
    values = evaluate_basis(dense, basis)
    # cumulative_trapezoid along decreasing psi_n integrates from 1 to psi_n
    expected = -cumulative_trapezoid(values[::-1], dense[::-1], initial=0)[::-1]
    np.testing.assert_allclose(
        evaluate_basis(dense[::1000], basis, "integral"), expected[::1000], rtol=0,
        atol=1e-8 * np.abs(values).max(),
    )


@pytest.mark.parametrize("f_boundary", [-1.7, 2.1])
def test_f_matches_efits_trapezoid_on_a_fine_grid(code_parameters, f_boundary):
    """F of the closed-form integral is EFIT's FPOL (ece.F90) without the error of the trapezoidal rule."""
    xml, bases = code_parameters
    psi_axis, psi_boundary = 0.2, -0.3
    psin = np.linspace(0, 1, 4097)
    ffprime = evaluate_basis(psin, bases[0]["ffprime"])
    f_profile = efit_f_profile(xml, 0, f_boundary, psi_axis, psi_boundary)
    np.testing.assert_allclose(f_profile(psin), efit_fpol(ffprime, f_boundary, psi_boundary - psi_axis), rtol=1e-7)
    assert f_profile(np.array([1.0]))[0] == f_boundary


def test_f_keeps_the_boundary_value_where_f_squared_is_negative():
    """As EFIT (ece.F90): F^2 = 0.01 - 200 (1 - psi_n) is negative inside psi_n = 0.99995."""
    ffprime = {"kind": "polynomial", "edge": 0.0, "coefficients": np.array([1.0])}
    np.testing.assert_array_equal(evaluate_f(ffprime, 0.1, 100.0, np.array([0.0, 0.5, 1.0])), [0.1, 0.1, 0.1])


@pytest.mark.parametrize("psi_n", [[-1e-3, 0.5], [0.5, 1.001]], ids=["below_axis", "outside_boundary"])
def test_psi_n_outside_the_plasma_raises(code_parameters, psi_n):
    xml, _ = code_parameters
    with pytest.raises(ValueError, match=r"psi_n must lie in \[0, 1\]"):
        efit_profile_basis(xml, 0)(np.array(psi_n))
    with pytest.raises(ValueError, match=r"psi_n must lie in \[0, 1\]"):
        efit_f_profile(xml, 0, 1.0, 0.0, 1.0)(np.array(psi_n))


@pytest.mark.parametrize("derivative", [False, True])
@pytest.mark.parametrize("i_time", [0, 1])
def test_xml_round_trip_is_exact(code_parameters, i_time, derivative):
    """The XML keeps the shortest exact form of every float, so the evaluator gives the bases' profiles bit for bit."""
    xml, bases = code_parameters
    psin = np.linspace(0, 1, 101).reshape(1, 101)
    pprime, ffprime = efit_profile_basis(xml, i_time)(psin, derivative=derivative)
    part = "derivative" if derivative else "value"
    assert pprime.shape == psin.shape
    np.testing.assert_array_equal(pprime, evaluate_basis(psin, bases[i_time]["pprime"], part))
    np.testing.assert_array_equal(ffprime, evaluate_basis(psin, bases[i_time]["ffprime"], part))


def test_write_basis_marks_fitted_knots():
    parent = ET.Element("time_slice")
    spline = {"kind": "tension_spline", "knots": np.array([0.0, 0.3, 1.0]), "tension": 10.0,
              "coefficients": np.arange(6.0)}
    write_basis(parent, "ffprime", spline, fitted_knots=True)
    write_basis(parent, "pprime", {"kind": "polynomial", "edge": 0.0, "coefficients": np.ones(2)}, fitted_knots=False)
    ffprime, pprime = parent.find("ffprime"), parent.find("pprime")
    assert ffprime.attrib == {"kind": "tension_spline", "tension": "10", "knots": "0 0.3 1", "fitted_knots": "true"}
    assert pprime.attrib == {"kind": "polynomial", "edge": "0"}


def test_evaluators_pickle(code_parameters):
    xml, _ = code_parameters
    psin = np.linspace(0, 1, 51)
    profile_basis = efit_profile_basis(xml, 1)
    f_profile = efit_f_profile(xml, 1, -1.7, 0.2, -0.3)
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(profile_basis))(psin), profile_basis(psin))
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(f_profile))(psin), f_profile(psin))


def test_evaluate_profiles_is_the_evaluator(code_parameters):
    xml, bases = code_parameters
    psin = np.linspace(0, 1, 11)
    np.testing.assert_array_equal(
        efit_profile_basis(xml, 0)(psin), evaluate_profiles(bases[0]["pprime"], bases[0]["ffprime"], psin)
    )


def test_parse_rejects_misnumbered_time_slices():
    xml = '<parameters><basis><time_slice index="1"><pprime kind="polynomial" edge="0">1</pprime>' \
          '<ffprime kind="polynomial" edge="0">1</ffprime></time_slice></basis></parameters>'
    with pytest.raises(ValueError, match="not numbered"):
        parse_basis(xml, 0)


def test_parse_rejects_time_slices_it_does_not_have(code_parameters):
    xml, _ = code_parameters
    with pytest.raises(ValueError, match="outside the 2 time slices"):
        parse_basis(xml, 2)


# snap files of polynomials of 2 coefficients with finite boundary values, without and with fitdelz
SNAP = " &efitin\n  kppcur=2  kffcur=2\n  pcurbd=0. fcurbd=0.\n /\n"
FITDELZ_SNAP = " &efitin\n  kppcur=2  kffcur=2\n  pcurbd=0. fcurbd=0.\n  fitdelz=.true.\n /\n"
NODE_NOT_FOUND = NoData("%TREE-W-NNF, Node Not Found")


def standard_tree_raw_data(
    composer: ImasComposer,
    snap,
    pprime_error: tuple = (0.0, 0.0),
    plasma: tuple = (1.0e6, 1.0e6),
    no_data: tuple = (),
) -> dict:
    """raw_data of equilibrium.code.parameters of a standard EFIT tree with two slices: polynomials of 2 coefficients
    with finite boundary values (pcurbd = fcurbd = 0), the g-file profiles their float32 sums times EFIT's output sign
    (write_g.f90: -1 where the measured Ip `plasma` of the slice is > 0) and 1 + pprime_error of the slice for p'.
    CPASMA is positive whatever `plasma` is. `snap` is the snap file namelist or a NoData, the nodes `no_data` (e.g.
    "XRSP") are NODE_NOT_FOUND."""
    status, requirements = composer.resolve([CODE_PARAMETERS, DPRESSURE_DPSI, F_DF_DPSI], SHOT, {})
    psin = np.linspace(0, 1, 65)
    xrsp = np.full((2, 8), 9.97e36, dtype=np.float32)
    xrsp[:, :4] = [[1.2e5, -4.0e4, 2.5, -0.8], [0.9e5, -3.0e4, 2.2, -0.6]]
    used = xrsp[:, :4].astype(np.float64)
    sign_out = np.where(np.array(plasma) > 0, -1.0, 1.0)[:, np.newaxis]
    pprime = np.stack([efit_polynomial_reference(psin, coefficients[:2], 0.0) for coefficients in used])
    ffprim = np.stack([EFIT_MU0 * efit_polynomial_reference(psin, coefficients[2:], 0.0) for coefficients in used])
    values = {
        "GTIME": np.array([1000.0, 2000.0]),
        "MTIME": np.array([1000.0, 2000.0]),
        "XRSP": xrsp,
        "PLASMA": np.array(plasma),
        "NAMELIST": snap,
        "PSIN": psin.astype(np.float32),
        "PPRIME": (sign_out * pprime * (1 + np.array(pprime_error)[:, np.newaxis])).astype(np.float32),
        "FFPRIM": (sign_out * ffprim).astype(np.float32),
        "BCENTR": np.array([-2.0, -2.0]),
        "CPASMA": np.array([1.0e6, 1.0e6]),
    }
    values.update({name: NODE_NOT_FOUND for name in no_data})
    # the last name of the MDSplus path, e.g. GTIME of \EFIT01::TOP.RESULTS.GEQDSK.GTIME
    return {req.as_key(): values[req.mds_path.split(".")[-1].split(":")[-1]] for req in requirements}


def basis_slices(xml: str) -> list:
    return ET.fromstring(xml).find("basis").findall("time_slice")


@pytest.mark.parametrize("efit_tree, efit_run_id, namelist", [
    pytest.param("EFIT01", "", "NAMELIST", id="standard_tree_snap_file"),
    pytest.param("EFIT", "01", "KEQDSKS", id="run_id_tree_k_files"),
])
def test_only_the_namelist_of_the_tree_is_required(efit_tree, efit_run_id, namelist):
    """A standard tree requires its snap file and no k-files, a run-id run its k-files and no snap file."""
    composer = ImasComposer(efit_tree=efit_tree, efit_run_id=efit_run_id)
    _, requirements = composer.resolve([CODE_PARAMETERS], SHOT, {})
    paths = [req.mds_path for req in requirements]
    other = {"NAMELIST": "KEQDSKS", "KEQDSKS": "TOP:NAMELIST"}[namelist]
    assert any(path.endswith(namelist) for path in paths)
    assert not any(other in path for path in paths)


@pytest.mark.parametrize("plasma", [
    pytest.param((1.0e6, 1.0e6), id="positive_ip"),
    pytest.param((-1.0e6, -1.0e6), id="negative_ip"),
    pytest.param((1.0e6, 0.0), id="measured_ip_zero"),
])
def test_basis_from_the_snap_file(plasma):
    """The basis follows the sign of the stored profiles slice by slice, set by the measured Ip, not CPASMA."""
    composer = ImasComposer()
    raw_data = standard_tree_raw_data(composer, SNAP, plasma=plasma)
    results = composer.compose([CODE_PARAMETERS, DPRESSURE_DPSI, F_DF_DPSI], SHOT, raw_data)
    xml = results[CODE_PARAMETERS]
    assert [element.get("source") for element in basis_slices(xml)] == ["snap", "snap"]
    psin = np.linspace(0, 1, 65)
    for index in range(2):
        pprime, ffprime = efit_profile_basis(xml, index)(psin)
        for values, stored in ((pprime, results[DPRESSURE_DPSI][index]), (ffprime, results[F_DF_DPSI][index])):
            np.testing.assert_allclose(values, stored, rtol=0, atol=RECOVERY_TOLERANCE * np.abs(stored).max())


def test_slice_whose_basis_misses_keeps_the_error():
    """A slice whose basis misses its stored profiles has an error instead of bases, the other slices keep theirs."""
    composer = ImasComposer()
    raw_data = standard_tree_raw_data(composer, SNAP, pprime_error=(0.0, 1e-3))
    xml = composer.compose([CODE_PARAMETERS], SHOT, raw_data)[CODE_PARAMETERS]
    first, second = basis_slices(xml)
    assert [child.tag for child in first] == ["pprime", "ffprime"] and "error" not in first.attrib
    assert len(second) == 0 and "misses the stored pprime (polynomial)" in second.get("error")
    efit_profile_basis(xml, 0)
    with pytest.raises(ValueError, match="Time slice 1 has no EFIT basis: EFIT's basis .* misses the stored pprime"):
        efit_profile_basis(xml, 1)
    with pytest.raises(ValueError, match="Time slice 1 has no EFIT basis"):
        efit_f_profile(xml, 1, 1.0, 0.0, 1.0)


@pytest.mark.parametrize("snap, error", [
    pytest.param(" &efitin\n  kppfnc=2\n /\n", "kppfnc = 2", id="unsupported_kind"),
    pytest.param(" &efitin\n  kppcur=3  kffcur=2\n /\n", "XRSP holds 4 coefficients", id="xrsp_count"),
    pytest.param(NODE_NOT_FOUND, "misses the stored pprime", id="defaults_miss"),
])
def test_slices_without_a_basis_keep_xrsp(snap, error):
    """Unsupported settings, and EFIT's defaults (polynomials of 3 and 1 coefficients vanishing at the boundary) of a
    run without a snap file, leave every slice without a basis, but the field composes with <xrsp>."""
    composer = ImasComposer()
    raw_data = standard_tree_raw_data(composer, snap)
    xml = composer.compose([CODE_PARAMETERS], SHOT, raw_data)[CODE_PARAMETERS]
    assert len(ET.fromstring(xml).find("xrsp")) == 2
    for element in basis_slices(xml):
        assert len(element) == 0 and error in element.get("error")


@pytest.mark.parametrize("no_data, snap, parts", [
    pytest.param(("PPRIME",), SNAP, ["xrsp"], id="no_profiles"),
    pytest.param(("PLASMA",), SNAP, ["xrsp"], id="no_measured_ip"),
    pytest.param(("XRSP",), FITDELZ_SNAP, ["fitdelz", "ifitdelz"], id="no_xrsp"),
    pytest.param(("GTIME",), FITDELZ_SNAP, None, id="no_time_slices"),
    pytest.param(("XRSP",), SNAP, None, id="no_xrsp_nor_fitdelz"),
])
def test_parts_whose_inputs_are_missing_are_left_out(no_data, snap, parts):
    """Every part is written when its inputs exist, NoData (parts None) only when there is none."""
    composer = ImasComposer()
    raw_data = standard_tree_raw_data(composer, snap, no_data=no_data)
    result = composer.compose([CODE_PARAMETERS], SHOT, raw_data)[CODE_PARAMETERS]
    if parts is None:
        assert result is NODE_NOT_FOUND
        return
    assert [child.tag for child in ET.fromstring(result)] == parts
    if "xrsp" in parts:
        with pytest.raises(ValueError, match="holds no <basis>"):
            efit_profile_basis(result, 0)
