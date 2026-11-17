"""
EFIT's basis functions of p' and FF'.

EFIT evaluates p' and FF' on every grid node as sums of basis functions (pressure.F90 ppcurr, current.f90 fpcurr):

    pprime(psi_n) = sum_j xrsp_j bsppel_j(psi_n),   ffprim(psi_n) = mu0 sum_j xrsp_(kppcur+j) bsffel_j(psi_n)

in g-file units. MDSplus keeps the coefficients (MEASUREMENTS.XRSP) and the profiles on nw points (GEQDSK.PPRIME,
FFPRIM), while the basis is an input of the run: its kind (kppfnc, kfffnc), the number of coefficients of each
profile and the parameters of the basis, set in the namelists of the run. Interpolating the nw points misses what
happens between them, e.g. a pedestal narrower than the point spacing. Two kinds of ppbasisfunc.f90 and
ffbasisfunc.f90 (the same functions with pp and ff names) are supported:

- Polynomials (kppfnc = kfffnc = 0, EFIT's default and the basis of the standard EFIT01 runs). Coefficient i of a
  profile with n coefficients multiplies x^(i-1) - edge x^n, with x = psi_n and edge EFIT's pcurbd (fcurbd for FF'),
  1 for a profile that vanishes at the boundary and 0 otherwise.
- Tension splines (kppfnc = kfffnc = 6, CAKE and EFIT02 runs). Knot k carries a hat (odd coefficient), which is 1 at
  the knot, and a tension bump (even coefficient), which vanishes at every knot. On an interval of width w next to the
  knot, with t the distance from the far end of the interval,

      hat = t / w,    bump = (sinh(tension t) / sinh(tension w) - t / w) / tension^2,

  where tension is EFIT's pptens2 = |pptens| (n_knots - 1) / (knot_last - knot_first). EFIT keeps the first
  derivative of the profile continuous at the knots (ppcnst), so the profile is C1 there.

The mapper of equilibrium.code.parameters builds the basis of every slice from the namelists (`basis_settings`),
checks it against the stored profiles (`resolve_basis`) and writes it into the XML (`write_basis`).
`efit_profile_basis` and `efit_f_profile` evaluate a slice of that XML on any psi_n.
"""

import functools
import xml.etree.ElementTree as ET
from itertools import product
from typing import Callable, Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

# relative rms of a basis against the stored profiles, which are float32; EFIT's basis reproduces them to ~1e-8
RECOVERY_TOLERANCE = 1e-6
# psi_n points of the cubic interpolant of the stored profile on which knots are searched
CROSSING_POINTS = 200001
# relative distance of the interpolant to a hat value within which a local minimum counts as touching it
TOUCH_TOLERANCE = 1e-3
# candidate knots closer than this in psi_n are one candidate
MERGE_DISTANCE = 1e-3
# a spline whose candidate knot combinations exceed this is not searched
MAX_KNOT_COMBINATIONS = 20000
# best ranked knot combinations that are refined
N_REFINED = 5
# if the refined knots miss RECOVERY_TOLERANCE, each knot is restarted from this many positions between its
# neighbours, for at most this many passes
N_RESCAN_STARTS = 8
MAX_RESCAN_PASSES = 3
# EFIT's mu0 of FF', twopi * tmu with tmu = 2e-7 (modules-efit.F90)
EFIT_MU0 = 4e-7 * np.pi

POLYNOMIAL = 0
TENSION_SPLINE = 6
# EFIT's defaults of the basis settings (set_defaults.f90). kppcur and kffcur are those of the full solution modes:
# the block of the k-file modes is guarded by (kdata.eq.5).and.(kdata.eq.6), which never holds.
BASIS_DEFAULTS = {
    "kppfnc": POLYNOMIAL,
    "kfffnc": POLYNOMIAL,
    "kppcur": 3,
    "kffcur": 1,
    "pcurbd": 1.0,
    "fcurbd": 1.0,
    "kppknt": 0,
    "kffknt": 0,
    "ppknt": [],
    "ffknt": [],
    "pptens": 0.0,
    "fftens": 0.0,
    "kautoknt": 0,
    "icprof": 0,
}
BASIS_KEYS = tuple(BASIS_DEFAULTS)
# kffcur, kppcur and fcurbd = pcurbd that icprof sets (set_basis_params, set_basis.f90)
ICPROF_SETTINGS = {1: (2, 2, 1.0), 2: (2, 2, 0.0), 3: (3, 2, 0.0)}
# namelist names of the settings of each profile
PROFILE_NAMES = {
    "pprime": {"kind": "kppfnc", "count": "kppcur", "edge": "pcurbd", "n_knots": "kppknt", "knots": "ppknt",
               "tension": "pptens"},
    "ffprime": {"kind": "kfffnc", "count": "kffcur", "edge": "fcurbd", "n_knots": "kffknt", "knots": "ffknt",
                "tension": "fftens"},
}


def basis_settings(values: dict) -> dict[str, dict]:
    """
    EFIT's basis settings of p' and FF' from the namelist values of a run.

    Values the namelist does not set take EFIT's defaults (`BASIS_DEFAULTS`). Then set_basis_params (set_basis.f90)
    applies: icprof = 1, 2, 3 overrides kffcur, kppcur, fcurbd and pcurbd, and a tension spline has 2 coefficients
    per knot, whatever kppcur (kffcur) the namelist sets.

    Args:
        values: Namelist values by lowercase name, those of `BASIS_KEYS` the namelist sets.

    Returns:
        `pprime` and `ffprime` settings, each with its `kind` ("polynomial" or "tension_spline") and `n_coefficients`.
        A polynomial has its `edge` (pcurbd, fcurbd), a tension spline its `knots`, its effective `tension` and
        whether EFIT moved the knots (`movable_knots`, kautoknt != 0).

    Raises:
        ValueError: For other kinds of basis, or a tension spline without its knots.
    """
    settings = {**BASIS_DEFAULTS, **values}
    if settings["icprof"] in ICPROF_SETTINGS:
        settings["kffcur"], settings["kppcur"], edge = ICPROF_SETTINGS[settings["icprof"]]
        settings["fcurbd"] = settings["pcurbd"] = edge

    profiles = {}
    for profile, names in PROFILE_NAMES.items():
        kind = settings[names["kind"]]
        if kind == POLYNOMIAL:
            profiles[profile] = {
                "kind": "polynomial",
                "n_coefficients": int(settings[names["count"]]),
                "edge": float(settings[names["edge"]]),
            }
        elif kind == TENSION_SPLINE:
            n_knots = int(settings[names["n_knots"]])
            knots = np.atleast_1d(np.asarray(settings[names["knots"]], dtype=np.float64))
            if n_knots < 2 or len(knots) < n_knots:
                raise ValueError(
                    f"{names['kind']} = 6 needs at least 2 knots, the namelist sets {names['n_knots']} = {n_knots} "
                    f"and {names['knots']} = {knots.tolist()}."
                )
            knots = knots[:n_knots]
            profiles[profile] = {
                "kind": "tension_spline",
                "n_coefficients": 2 * n_knots,
                "knots": knots,
                # pptens2 of ppbasisfunc.f90 (fftens2 of ffbasisfunc.f90)
                "tension": abs(float(settings[names["tension"]])) * (n_knots - 1) / (knots[-1] - knots[0]),
                "movable_knots": settings["kautoknt"] != 0,
            }
        else:
            raise ValueError(
                f"{names['kind']} = {kind}: only EFIT's polynomials ({POLYNOMIAL}) and tension splines "
                f"({TENSION_SPLINE}) are supported."
            )
    return profiles


def split_coefficients(coefficients: np.ndarray, settings: dict[str, dict]) -> dict[str, np.ndarray]:
    """
    The XRSP of one time slice split into the coefficients of p' and of FF'.

    Raises:
        ValueError: If XRSP does not hold exactly the coefficients of the two bases, e.g. with rotation (kwwcur),
            Er (keecur) or edge coefficients.
    """
    n_pprime, n_ffprime = settings["pprime"]["n_coefficients"], settings["ffprime"]["n_coefficients"]
    if len(coefficients) != n_pprime + n_ffprime:
        raise ValueError(
            f"XRSP holds {len(coefficients)} coefficients, the basis settings give {n_pprime} for p' (kppcur) and "
            f"{n_ffprime} for FF' (kffcur); further coefficients are not supported."
        )
    return {"pprime": coefficients[:n_pprime], "ffprime": coefficients[n_pprime:]}


def polynomial_basis(psin: np.ndarray, n_coefficients: int, edge: float, part: str = "value") -> np.ndarray:
    """
    EFIT's polynomial basis (ppbasisfunc.f90, case 0) on `psin`.

    Args:
        psin: (n_psin,) normalized flux.
        n_coefficients: Number of coefficients of the profile, EFIT's kppcur (kffcur for FF').
        edge: EFIT's pcurbd (fcurbd for FF'), 1 for a profile that vanishes at psi_n = 1 and 0 otherwise.
        part: "value" for x^(i-1) - edge x^n_coefficients, "derivative" for its derivative in psi_n, "integral" for
            its integral from psin to 1 (minus bsffin of ffbasisfunc.f90, case 0).

    Returns:
        (n_coefficients, n_psin) the basis functions i = 1 ... n_coefficients.
    """
    powers = np.arange(n_coefficients)[:, None]
    x = psin[None, :]
    n = n_coefficients
    if part == "value":
        return x**powers - edge * x**n
    if part == "derivative":
        return powers * x ** np.maximum(powers - 1, 0) - edge * n * x ** (n - 1)
    if part == "integral":
        return (1 - x ** (powers + 1)) / (powers + 1) - edge * (1 - x ** (n + 1)) / (n + 1)
    raise ValueError(f"Unknown part {part!r}, expected 'value', 'derivative' or 'integral'.")


def tension_spline_basis(psin: np.ndarray, knots: np.ndarray, tension: float, part: str = "value") -> np.ndarray:
    """
    EFIT's tension spline basis (ppbasisfunc.f90, case 6) on `psin`.

    In t, the distance from the far end of an interval of width w, the hat t / w and the bump
    (sinh(T t) / sinh(T w) - t / w) / T^2 have the derivatives 1 / w and (T cosh(T t) / sinh(T w) - 1 / w) / T^2 and
    the antiderivatives t^2 / (2 w) and (cosh(T t) / (T sinh(T w)) - t^2 / (2 w)) / T^2 (bsffin of ffbasisfunc.f90,
    case 6). An interval holds its low end but not its high end, except for the last one, so that at a knot the
    derivative of the single basis functions, which jumps there, is the one of the interval to its right.

    Args:
        psin: (n_psin,) normalized flux.
        knots: (n_knots,) increasing knot positions in psi_n.
        tension: EFIT's effective tension pptens2.
        part: "value", "derivative" in psi_n, or "integral" from psin to 1.

    Returns:
        (2 n_knots, n_psin) the hat and the tension bump of each knot, 0 outside the knots.
    """
    if part not in ("value", "derivative", "integral"):
        raise ValueError(f"Unknown part {part!r}, expected 'value', 'derivative' or 'integral'.")
    n_knots = len(knots)
    basis = np.zeros((2 * n_knots, len(psin)))
    for k in range(n_knots):
        for neighbour in (k - 1, k + 1):
            if not 0 <= neighbour < n_knots:
                continue
            low, high = sorted((knots[k], knots[neighbour]))
            width = high - low
            # dt / dpsi_n, t grows towards knot k
            slope = 1.0 if knots[k] > knots[neighbour] else -1.0
            if part == "integral":
                # int_psin^1 is the integral over [max(psin, low), high], 0 above the interval
                t_start = np.abs(np.clip(psin, low, high) - knots[neighbour])
                t_end = abs(high - knots[neighbour])
                basis[2 * k] += slope * (t_end**2 - t_start**2) / (2 * width)
                basis[2 * k + 1] += slope * (
                    (_cosh_ratio(tension, t_end, width) - _cosh_ratio(tension, t_start, width)) / tension
                    - (t_end**2 - t_start**2) / (2 * width)
                ) / tension**2
                continue
            inside = (psin >= low) & ((psin < high) | (high == knots[-1]))
            distance = np.abs(psin[inside] - knots[neighbour])
            if part == "value":
                basis[2 * k, inside] = distance / width
                basis[2 * k + 1, inside] = (_sinh_ratio(tension, distance, width) - distance / width) / tension**2
            else:
                basis[2 * k, inside] = slope / width
                basis[2 * k + 1, inside] = slope * (
                    tension * _cosh_ratio(tension, distance, width) - 1 / width
                ) / tension**2
    return basis


def _sinh_ratio(tension: float, distance: np.ndarray, width: float) -> np.ndarray:
    """sinh(tension distance) / sinh(tension width) for 0 <= distance <= width, without overflow at large tension."""
    a, b = abs(tension) * distance, abs(tension) * width
    return np.exp(a - b) * np.expm1(-2 * a) / np.expm1(-2 * b)


def _cosh_ratio(tension: float, distance: np.ndarray, width: float) -> np.ndarray:
    """cosh(tension distance) / sinh(tension width) for 0 <= distance <= width, without overflow at large tension."""
    a, b = abs(tension) * distance, abs(tension) * width
    return np.exp(a - b) * (1 + np.exp(-2 * a)) / -np.expm1(-2 * b)


def evaluate_basis(psin: np.ndarray, basis: dict, part: str = "value") -> np.ndarray:
    """
    Profile of a basis on `psin` of any shape: its value, its derivative in psi_n or its integral from psin to 1.

    Args:
        psin: Normalized flux.
        basis: `kind` and `coefficients`, with the `edge` of a "polynomial" or the `knots` and `tension` of a
            "tension_spline".
        part: "value", "derivative" or "integral".

    Returns:
        The profile, shaped as `psin`.
    """
    flat = np.ravel(np.asarray(psin, dtype=np.float64))
    if basis["kind"] == "polynomial":
        matrix = polynomial_basis(flat, len(basis["coefficients"]), basis["edge"], part)
    elif basis["kind"] == "tension_spline":
        matrix = tension_spline_basis(flat, basis["knots"], basis["tension"], part)
    else:
        raise ValueError(f"Unknown EFIT basis kind {basis['kind']!r}, expected 'polynomial' or 'tension_spline'.")
    return (basis["coefficients"] @ matrix).reshape(np.shape(psin))


def relative_rms(basis: dict, psin: np.ndarray, profile: np.ndarray) -> float:
    """rms of the basis against `profile` on `psin`, relative to max |profile|."""
    return float(np.sqrt(np.mean(((evaluate_basis(psin, basis) - profile) / _scale(profile)) ** 2)))


def _scale(profile: np.ndarray) -> float:
    """max |profile|, 1 for a profile that is 0 everywhere."""
    return float(np.abs(profile).max()) or 1.0


def resolve_basis(
    settings: dict, coefficients: np.ndarray, psin: np.ndarray, profile: np.ndarray
) -> tuple[dict, float, bool]:
    """
    EFIT's basis of one profile from its settings and coefficients, checked against the profile EFIT stored.

    EFIT can move the knots of a tension spline during the fit (kautoknt) and stores only the knots it started from.
    If the knots of the namelist miss the stored profile by more than `RECOVERY_TOLERANCE` and EFIT moved them, the
    interior knots are fitted to the stored profile, while the end knots and the tension stay those of the namelist:
    first by least squares from the knots of the namelist, then by the candidate search of `_search_knots`.

    Args:
        settings: Settings of the profile, from `basis_settings`.
        coefficients: (n_coefficients,) coefficients of the profile, scaled to the units of `profile`.
        psin: (n_psin,) normalized flux of the stored profile.
        profile: (n_psin,) stored profile.

    Returns:
        The basis (`kind`, `coefficients`, and `edge` or `knots` and `tension`), its relative rms against `profile`
        (`relative_rms`) and whether the knots were fitted. The caller decides whether the rms is acceptable.
    """
    if settings["kind"] == "polynomial":
        basis = {"kind": "polynomial", "edge": settings["edge"], "coefficients": coefficients}
        return basis, relative_rms(basis, psin, profile), False

    basis = {
        "kind": "tension_spline",
        "knots": settings["knots"],
        "tension": settings["tension"],
        "coefficients": coefficients,
    }
    rms = relative_rms(basis, psin, profile)
    if rms <= RECOVERY_TOLERANCE or not settings["movable_knots"] or len(basis["knots"]) < 3:
        return basis, rms, False

    fitted = _with_interior_knots(basis, _refine_knots(basis, psin, profile, basis["knots"][1:-1]).x)
    if relative_rms(fitted, psin, profile) > RECOVERY_TOLERANCE:
        fitted = _search_knots(basis, psin, profile)
    if fitted is None:
        return basis, rms, False
    return fitted, relative_rms(fitted, psin, profile), True


def _with_interior_knots(basis: dict, interior: np.ndarray) -> dict:
    """`basis` with the interior knots `interior` between its end knots."""
    return {**basis, "knots": np.r_[basis["knots"][0], interior, basis["knots"][-1]]}


def _knot_residual(interior: np.ndarray, basis: dict, psin: np.ndarray, profile: np.ndarray) -> np.ndarray:
    """Residual of the spline with the interior knots `interior` against `profile`, relative to max |profile|."""
    return (evaluate_basis(psin, _with_interior_knots(basis, interior)) - profile) / _scale(profile)


def _refine_knots(basis: dict, psin: np.ndarray, profile: np.ndarray, start: np.ndarray):
    """Levenberg-Marquardt fit of the interior knots of `basis` to `profile` from `start`."""
    return least_squares(_knot_residual, start, args=(basis, psin, profile), method="lm")


def _search_knots(basis: dict, psin: np.ndarray, profile: np.ndarray) -> Optional[dict]:
    """
    Interior knots of a tension spline from the stored profile, with the coefficients and tension of `basis`.

    Every hat coefficient is the profile at its knot, so the interior knots are among the points where the profile
    takes these values: EFIT keeps the first derivative continuous at the knots, so the profile crosses the value
    there, or touches it at an extremum. The increasing combinations of candidates are ranked and the best
    `N_REFINED` refined. If that misses `RECOVERY_TOLERANCE`, the knots are restarted one at a time between their
    neighbours: near a steep pedestal the cubic interpolant can place a candidate outside the basin of its knot
    (169621 at 1.44 s: 0.9956 for 0.9933).

    Returns:
        The basis with the best interior knots found, None if a hat value is never taken or there are too many
        candidates.
    """
    coefficients = basis["coefficients"]
    first, last = basis["knots"][0], basis["knots"][-1]
    n_knots = len(basis["knots"])
    scale = _scale(profile)

    dense = np.linspace(first, last, CROSSING_POINTS)
    values = CubicSpline(psin, profile)(dense)
    candidates = []
    for hat in coefficients[2:-2:2]:
        offset = values - hat
        crossings = np.flatnonzero(np.diff(np.sign(offset)) != 0)
        # an extremum on the knot touches the value without crossing it
        distance = np.abs(offset)
        touching = 1 + np.flatnonzero(
            (distance[1:-1] <= distance[:-2])
            & (distance[1:-1] <= distance[2:])
            & (distance[1:-1] < TOUCH_TOLERANCE * scale)
        )
        found = np.union1d(dense[crossings], dense[touching])
        if not len(found):
            return None
        candidates.append(found[np.r_[True, np.diff(found) > MERGE_DISTANCE]])
    if np.prod([len(knots) for knots in candidates], dtype=float) > MAX_KNOT_COMBINATIONS:
        return None

    ranked = sorted(
        (
            (np.linalg.norm(_knot_residual(np.asarray(interior), basis, psin, profile)), np.asarray(interior))
            for interior in product(*candidates)
            if np.all(np.diff(np.r_[first, interior, last]) > 0)
        ),
        key=lambda item: item[0],
    )
    best = None
    for _, interior in ranked[:N_REFINED]:
        result = _refine_knots(basis, psin, profile, interior)
        if best is None or result.cost < best.cost:
            best = result
    if best is None:
        return None

    for _ in range(MAX_RESCAN_PASSES):
        if np.sqrt(2 * best.cost / len(profile)) < RECOVERY_TOLERANCE:
            break
        improved = False
        for k in range(n_knots - 2):
            knots = np.r_[first, best.x, last]
            for position in np.linspace(knots[k], knots[k + 2], N_RESCAN_STARTS + 2)[1:-1]:
                start = best.x.copy()
                start[k] = position
                result = _refine_knots(basis, psin, profile, start)
                if result.cost < best.cost:
                    best, improved = result, True
        if not improved:
            break
    return _with_interior_knots(basis, best.x)


def format_floats(values: np.ndarray) -> str:
    """Values in their shortest exact scientific form (for their dtype), separated by spaces."""
    return " ".join(np.format_float_scientific(value, unique=True) for value in values)


def _format_attribute(value: float) -> str:
    """A float of an XML attribute in its shortest exact positional form, e.g. 0, 0.3, 10."""
    return np.format_float_positional(np.float64(value), unique=True, trim="-")


def write_basis(parent: ET.Element, profile: str, basis: dict, fitted_knots: bool) -> None:
    """
    Append the basis of one profile to the <time_slice> `parent` of <basis>.

    Writes <pprime> or <ffprime> with the coefficients as text and the `kind`, with the `edge` of a polynomial, or the
    `knots`, the effective `tension` and whether the knots were fitted (`fitted_knots`) of a tension spline.
    """
    element = ET.SubElement(parent, profile, kind=basis["kind"])
    if basis["kind"] == "polynomial":
        element.set("edge", _format_attribute(basis["edge"]))
    else:
        element.set("tension", _format_attribute(basis["tension"]))
        element.set("knots", " ".join(_format_attribute(knot) for knot in basis["knots"]))
        element.set("fitted_knots", str(fitted_knots).lower())
    element.text = format_floats(np.asarray(basis["coefficients"], dtype=np.float64))


def parse_basis(code_parameters: str, i_time: int) -> dict[str, dict]:
    """
    The bases of p' and FF' of one time slice from the XML of equilibrium.code.parameters.

    Args:
        code_parameters: XML of equilibrium.code.parameters.
        i_time: Index of the equilibrium time slice.

    Returns:
        `pprime` and `ffprime` bases for `evaluate_basis`.

    Raises:
        ValueError: If the time slices are not numbered 0 ... n - 1 or `i_time` is not one of them.
    """
    slices = ET.fromstring(code_parameters).find("basis").findall("time_slice")
    indices = [int(element.get("index")) for element in slices]
    if indices != list(range(len(slices))):
        raise ValueError(f"The basis time slices are not numbered 0..{len(slices) - 1}: {indices}.")
    if not 0 <= i_time < len(slices):
        raise ValueError(f"Time slice {i_time} is outside the {len(slices)} time slices of the basis.")
    return {profile: _parse_profile(slices[i_time].find(profile)) for profile in ("pprime", "ffprime")}


def _parse_profile(element: ET.Element) -> dict:
    """The basis of one profile from its <pprime> or <ffprime> element."""
    basis = {"kind": element.get("kind"), "coefficients": np.array(element.text.split(), dtype=np.float64)}
    if basis["kind"] == "polynomial":
        basis["edge"] = float(element.get("edge"))
    elif basis["kind"] == "tension_spline":
        basis["knots"] = np.array(element.get("knots").split(), dtype=np.float64)
        basis["tension"] = float(element.get("tension"))
    else:
        raise ValueError(f"Unknown EFIT basis kind {basis['kind']!r}, expected 'polynomial' or 'tension_spline'.")
    return basis


def _check_psi_n(psi_n: np.ndarray) -> np.ndarray:
    """`psi_n` as float64, which EFIT's bases cover from 0 to 1."""
    psi_n = np.asarray(psi_n, dtype=np.float64)
    if np.any((psi_n < 0) | (psi_n > 1)):
        raise ValueError(f"psi_n must lie in [0, 1], it covers {psi_n.min()} ... {psi_n.max()}.")
    return psi_n


def efit_profile_basis(code_parameters: str, i_time: int) -> Callable[..., tuple[np.ndarray, np.ndarray]]:
    """
    p' and FF' of EFIT's basis for time slice `i_time` of equilibrium.code.parameters.

    The callable takes psi_n (any shape, 0 <= psi_n <= 1) and `derivative`, and returns (pprime, ffprime) in the
    units and COCOS of profiles_1d.dpressure_dpsi and f_df_dpsi, or their derivatives d/dpsi_n with derivative=True.
    It is a `functools.partial` of `evaluate_profiles` and pickles.
    """
    basis = parse_basis(code_parameters, i_time)
    return functools.partial(evaluate_profiles, basis["pprime"], basis["ffprime"])


def evaluate_profiles(
    pprime: dict, ffprime: dict, psi_n: np.ndarray, derivative: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """p' and FF' of the bases `pprime` and `ffprime` on `psi_n`, or their derivatives d/dpsi_n."""
    psi_n = _check_psi_n(psi_n)
    part = "derivative" if derivative else "value"
    return evaluate_basis(psi_n, pprime, part), evaluate_basis(psi_n, ffprime, part)


def efit_f_profile(
    code_parameters: str, i_time: int, f_boundary: float, psi_axis: float, psi_boundary: float
) -> Callable[[np.ndarray], np.ndarray]:
    """
    F of EFIT's FF' basis for time slice `i_time` of equilibrium.code.parameters.

    F^2(psi_n) = F_b^2 - 2 (psi_b - psi_a) int_psi_n^1 FF' dpsi_n, with the integral of the basis in closed form.
    EFIT's stored FPOL is the same relation with the trapezoidal rule on its nw points (ece.F90), so the two differ by
    the error of that rule. The arguments are the mapped profiles_1d.f at the boundary and global_quantities.psi_axis
    and psi_boundary of the slice, in the COCOS of f_df_dpsi, in which (psi_b - psi_a) FF' is invariant.

    The callable takes psi_n (any shape, 0 <= psi_n <= 1) and returns F in the units and COCOS of profiles_1d.f. It is
    a `functools.partial` of `evaluate_f` and pickles.
    """
    ffprime = parse_basis(code_parameters, i_time)["ffprime"]
    return functools.partial(evaluate_f, ffprime, float(f_boundary), float(psi_boundary) - float(psi_axis))


def evaluate_f(ffprime: dict, f_boundary: float, delta_psi: float, psi_n: np.ndarray) -> np.ndarray:
    """F on `psi_n` from the FF' basis, its boundary value and psi_b - psi_a. Where F^2 would be negative, EFIT keeps
    the boundary value (ece.F90), and so does this."""
    f_squared = f_boundary**2 - 2 * delta_psi * evaluate_basis(_check_psi_n(psi_n), ffprime, "integral")
    return np.where(f_squared >= 0, np.sign(f_boundary) * np.sqrt(np.maximum(f_squared, 0)), f_boundary)
