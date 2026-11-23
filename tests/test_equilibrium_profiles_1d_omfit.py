"""
Test the equilibrium profiles_1d fields that OMAS does not map.

Inside the plasma the profiles are compared with an OMFIT reference (OMFITgeqdsk.to_omas), stored in
equilibrium_profiles_1d_omfit_reference.yaml by scripts/generate_omfit_equilibrium_reference.py because OMFIT is
not available in CI. The reference only holds already published slices (EFIT02, see the script). OMFIT traces the
flux surfaces with its own code, so the comparison has tolerances and starts at the fourth psi point: the innermost contours have only a few vertices, so their shape
is ill-defined. The values on the axis and at the boundary come from EFIT scalars and are tested against the
fields that carry them.
"""
from pathlib import Path

import numpy as np
import pytest
import yaml

from imas_composer import ImasComposer
from tests.conftest import resolve_and_compose, _missing_to_nan


REFERENCE = yaml.safe_load((Path(__file__).parent / "equilibrium_profiles_1d_omfit_reference.yaml").read_text())
PROFILES_1D = "equilibrium.time_slice.profiles_1d."
TIME_SLICE = "equilibrium.time_slice."

# (atol, rtol): largest difference to OMFIT on the reference slices, with margin
OMFIT_TOLERANCES = {
    "elongation": (1e-2, 0),
    "triangularity_upper": (3e-2, 0),
    "triangularity_lower": (3e-2, 0),
    "r_inboard": (3e-3, 0),
    "r_outboard": (3e-3, 0),
    "geometric_axis.r": (3e-3, 0),
    "geometric_axis.z": (3e-3, 0),
    "trapped_fraction": (1.5e-2, 0),
    "gm5": (0, 2e-2),
    "b_field_average": (0, 1e-2),
}
FIRST_COMPARED_INDEX = 3
# Composed from EFIT02 FLUXFUN, which atlas serves for the reference shots but FDP does not (yet)
ATLAS_ONLY_FIELDS = ["trapped_fraction", "gm5", "b_field_average"]

# Profile on the axis (column 0) -> field it equals, and how
AXIS_VALUES = {
    "triangularity_upper": (None, lambda value: np.zeros_like(value)),
    "triangularity_lower": (None, lambda value: np.zeros_like(value)),
    "trapped_fraction": (None, lambda value: np.zeros_like(value)),
    "r_inboard": ("global_quantities.magnetic_axis.r", lambda value: value),
    "r_outboard": ("global_quantities.magnetic_axis.r", lambda value: value),
    "geometric_axis.r": ("global_quantities.magnetic_axis.r", lambda value: value),
    "geometric_axis.z": ("global_quantities.magnetic_axis.z", lambda value: value),
    "b_field_average": ("global_quantities.magnetic_axis.b_field_tor", np.abs),
    "gm5": ("global_quantities.magnetic_axis.b_field_tor", np.square),
}
# Profile at the boundary (last column) -> boundary_separatrix field it equals, and how
BOUNDARY_VALUES = {
    "elongation": ("boundary_separatrix.elongation", lambda value, _: value),
    "triangularity_upper": ("boundary_separatrix.triangularity_upper", lambda value, _: value),
    "triangularity_lower": ("boundary_separatrix.triangularity_lower", lambda value, _: value),
    "geometric_axis.r": ("boundary_separatrix.geometric_axis.r", lambda value, _: value),
    "geometric_axis.z": ("boundary_separatrix.geometric_axis.z", lambda value, _: value),
    "r_inboard": ("boundary_separatrix.geometric_axis.r", lambda value, minor_radius: value - minor_radius),
    "r_outboard": ("boundary_separatrix.geometric_axis.r", lambda value, minor_radius: value + minor_radius),
}

COMPOSED_PATHS = (
    [PROFILES_1D + field for field in OMFIT_TOLERANCES]
    + [PROFILES_1D + "psi_norm", PROFILES_1D + "psi", TIME_SLICE + "time",
       TIME_SLICE + "boundary_separatrix.minor_radius"]
    + sorted({TIME_SLICE + path for path, _ in AXIS_VALUES.values() if path is not None}
             | {TIME_SLICE + path for path, _ in BOUNDARY_VALUES.values()})
)


@pytest.fixture(scope="module")
def composed(composer):
    """Compose all fields of a shot once and return them as float arrays (missing slices are NaN)."""
    cache = {}

    def _composed(shot):
        if shot not in cache:
            results = resolve_and_compose(composer, COMPOSED_PATHS, shot)
            cache[shot] = {path: np.asarray(_missing_to_nan(value), dtype=float) for path, value in results.items()}
        return cache[shot]

    return _composed


@pytest.fixture(scope="module")
def reference_composer():
    return ImasComposer(efit_tree=REFERENCE["efit_tree"])


@pytest.mark.parametrize("reference_slice", REFERENCE["slices"], ids=lambda reference_slice: str(reference_slice["shot"]))
@pytest.mark.parametrize("field", [
    pytest.param(field, marks=pytest.mark.requires_atlas) if field in ATLAS_ONLY_FIELDS else field
    for field in OMFIT_TOLERANCES
])
def test_matches_omfit(field, reference_slice, reference_composer):
    """Profiles inside the plasma match OMFIT on the reference slices."""
    shot = reference_slice["shot"]
    profiles = resolve_and_compose(reference_composer, [PROFILES_1D + field, TIME_SLICE + "time"], shot)
    times = np.asarray(profiles[TIME_SLICE + "time"])
    time_index = int(np.argmin(np.abs(times - reference_slice["time"])))
    assert times[time_index] == pytest.approx(reference_slice["time"], abs=1e-5)

    atol, rtol = OMFIT_TOLERANCES[field]
    np.testing.assert_allclose(
        np.asarray(_missing_to_nan(profiles[PROFILES_1D + field]), dtype=float)[time_index, FIRST_COMPARED_INDEX:-1],
        np.array(reference_slice["profiles"][field])[FIRST_COMPARED_INDEX:-1],
        rtol=rtol, atol=atol, err_msg=f"{field} differs from OMFIT for shot {shot}",
    )


@pytest.mark.parametrize("field", list(AXIS_VALUES))
def test_axis_value(field, test_shot, composed):
    """The axis column is zero or equals the magnetic axis quantity."""
    profiles = composed(test_shot)
    path, transform = AXIS_VALUES[field]
    axis = profiles[PROFILES_1D + field][:, 0]
    expected = transform(axis if path is None else profiles[TIME_SLICE + path])
    np.testing.assert_allclose(axis, expected, rtol=1e-6, atol=0)


def test_axis_elongation(test_shot, composed):
    """The axis elongation (AEQDSK.KAPPA0) is close to that of the innermost traced surface."""
    elongation = composed(test_shot)[PROFILES_1D + "elongation"]
    valid = np.isfinite(elongation[:, :2]).all(axis=1)
    assert np.median(np.abs(elongation[valid, 0] - elongation[valid, 1])) < 0.05


@pytest.mark.parametrize("field", list(BOUNDARY_VALUES))
def test_boundary_value(field, test_shot, composed):
    """The boundary column equals the boundary_separatrix quantity."""
    profiles = composed(test_shot)
    path, transform = BOUNDARY_VALUES[field]
    expected = transform(profiles[TIME_SLICE + path], profiles[TIME_SLICE + "boundary_separatrix.minor_radius"])
    np.testing.assert_allclose(profiles[PROFILES_1D + field][:, -1], expected, rtol=1e-6, atol=0)


@pytest.mark.parametrize("field", ["r_inboard", "r_outboard", "geometric_axis.r", "geometric_axis.z"])
def test_boundary_continues_profile(field, test_shot, composed):
    """
    The AEQDSK boundary lengths (RSURF, ZSURF, AMINOR) continue the traced surfaces, so they are in metres too.

    The X-point moves the boundary's lowest point, so geometric_axis.z jumps by ~3 cm from the last traced surface;
    a length in cm read as m would jump by metres.
    """
    profile = composed(test_shot)[PROFILES_1D + field]
    assert np.nanmedian(np.abs(profile[:, -1] - profile[:, -2])) < 0.1


def test_psi_norm(test_shot, composed):
    """psi_norm is profiles_1d.psi normalised between axis and boundary."""
    profiles = composed(test_shot)
    psi = profiles[PROFILES_1D + "psi"]
    expected = (psi - psi[:, :1]) / (psi[:, -1:] - psi[:, :1])
    np.testing.assert_allclose(profiles[PROFILES_1D + "psi_norm"], expected, rtol=0, atol=1e-6)
