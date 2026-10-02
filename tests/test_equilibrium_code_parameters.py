"""
Test equilibrium.code.parameters, the XML holding EFIT's p' and FF' basis function coefficients (XRSP).

OMAS has no mapping for it. The coefficients are checked against the p' profile: every basis function EFIT
offers for p' (polynomial or spline, ppbasisfunc.f90) vanishes on the magnetic axis except the first one,
which is 1 there, so p' at psi_n = 0 is the first coefficient. The g-file PPRIME is that sum (shapesurf.F90),
dpressure_dpsi is PPRIME / (2 pi) in COCOS 11 up to the sign.
"""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from imas_composer import ImasComposer
from imas_composer.fetchers import fetch_requirements

pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus]

CODE_PARAMETERS = "equilibrium.code.parameters"
TIME = "equilibrium.time"
PPRIME = "equilibrium.time_slice.profiles_1d.dpressure_dpsi"
# float32 storage of both XRSP and PPRIME
AXIS_TOLERANCE = 1e-6
# 237 GEQDSK and 282 MEASUREMENTS slices, see test_late_slices_map_to_their_own_measurements
UNEQUAL_TIME_BASE_SHOT = 202161


def compose(composer, shot: int) -> tuple[dict, dict]:
    """`CODE_PARAMETERS`, `TIME` and `PPRIME` of `shot`, and the raw data they were composed from."""
    paths = [CODE_PARAMETERS, TIME, PPRIME]
    raw_data = {}
    for _ in range(10):
        status, requirements = composer.resolve(paths, shot, raw_data)
        if all(status.values()):
            break
        for key, value in fetch_requirements(requirements).items():
            if isinstance(value, Exception):
                pytest.skip(f"MDSplus data unavailable for {key}: {value}")
            raw_data[key] = value
    return composer.compose(paths, shot, raw_data), raw_data


def parse_coefficients(xml: str) -> list[np.ndarray]:
    """XRSP of each time slice, parsed from the XML."""
    slices = ET.fromstring(xml).find("xrsp").findall("time_slice")
    assert [int(element.get("index")) for element in slices] == list(range(len(slices)))
    return [np.array(element.text.split(), dtype=np.float64) for element in slices]


def axis_mismatch(composed: dict) -> np.ndarray:
    """Relative difference of |xrsp_1| and 2 pi |dpressure_dpsi(psi_n = 0)| per slice."""
    pprime_axis = 2 * np.pi * np.abs(np.asarray(composed[PPRIME])[:, 0])
    first = np.abs([coefficients[0] for coefficients in parse_coefficients(composed[CODE_PARAMETERS])])
    return np.abs(first - pprime_axis) / pprime_axis


def raw_time_base(raw_data: dict, node: str) -> np.ndarray:
    return np.asarray(next(value for key, value in raw_data.items() if key[0].endswith(node)))


@pytest.fixture
def composed(composer, test_shot) -> tuple[dict, dict]:
    return compose(composer, test_shot)


def test_one_entry_per_equilibrium_time_slice(composed):
    data, _ = composed
    coefficients = parse_coefficients(data[CODE_PARAMETERS])
    assert len(coefficients) == len(data[TIME])
    for slice_coefficients in coefficients:
        assert len(slice_coefficients) > 0
        assert np.all(np.isfinite(slice_coefficients)) and np.all(np.abs(slice_coefficients) < 1e30)


def test_first_coefficient_is_pprime_on_axis(composed):
    """|xrsp_1| = 2 pi |dpressure_dpsi(psi_n = 0)| on every slice whose MEASUREMENTS slice is taken at its time."""
    data, raw_data = composed
    gtime = raw_time_base(raw_data, "GEQDSK.GTIME")
    mtime = raw_time_base(raw_data, "MEASUREMENTS.MTIME")
    # slices whose MEASUREMENTS index lies beyond the clamp of the time mapping are covered by the xfail below
    unaffected = (mtime.searchsorted(gtime) < len(gtime) - 1) if len(gtime) != len(mtime) else np.full(len(gtime), True)
    mismatch = axis_mismatch(data)[unaffected]
    assert np.all(mismatch < AXIS_TOLERANCE), f"worst slice deviates by {mismatch.max():.1e}"


@pytest.mark.xfail(
    strict=True,
    reason="pre-existing: _compose_constraint_time_indices clamps searchsorted(GTIME) to len(GTIME) - 1 instead of "
    "len(MTIME) - 1 (copied from OMAS _common.py). On 202161 (237 GEQDSK, 282 MEASUREMENTS slices) slices 216-236 "
    "all read MEASUREMENTS row 236, for every constraint field, not only XRSP",
)
def test_late_slices_map_to_their_own_measurements():
    data, _ = compose(ImasComposer(), UNEQUAL_TIME_BASE_SHOT)
    mismatch = axis_mismatch(data)
    assert np.all(mismatch < AXIS_TOLERANCE), f"{np.sum(mismatch >= AXIS_TOLERANCE)} slices read another slice's XRSP"
