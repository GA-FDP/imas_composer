"""
Test the GTIME -> MTIME index mapping used by all equilibrium constraint fields.

Between-shot EFIT filters remove slices from RESULTS but not from MEASUREMENTS, so MTIME
can be longer than GTIME. Each equilibrium slice must read the MEASUREMENTS row at its own time.
"""
import numpy as np
import pytest

from imas_composer.core import Requirement
from imas_composer.fetchers import fetch_requirements
from imas_composer.ids.equilibrium import EquilibriumMapper

SHOT = 200000


@pytest.fixture
def mapper():
    return EquilibriumMapper()


def make_raw_data(mapper, gtime, mtime):
    return {
        Requirement(f"{mapper.geqdsk_node}.GTIME", SHOT, mapper.efit_tree).as_key(): np.asarray(gtime, dtype=float),
        Requirement(f"{mapper.measurements_node}.MTIME", SHOT, mapper.efit_tree).as_key(): np.asarray(mtime, dtype=float),
    }


@pytest.mark.parametrize(
    "gtime, mtime, expected",
    [
        ([100, 120, 140], [100, 120, 140], [0, 1, 2]),
        ([100, 140, 160], [100, 120, 140, 160], [0, 2, 3]),
        # Old clamp to len(GTIME) - 1 mapped the last slice to row 1
        ([100, 120], [100, 120, 140, 160], [0, 1]),
        ([120, 140], [100, 120, 140, 160], [1, 2]),
        ([100, 160, 200], [100, 120, 140, 160, 180, 200, 220], [0, 3, 5]),
    ],
    ids=["equal_lengths", "removed_in_middle", "removed_at_end", "removed_at_start_and_end", "removed_everywhere"],
)
def test_constraint_time_indices_mapping(mapper, gtime, mtime, expected):
    indices = mapper._compose_constraint_time_indices(SHOT, make_raw_data(mapper, gtime, mtime))
    np.testing.assert_array_equal(indices, expected)


@pytest.mark.parametrize(
    "gtime, mtime",
    [
        ([100, 130], [100, 120, 140]),
        ([100, 160], [100, 120, 140]),
        ([100, 121], [100, 121.5, 140]),
    ],
    ids=["between_mtimes", "past_last_mtime", "outside_tolerance"],
)
def test_constraint_time_indices_raise_on_unmatched_gtime(mapper, gtime, mtime):
    with pytest.raises(ValueError, match="no matching MTIME"):
        mapper._compose_constraint_time_indices(SHOT, make_raw_data(mapper, gtime, mtime))


def test_constraint_indices_match_pprime_on_axis(mapper, test_shot):
    """
    The first p' coefficient in MEASUREMENTS (XRSP[:, 0]) is p' on axis, which GEQDSK also
    stores as PPRIME[:, 0] (up to sign). Every slice must read the XRSP row of its own time.
    """
    requirements = [
        Requirement(f"{mapper.geqdsk_node}.GTIME", test_shot, mapper.efit_tree),
        Requirement(f"{mapper.measurements_node}.MTIME", test_shot, mapper.efit_tree),
        Requirement(f"{mapper.measurements_node}.XRSP", test_shot, mapper.efit_tree),
        Requirement(f"{mapper.geqdsk_node}.PPRIME", test_shot, mapper.efit_tree),
    ]
    raw_data = fetch_requirements(requirements)
    xrsp, pprime = (np.asarray(raw_data[req.as_key()]) for req in requirements[2:])

    indices = mapper._compose_constraint_time_indices(test_shot, raw_data)

    np.testing.assert_allclose(np.abs(xrsp[indices, 0]), np.abs(pprime[:, 0]), rtol=1e-5)
