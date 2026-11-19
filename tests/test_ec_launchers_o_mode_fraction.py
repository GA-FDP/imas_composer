"""
Test the O-mode fraction of the EC beams, 1 - XMFRAC on the AZIANG time base.

The test shots all record XMFRAC, so the pure X-mode default for shots without it is tested here.
"""
import numpy as np
import pytest

from imas_composer.core import Requirement
from imas_composer.ids.ec_launchers import EcLaunchersMapper

SHOT = 200000
GYROTRON = "LEIA"
XMFRAC_PATH = f".ECH.{GYROTRON}.ECLEIXMFRAC"


@pytest.fixture
def mapper():
    return EcLaunchersMapper()


def make_raw_data(time, xmfrac):
    raw_data = {
        Requirement(".ECH.NUM_SYSTEMS", SHOT, "RF").as_key(): 1,
        Requirement(".ECH.SYSTEM_1.GYROTRON.NAME", SHOT, "RF").as_key(): GYROTRON,
        Requirement(f".ECH.{GYROTRON}.ECLEISTAT", SHOT, "RF").as_key(): 1,
        Requirement(f"dim_of(.ECH.{GYROTRON}.ECLEIAZIANG+01) / 1E3", SHOT, "RF").as_key(): np.asarray(time),
    }
    if xmfrac is not None:
        raw_data[Requirement(XMFRAC_PATH, SHOT, "RF").as_key()] = xmfrac
    return raw_data


@pytest.mark.parametrize(
    "time, xmfrac, expected",
    [
        ([0.1, 0.2, 0.3], np.array([1.0, 0.25, 0.0]), [0.0, 0.75, 1.0]),
        ([0.1, 0.2, 0.3], Exception("%TREE-E-NODATA"), [0.0, 0.0, 0.0]),
        ([0.1, 0.2, 0.3], None, [0.0, 0.0, 0.0]),
        ([0.1], Exception("%TREE-E-NODATA"), [0.0]),
    ],
    ids=["recorded", "not_recorded", "not_fetched", "not_recorded_single_time"],
)
def test_o_mode_fraction(mapper, time, xmfrac, expected):
    fraction = mapper._compose_o_mode_fraction(SHOT, make_raw_data(time, xmfrac))
    np.testing.assert_array_equal(fraction, [expected])
