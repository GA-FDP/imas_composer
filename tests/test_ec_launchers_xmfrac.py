"""
Test the ec_launchers fields read from XMFRAC, the X-mode power fraction of each gyrotron.

XMFRAC is not recorded on older shots (170000 in the OMAS comparison). Its NoData means pure
X-mode, as in OMAS; any other fetch error must raise instead of being served as X-mode.
"""
import numpy as np
import pytest

from imas_composer import ImasComposer, NoData

SHOT = 200000
XMFRAC_FIELDS = ["ec_launchers.beam.mode", "ec_launchers.beam.o_mode_fraction"]


def gyrotron_value(mds_path, time, xmfrac):
    if mds_path == ".ECH.NUM_SYSTEMS":
        return 1
    if mds_path.endswith("GYROTRON.NAME"):
        return "LEIA"
    if mds_path.endswith("STAT"):
        return 1
    if mds_path.endswith("XMFRAC"):
        return xmfrac
    if mds_path.startswith("dim_of"):
        return np.asarray(time)
    return np.ones(len(time))


def gyrotron_raw_data(composer, ids_paths, time, xmfrac):
    """One active gyrotron with every signal on the given time base."""
    raw_data = {}
    status, requirements = composer.resolve(ids_paths, SHOT, raw_data)
    while not all(status.values()):
        raw_data.update({req.as_key(): gyrotron_value(req.mds_path, time, xmfrac) for req in requirements})
        status, requirements = composer.resolve(ids_paths, SHOT, raw_data)
    return raw_data


@pytest.mark.parametrize(
    "time, xmfrac, mode, o_mode_fraction",
    [
        ([0.1, 0.2, 0.3], np.array([0.9, 0.75, 1.0]), -1, [0.1, 0.25, 0.0]),
        ([0.1, 0.2, 0.3], np.array([0.0, 0.1, 0.2]), 1, [1.0, 0.9, 0.8]),
        ([0.1, 0.2, 0.3], NoData("%TREE-E-NODATA, No data available for this node"), -1, [0.0, 0.0, 0.0]),
        ([0.1], NoData("%TREE-E-NODATA, No data available for this node"), -1, [0.0]),
    ],
    ids=["x_mode", "o_mode", "not_recorded", "not_recorded_single_time"],
)
def test_xmfrac_fields(time, xmfrac, mode, o_mode_fraction):
    composer = ImasComposer()
    results = composer.compose(XMFRAC_FIELDS, SHOT, gyrotron_raw_data(composer, XMFRAC_FIELDS, time, xmfrac))
    np.testing.assert_array_equal(results["ec_launchers.beam.mode"], [mode])
    np.testing.assert_allclose(results["ec_launchers.beam.o_mode_fraction"], [o_mode_fraction], atol=1e-15)


@pytest.mark.parametrize("ids_path", XMFRAC_FIELDS)
def test_xmfrac_fetch_error_raises(ids_path):
    composer = ImasComposer()
    error = ConnectionError("Error connecting to atlas.gat.com:8000")
    raw_data = gyrotron_raw_data(composer, [ids_path], [0.1, 0.2, 0.3], error)
    with pytest.raises(TypeError, match="ConnectionError"):
        composer.compose([ids_path], SHOT, raw_data)
