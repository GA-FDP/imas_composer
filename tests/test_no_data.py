"""
Test handling of missing MDSplus data.

Pins down how MDSplus/OMAS report the "no data" failure modes we map to NoData:
missing trees, empty (e.g. rejected) nodes, and nodes that did not exist yet
on old shots.
"""
import re

import awkward as ak
import pytest
from omas import mdsvalue

from imas_composer import ImasComposer, NoData, Requirement, fetch_requirements, simple_load
from imas_composer.fetchers import _as_no_data


# (treename, shot, mds_path, expected MDSplus status code)
FAILURE_MODES = [
    # Missing tree: the whole mdsvalue call raises
    pytest.param("OMFIT_PROFS", 118162, "\\TOP.PROFILES.ITEMPFIT", "FOPENR", id="missing_tree"),
    # Empty node: actively rejected CER channel
    pytest.param("IONS", 200000, "\\IONS::TOP.CER.CERQUICK.VERTICAL.CHANNEL08.ROTC", "NODATA",
                 id="rejected_cer_channel"),
    # Empty node on an old shot
    pytest.param("EFIT01", 118162, "\\EFIT01::TOP.MEASUREMENTS.SIGPASMA", "NODATA", id="old_shot_empty_node"),
    # Node did not exist yet: getMany reports a bare missing node as INVTREE instead of NNF
    pytest.param("NB", 111203, "\\pinj0", "INVTREE", id="old_shot_missing_node"),
    pytest.param("IONS", 118162, "\\IONS::TOP.IMPDENS.CERAUTO.ZEFF", "INVTREE", id="old_shot_missing_subtree"),
    # Node did not exist yet, wrapped in an expression: getMany reports NNF
    pytest.param("NB", 118162, "dim_of(\\NB::TOP.NB30L.PINJ_30L, 0)/1E3", "NNF",
                 id="old_shot_missing_node_expression"),
]


@pytest.mark.integration
@pytest.mark.requires_mdsplus
@pytest.mark.parametrize("treename, shot, mds_path, code", FAILURE_MODES)
def test_raw_mdsvalue_failure_mode(treename, shot, mds_path, code):
    """MDSplus/OMAS report the failure mode with the expected status code."""
    try:
        error = mdsvalue("d3d", treename=treename, pulse=shot, TDI={mds_path: mds_path}).raw()[mds_path]
    except Exception as e:
        error = e
    assert isinstance(error, Exception), f"Expected an error, got data: {error!r}"
    assert re.search(rf"%TREE-[A-Z]-{code}\b", str(error)), str(error)


@pytest.mark.integration
@pytest.mark.requires_mdsplus
@pytest.mark.parametrize("treename, shot, mds_path, code", FAILURE_MODES)
def test_fetch_requirements_returns_no_data(treename, shot, mds_path, code):
    """fetch_requirements stores missing data as NoData with the original MDSplus message."""
    req = Requirement(mds_path, shot, treename)
    value = fetch_requirements([req])[req.as_key()]
    assert isinstance(value, NoData), repr(value)
    assert re.search(rf"%TREE-[A-Z]-{code}\b", str(value)), str(value)


@pytest.mark.parametrize("error, is_no_data", [
    (Exception("%TREE-E-FOPENR, Error opening file read-only."), True),
    (Exception("%TREE-E-NODATA, No data available for this node"), True),
    (Exception("%TREE-W-NNF, Node Not Found"), True),
    (Exception("%TREE-E-INVTREE, Invalid tree identification structure"), True),
    (Exception("%TDI-E-SYNTAX, Bad punctuation or misspelled word or number"), False),
    (ConnectionError("Error connecting to atlas.gat.com:8000"), False),
])
def test_as_no_data_classification(error, is_no_data):
    """Only MDSplus "no data" status codes are converted to NoData."""
    result = _as_no_data(error)
    if is_no_data:
        assert isinstance(result, NoData)
        assert str(result) == str(error)
    else:
        assert result is error


@pytest.fixture
def no_data():
    return NoData("%TREE-E-NODATA, No data available for this node")


@pytest.mark.parametrize("ids_path", [
    # Pass-through of a single MDSplus node
    "equilibrium.time_slice.boundary_separatrix.elongation",
    # Computed from a ptdata signal
    "tf.b_field_tor_vacuum_r.data",
])
def test_compose_returns_no_data(ids_path, no_data):
    """compose() returns the NoData of a missing input instead of crashing."""
    composer = ImasComposer()
    raw_data = {}
    for _ in range(10):
        status, requirements = composer.resolve([ids_path], 200000, raw_data)
        if all(status.values()):
            break
        raw_data.update({req.as_key(): no_data for req in requirements})
    assert composer.compose([ids_path], 200000, raw_data)[ids_path] is no_data


# (ids_name, shot, composer kwargs, expected NoData coverage: "some" or "all" fields)
OLD_SHOT_CASES = [
    *[pytest.param(ids_name, 118162, {}, "some", id=f"{ids_name}-118162")
      for ids_name in ["equilibrium", "core_profiles", "ece", "charge_exchange", "reflectometer_profile"]],
    pytest.param("nbi", 118162, {}, "all", id="nbi-118162"),
    pytest.param("nbi", 111203, {}, "all", id="nbi-111203"),
    pytest.param("core_profiles", 118162, {"profiles_tree": "OMFIT_PROFS"}, "all", id="core_profiles_omfit-118162"),
]


@pytest.mark.integration
@pytest.mark.requires_mdsplus
@pytest.mark.parametrize("ids_name, shot, composer_kwargs, expected", OLD_SHOT_CASES)
def test_simple_load_old_shot(ids_name, shot, composer_kwargs, expected):
    """simple_load populates fields without data with NoData instead of crashing."""
    composer = ImasComposer(**composer_kwargs)
    fields = composer.get_supported_fields(ids_name)
    results = simple_load(fields, shot, composer=composer)
    assert set(results) == set(fields)
    missing = [path for path, value in results.items() if isinstance(value, NoData)]
    if expected == "all":
        # Static ids_properties do not depend on fetched data
        assert set(missing) == {path for path in fields if ".ids_properties." not in path}
    else:
        assert missing, f"Expected at least one NoData field for {ids_name} #{shot}"


RIP_MEASUREMENT_FIELDS = [
    "interferometer.channel.n_e_line.time",
    "interferometer.channel.n_e_line.data",
    "interferometer.channel.n_e_line.validity_timed",
    "interferometer.channel.n_e_line.data_error_upper",
]


@pytest.mark.integration
@pytest.mark.requires_mdsplus
@pytest.mark.parametrize("shot, has_rip", [
    pytest.param(190000, True, id="co2_and_rip"),
    pytest.param(175000, False, id="rip_nodata"),
    pytest.param(150000, False, id="no_rip_tree"),
])
def test_interferometer_rip_channels_without_data(shot, has_rip):
    """RIP channels are always present with include_rip; channels without data are empty."""
    composer = ImasComposer(include_rip=True)
    results = simple_load(composer.get_supported_fields("interferometer"), shot, composer=composer)
    assert not [path for path, value in results.items() if isinstance(value, NoData)]
    n_co2 = composer._mappers["interferometer"].N_CO2_CHANNELS
    n_channels = len(results["interferometer.channel.identifier"])
    assert n_channels > n_co2
    for path in RIP_MEASUREMENT_FIELDS:
        lengths = ak.num(results[path], axis=1).tolist()
        assert len(lengths) == n_channels, path
        assert all(n > 0 for n in lengths[:n_co2]), path
        assert all((n > 0) == has_rip for n in lengths[n_co2:]), path
