"""
Test the GTIME -> ATIME index mapping used by all equilibrium fields from RESULTS.AEQDSK.

EFIT writes no a-file for some slices that have a g-file, so ATIME can be shorter than GTIME
(e.g. 118162 EFIT01: 254 GTIME, 239 ATIME). Every AEQDSK field must still have one entry per GTIME,
with the slices that have no a-file missing.
"""
import awkward as ak
import numpy as np
import pytest

from imas_composer import simple_load
from imas_composer.core import Requirement
from imas_composer.fetchers import fetch_requirements
from imas_composer.ids.equilibrium import EquilibriumMapper

SHOT = 200000
SHOT_WITH_MISSING_AFILES = 118162

AEQDSK_NODES = [
    "RSURF", "SEPLIM", "TRITOP", "AMINOR", "LI3", "AREA", "Q95", "ERROR", "GAPIN", "GAPOUT", "GAPTOP", "GAPBOT",
    "RXPT1", "RXPT2", "ZXPT1", "ZXPT2", "RVSID", "RVSOD", "RVSIU", "RVSOU", "ZVSID", "ZVSOD", "ZVSIU", "ZVSOU",
]

# Missing slices: None for one value per slice, a row of None for the 4 gaps, an empty list for ragged points
FIELDS_MISSING_AS = {
    "equilibrium.time_slice.boundary_separatrix.geometric_axis.r": None,
    "equilibrium.time_slice.boundary_separatrix.closest_wall_point.distance": None,
    "equilibrium.time_slice.boundary_separatrix.triangularity_upper": None,
    "equilibrium.time_slice.boundary_separatrix.minor_radius": None,
    "equilibrium.time_slice.global_quantities.li_3": None,
    "equilibrium.time_slice.global_quantities.area": None,
    "equilibrium.time_slice.global_quantities.q_95": None,
    "equilibrium.time_slice.convergence.grad_shafranov_deviation_value": None,
    "equilibrium.time_slice.boundary_separatrix.gap.value": [None] * 4,
    "equilibrium.time_slice.boundary.x_point.r": [],
    "equilibrium.time_slice.boundary.x_point.z": [],
    "equilibrium.time_slice.boundary_separatrix.strike_point.r": [],
    "equilibrium.time_slice.boundary_separatrix.strike_point.z": [],
}

TIME_CASES = pytest.mark.parametrize(
    "gtime, atime, expected",
    [
        ([100, 120, 140], [100, 120, 140], [0, 1, 2]),
        ([100, 120, 140], [120, 140], [-1, 0, 1]),
        ([100, 120, 140, 160], [100, 160], [0, -1, -1, 1]),
        ([100, 120, 140], [100, 120], [0, 1, -1]),
        ([100, 120], [100, 120.5], [0, -1]),
    ],
    ids=["equal_lengths", "missing_at_start", "missing_in_middle", "missing_at_end", "outside_tolerance"],
)


@pytest.fixture
def mapper():
    return EquilibriumMapper()


def make_raw_data(mapper, gtime, atime):
    """GTIME/ATIME plus every AEQDSK node of the tested fields, with distinct positive values per a-file."""
    def key(node, name):
        return Requirement(f"{node}.{name}", SHOT, mapper.efit_tree).as_key()

    n_atime = len(atime)
    raw_data = {
        key(mapper.geqdsk_node, "GTIME"): np.asarray(gtime, dtype=float),
        key(mapper.aeqdsk_node, "ATIME"): np.asarray(atime, dtype=float),
        key(mapper.geqdsk_node, "BCENTR"): np.full(len(gtime), -2.0),
        key(mapper.geqdsk_node, "CPASMA"): np.full(len(gtime), 1e6),
    }
    for offset, name in enumerate(AEQDSK_NODES):
        raw_data[key(mapper.aeqdsk_node, name)] = 1.0 + np.arange(n_atime) + 10 * offset
    return raw_data


@TIME_CASES
def test_aeqdsk_time_indices_mapping(mapper, gtime, atime, expected):
    indices = mapper._compose_aeqdsk_time_indices(SHOT, make_raw_data(mapper, gtime, atime))
    np.testing.assert_array_equal(indices, expected)


@TIME_CASES
@pytest.mark.parametrize("ids_path", FIELDS_MISSING_AS)
def test_aeqdsk_field_on_gtime(mapper, gtime, atime, expected, ids_path):
    """Each GTIME gets the value of its own a-file, as composed when every slice has one."""
    compose = mapper.specs[ids_path].compose
    per_afile = ak.to_list(compose(SHOT, make_raw_data(mapper, atime, atime)))

    result = compose(SHOT, make_raw_data(mapper, gtime, atime))

    assert ak.to_list(result) == [FIELDS_MISSING_AS[ids_path] if index < 0 else per_afile[index] for index in expected]


def test_aeqdsk_matches_geqdsk_plasma_current():
    """
    AEQDSK stores the plasma current (IPMHD) that GEQDSK also stores (CPASMA). On a shot with missing
    a-files, every slice with an a-file must read the IPMHD of its own time.
    """
    mapper = EquilibriumMapper(efit_tree="EFIT01", efit_run_id="")
    shot = SHOT_WITH_MISSING_AFILES
    requirements = [
        Requirement(f"{mapper.geqdsk_node}.GTIME", shot, mapper.efit_tree),
        Requirement(f"{mapper.aeqdsk_node}.ATIME", shot, mapper.efit_tree),
        Requirement(f"{mapper.aeqdsk_node}.IPMHD", shot, mapper.efit_tree),
        Requirement(f"{mapper.geqdsk_node}.CPASMA", shot, mapper.efit_tree),
    ]
    raw_data = fetch_requirements(requirements)
    ipmhd, cpasma = (raw_data[req.as_key()] for req in requirements[2:])

    ipmhd_on_gtime = mapper._aeqdsk_on_gtime(shot, raw_data, ipmhd)

    present = ~ak.to_numpy(ak.is_none(ipmhd_on_gtime))
    assert len(cpasma) - present.sum() == 15
    np.testing.assert_allclose(ak.to_numpy(ipmhd_on_gtime[present]), cpasma[present], rtol=1e-6)


def test_aeqdsk_fields_on_equilibrium_time():
    mapper = EquilibriumMapper()
    paths = ["equilibrium.time"] + [
        ids_path for ids_path, spec in mapper.specs.items()
        if "equilibrium._aeqdsk_time_indices" in (spec.depends_on or [])
    ]

    result = simple_load(paths, SHOT_WITH_MISSING_AFILES, efit_tree="EFIT01", efit_run_id="")

    n_time = len(result["equilibrium.time"])
    assert {ids_path: len(result[ids_path]) for ids_path in paths} == dict.fromkeys(paths, n_time)
