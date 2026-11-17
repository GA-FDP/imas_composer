"""
Test equilibrium.code.parameters, the XML holding EFIT's p' and FF' basis function coefficients (XRSP), the bases of
p' and FF' and EFIT's rigid vertical shift settings (fitdelz, ifitdelz) from the k-files or the snap file of the run.

OMAS has no mapping for it. The coefficients are checked against the p' profile: every basis function EFIT
offers for p' (polynomial or spline, ppbasisfunc.f90) vanishes on the magnetic axis except the first one,
which is 1 there, so p' at psi_n = 0 is the first coefficient. The g-file PPRIME is that sum (shapesurf.F90),
dpressure_dpsi is PPRIME / (2 pi) in COCOS 11 up to the sign. The bases are checked against the stored profiles
on the cases IDA verifies its free-boundary solver with. Tests of the bases without data are in test_efit_basis.py.
"""
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from imas_composer import ImasComposer, NoData, Requirement
from imas_composer.efit_basis import RECOVERY_TOLERANCE, efit_f_profile, efit_profile_basis
from imas_composer.fetchers import fetch_requirements
from imas_composer.ids.equilibrium import FITDELZ_KEYS, EquilibriumMapper
from tests.conftest import efit_fpol

CODE_PARAMETERS = "equilibrium.code.parameters"
TIME = "equilibrium.time"
PPRIME = "equilibrium.time_slice.profiles_1d.dpressure_dpsi"
FFPRIME = "equilibrium.time_slice.profiles_1d.f_df_dpsi"
F = "equilibrium.time_slice.profiles_1d.f"
PSI_AXIS = "equilibrium.time_slice.global_quantities.psi_axis"
PSI_BOUNDARY = "equilibrium.time_slice.global_quantities.psi_boundary"
# IDA's basis cases (tests/models/test_free_boundary_gs_solver_efit.py): shot, tree, run id, time [s], where the
# settings come from and whether EFIT moved the knots (kautoknt) away from those of the k-file
BASIS_CASES = [
    pytest.param(203877, "EFIT", "01", 2.0, "kfile", False, id="203877_cake_01"),
    pytest.param(169621, "EFIT", "04", 2.0, "kfile", True, id="169621_cake_04_moved_knots"),
    pytest.param(169621, "EFIT", "04", 1.44, "kfile", False, id="169621_cake_04_pedestal"),
    pytest.param(173587, "EFIT", "01", 2.28, "kfile", True, id="173587_cake_01_moved_knots"),
    pytest.param(173587, "EFIT", "01", 5.4, "kfile", False, id="173587_cake_01_knots_at_one_half"),
    pytest.param(155149, "EFIT", "05", 3.0, "kfile", False, id="155149_cake_05"),
    pytest.param(203877, "EFIT01", "", 2.0, "snap", False, id="203877_efit01"),
    pytest.param(203877, "EFIT02", "", 2.0, "snap", False, id="203877_efit02"),
    pytest.param(118162, "EFIT01", "", 1.8, "snap", False, id="118162_efit01"),
]
# a basis case slice lies within this of its time [s]
CASE_TIME_TOLERANCE = 1e-3
# F of the basis differs from EFIT's FPOL by the error of EFIT's trapezoidal rule on nw points (ece.F90)
F_TRAPEZOID_TOLERANCE = 1e-4
# float32 storage of both XRSP and PPRIME
AXIS_TOLERANCE = 1e-6
# 237 GEQDSK and 282 MEASUREMENTS slices, see test_late_slices_map_to_their_own_measurements
UNEQUAL_TIME_BASE_SHOT = 202161
# EFIT run that stores its 229 k-files (NAMELISTS:KEQDSKS), all with FITDELZ = .true. and IFITDELZ = 1
KEQDSKS_SHOT = 155151
KEQDSKS_RUN_ID = "04"
KEQDSKS = Requirement("\\EFIT::TOP.NAMELISTS:KEQDSKS", int(f"{KEQDSKS_SHOT}{KEQDSKS_RUN_ID}"), "EFIT")
KEQDSKS_SHAPE = (229, 630)
# Standard EFIT01 runs store no k-files, only their snap file (TOP:NAMELIST). That of 203877 sets FITDELZ = .T. and
# IFITDELZ = 3, that of 155151 FITDELZ = .T. and leaves IFITDELZ at EFIT's default.
SNAP_SETTINGS = [
    pytest.param(203877, {"fitdelz": "true", "ifitdelz": "3"}, id="203877"),
    pytest.param(KEQDSKS_SHOT, {"fitdelz": "true"}, id="155151_default_ifitdelz"),
]
# The stored snap files hold a title line, the &EFITIN group and a description of its parameters
SNAP_TITLE = "EFIT_SNAP.DAT_JTA_F  (rename to EFIT_SNAP.DAT for EFITD)\n"
SNAP_DESCRIPTION = "\n  NAMELIST input parameters               03/07/91\n  kffcur : number of fitting parameters in FF'\n"


def compose(composer, shot: int, paths: tuple = (CODE_PARAMETERS, TIME, PPRIME)) -> tuple[dict, dict]:
    """`paths` of `shot`, by default `CODE_PARAMETERS`, `TIME` and `PPRIME`, and the raw data they were composed
    from."""
    paths = list(paths)
    raw_data = {}
    for _ in range(10):
        status, requirements = composer.resolve(paths, shot, raw_data)
        if all(status.values()):
            break
        for key, value in fetch_requirements(requirements).items():
            if isinstance(value, Exception) and not isinstance(value, NoData):
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


def parse_settings(xml: str, key: str) -> dict[int, str]:
    """The <key> setting of each time slice that has one, parsed from the XML."""
    return {int(element.get("index")): element.text for element in ET.fromstring(xml).find(key).findall("time_slice")}


@pytest.fixture
def composed(composer, test_shot) -> tuple[dict, dict]:
    return compose(composer, test_shot)


@pytest.mark.integration
@pytest.mark.requires_mdsplus
def test_one_entry_per_equilibrium_time_slice(composed):
    data, _ = composed
    coefficients = parse_coefficients(data[CODE_PARAMETERS])
    assert len(coefficients) == len(data[TIME])
    for slice_coefficients in coefficients:
        assert len(slice_coefficients) > 0
        assert np.all(np.isfinite(slice_coefficients)) and np.all(np.abs(slice_coefficients) < 1e30)


@pytest.mark.integration
@pytest.mark.requires_mdsplus
def test_first_coefficient_is_pprime_on_axis(composed):
    """|xrsp_1| = 2 pi |dpressure_dpsi(psi_n = 0)| on every slice."""
    data, _ = composed
    mismatch = axis_mismatch(data)
    assert np.all(mismatch < AXIS_TOLERANCE), f"worst slice deviates by {mismatch.max():.1e}"


@pytest.mark.integration
@pytest.mark.requires_mdsplus
def test_late_slices_map_to_their_own_measurements():
    data, _ = compose(ImasComposer(), UNEQUAL_TIME_BASE_SHOT)
    mismatch = axis_mismatch(data)
    assert np.all(mismatch < AXIS_TOLERANCE), f"{np.sum(mismatch >= AXIS_TOLERANCE)} slices read another slice's XRSP"


def relative_rms(values: np.ndarray, stored: np.ndarray) -> float:
    """rms of `values` against `stored`, relative to max |stored|, as efit_basis checks the bases."""
    return float(np.sqrt(np.mean(((values - stored) / np.abs(stored).max()) ** 2)))


@pytest.mark.integration
@pytest.mark.requires_mdsplus
def test_one_basis_per_equilibrium_time_slice(composed):
    data, _ = composed
    slices = ET.fromstring(data[CODE_PARAMETERS]).find("basis").findall("time_slice")
    assert [int(element.get("index")) for element in slices] == list(range(len(data[TIME])))
    for element in slices:
        assert [child.tag for child in element] == ["pprime", "ffprime"]


@pytest.mark.integration
@pytest.mark.requires_mdsplus
@pytest.mark.parametrize("shot, efit_tree, efit_run_id, time, source, fitted", BASIS_CASES)
def test_basis_reproduces_the_stored_profiles(shot, efit_tree, efit_run_id, time, source, fitted):
    """The basis of the slice reproduces the stored p' and FF', from the expected namelist, with the knots fitted
    only where EFIT moved them."""
    composer = ImasComposer(efit_tree=efit_tree, efit_run_id=efit_run_id)
    data, _ = compose(composer, shot, (CODE_PARAMETERS, TIME, PPRIME, FFPRIME))
    times = np.asarray(data[TIME])
    index = int(np.argmin(np.abs(times - time)))
    assert abs(times[index] - time) < CASE_TIME_TOLERANCE, f"no slice at {time} s, nearest {times[index]} s"

    element = ET.fromstring(data[CODE_PARAMETERS]).find("basis").findall("time_slice")[index]
    assert element.get("source") == source
    assert any(child.get("fitted_knots") == "true" for child in element) == fitted

    stored_pprime, stored_ffprime = np.asarray(data[PPRIME])[index], np.asarray(data[FFPRIME])[index]
    pprime, ffprime = efit_profile_basis(data[CODE_PARAMETERS], index)(np.linspace(0, 1, stored_pprime.size))
    assert relative_rms(pprime, stored_pprime) < RECOVERY_TOLERANCE
    assert relative_rms(ffprime, stored_ffprime) < RECOVERY_TOLERANCE


@pytest.mark.integration
@pytest.mark.requires_mdsplus
def test_f_follows_the_stored_f(composer, test_shot):
    """EFIT's trapezoidal rule (ece.F90) on the FF' of the basis gives the stored F, which checks the inputs of
    efit_f_profile and their COCOS. Its closed-form integral differs from it by the error of that rule."""
    data, _ = compose(composer, test_shot, (CODE_PARAMETERS, TIME, F, PSI_AXIS, PSI_BOUNDARY))
    f_stored = np.asarray(data[F], dtype=np.float64)
    psi_axis, psi_boundary = np.asarray(data[PSI_AXIS]), np.asarray(data[PSI_BOUNDARY])
    psin = np.linspace(0, 1, f_stored.shape[1])
    for index in (0, len(f_stored) // 2, len(f_stored) - 1):
        _, ffprime = efit_profile_basis(data[CODE_PARAMETERS], index)(psin)
        trapezoid = efit_fpol(ffprime, f_stored[index, -1], psi_boundary[index] - psi_axis[index])
        np.testing.assert_allclose(trapezoid, f_stored[index], rtol=AXIS_TOLERANCE)
        f_profile = efit_f_profile(
            data[CODE_PARAMETERS], index, f_stored[index, -1], psi_axis[index], psi_boundary[index]
        )
        np.testing.assert_allclose(f_profile(psin), f_stored[index], rtol=F_TRAPEZOID_TOLERANCE)


@pytest.mark.requires_atlas  # the EFIT run KEQDSKS_RUN_ID is not on FDP
@pytest.mark.integration
@pytest.mark.requires_mdsplus
@pytest.mark.parametrize("isolated, shape", [(True, KEQDSKS_SHAPE), (False, KEQDSKS_SHAPE[1:])])
def test_getmany_truncates_keqdsks(isolated, shape):
    """getMany returns only the first k-file of the 2D string array, an isolated fetch all of them."""
    req = Requirement(KEQDSKS.mds_path, KEQDSKS.shot, KEQDSKS.treename, isolated=isolated)
    assert np.asarray(fetch_requirements([req])[req.as_key()]).shape == shape


@pytest.mark.requires_atlas  # the EFIT run KEQDSKS_RUN_ID is not on FDP
@pytest.mark.integration
@pytest.mark.requires_mdsplus
@pytest.mark.parametrize("key, value", [("fitdelz", "true"), ("ifitdelz", "1")])
def test_fitdelz_per_time_slice(key, value):
    composer = ImasComposer(efit_tree="EFIT", efit_run_id=KEQDSKS_RUN_ID)
    data, _ = compose(composer, KEQDSKS_SHOT)
    assert parse_settings(data[CODE_PARAMETERS], key) == {index: value for index in range(len(data[TIME]))}


@pytest.mark.integration
@pytest.mark.requires_mdsplus
@pytest.mark.parametrize("shot, expected", SNAP_SETTINGS)
def test_fitdelz_from_snap_file(shot, expected):
    """Standard EFIT trees store no k-files, the settings of their snap file hold for every slice. A setting the snap
    file leaves out is left out of the XML too."""
    data, _ = compose(ImasComposer(), shot)
    all_slices = range(len(data[TIME]))
    for key in ("fitdelz", "ifitdelz"):
        settings = parse_settings(data[CODE_PARAMETERS], key)
        assert settings == ({index: expected[key] for index in all_slices} if key in expected else {}), key
    assert len(parse_coefficients(data[CODE_PARAMETERS])) == len(data[TIME])


@pytest.mark.parametrize("namelist, group, expected", [
    pytest.param("&IN1\n ISHOT = 155151\n/", "inwant", {}, id="no_inwant"),
    pytest.param("&INWANT\n NITERA = 8\n/", "inwant", {}, id="neither"),
    pytest.param("&INWANT\n FITDELZ = .true.\n/", "inwant", {"fitdelz": True}, id="fitdelz"),
    pytest.param("&INWANT\n IFITDELZ = 3\n/", "inwant", {"ifitdelz": 3}, id="ifitdelz"),
    pytest.param(
        "&INWANT\n FITDELZ = .false.\n IFITDELZ = 3\n/", "inwant", {"fitdelz": False, "ifitdelz": 3}, id="both"
    ),
    pytest.param("&INWANT\n FITDELZ = .true.\n/", "efitin", {}, id="other_group"),
    pytest.param(
        SNAP_TITLE + " &efitin\n  kffcur=3  kppcur=2\n  fwtcur=  1.     fitdelz = .T. kersil=3 ifitdelz=3\n /"
        + SNAP_DESCRIPTION,
        "efitin",
        {"fitdelz": True, "ifitdelz": 3},
        id="snap",
    ),
    pytest.param(
        SNAP_TITLE + " &efitin\n  kffcur=2  kppcur=2\n  fitdelz=T\n  relax=0.5\n /" + SNAP_DESCRIPTION,
        "efitin",
        {"fitdelz": True},
        id="snap_without_ifitdelz",
    ),
])
def test_parse_fitdelz(namelist, group, expected):
    """Only the settings the namelist sets in its group, no EFIT defaults."""
    assert EquilibriumMapper._parse_namelist(namelist, group, FITDELZ_KEYS) == expected
