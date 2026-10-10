"""
Test the ZIPFIT core_profiles poloidal flux grid axes.

ZIPFIT stores no poloidal flux coordinate, so grid.psi_norm, grid.rho_pol_norm and
grid.psi are reconstructed from the EFIT01 equilibrium. OMAS's ZIPFIT branch maps
none of them (d3d.py::core_profiles_profile_1d writes only grid.rho_tor_norm), so
the usual OMAS differential test does not apply. These axes are validated instead
against the equilibrium IDS, whose psi profile is itself OMAS-validated.

The module is deliberately not named test_core_profiles_* : conftest's
pytest_generate_tests fans the shared `composer` fixture across both profile trees
for modules whose name contains 'core_profiles', and this suite is ZIPFIT-only.
"""
import awkward as ak
import numpy as np
import pytest
from scipy.interpolate import PchipInterpolator

from imas_composer.composer import ImasComposer
from tests.conftest import resolve_and_compose

pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus]

RHO_TOR_NORM = 'core_profiles.profiles_1d.grid.rho_tor_norm'
RHO_POL_NORM = 'core_profiles.profiles_1d.grid.rho_pol_norm'
PSI_NORM = 'core_profiles.profiles_1d.grid.psi_norm'
PSI = 'core_profiles.profiles_1d.grid.psi'
CORE_PROFILES_TIME = 'core_profiles.time'

EQ_RHO_TOR_NORM = 'equilibrium.time_slice.profiles_1d.rho_tor_norm'
EQ_PSI = 'equilibrium.time_slice.profiles_1d.psi'
EQ_TIME = 'equilibrium.time'


@pytest.fixture(scope='module')
def zipfit_composer():
    """Composer on the ZIPFIT01 / EFIT01 defaults, serving both IDS under test."""
    return ImasComposer(profiles_tree='ZIPFIT01', efit_tree='EFIT01')


@pytest.fixture(scope='module')
def composed_cache():
    """Per-shot cache so each shot is fetched from MDSplus once for the module."""
    return {}


def _composed(composed_cache, composer, shot, paths):
    """Compose ``paths`` for ``shot``, reusing anything already fetched."""
    slot = composed_cache.setdefault(shot, {})
    for path in paths:
        if path not in slot:
            slot[path] = resolve_and_compose(composer, path, shot)
    return slot


@pytest.fixture
def grid(composed_cache, zipfit_composer, test_shot):
    """The four ZIPFIT grid axes plus the core_profiles time base, as numpy arrays."""
    composed = _composed(
        composed_cache, zipfit_composer, test_shot,
        [CORE_PROFILES_TIME, RHO_TOR_NORM, RHO_POL_NORM, PSI_NORM, PSI],
    )
    return {
        path: value if path == CORE_PROFILES_TIME else ak.to_numpy(value)
        for path, value in composed.items()
        if path in (CORE_PROFILES_TIME, RHO_TOR_NORM, RHO_POL_NORM, PSI_NORM, PSI)
    }


@pytest.fixture
def equilibrium_profiles(composed_cache, zipfit_composer, test_shot):
    """Equilibrium time, rho_tor_norm (RHOVN) and psi, on the EFIT01 GTIME base."""
    return _composed(
        composed_cache, zipfit_composer, test_shot,
        [EQ_TIME, EQ_RHO_TOR_NORM, EQ_PSI],
    )


def test_grid_axes_share_the_rho_axis(grid):
    """Every grid.* axis is defined on the same rho grid, for every time slice."""
    n_time = len(grid[CORE_PROFILES_TIME])
    reference = grid[RHO_TOR_NORM]
    assert reference.shape[0] == n_time

    for path in (PSI_NORM, RHO_POL_NORM, PSI):
        assert grid[path].shape == reference.shape, (
            f"{path} shape {grid[path].shape} does not match "
            f"grid.rho_tor_norm shape {reference.shape}"
        )


def test_rho_pol_norm_is_sqrt_of_psi_norm(grid):
    """rho_pol_norm is defined as sqrt(psi_norm), matching the OMFIT_PROFS mapper."""
    np.testing.assert_allclose(
        grid[RHO_POL_NORM], np.sqrt(grid[PSI_NORM]), rtol=0.0, atol=0.0
    )


def test_psi_norm_is_monotonic_and_normalized(grid):
    """psi_norm increases from the axis to the separatrix and stays normalized."""
    psi_norm = grid[PSI_NORM]

    assert np.all(np.diff(psi_norm, axis=1) > 0), (
        "psi_norm is not strictly increasing with rho_tor_norm"
    )
    assert np.all(psi_norm >= -1e-6), f"psi_norm below 0: min {psi_norm.min()}"
    assert np.all(psi_norm <= 1.0 + 1e-6), f"psi_norm above 1: max {psi_norm.max()}"


def test_psi_matches_the_equilibrium_ids(grid, equilibrium_profiles):
    """
    grid.psi reproduces the equilibrium psi profile on the ZIPFIT rho axis.

    This is the substantive check: it exercises the EFIT tree choice, the
    SSIMAG/SSIBRY denormalization and the COCOS sign end-to-end against a mapping
    that is itself validated against OMAS. Both IDS take their time base from
    EFIT01 GTIME, so the slices are compared index-for-index.

    The same shape-preserving interpolator as the mapper is used here: PCHIP
    commutes with the affine denormalization (its slope estimates are homogeneous
    in the sample slopes), so agreement is expected to near machine precision and
    any disagreement points at the tree, the denormalization or the COCOS factor.
    """
    # Same GTIME node, but equilibrium converts ms->s on the float32 array while
    # core_profiles divides inside the TDI expression, so the two differ by float32
    # rounding. Only the slice correspondence matters here.
    np.testing.assert_allclose(
        equilibrium_profiles[EQ_TIME], grid[CORE_PROFILES_TIME], rtol=1e-6, atol=0.0,
        err_msg="equilibrium and core_profiles are not on the same EFIT01 GTIME base",
    )

    eq_rho_tor = equilibrium_profiles[EQ_RHO_TOR_NORM]
    eq_psi = equilibrium_profiles[EQ_PSI]
    rho = grid[RHO_TOR_NORM]

    for i_time in range(len(grid[CORE_PROFILES_TIME])):
        expected = PchipInterpolator(eq_rho_tor[i_time], eq_psi[i_time])(rho[i_time])
        np.testing.assert_allclose(
            grid[PSI][i_time], expected, rtol=1e-8, atol=1e-10,
            err_msg=f"grid.psi disagrees with the equilibrium psi profile at slice {i_time}",
        )
