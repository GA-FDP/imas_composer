"""
Test equilibrium profiles_2d.j_tor, recovered from PSIRZ with EFIT's finite difference Grad-Shafranov operator.

OMAS has no derivation of a 2D current density, so the composed field is checked against quantities it is not
computed from: the plasma current and the 1D profiles p' and FF' of the same EFIT run.

Tolerances are measured on the 65x65 EFIT01 grids of the test shots and only apply to time slices that EFIT
converged (convergence error < CONVERGED_ERROR): EFIT evaluates its current on the previous iterate, so on the
other slices the flux and the profiles disagree by EFIT's own convergence error.
"""
import numpy as np
import pytest
from scipy.constants import mu_0

from imas_composer.fetchers import fetch_requirements

pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus]

EQ = "equilibrium.time_slice."
J_TOR = EQ + "profiles_2d.j_tor"
PATHS = [
    J_TOR,
    EQ + "profiles_2d.psi",
    EQ + "profiles_2d.grid.dim1",
    EQ + "profiles_2d.grid.dim2",
    EQ + "global_quantities.ip",
    EQ + "global_quantities.psi_axis",
    EQ + "global_quantities.psi_boundary",
    EQ + "profiles_1d.psi",
    EQ + "profiles_1d.dpressure_dpsi",
    EQ + "profiles_1d.f_df_dpsi",
    EQ + "convergence.grad_shafranov_deviation_value",
]

# EFIT convergence error (max |dpsi| / |psi_bnd - psi_axis| of the last iteration) of the slices that are checked
CONVERGED_ERROR = 1e-4
# Ip of the current within float32 PSIRZ and the truncation of the coil flux by the stencil, 3.1e-4 measured
IP_TOLERANCE = 1e-3
# relative rms of j_tor - (R p' + FF'/(mu0 R)) inside psi_n < CORE_PSIN, 0.9-3e-3 median and 8.1e-3 max
# measured. Mostly the O(h^2) truncation of the stencil, about 4x smaller on 129x129 grids. Further out the 1D
# profiles, linearly interpolated from nw points, cannot resolve the pedestal EFIT evaluates on the grid nodes.
CORE_TOLERANCE = 1e-2
CORE_PSIN = 0.9


@pytest.fixture(scope="module")
def composed_cache() -> dict:
    """Composed fields per shot, so that the tests of a shot fetch from MDSplus once."""
    return {}


@pytest.fixture
def composed(composer, test_shot, composed_cache) -> dict:
    """`PATHS` of `test_shot` as numpy arrays, restricted to the converged time slices."""
    if test_shot not in composed_cache:
        raw_data = {}
        for _ in range(10):
            status, requirements = composer.resolve(PATHS, test_shot, raw_data)
            if all(status.values()):
                break
            for key, value in fetch_requirements(requirements).items():
                if isinstance(value, Exception):
                    pytest.skip(f"MDSplus data unavailable for {key}: {value}")
                raw_data[key] = value
        data = {path: np.asarray(value) for path, value in composer.compose(PATHS, test_shot, raw_data).items()}
        converged = data[EQ + "convergence.grad_shafranov_deviation_value"] < CONVERGED_ERROR
        composed_cache[test_shot] = {
            path: value[converged] if path not in (EQ + "profiles_2d.grid.dim1", EQ + "profiles_2d.grid.dim2")
            else value for path, value in data.items()
        }
    return composed_cache[test_shot]


def cell_area(composed: dict) -> float:
    R = composed[EQ + "profiles_2d.grid.dim1"][0, 0]
    Z = composed[EQ + "profiles_2d.grid.dim2"][0, 0]
    return (R[1] - R[0]) * (Z[1] - Z[0])


def normalized_flux(composed: dict) -> np.ndarray:
    psi_axis = composed[EQ + "global_quantities.psi_axis"][:, None, None]
    psi_boundary = composed[EQ + "global_quantities.psi_boundary"][:, None, None]
    return (composed[EQ + "profiles_2d.psi"][:, 0] - psi_axis) / (psi_boundary - psi_axis)


def test_j_tor_carries_ip(composed):
    """The current integrates to EFIT's Ip, sign included."""
    ip_j_tor = composed[J_TOR][:, 0].sum(axis=(1, 2)) * cell_area(composed)
    np.testing.assert_allclose(ip_j_tor, composed[EQ + "global_quantities.ip"], rtol=IP_TOLERANCE)


def test_j_tor_vanishes_outside_the_plasma(composed):
    """No current outside psi_n = 1, EFIT's hard cut at the separatrix."""
    j_tor = composed[J_TOR][:, 0]
    assert np.all(j_tor[normalized_flux(composed) > 1] == 0)


def test_j_tor_matches_1d_profiles(composed):
    """In the core, j_tor = -2 pi (R p' + FF' / (mu0 R)) (COCOS 11) of the 1D profiles evaluated on the flux."""
    j_tor = composed[J_TOR][:, 0]
    psi_2d = composed[EQ + "profiles_2d.psi"][:, 0]
    R = composed[EQ + "profiles_2d.grid.dim1"][0, 0][:, None]
    core = (j_tor != 0) & (normalized_flux(composed) < CORE_PSIN)

    deviations = []
    for i_time in range(len(j_tor)):
        psi_1d = composed[EQ + "profiles_1d.psi"][i_time]
        order = np.argsort(psi_1d)
        pprime = np.interp(psi_2d[i_time], psi_1d[order], composed[EQ + "profiles_1d.dpressure_dpsi"][i_time][order])
        ffprime = np.interp(psi_2d[i_time], psi_1d[order], composed[EQ + "profiles_1d.f_df_dpsi"][i_time][order])
        j_tor_profiles = -2 * np.pi * (R * pprime + ffprime / (mu_0 * R))
        mask = core[i_time]
        deviations.append(np.linalg.norm((j_tor[i_time] - j_tor_profiles)[mask]) / np.linalg.norm(j_tor_profiles[mask]))

    assert max(deviations) < CORE_TOLERANCE, f"worst slice deviates by {max(deviations):.1e}"
