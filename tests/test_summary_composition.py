"""
Test Summary IDS data composition.

Verifies that summary.global_quantities.tau_energy.value and .time
are correctly fetched from the DIII-D TRANSPORT MDSplus tree and
returned as numpy arrays with physically reasonable values.

TRANSPORT.GLOBAL.TIMES.TAUE stores the energy confinement time in seconds
as computed by the transport analysis.  For DIII-D H-mode plasmas a
typical value is 0.05 – 0.40 s.

The composition tests do not compare against OMAS because the summary IDS
is a derived quantity in OMAS (computed from stored energy and power balance)
rather than read directly from MDSplus.  Instead we check:
  - The output is a 1-D numpy array of finite floats
  - The time array is monotonically increasing and in seconds (not ms)
  - tau_energy values are in a physically plausible range for DIII-D
"""
import numpy as np
import pytest

from tests.conftest import REFERENCE_SHOT, load_ids_fields
from imas_composer import ImasComposer


@pytest.mark.parametrize('ids_path', load_ids_fields('summary'))
def test_can_resolve_requirements(ids_path, composer):
    """Requirements resolve without error."""
    from tests.conftest import run_requirements_resolution
    run_requirements_resolution(ids_path, composer, REFERENCE_SHOT)


def test_tau_energy_value_is_array(composer):
    """summary.global_quantities.tau_energy.value returns a numpy array."""
    result = composer.simple_load(
        ['summary.global_quantities.tau_energy.value'], REFERENCE_SHOT
    )
    tau_e = result['summary.global_quantities.tau_energy.value']
    assert isinstance(tau_e, np.ndarray), "tau_energy.value should be a numpy array"
    assert tau_e.ndim == 1, "tau_energy.value should be 1-D (time trace)"


def test_tau_energy_value_finite(composer):
    """tau_energy.value contains only finite values."""
    result = composer.simple_load(
        ['summary.global_quantities.tau_energy.value'], REFERENCE_SHOT
    )
    tau_e = result['summary.global_quantities.tau_energy.value']
    assert np.all(np.isfinite(tau_e)), "tau_energy.value should contain only finite values"


def test_tau_energy_value_positive(composer):
    """tau_energy.value is strictly positive (confinement time cannot be negative)."""
    result = composer.simple_load(
        ['summary.global_quantities.tau_energy.value'], REFERENCE_SHOT
    )
    tau_e = result['summary.global_quantities.tau_energy.value']
    assert np.all(tau_e > 0), "tau_energy.value should be strictly positive"


def test_tau_energy_value_plausible_range(composer):
    """tau_energy mean is in the DIII-D range 0.01 – 1.0 s."""
    result = composer.simple_load(
        ['summary.global_quantities.tau_energy.value'], REFERENCE_SHOT
    )
    tau_e = result['summary.global_quantities.tau_energy.value']
    mean_tau = float(np.mean(tau_e))
    assert 0.01 < mean_tau < 1.0, (
        f"Mean tau_energy = {mean_tau:.4f} s is outside plausible DIII-D range [0.01, 1.0] s"
    )


def test_tau_energy_time_is_seconds(composer):
    """tau_energy.time values are in seconds (not milliseconds)."""
    result = composer.simple_load(
        ['summary.global_quantities.tau_energy.time'], REFERENCE_SHOT
    )
    t = result['summary.global_quantities.tau_energy.time']
    assert isinstance(t, np.ndarray)
    # Shot duration on DIII-D is < 10 s; if values > 1000 the unit is ms
    assert np.max(t) < 100.0, (
        f"tau_energy.time max = {np.max(t):.1f} — expected seconds, got milliseconds?"
    )


def test_tau_energy_time_monotonic(composer):
    """tau_energy.time is monotonically increasing."""
    result = composer.simple_load(
        ['summary.global_quantities.tau_energy.time'], REFERENCE_SHOT
    )
    t = result['summary.global_quantities.tau_energy.time']
    assert np.all(np.diff(t) > 0), "tau_energy.time should be monotonically increasing"


def test_tau_energy_value_and_time_same_length(composer):
    """tau_energy.value and .time have the same number of points."""
    result = composer.simple_load(
        [
            'summary.global_quantities.tau_energy.value',
            'summary.global_quantities.tau_energy.time',
        ],
        REFERENCE_SHOT,
    )
    tau_e = result['summary.global_quantities.tau_energy.value']
    t     = result['summary.global_quantities.tau_energy.time']
    assert len(tau_e) == len(t), (
        f"tau_energy value length {len(tau_e)} != time length {len(t)}"
    )
