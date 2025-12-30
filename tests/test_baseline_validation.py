"""
Test data fetching consistency across different environments.

This test loads baseline data from a pickle file and compares it against
freshly fetched data using simple_load. The primary purpose is to validate
that data fetching works consistently across different conda environments
and dependency versions.

IMPORTANT: These tests are OPT-IN and require manual setup.
They will only run when explicitly requested with the -m baseline_validation flag.

Usage:
    # Generate baseline data first (in reference environment)
    python scripts/generate_baseline_data.py --output baseline_data.pkl

    # Run validation tests (in new environment) - MUST use -m baseline_validation
    pytest -m baseline_validation -v

    # Test specific IDS
    pytest -m baseline_validation -k equilibrium -v

    # Test with custom baseline file
    pytest -m baseline_validation --baseline-file path/to/baseline.pkl -v

    # These will NOT run baseline validation tests:
    pytest                              # Skips baseline_validation
    pytest tests/test_baseline_validation.py  # Skips baseline_validation
"""

import pytest
import pickle
import numpy as np
import awkward as ak
from pathlib import Path

from imas_composer import simple_load, ImasComposer
from tests.conftest import load_test_config

# Make all tests in this module opt-in only
# Tests will be skipped unless explicitly run with: pytest -m baseline_validation
pytestmark = pytest.mark.baseline_validation


def compare_baseline_values(fresh_data, baseline_data, field_path, rtol=1e-10, atol_float=1e-12, atol_array=1e-6):
    """
    Compare freshly fetched data against baseline data.

    This is simpler than compare_values in conftest because both values come from
    imas_composer (via simple_load), so they have the same types and structure.

    Args:
        fresh_data: Freshly fetched data from simple_load
        baseline_data: Baseline data from pickle (also from simple_load)
        field_path: Full IDS path for error messages
        rtol: Relative tolerance for float comparisons
        atol_float: Absolute tolerance for scalar floats
        atol_array: Absolute tolerance for float arrays

    Raises:
        AssertionError: If data doesn't match
    """
    # Convert everything to awkward arrays for consistent comparison
    # This works for numpy arrays, scalars, and existing awkward arrays
    fresh_ak = ak.Array([fresh_data]) if not isinstance(fresh_data, ak.Array) else fresh_data
    baseline_ak = ak.Array([baseline_data]) if not isinstance(baseline_data, ak.Array) else baseline_data

    # Determine if we should use tolerant comparison by checking the dtype
    try:
        if ak.count(baseline_ak) > 0:
            sample = ak.flatten(baseline_ak, axis=None)[0]
            is_float = isinstance(sample, (float, np.floating))
        else:
            is_float = False
    except:
        is_float = False

    if is_float:
        # Floating point comparison with tolerance
        # Use atol_float for scalars, atol_array for arrays
        atol = atol_float if not isinstance(baseline_data, (np.ndarray, ak.Array)) else atol_array
        close = ak.isclose(fresh_ak, baseline_ak, rtol=rtol, atol=atol)

        if not ak.all(close):
            not_close = ~close
            raise AssertionError(
                f"{field_path}: values differ beyond tolerance\n"
                f"  rtol={rtol}, atol={atol}\n"
                f"  Number of differing elements: {ak.sum(ak.flatten(not_close))}"
            )
    else:
        # Exact comparison for integers/strings
        equal = fresh_ak == baseline_ak

        if not ak.all(equal):
            not_equal = ~equal
            raise AssertionError(
                f"{field_path}: values differ\n"
                f"  Number of differing elements: {ak.sum(ak.flatten(not_equal))}"
            )


@pytest.fixture(scope='module')
def baseline_data(request):
    """
    Load baseline data from pickle file.

    Returns:
        dict: Baseline data with structure {shot: {ids_name: {field_path: data}}}

    Raises:
        FileNotFoundError: If baseline file doesn't exist
    """
    baseline_file = request.config.getoption("--baseline-file")
    baseline_path = Path(baseline_file)

    if not baseline_path.exists():
        pytest.skip(
            f"Baseline file not found: {baseline_path}\n"
            f"Generate it first with: python scripts/generate_baseline_data.py"
        )

    with open(baseline_path, 'rb') as f:
        data = pickle.load(f)

    return data


def generate_test_parameters(baseline_data):
    """
    Generate pytest parameters from baseline data.

    Returns:
        list: List of (shot, ids_key, field_path, baseline_value) tuples
        list: List of test IDs
    """
    params = []
    ids_list = []

    for shot, shot_data in baseline_data['data'].items():
        for ids_key, fields_data in shot_data.items():
            for field_path, baseline_value in fields_data.items():
                # Skip fields that were skipped or failed in baseline generation
                if baseline_value is None:
                    continue
                if isinstance(baseline_value, Exception):
                    continue

                params.append((shot, ids_key, field_path, baseline_value))
                # Create readable test ID
                ids_list.append(f"shot{shot}::{ids_key}::{field_path}")

    return params, ids_list


# Generate test parameters at module level for pytest parametrize
def pytest_generate_tests(metafunc):
    """Dynamically parametrize tests based on baseline data."""
    if 'field_test_case' in metafunc.fixturenames:
        # Load baseline data
        baseline_file = metafunc.config.getoption("--baseline-file")
        baseline_path = Path(baseline_file)

        if not baseline_path.exists():
            # If baseline doesn't exist, create empty parametrization
            metafunc.parametrize('field_test_case', [], ids=[])
            return

        with open(baseline_path, 'rb') as f:
            baseline_data = pickle.load(f)

        # Generate parameters
        params, ids_list = generate_test_parameters(baseline_data)

        # Parametrize the test
        metafunc.parametrize('field_test_case', params, ids=ids_list)


def get_composer_for_ids(ids_key):
    """
    Create appropriate ImasComposer instance for the given IDS key.

    Args:
        ids_key: IDS key (e.g., 'equilibrium', 'core_profiles_ZIPFIT')

    Returns:
        ImasComposer: Configured composer instance
    """
    if ids_key.endswith('_ZIPFIT'):
        return ImasComposer(profiles_tree='ZIPFIT01')
    elif ids_key.endswith('_OMFIT_PROFS'):
        return ImasComposer(profiles_tree='OMFIT_PROFS', profiles_run_id='001')
    else:
        return ImasComposer()


def get_ids_name(ids_key):
    """
    Extract IDS name from IDS key.

    Args:
        ids_key: IDS key (e.g., 'equilibrium', 'core_profiles_ZIPFIT')

    Returns:
        str: IDS name (e.g., 'equilibrium', 'core_profiles')
    """
    if ids_key.startswith('core_profiles'):
        return 'core_profiles'
    return ids_key


def compare_data(fresh_data, baseline_data, field_path, ids_name):
    """
    Compare freshly fetched data against baseline data.

    Args:
        fresh_data: Freshly fetched data
        baseline_data: Baseline data from pickle
        field_path: Full IDS path
        ids_name: IDS name for loading tolerances

    Raises:
        AssertionError: If data doesn't match
    """
    # Load test config to get field-specific tolerances
    test_config = load_test_config(ids_name)
    field_tolerances = test_config.get('field_tolerances', {})

    # Get field-specific tolerances if configured
    field_tol = field_tolerances.get(field_path, {})
    rtol = field_tol.get('rtol', 1e-10)
    atol_float = field_tol.get('atol', 1e-12)
    atol_array = field_tol.get('atol', 1e-6)

    # Use our dedicated baseline comparison function
    compare_baseline_values(
        fresh_data,
        baseline_data,
        field_path,
        rtol=rtol,
        atol_float=atol_float,
        atol_array=atol_array
    )


def test_field_matches_baseline(field_test_case):
    """
    Test that freshly fetched data matches baseline data.

    This test validates that data fetching is consistent across different
    conda environments and dependency versions by comparing against a
    known-good baseline.

    Args:
        field_test_case: Tuple of (shot, ids_key, field_path, baseline_value)
    """
    shot, ids_key, field_path, baseline_value = field_test_case

    # Get appropriate composer for this IDS
    composer = get_composer_for_ids(ids_key)
    ids_name = get_ids_name(ids_key)

    # Fetch fresh data
    try:
        result = simple_load([field_path], shot, composer=composer)
        fresh_data = result[field_path]
    except Exception as e:
        pytest.fail(
            f"Failed to fetch {field_path} for shot {shot}: {e}\n"
            f"This may indicate a dependency or environment issue."
        )

    # Compare with baseline
    try:
        compare_data(fresh_data, baseline_value, field_path, ids_name)
    except AssertionError as e:
        # Provide helpful error message
        raise AssertionError(
            f"Data mismatch for {field_path} (shot {shot}):\n"
            f"{e}\n"
            f"This may indicate:\n"
            f"  - Different MDSplus server data\n"
            f"  - Dependency version changes affecting data fetching\n"
            f"  - Code changes in imas_composer\n"
            f"Consider regenerating baseline if changes are expected."
        ) from e


@pytest.mark.summary
def test_baseline_summary(baseline_data, capsys):
    """
    Print summary of baseline data for informational purposes.

    This test always passes but prints useful statistics about the
    baseline data being validated against.
    """
    print("\n" + "=" * 70)
    print("BASELINE DATA SUMMARY")
    print("=" * 70)
    print(f"Shots: {baseline_data.get('shots', [])}")
    print(f"Metadata: {baseline_data.get('metadata', {})}")
    print()

    # Summary per shot
    for shot in baseline_data.get('shots', []):
        if shot not in baseline_data['data']:
            continue

        print(f"Shot {shot}:")
        total_fields = 0
        successful_fields = 0

        for ids_key, fields_data in baseline_data['data'][shot].items():
            valid_fields = sum(
                1 for v in fields_data.values()
                if v is not None and not isinstance(v, Exception)
            )
            total_fields += len(fields_data)
            successful_fields += valid_fields
            print(f"  {ids_key}: {valid_fields}/{len(fields_data)} fields")

        print(f"  Total: {successful_fields}/{total_fields} fields")
        print()

    # Overall summary
    total_tests = sum(
        1 for shot_data in baseline_data['data'].values()
        for fields_data in shot_data.values()
        for v in fields_data.values()
        if v is not None and not isinstance(v, Exception)
    )
    print(f"Total parametrized tests: {total_tests}")
    print("=" * 70)

    # This test always passes - it's just for informational output
    assert True
