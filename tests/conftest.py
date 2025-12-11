"""
Shared pytest fixtures and utilities for imas_composer tests.

This file is automatically discovered by pytest and makes fixtures available
to all test files in this directory.
"""

import pytest
import numpy as np
import yaml
import awkward as ak
from pathlib import Path

from omas import ODS, mdsvalue
from omas.omas_machine import machine_to_omas

from imas_composer import ImasComposer
from imas_composer.core import Requirement


# Reference shot used across most tests
REFERENCE_SHOT = 200000

# Shots to test across (parametrized in test_shot fixture)
#          Bt | Ip
# 202161:  -  | -
# 203321:  +  | -
# 204602:  -  | +
# 204601:  +  | +

TEST_SHOTS = [202161, 203321, 204602, 204601]


# ============================================================================
# YAML Configuration Utilities
# ============================================================================

def load_test_config(ids_name):
    """
    Load test configuration for an IDS.

    Args:
        ids_name: IDS identifier (e.g., 'ece', 'thomson_scattering')

    Returns:
        dict: Test configuration with validation rules, exceptions, and OMAS path mapping
    """
    config_path = Path(__file__).parent / f'test_config_{ids_name}.yaml'

    if not config_path.exists():
        # Return default config if no custom config exists
        return {
            'field_exceptions': {},
            'requirement_validation': {
                'allow_different_shot': []
            },
            'omas_path_map': {}
        }

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


@pytest.fixture(params=TEST_SHOTS)
def test_shot(request):
    """
    Fixture that provides the test shot number for the current test.

    Parametrized across TEST_SHOTS to run tests on multiple shots.
    IDS-specific configs can exclude certain shots via 'exclude_shots' list.

    Determines the IDS name from the test file name (e.g., test_ece_*.py -> 'ece')
    and checks if the shot should be skipped for this IDS.

    Returns:
        int: Shot number to use for testing
    """
    shot = request.param

    # Extract IDS name from test file name
    # e.g., 'test_ece_requirements.py' -> 'ece'
    # e.g., 'test_ec_launchers_composition.py' -> 'ec_launchers'
    test_file = request.node.fspath.basename

    # Remove 'test_' prefix and '_requirements.py' or '_composition.py' suffix
    if test_file.startswith('test_'):
        test_file = test_file[5:]  # Remove 'test_'

    # Remove common suffixes
    for suffix in ['_requirements.py', '_composition.py', '.py']:
        if test_file.endswith(suffix):
            test_file = test_file[:-len(suffix)]
            break

    ids_name = test_file

    # Load config and check for excluded shots
    config = load_test_config(ids_name)
    exclude_shots = config.get('exclude_shots', [])

    if shot in exclude_shots:
        pytest.skip(f"Shot {shot} excluded for {ids_name} (no data available)")

    return shot


def load_ids_fields(ids_name):
    """
    Load field list from IDS YAML configuration file.

    Args:
        ids_name: IDS identifier (e.g., 'ece', 'thomson_scattering')

    Returns:
        List of full IDS paths with IDS prefix (e.g., ['ece.channel.name', ...])

    Example:
        >>> fields = load_ids_fields('ece')
        >>> print(fields[:3])
        ['ece.ids_properties.homogeneous_time', 'ece.line_of_sight.first_point.r', ...]
    """
    # Path to YAML file in ids directory
    yaml_path = Path(__file__).parent.parent / 'imas_composer' / 'ids' / f'{ids_name}.yaml'

    if not yaml_path.exists():
        raise FileNotFoundError(f"No YAML config found for IDS '{ids_name}' at {yaml_path}")

    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)

    # Get fields list from config
    fields = config.get('fields', [])

    # Prepend IDS name to each field to make full paths
    return [f'{ids_name}.{field}' for field in fields]


# ============================================================================
# Utility Functions
# ============================================================================

def fetch_requirements(requirements: list[Requirement]) -> dict:
    """
    Fetch multiple requirements from MDS+ in a single query.

    Args:
        requirements: List of Requirement objects to fetch

    Returns:
        Dict mapping requirement keys to fetched data
    """
    if not requirements:
        return {}

    # Group by (treename, shot)
    by_tree_shot = {}
    for req in requirements:
        key = (req.treename, req.shot)
        if key not in by_tree_shot:
            by_tree_shot[key] = []
        by_tree_shot[key].append(req)

    # Fetch each tree/shot combination separately
    raw_data = {}
    for (treename, shot), reqs in by_tree_shot.items():
        tdi_query = {req.mds_path: req.mds_path for req in reqs}

        try:
            result = mdsvalue('d3d', treename=treename, pulse=shot, TDI=tdi_query)
            tree_data = result.raw()

            for req in reqs:
                try:
                    raw_data[req.as_key()] = tree_data[req.mds_path]
                except Exception as e:
                    raw_data[req.as_key()] = e
        except Exception as e:
            for req in reqs:
                raw_data[req.as_key()] = e

    return raw_data


def get_ndim(arr):
    """
    Extract shape information from awkward or numpy arrays.

    Returns dimensionality of the data. Returns 0 for scalars.
    """
    if hasattr(arr, "ndim"):
        return arr.ndim
    else:
        return 0 # scalar

def resolve_and_compose(composer, ids_path, shot=REFERENCE_SHOT):
    """
    Resolve requirements and compose data using ImasComposer public API.

    Args:
        composer: ImasComposer instance
        ids_path: Full IDS path (e.g., 'ece.channel.t_e.data') or list of paths
        shot: Shot number (default: REFERENCE_SHOT)

    Returns:
        Composed value from imas_composer (single value if ids_path is str, dict if list)

    Raises:
        RuntimeError: If requirements cannot be resolved within max iterations
        Exception: If any requirement fetch fails (re-raises the fetch exception)
    """
    # Convert single path to list for batch API
    single_path = isinstance(ids_path, str)
    ids_paths = [ids_path] if single_path else ids_path

    raw_data = {}

    # Iteratively resolve requirements using batch API
    for _ in range(10):  # Max 10 iterations
        status, requirements = composer.resolve(ids_paths, shot, raw_data)

        if all(status.values()):
            break

        # Fetch requirements
        fetched = fetch_requirements(requirements)

        # Check if any fetched values are exceptions (from failed MDS+ access)
        for key, value in fetched.items():
            if isinstance(value, Exception):
                raise RuntimeError(f"Failed to fetch requirement {key}: {value}") from value

        raw_data.update(fetched)

    if not all(status.values()):
        unresolved = [path for path, resolved in status.items() if not resolved]
        raise RuntimeError(f"Could not resolve {unresolved} within 10 iterations")

    # Compose final data using batch API
    results = composer.compose(ids_paths, shot, raw_data)

    # Return single value if input was single path
    return results[ids_path] if single_path else results

def compare_values(composer_val, omas_val, label="value", rtol=1e-10, atol_float=1e-12, atol_array=1e-6):
    """
    Compare composer and OMAS values with appropriate method based on type.

    Args:
        composer_val: Value from imas_composer
        omas_val: Value from OMAS
        label: Description for error messages
        rtol: Relative tolerance for float comparisons (default: 1e-10)
        atol_float: Absolute tolerance for scalar floats (default: 1e-12)
        atol_array: Absolute tolerance for float arrays (default: 1e-6)

    Raises:
        AssertionError: If values don't match
    """
    if isinstance(omas_val, str):
        assert composer_val == omas_val, f"{label}: string mismatch"

    elif isinstance(omas_val, (int, np.integer)):
        assert composer_val == omas_val, f"{label}: int mismatch"

    elif isinstance(omas_val, (float, np.floating)):
        np.testing.assert_allclose(
            composer_val, omas_val,
            rtol=rtol, atol=atol_float,
            err_msg=f"{label}: float mismatch"
        )

    elif isinstance(omas_val, np.ndarray):
        # Convert awkward arrays to numpy if needed (for Thomson ragged data)
        if not isinstance(composer_val, np.ndarray):
            composer_val = np.asarray(composer_val)
        assert len(composer_val) == len(omas_val), f"{label}: array length mismatch"

        # Check dtype - handle strings and ints separately from floats
        if np.issubdtype(omas_val.dtype, np.str_) or np.issubdtype(omas_val.dtype, np.unicode_):
            # String arrays - use exact comparison
            np.testing.assert_array_equal(
                composer_val, omas_val,
                err_msg=f"{label}: string array mismatch"
            )
        elif np.issubdtype(omas_val.dtype, np.integer):
            # Integer arrays - use exact comparison
            np.testing.assert_array_equal(
                composer_val, omas_val,
                err_msg=f"{label}: int array mismatch"
            )
        else:
            # Float arrays - use tolerance-based comparison
            np.testing.assert_allclose(
                composer_val, omas_val,
                rtol=rtol, atol=atol_array,
                err_msg=f"{label}: float array mismatch"
            )

    else:
        # Fallback for other types
        assert composer_val == omas_val, f"{label}: value mismatch"


# ============================================================================
# Shared Fixtures
# ============================================================================

@pytest.fixture(scope='module')
def composer():
    """Create ImasComposer instance once per module."""
    return ImasComposer(efit_tree='EFIT01')


@pytest.fixture(scope='session')
def omas_data():
    """
    Factory fixture to fetch OMAS data for any IDS.

    Returns a function that takes an IDS name, optional field path, and optional shot number.
    Data is cached per shot number - all fetches for the same shot accumulate in the same ODS.
    This allows multiple fetch calls to build up a single ODS incrementally.

    Usage:
        # Fetch entire IDS (ECE, Thomson) for default shot
        omas_ece = omas_data('ece')

        # Fetch specific field (Equilibrium - avoids fetching huge 2D grids)
        omas_eq_time = omas_data('equilibrium', 'equilibrium.time')

        # Fetch time first, then gaps (both go to same ODS via cache)
        omas_data('equilibrium', 'equilibrium.time_slice.:.time')
        ods = omas_data('equilibrium', 'equilibrium.time_slice.:.boundary_separatrix.gap.*')

        # Fetch for different shot
        omas_data('equilibrium', 'equilibrium.time', shot=200001)

        # Reset cache before fetching (creates new ODS, but still caches it)
        ods = omas_data('equilibrium', 'equilibrium.time', reset_cache=True)

    Args:
        ids_name: IDS identifier (e.g., 'ece', 'thomson_scattering', 'equilibrium')
        ids_path: Optional specific IDS path to fetch (defaults to '{ids_name}.*')
        shot: Optional shot number (defaults to REFERENCE_SHOT=200000)
        reset_cache: If True, clear cache and start with fresh ODS (default: False)

    Returns:
        ODS object with fetched data (cached per shot)
    """
    cache = {}

    def _fetch_omas_data(ids_name, ids_path=None, shot=REFERENCE_SHOT, reset_cache=False):
        # Default to wildcard fetch for non-equilibrium IDS
        if ids_path is None:
            ids_path = f'{ids_name}.*'

        # Use shot as cache key to handle future tests that iterate over several shots
        # If reset_cache=True, clear cache and create new ODS
        if reset_cache or shot not in cache:
            ods = ODS()
            cache[shot] = ods
        else:
            ods = cache[shot]

        # For equilibrium with specific path, fetch only that field
        # For others, use wildcard
        machine_to_omas(ods, 'd3d', shot, ids_path, options={'EFIT_tree': 'EFIT01'})

        return cache[shot]

    return _fetch_omas_data


# ============================================================================
# Generic Test Functions
# ============================================================================

def run_requirements_resolution(ids_path, composer, shot=REFERENCE_SHOT, max_steps=10):
    """
    Generic helper function for requirement resolution testing.

    Iteratively resolves requirements using OMAS mdsvalue to fetch raw MDSplus data,
    verifying that full resolution is achieved and tracking resolution depth.

    Args:
        ids_path: IDS path to test (e.g., 'ece.channel.t_e.data')
        composer: ImasComposer instance
        shot: Shot number (defaults to REFERENCE_SHOT)
        max_steps: Maximum resolution iterations (defaults to 10)

    Returns:
        resolution_steps: Number of steps needed to fully resolve
    """
    # Load test configuration for this IDS
    ids_name = ids_path.split('.')[0]
    test_config = load_test_config(ids_name)
    allow_different_shot = test_config['requirement_validation']['allow_different_shot']

    raw_data = {}
    resolution_steps = 0

    # Iteratively resolve requirements
    for step in range(max_steps):
        fully_resolved, requirements = composer.resolve([ids_path], shot, raw_data)

        if fully_resolved:
            resolution_steps = step
            break

        # Requirements should be non-empty if not fully resolved
        assert len(requirements) > 0, f"{ids_path} not resolved but no requirements returned"

        # Validate requirement structure
        for req in requirements:
            assert hasattr(req, 'mds_path'), "Requirement must have mds_path"
            assert hasattr(req, 'shot'), "Requirement must have shot"
            assert hasattr(req, 'treename'), "Requirement must have treename"

            # Check if this specific MDS+ path is in the allow_different_shot list
            # allow_different_shot contains MDS+ paths (e.g., '.ts.BLESSED.header.calib_nums')
            is_calibration_data = req.mds_path in allow_different_shot

            # Check shot number (unless this MDS+ path is marked as calibration data)
            if not is_calibration_data:
                assert req.shot == shot, (
                    f"Requirement shot must match requested shot. "
                    f"Got {req.shot}, expected {shot} for MDS+ path '{req.mds_path}'. "
                    f"If this is expected (e.g., calibration data), add '{req.mds_path}' "
                    f"to allow_different_shot in test_config_{ids_name}.yaml"
                )

        # Fetch data from MDSplus via OMAS mdsvalue
        for req in requirements:
            try:
                mds = mdsvalue('d3d', req.treename, req.shot, req.mds_path)
                value = mds.raw()
                # IMPORTANT: Use tuple key (mds_path, shot, treename) to match Requirement.as_key()
                raw_data[(req.mds_path, req.shot, req.treename)] = value
            except Exception as e:
                pytest.fail(f"Failed to fetch {req.mds_path} from {req.treename}: {e}")

    # Should achieve full resolution within max_steps
    assert fully_resolved, f"{ids_path} could not be fully resolved in {max_steps} steps"

    return resolution_steps



def _compare_recursive(composer_value, ods, omas_path, rtol=1e-10, atol_float=1e-12, atol_array=1e-6):
    """
    Recursively compare composer value with OMAS data.

    Uses ndim to determine when to stop recursion and compare 1D arrays.
    Handles ragged arrays by slicing OMAS NaN-padded data to match composer length.

    Args:
        composer_value: Value from imas_composer
        ods: OMAS ODS object
        omas_path: OMAS path with .: or indices (e.g., 'ece.channel.:.t_e.data' or 'ece.channel.0.t_e.data')
        rtol: Relative tolerance for float comparisons
        atol_float: Absolute tolerance for scalar floats
        atol_array: Absolute tolerance for float arrays
    """
    if hasattr(composer_value, "ndim"):
        ndim = composer_value.ndim
    else:
        # Scalar
        ndim = 0

    # Base case: 0D (scalar) or 1D array - do comparison
    if ndim <= 1 or ":" not in omas_path:
        # Compare
        compare_values(composer_value, ods[omas_path], omas_path, rtol=rtol, atol_float=atol_float, atol_array=atol_array)

    else:
        # Recursive case: ndim > 1, iterate over outer dimension
        n_outer = len(composer_value)

        for i in range(n_outer):
            # Recurse into next level
            composer_elem = composer_value[i]

            # Replace first occurrence of ':' with the index for OMAS-style path
            # e.g., 'thomson_scattering.channel.:.n_e.time' -> 'thomson_scattering.channel.0.n_e.time'
            new_omas_path = omas_path.replace(':', str(i), 1)

            _compare_recursive(composer_elem, ods, new_omas_path, rtol=rtol, atol_float=atol_float, atol_array=atol_array)


def run_composition_against_omas(ids_path, composer, omas_data, ids_name, shot):
    """
    Generic helper function for composition validation against OMAS.

    Compares imas_composer output with OMAS reference implementation.
    Uses test config's omas_path_map to translate imas_composer paths to OMAS paths.

    For multi-dimensional data (especially ragged arrays from Thomson), recursively
    compares element-by-element, handling OMAS's NaN padding for ragged arrays.

    Args:
        ids_path: IDS path to test (e.g., 'ece.channel.t_e.data')
        composer: ImasComposer instance
        omas_data: OMAS data factory fixture
        ids_name: IDS name (e.g., 'ece', 'thomson_scattering')
        shot: Shot number (from test_shot fixture)
    """

    # Compose using imas_composer
    composer_value = resolve_and_compose(composer, ids_path, shot)

    # Load test config to get OMAS path mapping
    test_config = load_test_config(ids_name)
    omas_path_map = test_config.get('omas_path_map', {})
    omas_fetch_map = test_config.get('omas_fetch_map', {})
    field_tolerances = test_config.get('field_tolerances', {})

    # Get field-specific tolerances if configured
    field_tol = field_tolerances.get(ids_path, {})
    rtol = field_tol.get('rtol', 1e-10)
    atol_float = field_tol.get('atol', 1e-12)
    atol_array = field_tol.get('atol', 1e-6)

    # Determine fetch and access paths
    # omas_fetch_map overrides omas_path_map for machine_to_omas calls (supports wildcards and lists)
    # omas_path_map is used for ODS access (standard array notation)
    omas_fetch_spec = omas_fetch_map.get(ids_path, omas_path_map.get(ids_path, ids_path))
    omas_access_path = omas_path_map.get(ids_path, ids_path)

    # Fetch OMAS ODS object
    # For equilibrium and ec_launchers, fetch only specific fields to avoid loading unwanted data
    # For other IDS, still fetch entire IDS with wildcard for backward compatibility
    if ids_name in ['equilibrium', 'ec_launchers']:
        # Handle list of paths (for fetch order control) or single path
        if isinstance(omas_fetch_spec, list):
            # Fetch each path in order (important for OMAS broadcasting)
            # Reset cache on first fetch for equilibrium (to avoid field accumulation interference)
            reset_first = (ids_name == 'equilibrium')
            ods = omas_data(ids_name, omas_fetch_spec[0], shot=shot, reset_cache=reset_first)
            for fetch_path in omas_fetch_spec[1:]:
                omas_data(ids_name, fetch_path, shot=shot)  # Additional fetches to same ODS
        else:
            # Reset cache for equilibrium (to avoid field accumulation interference)
            reset = (ids_name == 'equilibrium')
            ods = omas_data(ids_name, omas_fetch_spec, shot=shot, reset_cache=reset)
    else:
        ods = omas_data(ids_name, shot=shot)

    # Recursively compare using ndim-based logic with field-specific tolerances
    _compare_recursive(composer_value, ods, omas_access_path, rtol=rtol, atol_float=atol_float, atol_array=atol_array)