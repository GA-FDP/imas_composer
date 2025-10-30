"""
Shared pytest fixtures and utilities for imas_composer tests.

This file is automatically discovered by pytest and makes fixtures available
to all test files in this directory.
"""

import pytest
import numpy as np
import yaml
from pathlib import Path
from omas import ODS, mdsvalue
from omas.omas_machine import machine_to_omas

from imas_composer import ImasComposer
from imas_composer.core import Requirement


# Reference shot used across all tests
REFERENCE_SHOT = 200000


# ============================================================================
# YAML Configuration Utilities
# ============================================================================

def load_test_config(ids_name):
    """
    Load test configuration for an IDS.

    Args:
        ids_name: IDS identifier (e.g., 'ece', 'thomson_scattering')

    Returns:
        dict: Test configuration with validation rules and exceptions
    """
    config_path = Path(__file__).parent / f'test_config_{ids_name}.yaml'

    if not config_path.exists():
        # Return default config if no custom config exists
        return {
            'field_exceptions': {},
            'requirement_validation': {
                'allow_different_shot': []
            }
        }

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


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


def resolve_and_compose(composer, ids_path, shot=REFERENCE_SHOT):
    """
    Resolve requirements and compose data using ImasComposer public API.

    Args:
        composer: ImasComposer instance
        ids_path: Full IDS path (e.g., 'ece.channel.t_e.data')
        shot: Shot number (default: REFERENCE_SHOT)

    Returns:
        Composed value from imas_composer
    """
    raw_data = {}

    # Iteratively resolve requirements
    for _ in range(10):  # Max 10 iterations
        fully_resolved, requirements = composer.resolve(ids_path, shot, raw_data)

        if fully_resolved:
            break

        # Fetch requirements
        fetched = fetch_requirements(requirements)
        raw_data.update(fetched)

    if not fully_resolved:
        raise RuntimeError(f"Could not resolve {ids_path}")

    # Compose final data
    return composer.compose(ids_path, shot, raw_data)


def get_omas_value(omas_data, ids_path):
    """
    Navigate OMAS nested structure to get value for IDS path.

    Handles both scalar and channel fields generically.

    Args:
        omas_data: ODS object from OMAS
        ids_path: Full IDS path (e.g., 'ece.channel.t_e.data')

    Returns:
        Value from OMAS (scalar or list for channel fields)

    Examples:
        'ece.channel.t_e.data' -> [channel[0]['t_e']['data'], channel[1]['t_e']['data'], ...]
        'ece.ids_properties.homogeneous_time' -> scalar value
    """
    # Split path and extract IDS name
    parts = ids_path.split('.')
    ids_name = parts[0]  # 'ece' or 'thomson_scattering'
    path_parts = parts[1:]  # Rest of path

    # For channel fields, return list of channel values
    if 'channel' in path_parts:
        channel_idx = path_parts.index('channel')
        prefix = path_parts[:channel_idx]
        suffix = path_parts[channel_idx + 1:]  # Skip 'channel'
        value = omas_data[ids_name]
        # Navigate to channel array
        if len(prefix) > 0:
            value = value[prefix]

        # Get channel array
        channels = value['channel']

        # Extract field from each channel
        result = []
        for ch in channels:
            result.append(channels[ch][suffix])

        return result

    # For non-channel fields, navigate directly
    value = omas_data[ids_name][path_parts]

    return value


def compare_values(composer_val, omas_val, label="value"):
    """
    Compare composer and OMAS values with appropriate method based on type.

    Args:
        composer_val: Value from imas_composer
        omas_val: Value from OMAS
        label: Description for error messages

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
            rtol=1e-10, atol=1e-12,
            err_msg=f"{label}: float mismatch"
        )

    elif isinstance(omas_val, np.ndarray):
        # Convert awkward arrays to numpy if needed (for Thomson ragged data)
        if not isinstance(composer_val, np.ndarray):
            composer_val = np.asarray(composer_val)

        np.testing.assert_allclose(
            composer_val, omas_val,
            rtol=1e-10, atol=1e-6,
            err_msg=f"{label}: array mismatch"
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

    Returns a function that takes an IDS name and optionally a specific field path.
    Data is cached per unique (ids_name, ids_path) combination.

    Usage:
        # Fetch entire IDS (ECE, Thomson)
        omas_ece = omas_data('ece')

        # Fetch specific field (Equilibrium - avoids fetching huge 2D grids)
        omas_eq_time = omas_data('equilibrium', 'equilibrium.time')

    Args:
        ids_name: IDS identifier (e.g., 'ece', 'thomson_scattering', 'equilibrium')
        ids_path: Optional specific IDS path to fetch (defaults to '{ids_name}.*')
                  Special case for equilibrium to avoid fetching all fields.

    Returns:
        ODS object with fetched data
    """
    cache = {}

    def _fetch_omas_data(ids_name, ids_path=None):
        # Default to wildcard fetch for non-equilibrium IDS
        if ids_path is None:
            ids_path = f'{ids_name}.*'

        # Use (ids_name, ids_path) as cache key to handle partial fetches
        cache_key = (ids_name, ids_path)

        if cache_key not in cache:
            ods = ODS()
            # For equilibrium with specific path, fetch only that field
            # For others, use wildcard
            machine_to_omas(ods, 'd3d', REFERENCE_SHOT, ids_path, options={'EFIT_tree': 'EFIT01'})
            cache[cache_key] = ods

        return cache[cache_key]

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
        fully_resolved, requirements = composer.resolve(ids_path, shot, raw_data)

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

            # Track which dependency path this requirement came from
            # by checking which unresolved dependencies could have generated it
            current_dep = None
            mapper, _ = composer._get_mapper_for_path(ids_path)
            for dep_path in _get_all_dependencies(mapper, ids_path):
                if dep_path in allow_different_shot:
                    current_dep = dep_path
                    break

            # Check shot number (unless this dependency allows different shots)
            if current_dep not in allow_different_shot:
                assert req.shot == shot, (
                    f"Requirement shot must match requested shot. "
                    f"Got {req.shot}, expected {shot}. "
                    f"If this is expected (e.g., calibration data), add '{current_dep or ids_path}' "
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


def _get_all_dependencies(mapper, ids_path):
    """
    Get all dependency paths for an IDS path (recursive).

    Args:
        mapper: IDS mapper instance
        ids_path: IDS path to analyze

    Returns:
        set: All dependency paths (direct and transitive)
    """
    deps = set()
    to_process = [ids_path]
    visited = set()

    while to_process:
        current = to_process.pop(0)
        if current in visited or current not in mapper.specs:
            continue
        visited.add(current)

        spec = mapper.specs[current]
        if spec.depends_on:
            for dep in spec.depends_on:
                deps.add(dep)
                to_process.append(dep)

    return deps


def run_composition_against_omas(ids_path, composer, omas_data, ids_name):
    """
    Generic helper function for composition validation against OMAS.

    Compares imas_composer output with OMAS reference implementation.

    Args:
        ids_path: IDS path to test (e.g., 'ece.channel.t_e.data')
        composer: ImasComposer instance
        omas_data: OMAS data factory fixture
        ids_name: IDS name (e.g., 'ece', 'thomson_scattering')
    """
    # Compose using imas_composer
    composer_value = resolve_and_compose(composer, ids_path)

    # Get OMAS value
    # For equilibrium, fetch only the specific field to avoid loading entire IDS
    if ids_name == 'equilibrium':
        omas_value = get_omas_value(omas_data(ids_name, ids_path), ids_path)
    else:
        # For other IDS, fetch entire IDS with wildcard
        omas_value = get_omas_value(omas_data(ids_name), ids_path)

    # Compare based on type
    if isinstance(omas_value, list):
        # Channel field - compare each channel
        assert len(composer_value) == len(omas_value), f"{ids_path}: length mismatch"

        for i in range(len(omas_value)):
            compare_values(composer_value[i], omas_value[i], f"{ids_path}[{i}]")
    else:
        # Scalar field - compare directly
        compare_values(composer_value, omas_value, ids_path)