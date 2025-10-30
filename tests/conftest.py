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

        # Navigate to channel array
        value = omas_data[ids_name]
        for key in prefix:
            value = value[key]

        # Get channel array
        channels = value['channel']

        # Extract field from each channel
        result = []
        for ch in channels:
            ch_value = ch
            for key in suffix:
                ch_value = ch_value[key]
            result.append(ch_value)

        return result

    # For non-channel fields, navigate directly
    value = omas_data[ids_name]
    for key in path_parts:
        value = value[key]

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
    return ImasComposer()


@pytest.fixture(scope='session')
def omas_data():
    """
    Factory fixture to fetch OMAS data for any IDS.

    Returns a function that takes an IDS name and returns cached OMAS data.
    Data is cached per IDS to avoid redundant fetches.

    Usage:
        @pytest.fixture(scope='session')
        def omas_ece(omas_data):
            return omas_data('ece')

    Args:
        ids_name: IDS identifier (e.g., 'ece', 'thomson_scattering')

    Returns:
        ODS object with fetched data
    """
    cache = {}

    def _fetch_omas_data(ids_name):
        if ids_name not in cache:
            ods = ODS()
            machine_to_omas(ods, 'd3d', REFERENCE_SHOT, f'{ids_name}.*')
            cache[ids_name] = ods
        return cache[ids_name]

    return _fetch_omas_data