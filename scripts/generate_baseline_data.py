"""
Utility script to generate baseline data for cross-environment testing.

This script fetches all parametrized test fields (across all TEST_SHOTS and all fields)
using simple_load and stores them in a pickle file. The baseline data can then be used
to validate data fetching consistency across different conda environments and dependencies.

Usage:
    python scripts/generate_baseline_data.py [--output OUTPUT] [--ids IDS_NAME]

    --output OUTPUT   Output pickle file path (default: baseline_data.pkl)
    --ids IDS_NAME    Specific IDS to fetch (default: all)
"""

import argparse
import pickle
from pathlib import Path
import sys

# Add parent directory to path to import imas_composer
sys.path.insert(0, str(Path(__file__).parent.parent))

from imas_composer import simple_load, ImasComposer
from tests.conftest import load_ids_fields, load_test_config, TEST_SHOTS


def get_all_test_fields():
    """
    Get all parametrized test fields from all IDS configurations.

    Returns:
        dict: Mapping of ids_name -> list of field paths
    """
    # List of all IDS configurations (based on existing test files)
    ids_list = [
        'equilibrium',
        'ece',
        'thomson_scattering',
        'ec_launchers',
        'core_profiles'
        'reflectometer_profile'
    ]

    all_fields = {}

    for ids_name in ids_list:
        try:
            if ids_name == 'core_profiles':
                # core_profiles has multiple tree configurations
                zipfit_fields = load_ids_fields(ids_name, tree_filter='ZIPFIT')
                omfit_fields = load_ids_fields(ids_name, tree_filter='OMFIT_PROFS')
                all_fields[f'{ids_name}_ZIPFIT'] = zipfit_fields
                all_fields[f'{ids_name}_OMFIT_PROFS'] = omfit_fields
            else:
                fields = load_ids_fields(ids_name)
                all_fields[ids_name] = fields
        except FileNotFoundError:
            print(f"Warning: No configuration found for {ids_name}, skipping...")
            continue

    return all_fields


def fetch_baseline_data_for_shot(shot, ids_filter=None):
    """
    Fetch all test fields for the given shot.

    Args:
        shot: Shot number
        ids_filter: Optional IDS name to filter (e.g., 'equilibrium')

    Returns:
        dict: Nested dict of {ids_name: {field_path: data}}
    """
    all_fields = get_all_test_fields()

    # Filter to specific IDS if requested
    if ids_filter:
        # Handle core_profiles tree variants
        if ids_filter == 'core_profiles':
            all_fields = {k: v for k, v in all_fields.items()
                         if k.startswith('core_profiles')}
        else:
            all_fields = {k: v for k, v in all_fields.items() if k == ids_filter}

    baseline_data = {}

    for ids_key, fields in all_fields.items():
        print(f"\n  Fetching {ids_key} ({len(fields)} fields)...")

        # Determine composer parameters based on IDS
        if ids_key.endswith('_ZIPFIT'):
            composer = ImasComposer(profiles_tree='ZIPFIT01')
            ids_name = 'core_profiles'
        elif ids_key.endswith('_OMFIT_PROFS'):
            composer = ImasComposer(profiles_tree='OMFIT_PROFS', profiles_run_id='001')
            ids_name = 'core_profiles'
        else:
            composer = ImasComposer()
            ids_name = ids_key

        baseline_data[ids_key] = {}

        # Load test config to check for shot exclusions
        test_config = load_test_config(ids_name)
        exclude_shots = test_config.get('exclude_shots', [])
        field_shot_exclusions = test_config.get('field_shot_exclusions', {})

        # Skip entire IDS if shot is excluded
        if shot in exclude_shots:
            print(f"    Skipping {ids_key} - shot {shot} excluded")
            continue

        # Fetch each field
        for i, field_path in enumerate(fields, 1):
            # Check field-specific shot exclusions
            if field_path in field_shot_exclusions:
                if shot in field_shot_exclusions[field_path]:
                    print(f"    [{i}/{len(fields)}] Skipping {field_path} - shot {shot} excluded")
                    baseline_data[ids_key][field_path] = None
                    continue

            try:
                print(f"    [{i}/{len(fields)}] Fetching {field_path}...", end=' ')
                result = simple_load([field_path], shot, composer=composer)
                baseline_data[ids_key][field_path] = result[field_path]
                print("✓")
            except Exception as e:
                print(f"✗ Error: {e}")
                baseline_data[ids_key][field_path] = e

    return baseline_data


def fetch_all_baseline_data(ids_filter=None):
    """
    Fetch all test fields for all parametrized shots.

    Args:
        ids_filter: Optional IDS name to filter (e.g., 'equilibrium')

    Returns:
        dict: Nested dict of {shot: {ids_name: {field_path: data}}}
    """
    all_baseline_data = {}

    print(f"Fetching data for {len(TEST_SHOTS)} shots: {TEST_SHOTS}")

    for shot in TEST_SHOTS:
        print(f"\n{'='*70}")
        print(f"Shot {shot}")
        print(f"{'='*70}")

        shot_data = fetch_baseline_data_for_shot(shot, ids_filter=ids_filter)
        all_baseline_data[shot] = shot_data

    return all_baseline_data


def main():
    parser = argparse.ArgumentParser(
        description='Generate baseline data for cross-environment testing'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output pickle file path (default: baseline_data.pkl in repo root)'
    )
    parser.add_argument(
        '--ids',
        type=str,
        default=None,
        help='Specific IDS to fetch (default: all)'
    )

    args = parser.parse_args()

    # Default output to repo root
    if args.output is None:
        # Get repo root (parent of parent of this script)
        repo_root = Path(__file__).parent.parent
        output_path = repo_root / 'baseline_data.pkl'
    else:
        output_path = Path(args.output)

    print(f"Generating baseline data for all parametrized test shots")
    print(f"Output file: {output_path}")
    if args.ids:
        print(f"Filtering to IDS: {args.ids}")

    # Fetch all data for all shots
    all_baseline_data = fetch_all_baseline_data(ids_filter=args.ids)

    # Add metadata
    data_with_metadata = {
        'shots': TEST_SHOTS,
        'data': all_baseline_data,
        'metadata': {
            'description': 'Baseline data for cross-environment testing',
            'generated_by': 'scripts/generate_baseline_data.py',
            'shots': TEST_SHOTS,
        }
    }

    # Save to pickle
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'wb') as f:
        pickle.dump(data_with_metadata, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\n{'='*70}")
    print(f"✓ Baseline data saved to {output_path}")
    print(f"{'='*70}")

    # Print summary statistics per shot
    print("\nSummary per shot:")
    for shot in TEST_SHOTS:
        total_fields = 0
        successful_fields = 0
        failed_fields = 0
        skipped_fields = 0

        if shot in all_baseline_data:
            for ids_key, fields_data in all_baseline_data[shot].items():
                for field_path, data in fields_data.items():
                    total_fields += 1
                    if data is None:
                        skipped_fields += 1
                    elif isinstance(data, Exception):
                        failed_fields += 1
                    else:
                        successful_fields += 1

        print(f"\n  Shot {shot}:")
        print(f"    Total fields: {total_fields}")
        print(f"    Successful: {successful_fields}")
        print(f"    Failed: {failed_fields}")
        print(f"    Skipped: {skipped_fields}")


if __name__ == '__main__':
    main()
