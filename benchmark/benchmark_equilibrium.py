#!/usr/bin/env python3
"""
Benchmark script comparing imas_composer vs OMAS performance for equilibrium data fetching.

This script measures and compares the time required to fetch all equilibrium fields
using both imas_composer and OMAS, providing insights into relative performance.

Usage:
    python benchmark_equilibrium.py [--shot SHOT] [--efit-tree TREE] [--verbose]

Example:
    python benchmark_equilibrium.py --shot 204601 --verbose
"""

import argparse
import time
import sys
from pathlib import Path

# Add parent directory to path to import imas_composer
sys.path.insert(0, str(Path(__file__).parent.parent))

from imas_composer import ImasComposer
from omas import ODS, mdsvalue
from omas.omas_machine import machine_to_omas


def fetch_requirements(requirements):
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


def benchmark_imas_composer(shot, efit_tree, verbose=False):
    """
    Benchmark fetching all equilibrium fields with imas_composer.

    Args:
        shot: Shot number to fetch
        efit_tree: EFIT tree to use (e.g., 'EFIT01')
        verbose: If True, print progress messages

    Returns:
        Tuple of (elapsed_time, num_fields, num_requirements)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Benchmarking imas_composer for shot {shot} ({efit_tree})")
        print(f"{'='*70}")

    # Create composer
    composer = ImasComposer(efit_tree=efit_tree)

    # Get all supported equilibrium fields
    fields = composer.get_supported_fields('equilibrium')

    if verbose:
        print(f"Total fields to fetch: {len(fields)}")

    # Start timing
    start_time = time.time()

    # Track total requirements
    total_requirements = 0

    # Fetch each field
    for i, field in enumerate(fields):
        if verbose and (i % 10 == 0 or i == len(fields) - 1):
            print(f"  Progress: {i+1}/{len(fields)} fields", end='\r')

        try:
            # Use resolve_and_compose method
            composer.resolve_and_compose(field, shot, fetch_requirements)

            # Count requirements (for statistics)
            raw_data = {}
            fully_resolved, requirements = composer.resolve(field, shot, raw_data)
            total_requirements += len(requirements)

        except Exception as e:
            if verbose:
                print(f"\n  WARNING: Failed to fetch {field}: {e}")

    elapsed_time = time.time() - start_time

    if verbose:
        print(f"\n  Completed in {elapsed_time:.2f} seconds")
        print(f"  Average requirements per field: {total_requirements / len(fields):.1f}")

    return elapsed_time, len(fields), total_requirements


def benchmark_omas(shot, efit_tree, verbose=False):
    """
    Benchmark fetching all equilibrium fields with OMAS.

    Args:
        shot: Shot number to fetch
        efit_tree: EFIT tree to use (e.g., 'EFIT01')
        verbose: If True, print progress messages

    Returns:
        Tuple of (elapsed_time, num_fields)
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Benchmarking OMAS for shot {shot} ({efit_tree})")
        print(f"{'='*70}")

    # Create ODS
    ods = ODS()

    # Start timing
    start_time = time.time()

    # Fetch all equilibrium data with wildcard
    if verbose:
        print("  Fetching equilibrium.*...")

    try:
        machine_to_omas(ods, 'd3d', shot, 'equilibrium.*', options={'EFIT_tree': efit_tree})
    except Exception as e:
        if verbose:
            print(f"  WARNING: OMAS fetch failed: {e}")
        return None, None

    elapsed_time = time.time() - start_time

    # Count fields in ODS (approximate)
    num_fields = count_ods_fields(ods, 'equilibrium')

    if verbose:
        print(f"  Completed in {elapsed_time:.2f} seconds")
        print(f"  Total fields populated: {num_fields}")

    return elapsed_time, num_fields


def count_ods_fields(ods, prefix):
    """
    Count number of populated fields in an ODS with a given prefix.

    Args:
        ods: OMAS ODS object
        prefix: Field prefix (e.g., 'equilibrium')

    Returns:
        Number of fields
    """
    count = 0

    def _count_recursive(obj, path):
        nonlocal count

        if hasattr(obj, 'keys'):
            for key in obj.keys():
                _count_recursive(obj[key], f"{path}.{key}")
        elif hasattr(obj, '__len__') and not isinstance(obj, (str, bytes)):
            # Array-like - count as one field
            count += 1
        else:
            # Scalar value
            count += 1

    if prefix in ods:
        _count_recursive(ods[prefix], prefix)

    return count


def print_comparison(composer_time, composer_fields, omas_time, omas_fields):
    """
    Print comparison summary.

    Args:
        composer_time: imas_composer elapsed time (seconds)
        composer_fields: Number of fields fetched by imas_composer
        omas_time: OMAS elapsed time (seconds)
        omas_fields: Number of fields fetched by OMAS
    """
    print(f"\n{'='*70}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*70}")

    print(f"\nimas_composer:")
    print(f"  Time:   {composer_time:.2f}s")
    print(f"  Fields: {composer_fields}")
    if composer_fields > 0:
        print(f"  Rate:   {composer_time / composer_fields * 1000:.1f}ms per field")

    print(f"\nOMAS:")
    print(f"  Time:   {omas_time:.2f}s")
    print(f"  Fields: {omas_fields}")
    if omas_fields > 0:
        print(f"  Rate:   {omas_time / omas_fields * 1000:.1f}ms per field")

    print(f"\nComparison:")
    speedup = omas_time / composer_time if composer_time > 0 else 0
    if speedup > 1:
        print(f"  imas_composer is {speedup:.2f}x FASTER than OMAS")
    elif speedup < 1 and speedup > 0:
        print(f"  imas_composer is {1/speedup:.2f}x SLOWER than OMAS")

    print(f"\n{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark imas_composer vs OMAS for equilibrium data fetching'
    )
    parser.add_argument(
        '--shot',
        type=int,
        default=204601,
        help='Shot number to benchmark (default: 204601)'
    )
    parser.add_argument(
        '--efit-tree',
        type=str,
        default='EFIT01',
        help='EFIT tree to use (default: EFIT01)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Print verbose progress messages'
    )
    parser.add_argument(
        '--composer-only',
        action='store_true',
        help='Only benchmark imas_composer (skip OMAS)'
    )
    parser.add_argument(
        '--omas-only',
        action='store_true',
        help='Only benchmark OMAS (skip imas_composer)'
    )

    args = parser.parse_args()

    print(f"\nEquilibrium Data Fetch Benchmark")
    print(f"Shot: {args.shot}")
    print(f"EFIT Tree: {args.efit_tree}")

    composer_time = None
    composer_fields = None
    omas_time = None
    omas_fields = None

    # Benchmark imas_composer
    if not args.omas_only:
        composer_time, composer_fields, _ = benchmark_imas_composer(
            args.shot,
            args.efit_tree,
            verbose=args.verbose
        )

    # Benchmark OMAS
    if not args.composer_only:
        omas_time, omas_fields = benchmark_omas(
            args.shot,
            args.efit_tree,
            verbose=args.verbose
        )

    # Print comparison
    if composer_time is not None and omas_time is not None:
        print_comparison(composer_time, composer_fields, omas_time, omas_fields)
    elif composer_time is not None:
        print(f"\nimas_composer completed in {composer_time:.2f}s ({composer_fields} fields)")
    elif omas_time is not None:
        print(f"\nOMAS completed in {omas_time:.2f}s ({omas_fields} fields)")


if __name__ == '__main__':
    main()
