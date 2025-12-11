"""
Benchmark script for OMAS equilibrium data fetching.

This script measures the time required to fetch all equilibrium fields using OMAS.
This provides a baseline for comparing against imas_composer performance.

For imas_composer benchmarks, use pytest-benchmark tests:
    pytest tests/test_performance_benchmark.py -v --benchmark-only

Usage:
    python benchmark_equilibrium.py [--shot SHOT] [--efit-tree TREE] [--verbose]

Example:
    python benchmark_equilibrium.py --shot 204601 --verbose
"""

import argparse
import time

from omas import ODS
from omas.omas_machine import machine_to_omas


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


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark OMAS equilibrium data fetching'
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

    args = parser.parse_args()

    print(f"\nOMAS Equilibrium Data Fetch Benchmark")
    print(f"Shot: {args.shot}")
    print(f"EFIT Tree: {args.efit_tree}")
    print(f"\nNote: For imas_composer benchmarks, use:")
    print(f"  pytest tests/test_performance_benchmark.py -v --benchmark-only\n")

    # Benchmark OMAS
    omas_time, omas_fields = benchmark_omas(
        args.shot,
        args.efit_tree,
        verbose=args.verbose
    )

    # Print results
    if omas_time is not None:
        print(f"\n{'='*70}")
        print(f"OMAS BENCHMARK RESULTS")
        print(f"{'='*70}")
        print(f"  Time:   {omas_time:.2f}s")
        print(f"  Fields: {omas_fields}")
        if omas_fields > 0:
            print(f"  Rate:   {omas_time / omas_fields * 1000:.1f}ms per field")
        print(f"{'='*70}\n")
    else:
        print("\nOMAS benchmark failed")


if __name__ == '__main__':
    main()
