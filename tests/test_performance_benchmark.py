"""
Performance benchmarks for ImasComposer batch API.

Tests the performance improvements of the batch-first API by comparing:
1. Batch resolve/compose of all fields at once
2. Sequential resolve/compose of fields one at a time

Run with: pytest tests/test_performance_benchmark.py -v --benchmark-only
View stats: pytest tests/test_performance_benchmark.py --benchmark-autosave
"""
import pytest
from imas_composer import ImasComposer
from tests.conftest import fetch_requirements, REFERENCE_SHOT


pytestmark = pytest.mark.benchmark


@pytest.fixture
def equilibrium_fields():
    """Get all equilibrium fields for benchmarking."""
    composer = ImasComposer()
    return composer.get_supported_fields('equilibrium')


@pytest.fixture
def ece_fields():
    """Get all ECE fields for benchmarking."""
    composer = ImasComposer()
    return composer.get_supported_fields('ece')


def benchmark_batch_resolve_compose(composer, fields, shot):
    """
    Benchmark resolving and composing all fields using batch API.

    This is the optimized path that processes all fields at once.
    """
    raw_data = {}

    # Resolve all fields at once
    for _ in range(10):  # Max iterations
        status, requirements = composer.resolve(fields, shot, raw_data)

        if all(status.values()):
            break

        # Fetch requirements
        fetched = fetch_requirements(requirements)
        raw_data.update(fetched)

    # Compose all fields at once
    results = composer.compose(fields, shot, raw_data)
    return results


def benchmark_sequential_resolve_compose(composer, fields, shot):
    """
    Benchmark resolving and composing fields sequentially (old approach).

    This simulates the old pattern where each field was processed separately.
    Note: This uses the new batch API but calls it in a loop to simulate
    the overhead of the old approach.
    """
    all_results = {}

    for field in fields:
        raw_data = {}

        # Resolve single field (as list of 1)
        for _ in range(10):  # Max iterations
            status, requirements = composer.resolve([field], shot, raw_data)

            if all(status.values()):
                break

            # Fetch requirements
            fetched = fetch_requirements(requirements)
            raw_data.update(fetched)

        # Compose single field (as list of 1)
        results = composer.compose([field], shot, raw_data)
        all_results[field] = results[field]

    return all_results


def test_benchmark_equilibrium_batch(benchmark, equilibrium_fields):
    """Benchmark batch resolve/compose for all equilibrium fields."""
    composer = ImasComposer()

    # Run benchmark
    results = benchmark(
        benchmark_batch_resolve_compose,
        composer,
        equilibrium_fields,
        REFERENCE_SHOT
    )

    # Verify we got results
    assert len(results) == len(equilibrium_fields)


@pytest.mark.slow
def test_benchmark_equilibrium_sequential(benchmark, equilibrium_fields):
    """Benchmark sequential resolve/compose for equilibrium fields (old approach)."""
    composer = ImasComposer()

    # Run benchmark
    results = benchmark(
        benchmark_sequential_resolve_compose,
        composer,
        equilibrium_fields,
        REFERENCE_SHOT
    )

    # Verify we got results
    assert len(results) == len(equilibrium_fields)


@pytest.mark.slow
def test_benchmark_ece_batch(benchmark, ece_fields):
    """Benchmark batch resolve/compose for all ECE fields."""
    composer = ImasComposer(efit_tree='EFIT01')

    # Run benchmark
    results = benchmark(
        benchmark_batch_resolve_compose,
        composer,
        ece_fields,
        REFERENCE_SHOT
    )

    # Verify we got results
    assert len(results) == len(ece_fields)


@pytest.mark.slow
def test_benchmark_ece_sequential(benchmark, ece_fields):
    """Benchmark sequential resolve/compose for ECE fields (old approach)."""
    composer = ImasComposer(efit_tree='EFIT01')

    # Run benchmark
    results = benchmark(
        benchmark_sequential_resolve_compose,
        composer,
        ece_fields,
        REFERENCE_SHOT
    )

    # Verify we got results
    assert len(results) == len(ece_fields)


@pytest.mark.slow
def test_benchmark_small_batch(benchmark):
    """Benchmark batch API with small number of fields (5 fields)."""
    composer = ImasComposer()
    fields = composer.get_supported_fields('equilibrium')[:5]

    results = benchmark(
        benchmark_batch_resolve_compose,
        composer,
        fields,
        REFERENCE_SHOT
    )

    assert len(results) == 5


@pytest.mark.slow
def test_benchmark_small_sequential(benchmark):
    """Benchmark sequential processing with small number of fields (5 fields)."""
    composer = ImasComposer()
    fields = composer.get_supported_fields('equilibrium')[:5]

    results = benchmark(
        benchmark_sequential_resolve_compose,
        composer,
        fields,
        REFERENCE_SHOT
    )

    assert len(results) == 5
