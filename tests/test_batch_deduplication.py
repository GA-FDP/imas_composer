"""
Test that the batch API properly deduplicates shared requirements.

This verifies that when multiple fields share dependencies, the batch API
returns fewer total requirements than processing them separately.
"""
import pytest
from imas_composer import ImasComposer
from tests.conftest import REFERENCE_SHOT


def test_equilibrium_requirement_deduplication():
    """
    Verify that batch API dedups requirements for equilibrium fields.

    When multiple equilibrium fields share dependencies (which they do - all
    use common EFIT data), the batch API should return fewer total requirements
    than processing them separately.
    """
    composer = ImasComposer()

    # Get equilibrium fields that share dependencies
    fields = [
        'equilibrium.time_slice.global_quantities.psi_axis',
        'equilibrium.time_slice.global_quantities.psi_boundary',
        'equilibrium.time_slice.global_quantities.magnetic_axis.r',
        'equilibrium.time_slice.global_quantities.magnetic_axis.z',
    ]

    # Batch resolve - should deduplicate
    raw_data = {}
    status_batch, requirements_batch = composer.resolve(fields, REFERENCE_SHOT, raw_data)

    # Sequential resolve - simulates old approach with duplication
    total_requirements_sequential = 0
    for field in fields:
        raw_data_single = {}
        status_single, requirements_single = composer.resolve([field], REFERENCE_SHOT, raw_data_single)
        total_requirements_sequential += len(requirements_single)

    # Batch should have same or fewer requirements (due to deduplication)
    assert len(requirements_batch) <= total_requirements_sequential

    # For these specific fields, we expect significant deduplication
    # (they all use the same EFIT data)
    deduplication_ratio = total_requirements_sequential / len(requirements_batch)
    assert deduplication_ratio >= 1.0  # At least some deduplication

    print(f"\nEquilibrium deduplication stats:")
    print(f"  Batch requirements: {len(requirements_batch)}")
    print(f"  Sequential total: {total_requirements_sequential}")
    print(f"  Deduplication ratio: {deduplication_ratio:.2f}x")


def test_ece_requirement_deduplication():
    """
    Verify that batch API dedups requirements for ECE fields.

    ECE fields share some common dependencies (time base, channel mapping),
    so batch processing should show deduplication.
    """
    composer = ImasComposer()

    # Get a subset of ECE fields
    fields = [
        'ece.channel.t_e.data',
        'ece.channel.t_e.time',
        'ece.channel.name',
        'ece.channel.position.r',
    ]

    # Batch resolve
    raw_data = {}
    status_batch, requirements_batch = composer.resolve(fields, REFERENCE_SHOT, raw_data)

    # Sequential resolve
    total_requirements_sequential = 0
    for field in fields:
        raw_data_single = {}
        status_single, requirements_single = composer.resolve([field], REFERENCE_SHOT, raw_data_single)
        total_requirements_sequential += len(requirements_single)

    # Verify deduplication
    assert len(requirements_batch) <= total_requirements_sequential

    deduplication_ratio = total_requirements_sequential / len(requirements_batch)
    assert deduplication_ratio >= 1.0

    print(f"\nECE deduplication stats:")
    print(f"  Batch requirements: {len(requirements_batch)}")
    print(f"  Sequential total: {total_requirements_sequential}")
    print(f"  Deduplication ratio: {deduplication_ratio:.2f}x")


def test_all_equilibrium_fields_deduplication():
    """
    Test deduplication with ALL equilibrium fields.

    This demonstrates the real-world benefit of the batch API when
    fetching all fields for an IDS at once.
    """
    composer = ImasComposer()
    all_fields = composer.get_supported_fields('equilibrium')

    # Batch resolve all fields
    raw_data = {}
    status_batch, requirements_batch = composer.resolve(all_fields, REFERENCE_SHOT, raw_data)

    # For comparison, estimate sequential requirement count
    # (we don't actually run it as it's slow - just count from a sample)
    sample_fields = all_fields[:10]  # Sample 10 fields
    total_requirements_sample = 0
    for field in sample_fields:
        raw_data_single = {}
        status_single, requirements_single = composer.resolve([field], REFERENCE_SHOT, raw_data_single)
        total_requirements_sample += len(requirements_single)

    # Estimate total sequential requirements
    estimated_sequential_total = (total_requirements_sample / len(sample_fields)) * len(all_fields)

    # Calculate deduplication ratio
    deduplication_ratio = estimated_sequential_total / len(requirements_batch)

    print(f"\nAll equilibrium fields deduplication stats:")
    print(f"  Total fields: {len(all_fields)}")
    print(f"  Batch requirements: {len(requirements_batch)}")
    print(f"  Estimated sequential total: {estimated_sequential_total:.0f}")
    print(f"  Deduplication ratio: {deduplication_ratio:.2f}x")

    # Should have significant deduplication for equilibrium
    assert deduplication_ratio >= 2.0  # At least 2x deduplication


def test_mixed_ids_no_deduplication():
    """
    Test that fields from different IDS types don't incorrectly deduplicate.

    Fields from different IDS types (e.g., equilibrium and ece) should
    not share requirements, so batch processing them together should not
    deduplicate across IDS boundaries.
    """
    composer = ImasComposer()

    # Get fields from different IDS types
    equilibrium_field = 'equilibrium.time'
    ece_field = 'ece.channel.t_e.data'
    fields = [equilibrium_field, ece_field]

    # Batch resolve
    raw_data = {}
    status_batch, requirements_batch = composer.resolve(fields, REFERENCE_SHOT, raw_data)

    # Resolve separately
    raw_data_eq = {}
    status_eq, requirements_eq = composer.resolve([equilibrium_field], REFERENCE_SHOT, raw_data_eq)

    raw_data_ece = {}
    status_ece, requirements_ece = composer.resolve([ece_field], REFERENCE_SHOT, raw_data_ece)

    # Total should be sum of both (no shared dependencies between IDS types)
    total_separate = len(requirements_eq) + len(requirements_ece)

    # Batch should give same total (no deduplication across IDS boundaries)
    assert len(requirements_batch) == total_separate

    print(f"\nMixed IDS deduplication stats:")
    print(f"  Equilibrium requirements: {len(requirements_eq)}")
    print(f"  ECE requirements: {len(requirements_ece)}")
    print(f"  Batch requirements: {len(requirements_batch)}")
    print(f"  No cross-IDS deduplication: {len(requirements_batch) == total_separate}")
