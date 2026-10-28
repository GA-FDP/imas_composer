"""
Test Summary IDS requirement resolution.

Verifies that resolve() can fully resolve all requirements for each
summary IDS field using the TRANSPORT MDSplus tree.
"""
import pytest

from tests.conftest import REFERENCE_SHOT, load_ids_fields, run_requirements_resolution


@pytest.mark.parametrize('ids_path', load_ids_fields('summary'))
def test_can_resolve_requirements(ids_path, composer):
    """Test that resolve() can fully resolve requirements for each summary field."""
    resolution_steps = run_requirements_resolution(ids_path, composer, REFERENCE_SHOT)
    print(f"\n{ids_path}: resolved in {resolution_steps} steps")
