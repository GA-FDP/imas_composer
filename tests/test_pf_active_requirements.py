"""
Test PF Active IDS requirement resolution.

Tests that the requirement system can fully resolve all dependencies
for pf_active fields through the three-stage requirement system.
"""

import pytest
from tests.conftest import load_ids_fields, run_requirements_resolution

pytestmark = [pytest.mark.requirements]

# Reference shot for requirement testing
REFERENCE_SHOT = 202161


@pytest.mark.parametrize('ids_path', load_ids_fields('pf_active'))
def test_can_resolve_requirements(ids_path, composer):
    """Test that resolve() can fully resolve requirements for each pf_active field."""
    resolution_steps = run_requirements_resolution(ids_path, composer, REFERENCE_SHOT)
    print(f"\n{ids_path}: resolved in {resolution_steps} steps")
