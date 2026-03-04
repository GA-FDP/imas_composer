"""
Test Charge Exchange IDS requirement resolution.

Single parametric test that verifies resolve() can fully resolve requirements
for each charge_exchange field, tracking resolution depth.
"""
import pytest

from tests.conftest import REFERENCE_SHOT, load_ids_fields, run_requirements_resolution

pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus]


@pytest.mark.parametrize('ids_path', load_ids_fields('charge_exchange'))
def test_can_resolve_requirements(ids_path, composer):
    """Test that resolve() can fully resolve requirements for each charge_exchange field."""
    resolution_steps = run_requirements_resolution(ids_path, composer, REFERENCE_SHOT)
    print(f"\n{ids_path}: resolved in {resolution_steps} steps")
