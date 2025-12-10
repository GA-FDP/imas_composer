"""
Test EC Launchers IDS requirement resolution.

Single parametric test that verifies resolve() can fully resolve all requirements
for each EC launchers field, tracking resolution depth.
"""
import pytest

from tests.conftest import load_ids_fields, run_requirements_resolution

pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus]
@pytest.mark.parametrize('ids_path', load_ids_fields('ec_launchers'))
def test_can_resolve_requirements(ids_path, composer, test_shot):
    """Test that resolve() can fully resolve requirements for each EC launchers field."""
    resolution_steps = run_requirements_resolution(ids_path, composer, test_shot)
    print(f"\n{ids_path}: resolved in {resolution_steps} steps")
