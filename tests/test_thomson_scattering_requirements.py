"""
Test Thomson Scattering IDS requirement resolution.

Single parametric test that verifies resolve() works for all Thomson fields.
"""

import pytest
from conftest import REFERENCE_SHOT, load_ids_fields

pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus]


@pytest.mark.parametrize('ids_path', load_ids_fields('thomson_scattering'))
def test_can_resolve_requirements(ids_path, composer):
    """Test that resolve() returns valid requirements for each Thomson field."""
    # Call resolve with empty raw_data
    fully_resolved, requirements = composer.resolve(ids_path, REFERENCE_SHOT, {})

    # Should not be fully resolved yet (no data fetched)
    assert not fully_resolved, f"{ids_path} should require data"

    # Should return some requirements
    assert len(requirements) > 0, f"{ids_path} should have requirements"

    # All requirements should be valid
    for req in requirements:
        assert hasattr(req, 'mds_path'), "Requirement should have mds_path"
        assert hasattr(req, 'shot'), "Requirement should have shot"
        assert hasattr(req, 'treename'), "Requirement should have treename"
        assert req.shot == REFERENCE_SHOT, "Requirement should use correct shot"
