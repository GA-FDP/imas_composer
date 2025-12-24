"""
Test Core Profiles IDS requirement resolution.

Single parametric test that verifies resolve() can fully resolve all requirements
for each core_profiles field, tracking resolution depth.

Note: ids_path is dynamically parametrized in conftest.py based on tree type
(ZIPFIT vs OMFIT_PROFS), so each tree only tests fields it supports.
"""
import pytest

from tests.conftest import REFERENCE_SHOT, run_requirements_resolution

pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus]

def test_can_resolve_requirements(ids_path, composer):
    """Test that resolve() can fully resolve requirements for each core_profiles field."""
    resolution_steps = run_requirements_resolution(ids_path, composer, REFERENCE_SHOT)
    print(f"\n{ids_path}: resolved in {resolution_steps} steps")
