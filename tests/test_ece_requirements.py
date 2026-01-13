"""
Test ECE IDS requirement resolution.

Single parametric test that verifies resolve() can fully resolve all requirements
for each ECE field, tracking resolution depth.

Parametrization is handled by pytest_generate_tests in conftest.py, which creates
test variants for fast_ece configuration.
"""
import pytest

from tests.conftest import REFERENCE_SHOT, run_requirements_resolution

def test_can_resolve_requirements(ids_path, composer):
    """Test that resolve() can fully resolve requirements for each ECE field."""
    resolution_steps = run_requirements_resolution(ids_path, composer, REFERENCE_SHOT)
    print(f"\n{ids_path}: resolved in {resolution_steps} steps")
