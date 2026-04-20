"""
Test Interferometer IDS data composition against OMAS.

Single parametric test that verifies compose() matches OMAS for all Interferometer fields.

Parametrization is handled by pytest_generate_tests in conftest.py, which creates
test variants for CO2_only and CO2_RIP configurations.
"""

import pytest
from tests.conftest import run_composition_against_omas

pytestmark = [pytest.mark.omas_validation]


def test_composition_matches_omas(ids_path, composer, omas_data, test_shot):
    """Test that composed data matches OMAS for each Interferometer field."""
    run_composition_against_omas(ids_path, composer, omas_data, 'interferometer', test_shot)
