"""
Test Core Profiles IDS data composition against OMAS.

Single parametric test that verifies compose() matches OMAS for all core_profiles fields.

Note: ids_path is dynamically parametrized in conftest.py based on tree type
(ZIPFIT vs OMFIT_PROFS), so each tree only tests fields it supports.
"""

import pytest
from tests.conftest import run_composition_against_omas

pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus, pytest.mark.omas_validation]


def test_composition_matches_omas(ids_path, composer, omas_data, test_shot):
    """Test that composed data matches OMAS for each core_profiles field."""
    run_composition_against_omas(ids_path, composer, omas_data, 'core_profiles', test_shot)
