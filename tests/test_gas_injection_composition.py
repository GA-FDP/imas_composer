"""
Test Gas Injection IDS data composition against OMAS.

Single parametric test that verifies compose() matches OMAS for all gas injection fields.
"""

import pytest
from tests.conftest import load_ids_fields, run_composition_against_omas

pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus, pytest.mark.omas_validation]


@pytest.mark.parametrize('ids_path', load_ids_fields('gas_injection'))
def test_composition_matches_omas(ids_path, composer, omas_data, test_shot):
    """Test that composed data matches OMAS for each gas injection field."""
    run_composition_against_omas(ids_path, composer, omas_data, 'gas_injection', test_shot)
