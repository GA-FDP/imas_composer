"""
Test Magnetics IDS data composition against OMAS.

Single parametric test that verifies compose() matches OMAS for all magnetics fields.
"""

import pytest
from tests.conftest import load_ids_fields, run_composition_against_omas

pytestmark = [pytest.mark.omas_validation]


@pytest.mark.parametrize('ids_path', load_ids_fields('magnetics'))
def test_composition_matches_omas(ids_path, composer, omas_data, test_shot):
    """Test that composed data matches OMAS for each magnetics field."""
    run_composition_against_omas(ids_path, composer, omas_data, 'magnetics', test_shot)
