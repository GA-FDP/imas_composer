"""Test reflectometer_profile composition against OMAS."""
import pytest
from tests.conftest import load_ids_fields, run_composition_against_omas

pytestmark = [pytest.mark.omas_validation]


@pytest.mark.parametrize('ids_path', load_ids_fields('reflectometer_profile'))
def test_composition_matches_omas(ids_path, composer, omas_data, test_shot):
    """Test that composed data matches OMAS for each reflectometer_profile."""
    run_composition_against_omas(ids_path, composer, omas_data, 'reflectometer_profile', test_shot)