"""
Test Equilibrium IDS data composition against OMAS.

Single parametric test that verifies compose() matches OMAS for all Equilibrium fields.
"""
import pytest

from tests.conftest import load_ids_fields, run_composition_against_omas

# Fields that require special handling (tested separately)
# profiles_2d magnetic fields use OMAS physics derivation (see test_equilibrium_profiles_2d_bfields.py)
EXCLUDED_FIELDS = [
    'equilibrium.time_slice.profiles_2d.b_field_tor',
    'equilibrium.time_slice.profiles_2d.b_field_r',
    'equilibrium.time_slice.profiles_2d.b_field_z',
]

# Filter out excluded fields from parametrization
all_fields = load_ids_fields('equilibrium')
test_fields = [f for f in all_fields if f not in EXCLUDED_FIELDS]

pytestmark = [pytest.mark.omas_validation]
@pytest.mark.parametrize('ids_path', test_fields)
def test_composition_matches_omas(ids_path, composer, omas_data, test_shot):
    """Test that composed data matches OMAS for each Equilibrium field."""
    run_composition_against_omas(ids_path, composer, omas_data, 'equilibrium', test_shot)
