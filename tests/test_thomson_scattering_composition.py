"""
Test Thomson Scattering IDS data composition against OMAS.

Single parametric test that verifies compose() matches OMAS for all Thomson fields.
"""

import pytest
from conftest import resolve_and_compose, get_omas_value, compare_values, load_ids_fields

pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus, pytest.mark.omas_validation]


@pytest.mark.parametrize('ids_path', load_ids_fields('thomson_scattering'))
def test_composition_matches_omas(ids_path, composer, omas_thomson_data_cached):
    """Test that composed data matches OMAS for each Thomson field."""
    # Compose using imas_composer
    composer_value = resolve_and_compose(composer, ids_path)

    # Get OMAS value
    omas_value = get_omas_value(omas_thomson_data_cached, ids_path)

    # Compare based on type
    if isinstance(omas_value, list):
        # Channel field - compare each channel
        assert len(composer_value) == len(omas_value), f"{ids_path}: length mismatch"

        for i in range(len(omas_value)):
            compare_values(composer_value[i], omas_value[i], f"{ids_path}[{i}]")
    else:
        # Scalar field - compare directly
        compare_values(composer_value, omas_value, ids_path)
