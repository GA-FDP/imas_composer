"""
Test NBI IDS data composition against OMAS.

Single parametric test that verifies compose() matches OMAS for all NBI fields.
"""

import pytest
from tests.conftest import load_ids_fields, run_composition_against_omas, resolve_and_compose, compare_values

pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus, pytest.mark.omas_validation]

# Per-beam fields compared by name in test_composition_matches_omas_by_beam_name,
# because imas_composer now selects active beams via the dedicated FIRED flag
# while the OMAS reference (nbi_active_hardware) still selects them by PINJ
# time-trace presence, so the two beam lists can differ in order and length.
PER_BEAM_FIELDS = [
    'power_launched.time', 'power_launched.data',
    'energy.time', 'energy.data',
    'species.a',
]


@pytest.mark.parametrize('ids_path', load_ids_fields('nbi'))
def test_composition_matches_omas(ids_path, composer, omas_data, test_shot):
    """Test that composed data matches OMAS for each NBI field."""
    run_composition_against_omas(ids_path, composer, omas_data, 'nbi', test_shot)


def test_composition_matches_omas_by_beam_name(composer, omas_data, test_shot):
    """
    Test per-beam fields against OMAS, matched by beam name rather than index.

    imas_composer selects active beams via NB{beam_name}.FIRED; OMAS's
    nbi_active_hardware still selects them via PINJ time-trace presence, which
    also includes beams with a recorded trace but no actual injection. The two
    lists can therefore differ in order and length, so beams are matched by
    name instead of compared index-by-index.
    """
    composer_names = resolve_and_compose(composer, 'nbi.unit.name', test_shot)
    composed = {field: resolve_and_compose(composer, f'nbi.unit.{field}', test_shot)
                for field in PER_BEAM_FIELDS}

    ods = omas_data('nbi', shot=test_shot)
    omas_names = [ods['nbi.unit'][i]['name'] for i in ods['nbi.unit']]

    for beam_idx, beam_name in enumerate(composer_names):
        assert beam_name in omas_names, f"{beam_name}: fired beam missing from OMAS reference"
        omas_idx = omas_names.index(beam_name)
        for field in PER_BEAM_FIELDS:
            compare_values(
                composed[field][beam_idx],
                ods['nbi.unit'][omas_idx][field],
                label=f'nbi.unit.{beam_name}.{field}'
            )
