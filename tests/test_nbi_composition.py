"""
Test NBI IDS data composition against OMAS.

Single parametric test that verifies compose() matches OMAS for all NBI fields.
"""

import numpy as np
import pytest
from tests.conftest import load_ids_fields, run_composition_against_omas, resolve_and_compose, compare_values

pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus, pytest.mark.omas_validation]

# Per-beam fields compared by name in test_composition_matches_omas_by_beam_name:
# imas_composer lists every beam, with empty arrays for a beam without data,
# while the OMAS reference (nbi_active_hardware) lists only the beams with a
# PINJ time base, so the two beam lists differ in length.
SIGNAL_FIELDS = [
    'power_launched.time', 'power_launched.data',
    'energy.time', 'energy.data',
]


@pytest.mark.parametrize('ids_path', load_ids_fields('nbi'))
def test_composition_matches_omas(ids_path, composer, omas_data, test_shot):
    """Test that composed data matches OMAS for each NBI field."""
    run_composition_against_omas(ids_path, composer, omas_data, 'nbi', test_shot)


def test_composition_matches_omas_by_beam_name(composer, omas_data, test_shot):
    """
    Test per-beam fields against OMAS, matched by beam name rather than index.

    imas_composer lists every beam; OMAS's nbi_active_hardware skips beams
    without a PINJ time base. Beams OMAS lists are compared field by field;
    the beams it skips must be empty in imas_composer. For a beam without GAS
    (it did not fire) imas_composer reports species.a as NaN where OMAS
    assumes deuterium, so species.a is only compared where it is known.
    """
    composer_names = list(resolve_and_compose(composer, 'nbi.unit.name', test_shot))
    composed = {field: resolve_and_compose(composer, f'nbi.unit.{field}', test_shot)
                for field in [*SIGNAL_FIELDS, 'species.a']}

    ods = omas_data('nbi', shot=test_shot)
    omas_names = [ods['nbi.unit'][i]['name'] for i in ods['nbi.unit']]
    assert set(omas_names) <= set(composer_names), f"OMAS beams missing from imas_composer: {omas_names}"

    for beam_idx, beam_name in enumerate(composer_names):
        if beam_name not in omas_names:
            for field in SIGNAL_FIELDS:
                assert len(composed[field][beam_idx]) == 0, \
                    f"nbi.unit.{beam_name}.{field}: OMAS has no PINJ time base but imas_composer has data"
            continue
        omas_idx = omas_names.index(beam_name)
        for field in SIGNAL_FIELDS:
            compare_values(
                composed[field][beam_idx],
                ods['nbi.unit'][omas_idx][field],
                label=f'nbi.unit.{beam_name}.{field}'
            )
        if not np.isnan(composed['species.a'][beam_idx]):
            compare_values(
                composed['species.a'][beam_idx],
                ods['nbi.unit'][omas_idx]['species.a'],
                label=f'nbi.unit.{beam_name}.species.a'
            )
