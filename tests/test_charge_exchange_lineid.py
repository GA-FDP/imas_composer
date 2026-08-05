"""
Regression tests for CER LINEID parsing.

Covers the hand-rolled Roman numeral conversion and the element table behind
charge_exchange.channel.ion.{label,a,z_ion,z_n}. These run offline (no MDSplus).

The same parser is duplicated in the vendored OMAS fork so that
omas/machine_mappings/d3d.py stays importable without imas_composer; the two are
pinned together here because the OMAS comparison test relies on them agreeing.
"""
import numpy as np
import pytest

from imas_composer.ids.charge_exchange import CER_ELEMENTS, _parse_lineid, _roman_to_int
from omas.machine_mappings.d3d import parse_cer_lineid


# Every line CERFIT can be configured to fit, with the identity it must produce.
# lineid -> (label, a, z_ion, z_n)
MASTER_LINES = {
    'D I 4-2': ('2H1', 2.0, 1.0, 1.0),
    'B V 7-6': ('11B5', 11.0, 5.0, 5.0),
    'Ne X 11-10': ('20Ne10', 20.0, 10.0, 10.0),
    'N VII 9-8': ('14N7', 14.0, 7.0, 7.0),
    'He II 4-3': ('4He2', 4.0, 2.0, 2.0),
    'O VIII 8-7': ('16O8', 16.0, 8.0, 8.0),
    'D I 3-2': ('2H1', 2.0, 1.0, 1.0),
    'C VI 8-7': ('12C6', 12.0, 6.0, 6.0),
    'C VI 7-6': ('12C6', 12.0, 6.0, 6.0),
    'Ar XVI 14-13': ('40Ar16', 40.0, 16.0, 18.0),
    'Ar XVI 13-12': ('40Ar16', 40.0, 16.0, 18.0),
    'Ar XVIII 14-13': ('40Ar18', 40.0, 18.0, 18.0),
    'C VI 9-8': ('12C6', 12.0, 6.0, 6.0),
    'F IX 9-8': ('19F9', 19.0, 9.0, 9.0),
    'F IX 10-9': ('19F9', 19.0, 9.0, 9.0),
    'Ca XVIII 14-13': ('40Ca18', 40.0, 18.0, 20.0),
    'Ca XX 15-14': ('40Ca20', 40.0, 20.0, 20.0),
    'Li I 2-1': ('7Li1', 7.0, 1.0, 3.0),
    'C IV 6h-7i': ('12C4', 12.0, 4.0, 6.0),
    'Li III 7-5': ('7Li3', 7.0, 3.0, 3.0),
    'Al XIII 12-11': ('27Al13', 27.0, 13.0, 13.0),
    'Si XIV 12-11': ('28Si14', 28.0, 14.0, 14.0),
    'Ca XVIII 15-14': ('40Ca18', 40.0, 18.0, 20.0),
    'Ne IX 11-10': ('20Ne9', 20.0, 9.0, 10.0),
    'C IV 6-5': ('12C4', 12.0, 4.0, 6.0),
    'Ar XVII 16-15': ('40Ar17', 40.0, 17.0, 18.0),
    'Ar XVI 15-14': ('40Ar16', 40.0, 16.0, 18.0),
    'Ca XVIII 16-15': ('40Ca18', 40.0, 18.0, 20.0),
    'Kr XXV 20-19': ('84Kr25', 84.0, 25.0, 36.0),
    'Kr XXVII 21-20': ('84Kr27', 84.0, 27.0, 36.0),
    'O VIII 10-9': ('16O8', 16.0, 8.0, 8.0),
    'O VIII 9-8': ('16O8', 16.0, 8.0, 8.0),
}

# CER charge states run I..XXVII; the whole range is pinned so a rewrite of the
# converter cannot silently change any reachable value.
ROMAN_NUMERALS = [
    ('I', 1), ('II', 2), ('III', 3), ('IV', 4), ('V', 5), ('VI', 6), ('VII', 7),
    ('VIII', 8), ('IX', 9), ('X', 10), ('XI', 11), ('XII', 12), ('XIII', 13),
    ('XIV', 14), ('XV', 15), ('XVI', 16), ('XVII', 17), ('XVIII', 18), ('XIX', 19),
    ('XX', 20), ('XXI', 21), ('XXII', 22), ('XXIII', 23), ('XXIV', 24), ('XXV', 25),
    ('XXVI', 26), ('XXVII', 27), ('XXVIII', 28), ('XXIX', 29), ('XXX', 30),
]


@pytest.mark.parametrize('roman,expected', ROMAN_NUMERALS)
def test_roman_to_int(roman, expected):
    """Roman numerals over the full CER charge-state range convert correctly."""
    assert _roman_to_int(roman) == expected


@pytest.mark.parametrize('roman,expected', [('XL', 40), ('L', 50), ('XC', 90), ('C', 100)])
def test_roman_to_int_handles_subtractive_pairs_above_range(roman, expected):
    """Subtractive notation keeps working past the charge states CER actually uses."""
    assert _roman_to_int(roman) == expected


@pytest.mark.parametrize('lineid,expected', sorted(MASTER_LINES.items()))
def test_parse_lineid_master_list(lineid, expected):
    """Every configurable CERFIT line parses to the documented ion identity."""
    result = _parse_lineid(lineid, 'TANGENTIAL', 1)
    assert (result['label'], result['a'], result['z_ion'], result['z_n']) == expected


@pytest.mark.parametrize('lineid', ['C IV 6h-7i', 'C IV 6-5', 'Ca XVIII 14-13',
                                    'Ne IX 11-10', 'Kr XXV 20-19', 'Li I 2-1'])
def test_nuclear_charge_is_not_the_charge_state(lineid):
    """z_n is the element's nuclear charge, which for these lines differs from z_ion.

    Charge exchange reports the pre-capture ion, so 'C IV' means z_ion 4 while carbon's
    z_n stays 6. Collapsing the two is the bug in the OMFIT reference implementation.
    """
    result = _parse_lineid(lineid, 'VERTICAL', 3)
    assert result['z_n'] != result['z_ion']
    assert result['z_n'] == CER_ELEMENTS[lineid.split()[0]][0]


def test_deuterium_is_labelled_as_hydrogen():
    """D is reported as the hydrogen isotope of mass 2, matching the ida_lite convention."""
    assert _parse_lineid('D I 3-2', 'TANGENTIAL', 1)['label'] == '2H1'


@pytest.mark.parametrize('lineid', [b'C VI 8-7', np.str_('C VI 8-7'), np.array('C VI 8-7'),
                                    '  C VI 8-7  ', 'CVI8-7'])
def test_parse_lineid_accepts_mdsplus_string_flavours(lineid):
    """MDSplus hands back bytes, numpy scalars or 0-d arrays depending on the fetch path."""
    assert _parse_lineid(lineid, 'TANGENTIAL', 1)['label'] == '12C6'


@pytest.mark.parametrize('lineid', [None, '', 'garbage', 'Xx IV 3-2', 'C 8-7', 'ZZ ZZ 1-0'])
def test_parse_lineid_rejects_bad_values(lineid):
    """Unusable LINEIDs raise rather than silently defaulting to carbon."""
    with pytest.raises(ValueError, match='VERTICAL channel 07'):
        _parse_lineid(lineid, 'VERTICAL', 7)


@pytest.mark.parametrize('lineid', sorted(MASTER_LINES))
def test_composer_and_omas_parsers_agree(lineid):
    """The duplicated OMAS-side parser must stay identical to the composer's.

    The charge_exchange OMAS comparison test is only meaningful while these match.
    """
    composer = _parse_lineid(lineid, 'TANGENTIAL', 1)
    label, a, z_ion, z_n = parse_cer_lineid(lineid, 'TANGENTIAL', 1)
    assert (label, a, z_ion, z_n) == (composer['label'], composer['a'],
                                      composer['z_ion'], composer['z_n'])
