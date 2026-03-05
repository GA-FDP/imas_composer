"""
Tests for ImasComposer.get_supported_fields().

Verifies that the method correctly returns leaf paths for both top-level
IDS names and arbitrary dotted path prefixes.
"""
import pytest
from imas_composer import ImasComposer


@pytest.fixture(scope='module')
def composer():
    return ImasComposer()


def test_top_level_ids_returns_all_fields(composer):
    """Top-level IDS name returns all computed fields for that IDS."""
    fields = composer.get_supported_fields('ece')
    assert len(fields) > 0
    assert all(f.startswith('ece.') for f in fields)


def test_prefix_filters_to_subtree(composer):
    """A dotted prefix returns only fields under that subtree."""
    all_fields = composer.get_supported_fields('ece')
    channel_fields = composer.get_supported_fields('ece.channel')
    assert len(channel_fields) > 0
    assert len(channel_fields) < len(all_fields)
    assert all(f.startswith('ece.channel') for f in channel_fields)
    assert set(channel_fields).issubset(set(all_fields))


def test_prefix_not_in_subtree_excluded(composer):
    """Fields outside the prefix subtree are not returned."""
    channel_fields = composer.get_supported_fields('ece.channel')
    non_channel = [f for f in channel_fields if not f.startswith('ece.channel')]
    assert non_channel == []


def test_unknown_ids_raises(composer):
    """Unknown IDS name raises ValueError."""
    with pytest.raises(ValueError, match="No mapper for"):
        composer.get_supported_fields('nonexistent_ids')


def test_unknown_ids_raises_for_dotted_path(composer):
    """Unknown IDS name in dotted path raises ValueError."""
    with pytest.raises(ValueError, match="No mapper for"):
        composer.get_supported_fields('nonexistent_ids.some.field')


def test_narrow_prefix_returns_fewer_fields(composer):
    """A more specific prefix returns fewer or equal fields than a broader one."""
    equilibrium_fields = composer.get_supported_fields('equilibrium')
    ts_fields = composer.get_supported_fields('equilibrium.time_slice')
    gq_fields = composer.get_supported_fields('equilibrium.time_slice.global_quantities')
    assert len(equilibrium_fields) >= len(ts_fields) >= len(gq_fields) > 0


def test_works_for_multiple_ids(composer):
    """Works correctly across different IDS mappers."""
    for ids_name in ['ece', 'equilibrium', 'tf', 'magnetics']:
        fields = composer.get_supported_fields(ids_name)
        assert len(fields) > 0
        assert all(f.startswith(ids_name + '.') for f in fields)
