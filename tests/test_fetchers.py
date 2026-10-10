"""
Tests for fetch_requirements, which fetches through toksearch_d3d.

The dedup and pass-through tests monkeypatch fetch_many_from_req and run
anywhere.  The remaining tests fetch real data and are skipped when toksearch_d3d
is not installed.
"""

import importlib.util

import pytest
import numpy as np

from imas_composer import ImasComposer, simple_load
from imas_composer.core import Requirement
from imas_composer.fetchers import fetch_requirements

from tests.conftest import REFERENCE_SHOT


TOKSEARCH_INSTALLED = importlib.util.find_spec("toksearch_d3d") is not None

requires_toksearch = pytest.mark.skipif(
    not TOKSEARCH_INSTALLED, reason="toksearch_d3d is not installed"
)


@pytest.fixture
def recorded_fetch(monkeypatch):
    """Replace fetch_many_from_req with a recorder, returning the list of call args."""
    calls = []

    def _record(reqs):
        calls.append(reqs)
        return {req.as_key(): np.zeros(3) for req in reqs}

    monkeypatch.setattr("imas_composer.fetchers.TOKSEARCH_AVAILABLE", True)
    monkeypatch.setattr("imas_composer.fetchers.fetch_many_from_req", _record)
    return calls


def test_requirements_passed_through_unmodified(recorded_fetch):
    """fetch_requirements hands the deduplicated reqs straight to fetch_many_from_req."""
    req = Requirement("BT", REFERENCE_SHOT, "__ptdata__")

    fetch_requirements([req])

    assert len(recorded_fetch) == 1
    assert recorded_fetch[0] == [req]


def test_duplicate_requirements_fetched_once(recorded_fetch):
    """Requirements sharing an as_key() are fetched a single time."""
    reqs = [Requirement("BT", REFERENCE_SHOT, "__ptdata__")] * 3

    result = fetch_requirements(reqs)

    assert len(recorded_fetch) == 1
    assert len(recorded_fetch[0]) == 1
    assert len(result) == 1


def test_failure_is_stored_in_band(monkeypatch):
    """An in-band Exception from fetch_many_from_req passes straight through."""
    def _fake_fetch_many(reqs):
        return {req.as_key(): ValueError("%TREE-E-NODATA") for req in reqs}

    monkeypatch.setattr("imas_composer.fetchers.TOKSEARCH_AVAILABLE", True)
    monkeypatch.setattr("imas_composer.fetchers.fetch_many_from_req", _fake_fetch_many)

    req = Requirement("NOSUCHPOINT", REFERENCE_SHOT, "__ptdata__")
    result = fetch_requirements([req])

    assert isinstance(result[req.as_key()], ValueError)


def test_empty_requirements_returns_empty():
    """An empty requirement list short-circuits before the availability check."""
    assert fetch_requirements([]) == {}


@pytest.mark.integration
@pytest.mark.requires_mdsplus
@pytest.mark.requires_toksearch
@requires_toksearch
class TestAgainstRealData:
    """Tests that fetch real DIII-D data through toksearch_d3d."""

    def test_ptdata_requirement_shape(self):
        """A __ptdata__ requirement returns the data/times/rarray dict."""
        req = Requirement("BT", REFERENCE_SHOT, "__ptdata__")

        value = fetch_requirements([req])[req.as_key()]

        assert not isinstance(value, Exception), value
        assert set(value) == {'data', 'times', 'rarray'}
        assert len(value['data']) > 1
        assert len(value['times']) == len(value['data'])
        assert len(value['rarray']) > 4

    def test_bad_pointname_returns_degenerate_data(self):
        """A nonexistent pointname is not an error: ptdata2() TDI returns a
        degenerate result rather than raising.  ptdata2 is legacy and deeply
        embedded, so we cannot make it signal a missing point; instead a bad
        point comes back as a single-sample zero 'data'/'times' (a real point
        has many samples), which callers must treat as "no data".
        """
        req = Requirement("NOSUCHPOINT", REFERENCE_SHOT, "__ptdata__")

        value = fetch_requirements([req])[req.as_key()]

        assert not isinstance(value, Exception), value
        assert set(value) == {'data', 'times', 'rarray'}
        assert len(value['data']) == 1
        assert np.all(np.asarray(value['data']) == 0)
        assert len(value['times']) == len(value['data'])

    def test_tree_requirement(self):
        """A named-tree requirement returns an array."""
        composer = ImasComposer()
        paths = composer.get_supported_fields('equilibrium')
        _, requirements = composer.resolve(paths, REFERENCE_SHOT, {})
        tree_reqs = [r for r in requirements if r.treename not in (None, "__ptdata__")]
        assert tree_reqs, "expected at least one named-tree requirement"

        req = tree_reqs[0]
        value = fetch_requirements([req])[req.as_key()]

        assert not isinstance(value, Exception), value
        assert np.asarray(value).size > 0

    @pytest.mark.parametrize("pointname", ["BT", "IP", "IPSPR15V", "DIAMAG3"])
    def test_ptdata_times_are_milliseconds(self, pointname):
        """Compose functions divide 'times' by 1000, so the fetcher must return ms."""
        req = Requirement(pointname, REFERENCE_SHOT, "__ptdata__")

        value = fetch_requirements([req])[req.as_key()]

        assert not isinstance(value, Exception), value
        times = np.asarray(value['times'])
        # DIII-D shots run for a few seconds; in ms that is O(1e3), not O(1)
        assert times[-1] > 100.0, f"{pointname}: times look like seconds, not ms"

    def test_simple_load_end_to_end(self):
        """simple_load resolves, fetches and composes through the toksearch backend."""
        paths = ['tf.b_field_tor_vacuum_r.data', 'tf.b_field_tor_vacuum_r.time']

        result = simple_load(paths, REFERENCE_SHOT)

        assert set(result) == set(paths)
        data = np.asarray(result['tf.b_field_tor_vacuum_r.data'])
        time = np.asarray(result['tf.b_field_tor_vacuum_r.time'])
        assert data.size > 1
        assert time.size == data.size
        # time is converted to seconds during compose
        assert time[-1] < 100.0
