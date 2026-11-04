"""
Test the Julia interface: awkward arrays are handed over as awkward buffers that
reconstruct the original array, everything else passes through unchanged.
"""
import json

import awkward as ak
import numpy as np
import pytest

from imas_composer import NoData
from imas_composer.julia import julia_value


# Node classes IMASdd.awkward_from_buffers can read
SUPPORTED_CLASSES = {"NumpyArray", "EmptyArray", "ListOffsetArray", "RegularArray", "ByteMaskedArray"}


def _regular_of_list_array():
    """RegularArray > ListArray, as charge_exchange produces it before packing."""
    content = ak.contents.NumpyArray(np.arange(6.0))
    starts = ak.index.Index64(np.array([0, 2, 3, 5]))
    stops = ak.index.Index64(np.array([2, 3, 5, 6]))
    lists = ak.contents.ListArray(starts, stops, content)
    return ak.Array(ak.contents.RegularArray(lists, 2))


ARRAYS = [
    pytest.param(lambda: ak.Array([[1.0, 2.0, 3.0], [4.0], []]), id="var_float64"),
    pytest.param(lambda: ak.values_astype(ak.Array([[1.0, 2.0], [3.0]]), np.float32), id="var_float32"),
    pytest.param(lambda: ak.mask(ak.values_astype(ak.Array([[1.0, 2.0], [3.0, 4.0]]), np.float32),
                                 ak.Array([[True, False], [True, True]])), id="var_option_float32"),
    pytest.param(lambda: ak.Array([[[1.0], [2.0, 3.0]], [[]]]), id="var_var_float64"),
    pytest.param(_regular_of_list_array, id="regular_list_array"),
    pytest.param(lambda: ak.Array([[0], [1, 2], []]), id="var_int64"),
    pytest.param(lambda: ak.Array([["ECOILA", "ECOILA_1"], ["F1A"]]), id="var_string"),
    pytest.param(lambda: ak.Array(np.arange(6.0).reshape(2, 3)), id="regular_float64"),
    pytest.param(lambda: ak.Array(np.arange(3.0)), id="flat_float64"),
    pytest.param(lambda: ak.Array([]), id="empty"),
    pytest.param(lambda: ak.Array([[], []]), id="var_empty"),
    pytest.param(lambda: ak.Array(np.zeros((3, 0))), id="regular_size_0"),
]


def _classes(form: dict) -> set:
    """All node classes in a form dict."""
    contents = [form["content"]] if "content" in form else form.get("contents", [])
    return {form["class"]}.union(*(_classes(content) for content in contents))


@pytest.fixture(params=ARRAYS)
def array(request):
    return request.param()


@pytest.fixture
def converted(array):
    return julia_value(array)


def test_round_trip(array, converted):
    """The buffers rebuild the original array."""
    rebuilt = ak.from_buffers(converted["form"], converted["length"], converted["buffers"])
    assert ak.array_equal(rebuilt, array)


def test_plain_types(converted):
    """Julia only sees a str form, an int length and numpy buffers."""
    assert isinstance(converted["form"], str)
    assert isinstance(converted["length"], int)
    assert all(isinstance(buffer, np.ndarray) for buffer in converted["buffers"].values())


def test_supported_classes(converted):
    """Packing leaves only node classes the Julia reader supports."""
    assert _classes(json.loads(converted["form"])) <= SUPPORTED_CLASSES


@pytest.mark.parametrize("value", [
    pytest.param(np.arange(3.0), id="numpy_float"),
    pytest.param(np.array(["a", "bc"]), id="numpy_unicode"),
    pytest.param(1.5, id="float"),
    pytest.param(3, id="int"),
    pytest.param("EFIT01", id="str"),
    pytest.param(NoData("no data"), id="no_data"),
])
def test_passthrough(value):
    """Values that are not awkward arrays are returned unchanged."""
    assert julia_value(value) is value
