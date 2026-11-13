"""
Test that filter_padding drops invalid entries instead of masking them.
"""
import awkward as ak
import numpy as np
import pytest

from imas_composer.ids.equilibrium import filter_padding


@pytest.mark.parametrize(
    "arr, expected",
    [
        ([[1.0, 2.0], [3.0, 4.0]], [[1.0, 2.0], [3.0, 4.0]]),
        ([[1.0, 0.0], [3.0, 4.0]], [[1.0], [3.0, 4.0]]),
        ([[0.0, 2.0], [3.0, 4.0]], [[2.0], [3.0, 4.0]]),
        ([[0.0, 0.0], [3.0, 0.0]], [[], [3.0]]),
        ([[0.0, 0.0], [0.0, 0.0]], [[], []]),
        ([[0.0, 2.0, 0.0, 4.0]], [[2.0, 4.0]]),
    ],
    ids=["all_valid", "trailing_missing", "leading_missing", "empty_row", "all_empty", "interior_gaps"],
)
def test_filter_padding_drops_missing(arr, expected):
    arr = np.array(arr)
    result = filter_padding(arr, arr != 0)
    assert result.tolist() == expected
    assert str(result.type) == f"{len(expected)} * var * float64"
