"""
Julia - Interface for calling imas_composer from Julia (IMASdd.jl via PythonCall).

Julia does not use the awkward API. Awkward arrays are handed over in awkward's own
buffer format: a JSON form, a length and a dict of flat numpy buffers (data, offsets,
masks), which IMASdd.awkward_from_buffers turns back into nested Julia arrays.

Public API:
    julia_value: Convert one composed value into types PythonCall maps without awkward
    simple_load_julia: simple_load with every value passed through julia_value
"""

from typing import Any, Dict, List, Optional

import awkward as ak

from .composer import ImasComposer
from .fetchers import simple_load


def julia_value(value: Any) -> Any:
    """
    Convert one composed value into types PythonCall maps without awkward.

    Args:
        value: Value returned by ImasComposer.compose or simple_load

    Returns:
        For an awkward array, a dict with the JSON "form", the "length" and the flat numpy
        "buffers" of ak.to_buffers on the packed array. Any other value unchanged.
    """
    if isinstance(value, ak.Array):
        form, length, buffers = ak.to_buffers(ak.to_packed(value))
        return {"form": form.to_json(), "length": length, "buffers": buffers}
    return value


def simple_load_julia(
    ids_paths: List[str],
    shot: int,
    composer: Optional[ImasComposer] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """
    Run simple_load and convert every value with julia_value.

    Args:
        ids_paths: List of full IDS paths to compose (e.g., ['ece.channel.t_e.data'])
        shot: Shot number
        composer: Optional pre-configured ImasComposer instance
        **kwargs: Passed on to simple_load

    Returns:
        Dict mapping each ids_path -> julia_value of the composed data, or NoData
    """
    data = simple_load(ids_paths, shot, composer=composer, **kwargs)
    return {ids_path: julia_value(value) for ids_path, value in data.items()}
