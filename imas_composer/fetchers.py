"""
Fetchers - Data retrieval utilities for imas_composer.

This module provides concrete fetching implementations for use with ImasComposer.
It is intentionally separate from composer.py so that ImasComposer itself has no
dependency on any specific data backend (MDSplus, ptdata, etc.).

Public API:
    fetch_requirements: Fetch a list of Requirement objects from MDSplus via OMAS
    simple_load: Convenience wrapper that runs the full resolve-fetch-compose loop
"""

from typing import Dict, List, Tuple, Any, Optional
from .core import Requirement
from .composer import ImasComposer

try:
    from omas import mdsvalue
    OMAS_AVAILABLE = True
except ImportError:
    OMAS_AVAILABLE = False
    mdsvalue = None


def fetch_requirements(requirements: List[Requirement]) -> Dict[Tuple[str, int, str], Any]:
    """
    Fetch a list of requirements from MDSplus via OMAS mdsvalue.

    Requirements are grouped by (treename, shot) for efficient batching.
    treename=None is used for TDI expressions that do not require a named tree
    (e.g. ptdata2("BT", shot)).

    Requirements with treename == "__ptdata__" are treated as ptdata signals:
    the mds_path is used as the signal name and three TDI expressions are built
    (ptdata2 for data, dim_of for time, pthead2/__rarray for the header).  The
    result is stored as a dict with keys 'data', 'times' (ms), and 'rarray',
    matching the format expected by mapper compose functions.

    Args:
        requirements: List of Requirement objects to fetch.

    Returns:
        Dict mapping each requirement's as_key() tuple to its fetched value,
        or to the Exception if fetching failed.

    Raises:
        RuntimeError: If OMAS is not installed.
    """
    if not requirements:
        return {}

    if not OMAS_AVAILABLE:
        raise RuntimeError(
            "OMAS is required for fetching requirements but is not installed."
        )

    ptdata_reqs = [r for r in requirements if r.treename == "__ptdata__"]
    mds_reqs = [r for r in requirements if r.treename != "__ptdata__"]

    raw_data = {}

    # --- __ptdata__ requirements: translate to ptdata2/pthead2 TDI via OMAS ---
    if ptdata_reqs:
        seen_keys = set()
        for req in ptdata_reqs:
            k = req.as_key()
            if k in seen_keys:
                continue
            seen_keys.add(k)
            sig = req.mds_path
            shot = req.shot
            tdi = {
                'data':   f'ptdata2("{sig}",{shot})',
                'times':  f'dim_of(ptdata2("{sig}",{shot}),0)',
                'rarray': f'pthead2("{sig}",{shot}), __rarray',
            }
            try:
                result = mdsvalue('d3d', treename=None, pulse=shot, TDI=tdi)
                tree_data = result.raw()
                raw_data[k] = {
                    'data':   tree_data['data'],
                    'times':  tree_data['times'],
                    'rarray': tree_data['rarray'],
                }
            except Exception as e:
                raw_data[k] = e

    # --- MDSplus requirements ---
    if mds_reqs:
        by_tree_shot = {}
        for req in mds_reqs:
            key = (req.treename, req.shot)
            if key not in by_tree_shot:
                by_tree_shot[key] = []
            by_tree_shot[key].append(req)

        for (treename, shot), reqs in by_tree_shot.items():
            tdi_query = {req.mds_path: req.mds_path for req in reqs}
            try:
                result = mdsvalue('d3d', treename=treename, pulse=shot, TDI=tdi_query)
                tree_data = result.raw()
                for req in reqs:
                    try:
                        raw_data[req.as_key()] = tree_data[req.mds_path]
                    except Exception as e:
                        raw_data[req.as_key()] = e
            except Exception as e:
                for req in reqs:
                    raw_data[req.as_key()] = e

    return raw_data


def simple_load(
    ids_paths: List[str],
    shot: int,
    composer: Optional[ImasComposer] = None,
    efit_tree: str = "EFIT01",
    efit_run_id: str = "",
    profiles_tree: str = "ZIPFIT01",
    profiles_run_id: str = "",
    fast_ece: bool = False,
    max_iterations: int = 10
) -> Dict[str, Any]:
    """
    Simple utility function to resolve and compose IDS data in one call.

    Runs the full resolve-fetch-compose loop using fetch_requirements for data
    retrieval (MDSplus via OMAS and ptdata). Requires OMAS to be installed.

    Args:
        ids_paths: List of full IDS paths to compose (e.g., ['ece.channel.t_e.data'])
        shot: Shot number
        composer: Optional pre-configured ImasComposer instance. If None, creates new instance.
        efit_tree: EFIT tree (default: 'EFIT01', ignored if composer provided)
        efit_run_id: Run id appended to pulse for 'EFIT' tree (default: '')
        profiles_tree: Profiles tree (default: 'ZIPFIT01', ignored if composer provided)
        profiles_run_id: Run ID appended to pulse for OMFIT_PROFS tree (default: '')
        fast_ece: Whether to load fast_ece data (default: False)
        max_iterations: Maximum resolve-fetch iterations (default: 10)

    Returns:
        Dict mapping each ids_path -> composed IDS data

    Raises:
        RuntimeError: If requirements cannot be resolved or any fetch fails.

    Example:
        >>> result = simple_load(['equilibrium.time'], 200000)
        >>> result = simple_load(['ece.channel.t_e.data'], 180000, efit_tree='EFIT02')
    """
    if composer is None:
        composer = ImasComposer(
            efit_tree=efit_tree,
            efit_run_id=efit_run_id,
            profiles_tree=profiles_tree,
            profiles_run_id=profiles_run_id,
            fast_ece=fast_ece
        )

    raw_data = {}

    for _ in range(max_iterations):
        status, requirements = composer.resolve(ids_paths, shot, raw_data)

        if all(status.values()):
            break

        fetched = fetch_requirements(requirements)

        for key, value in fetched.items():
            if isinstance(value, Exception):
                raise RuntimeError(
                    f"Failed to fetch requirement {key}: {value}"
                ) from value

        raw_data.update(fetched)
    else:
        unresolved = [path for path, resolved in status.items() if not resolved]
        raise RuntimeError(
            f"Could not resolve {unresolved} within {max_iterations} iterations"
        )

    return composer.compose(ids_paths, shot, raw_data)
