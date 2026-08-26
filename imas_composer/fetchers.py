"""
Fetchers - Data retrieval utilities for imas_composer.

This module provides concrete fetching implementations for use with ImasComposer.
It is intentionally separate from composer.py so that ImasComposer itself has no
dependency on any specific data backend (MDSplus, ptdata, etc.).

Public API:
    fetch_requirements: Fetch a list of Requirement objects via toksearch_d3d
    simple_load: Convenience wrapper that runs the full resolve-fetch-compose loop
"""

from typing import Dict, List, Tuple, Any, Optional

from .core import Requirement
from .composer import ImasComposer

try:
    from toksearch_d3d.interfaces.req_interface import fetch_many_from_req
    TOKSEARCH_AVAILABLE = True
except ImportError:
    TOKSEARCH_AVAILABLE = False
    fetch_many_from_req = None


def fetch_requirements(
    requirements: List[Requirement],
) -> Dict[Tuple[str, int, str], Any]:
    """
    Fetch a list of requirements via toksearch_d3d.

    Requirements with treename == "__ptdata__" are batched into a single
    tree-less getMany() of ptdata2/dim_of/pthead2 TDI, returning a dict with
    keys 'data', 'times' (ms), and 'rarray'.  Everything else is batched per
    (treename, shot) and evaluated as TDI against an MDSplus tree.  Every group
    is tried first against the FDP thin client and then atlas.

    Args:
        requirements: List of Requirement objects to fetch.

    Returns:
        Dict mapping each requirement's as_key() tuple to its fetched value,
        or to the Exception if fetching failed.

    Raises:
        RuntimeError: If toksearch_d3d is not installed.
    """
    if not requirements:
        return {}

    if not TOKSEARCH_AVAILABLE:
        raise RuntimeError(
            "toksearch_d3d is required for fetching requirements but is not "
            "installed. Install it with: conda install -c ga-fdp toksearch_d3d"
        )

    unique_requirements = []
    seen_keys = set()
    for req in requirements:
        key = req.as_key()
        if key in seen_keys:
            continue
        seen_keys.add(key)
        unique_requirements.append(req)

    return fetch_many_from_req(unique_requirements)


def simple_load(
    ids_paths: List[str],
    shot: int,
    composer: Optional[ImasComposer] = None,
    efit_tree: str = "EFIT01",
    efit_run_id: str = "",
    profiles_tree: str = "ZIPFIT01",
    profiles_run_id: str = "",
    fast_ece: bool = False,
    include_rip: bool = False,
    crop_core_profiles: bool = False,
    max_iterations: int = 10,
) -> Dict[str, Any]:
    """
    Simple utility function to resolve and compose IDS data in one call.

    Runs the full resolve-fetch-compose loop, fetching through toksearch_d3d.

    Args:
        ids_paths: List of full IDS paths to compose (e.g., ['ece.channel.t_e.data'])
        shot: Shot number
        composer: Optional pre-configured ImasComposer instance. If None, creates new instance.
        efit_tree: EFIT tree (default: 'EFIT01', ignored if composer provided)
        efit_run_id: Run id appended to pulse for 'EFIT' tree (default: '')
        profiles_tree: Profiles tree (default: 'ZIPFIT01', ignored if composer provided)
        profiles_run_id: Run ID appended to pulse for OMFIT_PROFS tree (default: '')
        fast_ece: Whether to load fast_ece data (default: False)
        include_rip: Whether to include RIP data for interferometer IDS (default: False)
        crop_core_profiles: Whether to crop core_profiles to inside the separatrix (rho <= 1)
            (default: False, keeps scrape-off layer data)
        max_iterations: Maximum resolve-fetch iterations (default: 10)

    Returns:
        Dict mapping each ids_path -> composed IDS data

    Raises:
        RuntimeError: If toksearch_d3d is not installed, or if requirements
            cannot be resolved or any fetch fails.

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
            fast_ece=fast_ece,
            include_rip=include_rip,
            crop_core_profiles=crop_core_profiles
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
