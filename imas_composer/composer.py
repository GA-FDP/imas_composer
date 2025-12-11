"""
ImasComposer - Public API for composing IMAS data from MDSplus sources.

This is the main interface for external applications to use imas_composer.
"""

from typing import Dict, List, Tuple, Any, Optional
from .core import Requirement, RequirementStage
from .ids.ece import ElectronCyclotronEmissionMapper
from .ids.thomson_scattering import ThomsonScatteringMapper
from .ids.equilibrium import EquilibriumMapper
from .ids.ec_launchers import ECLaunchersMapper


class ImasComposer:
    """
    Main interface for composing IMAS data.

    This class uses a batch-first API that processes multiple IDS paths efficiently.

    Usage:
        composer = ImasComposer()

        # Iteratively resolve requirements for multiple paths
        ids_paths = ['ece.channel.t_e.data', 'ece.channel.time']
        raw_data = {}
        while True:
            status, requirements = composer.resolve(ids_paths, 180000, raw_data)
            if all(status.values()):
                break
            # Fetch requirements from MDSplus/toksearch
            for req in requirements:
                raw_data[req.as_key()] = fetch_from_mds(req)

        # Compose final data for all paths at once
        results = composer.compose(ids_paths, 180000, raw_data)
        # results is a dict: {'ece.channel.t_e.data': array(...), 'ece.channel.time': array(...)}
    """

    def __init__(self, device: str = 'd3d', efit_tree: str = 'EFIT01'):
        """
        Initialize ImasComposer.

        Args:
            device: Device identifier (currently only 'd3d' supported)
            efit_tree: EFIT tree to use for equilibrium data (e.g., 'EFIT01', 'EFIT02')
        """
        self.device = device
        self.efit_tree = efit_tree
        self._mappers = {}

        # Register available mappers
        self._register_mapper('ece', ElectronCyclotronEmissionMapper(fast_ece=False))
        self._register_mapper('thomson_scattering', ThomsonScatteringMapper())
        self._register_mapper('equilibrium', EquilibriumMapper(efit_tree=efit_tree))
        self._register_mapper('ec_launchers', ECLaunchersMapper())

    def _register_mapper(self, ids_name: str, mapper):
        """Register an IDS mapper."""
        self._mappers[ids_name] = mapper

    def _get_mapper_for_path(self, ids_path: str):
        """
        Get the appropriate mapper for an IDS path.

        Args:
            ids_path: Full IDS path like 'ece.channel.t_e.data'

        Returns:
            Tuple of (mapper, ids_name)

        Raises:
            ValueError: If no mapper found for path
        """
        # Extract IDS name from path (first component)
        ids_name = ids_path.split('.')[0]

        if ids_name not in self._mappers:
            raise ValueError(
                f"No mapper registered for IDS '{ids_name}'. "
                f"Available: {list(self._mappers.keys())}"
            )

        return self._mappers[ids_name], ids_name

    def _group_paths_by_ids(self, ids_paths: List[str]) -> Dict[str, List[str]]:
        """
        Group IDS paths by their root IDS name.

        Args:
            ids_paths: List of full IDS paths

        Returns:
            Dict mapping ids_name -> list of paths for that IDS

        Example:
            >>> paths = ['ece.channel.t_e.data', 'equilibrium.time', 'ece.channel.time']
            >>> grouped = composer._group_paths_by_ids(paths)
            >>> grouped
            {'ece': ['ece.channel.t_e.data', 'ece.channel.time'],
             'equilibrium': ['equilibrium.time']}
        """
        paths_by_ids = {}
        for ids_path in ids_paths:
            ids_name = ids_path.split('.')[0]
            if ids_name not in paths_by_ids:
                paths_by_ids[ids_name] = []
            paths_by_ids[ids_name].append(ids_path)
        return paths_by_ids

    def resolve(
        self,
        ids_paths: List[str],
        shot: int,
        raw_data: Dict[str, Any]
    ) -> Tuple[Dict[str, bool], List[Requirement]]:
        """
        Resolve requirements for multiple IDS paths at once.

        This method determines what MDSplus data is needed to synthesize the requested
        IDS fields. It should be called iteratively, fetching requirements and updating
        raw_data, until all paths are fully resolved.

        Args:
            ids_paths: List of full IDS paths (e.g., ['ece.channel.t_e.data', 'ece.channel.time'])
            shot: Shot number
            raw_data: Dict of already-fetched data (requirement keys -> values)

        Returns:
            Tuple of (resolution_status, requirements):
                - resolution_status: Dict mapping path -> fully_resolved boolean
                - requirements: Deduplicated list of all requirements needed

        Example:
            >>> composer = ImasComposer()
            >>> raw_data = {}
            >>> status, reqs = composer.resolve(['ece.channel.t_e.data', 'ece.channel.time'], 180000, raw_data)
            >>> print(status)
            {'ece.channel.t_e.data': False, 'ece.channel.time': False}
            >>> print(len(reqs))  # Deduplicated requirements for both fields
        """
        # Group paths by IDS name for efficient batching
        paths_by_ids = self._group_paths_by_ids(ids_paths)

        # Collect requirements for all paths, grouped by IDS
        all_requirements = []
        seen_keys = set()
        resolution_status = {}

        for ids_name, paths in paths_by_ids.items():
            if ids_name not in self._mappers:
                raise ValueError(
                    f"No mapper registered for IDS '{ids_name}'. "
                    f"Available: {list(self._mappers.keys())}"
                )

            mapper = self._mappers[ids_name]

            # Validate all paths exist in mapper
            for ids_path in paths:
                if ids_path not in mapper.specs:
                    raise ValueError(
                        f"IDS path '{ids_path}' not found in {ids_name} mapper. "
                        f"Available: {list(mapper.specs.keys())}"
                    )

            # Batch collect requirements for all paths in this IDS
            ids_requirements = self._collect_requirements_batch(mapper, paths, shot, raw_data)

            # Deduplicate and track resolution status
            for ids_path, path_requirements in ids_requirements.items():
                # Filter out requirements we already have
                missing_requirements = [
                    req for req in path_requirements
                    if req.as_key() not in raw_data
                ]

                # Track resolution status
                resolution_status[ids_path] = len(missing_requirements) == 0

                # Add to deduplicated list
                for req in missing_requirements:
                    key = req.as_key()
                    if key not in seen_keys:
                        all_requirements.append(req)
                        seen_keys.add(key)

        return resolution_status, all_requirements

    def _collect_requirements_batch(
        self,
        mapper,
        ids_paths: List[str],
        shot: int,
        raw_data: Dict[str, Any]
    ) -> Dict[str, List[Requirement]]:
        """
        Collect requirements for multiple IDS paths at once, sharing dependency traversal.

        This performs multi-pass resolution for all paths simultaneously:
        1. Collect DIRECT requirements for all paths
        2. Use fetched data to derive DERIVED requirements
        3. Repeat until all requirements collected

        Args:
            mapper: IDS mapper instance
            ids_paths: List of IDS paths to collect requirements for
            shot: Shot number
            raw_data: Already-fetched data

        Returns:
            Dict mapping each ids_path -> List[Requirement]
        """
        # Track requirements per path
        requirements_by_path = {path: [] for path in ids_paths}

        # Shared visited set across all paths to avoid redundant traversal
        visited = set()

        # Process all paths together
        to_process = [(path, path, 0) for path in ids_paths]  # (original_path, current_path, depth)
        max_depth = 10

        while to_process:
            original_path, current_path, depth = to_process.pop(0)

            if depth > max_depth:
                raise RuntimeError(f"Max dependency depth exceeded for {current_path}")

            # Use a combined key for visiting to share across all paths
            visit_key = current_path
            if visit_key in visited:
                continue
            visited.add(visit_key)

            if current_path not in mapper.specs:
                continue

            spec = mapper.specs[current_path]

            # Add dependencies to process list (propagate original_path)
            if spec.depends_on:
                for dep in spec.depends_on:
                    to_process.append((original_path, dep, depth + 1))

            # Collect requirements based on stage
            if spec.stage == RequirementStage.DIRECT:
                for req in spec.static_requirements:
                    requirements_by_path[original_path].append(
                        Requirement(req.mds_path, shot, req.treename)
                    )

            elif spec.stage == RequirementStage.DERIVED:
                if spec.derive_requirements:
                    # Try to derive requirements if we have the dependency data
                    try:
                        derived_reqs = spec.derive_requirements(shot, raw_data)
                        requirements_by_path[original_path].extend(derived_reqs)
                    except (KeyError, Exception):
                        # Dependencies not yet available, will be resolved in next pass
                        pass

        return requirements_by_path

    def compose(
        self,
        ids_paths: List[str],
        shot: int,
        raw_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compose (synthesize) final IDS data from raw MDSplus data for multiple paths.

        This should only be called after resolve() returns all paths as fully_resolved=True.

        Args:
            ids_paths: List of full IDS paths (e.g., ['ece.channel.t_e.data', 'ece.channel.time'])
            shot: Shot number
            raw_data: Dict of fetched data (requirement keys -> values)

        Returns:
            Dict mapping each ids_path -> synthesized IDS data

        Raises:
            ValueError: If path not found or requirements not met
            RuntimeError: If called before all requirements are resolved

        Example:
            >>> composer = ImasComposer()
            >>> # ... resolve and fetch all requirements ...
            >>> results = composer.compose(['ece.channel.t_e.data', 'ece.channel.time'], 180000, raw_data)
            >>> print(results['ece.channel.t_e.data'].shape)  # (n_channels, n_time)
        """
        results = {}

        # Group paths by IDS for efficient processing
        paths_by_ids = self._group_paths_by_ids(ids_paths)

        # Compose all paths
        for ids_name, paths in paths_by_ids.items():
            if ids_name not in self._mappers:
                raise ValueError(
                    f"No mapper registered for IDS '{ids_name}'. "
                    f"Available: {list(self._mappers.keys())}"
                )

            mapper = self._mappers[ids_name]

            for ids_path in paths:
                # Check if spec exists
                if ids_path not in mapper.specs:
                    raise ValueError(
                        f"IDS path '{ids_path}' not found in {ids_name} mapper"
                    )

                spec = mapper.specs[ids_path]

                # Verify it's a COMPUTED stage
                if spec.stage != RequirementStage.COMPUTED:
                    raise ValueError(
                        f"Cannot compose '{ids_path}' - it is {spec.stage}, not COMPUTED. "
                        f"Only COMPUTED stage fields can be composed."
                    )

                # Verify it has a compose function
                if not spec.compose:
                    raise ValueError(
                        f"Cannot compose '{ids_path}' - no compose function defined"
                    )

                # Compose the data
                try:
                    results[ids_path] = spec.compose(shot, raw_data)
                except KeyError as e:
                    raise RuntimeError(
                        f"Missing required data for composing '{ids_path}': {e}. "
                        f"Did you call resolve() and fetch all requirements?"
                    ) from e

        return results

    def get_supported_fields(self, ids_name: str) -> List[str]:
        """
        Get list of supported fields for an IDS.

        Args:
            ids_name: IDS identifier (e.g., 'ece', 'thomson_scattering')

        Returns:
            List of full IDS paths that can be composed

        Example:
            >>> composer = ImasComposer()
            >>> fields = composer.get_supported_fields('ece')
            >>> print(fields[:3])
            ['ece.ids_properties.homogeneous_time', 'ece.channel.name', ...]
        """
        if ids_name not in self._mappers:
            raise ValueError(f"No mapper for '{ids_name}'")

        mapper = self._mappers[ids_name]

        # Return only COMPUTED fields (user-facing)
        return [
            path for path, spec in mapper.specs.items()
            if spec.stage == RequirementStage.COMPUTED
        ]
