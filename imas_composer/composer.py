"""
ImasComposer - Public API for composing IMAS data from MDSplus sources.

This is the main interface for external applications to use imas_composer.
"""

from typing import Dict, List, Tuple, Any, Optional
from .core import Requirement, RequirementStage
from .ids.ece import ElectronCyclotronEmissionMapper
from .ids.thomson_scattering import ThomsonScatteringMapper
from .ids.equilibrium import EquilibriumMapper
from .ids.core_profiles import CoreProfilesMapper
from .ids.ec_launchers import ECLaunchersMapper

class ImasComposer:
    """
    Main interface for composing IMAS data.

    Usage:
        composer = ImasComposer()

        # Iteratively resolve requirements
        raw_data = {}
        while True:
            fully_resolved, requirements = composer.resolve('ece.channel.t_e.data', 180000, raw_data)
            if fully_resolved:
                break
            # Fetch requirements from MDSplus/toksearch
            for req in requirements:
                raw_data[req.as_key()] = fetch_from_mds(req)

        # Compose final data
        result = composer.compose('ece.channel.t_e.data', 180000, raw_data)
    """

    def __init__(self, device: str = 'd3d', efit_tree: str = 'EFIT01', profiles_tree: str = 'ZIPFIT01'):
        """
        Initialize ImasComposer.

        Args:
            device: Device identifier (currently only 'd3d' supported)
            efit_tree: EFIT tree to use for equilibrium data (e.g., 'EFIT01', 'EFIT02')
            profiles_tree: Profiles tree to use for core_profiles data (e.g., 'ZIPFIT01', 'OMFIT_PROFS')
        """
        self.device = device
        self.efit_tree = efit_tree
        self.profiles_tree = profiles_tree
        self._mappers = {}

        # Register available mappers
        self._register_mapper('ece', ElectronCyclotronEmissionMapper(fast_ece=False))
        self._register_mapper('thomson_scattering', ThomsonScatteringMapper())
        self._register_mapper('equilibrium', EquilibriumMapper(efit_tree=efit_tree))
        self._register_mapper('core_profiles', CoreProfilesMapper(profiles_tree=profiles_tree))
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

    def resolve(
        self,
        ids_path: str,
        shot: int,
        raw_data: Dict[str, Any]
    ) -> Tuple[bool, List[Requirement]]:
        """
        Resolve requirements for an IDS path.

        This method determines what MDSplus data is needed to synthesize the requested
        IDS field. It should be called iteratively, fetching requirements and updating
        raw_data, until fully_resolved returns True.

        Args:
            ids_path: Full IDS path (e.g., 'ece.channel.t_e.data')
            shot: Shot number
            raw_data: Dict of already-fetched data (requirement keys -> values)

        Returns:
            Tuple of (fully_resolved, requirements):
                - fully_resolved: True if all dependencies are satisfied
                - requirements: List of Requirement objects still needed

        Example:
            >>> composer = ImasComposer()
            >>> raw_data = {}
            >>> resolved, reqs = composer.resolve('ece.channel.t_e.data', 180000, raw_data)
            >>> print(resolved)  # False - need to fetch requirements
            >>> print(len(reqs))  # N requirements needed
        """
        mapper, ids_name = self._get_mapper_for_path(ids_path)

        # Check if spec exists
        if ids_path not in mapper.specs:
            raise ValueError(
                f"IDS path '{ids_path}' not found in {ids_name} mapper. "
                f"Available: {list(mapper.specs.keys())}"
            )

        # Collect all requirements needed for this path
        # Use _collect_requirements_with_data to handle DERIVED stages
        all_requirements = self._collect_requirements_with_data(mapper, ids_path, shot, raw_data)

        # Filter out requirements we already have
        missing_requirements = [
            req for req in all_requirements
            if req.as_key() not in raw_data
        ]

        # Fully resolved if no missing requirements
        fully_resolved = len(missing_requirements) == 0

        return fully_resolved, missing_requirements

    def _collect_requirements_with_data(
        self,
        mapper,
        ids_path: str,
        shot: int,
        raw_data: Dict[str, Any]
    ) -> List[Requirement]:
        """
        Collect requirements with access to raw_data for DERIVED stages.

        This performs multi-pass resolution:
        1. Collect DIRECT requirements
        2. Use fetched data to derive DERIVED requirements
        3. Repeat until all requirements collected
        """
        all_requirements = []
        visited = set()

        # Start with the target path
        to_process = [(ids_path, 0)]  # (path, depth)
        max_depth = 10  # Prevent infinite loops

        while to_process:
            current_path, depth = to_process.pop(0)

            if depth > max_depth:
                raise RuntimeError(f"Max dependency depth exceeded for {current_path}")

            if current_path in visited:
                continue
            visited.add(current_path)

            if current_path not in mapper.specs:
                continue

            spec = mapper.specs[current_path]

            # Add dependencies to process list
            if spec.depends_on:
                for dep in spec.depends_on:
                    to_process.append((dep, depth + 1))

            # Collect requirements based on stage
            if spec.stage == RequirementStage.DIRECT:
                for req in spec.static_requirements:
                    all_requirements.append(Requirement(req.mds_path, shot, req.treename))

            elif spec.stage == RequirementStage.DERIVED:
                if spec.derive_requirements:
                    # Try to derive requirements if we have the dependency data
                    try:
                        derived_reqs = spec.derive_requirements(shot, raw_data)
                        all_requirements.extend(derived_reqs)
                    except (KeyError, Exception):
                        # Dependencies not yet available, will be resolved in next pass
                        pass

        return all_requirements

    def compose(
        self,
        ids_path: str,
        shot: int,
        raw_data: Dict[str, Any]
    ) -> Any:
        """
        Compose (synthesize) the final IDS data from raw MDSplus data.

        This should only be called after resolve() returns fully_resolved=True.

        Args:
            ids_path: Full IDS path (e.g., 'ece.channel.t_e.data')
            shot: Shot number
            raw_data: Dict of fetched data (requirement keys -> values)

        Returns:
            Synthesized IDS data (type depends on field - could be array, scalar, etc.)

        Raises:
            ValueError: If path not found or requirements not met
            RuntimeError: If called before all requirements are resolved

        Example:
            >>> composer = ImasComposer()
            >>> # ... resolve and fetch all requirements ...
            >>> t_e_data = composer.compose('ece.channel.t_e.data', 180000, raw_data)
            >>> print(t_e_data.shape)  # (n_channels, n_time)
        """
        mapper, ids_name = self._get_mapper_for_path(ids_path)

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
            result = spec.compose(shot, raw_data)
        except KeyError as e:
            raise RuntimeError(
                f"Missing required data for composing '{ids_path}': {e}. "
                f"Did you call resolve() and fetch all requirements?"
            ) from e

        return result

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

    def resolve_multiple(
        self,
        ids_paths: List[str],
        shot: int,
        raw_data: Dict[str, Any]
    ) -> Tuple[Dict[str, bool], List[Requirement]]:
        """
        Resolve requirements for multiple IDS paths at once.

        Args:
            ids_paths: List of IDS paths to resolve
            shot: Shot number
            raw_data: Dict of already-fetched data

        Returns:
            Tuple of (resolution_status, requirements):
                - resolution_status: Dict mapping path -> fully_resolved boolean
                - requirements: Deduplicated list of all requirements needed

        Example:
            >>> paths = ['ece.channel.t_e.data', 'ece.channel.time']
            >>> status, reqs = composer.resolve_multiple(paths, 180000, {})
            >>> print(status)
            {'ece.channel.t_e.data': False, 'ece.channel.time': False}
            >>> print(len(reqs))  # Deduplicated requirements for both fields
        """
        resolution_status = {}
        all_requirements = []
        seen_keys = set()

        for ids_path in ids_paths:
            fully_resolved, requirements = self.resolve(ids_path, shot, raw_data)
            resolution_status[ids_path] = fully_resolved

            # Deduplicate requirements
            for req in requirements:
                key = req.as_key()
                if key not in seen_keys:
                    all_requirements.append(req)
                    seen_keys.add(key)

        return resolution_status, all_requirements
