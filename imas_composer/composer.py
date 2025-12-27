"""
ImasComposer - Public API for composing IMAS data from MDSplus sources.

This is the main interface for external applications to use imas_composer.
"""

from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import yaml
from .core import Requirement, RequirementStage
from .ids.ids_factory import IDSFactory

# Optional OMAS import for simple_add utility
try:
    from omas import mdsvalue
    OMAS_AVAILABLE = True
except ImportError:
    OMAS_AVAILABLE = False
    mdsvalue = None



def _load_default_from_yaml(yaml_filename: str, key: str, fallback: Any) -> Any:
    """
    Load a default value from an IDS YAML configuration file.

    Args:
        yaml_filename: Name of the YAML file (e.g., 'equilibrium.yaml')
        key: Configuration key to retrieve (e.g., 'default_efit_tree')
        fallback: Fallback value if key not found

    Returns:
        Value from YAML config, or fallback if not found
    """
    yaml_path = Path(__file__).parent / 'ids' / yaml_filename
    if yaml_path.exists():
        try:
            with open(yaml_path, 'r') as f:
                config = yaml.safe_load(f) or {}
                return config.get(key, fallback)
        except Exception:
            return fallback
    return fallback

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

    def __init__(self,
                 efit_tree: str = "EFIT01",
                 efit_run_id: str = "01",
                 profiles_tree: str = "ZIPFIT01",
                 profiles_run_id: str = "001",
                 fast_ece:bool = False):
        """
        Initialize ImasComposer.

        Args:
            device: Device identifier (currently only 'd3d' supported)
            efit_tree: EFIT tree to use for equilibrium data (e.g., 'EFIT01', 'EFIT02').
            efit_run_id: Run id to append to pulse for 'EFIT' tree.
            profiles_tree: Profiles tree to use for core_profiles data (e.g., 'ZIPFIT01', 'OMFIT_PROFS').
            profiles_run_id: Run ID to append to pulse for OMFIT_PROFS tree.
            fast_ece: Whether to load fast_ece data, defaults to false.
        """
        self.efit_tree = efit_tree
        self.efit_run_id = efit_run_id
        self.profiles_tree = profiles_tree
        self.profiles_run_id = profiles_run_id
        self.fast_ece = fast_ece
        self.ids_factory = IDSFactory()
        self._mappers = {}
        for ids_name in self.ids_factory.list_ids():
            # Register available mappers using factory functions
            self._register_mapper(ids_name, self.ids_factory(ids_name, efit_tree=efit_tree, 
                                                             efit_run_id=efit_run_id, 
                                                             profiles_tree=self.profiles_tree,
                                                             profiles_run_id=self.profiles_run_id,
                                                             fast_ece=self.fast_ece))
            
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

    def _fetch_requirements(self, requirements: List[Requirement]) -> Dict[Tuple[str, int, str], Any]:
        """
        Fetch multiple requirements from MDSplus in a single query.

        This is a private method that requires OMAS to be installed.

        Args:
            requirements: List of Requirement objects to fetch

        Returns:
            Dict mapping requirement keys to fetched data

        Raises:
            RuntimeError: If OMAS is not available
        """
        if not OMAS_AVAILABLE:
            raise RuntimeError(
                "OMAS is required for _fetch_requirements but is not installed. "
                "Install OMAS or use your own fetching mechanism."
            )

        if not requirements:
            return {}

        # Group by (treename, shot)
        by_tree_shot = {}
        for req in requirements:
            key = (req.treename, req.shot)
            if key not in by_tree_shot:
                by_tree_shot[key] = []
            by_tree_shot[key].append(req)

        # Fetch each tree/shot combination separately
        raw_data = {}
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
    efit_run_id: str = "01",
    profiles_tree: str = "ZIPFIT01",
    profiles_run_id: str = "001",
    fast_ece: bool = False,
    max_iterations: int = 10
) -> Dict[str, Any]:
    """
    Simple utility function to resolve and compose IDS data in one call.

    This function performs the resolve-fetch loop automatically, similar to
    the benchmark_batch_resolve_compose pattern. It requires OMAS to be installed
    for fetching data from MDSplus.

    Args:
        ids_paths: List of full IDS paths to compose (e.g., ['ece.channel.t_e.data'])
        shot: Shot number
        composer: Optional pre-configured ImasComposer instance. If None, creates new instance.
        efit_tree: EFIT tree to use for equilibrium data (default: 'EFIT01', ignored if composer provided)
        efit_run_id: Run id to append to pulse for 'EFIT' tree (default: '01', ignored if composer provided)
        profiles_tree: Profiles tree to use for core_profiles data (default: 'ZIPFIT01', ignored if composer provided)
        profiles_run_id: Run ID to append to pulse for OMFIT_PROFS tree (default: '001', ignored if composer provided)
        fast_ece: Whether to load fast_ece data (default: False, ignored if composer provided)
        max_iterations: Maximum number of resolve-fetch iterations (default: 10)

    Returns:
        Dict mapping each ids_path -> composed IDS data

    Raises:
        RuntimeError: If OMAS is not available, or if requirements cannot be resolved
        Exception: If any requirement fetch fails

    Example:
        >>> # Simple single field
        >>> result = simple_load(['equilibrium.time'], 200000)
        >>> print(result['equilibrium.time'])

        >>> # Multiple fields with custom EFIT tree
        >>> result = simple_load(
        ...     ['ece.channel.t_e.data', 'ece.channel.time'],
        ...     180000,
        ...     efit_tree='EFIT02'
        ... )
        >>> print(result['ece.channel.t_e.data'].shape)

        >>> # With pre-configured composer
        >>> composer = ImasComposer(efit_tree='EFIT02')
        >>> result = simple_load(['equilibrium.time'], 200000, composer=composer)
    """
    if not OMAS_AVAILABLE:
        raise RuntimeError(
            "OMAS is required for simple_load but is not installed. "
            "Please install OMAS or use the composer.resolve()/compose() API "
            "with your own fetching mechanism."
        )

    # Use provided composer or create new instance
    if composer is None:
        composer = ImasComposer(
            efit_tree=efit_tree,
            efit_run_id=efit_run_id,
            profiles_tree=profiles_tree,
            profiles_run_id=profiles_run_id,
            fast_ece=fast_ece
        )

    raw_data = {}

    # Resolve-fetch loop
    for iteration in range(max_iterations):
        status, requirements = composer.resolve(ids_paths, shot, raw_data)

        # Check if all paths are resolved
        if all(status.values()):
            break

        # Fetch requirements using the private method
        fetched = composer._fetch_requirements(requirements)

        # Check for fetch errors
        for key, value in fetched.items():
            if isinstance(value, Exception):
                raise RuntimeError(
                    f"Failed to fetch requirement {key}: {value}"
                ) from value

        # Update raw_data
        raw_data.update(fetched)
    else:
        # Loop completed without breaking (not all resolved)
        unresolved = [path for path, resolved in status.items() if not resolved]
        raise RuntimeError(
            f"Could not resolve {unresolved} within {max_iterations} iterations"
        )

    # Compose final data
    results = composer.compose(ids_paths, shot, raw_data)
    return results
