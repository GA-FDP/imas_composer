"""
Base class for IDS mappers and factory pattern.

Provides common functionality shared across all IDS implementations.

IMPORTANT: All user-facing IDS fields must be RequirementStage.COMPUTED
-------------------------------------------------------------------
The ImasComposer.compose() method requires all fields to be COMPUTED stage.
Only internal dependency fields (conventionally prefixed with _) should be DIRECT.

Pattern for simple pass-through fields:
    # Internal dependency
    self.specs["ids._internal_field"] = IDSEntrySpec(
        stage=RequirementStage.DIRECT,
        static_requirements=[Requirement('path.to.data', 0, tree)],
        ids_path="ids._internal_field",
        docs_file=self.DOCS_PATH
    )

    # User-facing field (COMPUTED with passthrough lambda)
    self.specs["ids.user.field"] = IDSEntrySpec(
        stage=RequirementStage.COMPUTED,
        depends_on=["ids._internal_field"],
        compose=lambda shot, raw: raw[Requirement('path.to.data', shot, tree).as_key()],
        ids_path="ids.user.field",
        docs_file=self.DOCS_PATH
    )

Factory Pattern:
----------------
IDS mappers should use a factory function to create mapper instances.
This allows for tree-specific or configuration-specific mapper implementations.

Example:
    def create_equilibrium_mapper(efit_tree: str = 'EFIT01') -> IDSMapper:
        return EquilibriumMapper(efit_tree=efit_tree)
"""

from typing import Dict, List
from pathlib import Path
import yaml


class IDSMapper:
    """Base class for all IDS mappers."""

    # Subclasses should override these
    CONFIG_PATH: str = None  # e.g., "ece.yaml"
    DOCS_PATH: str = None    # e.g., "electron_cyclotron_emission.yaml"

    def __init__(self, **kwargs):
        """Initialize the IDS mapper."""
        # Load configuration (static values and field list)
        config = self._load_config()
        self.static_values = config.get('static_values', {})
        self.supported_fields = config.get('fields', [])

        # Subclasses should initialize self.specs
        self.specs: Dict = {}

    def _load_config(self) -> Dict:
        """
        Load IDS configuration from YAML file.

        Returns:
            Dict with 'static_values' and 'fields' keys
        """
        if not self.CONFIG_PATH:
            return {}

        yaml_path = Path(__file__).parent / self.CONFIG_PATH
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}

    def get_supported_fields(self) -> List[str]:
        """
        Get list of all supported IDS fields.

        Returns:
            List of IMAS schema paths (e.g., ['channel.name', 'channel.t_e.data'])
        """
        return self.supported_fields
