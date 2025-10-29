"""
Base class for IDS mappers.

Provides common functionality shared across all IDS implementations.
"""

from typing import Dict, List
from pathlib import Path
import yaml


class IDSMapper:
    """Base class for all IDS mappers."""

    # Subclasses should override these
    CONFIG_PATH: str = None  # e.g., "ece.yaml"
    DOCS_PATH: str = None    # e.g., "electron_cyclotron_emission.yaml"

    def __init__(self):
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
