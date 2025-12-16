"""
Gas Injection IDS Mapping for DIII-D

See OMAS: omas/machine_mappings/d3d.py::gas_injection_hardware

All data is static hardware geometry configuration, loaded from YAML.
"""

from typing import Dict
import numpy as np
import awkward as ak

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class GasInjectionMapper(IDSMapper):
    """Maps DIII-D gas injection hardware to IMAS gas_injection IDS."""

    # Configuration YAML file contains static values and field ledger
    DOCS_PATH = "gas_injection.yaml"
    CONFIG_PATH = "gas_injection.yaml"

    def __init__(self):
        """Initialize gas injection mapper."""
        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Load pipe and valve configurations from static_values
        config = self._load_config()
        self.pipes = config.get('static_values', {}).get('pipes', [])
        self.valves = config.get('static_values', {}).get('valves', [])

        # Build IDS specs
        self._build_specs()

    def _build_specs(self):
        """Build all IDS entry specifications"""

        # All fields are COMPUTED stage (no MDSplus data needed - all static)

        self.specs["gas_injection.pipe.name"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_pipe_name,
            ids_path="gas_injection.pipe.name",
            docs_file=self.DOCS_PATH
        )

        self.specs["gas_injection.pipe.valve_indices"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_pipe_valve_indices,
            ids_path="gas_injection.pipe.valve_indices",
            docs_file=self.DOCS_PATH
        )

        self.specs["gas_injection.pipe.exit_position.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_pipe_exit_r,
            ids_path="gas_injection.pipe.exit_position.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["gas_injection.pipe.exit_position.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_pipe_exit_z,
            ids_path="gas_injection.pipe.exit_position.z",
            docs_file=self.DOCS_PATH
        )

        self.specs["gas_injection.pipe.exit_position.phi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_pipe_exit_phi,
            ids_path="gas_injection.pipe.exit_position.phi",
            docs_file=self.DOCS_PATH
        )

        self.specs["gas_injection.pipe.second_point.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_pipe_second_r,
            ids_path="gas_injection.pipe.second_point.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["gas_injection.pipe.second_point.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_pipe_second_z,
            ids_path="gas_injection.pipe.second_point.z",
            docs_file=self.DOCS_PATH
        )

        self.specs["gas_injection.pipe.second_point.phi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_pipe_second_phi,
            ids_path="gas_injection.pipe.second_point.phi",
            docs_file=self.DOCS_PATH
        )

        self.specs["gas_injection.valve.identifier"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_valve_identifier,
            ids_path="gas_injection.valve.identifier",
            docs_file=self.DOCS_PATH
        )

        self.specs["gas_injection.valve.pipe_indices"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_valve_pipe_indices,
            ids_path="gas_injection.valve.pipe_indices",
            docs_file=self.DOCS_PATH
        )

    # Compose functions - all extract data from self.pipes and self.valves

    def _compose_pipe_name(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose pipe names array."""
        return np.array([pipe['name'] for pipe in self.pipes])

    def _compose_pipe_valve_indices(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose pipe valve_indices arrays (ragged)."""
        return ak.Array([pipe['valve_indices'] for pipe in self.pipes])

    def _compose_pipe_exit_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose pipe exit position R coordinates."""
        return np.array([pipe['exit_position']['r'] for pipe in self.pipes])

    def _compose_pipe_exit_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose pipe exit position Z coordinates."""
        return np.array([pipe['exit_position']['z'] for pipe in self.pipes])

    def _compose_pipe_exit_phi(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose pipe exit position phi coordinates."""
        return np.array([pipe['exit_position']['phi'] for pipe in self.pipes])

    def _compose_pipe_second_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose pipe second_point R coordinates."""
        return np.array([pipe['second_point']['r'] for pipe in self.pipes])

    def _compose_pipe_second_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose pipe second_point Z coordinates."""
        return np.array([pipe['second_point']['z'] for pipe in self.pipes])

    def _compose_pipe_second_phi(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose pipe second_point phi coordinates."""
        return np.array([pipe['second_point']['phi'] for pipe in self.pipes])

    def _compose_valve_identifier(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose valve identifiers."""
        return np.array([valve['identifier'] for valve in self.valves])

    def _compose_valve_pipe_indices(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose valve pipe_indices arrays (ragged)."""
        return ak.Array([valve['pipe_indices'] for valve in self.valves])

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
