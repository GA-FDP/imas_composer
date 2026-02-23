"""
Magnetics IDS Mapping for DIII-D

Maps plasma current and diamagnetic flux measurements to IMAS magnetics IDS.
See OMAS: omas/machine_mappings/d3d.py::ip_bt_dflux_data
"""

from typing import Dict, List
import numpy as np

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class MagneticsMapper(IDSMapper):
    """Maps DIII-D magnetics data to IMAS magnetics IDS."""

    CONFIG_PATH = "magnetics.yaml"

    def __init__(self, **kwargs):
        """Initialize Magnetics mapper."""
        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Build IDS specs
        self._build_specs()

    def _build_specs(self):
        """Build all IDS entry specifications"""

        # Plasma current (Ip) - auxiliary nodes
        self.specs["magnetics._ip_data"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_ip_data_requirements,
            ids_path="magnetics._ip_data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics._ip_time"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_ip_time_requirements,
            ids_path="magnetics._ip_time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics._ip_header"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_ip_header_requirements,
            ids_path="magnetics._ip_header",
            docs_file=self.CONFIG_PATH
        )

        # Diamagnetic flux - auxiliary nodes
        self.specs["magnetics._diamag_data"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_diamag_data_requirements,
            ids_path="magnetics._diamag_data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics._diamag_time"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_diamag_time_requirements,
            ids_path="magnetics._diamag_time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics._diamag_header"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_diamag_header_requirements,
            ids_path="magnetics._diamag_header",
            docs_file=self.CONFIG_PATH
        )

        # User-facing fields - COMPUTED stage
        # Note: ip and diamagnetic_flux are arrays (dimension 0 is measurement index)
        self.specs["magnetics.ip.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._ip_data"],
            compose=self._compose_ip_data,
            ids_path="magnetics.ip.data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.ip.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._ip_time"],
            compose=self._compose_ip_time,
            ids_path="magnetics.ip.time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.ip.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._ip_data", "magnetics._ip_header"],
            compose=self._compose_ip_data_error_upper,
            ids_path="magnetics.ip.data_error_upper",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.diamagnetic_flux.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._diamag_data"],
            compose=self._compose_diamagnetic_flux_data,
            ids_path="magnetics.diamagnetic_flux.data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.diamagnetic_flux.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._diamag_time"],
            compose=self._compose_diamagnetic_flux_time,
            ids_path="magnetics.diamagnetic_flux.time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.diamagnetic_flux.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._diamag_data", "magnetics._diamag_header"],
            compose=self._compose_diamagnetic_flux_data_error_upper,
            ids_path="magnetics.diamagnetic_flux.data_error_upper",
            docs_file=self.CONFIG_PATH
        )

    # Requirement derivation functions
    def _derive_ip_data_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for IP (data, time, and header bundled under __ptdata__ key)."""
        return [Requirement("IP", shot, "__ptdata__")]

    def _derive_ip_time_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for IP time (same key as data — deduplication handles it)."""
        return [Requirement("IP", shot, "__ptdata__")]

    def _derive_ip_header_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for IP header (same key as data — deduplication handles it)."""
        return [Requirement("IP", shot, "__ptdata__")]

    def _derive_diamag_data_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for DIAMAG3 (data, time, and header bundled under __ptdata__ key)."""
        return [Requirement("DIAMAG3", shot, "__ptdata__")]

    def _derive_diamag_time_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for DIAMAG3 time (same key as data — deduplication handles it)."""
        return [Requirement("DIAMAG3", shot, "__ptdata__")]

    def _derive_diamag_header_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for DIAMAG3 header (same key as data — deduplication handles it)."""
        return [Requirement("DIAMAG3", shot, "__ptdata__")]

    # Compose functions
    def _compose_ip_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get plasma current data.

        Returns array of shape (n_measurements, n_time).
        For DIII-D: (1, n_time) - single Ip measurement.
        """
        key = Requirement("IP", shot, "__ptdata__").as_key()
        # Add measurement dimension: (n_time,) -> (1, n_time)
        return raw_data[key]['data'][np.newaxis, :]

    def _compose_ip_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get plasma current time base with unit conversion.

        From OMAS: dim_of(...,0)/1000. (convert ms to s)
        Returns array of shape (n_measurements, n_time).
        For DIII-D: (1, n_time) - single Ip measurement.
        """
        key = Requirement("IP", shot, "__ptdata__").as_key()
        # Add measurement dimension: (n_time,) -> (1, n_time)
        return (raw_data[key]['times'] / 1000.0)[np.newaxis, :]

    def _compose_ip_data_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compute uncertainty for plasma current.

        From OMAS: abs(header[3] * header[4]) * ones(nt) * 10.0
        where header rarray comes from PtDataHeader.
        Returns array of shape (n_measurements, n_time).
        For DIII-D: (1, n_time) - single Ip measurement.
        """
        key = Requirement("IP", shot, "__ptdata__").as_key()
        nt = len(raw_data[key]['data'])
        header = raw_data[key]['rarray']

        # OMAS formula: abs(header[3] * header[4]) * ones(nt) * 10.0
        # Add measurement dimension: (n_time,) -> (1, n_time)
        return (np.abs(header[3] * header[4]) * np.ones(nt) * 10.0)[np.newaxis, :]

    def _compose_diamagnetic_flux_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get diamagnetic flux data with unit conversion.

        From OMAS: data * 1e-3 (convert to Weber)
        Returns array of shape (n_measurements, n_time).
        For DIII-D: (1, n_time) - single diamagnetic flux measurement.
        """
        key = Requirement("DIAMAG3", shot, "__ptdata__").as_key()
        # Add measurement dimension: (n_time,) -> (1, n_time)
        return (raw_data[key]['data'] * 1e-3)[np.newaxis, :]

    def _compose_diamagnetic_flux_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get diamagnetic flux time base with unit conversion.

        From OMAS: dim_of(...,0)/1000. (convert ms to s)
        Returns array of shape (n_measurements, n_time).
        For DIII-D: (1, n_time) - single diamagnetic flux measurement.
        """
        key = Requirement("DIAMAG3", shot, "__ptdata__").as_key()
        # Add measurement dimension: (n_time,) -> (1, n_time)
        return (raw_data[key]['times'] / 1000.0)[np.newaxis, :]

    def _compose_diamagnetic_flux_data_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compute uncertainty for diamagnetic flux.

        From OMAS: abs(header[3] * header[4]) * ones(nt) * 10.0 / 1000.0
        where header rarray comes from PtDataHeader.
        Returns array of shape (n_measurements, n_time).
        For DIII-D: (1, n_time) - single diamagnetic flux measurement.
        """
        key = Requirement("DIAMAG3", shot, "__ptdata__").as_key()
        nt = len(raw_data[key]['data'])
        header = raw_data[key]['rarray']

        # OMAS formula: abs(header[3] * header[4]) * ones(nt) / 100.0
        # Add measurement dimension: (n_time,) -> (1, n_time)
        return (np.abs(header[3] * header[4]) * np.ones(nt) / 100.0)[np.newaxis, :]

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
