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
        self.specs["magnetics.ip.0.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._ip_data"],
            compose=self._compose_ip_data,
            ids_path="magnetics.ip.0.data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.ip.0.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._ip_time"],
            compose=self._compose_ip_time,
            ids_path="magnetics.ip.0.time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.ip.0.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._ip_data", "magnetics._ip_header"],
            compose=self._compose_ip_data_error_upper,
            ids_path="magnetics.ip.0.data_error_upper",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.diamagnetic_flux.0.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._diamag_data"],
            compose=self._compose_diamagnetic_flux_data,
            ids_path="magnetics.diamagnetic_flux.0.data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.diamagnetic_flux.0.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._diamag_time"],
            compose=self._compose_diamagnetic_flux_time,
            ids_path="magnetics.diamagnetic_flux.0.time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.diamagnetic_flux.0.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._diamag_data", "magnetics._diamag_header"],
            compose=self._compose_diamagnetic_flux_data_error_upper,
            ids_path="magnetics.diamagnetic_flux.0.data_error_upper",
            docs_file=self.CONFIG_PATH
        )

    # Requirement derivation functions
    def _derive_ip_data_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for IP data (needs shot number in TDI expression)."""
        return [Requirement(f'ptdata2("IP",{shot})', shot, None)]

    def _derive_ip_time_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for IP time dimension (needs shot number in TDI expression)."""
        return [Requirement(f'dim_of(ptdata2("IP",{shot}),0)', shot, None)]

    def _derive_ip_header_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for IP header (needs shot number in TDI expression)."""
        return [Requirement(f'pthead2("IP",{shot}), __rarray', shot, None)]

    def _derive_diamag_data_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for DIAMAG3 data (needs shot number in TDI expression)."""
        return [Requirement(f'ptdata2("DIAMAG3",{shot})', shot, None)]

    def _derive_diamag_time_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for DIAMAG3 time dimension (needs shot number in TDI expression)."""
        return [Requirement(f'dim_of(ptdata2("DIAMAG3",{shot}),0)', shot, None)]

    def _derive_diamag_header_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for DIAMAG3 header (needs shot number in TDI expression)."""
        return [Requirement(f'pthead2("DIAMAG3",{shot}), __rarray', shot, None)]

    # Compose functions
    def _compose_ip_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """Get plasma current data."""
        data_key = Requirement(f'ptdata2("IP",{shot})', shot, None).as_key()
        return raw_data[data_key]

    def _compose_ip_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get plasma current time base with unit conversion.

        From OMAS: dim_of(...,0)/1000. (convert ms to s)
        """
        time_key = Requirement(f'dim_of(ptdata2("IP",{shot}),0)', shot, None).as_key()
        return raw_data[time_key] / 1000.0

    def _compose_ip_data_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compute uncertainty for plasma current.

        From OMAS: abs(header[3] * header[4]) * ones(nt) * 10.0
        where header is from pthead2("IP", shot)
        """
        # Get the data to determine time length
        data_key = Requirement(f'ptdata2("IP",{shot})', shot, None).as_key()
        nt = len(raw_data[data_key])

        # Get header information
        header_key = Requirement(f'pthead2("IP",{shot}), __rarray', shot, None).as_key()
        header = raw_data[header_key]

        # OMAS formula: abs(header[3] * header[4]) * ones(nt) * 10.0
        return np.abs(header[3] * header[4]) * np.ones(nt) * 10.0

    def _compose_diamagnetic_flux_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get diamagnetic flux data with unit conversion.

        From OMAS: data * 1e-3 (convert to Weber)
        """
        data_key = Requirement(f'ptdata2("DIAMAG3",{shot})', shot, None).as_key()
        return raw_data[data_key] * 1e-3

    def _compose_diamagnetic_flux_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get diamagnetic flux time base with unit conversion.

        From OMAS: dim_of(...,0)/1000. (convert ms to s)
        """
        time_key = Requirement(f'dim_of(ptdata2("DIAMAG3",{shot}),0)', shot, None).as_key()
        return raw_data[time_key] / 1000.0

    def _compose_diamagnetic_flux_data_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compute uncertainty for diamagnetic flux.

        From OMAS: abs(header[3] * header[4]) * ones(nt) * 10.0
        where header is from pthead2("DIAMAG3", shot)
        """
        # Get the data to determine time length
        data_key = Requirement(f'ptdata2("DIAMAG3",{shot})', shot, None).as_key()
        nt = len(raw_data[data_key])

        # Get header information
        header_key = Requirement(f'pthead2("DIAMAG3",{shot}), __rarray', shot, None).as_key()
        header = raw_data[header_key]

        # OMAS formula: abs(header[3] * header[4]) * ones(nt) * 10.0 /1000.0
        return np.abs(header[3] * header[4]) * np.ones(nt) / 100.0

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
