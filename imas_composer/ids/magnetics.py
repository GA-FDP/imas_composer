"""
Magnetics IDS Mapping for DIII-D

Maps plasma current and diamagnetic flux measurements to IMAS magnetics IDS.
See OMAS: omas/machine_mappings/d3d.py::ip_bt_dflux_data
"""

from typing import Dict, List
import numpy as np
import awkward as ak

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

        # IPSPR15V (second IP source) - auxiliary nodes
        self.specs["magnetics._ipspr15v_data"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_ipspr15v_data_requirements,
            ids_path="magnetics._ipspr15v_data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics._ipspr15v_time"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_ipspr15v_time_requirements,
            ids_path="magnetics._ipspr15v_time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics._ipspr15v_header"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_ipspr15v_header_requirements,
            ids_path="magnetics._ipspr15v_header",
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
            depends_on=["magnetics._ip_data", "magnetics._ipspr15v_data"],
            compose=self._compose_ip_data,
            ids_path="magnetics.ip.data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.ip.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._ip_time", "magnetics._ipspr15v_time"],
            compose=self._compose_ip_time,
            ids_path="magnetics.ip.time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.ip.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._ip_data", "magnetics._ip_header",
                        "magnetics._ipspr15v_data", "magnetics._ipspr15v_header"],
            compose=self._compose_ip_data_error_upper,
            ids_path="magnetics.ip.data_error_upper",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.ip.method_name"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_ip_method_name,
            ids_path="magnetics.ip.method_name",
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
        """Derive requirements for IP data (needs shot number in TDI expression)."""
        return [Requirement(f'ptdata2("IP",{shot})', shot, None)]

    def _derive_ip_time_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for IP time dimension (needs shot number in TDI expression)."""
        return [Requirement(f'dim_of(ptdata2("IP",{shot}),0)', shot, None)]

    def _derive_ip_header_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for IP header (needs shot number in TDI expression)."""
        return [Requirement(f'pthead2("IP",{shot}), __rarray', shot, None)]

    def _derive_ipspr15v_data_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for IPSPR15V data (needs shot number in TDI expression)."""
        return [Requirement(f'ptdata2("IPSPR15V",{shot})', shot, None)]

    def _derive_ipspr15v_time_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for IPSPR15V time dimension (needs shot number in TDI expression)."""
        return [Requirement(f'dim_of(ptdata2("IPSPR15V",{shot}),0)', shot, None)]

    def _derive_ipspr15v_header_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for IPSPR15V header (needs shot number in TDI expression)."""
        return [Requirement(f'pthead2("IPSPR15V",{shot}), __rarray', shot, None)]

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
    def _compose_ip_data(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Get plasma current data.

        Returns awkward array of shape (n_measurements,) where each element is a 1D time series.
        For DIII-D: 2 measurements (IP and IPSPR15V) with different time bases.
        """
        # Get first IP source (already in Amperes)
        ip_key = Requirement(f'ptdata2("IP",{shot})', shot, None).as_key()
        ip_data = raw_data[ip_key]

        # Get second IP source (IPSPR15V) and convert units
        # IPSPR15V is in units of 2 V/MA, multiply by 500e3 to convert to Amperes
        ipspr15v_key = Requirement(f'ptdata2("IPSPR15V",{shot})', shot, None).as_key()
        ipspr15v_data = raw_data[ipspr15v_key] * 500e3

        # Return as awkward array (ragged array for different time bases)
        return ak.Array([ip_data, ipspr15v_data])

    def _compose_ip_time(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Get plasma current time base with unit conversion.

        From OMAS: dim_of(...,0)/1000. (convert ms to s)
        Returns awkward array of shape (n_measurements,) where each element is a 1D time series.
        For DIII-D: 2 measurements (IP and IPSPR15V) with potentially different time bases.
        """
        # Get first IP source time
        ip_time_key = Requirement(f'dim_of(ptdata2("IP",{shot}),0)', shot, None).as_key()
        ip_time = raw_data[ip_time_key] / 1000.0

        # Get second IP source time (IPSPR15V)
        ipspr15v_time_key = Requirement(f'dim_of(ptdata2("IPSPR15V",{shot}),0)', shot, None).as_key()
        ipspr15v_time = raw_data[ipspr15v_time_key] / 1000.0

        # Return as awkward array (ragged array for different time bases)
        return ak.Array([ip_time, ipspr15v_time])

    def _compose_ip_data_error_upper(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compute uncertainty for plasma current.

        From OMAS: abs(header[3] * header[4]) * ones(nt) * 10.0
        where header is from pthead2("IP", shot)
        Returns awkward array of shape (n_measurements,) where each element is a 1D time series.
        For DIII-D: 2 measurements (IP and IPSPR15V) with different time bases.
        """
        # Get the first IP source data to determine time length
        ip_data_key = Requirement(f'ptdata2("IP",{shot})', shot, None).as_key()
        nt_ip = len(raw_data[ip_data_key])

        # Get first IP header information
        ip_header_key = Requirement(f'pthead2("IP",{shot}), __rarray', shot, None).as_key()
        ip_header = raw_data[ip_header_key]

        # OMAS formula for IP: abs(header[3] * header[4]) * ones(nt) * 10.0
        ip_error = np.abs(ip_header[3] * ip_header[4]) * np.ones(nt_ip) * 10.0

        # Get the second IP source (IPSPR15V) data to determine time length
        ipspr15v_data_key = Requirement(f'ptdata2("IPSPR15V",{shot})', shot, None).as_key()
        nt_ipspr15v = len(raw_data[ipspr15v_data_key])

        # Get second IP header information
        ipspr15v_header_key = Requirement(f'pthead2("IPSPR15V",{shot}), __rarray', shot, None).as_key()
        ipspr15v_header = raw_data[ipspr15v_header_key]

        # OMAS formula for IPSPR15V: abs(header[3] * header[4]) * ones(nt) * 10.0 * unit_conversion
        # Since IPSPR15V is in units of 2 V/MA, the error also needs to be scaled by 500e3
        ipspr15v_error = np.abs(ipspr15v_header[3] * ipspr15v_header[4]) * np.ones(nt_ipspr15v) * 10.0 * 500e3

        # Return as awkward array (ragged array for different time bases)
        return ak.Array([ip_error, ipspr15v_error])

    def _compose_ip_method_name(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get method names for IP measurements.

        Returns array of strings with length n_measurements.
        For DIII-D: ["IP", "IPSPR15V"] - two IP source names.
        """
        return np.array(["IP", "IPSPR15V"], dtype=str)

    def _compose_diamagnetic_flux_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get diamagnetic flux data with unit conversion.

        From OMAS: data * 1e-3 (convert to Weber)
        Returns array of shape (n_measurements, n_time).
        For DIII-D: (1, n_time) - single diamagnetic flux measurement.
        """
        data_key = Requirement(f'ptdata2("DIAMAG3",{shot})', shot, None).as_key()
        # Add measurement dimension: (n_time,) -> (1, n_time)
        return (raw_data[data_key] * 1e-3)[np.newaxis, :]

    def _compose_diamagnetic_flux_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get diamagnetic flux time base with unit conversion.

        From OMAS: dim_of(...,0)/1000. (convert ms to s)
        Returns array of shape (n_measurements, n_time).
        For DIII-D: (1, n_time) - single diamagnetic flux measurement.
        """
        time_key = Requirement(f'dim_of(ptdata2("DIAMAG3",{shot}),0)', shot, None).as_key()
        # Add measurement dimension: (n_time,) -> (1, n_time)
        return (raw_data[time_key] / 1000.0)[np.newaxis, :]

    def _compose_diamagnetic_flux_data_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compute uncertainty for diamagnetic flux.

        From OMAS: abs(header[3] * header[4]) * ones(nt) * 10.0 / 1000.0
        where header is from pthead2("DIAMAG3", shot)
        Returns array of shape (n_measurements, n_time).
        For DIII-D: (1, n_time) - single diamagnetic flux measurement.
        """
        # Get the data to determine time length
        data_key = Requirement(f'ptdata2("DIAMAG3",{shot})', shot, None).as_key()
        nt = len(raw_data[data_key])

        # Get header information
        header_key = Requirement(f'pthead2("DIAMAG3",{shot}), __rarray', shot, None).as_key()
        header = raw_data[header_key]

        # OMAS formula: abs(header[3] * header[4]) * ones(nt) / 100.0
        # Add measurement dimension: (n_time,) -> (1, n_time)
        return (np.abs(header[3] * header[4]) * np.ones(nt) / 100.0)[np.newaxis, :]

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
