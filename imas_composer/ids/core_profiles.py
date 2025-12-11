"""
Core Profiles IDS Mapping for DIII-D

See OMAS: omas/machine_mappings/d3d.py::core_profiles_profile_1d
ZIPFIT section (lines 1664-1713)
"""

from typing import Dict, Any
import numpy as np
from scipy.interpolate import interp1d

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class CoreProfilesMapper(IDSMapper):
    """Maps DIII-D core profiles data to IMAS core_profiles IDS."""

    DOCS_PATH = "core_profiles.yaml"
    CONFIG_PATH = "core_profiles.yaml"

    def __init__(self, profiles_tree: str = 'ZIPFIT01'):
        """
        Initialize CoreProfilesMapper.

        Args:
            profiles_tree: Profile tree to use (e.g., 'ZIPFIT01', 'OMFIT_PROFS')
        """
        self.profiles_tree = profiles_tree

        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Build IDS specs
        self._build_specs()

    def _build_specs(self):
        """Build all IDS entry specifications."""

        # ============================================================
        # Internal dependencies - DIRECT stage
        # ============================================================

        # Electron density - data, time, and rho dimensions
        self.specs["core_profiles.profiles_1d._density_data"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('\\TOP.PROFILES.EDENSFIT', 0, self.profiles_tree)
            ],
            ids_path="core_profiles.profiles_1d._density_data",
            docs_file=self.DOCS_PATH
        )

        self.specs["core_profiles.profiles_1d._density_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('dim_of(\\TOP.PROFILES.EDENSFIT,1)', 0, self.profiles_tree)
            ],
            ids_path="core_profiles.profiles_1d._density_time",
            docs_file=self.DOCS_PATH
        )

        self.specs["core_profiles.profiles_1d._density_rho"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('dim_of(\\TOP.PROFILES.EDENSFIT,0)', 0, self.profiles_tree)
            ],
            ids_path="core_profiles.profiles_1d._density_rho",
            docs_file=self.DOCS_PATH
        )

        # Electron temperature - data, time, and rho dimensions
        self.specs["core_profiles.profiles_1d._temperature_data"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('\\TOP.PROFILES.ETEMPFIT', 0, self.profiles_tree)
            ],
            ids_path="core_profiles.profiles_1d._temperature_data",
            docs_file=self.DOCS_PATH
        )

        self.specs["core_profiles.profiles_1d._temperature_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('dim_of(\\TOP.PROFILES.ETEMPFIT,1)', 0, self.profiles_tree)
            ],
            ids_path="core_profiles.profiles_1d._temperature_time",
            docs_file=self.DOCS_PATH
        )

        self.specs["core_profiles.profiles_1d._temperature_rho"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('dim_of(\\TOP.PROFILES.ETEMPFIT,0)', 0, self.profiles_tree)
            ],
            ids_path="core_profiles.profiles_1d._temperature_rho",
            docs_file=self.DOCS_PATH
        )

        # Ion temperature (both D and C use ITEMPFIT) - data, time, and rho dimensions
        self.specs["core_profiles.profiles_1d._ion_temperature_data"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('\\TOP.PROFILES.ITEMPFIT', 0, self.profiles_tree)
            ],
            ids_path="core_profiles.profiles_1d._ion_temperature_data",
            docs_file=self.DOCS_PATH
        )

        self.specs["core_profiles.profiles_1d._ion_temperature_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('dim_of(\\TOP.PROFILES.ITEMPFIT,1)', 0, self.profiles_tree)
            ],
            ids_path="core_profiles.profiles_1d._ion_temperature_time",
            docs_file=self.DOCS_PATH
        )

        self.specs["core_profiles.profiles_1d._ion_temperature_rho"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('dim_of(\\TOP.PROFILES.ITEMPFIT,0)', 0, self.profiles_tree)
            ],
            ids_path="core_profiles.profiles_1d._ion_temperature_rho",
            docs_file=self.DOCS_PATH
        )

        # Carbon density - data, time, and rho dimensions
        self.specs["core_profiles.profiles_1d._carbon_density_data"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('\\TOP.PROFILES.ZDENSFIT', 0, self.profiles_tree)
            ],
            ids_path="core_profiles.profiles_1d._carbon_density_data",
            docs_file=self.DOCS_PATH
        )

        self.specs["core_profiles.profiles_1d._carbon_density_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('dim_of(\\TOP.PROFILES.ZDENSFIT,1)', 0, self.profiles_tree)
            ],
            ids_path="core_profiles.profiles_1d._carbon_density_time",
            docs_file=self.DOCS_PATH
        )

        self.specs["core_profiles.profiles_1d._carbon_density_rho"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('dim_of(\\TOP.PROFILES.ZDENSFIT,0)', 0, self.profiles_tree)
            ],
            ids_path="core_profiles.profiles_1d._carbon_density_rho",
            docs_file=self.DOCS_PATH
        )

        # Carbon rotation - data, time, and rho dimensions
        self.specs["core_profiles.profiles_1d._carbon_rotation_data"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('\\TOP.PROFILES.TROTFIT', 0, self.profiles_tree)
            ],
            ids_path="core_profiles.profiles_1d._carbon_rotation_data",
            docs_file=self.DOCS_PATH
        )

        self.specs["core_profiles.profiles_1d._carbon_rotation_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('dim_of(\\TOP.PROFILES.TROTFIT,1)', 0, self.profiles_tree)
            ],
            ids_path="core_profiles.profiles_1d._carbon_rotation_time",
            docs_file=self.DOCS_PATH
        )

        self.specs["core_profiles.profiles_1d._carbon_rotation_rho"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('dim_of(\\TOP.PROFILES.TROTFIT,0)', 0, self.profiles_tree)
            ],
            ids_path="core_profiles.profiles_1d._carbon_rotation_rho",
            docs_file=self.DOCS_PATH
        )

        # V_loop data (for global_quantities)
        # Note: treename=None means this comes from ptdata2 (not a specific tree)
        # Using DERIVED stage because ptdata2 requires the shot number in the TDI expression
        self.specs["core_profiles._vloop_data"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_vloop_data_requirements,
            ids_path="core_profiles._vloop_data",
            docs_file=self.DOCS_PATH
        )

        self.specs["core_profiles._vloop_time"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_vloop_time_requirements,
            ids_path="core_profiles._vloop_time",
            docs_file=self.DOCS_PATH
        )

        # ============================================================
        # User-facing fields - COMPUTED stage
        # ============================================================

        # Grid: rho_tor_norm
        # Unified grid from all rho dimensions
        self.specs["core_profiles.profiles_1d.grid.rho_tor_norm"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles.profiles_1d._density_rho",
                "core_profiles.profiles_1d._temperature_rho",
                "core_profiles.profiles_1d._ion_temperature_rho",
                "core_profiles.profiles_1d._carbon_density_rho",
                "core_profiles.profiles_1d._carbon_rotation_rho"
            ],
            compose=self._compose_rho_tor_norm,
            ids_path="core_profiles.profiles_1d.grid.rho_tor_norm",
            docs_file=self.DOCS_PATH
        )

        # Electrons: density_thermal
        # OMAS: multiply by 1E19 to get m^-3, interpolate to common grid
        self.specs["core_profiles.profiles_1d.electrons.density_thermal"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles.profiles_1d._density_data",
                "core_profiles.profiles_1d._density_time",
                "core_profiles.profiles_1d._density_rho",
                "core_profiles.profiles_1d._temperature_time",
                "core_profiles.profiles_1d._temperature_rho",
                "core_profiles.profiles_1d._ion_temperature_time",
                "core_profiles.profiles_1d._ion_temperature_rho",
                "core_profiles.profiles_1d._carbon_density_time",
                "core_profiles.profiles_1d._carbon_density_rho",
                "core_profiles.profiles_1d._carbon_rotation_time",
                "core_profiles.profiles_1d._carbon_rotation_rho"
            ],
            compose=self._compose_density_thermal,
            ids_path="core_profiles.profiles_1d.electrons.density_thermal",
            docs_file=self.DOCS_PATH
        )

        # Electrons: temperature
        # OMAS: multiply by 1E3 to get eV, interpolate to common grid
        self.specs["core_profiles.profiles_1d.electrons.temperature"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles.profiles_1d._temperature_data",
                "core_profiles.profiles_1d._temperature_time",
                "core_profiles.profiles_1d._temperature_rho",
                "core_profiles.profiles_1d._density_time",
                "core_profiles.profiles_1d._density_rho",
                "core_profiles.profiles_1d._ion_temperature_time",
                "core_profiles.profiles_1d._ion_temperature_rho",
                "core_profiles.profiles_1d._carbon_density_time",
                "core_profiles.profiles_1d._carbon_density_rho",
                "core_profiles.profiles_1d._carbon_rotation_time",
                "core_profiles.profiles_1d._carbon_rotation_rho"
            ],
            compose=self._compose_temperature,
            ids_path="core_profiles.profiles_1d.electrons.temperature",
            docs_file=self.DOCS_PATH
        )

        # ============================================================
        # Ion[0] (Deuterium) fields
        # ============================================================

        # Ion[0]: temperature (from ITEMPFIT)
        self.specs["core_profiles.profiles_1d.ion.0.temperature"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles.profiles_1d._ion_temperature_data",
                "core_profiles.profiles_1d._ion_temperature_time",
                "core_profiles.profiles_1d._ion_temperature_rho",
                "core_profiles.profiles_1d._density_time",
                "core_profiles.profiles_1d._density_rho",
                "core_profiles.profiles_1d._temperature_time",
                "core_profiles.profiles_1d._temperature_rho",
                "core_profiles.profiles_1d._carbon_density_time",
                "core_profiles.profiles_1d._carbon_density_rho",
                "core_profiles.profiles_1d._carbon_rotation_time",
                "core_profiles.profiles_1d._carbon_rotation_rho"
            ],
            compose=self._compose_ion_temperature,
            ids_path="core_profiles.profiles_1d.ion.0.temperature",
            docs_file=self.DOCS_PATH
        )

        # Ion[0]: density_thermal (from quasineutrality: n_D = n_e - 6*n_C)
        self.specs["core_profiles.profiles_1d.ion.0.density_thermal"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles.profiles_1d.electrons.density_thermal",
                "core_profiles.profiles_1d.ion.1.density_thermal"
            ],
            compose=self._compose_deuterium_density,
            ids_path="core_profiles.profiles_1d.ion.0.density_thermal",
            docs_file=self.DOCS_PATH
        )

        # Ion[0]: label
        self.specs["core_profiles.profiles_1d.ion.0.label"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, raw: "D",
            ids_path="core_profiles.profiles_1d.ion.0.label",
            docs_file=self.DOCS_PATH
        )

        # Ion[0]: element[0].z_n (atomic number)
        self.specs["core_profiles.profiles_1d.ion.0.element.0.z_n"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, raw: 1.0,
            ids_path="core_profiles.profiles_1d.ion.0.element.0.z_n",
            docs_file=self.DOCS_PATH
        )

        # Ion[0]: element[0].a (atomic mass)
        self.specs["core_profiles.profiles_1d.ion.0.element.0.a"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, raw: 2.0141,
            ids_path="core_profiles.profiles_1d.ion.0.element.0.a",
            docs_file=self.DOCS_PATH
        )

        # ============================================================
        # Ion[1] (Carbon) fields
        # ============================================================

        # Ion[1]: density_thermal (from ZDENSFIT)
        self.specs["core_profiles.profiles_1d.ion.1.density_thermal"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles.profiles_1d._carbon_density_data",
                "core_profiles.profiles_1d._carbon_density_time",
                "core_profiles.profiles_1d._carbon_density_rho",
                "core_profiles.profiles_1d._density_time",
                "core_profiles.profiles_1d._density_rho",
                "core_profiles.profiles_1d._temperature_time",
                "core_profiles.profiles_1d._temperature_rho",
                "core_profiles.profiles_1d._ion_temperature_time",
                "core_profiles.profiles_1d._ion_temperature_rho",
                "core_profiles.profiles_1d._carbon_rotation_time",
                "core_profiles.profiles_1d._carbon_rotation_rho"
            ],
            compose=self._compose_carbon_density,
            ids_path="core_profiles.profiles_1d.ion.1.density_thermal",
            docs_file=self.DOCS_PATH
        )

        # Ion[1]: temperature (from ITEMPFIT, same as ion[0])
        self.specs["core_profiles.profiles_1d.ion.1.temperature"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles.profiles_1d._ion_temperature_data",
                "core_profiles.profiles_1d._ion_temperature_time",
                "core_profiles.profiles_1d._ion_temperature_rho",
                "core_profiles.profiles_1d._density_time",
                "core_profiles.profiles_1d._density_rho",
                "core_profiles.profiles_1d._temperature_time",
                "core_profiles.profiles_1d._temperature_rho",
                "core_profiles.profiles_1d._carbon_density_time",
                "core_profiles.profiles_1d._carbon_density_rho",
                "core_profiles.profiles_1d._carbon_rotation_time",
                "core_profiles.profiles_1d._carbon_rotation_rho"
            ],
            compose=self._compose_ion_temperature,
            ids_path="core_profiles.profiles_1d.ion.1.temperature",
            docs_file=self.DOCS_PATH
        )

        # Ion[1]: rotation_frequency_tor (from TROTFIT)
        self.specs["core_profiles.profiles_1d.ion.1.rotation_frequency_tor"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles.profiles_1d._carbon_rotation_data",
                "core_profiles.profiles_1d._carbon_rotation_time",
                "core_profiles.profiles_1d._carbon_rotation_rho",
                "core_profiles.profiles_1d._density_time",
                "core_profiles.profiles_1d._density_rho",
                "core_profiles.profiles_1d._temperature_time",
                "core_profiles.profiles_1d._temperature_rho",
                "core_profiles.profiles_1d._ion_temperature_time",
                "core_profiles.profiles_1d._ion_temperature_rho",
                "core_profiles.profiles_1d._carbon_density_time",
                "core_profiles.profiles_1d._carbon_density_rho"
            ],
            compose=self._compose_carbon_rotation,
            ids_path="core_profiles.profiles_1d.ion.1.rotation_frequency_tor",
            docs_file=self.DOCS_PATH
        )

        # Ion[1]: label
        self.specs["core_profiles.profiles_1d.ion.1.label"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, raw: "C",
            ids_path="core_profiles.profiles_1d.ion.1.label",
            docs_file=self.DOCS_PATH
        )

        # Ion[1]: element[0].z_n (atomic number)
        self.specs["core_profiles.profiles_1d.ion.1.element.0.z_n"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, raw: 6.0,
            ids_path="core_profiles.profiles_1d.ion.1.element.0.z_n",
            docs_file=self.DOCS_PATH
        )

        # Ion[1]: element[0].a (atomic mass)
        self.specs["core_profiles.profiles_1d.ion.1.element.0.a"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, raw: 12.011,
            ids_path="core_profiles.profiles_1d.ion.1.element.0.a",
            docs_file=self.DOCS_PATH
        )

        # ============================================================
        # Time and metadata fields
        # ============================================================

        # time: unified time array from all profile signals
        self.specs["core_profiles.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles.profiles_1d._density_time",
                "core_profiles.profiles_1d._temperature_time",
                "core_profiles.profiles_1d._ion_temperature_time",
                "core_profiles.profiles_1d._carbon_density_time",
                "core_profiles.profiles_1d._carbon_rotation_time"
            ],
            compose=self._get_unified_time,
            ids_path="core_profiles.time",
            docs_file=self.DOCS_PATH
        )

        # profiles_1d.time: same as core_profiles.time (for each time slice)
        self.specs["core_profiles.profiles_1d.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["core_profiles.time"],
            compose=lambda shot, raw: self._get_unified_time(shot, raw),
            ids_path="core_profiles.profiles_1d.time",
            docs_file=self.DOCS_PATH
        )

        # ids_properties.homogeneous_time: static value from config
        self.specs["core_profiles.ids_properties.homogeneous_time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda _shot, _raw: self.static_values['ids_properties.homogeneous_time'],
            ids_path="core_profiles.ids_properties.homogeneous_time",
            docs_file=self.DOCS_PATH
        )

        # ============================================================
        # Global quantities
        # ============================================================

        # global_quantities.v_loop: loop voltage interpolated to profile time
        self.specs["core_profiles.global_quantities.v_loop"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles._vloop_data",
                "core_profiles._vloop_time",
                "core_profiles.time"
            ],
            compose=self._compose_v_loop,
            ids_path="core_profiles.global_quantities.v_loop",
            docs_file=self.DOCS_PATH
        )

    def _compose_rho_tor_norm(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compose grid.rho_tor_norm from all profile data dimensions.

        OMAS reference (d3d.py:1694-1695):
            rho_tor_norm = np.unique(np.concatenate([[1.0],np.concatenate([data[entry]
                for entry in query.keys() if entry.startswith("rho__") ...])]))
            rho_tor_norm = rho_tor_norm[rho_tor_norm<=1.0]

        Returns:
            1D array of normalized toroidal flux coordinates (rho_tor_norm)
        """
        # Collect all rho dimensions
        rho_arrays = []

        density_rho_key = Requirement('dim_of(\\TOP.PROFILES.EDENSFIT,0)', shot, self.profiles_tree).as_key()
        temp_rho_key = Requirement('dim_of(\\TOP.PROFILES.ETEMPFIT,0)', shot, self.profiles_tree).as_key()
        ion_temp_rho_key = Requirement('dim_of(\\TOP.PROFILES.ITEMPFIT,0)', shot, self.profiles_tree).as_key()
        carbon_density_rho_key = Requirement('dim_of(\\TOP.PROFILES.ZDENSFIT,0)', shot, self.profiles_tree).as_key()
        carbon_rotation_rho_key = Requirement('dim_of(\\TOP.PROFILES.TROTFIT,0)', shot, self.profiles_tree).as_key()

        rho_arrays.append(raw_data[density_rho_key])
        rho_arrays.append(raw_data[temp_rho_key])
        rho_arrays.append(raw_data[ion_temp_rho_key])
        rho_arrays.append(raw_data[carbon_density_rho_key])
        rho_arrays.append(raw_data[carbon_rotation_rho_key])

        # Concatenate all rho values with [1.0] and get unique values
        # OMAS concatenates [1.0] with all unique rho values, then filters to <= 1.0
        rho_tor_norm = np.unique(np.concatenate([[1.0]] + rho_arrays))
        rho_tor_norm = rho_tor_norm[rho_tor_norm <= 1.0]

        return rho_tor_norm

    def _compose_density_thermal(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compose electrons.density_thermal.

        OMAS reference (d3d.py:1666, 1687, 1693, 1710):
            query["electrons.density_thermal"] = "\\TOP.PROFILES.EDENSFIT"
            data[entry] *= 1E19  # in [m^-3]
            time = np.unique(np.concatenate([data[entry] for entry in query.keys()
                if entry.startswith("time__") ...]))
            time_index = np.argmin(np.abs(data["time__" + entry] - time0))
            interp1d(data["rho__" + entry], data[entry][time_index],
                     bounds_error=False, fill_value=np.nan)(rho_tor_norm)

        Returns:
            2D array of shape (n_time, n_rho) with electron density in m^-3
        """
        # Get raw data
        data_key = Requirement('\\TOP.PROFILES.EDENSFIT', shot, self.profiles_tree).as_key()
        density_raw = raw_data[data_key]

        # Get time and rho dimensions for this signal
        time_key = Requirement('dim_of(\\TOP.PROFILES.EDENSFIT,1)', shot, self.profiles_tree).as_key()
        rho_key = Requirement('dim_of(\\TOP.PROFILES.EDENSFIT,0)', shot, self.profiles_tree).as_key()
        signal_time = raw_data[time_key] * 1e-3  # Convert ms to s
        signal_rho = raw_data[rho_key]

        # Get unified time array (from all signals)
        unified_time = self._get_unified_time(shot, raw_data)

        # Get common rho_tor_norm grid
        rho_tor_norm = self._compose_rho_tor_norm(shot, raw_data)

        # Interpolate to common grid
        # Shape: (n_time, n_rho)
        result = np.zeros((len(unified_time), len(rho_tor_norm)))

        for i_time, time0 in enumerate(unified_time):
            # Find nearest time index in this signal's native time array
            time_index = np.argmin(np.abs(signal_time - time0))

            # Get data at this time and convert to m^-3 (OMAS multiplies by 1E19)
            data_at_time = density_raw[time_index, :] * 1e19

            # Interpolate from signal's native rho grid to common rho_tor_norm
            result[i_time, :] = interp1d(
                signal_rho,
                data_at_time,
                bounds_error=False,
                fill_value=np.nan
            )(rho_tor_norm)

        return result

    def _compose_temperature(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compose electrons.temperature.

        OMAS reference (d3d.py:1667, 1689, 1693, 1710):
            query["electrons.temperature"] = "\\TOP.PROFILES.ETEMPFIT"
            data[entry] *= 1E3  # in [eV]
            time = np.unique(np.concatenate([data[entry] for entry in query.keys()
                if entry.startswith("time__") ...]))
            time_index = np.argmin(np.abs(data["time__" + entry] - time0))
            interp1d(data["rho__" + entry], data[entry][time_index],
                     bounds_error=False, fill_value=np.nan)(rho_tor_norm)

        Returns:
            2D array of shape (n_time, n_rho) with electron temperature in eV
        """
        # Get raw data
        data_key = Requirement('\\TOP.PROFILES.ETEMPFIT', shot, self.profiles_tree).as_key()
        temperature_raw = raw_data[data_key]

        # Get time and rho dimensions for this signal
        time_key = Requirement('dim_of(\\TOP.PROFILES.ETEMPFIT,1)', shot, self.profiles_tree).as_key()
        rho_key = Requirement('dim_of(\\TOP.PROFILES.ETEMPFIT,0)', shot, self.profiles_tree).as_key()
        signal_time = raw_data[time_key] * 1e-3  # Convert ms to s
        signal_rho = raw_data[rho_key]

        # Get unified time array (from all signals)
        unified_time = self._get_unified_time(shot, raw_data)

        # Get common rho_tor_norm grid
        rho_tor_norm = self._compose_rho_tor_norm(shot, raw_data)

        # Interpolate to common grid
        # Shape: (n_time, n_rho)
        result = np.zeros((len(unified_time), len(rho_tor_norm)))

        for i_time, time0 in enumerate(unified_time):
            # Find nearest time index in this signal's native time array
            time_index = np.argmin(np.abs(signal_time - time0))

            # Get data at this time and convert to eV (OMAS multiplies by 1E3)
            data_at_time = temperature_raw[time_index, :] * 1e3

            # Interpolate from signal's native rho grid to common rho_tor_norm
            result[i_time, :] = interp1d(
                signal_rho,
                data_at_time,
                bounds_error=False,
                fill_value=np.nan
            )(rho_tor_norm)

        return result

    def _get_unified_time(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Get unified time array from all profile signals.

        OMAS reference (d3d.py:1693):
            time = np.unique(np.concatenate([data[entry] for entry in query.keys()
                if entry.startswith("time__") and not isinstance(data[entry], Exception)
                and len(data[entry])>0]))

        Returns:
            1D array of unique time points in seconds
        """
        time_arrays = []

        # Collect time dimensions from all signals
        time_keys = [
            Requirement('dim_of(\\TOP.PROFILES.EDENSFIT,1)', shot, self.profiles_tree).as_key(),
            Requirement('dim_of(\\TOP.PROFILES.ETEMPFIT,1)', shot, self.profiles_tree).as_key(),
            Requirement('dim_of(\\TOP.PROFILES.ITEMPFIT,1)', shot, self.profiles_tree).as_key(),
            Requirement('dim_of(\\TOP.PROFILES.ZDENSFIT,1)', shot, self.profiles_tree).as_key(),
            Requirement('dim_of(\\TOP.PROFILES.TROTFIT,1)', shot, self.profiles_tree).as_key(),
        ]

        # Convert from ms to s and collect
        for key in time_keys:
            if key in raw_data and len(raw_data[key]) > 0:
                time_arrays.append(raw_data[key] * 1e-3)

        # Get unique time points
        unified_time = np.unique(np.concatenate(time_arrays))

        return unified_time

    def _compose_ion_temperature(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compose ion temperature (both D and C use same ITEMPFIT data).

        OMAS reference (d3d.py:1669-1670, 1689):
            query["ion[0].temperature"] = "\\TOP.PROFILES.ITEMPFIT"
            query["ion[1].temperature"] = "\\TOP.PROFILES.ITEMPFIT"
            data[entry] *= 1E3  # in [eV]

        Returns:
            2D array of shape (n_time, n_rho) with ion temperature in eV
        """
        # Get raw data
        data_key = Requirement('\\TOP.PROFILES.ITEMPFIT', shot, self.profiles_tree).as_key()
        ion_temp_raw = raw_data[data_key]

        # Get time and rho dimensions for this signal
        time_key = Requirement('dim_of(\\TOP.PROFILES.ITEMPFIT,1)', shot, self.profiles_tree).as_key()
        rho_key = Requirement('dim_of(\\TOP.PROFILES.ITEMPFIT,0)', shot, self.profiles_tree).as_key()
        signal_time = raw_data[time_key] * 1e-3  # Convert ms to s
        signal_rho = raw_data[rho_key]

        # Get unified time array (from all signals)
        unified_time = self._get_unified_time(shot, raw_data)

        # Get common rho_tor_norm grid
        rho_tor_norm = self._compose_rho_tor_norm(shot, raw_data)

        # Interpolate to common grid
        # Shape: (n_time, n_rho)
        result = np.zeros((len(unified_time), len(rho_tor_norm)))

        for i_time, time0 in enumerate(unified_time):
            # Find nearest time index in this signal's native time array
            time_index = np.argmin(np.abs(signal_time - time0))

            # Get data at this time and convert to eV (OMAS multiplies by 1E3)
            data_at_time = ion_temp_raw[time_index, :] * 1e3

            # Interpolate from signal's native rho grid to common rho_tor_norm
            result[i_time, :] = interp1d(
                signal_rho,
                data_at_time,
                bounds_error=False,
                fill_value=np.nan
            )(rho_tor_norm)

        return result

    def _compose_carbon_density(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compose carbon density.

        OMAS reference (d3d.py:1668, 1687):
            query["ion[1].density_thermal"] = "\\TOP.PROFILES.ZDENSFIT"
            data[entry] *= 1E19  # in [m^-3]

        Returns:
            2D array of shape (n_time, n_rho) with carbon density in m^-3
        """
        # Get raw data
        data_key = Requirement('\\TOP.PROFILES.ZDENSFIT', shot, self.profiles_tree).as_key()
        carbon_density_raw = raw_data[data_key]

        # Get time and rho dimensions for this signal
        time_key = Requirement('dim_of(\\TOP.PROFILES.ZDENSFIT,1)', shot, self.profiles_tree).as_key()
        rho_key = Requirement('dim_of(\\TOP.PROFILES.ZDENSFIT,0)', shot, self.profiles_tree).as_key()
        signal_time = raw_data[time_key] * 1e-3  # Convert ms to s
        signal_rho = raw_data[rho_key]

        # Get unified time array (from all signals)
        unified_time = self._get_unified_time(shot, raw_data)

        # Get common rho_tor_norm grid
        rho_tor_norm = self._compose_rho_tor_norm(shot, raw_data)

        # Interpolate to common grid
        # Shape: (n_time, n_rho)
        result = np.zeros((len(unified_time), len(rho_tor_norm)))

        for i_time, time0 in enumerate(unified_time):
            # Find nearest time index in this signal's native time array
            time_index = np.argmin(np.abs(signal_time - time0))

            # Get data at this time and convert to m^-3 (OMAS multiplies by 1E19)
            data_at_time = carbon_density_raw[time_index, :] * 1e19

            # Interpolate from signal's native rho grid to common rho_tor_norm
            result[i_time, :] = interp1d(
                signal_rho,
                data_at_time,
                bounds_error=False,
                fill_value=np.nan
            )(rho_tor_norm)

        return result

    def _compose_carbon_rotation(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compose carbon rotation frequency.

        OMAS reference (d3d.py:1671, 1691):
            query["ion[1].rotation_frequency_tor"] = "\\TOP.PROFILES.TROTFIT"
            data[entry] *= 1E3  # in [rad/s]

        Returns:
            2D array of shape (n_time, n_rho) with rotation frequency in rad/s
        """
        # Get raw data
        data_key = Requirement('\\TOP.PROFILES.TROTFIT', shot, self.profiles_tree).as_key()
        carbon_rotation_raw = raw_data[data_key]

        # Get time and rho dimensions for this signal
        time_key = Requirement('dim_of(\\TOP.PROFILES.TROTFIT,1)', shot, self.profiles_tree).as_key()
        rho_key = Requirement('dim_of(\\TOP.PROFILES.TROTFIT,0)', shot, self.profiles_tree).as_key()
        signal_time = raw_data[time_key] * 1e-3  # Convert ms to s
        signal_rho = raw_data[rho_key]

        # Get unified time array (from all signals)
        unified_time = self._get_unified_time(shot, raw_data)

        # Get common rho_tor_norm grid
        rho_tor_norm = self._compose_rho_tor_norm(shot, raw_data)

        # Interpolate to common grid
        # Shape: (n_time, n_rho)
        result = np.zeros((len(unified_time), len(rho_tor_norm)))

        for i_time, time0 in enumerate(unified_time):
            # Find nearest time index in this signal's native time array
            time_index = np.argmin(np.abs(signal_time - time0))

            # Get data at this time and convert to rad/s (OMAS multiplies by 1E3)
            data_at_time = carbon_rotation_raw[time_index, :] * 1e3

            # Interpolate from signal's native rho grid to common rho_tor_norm
            result[i_time, :] = interp1d(
                signal_rho,
                data_at_time,
                bounds_error=False,
                fill_value=np.nan
            )(rho_tor_norm)

        return result

    def _compose_deuterium_density(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compose deuterium density from quasineutrality.

        OMAS reference (d3d.py:1712):
            # deuterium from quasineutrality
            ods[f"{sh}[{i_time}].ion[0].density_thermal"] =
                ods[f"{sh}[{i_time}].electrons.density_thermal"] -
                ods[f"{sh}[{i_time}].ion[1].density_thermal"] * 6

        Returns:
            2D array of shape (n_time, n_rho) with deuterium density in m^-3
        """
        # Get already-composed electron and carbon densities
        n_e = self._compose_density_thermal(shot, raw_data)
        n_C = self._compose_carbon_density(shot, raw_data)

        # Quasineutrality: n_D = n_e - Z_C * n_C, where Z_C = 6
        n_D = n_e - 6.0 * n_C

        return n_D

    def _derive_vloop_data_requirements(self, shot: int, _raw_data: Dict[str, Any]) -> list:
        """Derive requirements for v_loop data (needs shot number in TDI expression)."""
        return [Requirement(f'ptdata2("VLOOP",{shot})', shot, None)]

    def _derive_vloop_time_requirements(self, shot: int, _raw_data: Dict[str, Any]) -> list:
        """Derive requirements for v_loop time dimension (needs shot number in TDI expression)."""
        return [Requirement(f'dim_of(ptdata2("VLOOP",{shot}),0)', shot, None)]

    def _compose_v_loop(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compose loop voltage interpolated to profile time.

        OMAS reference (d3d.py:1740-1741):
            m = mdsvalue('d3d', pulse=pulse, TDI=f"ptdata2(\"VLOOP\",{pulse})", treename=None)
            gq['v_loop'] = interp1d(m.dim_of(0) * 1e-3, m.data(),
                                    bounds_error=False, fill_value=np.nan)(t)

        Returns:
            1D array of loop voltage in V, interpolated to profile time
        """
        # Get v_loop data and time
        vloop_data_key = Requirement(f'ptdata2("VLOOP",{shot})', shot, None).as_key()
        vloop_time_key = Requirement(f'dim_of(ptdata2("VLOOP",{shot}),0)', shot, None).as_key()

        vloop_data = raw_data[vloop_data_key]
        vloop_time = raw_data[vloop_time_key] * 1e-3  # Convert ms to s

        # Get profile time
        profile_time = self._get_unified_time(shot, raw_data)

        # Interpolate v_loop to profile time
        v_loop = interp1d(
            vloop_time,
            vloop_data,
            bounds_error=False,
            fill_value=np.nan
        )(profile_time)

        return v_loop
