"""
Core Profiles IDS Mapping for DIII-D

See OMAS: omas/machine_mappings/d3d.py::core_profiles_profile_1d
ZIPFIT section (lines 1664-1713)
"""

from typing import Dict, Any
import numpy as np
from scipy.interpolate import interp1d, InterpolatedUnivariateSpline
import awkward as ak

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class CoreProfilesMapper(IDSMapper):
    """Maps DIII-D core profiles data to IMAS core_profiles IDS."""

    DOCS_PATH = "core_profiles.yaml"
    CONFIG_PATH = "core_profiles.yaml"

    def __init__(self, profiles_tree: str = 'ZIPFIT01', run_id: str = '001'):
        """
        Initialize CoreProfilesMapper.

        Args:
            profiles_tree: Profile tree to use (e.g., 'ZIPFIT01', 'OMFIT_PROFS')
            run_id: Run ID to append to pulse for OMFIT_PROFS tree (default: '001')
        """
        self.profiles_tree = profiles_tree
        self.run_id = run_id
        self.is_omfit_profs = 'OMFIT_PROFS' in profiles_tree

        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Build IDS specs
        self._build_specs()

    def _get_pulse_id(self, shot: int) -> int:
        """
        Get the pulse ID to use for MDSplus queries.

        For OMFIT_PROFS tree, appends run_id to shot number.
        For ZIPFIT, returns shot unchanged.

        Args:
            shot: Base shot number

        Returns:
            Pulse ID for MDSplus query
        """
        if self.is_omfit_profs and self.run_id is not None:
            return int(str(shot) + self.run_id)
        return shot

    def _get_mds_path(self, field_type: str) -> str:
        """
        Get the MDSplus TDI path for a given field type.

        Args:
            field_type: One of 'density', 'temperature', 'ion_temperature',
                       'deuterium_density', 'carbon_density', 'carbon_rotation'

        Returns:
            MDSplus path string
        """
        if self.is_omfit_profs:
            path_map = {
                'density': '\\TOP.N_E',
                'temperature': '\\TOP.T_E',
                'ion_temperature': '\\TOP.T_D',
                'deuterium_density': '\\TOP.N_D',
                'carbon_density': '\\TOP.N_C',
                'carbon_rotation': '\\TOP.V_TOR_C'
            }
        else:  # ZIPFIT
            path_map = {
                'density': '\\TOP.PROFILES.EDENSFIT',
                'temperature': '\\TOP.PROFILES.ETEMPFIT',
                'ion_temperature': '\\TOP.PROFILES.ITEMPFIT',
                'carbon_density': '\\TOP.PROFILES.ZDENSFIT',
                'carbon_rotation': '\\TOP.PROFILES.TROTFIT'
            }

        if field_type not in path_map:
            raise ValueError(f"Unknown field type: {field_type}")

        return path_map[field_type]

    def _get_unit_conversion(self, field_type: str) -> float:
        """
        Get the unit conversion factor for a given field type.

        OMFIT_PROFS data is already in correct units (m^-3, eV, rad/s).
        ZIPFIT data needs conversion (1E19 for density, 1E3 for temp/rotation).

        Args:
            field_type: One of 'density', 'temperature', 'rotation'

        Returns:
            Conversion factor to multiply raw data by
        """
        if self.is_omfit_profs:
            # OMFIT_PROFS already in correct units
            return 1.0
        else:  # ZIPFIT
            conversion_map = {
                'density': 1e19,      # Convert to m^-3
                'temperature': 1e3,   # Convert to eV
                'rotation': 1e3       # Convert to rad/s
            }
            return conversion_map.get(field_type, 1.0)

    def _create_profile_field_spec(self, field_name: str, field_type: str, dim: int = None) -> IDSEntrySpec:
        """
        Helper to create IDSEntrySpec for profile fields.

        Handles both OMFIT_PROFS (DERIVED) and ZIPFIT (DIRECT) trees.

        For OMFIT_PROFS: Only data fields (dim=None) are needed, not time/rho dims.
        For ZIPFIT: All fields (data, time, rho) are needed.

        Args:
            field_name: Internal field name (e.g., '_density_data', '_density_time')
            field_type: Field type for MDSplus path lookup (e.g., 'density', 'temperature')
            dim: Dimension index for dim_of() call, or None for data field

        Returns:
            IDSEntrySpec configured for the current tree type, or None if field not needed
        """
        # For OMFIT_PROFS, skip time/rho dimensions (already on common grid)
        if self.is_omfit_profs and dim is not None:
            return None

        ids_path = f"core_profiles.profiles_1d.{field_name}"
        mds_path = self._get_mds_path(field_type)

        if dim is not None:
            mds_path = f'dim_of({mds_path},{dim})'

        if self.is_omfit_profs:
            # OMFIT_PROFS uses DERIVED stage with pulse_id
            return IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw, p=mds_path: [
                    Requirement(p, self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path=ids_path,
                docs_file=self.DOCS_PATH
            )
        else:
            # ZIPFIT uses DIRECT stage with shot=0
            return IDSEntrySpec(
                stage=RequirementStage.DIRECT,
                static_requirements=[
                    Requirement(mds_path, 0, self.profiles_tree)
                ],
                ids_path=ids_path,
                docs_file=self.DOCS_PATH
            )

    def _build_specs(self):
        """Build all IDS entry specifications."""

        # ============================================================
        # Internal dependencies - DIRECT or DERIVED stage
        # For OMFIT_PROFS, use DERIVED to handle run_id appended to pulse
        # For ZIPFIT, use DIRECT (standard static requirements)
        # ============================================================

        # Electron density - data, time, and rho dimensions
        self.specs["core_profiles.profiles_1d._density_data"] = self._create_profile_field_spec(
            '_density_data', 'density', dim=None)
        spec = self._create_profile_field_spec('_density_time', 'density', dim=1)
        if spec:
            self.specs["core_profiles.profiles_1d._density_time"] = spec
        spec = self._create_profile_field_spec('_density_rho', 'density', dim=0)
        if spec:
            self.specs["core_profiles.profiles_1d._density_rho"] = spec

        # Electron temperature - data, time, and rho dimensions
        self.specs["core_profiles.profiles_1d._temperature_data"] = self._create_profile_field_spec(
            '_temperature_data', 'temperature', dim=None)
        spec = self._create_profile_field_spec('_temperature_time', 'temperature', dim=1)
        if spec:
            self.specs["core_profiles.profiles_1d._temperature_time"] = spec
        spec = self._create_profile_field_spec('_temperature_rho', 'temperature', dim=0)
        if spec:
            self.specs["core_profiles.profiles_1d._temperature_rho"] = spec

        # Ion temperature (both D and C use ITEMPFIT/T_D) - data, time, and rho dimensions
        self.specs["core_profiles.profiles_1d._ion_temperature_data"] = self._create_profile_field_spec(
            '_ion_temperature_data', 'ion_temperature', dim=None)
        spec = self._create_profile_field_spec('_ion_temperature_time', 'ion_temperature', dim=1)
        if spec:
            self.specs["core_profiles.profiles_1d._ion_temperature_time"] = spec
        spec = self._create_profile_field_spec('_ion_temperature_rho', 'ion_temperature', dim=0)
        if spec:
            self.specs["core_profiles.profiles_1d._ion_temperature_rho"] = spec

        # Carbon density - data, time, and rho dimensions
        self.specs["core_profiles.profiles_1d._carbon_density_data"] = self._create_profile_field_spec(
            '_carbon_density_data', 'carbon_density', dim=None)
        spec = self._create_profile_field_spec('_carbon_density_time', 'carbon_density', dim=1)
        if spec:
            self.specs["core_profiles.profiles_1d._carbon_density_time"] = spec
        spec = self._create_profile_field_spec('_carbon_density_rho', 'carbon_density', dim=0)
        if spec:
            self.specs["core_profiles.profiles_1d._carbon_density_rho"] = spec

        # Carbon rotation - data, time, and rho dimensions
        # Note: Only available in ZIPFIT, not OMFIT_PROFS
        if not self.is_omfit_profs:
            self.specs["core_profiles.profiles_1d._carbon_rotation_data"] = self._create_profile_field_spec(
                '_carbon_rotation_data', 'carbon_rotation', dim=None)
            spec = self._create_profile_field_spec('_carbon_rotation_time', 'carbon_rotation', dim=1)
            if spec:
                self.specs["core_profiles.profiles_1d._carbon_rotation_time"] = spec
            spec = self._create_profile_field_spec('_carbon_rotation_rho', 'carbon_rotation', dim=0)
            if spec:
                self.specs["core_profiles.profiles_1d._carbon_rotation_rho"] = spec

        # Deuterium density - only for OMFIT_PROFS (ZIPFIT uses quasineutrality)
        if self.is_omfit_profs:
            self.specs["core_profiles.profiles_1d._deuterium_density_data"] = self._create_profile_field_spec(
                '_deuterium_density_data', 'deuterium_density', dim=None)

        # Uncertainty fields - only for OMFIT_PROFS
        # OMAS d3d.py line 1579: query[entry + "_error_upper"] = "error_of(" + query[entry] + ")"
        if self.is_omfit_profs:
            # Electron density uncertainty
            self.specs["core_profiles.profiles_1d._density_error"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('error_of(\\TOP.N_E)', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._density_error",
                docs_file=self.DOCS_PATH
            )
            # Electron temperature uncertainty
            self.specs["core_profiles.profiles_1d._temperature_error"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('error_of(\\TOP.T_E)', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._temperature_error",
                docs_file=self.DOCS_PATH
            )
            # Ion temperature uncertainty (same for D and C)
            self.specs["core_profiles.profiles_1d._ion_temperature_error"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('error_of(\\TOP.T_D)', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._ion_temperature_error",
                docs_file=self.DOCS_PATH
            )
            # Deuterium density uncertainty
            self.specs["core_profiles.profiles_1d._deuterium_density_error"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('error_of(\\TOP.N_D)', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._deuterium_density_error",
                docs_file=self.DOCS_PATH
            )
            # Carbon density uncertainty
            self.specs["core_profiles.profiles_1d._carbon_density_error"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('error_of(\\TOP.N_C)', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._carbon_density_error",
                docs_file=self.DOCS_PATH
            )
            # Carbon temperature uncertainty
            self.specs["core_profiles.profiles_1d._carbon_temperature_error"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('error_of(\\TOP.T_C)', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._carbon_temperature_error",
                docs_file=self.DOCS_PATH
            )
            # E-field radial component
            self.specs["core_profiles.profiles_1d._e_field_radial_data"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('\\TOP.ER_C', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._e_field_radial_data",
                docs_file=self.DOCS_PATH
            )

            # ============================================================
            # Fit fields (raw measurements) - OMFIT_PROFS only
            # ============================================================

            # Electron density fit fields
            # measured: from RW_N_E
            self.specs["core_profiles.profiles_1d._electron_density_fit_measured"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('\\TOP.RW_N_E', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._electron_density_fit_measured",
                docs_file=self.DOCS_PATH
            )
            # psi_norm: from PS_N_E
            self.specs["core_profiles.profiles_1d._electron_density_fit_psi_norm"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('\\TOP.PS_N_E', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._electron_density_fit_psi_norm",
                docs_file=self.DOCS_PATH
            )
            # measured_error_upper: from error_of(RW_N_E)
            self.specs["core_profiles.profiles_1d._electron_density_fit_error"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('error_of(\\TOP.RW_N_E)', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._electron_density_fit_error",
                docs_file=self.DOCS_PATH
            )

            # Electron temperature fit fields
            # measured: from RW_T_E
            self.specs["core_profiles.profiles_1d._electron_temperature_fit_measured"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('\\TOP.RW_T_E', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._electron_temperature_fit_measured",
                docs_file=self.DOCS_PATH
            )
            # psi_norm: from PS_T_E
            self.specs["core_profiles.profiles_1d._electron_temperature_fit_psi_norm"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('\\TOP.PS_T_E', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._electron_temperature_fit_psi_norm",
                docs_file=self.DOCS_PATH
            )
            # measured_error_upper: from error_of(RW_T_E)
            self.specs["core_profiles.profiles_1d._electron_temperature_fit_error"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('error_of(\\TOP.RW_T_E)', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._electron_temperature_fit_error",
                docs_file=self.DOCS_PATH
            )

            # Carbon density fit fields
            # measured: from RW_N_C
            self.specs["core_profiles.profiles_1d._carbon_density_fit_measured"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('\\TOP.RW_N_C', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._carbon_density_fit_measured",
                docs_file=self.DOCS_PATH
            )
            # psi_norm: from PS_N_C
            self.specs["core_profiles.profiles_1d._carbon_density_fit_psi_norm"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('\\TOP.PS_N_C', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._carbon_density_fit_psi_norm",
                docs_file=self.DOCS_PATH
            )
            # measured_error_upper: from error_of(RW_N_C)
            self.specs["core_profiles.profiles_1d._carbon_density_fit_error"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('error_of(\\TOP.RW_N_C)', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._carbon_density_fit_error",
                docs_file=self.DOCS_PATH
            )

            # Carbon temperature fit fields
            # measured: from RW_T_C
            self.specs["core_profiles.profiles_1d._carbon_temperature_fit_measured"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('\\TOP.RW_T_C', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._carbon_temperature_fit_measured",
                docs_file=self.DOCS_PATH
            )
            # psi_norm: from PS_T_C
            self.specs["core_profiles.profiles_1d._carbon_temperature_fit_psi_norm"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('\\TOP.PS_T_C', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._carbon_temperature_fit_psi_norm",
                docs_file=self.DOCS_PATH
            )
            # measured_error_upper: from error_of(RW_T_C)
            self.specs["core_profiles.profiles_1d._carbon_temperature_fit_error"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('error_of(\\TOP.RW_T_C)', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._carbon_temperature_fit_error",
                docs_file=self.DOCS_PATH
            )

        # For OMFIT_PROFS, we need the common rho grid and common time
        # OMAS d3d.py line 1589: data['time'] = dim_info.dim_of(1) * 1.e-3
        if self.is_omfit_profs:
            self.specs["core_profiles.profiles_1d._omfit_rho"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('\\TOP.rho', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._omfit_rho",
                docs_file=self.DOCS_PATH
            )
            # For OMFIT_PROFS, time comes from dim_of(\TOP.n_e, 1)
            self.specs["core_profiles.profiles_1d._omfit_time"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('dim_of(\\TOP.N_E,1)', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._omfit_time",
                docs_file=self.DOCS_PATH
            )
            # For OMFIT_PROFS, psi_norm comes from dim_of(\TOP.n_e, 0)
            # Used to calculate rho_pol_norm = sqrt(psi_norm)
            self.specs["core_profiles.profiles_1d._omfit_psi_norm"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw: [
                    Requirement('dim_of(\\TOP.N_E,0)', self._get_pulse_id(shot), self.profiles_tree)
                ],
                ids_path="core_profiles.profiles_1d._omfit_psi_norm",
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
        # Unified grid from all rho dimensions (ZIPFIT) or from TOP.rho (OMFIT_PROFS)
        deps = []
        if self.is_omfit_profs:
            deps = ["core_profiles.profiles_1d._omfit_rho"]
        else:
            deps = [
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
            ]

        self.specs["core_profiles.profiles_1d.grid.rho_tor_norm"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=deps,
            compose=self._compose_rho_tor_norm,
            ids_path="core_profiles.profiles_1d.grid.rho_tor_norm",
            docs_file=self.DOCS_PATH
        )

        # Grid: rho_pol_norm (OMFIT_PROFS only)
        # rho_pol_norm = sqrt(psi_norm), where psi_norm comes from dim_of(\TOP.N_E, 0)
        if self.is_omfit_profs:
            self.specs["core_profiles.profiles_1d.grid.rho_pol_norm"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=["core_profiles.profiles_1d._omfit_psi_norm", "core_profiles.profiles_1d._omfit_rho"],
                compose=self._compose_rho_pol_norm,
                ids_path="core_profiles.profiles_1d.grid.rho_pol_norm",
                docs_file=self.DOCS_PATH
            )

        # Electrons: density_thermal
        # For OMFIT_PROFS: only need data + rho, for ZIPFIT: need all time/rho dims
        density_deps = ["core_profiles.profiles_1d._density_data"]
        if self.is_omfit_profs:
            density_deps.append("core_profiles.profiles_1d._omfit_rho")
        else:
            density_deps.extend([
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
            ])

        self.specs["core_profiles.profiles_1d.electrons.density_thermal"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=density_deps,
            compose=self._compose_density_thermal,
            ids_path="core_profiles.profiles_1d.electrons.density_thermal",
            docs_file=self.DOCS_PATH
        )

        # Electrons: density_thermal_error_upper (OMFIT_PROFS only)
        if self.is_omfit_profs:
            self.specs["core_profiles.profiles_1d.electrons.density_thermal_error_upper"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=["core_profiles.profiles_1d._density_error", "core_profiles.profiles_1d._omfit_rho"],
                compose=self._compose_density_error,
                ids_path="core_profiles.profiles_1d.electrons.density_thermal_error_upper",
                docs_file=self.DOCS_PATH
            )

        # Electrons: temperature
        # For OMFIT_PROFS: only need data + rho, for ZIPFIT: need all time/rho dims
        temperature_deps = ["core_profiles.profiles_1d._temperature_data"]
        if self.is_omfit_profs:
            temperature_deps.append("core_profiles.profiles_1d._omfit_rho")
        else:
            temperature_deps.extend([
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
            ])

        self.specs["core_profiles.profiles_1d.electrons.temperature"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=temperature_deps,
            compose=self._compose_temperature,
            ids_path="core_profiles.profiles_1d.electrons.temperature",
            docs_file=self.DOCS_PATH
        )

        # Electrons: temperature_error_upper (OMFIT_PROFS only)
        if self.is_omfit_profs:
            self.specs["core_profiles.profiles_1d.electrons.temperature_error_upper"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=["core_profiles.profiles_1d._temperature_error", "core_profiles.profiles_1d._omfit_rho"],
                compose=self._compose_temperature_error,
                ids_path="core_profiles.profiles_1d.electrons.temperature_error_upper",
                docs_file=self.DOCS_PATH
            )

        # ============================================================
        # Ion[0] (Deuterium) fields
        # ============================================================

        # Ion[0]: temperature (from ITEMPFIT)
        # For OMFIT_PROFS: only need data + rho, for ZIPFIT: need all time/rho dims
        ion_temp_deps = ["core_profiles.profiles_1d._ion_temperature_data"]
        if self.is_omfit_profs:
            ion_temp_deps.append("core_profiles.profiles_1d._omfit_rho")
        else:
            ion_temp_deps.extend([
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
            ])

        self.specs["core_profiles.profiles_1d.ion.0.temperature"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=ion_temp_deps,
            compose=self._compose_ion_temperature,
            ids_path="core_profiles.profiles_1d.ion.0.temperature",
            docs_file=self.DOCS_PATH
        )

        # Ion[0]: temperature_error_upper (OMFIT_PROFS only)
        if self.is_omfit_profs:
            self.specs["core_profiles.profiles_1d.ion.0.temperature_error_upper"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=["core_profiles.profiles_1d._ion_temperature_error", "core_profiles.profiles_1d._omfit_rho"],
                compose=self._compose_ion_temperature_error,
                ids_path="core_profiles.profiles_1d.ion.0.temperature_error_upper",
                docs_file=self.DOCS_PATH
            )

        # Ion[0]: density_thermal
        # For OMFIT_PROFS: comes from \TOP.N_D directly
        # For ZIPFIT: calculated from quasineutrality (n_D = n_e - 6*n_C)
        deuterium_deps = []
        if self.is_omfit_profs:
            deuterium_deps = [
                "core_profiles.profiles_1d._deuterium_density_data",
                "core_profiles.profiles_1d._omfit_rho"
            ]
        else:
            deuterium_deps = [
                "core_profiles.profiles_1d.electrons.density_thermal",
                "core_profiles.profiles_1d.ion.1.density_thermal"
            ]

        self.specs["core_profiles.profiles_1d.ion.0.density_thermal"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=deuterium_deps,
            compose=self._compose_deuterium_density,
            ids_path="core_profiles.profiles_1d.ion.0.density_thermal",
            docs_file=self.DOCS_PATH
        )

        # Ion[0]: density_thermal_error_upper (OMFIT_PROFS only)
        if self.is_omfit_profs:
            self.specs["core_profiles.profiles_1d.ion.0.density_thermal_error_upper"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=["core_profiles.profiles_1d._deuterium_density_error", "core_profiles.profiles_1d._omfit_rho"],
                compose=self._compose_deuterium_density_error,
                ids_path="core_profiles.profiles_1d.ion.0.density_thermal_error_upper",
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
        # For OMFIT_PROFS: only need data + rho, for ZIPFIT: need all time/rho dims
        carbon_density_deps = ["core_profiles.profiles_1d._carbon_density_data"]
        if self.is_omfit_profs:
            carbon_density_deps.append("core_profiles.profiles_1d._omfit_rho")
        else:
            carbon_density_deps.extend([
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
            ])

        self.specs["core_profiles.profiles_1d.ion.1.density_thermal"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=carbon_density_deps,
            compose=self._compose_carbon_density,
            ids_path="core_profiles.profiles_1d.ion.1.density_thermal",
            docs_file=self.DOCS_PATH
        )

        # Ion[1]: density_thermal_error_upper (OMFIT_PROFS only)
        if self.is_omfit_profs:
            self.specs["core_profiles.profiles_1d.ion.1.density_thermal_error_upper"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=["core_profiles.profiles_1d._carbon_density_error", "core_profiles.profiles_1d._omfit_rho"],
                compose=self._compose_carbon_density_error,
                ids_path="core_profiles.profiles_1d.ion.1.density_thermal_error_upper",
                docs_file=self.DOCS_PATH
            )

        # Ion[1]: temperature (from ITEMPFIT, same as ion[0])
        self.specs["core_profiles.profiles_1d.ion.1.temperature"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=ion_temp_deps,  # Reuse ion_temp_deps from ion[0]
            compose=self._compose_ion_temperature,
            ids_path="core_profiles.profiles_1d.ion.1.temperature",
            docs_file=self.DOCS_PATH
        )

        # Ion[1]: temperature_error_upper (OMFIT_PROFS only)
        if self.is_omfit_profs:
            self.specs["core_profiles.profiles_1d.ion.1.temperature_error_upper"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=["core_profiles.profiles_1d._carbon_temperature_error", "core_profiles.profiles_1d._omfit_rho"],
                compose=self._compose_carbon_temperature_error,
                ids_path="core_profiles.profiles_1d.ion.1.temperature_error_upper",
                docs_file=self.DOCS_PATH
            )

        # Ion[1]: rotation_frequency_tor (from TROTFIT)
        # Note: Only available in ZIPFIT, not OMFIT_PROFS
        if not self.is_omfit_profs:
            carbon_rotation_deps = [
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
            ]

            self.specs["core_profiles.profiles_1d.ion.1.rotation_frequency_tor"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=carbon_rotation_deps,
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
        # E-field (OMFIT_PROFS only)
        # ============================================================

        # e_field.radial
        if self.is_omfit_profs:
            self.specs["core_profiles.profiles_1d.e_field.radial"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=["core_profiles.profiles_1d._e_field_radial_data", "core_profiles.profiles_1d._omfit_rho"],
                compose=self._compose_e_field_radial,
                ids_path="core_profiles.profiles_1d.e_field.radial",
                docs_file=self.DOCS_PATH
            )

        # ============================================================
        # Fit fields (raw measurements) - OMFIT_PROFS only
        # ============================================================

        if self.is_omfit_profs:
            # Electron density_fit.* fields
            self.specs["core_profiles.profiles_1d.electrons.density_fit.measured"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._electron_density_fit_measured",
                    "core_profiles.profiles_1d._omfit_rho"
                ],
                compose=self._compose_electron_density_fit_measured,
                ids_path="core_profiles.profiles_1d.electrons.density_fit.measured",
                docs_file=self.DOCS_PATH
            )
            self.specs["core_profiles.profiles_1d.electrons.density_fit.psi_norm"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._electron_density_fit_psi_norm",
                    "core_profiles.profiles_1d._electron_density_fit_measured"
                ],
                compose=self._compose_electron_density_fit_psi_norm,
                ids_path="core_profiles.profiles_1d.electrons.density_fit.psi_norm",
                docs_file=self.DOCS_PATH
            )
            self.specs["core_profiles.profiles_1d.electrons.density_fit.rho_tor_norm"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._electron_density_fit_psi_norm",
                    "core_profiles.profiles_1d._electron_density_fit_measured",
                    "core_profiles.profiles_1d._omfit_psi_norm",
                    "core_profiles.profiles_1d._omfit_rho"
                ],
                compose=self._compose_electron_density_fit_rho_tor_norm,
                ids_path="core_profiles.profiles_1d.electrons.density_fit.rho_tor_norm",
                docs_file=self.DOCS_PATH
            )
            self.specs["core_profiles.profiles_1d.electrons.density_fit.measured_error_upper"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._electron_density_fit_error",
                    "core_profiles.profiles_1d._electron_density_fit_measured"
                ],
                compose=self._compose_electron_density_fit_error,
                ids_path="core_profiles.profiles_1d.electrons.density_fit.measured_error_upper",
                docs_file=self.DOCS_PATH
            )

            # Electron temperature_fit.* fields
            self.specs["core_profiles.profiles_1d.electrons.temperature_fit.measured"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._electron_temperature_fit_measured",
                    "core_profiles.profiles_1d._omfit_rho"
                ],
                compose=self._compose_electron_temperature_fit_measured,
                ids_path="core_profiles.profiles_1d.electrons.temperature_fit.measured",
                docs_file=self.DOCS_PATH
            )
            self.specs["core_profiles.profiles_1d.electrons.temperature_fit.psi_norm"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._electron_temperature_fit_psi_norm",
                    "core_profiles.profiles_1d._electron_temperature_fit_measured"
                ],
                compose=self._compose_electron_temperature_fit_psi_norm,
                ids_path="core_profiles.profiles_1d.electrons.temperature_fit.psi_norm",
                docs_file=self.DOCS_PATH
            )
            self.specs["core_profiles.profiles_1d.electrons.temperature_fit.rho_tor_norm"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._electron_temperature_fit_psi_norm",
                    "core_profiles.profiles_1d._electron_temperature_fit_measured",
                    "core_profiles.profiles_1d._omfit_psi_norm",
                    "core_profiles.profiles_1d._omfit_rho"
                ],
                compose=self._compose_electron_temperature_fit_rho_tor_norm,
                ids_path="core_profiles.profiles_1d.electrons.temperature_fit.rho_tor_norm",
                docs_file=self.DOCS_PATH
            )
            self.specs["core_profiles.profiles_1d.electrons.temperature_fit.measured_error_upper"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._electron_temperature_fit_error",
                    "core_profiles.profiles_1d._electron_temperature_fit_measured"
                ],
                compose=self._compose_electron_temperature_fit_error,
                ids_path="core_profiles.profiles_1d.electrons.temperature_fit.measured_error_upper",
                docs_file=self.DOCS_PATH
            )

            # Ion[1] (Carbon) density_fit.* fields
            self.specs["core_profiles.profiles_1d.ion.1.density_fit.measured"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._carbon_density_fit_measured",
                    "core_profiles.profiles_1d._omfit_rho"
                ],
                compose=self._compose_carbon_density_fit_measured,
                ids_path="core_profiles.profiles_1d.ion.1.density_fit.measured",
                docs_file=self.DOCS_PATH
            )
            self.specs["core_profiles.profiles_1d.ion.1.density_fit.psi_norm"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._carbon_density_fit_psi_norm",
                    "core_profiles.profiles_1d._carbon_density_fit_measured"
                ],
                compose=self._compose_carbon_density_fit_psi_norm,
                ids_path="core_profiles.profiles_1d.ion.1.density_fit.psi_norm",
                docs_file=self.DOCS_PATH
            )
            self.specs["core_profiles.profiles_1d.ion.1.density_fit.rho_tor_norm"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._carbon_density_fit_psi_norm",
                    "core_profiles.profiles_1d._carbon_density_fit_measured",
                    "core_profiles.profiles_1d._omfit_psi_norm",
                    "core_profiles.profiles_1d._omfit_rho"
                ],
                compose=self._compose_carbon_density_fit_rho_tor_norm,
                ids_path="core_profiles.profiles_1d.ion.1.density_fit.rho_tor_norm",
                docs_file=self.DOCS_PATH
            )
            self.specs["core_profiles.profiles_1d.ion.1.density_fit.measured_error_upper"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._carbon_density_fit_error",
                    "core_profiles.profiles_1d._carbon_density_fit_measured"
                ],
                compose=self._compose_carbon_density_fit_error,
                ids_path="core_profiles.profiles_1d.ion.1.density_fit.measured_error_upper",
                docs_file=self.DOCS_PATH
            )

            # Ion[1] (Carbon) temperature_fit.* fields
            self.specs["core_profiles.profiles_1d.ion.1.temperature_fit.measured"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._carbon_temperature_fit_measured",
                    "core_profiles.profiles_1d._omfit_rho"
                ],
                compose=self._compose_carbon_temperature_fit_measured,
                ids_path="core_profiles.profiles_1d.ion.1.temperature_fit.measured",
                docs_file=self.DOCS_PATH
            )
            self.specs["core_profiles.profiles_1d.ion.1.temperature_fit.psi_norm"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._carbon_temperature_fit_psi_norm",
                    "core_profiles.profiles_1d._carbon_temperature_fit_measured"
                ],
                compose=self._compose_carbon_temperature_fit_psi_norm,
                ids_path="core_profiles.profiles_1d.ion.1.temperature_fit.psi_norm",
                docs_file=self.DOCS_PATH
            )
            self.specs["core_profiles.profiles_1d.ion.1.temperature_fit.rho_tor_norm"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._carbon_temperature_fit_psi_norm",
                    "core_profiles.profiles_1d._carbon_temperature_fit_measured",
                    "core_profiles.profiles_1d._omfit_psi_norm",
                    "core_profiles.profiles_1d._omfit_rho"
                ],
                compose=self._compose_carbon_temperature_fit_rho_tor_norm,
                ids_path="core_profiles.profiles_1d.ion.1.temperature_fit.rho_tor_norm",
                docs_file=self.DOCS_PATH
            )
            self.specs["core_profiles.profiles_1d.ion.1.temperature_fit.measured_error_upper"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[
                    "core_profiles.profiles_1d._carbon_temperature_fit_error",
                    "core_profiles.profiles_1d._carbon_temperature_fit_measured"
                ],
                compose=self._compose_carbon_temperature_fit_error,
                ids_path="core_profiles.profiles_1d.ion.1.temperature_fit.measured_error_upper",
                docs_file=self.DOCS_PATH
            )

        # ============================================================
        # Time and metadata fields
        # ============================================================

        # time: unified time array from all profile signals
        # Time: for OMFIT_PROFS use _omfit_time, for ZIPFIT use all individual time dims
        time_deps = []
        if self.is_omfit_profs:
            time_deps = ["core_profiles.profiles_1d._omfit_time"]
        else:
            time_deps = [
                "core_profiles.profiles_1d._density_time",
                "core_profiles.profiles_1d._temperature_time",
                "core_profiles.profiles_1d._ion_temperature_time",
                "core_profiles.profiles_1d._carbon_density_time",
                "core_profiles.profiles_1d._carbon_rotation_time"
            ]

        self.specs["core_profiles.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=time_deps,
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

    def _get_requirement_key(self, field_type: str, shot: int, dim: int = None) -> str:
        """
        Get the requirement key for fetching from raw_data.

        Args:
            field_type: Field type (e.g., 'density', 'temperature')
            shot: Shot number
            dim: Dimension index for dim_of(), or None for data

        Returns:
            Requirement key string
        """
        mds_path = self._get_mds_path(field_type)
        if dim is not None:
            mds_path = f'dim_of({mds_path},{dim})'

        pulse_id = self._get_pulse_id(shot)
        return Requirement(mds_path, pulse_id, self.profiles_tree).as_key()

    def _compose_rho_tor_norm(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compose grid.rho_tor_norm from all profile data dimensions.

        For ZIPFIT (d3d.py:1694-1695):
            rho_tor_norm = np.unique(np.concatenate([[1.0],np.concatenate([data[entry]
                for entry in query.keys() if entry.startswith("rho__") ...])]))
            rho_tor_norm = rho_tor_norm[rho_tor_norm<=1.0]

        For OMFIT_PROFS (d3d.py:1570):
            query["grid.rho_tor_norm"] = "rho"  # -> \\TOP.rho
            # Returns 2D array [time, rho]

        Returns:
            1D array of normalized toroidal flux coordinates (rho_tor_norm)
        """
        if self.is_omfit_profs:
            # For OMFIT_PROFS, rho comes directly from \TOP.rho
            # It's a 2D array [time, rho], but all time slices have the same rho grid
            # So we just take the first time slice
            rho_key = Requirement('\\TOP.rho', self._get_pulse_id(shot), self.profiles_tree).as_key()
            rho_2d = raw_data[rho_key]

            # Filter to <= 1.0 if core_profiles_strict_grid
            # Broadcast to match the 2D shape [time, rho]
            n_time = rho_2d.shape[0]
            result = []
            for i_time in range(n_time):
                mask = rho_2d[i_time, :] <= 1.0
                result.append(rho_2d[i_time, mask])

            return result
        else:
            # For ZIPFIT, collect all rho dimensions using helper method
            rho_arrays = []

            density_rho_key = self._get_requirement_key('density', shot, dim=0)
            temp_rho_key = self._get_requirement_key('temperature', shot, dim=0)
            ion_temp_rho_key = self._get_requirement_key('ion_temperature', shot, dim=0)
            carbon_density_rho_key = self._get_requirement_key('carbon_density', shot, dim=0)
            carbon_rotation_rho_key = self._get_requirement_key('carbon_rotation', shot, dim=0)

            rho_arrays.append(raw_data[density_rho_key])
            rho_arrays.append(raw_data[temp_rho_key])
            rho_arrays.append(raw_data[ion_temp_rho_key])
            rho_arrays.append(raw_data[carbon_density_rho_key])
            rho_arrays.append(raw_data[carbon_rotation_rho_key])

            # Concatenate all rho values with [1.0] and get unique values
            # OMAS concatenates [1.0] with all unique rho values, then filters to <= 1.0
            rho_tor_norm = np.unique(np.concatenate([[1.0]] + rho_arrays))
            rho_tor_norm = rho_tor_norm[rho_tor_norm <= 1.0]

            # Get unified time array (from all signals)
            unified_time = self._get_unified_time(shot, raw_data)

            # Broadcast to match the 2D shape [time, rho]
            result = []
            for i_time in range(len(unified_time)):
                result.append(rho_tor_norm)

            return result

    def _compose_rho_pol_norm(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compose grid.rho_pol_norm for OMFIT_PROFS.

        OMAS d3d.py lines 1590-1592, 1602:
            psi_n = dim_info.dim_of(0)  # Get psi_norm from dimension 0 of n_e
            data['grid.rho_pol_norm'] = np.zeros((data['time'].shape + psi_n.shape))
            data['grid.rho_pol_norm'][:] = np.sqrt(psi_n)
            # Then apply mask: data['grid.rho_pol_norm'][i_time][mask[i_time]]

        Returns:
            2D array of shape (n_time, n_rho) with normalized poloidal flux coordinates
        """
        # Get psi_norm from dim_of(\TOP.N_E, 0)
        psi_key = Requirement('dim_of(\\TOP.N_E,0)', self._get_pulse_id(shot), self.profiles_tree).as_key()
        psi_n = raw_data[psi_key]

        # Get rho for masking
        rho_key = Requirement('\\TOP.rho', self._get_pulse_id(shot), self.profiles_tree).as_key()
        rho_2d = raw_data[rho_key]

        # Calculate rho_pol_norm = sqrt(psi_norm)
        # Broadcast psi_n to match the 2D shape [time, rho]
        n_time = rho_2d.shape[0]
        rho_pol_norm_full = np.broadcast_to(np.sqrt(psi_n), (n_time, len(psi_n)))

        # Apply rho masking per time slice (same as other OMFIT_PROFS fields)
        result = []
        for i_time in range(n_time):
            mask = rho_2d[i_time, :] <= 1.0
            result.append(rho_pol_norm_full[i_time, mask])

        return np.array(result)

    def _compose_density_thermal(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compose electrons.density_thermal.

        For OMFIT_PROFS (d3d.py:1570):
            query["electrons.density_thermal"] = "n_e"  # -> \\TOP.N_E
            # Data already on common [time, rho] grid, no unit conversion needed
            # Apply rho masking per time slice: mask = data["grid.rho_tor_norm"] <= 1.0

        For ZIPFIT (d3d.py:1666, 1687, 1693, 1710):
            query["electrons.density_thermal"] = "\\TOP.PROFILES.EDENSFIT"
            data[entry] *= 1E19  # in [m^-3]
            # Interpolate to common grid

        Returns:
            2D array of shape (n_time, n_rho) with electron density in m^-3
        """
        # Get raw data using helper method
        data_key = self._get_requirement_key('density', shot, dim=None)
        density_raw = raw_data[data_key]

        # Get unit conversion factor (1e19 for ZIPFIT, 1.0 for OMFIT_PROFS)
        unit_factor = self._get_unit_conversion('density')

        if self.is_omfit_profs:
            # For OMFIT_PROFS, data is already on common [time, rho] grid
            # Just need to apply rho masking per time slice
            rho_key = Requirement('\\TOP.rho', self._get_pulse_id(shot), self.profiles_tree).as_key()
            rho_2d = raw_data[rho_key]

            # Apply unit conversion and rho masking per time slice
            result = []
            for i_time in range(density_raw.shape[0]):
                mask = rho_2d[i_time, :] <= 1.0
                result.append(density_raw[i_time, mask] * unit_factor)

            # All time slices should have same rho grid, so stack into 2D array
            return np.array(result)
        else:
            # For ZIPFIT, interpolate to common grid
            # Get time and rho dimensions for this signal
            time_key = self._get_requirement_key('density', shot, dim=1)
            rho_key = self._get_requirement_key('density', shot, dim=0)
            signal_time = raw_data[time_key] * 1e-3  # Convert ms to s
            signal_rho = raw_data[rho_key]

            # Get unified time array (from all signals)
            unified_time = self._get_unified_time(shot, raw_data)

            # Get common rho_tor_norm grid
            rho_tor_norm_full = self._compose_rho_tor_norm(shot, raw_data)
            rho_tor_norm = rho_tor_norm_full[0]

            # Interpolate to common grid
            # Shape: (n_time, n_rho)
            result = np.zeros((len(unified_time), len(rho_tor_norm)))

            for i_time, time0 in enumerate(unified_time):
                # Find nearest time index in this signal's native time array
                time_index = np.argmin(np.abs(signal_time - time0))

                # Get data at this time and apply unit conversion
                data_at_time = density_raw[time_index, :] * unit_factor

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

        For OMFIT_PROFS (d3d.py:1570):
            query["electrons.temperature"] = "t_e"  # -> \\TOP.T_E
            # Data already on common [time, rho] grid, no unit conversion needed

        For ZIPFIT (d3d.py:1667, 1689, 1693, 1710):
            query["electrons.temperature"] = "\\TOP.PROFILES.ETEMPFIT"
            data[entry] *= 1E3  # in [eV]
            # Interpolate to common grid

        Returns:
            2D array of shape (n_time, n_rho) with electron temperature in eV
        """
        # Get raw data using helper method
        data_key = self._get_requirement_key('temperature', shot, dim=None)
        temperature_raw = raw_data[data_key]

        # Get unit conversion factor (1e3 for ZIPFIT, 1.0 for OMFIT_PROFS)
        unit_factor = self._get_unit_conversion('temperature')

        if self.is_omfit_profs:
            # For OMFIT_PROFS, data is already on common [time, rho] grid
            rho_key = Requirement('\\TOP.rho', self._get_pulse_id(shot), self.profiles_tree).as_key()
            rho_2d = raw_data[rho_key]

            # Apply unit conversion and rho masking per time slice
            result = []
            for i_time in range(temperature_raw.shape[0]):
                mask = rho_2d[i_time, :] <= 1.0
                result.append(temperature_raw[i_time, mask] * unit_factor)

            return np.array(result)
        else:
            # For ZIPFIT, interpolate to common grid
            # Get time and rho dimensions for this signal
            time_key = self._get_requirement_key('temperature', shot, dim=1)
            rho_key = self._get_requirement_key('temperature', shot, dim=0)
            signal_time = raw_data[time_key] * 1e-3  # Convert ms to s
            signal_rho = raw_data[rho_key]

            # Get unified time array (from all signals)
            unified_time = self._get_unified_time(shot, raw_data)

            # Get common rho_tor_norm grid
            rho_tor_norm_full = self._compose_rho_tor_norm(shot, raw_data)
            rho_tor_norm = rho_tor_norm_full[0]

            # Interpolate to common grid
            # Shape: (n_time, n_rho)
            result = np.zeros((len(unified_time), len(rho_tor_norm)))

            for i_time, time0 in enumerate(unified_time):
                # Find nearest time index in this signal's native time array
                time_index = np.argmin(np.abs(signal_time - time0))

                # Get data at this time and apply unit conversion
                data_at_time = temperature_raw[time_index, :] * unit_factor

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

        For OMFIT_PROFS (d3d.py:1589):
            data['time'] = dim_info.dim_of(1) * 1.e-3
            # All data on common time grid

        For ZIPFIT (d3d.py:1693):
            time = np.unique(np.concatenate([data[entry] for entry in query.keys()
                if entry.startswith("time__") ...]))

        Returns:
            1D array of unique time points in seconds
        """
        if self.is_omfit_profs:
            # For OMFIT_PROFS, time comes from dim_of(\TOP.N_E, 1)
            time_key = Requirement('dim_of(\\TOP.N_E,1)', self._get_pulse_id(shot), self.profiles_tree).as_key()
            unified_time = raw_data[time_key] * 1e-3  # Convert ms to s
            return unified_time
        else:
            # For ZIPFIT, collect time dimensions from all signals
            time_arrays = []

            # Collect time dimensions from all signals using helper method
            time_keys = [
                self._get_requirement_key('density', shot, dim=1),
                self._get_requirement_key('temperature', shot, dim=1),
                self._get_requirement_key('ion_temperature', shot, dim=1),
                self._get_requirement_key('carbon_density', shot, dim=1),
                self._get_requirement_key('carbon_rotation', shot, dim=1),
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
        Compose ion temperature (both D and C use same ITEMPFIT/T_D data).

        For OMFIT_PROFS (d3d.py:1570):
            query["ion[0].temperature"] = "t_d"  # -> \\TOP.T_D
            query["ion[1].temperature"] = "t_d"  # -> \\TOP.T_D
            # Data already on common grid

        For ZIPFIT (d3d.py:1669-1670, 1689):
            query["ion[0].temperature"] = "\\TOP.PROFILES.ITEMPFIT"
            query["ion[1].temperature"] = "\\TOP.PROFILES.ITEMPFIT"
            data[entry] *= 1E3  # in [eV]

        Returns:
            2D array of shape (n_time, n_rho) with ion temperature in eV
        """
        # Get raw data using helper method
        data_key = self._get_requirement_key('ion_temperature', shot, dim=None)
        ion_temp_raw = raw_data[data_key]

        # Get unit conversion factor (1e3 for ZIPFIT, 1.0 for OMFIT_PROFS)
        unit_factor = self._get_unit_conversion('temperature')

        if self.is_omfit_profs:
            # For OMFIT_PROFS, data is already on common [time, rho] grid
            rho_key = Requirement('\\TOP.rho', self._get_pulse_id(shot), self.profiles_tree).as_key()
            rho_2d = raw_data[rho_key]

            # Apply unit conversion and rho masking per time slice
            result = []
            for i_time in range(ion_temp_raw.shape[0]):
                mask = rho_2d[i_time, :] <= 1.0
                result.append(ion_temp_raw[i_time, mask] * unit_factor)

            return np.array(result)
        else:
            # For ZIPFIT, interpolate to common grid
            # Get time and rho dimensions for this signal
            time_key = self._get_requirement_key('ion_temperature', shot, dim=1)
            rho_key = self._get_requirement_key('ion_temperature', shot, dim=0)
            signal_time = raw_data[time_key] * 1e-3  # Convert ms to s
            signal_rho = raw_data[rho_key]

            # Get unified time array (from all signals)
            unified_time = self._get_unified_time(shot, raw_data)

            # Get common rho_tor_norm grid
            rho_tor_norm_full = self._compose_rho_tor_norm(shot, raw_data)
            rho_tor_norm = rho_tor_norm_full[0]

            # Interpolate to common grid
            # Shape: (n_time, n_rho)
            result = np.zeros((len(unified_time), len(rho_tor_norm)))

            for i_time, time0 in enumerate(unified_time):
                # Find nearest time index in this signal's native time array
                time_index = np.argmin(np.abs(signal_time - time0))

                # Get data at this time and apply unit conversion
                data_at_time = ion_temp_raw[time_index, :] * unit_factor

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

        For OMFIT_PROFS (d3d.py:1570):
            query["ion[1].density_thermal"] = "n_c"  # -> \\TOP.N_C
            # Data already on common grid

        For ZIPFIT (d3d.py:1668, 1687):
            query["ion[1].density_thermal"] = "\\TOP.PROFILES.ZDENSFIT"
            data[entry] *= 1E19  # in [m^-3]

        Returns:
            2D array of shape (n_time, n_rho) with carbon density in m^-3
        """
        # Get raw data using helper method
        data_key = self._get_requirement_key('carbon_density', shot, dim=None)
        carbon_density_raw = raw_data[data_key]

        # Get unit conversion factor (1e19 for ZIPFIT, 1.0 for OMFIT_PROFS)
        unit_factor = self._get_unit_conversion('density')

        if self.is_omfit_profs:
            # For OMFIT_PROFS, data is already on common [time, rho] grid
            rho_key = Requirement('\\TOP.rho', self._get_pulse_id(shot), self.profiles_tree).as_key()
            rho_2d = raw_data[rho_key]

            # Apply unit conversion and rho masking per time slice
            result = []
            for i_time in range(carbon_density_raw.shape[0]):
                mask = rho_2d[i_time, :] <= 1.0
                result.append(carbon_density_raw[i_time, mask] * unit_factor)

            return np.array(result)
        else:
            # For ZIPFIT, interpolate to common grid
            # Get time and rho dimensions for this signal
            time_key = self._get_requirement_key('carbon_density', shot, dim=1)
            rho_key = self._get_requirement_key('carbon_density', shot, dim=0)
            signal_time = raw_data[time_key] * 1e-3  # Convert ms to s
            signal_rho = raw_data[rho_key]

            # Get unified time array (from all signals)
            unified_time = self._get_unified_time(shot, raw_data)

            # Get common rho_tor_norm grid
            rho_tor_norm_full = self._compose_rho_tor_norm(shot, raw_data)
            rho_tor_norm = rho_tor_norm_full[0]

            # Interpolate to common grid
            # Shape: (n_time, n_rho)
            result = np.zeros((len(unified_time), len(rho_tor_norm)))

            for i_time, time0 in enumerate(unified_time):
                # Find nearest time index in this signal's native time array
                time_index = np.argmin(np.abs(signal_time - time0))

                # Get data at this time and apply unit conversion
                data_at_time = carbon_density_raw[time_index, :] * unit_factor

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

        For OMFIT_PROFS (d3d.py:1570):
            query["ion[1].rotation_frequency_tor"] = "v_tor_c"  # -> \\TOP.V_TOR_C
            # Data already on common grid

        For ZIPFIT (d3d.py:1671, 1691):
            query["ion[1].rotation_frequency_tor"] = "\\TOP.PROFILES.TROTFIT"
            data[entry] *= 1E3  # in [rad/s]

        Returns:
            2D array of shape (n_time, n_rho) with rotation frequency in rad/s
        """
        # Get raw data using helper method
        data_key = self._get_requirement_key('carbon_rotation', shot, dim=None)
        carbon_rotation_raw = raw_data[data_key]

        # Get unit conversion factor (1e3 for ZIPFIT, 1.0 for OMFIT_PROFS)
        unit_factor = self._get_unit_conversion('rotation')

        if self.is_omfit_profs:
            # For OMFIT_PROFS, data is already on common [time, rho] grid
            rho_key = Requirement('\\TOP.rho', self._get_pulse_id(shot), self.profiles_tree).as_key()
            rho_2d = raw_data[rho_key]

            # Apply unit conversion and rho masking per time slice
            result = []
            for i_time in range(carbon_rotation_raw.shape[0]):
                mask = rho_2d[i_time, :] <= 1.0
                result.append(carbon_rotation_raw[i_time, mask] * unit_factor)

            return np.array(result)
        else:
            # For ZIPFIT, interpolate to common grid
            # Get time and rho dimensions for this signal
            time_key = self._get_requirement_key('carbon_rotation', shot, dim=1)
            rho_key = self._get_requirement_key('carbon_rotation', shot, dim=0)
            signal_time = raw_data[time_key] * 1e-3  # Convert ms to s
            signal_rho = raw_data[rho_key]

            # Get unified time array (from all signals)
            unified_time = self._get_unified_time(shot, raw_data)

            # Get common rho_tor_norm grid
            rho_tor_norm_full = self._compose_rho_tor_norm(shot, raw_data)
            rho_tor_norm = rho_tor_norm_full[0]
            
            # Interpolate to common grid
            # Shape: (n_time, n_rho)
            result = np.zeros((len(unified_time), len(rho_tor_norm)))

            for i_time, time0 in enumerate(unified_time):
                # Find nearest time index in this signal's native time array
                time_index = np.argmin(np.abs(signal_time - time0))

                # Get data at this time and apply unit conversion
                data_at_time = carbon_rotation_raw[time_index, :] * unit_factor

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
        Compose deuterium density.

        For OMFIT_PROFS (d3d.py:1554):
            query["ion[0].density_thermal"] = "N_D"  # -> \\TOP.N_D
            # Direct from MDSplus, no quasineutrality

        For ZIPFIT (d3d.py:1712):
            # deuterium from quasineutrality
            ods[f"{sh}[{i_time}].ion[0].density_thermal"] =
                ods[f"{sh}[{i_time}].electrons.density_thermal"] -
                ods[f"{sh}[{i_time}].ion[1].density_thermal"] * 6

        Returns:
            2D array of shape (n_time, n_rho) with deuterium density in m^-3
        """
        if self.is_omfit_profs:
            # For OMFIT_PROFS, deuterium density comes directly from \TOP.N_D
            data_key = self._get_requirement_key('deuterium_density', shot, dim=None)
            deuterium_raw = raw_data[data_key]

            # Get unit conversion factor (1.0 for OMFIT_PROFS)
            unit_factor = self._get_unit_conversion('density')

            # Apply rho masking per time slice
            rho_key = Requirement('\\TOP.rho', self._get_pulse_id(shot), self.profiles_tree).as_key()
            rho_2d = raw_data[rho_key]

            result = []
            for i_time in range(deuterium_raw.shape[0]):
                mask = rho_2d[i_time, :] <= 1.0
                result.append(deuterium_raw[i_time, mask] * unit_factor)

            return np.array(result)
        else:
            # For ZIPFIT, use quasineutrality
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

        # Get profile time - use the already-composed time to avoid bypassing dependencies
        # Note: We depend on "core_profiles.time" so we need to compose it first
        profile_time = self.specs["core_profiles.time"].compose(shot, raw_data)

        # Interpolate v_loop to profile time
        v_loop = interp1d(
            vloop_time,
            vloop_data,
            bounds_error=False,
            fill_value=np.nan
        )(profile_time)

        return v_loop

    def _compose_omfit_error_field(self, shot: int, raw_data: Dict[str, Any],
                                   error_key_name: str, unit_type: str) -> np.ndarray:
        """
        Generic helper to compose OMFIT_PROFS error fields.

        All OMFIT_PROFS error fields follow the same pattern:
        1. Get error data from error_of(\\TOP.FIELD) (already in raw_data)
        2. Apply unit conversion (same as the main field)
        3. Apply rho masking per time slice

        Args:
            shot: Shot number
            raw_data: Dictionary of raw data
            error_key_name: Internal dependency key (e.g., 'core_profiles.profiles_1d._density_error')
            unit_type: Unit type for conversion ('density', 'temperature', 'rotation')

        Returns:
            2D array of shape (n_time, n_rho) with error values
        """
        # Extract the actual key from the dependency name
        # error_key_name is like "core_profiles.profiles_1d._density_error"
        # We need to get it from specs to get the proper Requirement
        spec = self.specs.get(error_key_name)
        if spec and spec.derive_requirements:
            reqs = spec.derive_requirements(shot, raw_data)
            if reqs:
                error_data_key = reqs[0].as_key()
                error_raw = raw_data[error_data_key]
            else:
                raise ValueError(f"No requirements for {error_key_name}")
        else:
            raise ValueError(f"Cannot find spec or requirements for {error_key_name}")

        # Get unit conversion factor
        unit_factor = self._get_unit_conversion(unit_type)

        # Get rho for masking
        rho_key = Requirement('\\TOP.rho', self._get_pulse_id(shot), self.profiles_tree).as_key()
        rho_2d = raw_data[rho_key]

        # Apply unit conversion and rho masking per time slice
        result = []
        for i_time in range(error_raw.shape[0]):
            mask = rho_2d[i_time, :] <= 1.0
            result.append(error_raw[i_time, mask] * unit_factor)

        return np.array(result)

    def _compose_density_error(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """Compose electrons.density_thermal_error_upper for OMFIT_PROFS."""
        return self._compose_omfit_error_field(
            shot, raw_data,
            "core_profiles.profiles_1d._density_error",
            "density"
        )

    def _compose_temperature_error(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """Compose electrons.temperature_error_upper for OMFIT_PROFS."""
        return self._compose_omfit_error_field(
            shot, raw_data,
            "core_profiles.profiles_1d._temperature_error",
            "temperature"
        )

    def _compose_ion_temperature_error(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """Compose ion[0] and ion[1] temperature_error_upper for OMFIT_PROFS."""
        return self._compose_omfit_error_field(
            shot, raw_data,
            "core_profiles.profiles_1d._ion_temperature_error",
            "temperature"
        )

    def _compose_deuterium_density_error(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """Compose ion[0].density_thermal_error_upper for OMFIT_PROFS."""
        return self._compose_omfit_error_field(
            shot, raw_data,
            "core_profiles.profiles_1d._deuterium_density_error",
            "density"
        )

    def _compose_carbon_density_error(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """Compose ion[1].density_thermal_error_upper for OMFIT_PROFS."""
        return self._compose_omfit_error_field(
            shot, raw_data,
            "core_profiles.profiles_1d._carbon_density_error",
            "density"
        )

    def _compose_carbon_temperature_error(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """Compose ion[1].temperature_error_upper for OMFIT_PROFS."""
        return self._compose_omfit_error_field(
            shot, raw_data,
            "core_profiles.profiles_1d._carbon_temperature_error",
            "temperature"
        )

    def _compose_e_field_radial(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compose e_field.radial for OMFIT_PROFS.

        OMAS d3d.py line 1569, 1645:
            query["e_field.radial"] = "ER_C"  # -> \\TOP.ER_C
            # No unit conversion, already in V/m
            ods[f"{sh}[{i_time}].e_field.radial"] = data[entry][i_time][mask[i_time]]

        Returns:
            2D array of shape (n_time, n_rho) with radial electric field in V/m
        """
        # Get e-field data
        e_field_key = Requirement('\\TOP.ER_C', self._get_pulse_id(shot), self.profiles_tree).as_key()
        e_field_raw = raw_data[e_field_key]

        # Get rho for masking
        rho_key = Requirement('\\TOP.rho', self._get_pulse_id(shot), self.profiles_tree).as_key()
        rho_2d = raw_data[rho_key]

        # Apply rho masking per time slice (no unit conversion needed)
        result = []
        for i_time in range(e_field_raw.shape[0]):
            mask = rho_2d[i_time, :] <= 1.0
            result.append(e_field_raw[i_time, mask])

        return np.array(result)

    # ============================================================
    # Fit field compose methods (OMFIT_PROFS only)
    # ============================================================

    def _compose_fit_measured(self, shot: int, raw_data: Dict[str, Any],
                             measured_key_name: str, unit_type: str) -> ak.Array:
        """
        Generic helper to compose *_fit.measured fields for OMFIT_PROFS.

        OMAS d3d.py lines 1615-1620:
            if "_fit.measured" in entry:
                data_mask = np.isfinite(data[entry][i_time])
                ods[f"{sh}[{i_time}]." + entry] = data[entry][i_time][data_mask]

        Args:
            shot: Shot number
            raw_data: Dictionary of raw data
            measured_key_name: Internal dependency key (e.g., 'core_profiles.profiles_1d._electron_density_fit_measured')
            unit_type: Unit type for conversion ('density', 'temperature')

        Returns:
            Ragged awkward array (n_time, var) with measured values, isfinite masked
        """
        # Get measured data from specs
        spec = self.specs.get(measured_key_name)
        if spec and spec.derive_requirements:
            reqs = spec.derive_requirements(shot, raw_data)
            if reqs:
                measured_data_key = reqs[0].as_key()
                measured_raw = raw_data[measured_data_key]
            else:
                raise ValueError(f"No requirements for {measured_key_name}")
        else:
            raise ValueError(f"Cannot find spec or requirements for {measured_key_name}")

        # Get unit conversion factor (OMFIT_PROFS should be 1.0, but keeping consistent)
        unit_factor = self._get_unit_conversion(unit_type)

        # Apply isfinite masking per time slice (not rho masking!)
        # This is different from regular fields which use rho <= 1.0
        result = []
        for i_time in range(measured_raw.shape[0]):
            mask = np.isfinite(measured_raw[i_time])
            result.append(measured_raw[i_time, mask] * unit_factor)

        return ak.Array(result)

    def _compose_fit_psi_norm(self, shot: int, raw_data: Dict[str, Any],
                             psi_norm_key_name: str, measured_key_name: str) -> ak.Array:
        """
        Generic helper to compose *_fit.psi_norm fields for OMFIT_PROFS.

        OMAS d3d.py lines 1641-1643:
            if "_fit.psi_norm" in entry:
                data_mask = np.isfinite(data[entry.replace("psi_norm","measured")][i_time])
                ods[f"{sh}[{i_time}]."+entry] = data[entry][i_time][data_mask]

        Args:
            shot: Shot number
            raw_data: Dictionary of raw data
            psi_norm_key_name: Internal dependency key for psi_norm data
            measured_key_name: Internal dependency key for measured data (to get mask)

        Returns:
            Ragged awkward array (n_time, var) with psi_norm values, isfinite masked
        """
        # Get psi_norm data
        spec = self.specs.get(psi_norm_key_name)
        if spec and spec.derive_requirements:
            reqs = spec.derive_requirements(shot, raw_data)
            if reqs:
                psi_norm_data_key = reqs[0].as_key()
                psi_norm_raw = raw_data[psi_norm_data_key]
            else:
                raise ValueError(f"No requirements for {psi_norm_key_name}")
        else:
            raise ValueError(f"Cannot find spec or requirements for {psi_norm_key_name}")

        # Get measured data to determine mask
        spec = self.specs.get(measured_key_name)
        if spec and spec.derive_requirements:
            reqs = spec.derive_requirements(shot, raw_data)
            if reqs:
                measured_data_key = reqs[0].as_key()
                measured_raw = raw_data[measured_data_key]
            else:
                raise ValueError(f"No requirements for {measured_key_name}")
        else:
            raise ValueError(f"Cannot find spec or requirements for {measured_key_name}")

        # Apply isfinite masking based on measured data
        result = []
        for i_time in range(measured_raw.shape[0]):
            mask = np.isfinite(measured_raw[i_time])
            result.append(psi_norm_raw[i_time, mask])

        return ak.Array(result)

    def _compose_fit_rho_tor_norm(self, shot: int, raw_data: Dict[str, Any],
                                  psi_norm_key_name: str, measured_key_name: str) -> ak.Array:
        """
        Generic helper to compose *_fit.rho_tor_norm fields for OMFIT_PROFS.

        OMAS d3d.py lines 1600-1604, 1615-1618:
            # Create spline for each time slice
            rho_spl.append(InterpolatedUnivariateSpline(psi_n, data["grid.rho_tor_norm"][i_time]))
            ...
            if "_fit.measured" in entry:
                data_mask = np.isfinite(data[entry][i_time])
                # Set rho_tor before we set anything else
                ods[f"{sh}[{i_time}]." + entry.replace("measured", "rho_tor_norm")] =
                    rho_spl[i_time](data[entry.replace("measured", "psi_norm")][i_time][data_mask])

        Args:
            shot: Shot number
            raw_data: Dictionary of raw data
            psi_norm_key_name: Internal dependency key for psi_norm data
            measured_key_name: Internal dependency key for measured data (to get mask)

        Returns:
            Ragged awkward array (n_time, var) with rho_tor_norm values computed via spline
        """
        # Get psi_norm for the fit field
        spec = self.specs.get(psi_norm_key_name)
        if spec and spec.derive_requirements:
            reqs = spec.derive_requirements(shot, raw_data)
            if reqs:
                fit_psi_norm_key = reqs[0].as_key()
                fit_psi_norm = raw_data[fit_psi_norm_key]
            else:
                raise ValueError(f"No requirements for {psi_norm_key_name}")
        else:
            raise ValueError(f"Cannot find spec or requirements for {psi_norm_key_name}")

        # Get measured data to determine mask
        spec = self.specs.get(measured_key_name)
        if spec and spec.derive_requirements:
            reqs = spec.derive_requirements(shot, raw_data)
            if reqs:
                measured_data_key = reqs[0].as_key()
                measured_raw = raw_data[measured_data_key]
            else:
                raise ValueError(f"No requirements for {measured_key_name}")
        else:
            raise ValueError(f"Cannot find spec or requirements for {measured_key_name}")

        # Get psi_n for the main grid (from dim_of(\TOP.N_E, 0))
        grid_psi_key = Requirement('dim_of(\\TOP.N_E,0)', self._get_pulse_id(shot), self.profiles_tree).as_key()
        grid_psi_n = raw_data[grid_psi_key]

        # Get rho_tor_norm for the main grid
        rho_key = Requirement('\\TOP.rho', self._get_pulse_id(shot), self.profiles_tree).as_key()
        rho_2d = raw_data[rho_key]

        # Create spline for each time slice and interpolate
        result = []
        for i_time in range(measured_raw.shape[0]):
            # Create spline from grid psi_n to grid rho_tor_norm
            rho_spl = InterpolatedUnivariateSpline(grid_psi_n, rho_2d[i_time, :])

            # Apply isfinite mask to fit psi_norm
            mask = np.isfinite(measured_raw[i_time])
            fit_psi_masked = fit_psi_norm[i_time, mask]

            # Interpolate to get rho_tor_norm for the fit field
            result.append(rho_spl(fit_psi_masked))

        return ak.Array(result)

    def _compose_fit_error(self, shot: int, raw_data: Dict[str, Any],
                          error_key_name: str, measured_key_name: str, unit_type: str) -> ak.Array:
        """
        Generic helper to compose *_fit.measured_error_upper fields for OMFIT_PROFS.

        OMAS d3d.py lines 1615-1621:
            if "_fit.measured" in entry:
                data_mask = np.isfinite(data[entry][i_time])
                ...
                ods[f"{sh}[{i_time}]." + entry + "_error_upper"] = data[entry + "_error_upper"][i_time][data_mask]

        Args:
            shot: Shot number
            raw_data: Dictionary of raw data
            error_key_name: Internal dependency key for error data
            measured_key_name: Internal dependency key for measured data (to get mask)
            unit_type: Unit type for conversion ('density', 'temperature')

        Returns:
            Ragged awkward array (n_time, var) with error values, isfinite masked
        """
        # Get error data
        spec = self.specs.get(error_key_name)
        if spec and spec.derive_requirements:
            reqs = spec.derive_requirements(shot, raw_data)
            if reqs:
                error_data_key = reqs[0].as_key()
                error_raw = raw_data[error_data_key]
            else:
                raise ValueError(f"No requirements for {error_key_name}")
        else:
            raise ValueError(f"Cannot find spec or requirements for {error_key_name}")

        # Get measured data to determine mask
        spec = self.specs.get(measured_key_name)
        if spec and spec.derive_requirements:
            reqs = spec.derive_requirements(shot, raw_data)
            if reqs:
                measured_data_key = reqs[0].as_key()
                measured_raw = raw_data[measured_data_key]
            else:
                raise ValueError(f"No requirements for {measured_key_name}")
        else:
            raise ValueError(f"Cannot find spec or requirements for {measured_key_name}")

        # Get unit conversion factor
        unit_factor = self._get_unit_conversion(unit_type)

        # Apply isfinite masking based on measured data
        result = []
        for i_time in range(measured_raw.shape[0]):
            mask = np.isfinite(measured_raw[i_time])
            result.append(error_raw[i_time, mask] * unit_factor)

        return ak.Array(result)

    # Electron density_fit compose methods
    def _compose_electron_density_fit_measured(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose electrons.density_fit.measured for OMFIT_PROFS."""
        return self._compose_fit_measured(
            shot, raw_data,
            "core_profiles.profiles_1d._electron_density_fit_measured",
            "density"
        )

    def _compose_electron_density_fit_psi_norm(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose electrons.density_fit.psi_norm for OMFIT_PROFS."""
        return self._compose_fit_psi_norm(
            shot, raw_data,
            "core_profiles.profiles_1d._electron_density_fit_psi_norm",
            "core_profiles.profiles_1d._electron_density_fit_measured"
        )

    def _compose_electron_density_fit_rho_tor_norm(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose electrons.density_fit.rho_tor_norm for OMFIT_PROFS."""
        return self._compose_fit_rho_tor_norm(
            shot, raw_data,
            "core_profiles.profiles_1d._electron_density_fit_psi_norm",
            "core_profiles.profiles_1d._electron_density_fit_measured"
        )

    def _compose_electron_density_fit_error(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose electrons.density_fit.measured_error_upper for OMFIT_PROFS."""
        return self._compose_fit_error(
            shot, raw_data,
            "core_profiles.profiles_1d._electron_density_fit_error",
            "core_profiles.profiles_1d._electron_density_fit_measured",
            "density"
        )

    # Electron temperature_fit compose methods
    def _compose_electron_temperature_fit_measured(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose electrons.temperature_fit.measured for OMFIT_PROFS."""
        return self._compose_fit_measured(
            shot, raw_data,
            "core_profiles.profiles_1d._electron_temperature_fit_measured",
            "temperature"
        )

    def _compose_electron_temperature_fit_psi_norm(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose electrons.temperature_fit.psi_norm for OMFIT_PROFS."""
        return self._compose_fit_psi_norm(
            shot, raw_data,
            "core_profiles.profiles_1d._electron_temperature_fit_psi_norm",
            "core_profiles.profiles_1d._electron_temperature_fit_measured"
        )

    def _compose_electron_temperature_fit_rho_tor_norm(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose electrons.temperature_fit.rho_tor_norm for OMFIT_PROFS."""
        return self._compose_fit_rho_tor_norm(
            shot, raw_data,
            "core_profiles.profiles_1d._electron_temperature_fit_psi_norm",
            "core_profiles.profiles_1d._electron_temperature_fit_measured"
        )

    def _compose_electron_temperature_fit_error(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose electrons.temperature_fit.measured_error_upper for OMFIT_PROFS."""
        return self._compose_fit_error(
            shot, raw_data,
            "core_profiles.profiles_1d._electron_temperature_fit_error",
            "core_profiles.profiles_1d._electron_temperature_fit_measured",
            "temperature"
        )

    # Carbon density_fit compose methods
    def _compose_carbon_density_fit_measured(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose ion[1].density_fit.measured for OMFIT_PROFS."""
        return self._compose_fit_measured(
            shot, raw_data,
            "core_profiles.profiles_1d._carbon_density_fit_measured",
            "density"
        )

    def _compose_carbon_density_fit_psi_norm(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose ion[1].density_fit.psi_norm for OMFIT_PROFS."""
        return self._compose_fit_psi_norm(
            shot, raw_data,
            "core_profiles.profiles_1d._carbon_density_fit_psi_norm",
            "core_profiles.profiles_1d._carbon_density_fit_measured"
        )

    def _compose_carbon_density_fit_rho_tor_norm(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose ion[1].density_fit.rho_tor_norm for OMFIT_PROFS."""
        return self._compose_fit_rho_tor_norm(
            shot, raw_data,
            "core_profiles.profiles_1d._carbon_density_fit_psi_norm",
            "core_profiles.profiles_1d._carbon_density_fit_measured"
        )

    def _compose_carbon_density_fit_error(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose ion[1].density_fit.measured_error_upper for OMFIT_PROFS."""
        return self._compose_fit_error(
            shot, raw_data,
            "core_profiles.profiles_1d._carbon_density_fit_error",
            "core_profiles.profiles_1d._carbon_density_fit_measured",
            "density"
        )

    # Carbon temperature_fit compose methods
    def _compose_carbon_temperature_fit_measured(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose ion[1].temperature_fit.measured for OMFIT_PROFS."""
        return self._compose_fit_measured(
            shot, raw_data,
            "core_profiles.profiles_1d._carbon_temperature_fit_measured",
            "temperature"
        )

    def _compose_carbon_temperature_fit_psi_norm(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose ion[1].temperature_fit.psi_norm for OMFIT_PROFS."""
        return self._compose_fit_psi_norm(
            shot, raw_data,
            "core_profiles.profiles_1d._carbon_temperature_fit_psi_norm",
            "core_profiles.profiles_1d._carbon_temperature_fit_measured"
        )

    def _compose_carbon_temperature_fit_rho_tor_norm(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose ion[1].temperature_fit.rho_tor_norm for OMFIT_PROFS."""
        return self._compose_fit_rho_tor_norm(
            shot, raw_data,
            "core_profiles.profiles_1d._carbon_temperature_fit_psi_norm",
            "core_profiles.profiles_1d._carbon_temperature_fit_measured"
        )

    def _compose_carbon_temperature_fit_error(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """Compose ion[1].temperature_fit.measured_error_upper for OMFIT_PROFS."""
        return self._compose_fit_error(
            shot, raw_data,
            "core_profiles.profiles_1d._carbon_temperature_fit_error",
            "core_profiles.profiles_1d._carbon_temperature_fit_measured",
            "temperature"
        )
