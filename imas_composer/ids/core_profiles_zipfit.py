"""
Core Profiles IDS Mapping for DIII-D ZIPFIT Tree

This mapper handles ZIPFIT-specific data from the ZIPFIT01 tree.
See OMAS: omas/machine_mappings/d3d.py::core_profiles_profile_1d (lines 1664-1713)
"""

from typing import Dict, Any
import numpy as np
import awkward as ak
from scipy.interpolate import interp1d

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class CoreProfilesZipfitMapper(IDSMapper):
    """Maps DIII-D ZIPFIT tree data to IMAS core_profiles IDS."""

    DOCS_PATH = "core_profiles_zipfit.yaml"
    CONFIG_PATH = "core_profiles_zipfit.yaml"

    # Maximum time difference (seconds) to consider a ZIPFIT time as matching a GTIME point
    _TIME_MATCH_TOL = 1e-3

    def __init__(self, zipfit_tree: str = 'ZIPFIT01', **kwargs):
        """
        Initialize CoreProfilesZIPFITMapper.

        Args:
            zipfit_tree: ZIPFIT tree to use (default: 'ZIPFIT01')
        """
        self.profiles_tree = zipfit_tree

        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Ion species list from YAML (defines ordering in ak.Array ion dimension)
        self.ions = self.config.get('ions', [])

        # Build IDS specs
        self._build_specs()

    def _get_pulse_id(self, shot: int) -> int:
        """
        Get the pulse ID to use for MDSplus queries.

        For ZIPFIT, returns shot unchanged.

        Args:
            shot: Base shot number

        Returns:
            Pulse ID for MDSplus query
        """
        return shot

    def _get_mds_path(self, field_type: str) -> str:
        """
        Get the MDSplus TDI path for a given field type.

        Args:
            field_type: One of 'density', 'temperature', 'ion_temperature',
                       'carbon_density', 'carbon_rotation'

        Returns:
            MDSplus path string
        """
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

        ZIPFIT data needs conversion (1E19 for density, 1E3 for temp/rotation).

        Args:
            field_type: One of 'density', 'temperature', 'rotation'

        Returns:
            Conversion factor to multiply raw data by
        """
        conversion_map = {
            'density': 1e19,      # Convert to m^-3
            'temperature': 1e3,   # Convert to eV
            'rotation': 1e3       # Convert to rad/s
        }
        return conversion_map.get(field_type, 1.0)

    def _create_profile_field_spec(self, field_name: str, field_type: str, dim: int = None) -> IDSEntrySpec:
        """
        Helper to create IDSEntrySpec for profile fields.

        For ZIPFIT: All fields (data, time, rho) use DIRECT stage.

        Args:
            field_name: Internal field name (e.g., '_density_data', '_density_time')
            field_type: Field type for MDSplus path lookup (e.g., 'density', 'temperature')
            dim: Dimension index for dim_of() call, or None for data field

        Returns:
            IDSEntrySpec configured for ZIPFIT tree
        """
        ids_path = f"core_profiles.profiles_1d.{field_name}"
        mds_path = self._get_mds_path(field_type)

        if dim is not None:
            mds_path = f'dim_of({mds_path},{dim})'

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
        # Internal dependencies - DIRECT stage
        # ============================================================

        # Common time basis from EFIT01 (seconds, already divided by 1000 in TDI)
        self.specs["core_profiles._gtime"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('\\EFIT01::TOP.RESULTS.GEQDSK.GTIME/1000.', 0, 'EFIT01')
            ],
            ids_path="core_profiles._gtime",
            docs_file=self.DOCS_PATH
        )

        # Electron density - data, time, and rho dimensions
        self.specs["core_profiles.profiles_1d._density_data"] = self._create_profile_field_spec(
            '_density_data', 'density', dim=None)
        self.specs["core_profiles.profiles_1d._density_time"] = self._create_profile_field_spec(
            '_density_time', 'density', dim=1)
        self.specs["core_profiles.profiles_1d._density_rho"] = self._create_profile_field_spec(
            '_density_rho', 'density', dim=0)

        # Electron temperature - data, time, and rho dimensions
        self.specs["core_profiles.profiles_1d._temperature_data"] = self._create_profile_field_spec(
            '_temperature_data', 'temperature', dim=None)
        self.specs["core_profiles.profiles_1d._temperature_time"] = self._create_profile_field_spec(
            '_temperature_time', 'temperature', dim=1)
        self.specs["core_profiles.profiles_1d._temperature_rho"] = self._create_profile_field_spec(
            '_temperature_rho', 'temperature', dim=0)

        # Ion temperature (both D and C use ITEMPFIT) - data, time, and rho dimensions
        self.specs["core_profiles.profiles_1d._ion_temperature_data"] = self._create_profile_field_spec(
            '_ion_temperature_data', 'ion_temperature', dim=None)
        self.specs["core_profiles.profiles_1d._ion_temperature_time"] = self._create_profile_field_spec(
            '_ion_temperature_time', 'ion_temperature', dim=1)
        self.specs["core_profiles.profiles_1d._ion_temperature_rho"] = self._create_profile_field_spec(
            '_ion_temperature_rho', 'ion_temperature', dim=0)

        # Carbon density - data, time, and rho dimensions
        self.specs["core_profiles.profiles_1d._carbon_density_data"] = self._create_profile_field_spec(
            '_carbon_density_data', 'carbon_density', dim=None)
        self.specs["core_profiles.profiles_1d._carbon_density_time"] = self._create_profile_field_spec(
            '_carbon_density_time', 'carbon_density', dim=1)
        self.specs["core_profiles.profiles_1d._carbon_density_rho"] = self._create_profile_field_spec(
            '_carbon_density_rho', 'carbon_density', dim=0)

        # Carbon rotation - data, time, and rho dimensions
        self.specs["core_profiles.profiles_1d._carbon_rotation_data"] = self._create_profile_field_spec(
            '_carbon_rotation_data', 'carbon_rotation', dim=None)
        self.specs["core_profiles.profiles_1d._carbon_rotation_time"] = self._create_profile_field_spec(
            '_carbon_rotation_time', 'carbon_rotation', dim=1)
        self.specs["core_profiles.profiles_1d._carbon_rotation_rho"] = self._create_profile_field_spec(
            '_carbon_rotation_rho', 'carbon_rotation', dim=0)

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

        # Grid: rho_tor_norm — density's rho axis is representative for all signals
        self.specs["core_profiles.profiles_1d.grid.rho_tor_norm"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles._gtime",
                "core_profiles.profiles_1d._density_rho",
            ],
            compose=self._compose_rho_tor_norm,
            ids_path="core_profiles.profiles_1d.grid.rho_tor_norm",
            docs_file=self.DOCS_PATH
        )

        # Electrons: density_thermal
        self.specs["core_profiles.profiles_1d.electrons.density"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles._gtime",
                "core_profiles.profiles_1d._density_data",
                "core_profiles.profiles_1d._density_time",
                "core_profiles.profiles_1d._density_rho",
            ],
            compose=self._compose_density_thermal,
            ids_path="core_profiles.profiles_1d.electrons.density",
            docs_file=self.DOCS_PATH
        )

        # Electrons: temperature
        self.specs["core_profiles.profiles_1d.electrons.temperature"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles._gtime",
                "core_profiles.profiles_1d._temperature_data",
                "core_profiles.profiles_1d._temperature_time",
                "core_profiles.profiles_1d._temperature_rho",
            ],
            compose=self._compose_temperature,
            ids_path="core_profiles.profiles_1d.electrons.temperature",
            docs_file=self.DOCS_PATH
        )

        # ============================================================
        # Ion fields (unified ion dimension - ak.Array [time, ion, rho])
        # Ion ordering follows the `ions:` YAML section (D=index 0, C=index 1)
        # ============================================================

        # ion.temperature: both D and C use ITEMPFIT → same data, stacked
        self.specs["core_profiles.profiles_1d.ion.temperature"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles._gtime",
                "core_profiles.profiles_1d._ion_temperature_data",
                "core_profiles.profiles_1d._ion_temperature_time",
                "core_profiles.profiles_1d._ion_temperature_rho",
            ],
            compose=self._compose_all_ion_temperature,
            ids_path="core_profiles.profiles_1d.ion.temperature",
            docs_file=self.DOCS_PATH
        )

        # ion.density_thermal: D from quasineutrality (n_e - 6*n_C), C from ZDENSFIT
        self.specs["core_profiles.profiles_1d.ion.density_thermal"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles._gtime",
                "core_profiles.profiles_1d._density_data",
                "core_profiles.profiles_1d._density_time",
                "core_profiles.profiles_1d._density_rho",
                "core_profiles.profiles_1d._carbon_density_data",
                "core_profiles.profiles_1d._carbon_density_time",
                "core_profiles.profiles_1d._carbon_density_rho",
            ],
            compose=self._compose_all_ion_density_thermal,
            ids_path="core_profiles.profiles_1d.ion.density_thermal",
            docs_file=self.DOCS_PATH
        )

        # ion.rotation_frequency_tor: D = empty, C from TROTFIT
        self.specs["core_profiles.profiles_1d.ion.rotation_frequency_tor"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "core_profiles._gtime",
                "core_profiles.profiles_1d._carbon_rotation_data",
                "core_profiles.profiles_1d._carbon_rotation_time",
                "core_profiles.profiles_1d._carbon_rotation_rho",
            ],
            compose=self._compose_all_ion_rotation,
            ids_path="core_profiles.profiles_1d.ion.rotation_frequency_tor",
            docs_file=self.DOCS_PATH
        )

        # time: single time basis from EFIT01 GTIME
        time_deps = ["core_profiles._gtime"]

        # Static metadata: label, z_n, a — 1D arrays indexed by ion (from YAML ions: list)
        self.specs["core_profiles.profiles_1d.ion.label"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=time_deps,
            compose=self._compose_all_ion_label,
            ids_path="core_profiles.profiles_1d.ion.label",
            docs_file=self.DOCS_PATH
        )

        self.specs["core_profiles.profiles_1d.ion.element.z_n"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=time_deps,
            compose=self._compose_all_ion_element_z_n,
            ids_path="core_profiles.profiles_1d.ion.element.z_n",
            docs_file=self.DOCS_PATH
        )

        self.specs["core_profiles.profiles_1d.ion.element.a"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=time_deps,
            compose=self._compose_all_ion_element_a,
            ids_path="core_profiles.profiles_1d.ion.element.a",
            docs_file=self.DOCS_PATH
        )

        # ============================================================
        # Time and metadata fields
        # ============================================================

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

    def _compose_all_ion_temperature(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """
        Compose ion.temperature for all ions: jagged (n_gtime, n_ion, n_rho) ak.Array.

        Both D and C use ITEMPFIT (same underlying data). At GTIME points where
        ITEMPFIT has no data the slot is empty ([]).
        """
        t_ion_jagged = self._compose_ion_temperature(shot, raw_data)
        result = []
        for t_ion in t_ion_jagged:
            if len(t_ion) > 0:
                t_arr = np.asarray(t_ion)
                result.append([t_arr for _ in self.ions])
            else:
                result.append([])
        return ak.Array(result)

    def _compose_all_ion_density_thermal(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """
        Compose ion.density_thermal for all ions: jagged (n_gtime, n_ion, n_rho) ak.Array.

        D from quasineutrality (n_e - 6*n_C), C from ZDENSFIT. A slot is empty
        unless both n_e and n_C data are present at that GTIME point.
        """
        n_D_jagged = self._compose_deuterium_density(shot, raw_data)
        n_C_jagged = self._compose_carbon_density(shot, raw_data)
        result = []
        for n_D, n_C in zip(n_D_jagged, n_C_jagged):
            if len(n_D) > 0 and len(n_C) > 0:
                result.append([np.asarray(n_D), np.asarray(n_C)])
            else:
                result.append([])
        return ak.Array(result)

    def _compose_all_ion_rotation(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """
        Compose ion.rotation_frequency_tor for all ions: jagged (n_gtime, n_ion, n_rho) ak.Array.

        C from TROTFIT, D has no measurement (empty inner array). A slot is empty
        unless TROTFIT has data at that GTIME point.
        """
        c_rotation_jagged = self._compose_carbon_rotation(shot, raw_data)
        result = []
        for c_rot in c_rotation_jagged:
            if len(c_rot) > 0:
                result.append([np.array([]), np.asarray(c_rot)])
            else:
                result.append([])
        return ak.Array(result)

    def _compose_all_ion_label(self, shot: int, raw_data: Dict[str, Any]) -> list:
        """Compose array of ion labels in YAML order (n_time, n_ion)."""
        unified_time = self._get_unified_time(shot, raw_data)
        return np.tile([ion['label'] for ion in self.ions], (len(unified_time), 1))

    def _compose_all_ion_element_z_n(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compose array of atomic numbers in YAML order (n_time, n_ion, n_element).

        n_element is assumed to be 1 for all plasma species.
        """
        unified_time = self._get_unified_time(shot, raw_data)
        return np.tile([ion['z_n'] for ion in self.ions], (len(unified_time), 1))[:, :, np.newaxis]

    def _compose_all_ion_element_a(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Compose array of atomic masses in YAML order (n_time, n_ion, n_element).

        n_element is assumed to be 1 for all plasma species.
        """
        unified_time = self._get_unified_time(shot, raw_data)
        return np.tile([ion['a'] for ion in self.ions], (len(unified_time), 1))[:, :, np.newaxis]

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

    def _get_signal_indices_for_gtime(
        self, gtime: np.ndarray, signal_time: np.ndarray
    ) -> np.ndarray:
        """
        For each GTIME point return the nearest signal time index, or -1 if
        the nearest point is farther than _TIME_MATCH_TOL.

        Args:
            gtime: GTIME array in seconds (n_gtime,)
            signal_time: Signal native time array in seconds (n_signal,)

        Returns:
            Integer array (n_gtime,); -1 means no match at that GTIME slot.
        """
        indices = np.full(len(gtime), -1, dtype=int)
        for i, t in enumerate(gtime):
            j = int(np.argmin(np.abs(signal_time - t)))
            if np.abs(signal_time[j] - t) <= self._TIME_MATCH_TOL:
                indices[i] = j
        return indices

    def _compose_profile_field(
        self,
        shot: int,
        raw_data: Dict[str, Any],
        field_type: str,
        unit_category: str,
    ) -> ak.Array:
        """
        Compose a single profile field as a jagged ak.Array over GTIME.

        At each GTIME point the signal's native time is checked against
        _TIME_MATCH_TOL. Matching slots receive the signal's rho-filtered data
        (values at rho <= 1.0); non-matching slots are empty ([]).

        Args:
            field_type: Key for _get_mds_path / _get_requirement_key
            unit_category: Key for _get_unit_conversion

        Returns:
            ak.Array of shape (n_gtime,) where each entry is either a 1-D
            array of length n_rho or an empty array.
        """
        data_key = self._get_requirement_key(field_type, shot, dim=None)
        time_key = self._get_requirement_key(field_type, shot, dim=1)
        rho_key  = self._get_requirement_key(field_type, shot, dim=0)

        raw         = raw_data[data_key]
        signal_time = raw_data[time_key] * 1e-3  # ms → s
        signal_rho  = raw_data[rho_key]
        unit_factor = self._get_unit_conversion(unit_category)

        rho_mask    = signal_rho <= 1.0
        gtime       = self._get_unified_time(shot, raw_data)
        sig_indices = self._get_signal_indices_for_gtime(gtime, signal_time)

        result = []
        for sig_idx in sig_indices:
            if sig_idx >= 0:
                result.append(raw[sig_idx, rho_mask] * unit_factor)
            else:
                result.append(np.array([]))
        return ak.Array(result)

    def _compose_rho_tor_norm(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """
        Compose grid.rho_tor_norm broadcast to all GTIME points.

        All ZIPFIT signals share the same rho grid; density's rho axis is used
        as the representative. Values above 1.0 are excluded.

        Returns:
            ak.Array of shape (n_gtime, n_rho)
        """
        rho_key = self._get_requirement_key('density', shot, dim=0)
        rho = raw_data[rho_key]
        rho = rho[rho <= 1.0]
        gtime = self._get_unified_time(shot, raw_data)
        return ak.Array([rho for _ in gtime])

    def _compose_density_thermal(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """
        Compose electrons.density as a jagged ak.Array (m^-3).

        Slots in GTIME where EDENSFIT has no matching time are empty.
        """
        return self._compose_profile_field(shot, raw_data, 'density', 'density')

    def _compose_temperature(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """
        Compose electrons.temperature as a jagged ak.Array (eV).

        Slots in GTIME where ETEMPFIT has no matching time are empty.
        """
        return self._compose_profile_field(shot, raw_data, 'temperature', 'temperature')

    def _get_unified_time(self, shot: int, raw_data: Dict[str, Any]) -> np.ndarray:
        """
        Return the common time basis from EFIT01 GTIME (already in seconds).

        GTIME is fetched as \\EFIT01::TOP.RESULTS.GEQDSK.GTIME/1000. so no
        unit conversion is needed here.
        """
        gtime_key = Requirement('\\EFIT01::TOP.RESULTS.GEQDSK.GTIME/1000.', shot, 'EFIT01').as_key()
        return raw_data[gtime_key]

    def _compose_ion_temperature(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """
        Compose ion temperature as a jagged ak.Array (eV).

        Slots in GTIME where ITEMPFIT has no matching time are empty.
        """
        return self._compose_profile_field(shot, raw_data, 'ion_temperature', 'temperature')

    def _compose_carbon_density(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """
        Compose carbon density as a jagged ak.Array (m^-3).

        Slots in GTIME where ZDENSFIT has no matching time are empty.
        """
        return self._compose_profile_field(shot, raw_data, 'carbon_density', 'density')

    def _compose_carbon_rotation(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """
        Compose carbon rotation frequency as a jagged ak.Array (rad/s).

        Slots in GTIME where TROTFIT has no matching time are empty.
        """
        return self._compose_profile_field(shot, raw_data, 'carbon_rotation', 'rotation')

    def _compose_deuterium_density(self, shot: int, raw_data: Dict[str, Any]) -> ak.Array:
        """
        Compose deuterium density via quasineutrality as a jagged ak.Array (m^-3).

        n_D = n_e - 6 * n_C. A slot is empty unless both n_e and n_C are
        present at that GTIME point.
        """
        n_e_jagged = self._compose_density_thermal(shot, raw_data)
        n_C_jagged = self._compose_carbon_density(shot, raw_data)
        result = []
        for n_e, n_C in zip(n_e_jagged, n_C_jagged):
            if len(n_e) > 0 and len(n_C) > 0:
                result.append(np.asarray(n_e) - 6.0 * np.asarray(n_C))
            else:
                result.append(np.array([]))
        return ak.Array(result)

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
