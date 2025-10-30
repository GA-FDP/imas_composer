"""
Thomson Scattering IDS Mapping for DIII-D

See OMAS: omas/machine_mappings/d3d.py::thomson_scattering_data
"""

from typing import Dict, List, Tuple
import numpy as np
import awkward as ak

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class ThomsonScatteringMapper(IDSMapper):
    """Maps DIII-D Thomson scattering data to IMAS thomson_scattering IDS."""

    SYSTEMS = ['TANGENTIAL', 'DIVERTOR', 'CORE']
    REVISION = 'BLESSED'

    DOCS_PATH = "thomson_scattering.yaml"
    CONFIG_PATH = "thomson_scattering.yaml"

    def __init__(self):
        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Build IDS specs
        self._build_specs()

    def _get_calib_nums_path(self) -> str:
        """Get MDS+ path for calibration numbers."""
        return f'.ts.{self.REVISION}.header.calib_nums'

    def _get_system_coordinate_path(self, system: str, coordinate: str) -> str:
        """Get MDS+ path for system coordinate (R, Z, PHI)."""
        return f'.TS.{self.REVISION}.{system}:{coordinate}'

    def _get_system_measurement_path(self, system: str, quantity: str) -> str:
        """Get MDS+ path for system measurement (DENSITY, TEMP, TIME, etc)."""
        return f'.TS.{self.REVISION}.{system}:{quantity}'

    def _get_hwmap_path(self, system: str) -> str:
        """Get MDS+ path for hardware map."""
        return f'.{system}.hwmapints'
    
    def _build_specs(self):
        """Build all IDS entry specifications"""
        
        # Internal dependencies
        self.specs["thomson_scattering._calib_nums"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(self._get_calib_nums_path(), 0, 'ELECTRONS')
            ],
            ids_path="thomson_scattering._calib_nums",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["thomson_scattering._hwmap"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["thomson_scattering._calib_nums"],
            derive_requirements=self._derive_hwmap_requirements,
            ids_path="thomson_scattering._hwmap",
            docs_file=self.DOCS_PATH
        )
        
        # Split position requirements by coordinate to avoid unnecessary coupling
        self.specs["thomson_scattering._position_r"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["thomson_scattering._calib_nums"],
            derive_requirements=lambda shot, raw: self._derive_position_requirements(shot, raw, 'R'),
            ids_path="thomson_scattering._position_r",
            docs_file=self.DOCS_PATH
        )

        self.specs["thomson_scattering._position_z"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["thomson_scattering._calib_nums"],
            derive_requirements=lambda shot, raw: self._derive_position_requirements(shot, raw, 'Z'),
            ids_path="thomson_scattering._position_z",
            docs_file=self.DOCS_PATH
        )

        self.specs["thomson_scattering._position_phi"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["thomson_scattering._calib_nums"],
            derive_requirements=lambda shot, raw: self._derive_position_requirements(shot, raw, 'PHI'),
            ids_path="thomson_scattering._position_phi",
            docs_file=self.DOCS_PATH
        )

        # Keep _system_availability for fields that need to know which systems are active
        # but don't need position data (e.g., channel.name, channel.identifier)
        self.specs["thomson_scattering._system_availability"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["thomson_scattering._calib_nums"],
            derive_requirements=lambda shot, raw: self._derive_position_requirements(shot, raw, 'R'),
            ids_path="thomson_scattering._system_availability",
            docs_file=self.DOCS_PATH
        )
        
        # Channel metadata
        self.specs["thomson_scattering.channel.name"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "thomson_scattering._hwmap",
                "thomson_scattering._system_availability"
            ],
            compose=self._compose_channel_name,
            ids_path="thomson_scattering.channel.name",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["thomson_scattering.channel.identifier"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["thomson_scattering._system_availability"],
            compose=self._compose_channel_identifier,
            ids_path="thomson_scattering.channel.identifier",
            docs_file=self.DOCS_PATH
        )
        
        # Position coordinates - each independently resolvable
        self.specs["thomson_scattering.channel.position.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["thomson_scattering._position_r"],
            compose=self._compose_position_r,
            ids_path="thomson_scattering.channel.position.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["thomson_scattering.channel.position.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["thomson_scattering._position_z"],
            compose=self._compose_position_z,
            ids_path="thomson_scattering.channel.position.z",
            docs_file=self.DOCS_PATH
        )

        self.specs["thomson_scattering.channel.position.phi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["thomson_scattering._position_phi"],
            compose=self._compose_position_phi,
            ids_path="thomson_scattering.channel.position.phi",
            docs_file=self.DOCS_PATH
        )
        
        # Measurements
        self._add_measurement_specs('n_e', 'DENSITY')
        self._add_measurement_specs('t_e', 'TEMP')
    
    def _add_measurement_specs(self, measurement: str, mds_quantity: str):
        """Add specs for a measurement (time, data, and data_error_upper)"""

        # Split requirements into separate specs to avoid unnecessary bundling
        # Each field should only depend on exactly what it needs

        # Time requirements: only TIME data from each system
        time_reqs = []
        for system in self.SYSTEMS:
            time_reqs.append(
                Requirement(self._get_system_measurement_path(system, 'TIME'), 0, 'ELECTRONS')
            )

        self.specs[f"thomson_scattering._{measurement}_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=time_reqs,
            ids_path=f"thomson_scattering._{measurement}_time",
            docs_file=self.DOCS_PATH
        )

        # Data requirements: only measurement data from each system
        data_reqs = []
        for system in self.SYSTEMS:
            data_reqs.append(
                Requirement(self._get_system_measurement_path(system, mds_quantity), 0, 'ELECTRONS')
            )

        self.specs[f"thomson_scattering._{measurement}_data"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=data_reqs,
            ids_path=f"thomson_scattering._{measurement}_data",
            docs_file=self.DOCS_PATH
        )

        # Error requirements: only error data from each system
        error_reqs = []
        for system in self.SYSTEMS:
            error_reqs.append(
                Requirement(self._get_system_measurement_path(system, f'{mds_quantity}_E'), 0, 'ELECTRONS')
            )

        self.specs[f"thomson_scattering._{measurement}_error"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=error_reqs,
            ids_path=f"thomson_scattering._{measurement}_error",
            docs_file=self.DOCS_PATH
        )

        # Time array (shared between data and error)
        # See OMAS d3d.py:72,74 - uses tsdat[f'{system}_TIME'] / 1e3
        self.specs[f"thomson_scattering.channel.{measurement}.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                f"thomson_scattering._{measurement}_time",
                "thomson_scattering._system_availability"
            ],
            compose=lambda shot, raw: self._compose_channel_time(shot, raw),
            ids_path=f"thomson_scattering.channel.{measurement}.time",
            docs_file=self.DOCS_PATH
        )

        # Synthesized data
        self.specs[f"thomson_scattering.channel.{measurement}.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                f"thomson_scattering._{measurement}_data",
                "thomson_scattering._system_availability"
            ],
            compose=lambda shot, raw, m=measurement, q=mds_quantity:
                self._compose_channel_measurement_data(shot, raw, m, q),
            ids_path=f"thomson_scattering.channel.{measurement}.data",
            docs_file=self.DOCS_PATH
        )

        # Synthesized error
        self.specs[f"thomson_scattering.channel.{measurement}.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                f"thomson_scattering._{measurement}_error",
                "thomson_scattering._system_availability"
            ],
            compose=lambda shot, raw, m=measurement, q=mds_quantity:
                self._compose_channel_measurement_data_error_upper(shot, raw, m, q),
            ids_path=f"thomson_scattering.channel.{measurement}.data_error_upper",
            docs_file=self.DOCS_PATH
        )
    
    # Requirement derivation functions
    
    def _derive_hwmap_requirements(self, shot: int, raw_data: dict) -> List[Requirement]:
        """Determine hardware map requirements using calibration numbers"""
        calib_key = Requirement(self._get_calib_nums_path(), shot, 'ELECTRONS').as_key()
        cal_set = raw_data[calib_key][0]

        return [
            Requirement(self._get_hwmap_path(system), cal_set, 'TSCAL')
            for system in self.SYSTEMS
        ]
    
    def _derive_position_requirements(self, shot: int, raw_data: dict, coordinate: str) -> List[Requirement]:
        """Request position data for a specific coordinate across all systems"""
        requirements = []
        for system in self.SYSTEMS:
            requirements.append(
                Requirement(self._get_system_coordinate_path(system, coordinate), shot, 'ELECTRONS')
            )
        return requirements

    # Synthesis functions
    
    def _compose_channel_name(self, shot: int, raw_data: dict) -> np.ndarray:
        """Generate channel names: TS_{system}_r{lens}_{channel}"""
        names = []

        for system in self.SYSTEMS:
            # Check if R coordinate exists (indicates active system)
            r_key = Requirement(
                self._get_system_coordinate_path(system, 'R'), shot, 'ELECTRONS'
            ).as_key()

            if r_key not in raw_data:
                continue

            r_data = raw_data[r_key]
            if isinstance(r_data, Exception) or len(r_data) == 0:
                continue

            nc = self._get_system_channel_count(system, shot, raw_data)
            hwmap_key = self._get_hwmap_key(system, shot, raw_data)
            ints = raw_data[hwmap_key]

            if len(np.shape(ints)) < 2:
                ints = ints.reshape(1, -1)
            lenses = ints[:, 2]

            for j in range(nc):
                lens_idx = min(j, len(lenses) - 1)
                names.append(f'TS_{system.lower()}_r{lenses[lens_idx]:+0d}_{j}')

        return np.array(names)
    
    def _compose_channel_identifier(self, shot: int, raw_data: dict) -> np.ndarray:
        """Generate short identifiers: {T|D|C}{channel:02d}"""
        identifiers = []

        for system in self.SYSTEMS:
            # Check if R coordinate exists (indicates active system)
            r_key = Requirement(
                self._get_system_coordinate_path(system, 'R'), shot, 'ELECTRONS'
            ).as_key()

            if r_key not in raw_data:
                continue

            r_data = raw_data[r_key]
            if isinstance(r_data, Exception) or len(r_data) == 0:
                continue

            nc = self._get_system_channel_count(system, shot, raw_data)
            system_letter = system[0]

            for j in range(nc):
                identifiers.append(f'{system_letter}{j:02d}')

        return np.array(identifiers)
    
    def _compose_position_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """Synthesize R coordinate across all active systems"""
        positions = []

        for system in self.SYSTEMS:
            coord_key = Requirement(
                self._get_system_coordinate_path(system, 'R'), shot, 'ELECTRONS'
            ).as_key()

            # Only extend if data exists and is non-empty
            if coord_key in raw_data:
                coord_data = raw_data[coord_key]
                if len(coord_data) > 0:
                    positions.extend(coord_data)

        return np.array(positions)

    def _compose_position_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """Synthesize Z coordinate across all active systems"""
        positions = []

        for system in self.SYSTEMS:
            coord_key = Requirement(
                self._get_system_coordinate_path(system, 'Z'), shot, 'ELECTRONS'
            ).as_key()

            # Only extend if data exists and is non-empty
            if coord_key in raw_data:
                coord_data = raw_data[coord_key]
                if len(coord_data) > 0:
                    positions.extend(coord_data)

        return np.array(positions)

    def _compose_position_phi(self, shot: int, raw_data: dict) -> np.ndarray:
        """Synthesize phi coordinate across all active systems"""
        positions = []

        for system in self.SYSTEMS:
            coord_key = Requirement(
                self._get_system_coordinate_path(system, 'PHI'), shot, 'ELECTRONS'
            ).as_key()

            # Only extend if data exists and is non-empty
            if coord_key in raw_data:
                coord_data = raw_data[coord_key]
                if len(coord_data) > 0:
                    # Convert to IMAS convention: negative radians
                    coord_data = -coord_data * np.pi / 180.0
                    positions.extend(coord_data)

        return np.array(positions)

    def _compose_channel_time(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Synthesize time arrays across all active systems.

        Implements thomson_scattering.channel.{n_e|t_e}.time

        See OMAS d3d.py:72,74 - ch['n_e.time'] = tsdat[f'{system}_TIME'] / 1e3

        Args:
            shot: Shot number
            raw_data: Raw MDS+ data

        Returns:
            Awkward array with ragged time data per channel (milliseconds)
        """
        all_time = []

        for system in self.SYSTEMS:
            time_key = Requirement(
                self._get_system_measurement_path(system, 'TIME'), shot, 'ELECTRONS'
            ).as_key()

            # Only process if data exists and is non-empty
            if time_key not in raw_data:
                continue

            # Get time for this system
            system_time = raw_data[time_key]
            if len(system_time) == 0:
                continue

            # Convert from microseconds to seconds (OMAS uses / 1e3 for ms)
            system_time = system_time / 1e3

            # Get number of channels for this system
            nc = self._get_system_channel_count(system, shot, raw_data)

            # Each channel gets the same time array (but ragged across systems)
            for _ in range(nc):
                all_time.append(system_time)

        return ak.Array(all_time)

    def _compose_channel_measurement_data(self, shot: int, raw_data: dict,
                                             measurement: str, quantity: str) -> ak.Array:
        """
        Synthesize measurement data across all active systems.

        Implements thomson_scattering.channel.{n_e|t_e}.data

        Args:
            shot: Shot number
            raw_data: Raw MDS+ data
            measurement: IMAS measurement name (e.g., 'n_e', 't_e')
            quantity: MDS+ quantity name (e.g., 'DENSITY', 'TEMP')

        Returns:
            Awkward array with ragged measurement data per channel
        """
        all_data = []

        for system in self.SYSTEMS:
            data_key = Requirement(
                self._get_system_measurement_path(system, quantity), shot, 'ELECTRONS'
            ).as_key()

            # Only process if data exists
            if data_key not in raw_data:
                continue

            # Get data for this system (shape: n_channels_this_system, n_time)
            system_data = raw_data[data_key]

            # Append each channel (skip if empty)
            if system_data.ndim == 1:
                # Single channel system
                if len(system_data) > 0:
                    all_data.append(system_data)
            else:
                # Multiple channels - add each channel (ragged time bases)
                for channel_data in system_data:
                    if len(channel_data) > 0:
                        all_data.append(channel_data)

        return ak.Array(all_data)

    def _compose_channel_measurement_data_error_upper(self, shot: int, raw_data: dict,
                                                         measurement: str, quantity: str) -> ak.Array:
        """
        Synthesize measurement upper uncertainties across all active systems.

        Implements thomson_scattering.channel.{n_e|t_e}.data_error_upper

        Args:
            shot: Shot number
            raw_data: Raw MDS+ data
            measurement: IMAS measurement name (e.g., 'n_e', 't_e')
            quantity: MDS+ quantity name (e.g., 'DENSITY', 'TEMP')

        Returns:
            Awkward array with ragged uncertainty data per channel
        """
        all_errors = []

        for system in self.SYSTEMS:
            error_key = Requirement(
                self._get_system_measurement_path(system, f'{quantity}_E'), shot, 'ELECTRONS'
            ).as_key()

            # Only process if data exists
            if error_key not in raw_data:
                continue

            # Get error for this system (shape: n_channels_this_system, n_time)
            system_error = raw_data[error_key]

            # Append each channel (skip if empty)
            if system_error.ndim == 1:
                # Single channel system
                if len(system_error) > 0:
                    all_errors.append(system_error)
            else:
                # Multiple channels - add each channel (ragged time bases)
                for channel_error in system_error:
                    if len(channel_error) > 0:
                        all_errors.append(channel_error)

        return ak.Array(all_errors)
    
    # Helper functions

    def _get_system_channel_count(self, system: str, shot: int, raw_data: dict) -> int:
        """Get number of channels in a system"""
        r_key = Requirement(
            self._get_system_coordinate_path(system, 'R'), shot, 'ELECTRONS'
        ).as_key()
        return len(raw_data[r_key])

    def _get_hwmap_key(self, system: str, shot: int, raw_data: dict) -> Tuple:
        """Get the raw_data key for a system's hardware map"""
        calib_key = Requirement(self._get_calib_nums_path(), shot, 'ELECTRONS').as_key()
        cal_set = raw_data[calib_key][0]
        return Requirement(self._get_hwmap_path(system), cal_set, 'TSCAL').as_key()
    
    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        """Return all IDS entry specifications"""
        return self.specs