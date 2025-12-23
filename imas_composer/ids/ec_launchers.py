"""
Electron Cyclotron Launchers IDS Mapping for DIII-D

See OMAS: omas/machine_mappings/d3d.py::ec_launcher_active_hardware
"""

from typing import Dict, List
import numpy as np

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class EcLaunchersMapper(IDSMapper):
    """Maps DIII-D EC launcher data to IMAS ec_launchers IDS."""

    # Configuration YAML file contains static values and field ledger
    DOCS_PATH = "ec_launchers.yaml"
    CONFIG_PATH = "ec_launchers.yaml"

    def __init__(self, **kwargs):
        """Initialize EC launchers mapper."""
        # MDSplus path prefixes
        self.setup_node = '.ECH.'

        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Build IDS specs
        self._build_specs()

    def _build_specs(self):
        """Build all IDS entry specifications"""

        # Internal dependencies - fetch NUM_SYSTEMS to know how many beams exist
        # Note: This is shot-dependent (different shots may have different active systems)
        self.specs["ec_launchers._num_systems"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=[],
            derive_requirements=lambda shot, raw: [
                Requirement('.ECH.NUM_SYSTEMS', shot, 'RF')
            ],
            ids_path="ec_launchers._num_systems",
            docs_file=self.DOCS_PATH
        )

        # DERIVED: Gyrotron names for all systems (needed by most fields)
        self.specs["ec_launchers._gyrotron_names"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["ec_launchers._num_systems"],
            derive_requirements=lambda shot, raw: self._derive_system_level_requirements(
                shot, raw, '.ECH.SYSTEM_{system_no}.GYROTRON.NAME'
            ),
            ids_path="ec_launchers._gyrotron_names",
            docs_file=self.DOCS_PATH
        )

        # DERIVED: Gyrotron-specific data (power, angles, etc.)
        self.specs["ec_launchers._gyrotron_data"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["ec_launchers._gyrotron_names"],
            derive_requirements=self._derive_gyrotron_requirements,
            ids_path="ec_launchers._gyrotron_data",
            docs_file=self.DOCS_PATH
        )

        # DERIVED: Frequency data (per-field requirement)
        self.specs["ec_launchers._frequency"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["ec_launchers._num_systems"],
            derive_requirements=lambda shot, raw: self._derive_system_level_requirements(
                shot, raw, '.ECH.SYSTEM_{system_no}.GYROTRON.FREQUENCY'
            ),
            ids_path="ec_launchers._frequency",
            docs_file=self.DOCS_PATH
        )

        # DERIVED: Launch R position (per-field requirement)
        self.specs["ec_launchers._launch_r"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["ec_launchers._num_systems"],
            derive_requirements=lambda shot, raw: self._derive_system_level_requirements(
                shot, raw, '.ECH.SYSTEM_{system_no}.ANTENNA.LAUNCH_R'
            ),
            ids_path="ec_launchers._launch_r",
            docs_file=self.DOCS_PATH
        )

        # DERIVED: Launch Z position (per-field requirement)
        self.specs["ec_launchers._launch_z"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["ec_launchers._num_systems"],
            derive_requirements=lambda shot, raw: self._derive_system_level_requirements(
                shot, raw, '.ECH.SYSTEM_{system_no}.ANTENNA.LAUNCH_Z'
            ),
            ids_path="ec_launchers._launch_z",
            docs_file=self.DOCS_PATH
        )

        # DERIVED: Port/phi data (per-field requirement)
        self.specs["ec_launchers._port"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["ec_launchers._num_systems"],
            derive_requirements=lambda shot, raw: self._derive_system_level_requirements(
                shot, raw, '.ECH.SYSTEM_{system_no}.ANTENNA.PORT'
            ),
            ids_path="ec_launchers._port",
            docs_file=self.DOCS_PATH
        )

        # DERIVED: Gaussian beam curvature (optional, per-field requirement)
        self.specs["ec_launchers._gb_rcurve"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["ec_launchers._num_systems"],
            derive_requirements=lambda shot, raw: self._derive_system_level_requirements(
                shot, raw, '.ECH.SYSTEM_{system_no}.ANTENNA.GB_RCURVE'
            ),
            ids_path="ec_launchers._gb_rcurve",
            docs_file=self.DOCS_PATH
        )

        # DERIVED: Gaussian beam waist (optional, per-field requirement)
        self.specs["ec_launchers._gb_waist"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["ec_launchers._num_systems"],
            derive_requirements=lambda shot, raw: self._derive_system_level_requirements(
                shot, raw, '.ECH.SYSTEM_{system_no}.ANTENNA.GB_WAIST'
            ),
            ids_path="ec_launchers._gb_waist",
            docs_file=self.DOCS_PATH
        )

        # Public IDS fields - all COMPUTED stage

        self.specs["ec_launchers.beam.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._gyrotron_data"],
            compose=self._compose_beam_time,
            ids_path="ec_launchers.beam.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.identifier"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._gyrotron_names", "ec_launchers._gyrotron_data"],
            compose=self._compose_beam_identifier,
            ids_path="ec_launchers.beam.identifier",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.name"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._gyrotron_names", "ec_launchers._gyrotron_data"],
            compose=self._compose_beam_name,
            ids_path="ec_launchers.beam.name",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.frequency.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._gyrotron_data"],
            compose=self._compose_frequency_time,
            ids_path="ec_launchers.beam.frequency.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.frequency.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._frequency", "ec_launchers._gyrotron_data"],
            compose=self._compose_frequency_data,
            ids_path="ec_launchers.beam.frequency.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.launching_position.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._launch_r", "ec_launchers._gyrotron_data"],
            compose=self._compose_launching_position_r,
            ids_path="ec_launchers.beam.launching_position.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.launching_position.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._launch_z", "ec_launchers._gyrotron_data"],
            compose=self._compose_launching_position_z,
            ids_path="ec_launchers.beam.launching_position.z",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.launching_position.phi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._port", "ec_launchers._gyrotron_data"],
            compose=self._compose_launching_position_phi,
            ids_path="ec_launchers.beam.launching_position.phi",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.mode"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._gyrotron_data"],
            compose=self._compose_mode,
            ids_path="ec_launchers.beam.mode",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.phase.angle"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._gyrotron_data"],
            compose=self._compose_phase_angle,
            ids_path="ec_launchers.beam.phase.angle",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.phase.curvature"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._gb_rcurve", "ec_launchers._gyrotron_data"],
            compose=self._compose_phase_curvature,
            ids_path="ec_launchers.beam.phase.curvature",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.power_launched.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._gyrotron_data"],
            compose=self._compose_power_launched_time,
            ids_path="ec_launchers.beam.power_launched.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.power_launched.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._gyrotron_data"],
            compose=self._compose_power_launched_data,
            ids_path="ec_launchers.beam.power_launched.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.spot.angle"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._gyrotron_data"],
            compose=self._compose_spot_angle,
            ids_path="ec_launchers.beam.spot.angle",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.spot.size"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._gb_waist", "ec_launchers._gyrotron_data"],
            compose=self._compose_spot_size,
            ids_path="ec_launchers.beam.spot.size",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.steering_angle_pol"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._gyrotron_data"],
            compose=self._compose_steering_angle_pol,
            ids_path="ec_launchers.beam.steering_angle_pol",
            docs_file=self.DOCS_PATH
        )

        self.specs["ec_launchers.beam.steering_angle_tor"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ec_launchers._gyrotron_data"],
            compose=self._compose_steering_angle_tor,
            ids_path="ec_launchers.beam.steering_angle_tor",
            docs_file=self.DOCS_PATH
        )

    # Requirement derivation helper
    def _derive_system_level_requirements(self, shot: int, raw_data: dict, mds_path_template: str) -> List[Requirement]:
        """
        Generic helper to derive requirements for system-level data.

        Args:
            shot: Shot number
            raw_data: Raw data dictionary
            mds_path_template: MDSplus path template with {system_no} placeholder
                              e.g., '.ECH.SYSTEM_{system_no}.GYROTRON.NAME'

        Returns:
            List of requirements, one per system
        """
        num_systems_key = Requirement('.ECH.NUM_SYSTEMS', shot, 'RF').as_key()
        num_systems = int(raw_data[num_systems_key])

        requirements = []
        for system_no in range(1, num_systems + 1):
            mds_path = mds_path_template.format(system_no=system_no)
            requirements.append(Requirement(mds_path, shot, 'RF'))

        return requirements

    def _derive_gyrotron_requirements(self, shot: int, raw_data: dict) -> List[Requirement]:
        """Derive requirements for gyrotron-specific data based on gyrotron names."""
        num_systems_key = Requirement('.ECH.NUM_SYSTEMS', shot, 'RF').as_key()
        num_systems = int(raw_data[num_systems_key])

        requirements = []
        for system_no in range(1, num_systems + 1):
            # Get gyrotron name from _gyrotron_names data
            gyrotron_name_key = Requirement(f'.ECH.SYSTEM_{system_no}.GYROTRON.NAME', shot, 'RF').as_key()

            # Skip if gyrotron name is empty (no gyrotron connected)
            if gyrotron_name_key not in raw_data:
                continue

            gyrotron_name = raw_data[gyrotron_name_key]
            if isinstance(gyrotron_name, Exception) or len(gyrotron_name) == 0:
                continue

            # Build gyrotron-specific paths
            gyr = gyrotron_name.upper()[:3]

            requirements.extend([
                Requirement(f'.ECH.{gyrotron_name.upper()}.EC{gyr}STAT', shot, 'RF'),
                Requirement(f'.ECH.{gyrotron_name.upper()}.EC{gyr}XMFRAC', shot, 'RF'),
                Requirement(f'.ECH.{gyrotron_name.upper()}.EC{gyr}FPWRC', shot, 'RF'),
                Requirement(f'dim_of(.ECH.{gyrotron_name.upper()}.EC{gyr}FPWRC+01) / 1E3', shot, 'RF'),
                Requirement(f'.ECH.{gyrotron_name.upper()}.EC{gyr}AZIANG', shot, 'RF'),
                Requirement(f'dim_of(.ECH.{gyrotron_name.upper()}.EC{gyr}AZIANG+01) / 1E3', shot, 'RF'),
                Requirement(f'.ECH.{gyrotron_name.upper()}.EC{gyr}POLANG', shot, 'RF'),
            ])

        return requirements

    # Helper functions
    def _get_active_systems(self, shot: int, raw_data: dict) -> List[int]:
        """Get list of active system indices (1-based)."""
        num_systems_key = Requirement('.ECH.NUM_SYSTEMS', shot, 'RF').as_key()
        num_systems = int(raw_data[num_systems_key])

        active_systems = []
        for system_no in range(1, num_systems + 1):
            # Check if gyrotron is active
            gyrotron_name_key = Requirement(f'.ECH.SYSTEM_{system_no}.GYROTRON.NAME', shot, 'RF').as_key()
            if gyrotron_name_key not in raw_data:
                continue

            gyrotron_name = raw_data[gyrotron_name_key]
            if isinstance(gyrotron_name, Exception) or len(gyrotron_name) == 0:
                continue

            # Check STAT flag
            gyr = gyrotron_name.upper()[:3]
            stat_key = Requirement(f'.ECH.{gyrotron_name.upper()}.EC{gyr}STAT', shot, 'RF').as_key()
            if stat_key in raw_data and raw_data[stat_key] != 0:
                active_systems.append(system_no)

        return active_systems

    def _get_gyrotron_name(self, shot: int, raw_data: dict, system_no: int) -> str:
        """Get gyrotron name for a given system number."""
        gyrotron_name_key = Requirement(f'.ECH.SYSTEM_{system_no}.GYROTRON.NAME', shot, 'RF').as_key()
        return raw_data[gyrotron_name_key]

    # Compose functions
    def _compose_beam_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose beam time arrays."""
        active_systems = self._get_active_systems(shot, raw_data)

        time_arrays = []
        for system_no in active_systems:
            gyrotron_name = self._get_gyrotron_name(shot, raw_data, system_no)
            gyr = gyrotron_name.upper()[:3]

            time_key = Requirement(f'dim_of(.ECH.{gyrotron_name.upper()}.EC{gyr}AZIANG+01) / 1E3', shot, 'RF').as_key()
            time = np.atleast_1d(raw_data[time_key])

            # OMAS uses 0 if only single time point
            if len(time) == 1:
                time = np.atleast_1d(0)

            time_arrays.append(time)

        return np.array(time_arrays)

    def _compose_beam_identifier(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose beam identifiers (gyrotron names)."""
        active_systems = self._get_active_systems(shot, raw_data)

        identifiers = []
        for system_no in active_systems:
            gyrotron_name = self._get_gyrotron_name(shot, raw_data, system_no)
            identifiers.append(gyrotron_name)

        return np.array(identifiers)

    def _compose_beam_name(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose beam names (same as identifiers)."""
        return self._compose_beam_identifier(shot, raw_data)

    def _compose_frequency_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose frequency time arrays (static, so just [0])."""
        active_systems = self._get_active_systems(shot, raw_data)

        freq_times = []
        for _ in active_systems:
            freq_times.append(np.atleast_1d(0))

        return np.array(freq_times)

    def _compose_frequency_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose frequency data arrays."""
        active_systems = self._get_active_systems(shot, raw_data)

        freq_arrays = []
        for system_no in active_systems:
            freq_key = Requirement(f'.ECH.SYSTEM_{system_no}.GYROTRON.FREQUENCY', shot, 'RF').as_key()

            if freq_key in raw_data and not isinstance(raw_data[freq_key], Exception):
                freq = np.atleast_1d(raw_data[freq_key])
            else:
                # Old shots did not record frequency, assume 110 GHz
                freq = np.ones(1) * 110e9

            freq_arrays.append(freq)

        return np.array(freq_arrays)

    def _compose_launching_position_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose launching position R coordinates."""
        active_systems = self._get_active_systems(shot, raw_data)

        r_arrays = []
        for system_no in active_systems:
            gyrotron_name = self._get_gyrotron_name(shot, raw_data, system_no)
            gyr = gyrotron_name.upper()[:3]

            # Get time array length
            time_key = Requirement(f'dim_of(.ECH.{gyrotron_name.upper()}.EC{gyr}AZIANG+01) / 1E3', shot, 'RF').as_key()
            time = np.atleast_1d(raw_data[time_key])
            ntime = len(time) if len(time) > 1 else 1

            r_key = Requirement(f'.ECH.SYSTEM_{system_no}.ANTENNA.LAUNCH_R', shot, 'RF').as_key()
            r_value = np.atleast_1d(raw_data[r_key])[0]

            r_arrays.append(r_value * np.ones(ntime))

        return np.array(r_arrays)

    def _compose_launching_position_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose launching position Z coordinates."""
        active_systems = self._get_active_systems(shot, raw_data)

        z_arrays = []
        for system_no in active_systems:
            gyrotron_name = self._get_gyrotron_name(shot, raw_data, system_no)
            gyr = gyrotron_name.upper()[:3]

            # Get time array length
            time_key = Requirement(f'dim_of(.ECH.{gyrotron_name.upper()}.EC{gyr}AZIANG+01) / 1E3', shot, 'RF').as_key()
            time = np.atleast_1d(raw_data[time_key])
            ntime = len(time) if len(time) > 1 else 1

            z_key = Requirement(f'.ECH.SYSTEM_{system_no}.ANTENNA.LAUNCH_Z', shot, 'RF').as_key()
            z_value = np.atleast_1d(raw_data[z_key])[0]

            z_arrays.append(z_value * np.ones(ntime))

        return np.array(z_arrays)

    def _compose_launching_position_phi(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose launching position phi coordinates."""
        active_systems = self._get_active_systems(shot, raw_data)

        phi_arrays = []
        for system_no in active_systems:
            gyrotron_name = self._get_gyrotron_name(shot, raw_data, system_no)
            gyr = gyrotron_name.upper()[:3]

            # Get time array length
            time_key = Requirement(f'dim_of(.ECH.{gyrotron_name.upper()}.EC{gyr}AZIANG+01) / 1E3', shot, 'RF').as_key()
            time = np.atleast_1d(raw_data[time_key])
            ntime = len(time) if len(time) > 1 else 1

            port_key = Requirement(f'.ECH.SYSTEM_{system_no}.ANTENNA.PORT', shot, 'RF').as_key()
            port_string = raw_data[port_key]
            phi = np.deg2rad(float(port_string.split(' ')[0]))

            phi_arrays.append(phi * np.ones(ntime))

        return np.array(phi_arrays)

    def _compose_mode(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose mode values (O-mode = +1, X-mode = -1)."""
        active_systems = self._get_active_systems(shot, raw_data)

        modes = []
        for system_no in active_systems:
            gyrotron_name = self._get_gyrotron_name(shot, raw_data, system_no)
            gyr = gyrotron_name.upper()[:3]

            xfrac_key = Requirement(f'.ECH.{gyrotron_name.upper()}.EC{gyr}XMFRAC', shot, 'RF').as_key()

            if xfrac_key in raw_data and not isinstance(raw_data[xfrac_key], Exception):
                xfrac = raw_data[xfrac_key]
                mode = int(np.round(1.0 - 2.0 * max(np.atleast_1d(xfrac))))
            else:
                # Assume X-mode if XMFRAC not recorded
                mode = -1

            modes.append(mode)

        return np.array(modes)

    def _compose_phase_angle(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose phase angle arrays (zeros)."""
        active_systems = self._get_active_systems(shot, raw_data)

        phase_arrays = []
        for system_no in active_systems:
            gyrotron_name = self._get_gyrotron_name(shot, raw_data, system_no)
            gyr = gyrotron_name.upper()[:3]

            time_key = Requirement(f'dim_of(.ECH.{gyrotron_name.upper()}.EC{gyr}AZIANG+01) / 1E3', shot, 'RF').as_key()
            time = np.atleast_1d(raw_data[time_key])
            ntime = len(time) if len(time) > 1 else 1

            phase_arrays.append(np.zeros(ntime))

        return np.array(phase_arrays)

    def _compose_phase_curvature(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose phase curvature arrays."""
        active_systems = self._get_active_systems(shot, raw_data)

        curvature_arrays = []
        for system_no in active_systems:
            gyrotron_name = self._get_gyrotron_name(shot, raw_data, system_no)
            gyr = gyrotron_name.upper()[:3]

            time_key = Requirement(f'dim_of(.ECH.{gyrotron_name.upper()}.EC{gyr}AZIANG+01) / 1E3', shot, 'RF').as_key()
            time = np.atleast_1d(raw_data[time_key])
            ntime = len(time) if len(time) > 1 else 1

            # Try to get curvature from MDSplus (optional field)
            rcurve_key = Requirement(f'.ECH.SYSTEM_{system_no}.ANTENNA.GB_RCURVE', shot, 'RF').as_key()

            if rcurve_key in raw_data and not isinstance(raw_data[rcurve_key], Exception):
                # DIII-D uses negative for divergent, IMAS uses positive
                curvature = np.zeros([2, ntime])
                curvature[:] = -1.0 / raw_data[rcurve_key]
            else:
                # Default: paraxial at launching point (zero curvature)
                curvature = np.zeros([2, ntime])

            curvature_arrays.append(curvature)

        return np.array(curvature_arrays)

    def _compose_power_launched_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose power launched time arrays."""
        active_systems = self._get_active_systems(shot, raw_data)

        time_arrays = []
        for system_no in active_systems:
            gyrotron_name = self._get_gyrotron_name(shot, raw_data, system_no)
            gyr = gyrotron_name.upper()[:3]

            time_key = Requirement(f'dim_of(.ECH.{gyrotron_name.upper()}.EC{gyr}FPWRC+01) / 1E3', shot, 'RF').as_key()
            times = np.atleast_1d(raw_data[time_key])

            time_arrays.append(times)

        return np.array(time_arrays)

    def _compose_power_launched_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose power launched data arrays."""
        active_systems = self._get_active_systems(shot, raw_data)

        power_arrays = []
        for system_no in active_systems:
            gyrotron_name = self._get_gyrotron_name(shot, raw_data, system_no)
            gyr = gyrotron_name.upper()[:3]

            power_key = Requirement(f'.ECH.{gyrotron_name.upper()}.EC{gyr}FPWRC', shot, 'RF').as_key()
            power = np.atleast_1d(raw_data[power_key])

            power_arrays.append(power)

        return np.array(power_arrays)

    def _compose_spot_angle(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose spot angle arrays (zeros)."""
        active_systems = self._get_active_systems(shot, raw_data)

        angle_arrays = []
        for system_no in active_systems:
            gyrotron_name = self._get_gyrotron_name(shot, raw_data, system_no)
            gyr = gyrotron_name.upper()[:3]

            time_key = Requirement(f'dim_of(.ECH.{gyrotron_name.upper()}.EC{gyr}AZIANG+01) / 1E3', shot, 'RF').as_key()
            time = np.atleast_1d(raw_data[time_key])
            ntime = len(time) if len(time) > 1 else 1

            angle_arrays.append(np.zeros(ntime))

        return np.array(angle_arrays)

    def _compose_spot_size(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose spot size arrays."""
        active_systems = self._get_active_systems(shot, raw_data)

        size_arrays = []
        for system_no in active_systems:
            gyrotron_name = self._get_gyrotron_name(shot, raw_data, system_no)
            gyr = gyrotron_name.upper()[:3]

            time_key = Requirement(f'dim_of(.ECH.{gyrotron_name.upper()}.EC{gyr}AZIANG+01) / 1E3', shot, 'RF').as_key()
            time = np.atleast_1d(raw_data[time_key])
            ntime = len(time) if len(time) > 1 else 1

            # Try to get spot size from MDSplus
            waist_key = Requirement(f'.ECH.SYSTEM_{system_no}.ANTENNA.GB_WAIST', shot, 'RF').as_key()

            if waist_key in raw_data and not isinstance(raw_data[waist_key], Exception):
                waist = raw_data[waist_key]
                size = np.zeros([2, ntime])
                size[0, :] = waist
                size[1, :] = waist
            else:
                # Default: 1.72 cm beam waist
                size = 0.0172 * np.ones([2, ntime])

            size_arrays.append(size)

        return np.array(size_arrays)

    def _compose_steering_angle_pol(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose steering angle poloidal arrays."""
        active_systems = self._get_active_systems(shot, raw_data)

        angle_arrays = []
        for system_no in active_systems:
            gyrotron_name = self._get_gyrotron_name(shot, raw_data, system_no)
            gyr = gyrotron_name.upper()[:3]

            time_key = Requirement(f'dim_of(.ECH.{gyrotron_name.upper()}.EC{gyr}AZIANG+01) / 1E3', shot, 'RF').as_key()
            time = np.atleast_1d(raw_data[time_key])

            aziang_key = Requirement(f'.ECH.{gyrotron_name.upper()}.EC{gyr}AZIANG', shot, 'RF').as_key()
            polang_key = Requirement(f'.ECH.{gyrotron_name.upper()}.EC{gyr}POLANG', shot, 'RF').as_key()

            phi_tor = np.atleast_1d(np.deg2rad(raw_data[aziang_key] - 180.0))
            theta_pol = np.atleast_1d(np.deg2rad(raw_data[polang_key] - 90.0))

            # Broadcast if needed
            if len(phi_tor) == 1 and len(phi_tor) != len(time):
                phi_tor = np.ones(len(time)) * phi_tor[0]
            if len(theta_pol) == 1 and len(theta_pol) != len(time):
                theta_pol = np.ones(len(time)) * theta_pol[0]

            steering_pol = np.arctan2(np.tan(theta_pol), np.cos(phi_tor))
            angle_arrays.append(steering_pol)

        return np.array(angle_arrays)

    def _compose_steering_angle_tor(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose steering angle toroidal arrays."""
        active_systems = self._get_active_systems(shot, raw_data)

        angle_arrays = []
        for system_no in active_systems:
            gyrotron_name = self._get_gyrotron_name(shot, raw_data, system_no)
            gyr = gyrotron_name.upper()[:3]

            time_key = Requirement(f'dim_of(.ECH.{gyrotron_name.upper()}.EC{gyr}AZIANG+01) / 1E3', shot, 'RF').as_key()
            time = np.atleast_1d(raw_data[time_key])

            aziang_key = Requirement(f'.ECH.{gyrotron_name.upper()}.EC{gyr}AZIANG', shot, 'RF').as_key()
            polang_key = Requirement(f'.ECH.{gyrotron_name.upper()}.EC{gyr}POLANG', shot, 'RF').as_key()

            phi_tor = np.atleast_1d(np.deg2rad(raw_data[aziang_key] - 180.0))
            theta_pol = np.atleast_1d(np.deg2rad(raw_data[polang_key] - 90.0))

            # Broadcast if needed
            if len(phi_tor) == 1 and len(phi_tor) != len(time):
                phi_tor = np.ones(len(time)) * phi_tor[0]
            if len(theta_pol) == 1 and len(theta_pol) != len(time):
                theta_pol = np.ones(len(time)) * theta_pol[0]

            steering_tor = -np.arcsin(np.cos(theta_pol) * np.sin(phi_tor))
            angle_arrays.append(steering_tor)

        return np.array(angle_arrays)

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
