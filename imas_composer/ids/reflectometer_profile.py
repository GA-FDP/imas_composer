"""
Reflectometer Profile IDS Mapping for DIII-D

Hardware geometry: OMAS d3d.py::reflectometer_hardware
Measurement data: OMAS d3d.py::reflectometer_data

Combines static hardware configuration (from YAML) with measurement data (from MDSplus).
"""

from typing import Dict, List
import numpy as np
import awkward as ak

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class ReflectometerProfileMapper(IDSMapper):
    """Maps DIII-D reflectometer hardware and measurement data to IMAS reflectometer_profile IDS."""

    # Configuration YAML file contains static values and field ledger
    DOCS_PATH = "reflectometer_profile.yaml"
    CONFIG_PATH = "reflectometer_profile.yaml"

    def __init__(self):
        """Initialize reflectometer profile mapper."""
        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Load channel configurations from static_values
        config = self._load_config()
        self.channels = config.get('static_values', {}).get('channels', [])
        self.tree = 'ELECTRONS'
        self.position_z = config.get('static_values', {}).get('position_z', 0.0254)

        # Build IDS specs
        self._build_specs()

    def _build_specs(self):
        """Build all IDS entry specifications"""

        # All fields are COMPUTED stage (no MDSplus data needed - all static)

        self.specs["reflectometer_profile.channel.identifier"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_channel_identifier,
            ids_path="reflectometer_profile.channel.identifier",
            docs_file=self.DOCS_PATH
        )

        self.specs["reflectometer_profile.channel.name"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_channel_name,
            ids_path="reflectometer_profile.channel.name",
            docs_file=self.DOCS_PATH
        )

        self.specs["reflectometer_profile.channel.mode"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_channel_mode,
            ids_path="reflectometer_profile.channel.mode",
            docs_file=self.DOCS_PATH
        )

        self.specs["reflectometer_profile.channel.line_of_sight_emission.first_point.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_los_first_r,
            ids_path="reflectometer_profile.channel.line_of_sight_emission.first_point.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["reflectometer_profile.channel.line_of_sight_emission.first_point.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_los_first_z,
            ids_path="reflectometer_profile.channel.line_of_sight_emission.first_point.z",
            docs_file=self.DOCS_PATH
        )

        self.specs["reflectometer_profile.channel.line_of_sight_emission.first_point.phi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_los_first_phi,
            ids_path="reflectometer_profile.channel.line_of_sight_emission.first_point.phi",
            docs_file=self.DOCS_PATH
        )

        self.specs["reflectometer_profile.channel.line_of_sight_emission.second_point.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_los_second_r,
            ids_path="reflectometer_profile.channel.line_of_sight_emission.second_point.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["reflectometer_profile.channel.line_of_sight_emission.second_point.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_los_second_z,
            ids_path="reflectometer_profile.channel.line_of_sight_emission.second_point.z",
            docs_file=self.DOCS_PATH
        )

        self.specs["reflectometer_profile.channel.line_of_sight_emission.second_point.phi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_los_second_phi,
            ids_path="reflectometer_profile.channel.line_of_sight_emission.second_point.phi",
            docs_file=self.DOCS_PATH
        )

        # ===== Measurement Data Fields =====
        # These require MDSplus data from ELECTRONS tree

        # Internal dependencies: Per-channel raw data (DIRECT stage)
        for channel in self.channels:
            identifier = channel['identifier']
            path_prefix = f'\\ELECTRONS::TOP.REFLECT.{identifier}.PROCESSED:'

            # Time, frequency, and phase for each channel
            self.specs[f"reflectometer_profile._channel_{identifier}_time"] = IDSEntrySpec(
                stage=RequirementStage.DIRECT,
                static_requirements=[
                    Requirement(f'{path_prefix}TIME', None, self.tree)
                ],
                ids_path=f"reflectometer_profile._channel_{identifier}_time",
                docs_file=self.DOCS_PATH
            )

            self.specs[f"reflectometer_profile._channel_{identifier}_frequency"] = IDSEntrySpec(
                stage=RequirementStage.DIRECT,
                static_requirements=[
                    Requirement(f'{path_prefix}FREQUENCY', None, self.tree)
                ],
                ids_path=f"reflectometer_profile._channel_{identifier}_frequency",
                docs_file=self.DOCS_PATH
            )

            self.specs[f"reflectometer_profile._channel_{identifier}_phase"] = IDSEntrySpec(
                stage=RequirementStage.DIRECT,
                static_requirements=[
                    Requirement(f'{path_prefix}PHASE', None, self.tree)
                ],
                ids_path=f"reflectometer_profile._channel_{identifier}_phase",
                docs_file=self.DOCS_PATH
            )

        # Internal dependencies: Full profile raw data (DIRECT stage)
        full_prof_prefix = '\\ELECTRONS::TOP.REFLECT.FULL_PROF:'

        self.specs["reflectometer_profile._full_profile_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{full_prof_prefix}TIME', None, self.tree)
            ],
            ids_path="reflectometer_profile._full_profile_time",
            docs_file=self.DOCS_PATH
        )

        self.specs["reflectometer_profile._full_profile_R"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{full_prof_prefix}R', None, self.tree)
            ],
            ids_path="reflectometer_profile._full_profile_R",
            docs_file=self.DOCS_PATH
        )

        self.specs["reflectometer_profile._full_profile_density"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{full_prof_prefix}DENSITY', None, self.tree)
            ],
            ids_path="reflectometer_profile._full_profile_density",
            docs_file=self.DOCS_PATH
        )

        # User-facing fields: Per-channel phase data (COMPUTED stage)
        # Dependencies: All channel raw data (for time alignment)
        channel_deps = []
        for channel in self.channels:
            identifier = channel['identifier']
            channel_deps.extend([
                f"reflectometer_profile._channel_{identifier}_time",
                f"reflectometer_profile._channel_{identifier}_frequency",
                f"reflectometer_profile._channel_{identifier}_phase"
            ])

        self.specs["reflectometer_profile.channel.phase.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=channel_deps,
            compose=self._compose_channel_phase_time,
            ids_path="reflectometer_profile.channel.phase.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["reflectometer_profile.channel.frequencies"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=channel_deps,
            compose=self._compose_channel_frequencies,
            ids_path="reflectometer_profile.channel.frequencies",
            docs_file=self.DOCS_PATH
        )

        self.specs["reflectometer_profile.channel.phase.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=channel_deps,
            compose=self._compose_channel_phase_data,
            ids_path="reflectometer_profile.channel.phase.data",
            docs_file=self.DOCS_PATH
        )

        # User-facing fields: Full profile data (COMPUTED stage)
        profile_deps = [
            "reflectometer_profile._full_profile_time",
            "reflectometer_profile._full_profile_R",
            "reflectometer_profile._full_profile_density"
        ] + channel_deps  # Need channel data for time alignment

        self.specs["reflectometer_profile.position.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=profile_deps,
            compose=self._compose_position_r,
            ids_path="reflectometer_profile.position.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["reflectometer_profile.position.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=profile_deps,
            compose=self._compose_position_z,
            ids_path="reflectometer_profile.position.z",
            docs_file=self.DOCS_PATH
        )

        self.specs["reflectometer_profile.n_e.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=profile_deps,
            compose=self._compose_n_e_time,
            ids_path="reflectometer_profile.n_e.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["reflectometer_profile.n_e.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=profile_deps,
            compose=self._compose_n_e_data,
            ids_path="reflectometer_profile.n_e.data",
            docs_file=self.DOCS_PATH
        )

    # ===== Hardware Compose Functions =====
    # All extract data from self.channels

    def _compose_channel_identifier(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose channel identifiers."""
        return np.array([channel['identifier'] for channel in self.channels])

    def _compose_channel_name(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose channel names."""
        return np.array([channel['name'] for channel in self.channels])

    def _compose_channel_mode(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose channel polarization modes (X or O)."""
        return np.array([channel['mode'] for channel in self.channels])

    def _compose_los_first_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose line of sight first point R coordinates."""
        return np.array([channel['line_of_sight_emission']['first_point']['r'] for channel in self.channels])

    def _compose_los_first_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose line of sight first point Z coordinates."""
        return np.array([channel['line_of_sight_emission']['first_point']['z'] for channel in self.channels])

    def _compose_los_first_phi(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose line of sight first point phi coordinates."""
        return np.array([channel['line_of_sight_emission']['first_point']['phi'] for channel in self.channels])

    def _compose_los_second_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose line of sight second point R coordinates."""
        return np.array([channel['line_of_sight_emission']['second_point']['r'] for channel in self.channels])

    def _compose_los_second_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose line of sight second point Z coordinates."""
        return np.array([channel['line_of_sight_emission']['second_point']['z'] for channel in self.channels])

    def _compose_los_second_phi(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose line of sight second point phi coordinates."""
        return np.array([channel['line_of_sight_emission']['second_point']['phi'] for channel in self.channels])

    # ===== Measurement Data Compose Functions =====

    def _get_aligned_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get common time base for all channels.

        OMAS aligns all channels to the first channel's time base to ensure equal size.
        Returns None if no valid channel data exists.
        """
        for channel in self.channels:
            identifier = channel['identifier']
            path_prefix = f'\\ELECTRONS::TOP.REFLECT.{identifier}.PROCESSED:'
            time_key = Requirement(f'{path_prefix}TIME', shot, self.tree).as_key()

            if time_key in raw_data and not isinstance(raw_data[time_key], Exception):
                time_ms = raw_data[time_key]
                # Convert from ms to seconds (OMAS does /1e3)
                return np.array(time_ms) / 1e3

        return None

    def _compose_channel_phase_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose aligned time base for phase data.

        All channels share the same time base (from first valid channel).
        """
        time = []
        for channel in self.channels:
            time.append(self._get_aligned_time(shot, raw_data))
            if time is None:
                time.append(np.array([]))
        return time

    def _compose_channel_frequencies(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compose frequency arrays for each channel.

        Returns ragged array with shape: (n_channels, var * n_frequencies)
        Channels can have different numbers of frequencies or be missing entirely.
        """
        frequencies = []
        for channel in self.channels:
            identifier = channel['identifier']
            path_prefix = f'\\ELECTRONS::TOP.REFLECT.{identifier}.PROCESSED:'
            freq_key = Requirement(f'{path_prefix}FREQUENCY', shot, self.tree).as_key()

            if freq_key in raw_data and not isinstance(raw_data[freq_key], Exception):
                frequencies.append(raw_data[freq_key])
            else:
                # If channel is missing, append empty array
                frequencies.append(np.array([]))

        return ak.Array(frequencies)

    def _compose_channel_phase_data(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compose phase data for each channel, aligned to common time base.

        OMAS implementation:
        - Transposes phase data (.T)
        - Aligns all channels to first channel's time
        - Filters invalid values (abs > 2e3)

        Returns ragged array with shape: (n_channels, var * n_frequencies, n_time)
        Channels can have different numbers of frequencies or be missing entirely.
        """
        time = self._get_aligned_time(shot, raw_data)
        if time is None:
            return ak.Array([])

        phase_arrays = []
        for channel in self.channels:
            identifier = channel['identifier']
            path_prefix = f'\\ELECTRONS::TOP.REFLECT.{identifier}.PROCESSED:'
            time_key = Requirement(f'{path_prefix}TIME', shot, self.tree).as_key()
            phase_key = Requirement(f'{path_prefix}PHASE', shot, self.tree).as_key()

            if (time_key in raw_data and not isinstance(raw_data[time_key], Exception) and
                phase_key in raw_data and not isinstance(raw_data[phase_key], Exception)):

                time_channel = np.array(raw_data[time_key]) / 1e3  # Convert ms to s
                phase = np.array(raw_data[phase_key]).T  # Transpose (OMAS does .T)

                # Align to common time base
                if np.ndim(phase) == 0 or len(phase) < 3:
                    # Not enough data - append empty array for this channel
                    phase_arrays.append(np.array([]))
                else:
                    # Find indices in common time that exist in this channel's time
                    it = np.minimum(np.searchsorted(time_channel, time), len(time_channel) - 1)
                    phase_aligned = phase[it]

                    # Filter invalid values (OMAS: invalid = abs(phase) > 2e3, set to 0)
                    invalid = np.any(np.abs(phase_aligned) > 2e3, axis=1)
                    phase_aligned[invalid] = 0

                    phase_arrays.append(phase_aligned.T)  # Transpose: (n_frequencies, n_time)
            else:
                # Channel missing - append empty array
                phase_arrays.append(np.array([]))

        return ak.Array(phase_arrays)

    def _compose_position_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose R positions for density profile.

        OMAS aligns full profile time to channel time base.
        Returns shape: (n_radial, n_time)
        """
        time = self._get_aligned_time(shot, raw_data)

        full_prof_prefix = '\\ELECTRONS::TOP.REFLECT.FULL_PROF:'
        full_time_key = Requirement(f'{full_prof_prefix}TIME', shot, self.tree).as_key()
        full_R_key = Requirement(f'{full_prof_prefix}R', shot, self.tree).as_key()

        if (full_time_key not in raw_data or isinstance(raw_data[full_time_key], Exception) or
            full_R_key not in raw_data or isinstance(raw_data[full_R_key], Exception)):
            return np.array([])

        _time = np.array(raw_data[full_time_key]) / 1e3  # Convert ms to s
        R = np.array(raw_data[full_R_key]).T  # Transpose

        if np.ndim(_time) == 0 or len(_time) < 3:
            return np.array([])

        if time is None:
            # No channel time - use all times
            it = np.arange(_time.shape[0])
        else:
            # Align to channel time using argmin(abs difference)
            it = np.argmin(np.abs(_time - time[:, None]), axis=1)

        return R[it].T  # Transpose back

    def _compose_position_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose Z positions for density profile.

        Z is constant (self.position_z), broadcasted to match R shape.
        Returns shape: (n_radial, n_time)
        """
        R = self._compose_position_r(shot, raw_data)
        if R.size == 0:
            return np.array([])

        # Z is constant, same shape as R (OMAS: 0.0254 + 0 * R)
        return self.position_z + 0 * R

    def _compose_n_e_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose time base for density profile.

        Returns time indices aligned to channel time.
        """
        time = self._get_aligned_time(shot, raw_data)

        full_prof_prefix = '\\ELECTRONS::TOP.REFLECT.FULL_PROF:'
        full_time_key = Requirement(f'{full_prof_prefix}TIME', shot, self.tree).as_key()

        if full_time_key not in raw_data or isinstance(raw_data[full_time_key], Exception):
            return np.array([])

        _time = np.array(raw_data[full_time_key]) / 1e3  # Convert ms to s

        if np.ndim(_time) == 0 or len(_time) < 3:
            return np.array([])

        if time is None:
            # No channel time - use all times
            it = np.arange(_time.shape[0])
        else:
            # Align to channel time
            it = np.argmin(np.abs(_time - time[:, None]), axis=1)

        return _time[it]

    def _compose_n_e_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose electron density profile data.

        Returns shape: (n_radial, n_time)
        """
        time = self._get_aligned_time(shot, raw_data)

        full_prof_prefix = '\\ELECTRONS::TOP.REFLECT.FULL_PROF:'
        full_time_key = Requirement(f'{full_prof_prefix}TIME', shot, self.tree).as_key()
        full_density_key = Requirement(f'{full_prof_prefix}DENSITY', shot, self.tree).as_key()

        if (full_time_key not in raw_data or isinstance(raw_data[full_time_key], Exception) or
            full_density_key not in raw_data or isinstance(raw_data[full_density_key], Exception)):
            return np.array([])

        _time = np.array(raw_data[full_time_key]) / 1e3  # Convert ms to s
        density = np.array(raw_data[full_density_key]).T  # Transpose

        if np.ndim(_time) == 0 or len(_time) < 3:
            return np.array([])

        if time is None:
            # No channel time - use all times
            it = np.arange(_time.shape[0])
        else:
            # Align to channel time
            it = np.argmin(np.abs(_time - time[:, None]), axis=1)

        return density[it].T  # Transpose back

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
