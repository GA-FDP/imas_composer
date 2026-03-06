"""
Interferometer IDS Mapping for DIII-D CO2 Interferometer and RIP

See OMAS: omas/machine_mappings/d3d.py::interferometer_polarimeter_hardware
and interferometer_polarimeter_data
"""

from typing import Dict, List
import numpy as np
from scipy.interpolate import interp1d

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class InterferometerMapper(IDSMapper):
    """
    Maps DIII-D interferometer data to IMAS interferometer IDS.

    Supports two systems:
    - CO2 Interferometer (BCI tree): 4 channels (r0, v1, v2, v3)
    - Radial Interferometer Polarimeter (RIP tree): 3-4 channels (Z, P, N, [T])
      - RIP only available for shots >= 168823
      - T channel only available for shots > 202680
    """

    # Configuration YAML file contains static values and field ledger
    DOCS_PATH = "interferometer.yaml"
    CONFIG_PATH = "interferometer.yaml"

    # CO2 channel names
    CO2_CHANNEL_NAMES = ['r0', 'v1', 'v2', 'v3']
    N_CO2_CHANNELS = 4

    # RIP channel names (varies by shot)
    RIP_CHANNEL_NAMES_BASE = ['Z', 'P', 'N']
    RIP_CHANNEL_NAME_T = 'T'

    def __init__(self, include_rip: bool = False, **kwargs):
        """
        Initialize Interferometer mapper.

        Args:
            include_rip: If True, include RIP channels in addition to CO2 channels
            **kwargs: Additional parameters passed to base class
        """
        self.include_rip = include_rip

        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__(**kwargs)

        # Build IDS specs
        self._build_specs()

    def _get_bci_tree(self, pulse: int) -> str:
        """Determine BCI tree path based on pulse number."""
        if pulse <= 197528:
            return "BCI::TOP"
        else:
            return "BCI::TOP.MAIN"

    def _get_rip_channel_names(self, pulse: int) -> List[str]:
        """
        Get RIP channel names based on shot number.

        Returns empty list if shot < 168823 (no RIP available).
        """
        if pulse < 168823:
            return []

        channels = self.RIP_CHANNEL_NAMES_BASE.copy()
        if pulse > 202680:
            channels.append(self.RIP_CHANNEL_NAME_T)

        return channels

    def _get_rip_identifier(self, channel: str, pulse: int) -> str:
        """
        Get RIP channel identifier based on shot number.

        Identifier format changes at pulse 177052.
        """
        if pulse < 177052:
            # Old format: RPICH{n}PHI[S]
            idx = self.RIP_CHANNEL_NAMES_BASE.index(channel) if channel in self.RIP_CHANNEL_NAMES_BASE else 3
            identifier = f'RPICH{2*idx+2}PHI'
            if pulse > 169007:
                identifier += 'S'
            return identifier
        else:
            # New format: RIP{channel}
            return f'RIP{channel}'

    def _build_specs(self):
        """Build all IDS entry specifications"""

        # CO2 measurement data requirements
        for i, name in enumerate(self.CO2_CHANNEL_NAMES):
            identifier = name.upper()
            self.specs[f"interferometer._co2_channel_{i}_density"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                depends_on=[],
                derive_requirements=lambda shot, raw, ident=identifier:
                    [Requirement(f"\\{self._get_bci_tree(shot)}.DEN{ident}", shot, 'BCI')],
                ids_path=f"interferometer._co2_channel_{i}_density",
                docs_file=self.DOCS_PATH
            )

            self.specs[f"interferometer._co2_channel_{i}_validity"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                depends_on=[],
                derive_requirements=lambda shot, raw, ident=identifier:
                    [Requirement(f"\\{self._get_bci_tree(shot)}.STAT{ident}", shot, 'BCI')],
                ids_path=f"interferometer._co2_channel_{i}_validity",
                docs_file=self.DOCS_PATH
            )

        # CO2 time bases
        self.specs["interferometer._co2_time"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=[],
            derive_requirements=lambda shot, raw:
                [Requirement(f"dim_of(\\{self._get_bci_tree(shot)}.DENR0)", shot, 'BCI')],
            ids_path="interferometer._co2_time",
            docs_file=self.DOCS_PATH
        )

        self.specs["interferometer._co2_time_valid"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=[],
            derive_requirements=lambda shot, raw:
                [Requirement(f"dim_of(\\{self._get_bci_tree(shot)}.STATR0)", shot, 'BCI')],
            ids_path="interferometer._co2_time_valid",
            docs_file=self.DOCS_PATH
        )

        # RIP measurement data requirements (if enabled)
        if self.include_rip:
            # RIP requirements are DERIVED based on shot number
            self.specs["interferometer._rip_data"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                depends_on=[],
                derive_requirements=self._derive_rip_requirements,
                ids_path="interferometer._rip_data",
                docs_file=self.DOCS_PATH
            )

            self.specs["interferometer._rip_time"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                depends_on=[],
                derive_requirements=self._derive_rip_time_requirement,
                ids_path="interferometer._rip_time",
                docs_file=self.DOCS_PATH
            )

        # Channel metadata (arrays across all channels)
        self.specs["interferometer.channel.identifier"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_channel_identifier,
            ids_path="interferometer.channel.identifier",
            docs_file=self.DOCS_PATH
        )

        self.specs["interferometer.channel.name"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_channel_name,
            ids_path="interferometer.channel.name",
            docs_file=self.DOCS_PATH
        )

        # Line of sight geometry (arrays across all channels)
        self.specs["interferometer.channel.line_of_sight.first_point.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_first_point_r,
            ids_path="interferometer.channel.line_of_sight.first_point.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["interferometer.channel.line_of_sight.first_point.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_first_point_z,
            ids_path="interferometer.channel.line_of_sight.first_point.z",
            docs_file=self.DOCS_PATH
        )

        self.specs["interferometer.channel.line_of_sight.first_point.phi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_first_point_phi,
            ids_path="interferometer.channel.line_of_sight.first_point.phi",
            docs_file=self.DOCS_PATH
        )

        self.specs["interferometer.channel.line_of_sight.second_point.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_second_point_r,
            ids_path="interferometer.channel.line_of_sight.second_point.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["interferometer.channel.line_of_sight.second_point.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_second_point_z,
            ids_path="interferometer.channel.line_of_sight.second_point.z",
            docs_file=self.DOCS_PATH
        )

        self.specs["interferometer.channel.line_of_sight.second_point.phi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_second_point_phi,
            ids_path="interferometer.channel.line_of_sight.second_point.phi",
            docs_file=self.DOCS_PATH
        )

        self.specs["interferometer.channel.line_of_sight.third_point.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_third_point_r,
            ids_path="interferometer.channel.line_of_sight.third_point.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["interferometer.channel.line_of_sight.third_point.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_third_point_z,
            ids_path="interferometer.channel.line_of_sight.third_point.z",
            docs_file=self.DOCS_PATH
        )

        self.specs["interferometer.channel.line_of_sight.third_point.phi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_third_point_phi,
            ids_path="interferometer.channel.line_of_sight.third_point.phi",
            docs_file=self.DOCS_PATH
        )

        # Wavelength (array across all channels)
        self.specs["interferometer.channel.wavelength.0.value"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_wavelength,
            ids_path="interferometer.channel.wavelength.0.value",
            docs_file=self.DOCS_PATH
        )

        # Measurement data (arrays across all channels)
        self.specs["interferometer.channel.n_e_line.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["interferometer._co2_time"] + (["interferometer._rip_time"] if self.include_rip else []),
            compose=self._compose_n_e_line_time,
            ids_path="interferometer.channel.n_e_line.time",
            docs_file=self.DOCS_PATH
        )

        # Density data depends on all channel density requirements
        density_deps = [f"interferometer._co2_channel_{i}_density" for i in range(self.N_CO2_CHANNELS)]
        if self.include_rip:
            density_deps.append("interferometer._rip_data")
            density_deps.append("interferometer._rip_time")

        self.specs["interferometer.channel.n_e_line.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=density_deps,
            compose=self._compose_n_e_line_data,
            ids_path="interferometer.channel.n_e_line.data",
            docs_file=self.DOCS_PATH
        )

        # Error estimate depends on density, validity, and time data
        error_deps = density_deps.copy()
        error_deps.extend([f"interferometer._co2_channel_{i}_validity" for i in range(self.N_CO2_CHANNELS)])
        error_deps.extend(["interferometer._co2_time", "interferometer._co2_time_valid"])

        self.specs["interferometer.channel.n_e_line.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=error_deps,
            compose=self._compose_n_e_line_data_error_upper,
            ids_path="interferometer.channel.n_e_line.data_error_upper",
            docs_file=self.DOCS_PATH
        )

        # Validity depends on all channel validity requirements plus time bases
        validity_deps = [f"interferometer._co2_channel_{i}_validity" for i in range(self.N_CO2_CHANNELS)]
        validity_deps.extend(["interferometer._co2_time", "interferometer._co2_time_valid"])
        if self.include_rip:
            validity_deps.extend(["interferometer._rip_data", "interferometer._rip_time"])

        self.specs["interferometer.channel.n_e_line.validity_timed"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=validity_deps,
            compose=self._compose_n_e_line_validity_timed,
            ids_path="interferometer.channel.n_e_line.validity_timed",
            docs_file=self.DOCS_PATH
        )

    # RIP requirement derivation functions
    def _derive_rip_requirements(self, shot: int, raw_data: dict) -> List[Requirement]:
        """
        Derive RIP data requirements based on shot number.

        Returns empty list if shot < 168823 (no RIP available).
        """
        rip_channels = self._get_rip_channel_names(shot)
        if not rip_channels:
            return []

        requirements = []
        tree = 'RPI'

        # Interferometer measurements: a1, a3, b1, b3
        interferometer_measurements = ['a1', 'a3', 'b1', 'b3']

        for ch in rip_channels:
            identifier = self._get_rip_identifier(ch, shot)

            if shot < 177052:
                requirements.append(Requirement(f"\\{tree}::{identifier}", shot, tree))
            else:
                for m in interferometer_measurements:
                    requirements.append(Requirement(f"\\{tree}::{identifier}{m}PHIS", shot, tree))

        return requirements

    def _derive_rip_time_requirement(self, shot: int, raw_data: dict) -> List[Requirement]:
        """Derive RIP time requirements for all channels."""
        rip_channels = self._get_rip_channel_names(shot)
        if not rip_channels:
            return []

        tree = 'RPI'
        requirements = []

        for ch in rip_channels:
            identifier = self._get_rip_identifier(ch, shot)

            if shot < 177052:
                time_path = f"\\{tree}::{identifier}"
            else:
                time_path = f"\\{tree}::{identifier}A1PHIS"

            requirements.append(Requirement(f"dim_of({time_path})", shot, tree))

        return requirements

    def _process_rip_phase_data(self, shot: int, raw_data: dict, channel: str):
        """
        Process RIP phase data for a single channel, applying all corrections.

        Returns:
            tuple: (phase, phase_error) where phase_error is None for shots < 177052
        """
        tree = 'RPI'
        identifier = self._get_rip_identifier(channel, shot)
        interferometer_measurements = ['a1', 'a3', 'b1', 'b3']
        n_int = len(interferometer_measurements)

        # Get time and ioff
        if shot < 177052:
            time_path = f"\\{tree}::{identifier}"
        else:
            time_path = f"\\{tree}::{identifier}A1PHIS"
        time_key = Requirement(f"dim_of({time_path})", shot, tree).as_key()
        time_ms = raw_data[time_key]
        time_s = time_ms / 1e3
        ioff = np.searchsorted(time_s, 0)

        if shot < 177052:
            # Old format - single measurement (no error estimate)
            rip_key = Requirement(f"\\{tree}::{identifier}", shot, tree).as_key()
            phase_data = raw_data[rip_key].copy()

            # Offset correction: subtract mean before t=0
            if ioff > 1:
                offset = phase_data[1:ioff].mean()
                phase_data -= offset

            # Sign correction: ensure positivity
            phase = phase_data * np.sign(phase_data.mean())
            phase_error = None
        else:
            # New format - process 4 measurements with unwrapping and fringe corrections
            phases = np.zeros([n_int, len(time_s)])

            for i, m in enumerate(interferometer_measurements):
                rip_key = Requirement(f"\\{tree}::{identifier}{m}PHIS", shot, tree).as_key()
                phase_data = raw_data[rip_key].copy()

                # Offset correction: subtract mean before t=0
                if ioff > 1:
                    offset = phase_data[1:ioff].mean()
                    phase_data -= offset

                # Sign correction: ensure positivity
                phases[i] = phase_data * np.sign(phase_data.mean())

            # Phase unwrapping (default axis=-1 = time axis)
            phases = np.rad2deg(np.unwrap(np.deg2rad(phases)))

            # Fringe jump corrections
            # Subtract time-median from each measurement, then unwrap the result
            jumps = phases - np.median(phases, axis=1, keepdims=True)
            jumps = jumps - np.rad2deg(np.unwrap(np.deg2rad(jumps)))
            phases -= jumps

            # Use median across measurements (axis 0) for each time point
            phase = np.median(phases, axis=0)

            # Calculate phase error (std across measurements + mean std before t=0)
            phase_error = phases.std(axis=0) + phases[:, 1:ioff].std(axis=1).mean()

        return phase, phase_error

    # Helper methods for geometry
    def _get_co2_geometry(self, channel_idx: int) -> dict:
        """
        Get hardcoded CO2 geometry for a channel.

        As of 2018 June 07, DIII-D has four CO2 interferometers:
        - r0: Radial chord at midplane
        - v1, v2, v3: Vertical chords at different radii

        Endpoints are approximative from OMFITprofiles.
        """
        if channel_idx == 0:  # r0 - radial chord
            return {
                'first_point': {'r': 2.36, 'z': 0.0, 'phi': 225 * (-np.pi / 180.0)},
                'second_point': {'r': 1.01, 'z': 0.0, 'phi': 225 * (-np.pi / 180.0)}
            }
        else:  # v1, v2, v3 - vertical chords
            r_values = [1.48, 1.94, 2.10]
            r = r_values[channel_idx - 1]
            Z_top = 1.24
            Z_bottom = -1.375
            return {
                'first_point': {'r': r, 'z': Z_top, 'phi': 240 * (-np.pi / 180.0)},
                'second_point': {'r': r, 'z': Z_bottom, 'phi': 240 * (-np.pi / 180.0)}
            }

    def _get_rip_geometry(self, channel: str, shot: int) -> dict:
        """
        Get hardcoded RIP geometry for a channel.

        From IDA-lite:
        https://github.com/GA-IDA/ida_lite/blob/c1398c826b7a327d6629b5518c3219b8870436ce/D3D/synt_diags/RIP.py#L47
        """
        Rin = 1.017
        Rout = 2.36
        z_positions = {'Z': 0.0, 'P': 0.135, 'N': -0.135, 'T': 0.0}
        z = z_positions[channel]

        # T channel has different phi angle
        if channel == 'T':
            phi = 283 * (-np.pi / 180.0)
        else:
            phi = 286.6 * (-np.pi / 180.0)

        return {
            'first_point': {'r': Rout, 'z': z, 'phi': phi},
            'second_point': {'r': Rin, 'z': z, 'phi': phi}
        }

    def _get_rip_conversion_factor(self, channel: str) -> float:
        """Get RIP phase-to-density conversion factor."""
        if channel == 'T':
            return 1.436
        else:
            return 6.71e15  # m^-2/rad

    # Compose functions - Channel metadata
    def _compose_channel_identifier(self, shot: int, raw_data: dict) -> np.ndarray:
        """Return array of channel identifiers for all systems."""
        identifiers = self.CO2_CHANNEL_NAMES.copy()

        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            for ch in rip_channels:
                identifiers.append(self._get_rip_identifier(ch, shot))

        return np.array(identifiers)

    def _compose_channel_name(self, shot: int, raw_data: dict) -> np.ndarray:
        """Return array of channel names for all systems."""
        names = self.CO2_CHANNEL_NAMES.copy()

        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            for ch in rip_channels:
                names.append(self._get_rip_identifier(ch, shot))

        return np.array(names)

    # Compose functions - Line of sight geometry
    def _compose_first_point_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """Return array of first point R coordinates for all channels."""
        r_values = [self._get_co2_geometry(i)['first_point']['r'] for i in range(self.N_CO2_CHANNELS)]

        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            for ch in rip_channels:
                r_values.append(self._get_rip_geometry(ch, shot)['first_point']['r'])

        return np.array(r_values)

    def _compose_first_point_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """Return array of first point Z coordinates for all channels."""
        z_values = [self._get_co2_geometry(i)['first_point']['z'] for i in range(self.N_CO2_CHANNELS)]

        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            for ch in rip_channels:
                z_values.append(self._get_rip_geometry(ch, shot)['first_point']['z'])

        return np.array(z_values)

    def _compose_first_point_phi(self, shot: int, raw_data: dict) -> np.ndarray:
        """Return array of first point phi coordinates for all channels."""
        phi_values = [self._get_co2_geometry(i)['first_point']['phi'] for i in range(self.N_CO2_CHANNELS)]

        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            for ch in rip_channels:
                phi_values.append(self._get_rip_geometry(ch, shot)['first_point']['phi'])

        return np.array(phi_values)

    def _compose_second_point_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """Return array of second point R coordinates for all channels."""
        r_values = [self._get_co2_geometry(i)['second_point']['r'] for i in range(self.N_CO2_CHANNELS)]

        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            for ch in rip_channels:
                r_values.append(self._get_rip_geometry(ch, shot)['second_point']['r'])

        return np.array(r_values)

    def _compose_second_point_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """Return array of second point Z coordinates for all channels."""
        z_values = [self._get_co2_geometry(i)['second_point']['z'] for i in range(self.N_CO2_CHANNELS)]

        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            for ch in rip_channels:
                z_values.append(self._get_rip_geometry(ch, shot)['second_point']['z'])

        return np.array(z_values)

    def _compose_second_point_phi(self, shot: int, raw_data: dict) -> np.ndarray:
        """Return array of second point phi coordinates for all channels."""
        phi_values = [self._get_co2_geometry(i)['second_point']['phi'] for i in range(self.N_CO2_CHANNELS)]

        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            for ch in rip_channels:
                phi_values.append(self._get_rip_geometry(ch, shot)['second_point']['phi'])

        return np.array(phi_values)

    def _compose_third_point_r(self, shot: int, raw_data: dict):
        """
        Return array of third point R coordinates.

        CO2: Same as first point (closed line of sight) - 4 values
        RIP: Empty (only has 2 points) - 0 values per channel

        Returns awkward array with variable length per channel.
        """
        import awkward as ak

        # CO2 channels: third_point exists (same as first_point)
        r_values = [self._get_co2_geometry(i)['first_point']['r'] for i in range(self.N_CO2_CHANNELS)]

        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            # RIP channels: third_point doesn't exist (only 2 points)
            # Don't add anything for RIP channels
            pass

        return ak.Array(r_values)

    def _compose_third_point_z(self, shot: int, raw_data: dict):
        """
        Return array of third point Z coordinates.

        CO2: Same as first point (closed line of sight) - 4 values
        RIP: Empty (only has 2 points) - 0 values per channel

        Returns awkward array with variable length per channel.
        """
        import awkward as ak

        # CO2 channels: third_point exists (same as first_point)
        z_values = [self._get_co2_geometry(i)['first_point']['z'] for i in range(self.N_CO2_CHANNELS)]

        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            # RIP channels: third_point doesn't exist (only 2 points)
            # Don't add anything for RIP channels
            pass

        return ak.Array(z_values)

    def _compose_third_point_phi(self, shot: int, raw_data: dict):
        """
        Return array of third point phi coordinates.

        CO2: Same as first point (closed line of sight) - 4 values
        RIP: Empty (only has 2 points) - 0 values per channel

        Returns awkward array with variable length per channel.
        """
        import awkward as ak

        # CO2 channels: third_point exists (same as first_point)
        phi_values = [self._get_co2_geometry(i)['first_point']['phi'] for i in range(self.N_CO2_CHANNELS)]

        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            # RIP channels: third_point doesn't exist (only 2 points)
            # Don't add anything for RIP channels
            pass

        return ak.Array(phi_values)

    def _compose_wavelength(self, shot: int, raw_data: dict):
        """
        Return array of wavelengths: CO2 = 10.6 μm, RIP = 461.5 μm.

        Since wavelength is an AOS (Array of Structures), we need to return
        an array per channel. Each channel has one wavelength measurement,
        so we return shape (n_channels, 1).

        Returns awkward array with shape (n_channels, 1).
        """
        import awkward as ak

        # Each channel has one wavelength, but wavelength is an AOS
        # So we wrap each value in a list
        wavelengths = [[10.6e-6]] * self.N_CO2_CHANNELS  # CO2 laser

        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            wavelengths.extend([[461.5e-6]] * len(rip_channels))  # RIP laser

        return ak.Array(wavelengths)

    # Compose functions - Measurement data
    def _compose_n_e_line_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Return time array for all channels.

        CO2 channels share same time base. RIP channels share their own time base.

        Returns:
            Array of shape (n_channels, n_time) - note that time arrays may be ragged
        """
        bci_tree = self._get_bci_tree(shot)
        time_key = Requirement(f"dim_of(\\{bci_tree}.DENR0)", shot, 'BCI').as_key()
        co2_time_ms = raw_data[time_key]
        co2_time_s = co2_time_ms / 1.0e3  # Convert ms to seconds

        # All CO2 channels share the same time
        all_times = [co2_time_s] * self.N_CO2_CHANNELS

        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            if rip_channels:
                # Get RIP time
                tree = 'RPI'
                ch = rip_channels[0]
                identifier = self._get_rip_identifier(ch, shot)

                if shot < 177052:
                    time_path = f"\\{tree}::{identifier}"
                else:
                    time_path = f"\\{tree}::{identifier}A1PHIS"

                rip_time_key = Requirement(f"dim_of({time_path})", shot, tree).as_key()
                rip_time_ms = raw_data[rip_time_key]
                rip_time_s = rip_time_ms / 1.0e3

                # All RIP channels share the same time
                all_times.extend([rip_time_s] * len(rip_channels))

        # Return as awkward array since time bases may differ
        import awkward as ak
        return ak.Array(all_times)

    def _compose_n_e_line_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Return line-integrated density data for all channels.

        Returns:
            Array of shape (n_channels, n_time) or awkward array if time bases differ
        """
        bci_tree = self._get_bci_tree(shot)
        all_data = []

        # CO2 data
        for name in self.CO2_CHANNEL_NAMES:
            identifier = name.upper()
            density_key = Requirement(f"\\{bci_tree}.DEN{identifier}", shot, 'BCI').as_key()
            density = raw_data[density_key]
            all_data.append(density * 1e6)

        # RIP data (if enabled and available)
        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            if rip_channels:
                for ch in rip_channels:
                    # Process phase data using helper method
                    phase, phase_error = self._process_rip_phase_data(shot, raw_data, ch)

                    # Apply conversion factor (phase to density)
                    conv = self._get_rip_conversion_factor(ch)
                    all_data.append(phase * conv)

        # Return as awkward array if time bases differ
        import awkward as ak
        return ak.Array(all_data)

    def _compose_n_e_line_data_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Return upper error estimate for line-integrated density data.

        For CO2 channels: Error = median of absolute values before t=0 (pre-shot noise)
        For RIP channels with shot >= 177052: Error from std of 4 measurements
        For RIP channels with shot < 177052: No error estimate (empty arrays)

        Returns:
            Awkward array with shape matching n_e_line.data
        """
        import awkward as ak
        all_errors = []

        # CO2 channels: Error based on pre-shot noise level
        bci_tree = self._get_bci_tree(shot)

        # Get time array to find pre-shot region
        time_key = Requirement(f"dim_of(\\{bci_tree}.DENR0)", shot, 'BCI').as_key()
        time_ms = raw_data[time_key]
        time_s = time_ms / 1e3

        # Get validity for each CO2 channel
        time_valid_key = Requirement(f"dim_of(\\{bci_tree}.STATR0)", shot, 'BCI').as_key()
        time_valid_ms = raw_data[time_valid_key]
        for name in self.CO2_CHANNEL_NAMES:
            identifier = name.upper()
            density_key = Requirement(f"\\{bci_tree}.DEN{identifier}", shot, 'BCI').as_key()
            density = raw_data[density_key]

            # Get validity for this channel
            validity_key = Requirement(f"\\{bci_tree}.STAT{identifier}", shot, 'BCI').as_key()
            validity = raw_data[validity_key]

            # Interpolate validity to match data time base
            from scipy.interpolate import interp1d
            validity_interp = interp1d(
                time_valid_ms / 1.0e3,
                -validity,
                kind='nearest',
                bounds_error=False,
                fill_value='extrapolate',
                assume_sorted=True,
            )(time_s)

            # Calculate error as median of absolute values before t=0
            ne_err = np.zeros(density.shape)
            ne_err[:] = np.std(np.abs(density[np.logical_and(time_s < 0, time_s > -0.5)])) * 1e6

            # Set error to inf where validity is negative (invalid)
            ne_err[validity_interp < 0] = np.inf

            all_errors.append(ne_err)

        # RIP data (if enabled and available)
        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            if rip_channels:
                for ch in rip_channels:
                    # Process phase data using helper method
                    phase, phase_error = self._process_rip_phase_data(shot, raw_data, ch)

                    if phase_error is not None:
                        # Convert phase error to density error
                        conv = self._get_rip_conversion_factor(ch)
                        ne_line_err = phase_error * conv

                        # Also need to get n_e_line data to set invalid points to inf
                        ne_line = phase * conv

                        # Set error to inf where data is negative (invalid)
                        ne_line_err[ne_line < 0] = np.inf

                        all_errors.append(ne_line_err)
                    else:
                        # Old shots (< 177052): no error estimate
                        all_errors.append(np.array([]))

        return ak.Array(all_errors)

    def _compose_n_e_line_validity_timed(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Return interpolated validity data for all channels.

        Returns:
            Array of shape (n_channels, n_time) or awkward array if time bases differ
        """
        bci_tree = self._get_bci_tree(shot)

        # Get CO2 time bases
        time_key = Requirement(f"dim_of(\\{bci_tree}.DENR0)", shot, 'BCI').as_key()
        time_valid_key = Requirement(f"dim_of(\\{bci_tree}.STATR0)", shot, 'BCI').as_key()
        time_ms = raw_data[time_key]
        time_valid_ms = raw_data[time_valid_key]

        all_validity = []
        # CO2 validity
        for name in self.CO2_CHANNEL_NAMES:
            identifier = name.upper()
            validity_key = Requirement(f"\\{bci_tree}.STAT{identifier}", shot, 'BCI').as_key()
            validity = raw_data[validity_key]

            # Interpolate validity to match measurement time base
            # Note: OMAS uses negative validity
            validity_interp = interp1d(
                time_valid_ms / 1.0e3,
                -validity,
                kind='nearest',
                bounds_error=False,
                fill_value='extrapolate',
                assume_sorted=True,
            )(time_ms / 1.0e3)

            all_validity.append(validity_interp)

        # RIP validity (if enabled and available)
        if self.include_rip:
            rip_channels = self._get_rip_channel_names(shot)
            if rip_channels:
                for ch in rip_channels:
                    # Process phase data using helper method
                    phase, phase_error = self._process_rip_phase_data(shot, raw_data, ch)

                    # Convert to density
                    conv = self._get_rip_conversion_factor(ch)
                    ne_line = phase * conv

                    # Compute validity based on ne_line values (matching OMAS logic)
                    n_time = len(ne_line)
                    valid = np.zeros(n_time, dtype=int)
                    if ne_line.mean() < 1e18:
                        # Issue with old discharges - wrong calibration
                        valid[:] = -1
                    else:
                        # Enforce positivity: mark negative values as invalid
                        valid[ne_line < 0] = -1

                    all_validity.append(valid)

        # Return as awkward array if time bases differ
        import awkward as ak
        return ak.Array(all_validity)

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        """Return all IDS entry specifications"""
        return self.specs
