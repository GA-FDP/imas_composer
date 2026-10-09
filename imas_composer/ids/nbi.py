"""
NBI (Neutral Beam Injection) IDS Mapping for DIII-D

See OMAS: omas/machine_mappings/d3d.py::nbi_active_hardware
"""

from typing import Dict, List
import numpy as np

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class NbiMapper(IDSMapper):
    """Maps DIII-D NBI data to IMAS nbi IDS."""

    # Configuration YAML file contains static values and field ledger
    DOCS_PATH = "nbi.yaml"
    CONFIG_PATH = "nbi.yaml"

    def __init__(self, **kwargs):
        """Initialize NBI mapper."""
        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Load beam names and GAS -> species lookup from config
        config = self._load_config()
        self.beam_names = config.get('beam_names', [])
        self.gas_species = config.get('gas_species', {})

        # Build IDS specs
        self._build_specs()

    def _build_specs(self):
        """Build all IDS entry specifications"""

        # Internal dependencies - fetch data for all beams

        # Fetch PINJ (power) and time for all beams
        self.specs["nbi._pinj_data"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._create_beam_requirements("PINJ"),
            ids_path="nbi._pinj_data",
            docs_file=self.DOCS_PATH
        )

        self.specs["nbi._pinj_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._create_time_requirements("PINJ"),
            ids_path="nbi._pinj_time",
            docs_file=self.DOCS_PATH
        )

        # Fetch VBEAM (beam voltage/energy) for all beams
        self.specs["nbi._vbeam_data"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._create_beam_requirements("VBEAM"),
            ids_path="nbi._vbeam_data",
            docs_file=self.DOCS_PATH
        )

        # Fetch GAS (species information) for all beams
        self.specs["nbi._gas_data"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._create_beam_requirements("GAS"),
            ids_path="nbi._gas_data",
            docs_file=self.DOCS_PATH
        )

        # Fetch FIRED (whether the beam actually injected) for all beams
        self.specs["nbi._fired_data"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._create_beam_requirements("FIRED"),
            ids_path="nbi._fired_data",
            docs_file=self.DOCS_PATH
        )

        # Public IDS fields - all COMPUTED stage

        self.specs["nbi.unit.name"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["nbi._pinj_time", "nbi._fired_data"],
            compose=self._compose_unit_name,
            ids_path="nbi.unit.name",
            docs_file=self.DOCS_PATH
        )

        self.specs["nbi.unit.power_launched.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["nbi._pinj_time", "nbi._fired_data"],
            compose=self._compose_power_launched_time,
            ids_path="nbi.unit.power_launched.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["nbi.unit.power_launched.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["nbi._pinj_time", "nbi._pinj_data", "nbi._fired_data"],
            compose=self._compose_power_launched_data,
            ids_path="nbi.unit.power_launched.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["nbi.unit.energy.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["nbi._pinj_time", "nbi._fired_data"],
            compose=self._compose_energy_time,
            ids_path="nbi.unit.energy.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["nbi.unit.energy.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["nbi._pinj_time", "nbi._vbeam_data", "nbi._fired_data"],
            compose=self._compose_energy_data,
            ids_path="nbi.unit.energy.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["nbi.unit.species.a"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["nbi._pinj_time", "nbi._gas_data", "nbi._fired_data"],
            compose=self._compose_species_a,
            ids_path="nbi.unit.species.a",
            docs_file=self.DOCS_PATH
        )

        self.specs["nbi.unit.species.z_n"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["nbi._pinj_time", "nbi._gas_data", "nbi._fired_data"],
            compose=self._compose_species_z_n,
            ids_path="nbi.unit.species.z_n",
            docs_file=self.DOCS_PATH
        )

        self.specs["nbi.unit.species.label"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["nbi._pinj_time", "nbi._gas_data", "nbi._fired_data"],
            compose=self._compose_species_label,
            ids_path="nbi.unit.species.label",
            docs_file=self.DOCS_PATH
        )

    def _create_beam_requirements(self, field: str) -> List[Requirement]:
        """
        Create requirements for a beam field (PINJ, VBEAM, GAS, etc.).

        Args:
            field: Field name (e.g., 'PINJ', 'VBEAM', 'GAS')

        Returns:
            List of requirements for all beams
        """
        requirements = []
        for beam_name in self.beam_names:
            if field in ["PINJ", "TINJ"]:
                # Power and time injection use beam-specific naming
                mds_path = f'\\NB::TOP.NB{beam_name}.{field}_{beam_name}'
            else:
                # Other fields use simpler naming
                mds_path = f'\\NB::TOP.NB{beam_name}.{field}'
            requirements.append(Requirement(mds_path, 0, 'NB'))

        return requirements

    def _create_time_requirements(self, field: str) -> List[Requirement]:
        """
        Create requirements for time dimension of a field.

        Args:
            field: Field name (e.g., 'PINJ')

        Returns:
            List of time requirements for all beams
        """
        requirements = []
        for beam_name in self.beam_names:
            # Time is derived from the dimension of the field
            mds_path = f'dim_of(\\NB::TOP.NB{beam_name}.{field}_{beam_name}, 0)/1E3'
            requirements.append(Requirement(mds_path, 0, 'NB'))

        return requirements

    def _get_active_beams(self, shot: int, raw_data: dict) -> List[str]:
        """
        Get list of active beam names (beams with FIRED == 1).

        Args:
            shot: Shot number
            raw_data: Raw data dictionary

        Returns:
            List of active beam names
        """
        active_beams = []
        for beam_name in self.beam_names:
            fired_path = f'\\NB::TOP.NB{beam_name}.FIRED'
            fired_key = Requirement(fired_path, shot, 'NB').as_key()

            # Check if beam actually fired
            if fired_key in raw_data and not isinstance(raw_data[fired_key], Exception):
                if raw_data[fired_key]:
                    active_beams.append(beam_name)

        return active_beams

    # Compose functions
    def _compose_unit_name(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose beam unit names."""
        active_beams = self._get_active_beams(shot, raw_data)
        return np.array(active_beams)

    def _compose_power_launched_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose power launched time arrays."""
        active_beams = self._get_active_beams(shot, raw_data)

        time_arrays = []
        for beam_name in active_beams:
            time_path = f'dim_of(\\NB::TOP.NB{beam_name}.PINJ_{beam_name}, 0)/1E3'
            time_key = Requirement(time_path, shot, 'NB').as_key()
            time_arrays.append(raw_data[time_key])

        return np.array(time_arrays)

    def _compose_power_launched_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose power launched data arrays."""
        active_beams = self._get_active_beams(shot, raw_data)

        power_arrays = []
        for beam_name in active_beams:
            power_path = f'\\NB::TOP.NB{beam_name}.PINJ_{beam_name}'
            power_key = Requirement(power_path, shot, 'NB').as_key()
            power_arrays.append(raw_data[power_key])

        return np.array(power_arrays)

    def _compose_energy_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose energy time arrays.

        Note: VBEAM uses same time base as PINJ (see OMAS line 657).
        """
        active_beams = self._get_active_beams(shot, raw_data)

        time_arrays = []
        for beam_name in active_beams:
            # VBEAM uses PINJ time (OMAS: data[f"{beam_name}.VBEAM_time"] = data[f"{beam_name}.PINJ_time"])
            time_path = f'dim_of(\\NB::TOP.NB{beam_name}.PINJ_{beam_name}, 0)/1E3'
            time_key = Requirement(time_path, shot, 'NB').as_key()
            time_arrays.append(raw_data[time_key])

        return np.array(time_arrays)

    def _compose_energy_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose energy data arrays.

        Note: If VBEAM is missing, defaults to 80 keV (see OMAS line 659).
        """
        active_beams = self._get_active_beams(shot, raw_data)

        energy_arrays = []
        for beam_name in active_beams:
            vbeam_path = f'\\NB::TOP.NB{beam_name}.VBEAM'
            vbeam_key = Requirement(vbeam_path, shot, 'NB').as_key()

            # Get time array for broadcasting if needed
            time_path = f'dim_of(\\NB::TOP.NB{beam_name}.PINJ_{beam_name}, 0)/1E3'
            time_key = Requirement(time_path, shot, 'NB').as_key()
            time = raw_data[time_key]

            # Check if VBEAM exists and is valid
            if vbeam_key in raw_data and not isinstance(raw_data[vbeam_key], Exception):
                energy = raw_data[vbeam_key]
            else:
                # Default to 80 keV when beam voltage is missing
                energy = np.zeros_like(time) + 80e3

            energy_arrays.append(energy)

        return np.array(energy_arrays)

    def _compose_species_a(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose species mass number (A).

        Note: Looked up from GAS string via config's gas_species (e.g., 'D2' -> 2).
        Every active beam has FIRED, so GAS is expected to be non-empty.
        """
        active_beams = self._get_active_beams(shot, raw_data)

        species_a = []
        for beam_name in active_beams:
            gas_path = f'\\NB::TOP.NB{beam_name}.GAS'
            gas_key = Requirement(gas_path, shot, 'NB').as_key()

            assert gas_key in raw_data and not isinstance(raw_data[gas_key], Exception), \
                f"{beam_name}: fired but GAS data is missing"
            gas = raw_data[gas_key].strip()
            assert gas in self.gas_species, f"Unexpected NBI GAS value: {gas!r}"
            species_a.append(self.gas_species[gas]['a'])

        return np.array(species_a)

    def _compose_species_z_n(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose species nuclear charge (z_n).

        Note: Looked up from GAS string via config's gas_species (e.g., 'D2' -> 1).
        Every active beam has FIRED, so GAS is expected to be non-empty.
        """
        active_beams = self._get_active_beams(shot, raw_data)

        species_z_n = []
        for beam_name in active_beams:
            gas_path = f'\\NB::TOP.NB{beam_name}.GAS'
            gas_key = Requirement(gas_path, shot, 'NB').as_key()

            assert gas_key in raw_data and not isinstance(raw_data[gas_key], Exception), \
                f"{beam_name}: fired but GAS data is missing"
            gas = raw_data[gas_key].strip()
            assert gas in self.gas_species, f"Unexpected NBI GAS value: {gas!r}"
            species_z_n.append(self.gas_species[gas]['z_n'])

        return np.array(species_z_n)

    def _compose_species_label(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose species label.

        Note: Looked up from GAS string via config's gas_species (e.g., 'D2' -> 'D').
        Every active beam has FIRED, so GAS is expected to be non-empty.
        """
        active_beams = self._get_active_beams(shot, raw_data)

        species_label = []
        for beam_name in active_beams:
            gas_path = f'\\NB::TOP.NB{beam_name}.GAS'
            gas_key = Requirement(gas_path, shot, 'NB').as_key()

            assert gas_key in raw_data and not isinstance(raw_data[gas_key], Exception), \
                f"{beam_name}: fired but GAS data is missing"
            gas = raw_data[gas_key].strip()
            assert gas in self.gas_species, f"Unexpected NBI GAS value: {gas!r}"
            species_label.append(self.gas_species[gas]['label'])

        return np.array(species_label)

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
