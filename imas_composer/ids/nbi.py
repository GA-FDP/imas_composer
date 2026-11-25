"""
NBI (Neutral Beam Injection) IDS Mapping for DIII-D

See OMAS: omas/machine_mappings/d3d.py::nbi_active_hardware
"""

from typing import Dict, List
import numpy as np
import awkward as ak

from ..core import RequirementStage, Requirement, IDSEntrySpec, NoData
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
            depends_on=[],
            compose=self._compose_unit_name,
            ids_path="nbi.unit.name",
            docs_file=self.DOCS_PATH
        )

        self.specs["nbi.unit.power_launched.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["nbi._pinj_time"],
            compose=self._compose_power_launched_time,
            ids_path="nbi.unit.power_launched.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["nbi.unit.power_launched.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["nbi._pinj_data"],
            compose=self._compose_power_launched_data,
            ids_path="nbi.unit.power_launched.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["nbi.unit.energy.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["nbi._pinj_time"],
            compose=self._compose_energy_time,
            ids_path="nbi.unit.energy.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["nbi.unit.energy.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["nbi._pinj_time", "nbi._vbeam_data"],
            compose=self._compose_energy_data,
            ids_path="nbi.unit.energy.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["nbi.unit.species.a"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["nbi._gas_data", "nbi._fired_data"],
            compose=self._compose_species_a,
            ids_path="nbi.unit.species.a",
            docs_file=self.DOCS_PATH
        )

        self.specs["nbi.unit.species.z_n"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["nbi._gas_data", "nbi._fired_data"],
            compose=self._compose_species_z_n,
            ids_path="nbi.unit.species.z_n",
            docs_file=self.DOCS_PATH
        )

        self.specs["nbi.unit.species.label"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["nbi._gas_data", "nbi._fired_data"],
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

    # Compose functions. Every beam in beam_names keeps its slot; a beam without
    # data (e.g. one that was not installed or did not fire) has empty arrays.
    def _compose_unit_name(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose beam unit names."""
        return np.array(self.beam_names)

    def _compose_power_launched_time(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose power launched time arrays."""
        time_arrays = []
        for beam_name in self.beam_names:
            time_path = f'dim_of(\\NB::TOP.NB{beam_name}.PINJ_{beam_name}, 0)/1E3'
            time = raw_data[Requirement(time_path, shot, 'NB').as_key()]
            time_arrays.append(np.array([]) if isinstance(time, NoData) else time)

        return ak.Array(time_arrays)

    def _compose_power_launched_data(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose power launched data arrays."""
        power_arrays = []
        for beam_name in self.beam_names:
            power_path = f'\\NB::TOP.NB{beam_name}.PINJ_{beam_name}'
            power = raw_data[Requirement(power_path, shot, 'NB').as_key()]
            power_arrays.append(np.array([]) if isinstance(power, NoData) else power)

        return ak.Array(power_arrays)

    def _compose_energy_time(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compose energy time arrays.

        Note: VBEAM uses same time base as PINJ (see OMAS line 657).
        """
        time_arrays = []
        for beam_name in self.beam_names:
            # VBEAM uses PINJ time (OMAS: data[f"{beam_name}.VBEAM_time"] = data[f"{beam_name}.PINJ_time"])
            time_path = f'dim_of(\\NB::TOP.NB{beam_name}.PINJ_{beam_name}, 0)/1E3'
            time = raw_data[Requirement(time_path, shot, 'NB').as_key()]
            time_arrays.append(np.array([]) if isinstance(time, NoData) else time)

        return ak.Array(time_arrays)

    def _compose_energy_data(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compose energy data arrays.

        Note: If VBEAM is missing, defaults to 80 keV (see OMAS line 659).
        """
        energy_arrays = []
        for beam_name in self.beam_names:
            vbeam_path = f'\\NB::TOP.NB{beam_name}.VBEAM'
            vbeam_key = Requirement(vbeam_path, shot, 'NB').as_key()

            # Get time array for broadcasting if needed
            time_path = f'dim_of(\\NB::TOP.NB{beam_name}.PINJ_{beam_name}, 0)/1E3'
            time = raw_data[Requirement(time_path, shot, 'NB').as_key()]
            if isinstance(time, NoData):
                energy_arrays.append(np.array([]))
                continue

            # Check if VBEAM exists and is valid
            if vbeam_key in raw_data and not isinstance(raw_data[vbeam_key], Exception):
                energy = raw_data[vbeam_key]
            else:
                # Default to 80 keV when beam voltage is missing
                energy = np.zeros_like(time) + 80e3

            energy_arrays.append(energy)

        return ak.Array(energy_arrays)

    def _compose_species_a(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose species mass number (A).

        Note: Looked up from GAS string via config's gas_species (e.g., 'D2' -> 2).
        A beam that did not fire gets NaN; a beam that fired is expected to have
        a GAS value.
        """
        species_a = []
        for beam_name in self.beam_names:
            fired = raw_data[Requirement(f'\\NB::TOP.NB{beam_name}.FIRED', shot, 'NB').as_key()]
            if isinstance(fired, NoData):
                return fired
            if not fired:
                # The GAS string of a beam that did not fire is not meaningful (e.g. '2O' on #118162)
                species_a.append(np.nan)
                continue
            gas = raw_data[Requirement(f'\\NB::TOP.NB{beam_name}.GAS', shot, 'NB').as_key()]
            assert not isinstance(gas, NoData) and gas.strip(), f"{beam_name}: fired but GAS data is missing"
            gas = gas.strip()
            assert gas in self.gas_species, f"Unexpected NBI GAS value: {gas!r}"
            species_a.append(self.gas_species[gas]['a'])

        return np.array(species_a)

    def _compose_species_z_n(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose species nuclear charge (z_n).

        Note: Looked up from GAS string via config's gas_species (e.g., 'D2' -> 1).
        A beam that did not fire gets NaN; a beam that fired is expected to have
        a GAS value.
        """
        species_z_n = []
        for beam_name in self.beam_names:
            fired = raw_data[Requirement(f'\\NB::TOP.NB{beam_name}.FIRED', shot, 'NB').as_key()]
            if isinstance(fired, NoData):
                return fired
            if not fired:
                # The GAS string of a beam that did not fire is not meaningful (e.g. '2O' on #118162)
                species_z_n.append(np.nan)
                continue
            gas = raw_data[Requirement(f'\\NB::TOP.NB{beam_name}.GAS', shot, 'NB').as_key()]
            assert not isinstance(gas, NoData) and gas.strip(), f"{beam_name}: fired but GAS data is missing"
            gas = gas.strip()
            assert gas in self.gas_species, f"Unexpected NBI GAS value: {gas!r}"
            species_z_n.append(self.gas_species[gas]['z_n'])

        return np.array(species_z_n)

    def _compose_species_label(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose species label.

        Note: Looked up from GAS string via config's gas_species (e.g., 'D2' -> 'D').
        A beam that did not fire gets an empty label; a beam that fired is expected
        to have a GAS value.
        """
        species_label = []
        for beam_name in self.beam_names:
            fired = raw_data[Requirement(f'\\NB::TOP.NB{beam_name}.FIRED', shot, 'NB').as_key()]
            if isinstance(fired, NoData):
                return fired
            if not fired:
                # The GAS string of a beam that did not fire is not meaningful (e.g. '2O' on #118162)
                species_label.append('')
                continue
            gas = raw_data[Requirement(f'\\NB::TOP.NB{beam_name}.GAS', shot, 'NB').as_key()]
            assert not isinstance(gas, NoData) and gas.strip(), f"{beam_name}: fired but GAS data is missing"
            gas = gas.strip()
            assert gas in self.gas_species, f"Unexpected NBI GAS value: {gas!r}"
            species_label.append(self.gas_species[gas]['label'])

        return np.array(species_label)

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
