"""
Electron Cyclotron Emission (ECE) IDS Mapping for DIII-D

See OMAS: omas/machine_mappings/d3d.py::electron_cyclotron_emission_data
"""

from typing import Dict, List
import numpy as np
import yaml
from pathlib import Path

from ..core import RequirementStage, Requirement, IDSEntrySpec


class ElectronCyclotronEmissionMapper:
    """Maps DIII-D ECE data to IMAS ece IDS."""
    
    # Documentation is loaded from YAML file
    DOCS_PATH = "electron_cyclotron_emission.yaml"
    STATIC_VALUES_PATH = "ece_static_values.yaml"

    def __init__(self, fast_ece: bool = False):
        """
        Initialize ECE mapper.

        Args:
            fast_ece: If True, use high-frequency sampling data (TECEF nodes)
        """
        self.fast_ece = fast_ece
        self.fast_suffix = 'F' if fast_ece else ''

        # MDS+ path prefixes
        self.setup_node = '\\ECE::TOP.SETUP.'
        self.cal_node = f'\\ECE::TOP.CAL{self.fast_suffix}.'
        self.tece_node = f'\\ECE::TOP.TECE.TECE{self.fast_suffix}'

        # Load static values from YAML
        self.static_values = self._load_static_values()

        self.specs: Dict[str, IDSEntrySpec] = {}
        self._build_specs()

    def _load_static_values(self) -> Dict:
        """Load static IDS values from YAML file."""
        yaml_path = Path(__file__).parent / self.STATIC_VALUES_PATH
        if yaml_path.exists():
            with open(yaml_path, 'r') as f:
                return yaml.safe_load(f) or {}
        return {}
    
    def _get_numch_path(self) -> str:
        """Get MDS+ path for NUMCH node."""
        return f'{self.cal_node}NUMCH{self.fast_suffix}'

    def _build_specs(self):
        """Build all IDS entry specifications"""

        # Internal dependencies
        self.specs["ece._numch"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(self._get_numch_path(), 0, 'ELECTRONS'),
            ],
            ids_path="ece._numch",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["ece._geometry_setup"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.setup_node}ECEPHI', 0, 'ELECTRONS'),
                Requirement(f'{self.setup_node}ECETHETA', 0, 'ELECTRONS'),
                Requirement(f'{self.setup_node}ECEZH', 0, 'ELECTRONS'),
            ],
            ids_path="ece._geometry_setup",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["ece._frequency_setup"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.setup_node}FREQ', 0, 'ELECTRONS'),
                Requirement(f'{self.setup_node}FLTRWID', 0, 'ELECTRONS'),
            ],
            ids_path="ece._frequency_setup",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["ece._time_base"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f"dim_of({self.tece_node}01)", 0, 'ELECTRONS')
            ],
            ids_path="ece._time_base",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["ece._temperature_data"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["ece._numch"],
            derive_requirements=self._derive_temperature_requirements,
            ids_path="ece._temperature_data",
            docs_file=self.DOCS_PATH
        )
        
        # IDS Properties
        self.specs["ece.ids_properties.homogeneous_time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            synthesize=lambda shot, raw: self.static_values.get('ids_properties.homogeneous_time', 0),
            ids_path="ece.ids_properties.homogeneous_time",
            docs_file=self.DOCS_PATH
        )
        
        # Line of sight geometry
        self.specs["ece.line_of_sight.first_point.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            synthesize=lambda shot, raw: self.static_values.get('line_of_sight.first_point.r', 2.5),
            ids_path="ece.line_of_sight.first_point.r",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["ece.line_of_sight.first_point.phi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ece._geometry_setup"],
            synthesize=self._synthesize_first_point_phi,
            ids_path="ece.line_of_sight.first_point.phi",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["ece.line_of_sight.first_point.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ece._geometry_setup"],
            synthesize=self._synthesize_first_point_z,
            ids_path="ece.line_of_sight.first_point.z",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["ece.line_of_sight.second_point.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            synthesize=lambda shot, raw: self.static_values.get('line_of_sight.second_point.r', 0.8),
            ids_path="ece.line_of_sight.second_point.r",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["ece.line_of_sight.second_point.phi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ece._geometry_setup"],
            synthesize=self._synthesize_second_point_phi,
            ids_path="ece.line_of_sight.second_point.phi",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["ece.line_of_sight.second_point.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ece._geometry_setup"],
            synthesize=self._synthesize_second_point_z,
            ids_path="ece.line_of_sight.second_point.z",
            docs_file=self.DOCS_PATH
        )
        
        # Channel metadata
        self.specs["ece.channel.name"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ece._numch"],
            synthesize=self._synthesize_channel_name,
            ids_path="ece.channel.name",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["ece.channel.identifier"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ece._numch"],
            synthesize=self._synthesize_channel_identifier,
            ids_path="ece.channel.identifier",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["ece.channel.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ece._time_base", "ece._numch"],
            synthesize=self._synthesize_channel_time,
            ids_path="ece.channel.time",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["ece.channel.frequency.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ece._frequency_setup", "ece._time_base", "ece._numch"],
            synthesize=self._synthesize_channel_frequency,
            ids_path="ece.channel.frequency.data",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["ece.channel.if_bandwidth"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ece._frequency_setup", "ece._numch"],
            synthesize=self._synthesize_channel_if_bandwidth,
            ids_path="ece.channel.if_bandwidth",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["ece.channel.t_e.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ece._temperature_data"],
            synthesize=self._synthesize_channel_t_e_data,
            ids_path="ece.channel.t_e.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["ece.channel.t_e.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["ece._temperature_data"],
            synthesize=self._synthesize_channel_t_e_data_error_upper,
            ids_path="ece.channel.t_e.data_error_upper",
            docs_file=self.DOCS_PATH
        )
    
    # Requirement derivation functions
    def _derive_temperature_requirements(self, shot: int, raw_data: dict) -> List[Requirement]:
        numch_key = Requirement(self._get_numch_path(), shot, 'ELECTRONS').as_key()
        n_channels = int(raw_data[numch_key])

        return [
            Requirement(f'{self.tece_node}{ich:02d}', shot, 'ELECTRONS')
            for ich in range(1, n_channels + 1)
        ]

    # Synthesis functions
    def _get_numch(self, shot: int, raw_data: dict) -> int:
        numch_key = Requirement(self._get_numch_path(), shot, 'ELECTRONS').as_key()
        return int(raw_data[numch_key])
    
    def _synthesize_first_point_phi(self, shot: int, raw_data: dict) -> float:
        phi_key = Requirement(f'{self.setup_node}ECEPHI', shot, 'ELECTRONS').as_key()
        return np.deg2rad(raw_data[phi_key])
    
    def _synthesize_first_point_z(self, shot: int, raw_data: dict) -> float:
        z_key = Requirement(f'{self.setup_node}ECEZH', shot, 'ELECTRONS').as_key()
        return raw_data[z_key]
    
    def _synthesize_second_point_phi(self, shot: int, raw_data: dict) -> float:
        return self._synthesize_first_point_phi(shot, raw_data)
    
    def _synthesize_second_point_z(self, shot: int, raw_data: dict) -> float:
        z_key = Requirement(f'{self.setup_node}ECEZH', shot, 'ELECTRONS').as_key()
        theta_key = Requirement(f'{self.setup_node}ECETHETA', shot, 'ELECTRONS').as_key()
        z_first = raw_data[z_key]
        theta_rad = np.deg2rad(raw_data[theta_key])
        return z_first + np.sin(theta_rad)
    
    def _synthesize_channel_name(self, shot: int, raw_data: dict) -> np.ndarray:
        n_channels = self._get_numch(shot, raw_data)
        return np.array([f'ECE{ich}' for ich in range(1, n_channels + 1)])
    
    def _synthesize_channel_identifier(self, shot: int, raw_data: dict) -> np.ndarray:
        n_channels = self._get_numch(shot, raw_data)
        return np.array([f'{self.tece_node}{ich:02d}' for ich in range(1, n_channels + 1)])
    
    def _synthesize_channel_time(self, shot: int, raw_data: dict) -> np.ndarray:
        time_key = Requirement(f"dim_of({self.tece_node}01)", shot, 'ELECTRONS').as_key()
        time_s = raw_data[time_key] * 1e-3
        n_channels = self._get_numch(shot, raw_data)
        return np.tile(time_s, (n_channels, 1))
    
    def _synthesize_channel_frequency(self, shot: int, raw_data: dict) -> np.ndarray:
        freq_key = Requirement(f'{self.setup_node}FREQ', shot, 'ELECTRONS').as_key()
        # Convert to float64 before multiplication to preserve precision at ~1e11 Hz
        freq_ghz = np.asarray(raw_data[freq_key], dtype=np.float64)
        freq_hz = freq_ghz * 1e9
        time_key = Requirement(f"dim_of({self.tece_node}01)", shot, 'ELECTRONS').as_key()
        n_time = len(raw_data[time_key])
        return np.tile(freq_hz[:, np.newaxis], (1, n_time))
    
    def _synthesize_channel_if_bandwidth(self, shot: int, raw_data: dict) -> np.ndarray:
        bw_key = Requirement(f'{self.setup_node}FLTRWID', shot, 'ELECTRONS').as_key()
        return raw_data[bw_key] * 1e9
    
    def _synthesize_channel_t_e_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Synthesize electron temperature data (convert keV to eV).

        Returns array of shape (n_channels, n_time).
        """
        n_channels = self._get_numch(shot, raw_data)
        temps_ev = []

        for ich in range(1, n_channels + 1):
            temp_key = Requirement(f'{self.tece_node}{ich:02d}', shot, 'ELECTRONS').as_key()
            t_kev = raw_data[temp_key]
            t_ev = t_kev * 1e3  # Convert keV to eV
            temps_ev.append(t_ev)

        return np.array(temps_ev)

    def _synthesize_channel_t_e_data_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Synthesize electron temperature upper uncertainties (ece.channel.t_e.data_error_upper).

        Uncertainty model: 7% calibration error + Poisson uncertainty
        Returns array of shape (n_channels, n_time) in eV.
        """
        n_channels = self._get_numch(shot, raw_data)
        uncertainties = []

        for ich in range(1, n_channels + 1):
            temp_key = Requirement(f'{self.tece_node}{ich:02d}', shot, 'ELECTRONS').as_key()
            t_kev = raw_data[temp_key]

            # Uncertainty in eV: sqrt(T[eV]) + 7% calibration error
            # Note: sqrt operates on eV, so convert first
            sigma_ev = np.sqrt(np.abs(t_kev * 1e3)) + 70 * np.abs(t_kev)
            uncertainties.append(sigma_ev)

        return np.array(uncertainties)
    
    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs