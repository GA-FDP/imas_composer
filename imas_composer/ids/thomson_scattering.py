"""
Thomson Scattering IDS Mapping for DIII-D

See OMAS: omas/machine_mappings/d3d.py::thomson_scattering_data
"""

from typing import Dict, List, Tuple
import numpy as np

from ..core import RequirementStage, Requirement, IDSEntrySpec


class ThomsonScatteringMapper:
    """Maps DIII-D Thomson scattering data to IMAS thomson_scattering IDS."""
    
    SYSTEMS = ['TANGENTIAL', 'DIVERTOR', 'CORE']
    REVISION = 'BLESSED'
    
    DOCS_PATH = "thomson_scattering.yaml"
    
    def __init__(self):
        self.specs: Dict[str, IDSEntrySpec] = {}
        self._build_specs()
    
    def _build_specs(self):
        """Build all IDS entry specifications"""
        
        # Internal dependencies
        self.specs["thomson_scattering._calib_nums"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'.ts.{self.REVISION}.header.calib_nums', 0, 'ELECTRONS')
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
        
        self.specs["thomson_scattering._system_availability"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["thomson_scattering._calib_nums"],
            derive_requirements=self._derive_system_position_requirements,
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
            synthesize=self._synthesize_channel_name,
            ids_path="thomson_scattering.channel.name",
            docs_file=self.DOCS_PATH
        )
        
        self.specs["thomson_scattering.channel.identifier"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["thomson_scattering._system_availability"],
            synthesize=self._synthesize_channel_identifier,
            ids_path="thomson_scattering.channel.identifier",
            docs_file=self.DOCS_PATH
        )
        
        # Position coordinates
        for coord in ['r', 'z', 'phi']:
            self.specs[f"thomson_scattering.channel.position.{coord}"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=["thomson_scattering._system_availability"],
                synthesize=lambda shot, raw, c=coord: self._synthesize_position(shot, raw, c),
                ids_path=f"thomson_scattering.channel.position.{coord}",
                docs_file=self.DOCS_PATH
            )
        
        # Measurements
        self._add_measurement_specs('n_e', 'DENSITY')
        self._add_measurement_specs('t_e', 'TEMP')
    
    def _add_measurement_specs(self, measurement: str, mds_quantity: str):
        """Add specs for a measurement (both .time and .data)"""
        
        self.specs[f"thomson_scattering.channel.{measurement}.time"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["thomson_scattering._system_availability"],
            derive_requirements=self._derive_time_requirements,
            ids_path=f"thomson_scattering.channel.{measurement}.time",
            docs_file=self.DOCS_PATH
        )
        
        self.specs[f"thomson_scattering.channel.{measurement}.data"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=["thomson_scattering._system_availability"],
            derive_requirements=lambda shot, raw, q=mds_quantity: 
                self._derive_measurement_requirements(shot, raw, q),
            ids_path=f"thomson_scattering.channel.{measurement}.data",
            docs_file=self.DOCS_PATH
        )
    
    # Requirement derivation functions
    
    def _derive_hwmap_requirements(self, shot: int, raw_data: dict) -> List[Requirement]:
        """Determine hardware map requirements using calibration numbers"""
        calib_key = Requirement(
            f'.ts.{self.REVISION}.header.calib_nums', shot, 'ELECTRONS'
        ).as_key()
        cal_set = raw_data[calib_key][0]
        
        return [
            Requirement(f'.{system}.hwmapints', cal_set, 'TSCAL')
            for system in self.SYSTEMS
        ]
    
    def _derive_system_position_requirements(self, shot: int, raw_data: dict) -> List[Requirement]:
        """Request position data for all systems"""
        requirements = []
        for system in self.SYSTEMS:
            for quantity in ['R', 'Z', 'PHI']:
                requirements.append(
                    Requirement(f'.TS.{self.REVISION}.{system}:{quantity}', shot, 'ELECTRONS')
                )
        return requirements
    
    def _derive_time_requirements(self, shot: int, raw_data: dict) -> List[Requirement]:
        """Request time base only for active systems"""
        requirements = []
        for system in self.SYSTEMS:
            if self._is_system_active(system, shot, raw_data):
                requirements.append(
                    Requirement(f'.TS.{self.REVISION}.{system}:TIME', shot, 'ELECTRONS')
                )
        return requirements
    
    def _derive_measurement_requirements(self, shot: int, raw_data: dict, 
                                        quantity: str) -> List[Requirement]:
        """Request measurement data and errors only for active systems"""
        requirements = []
        for system in self.SYSTEMS:
            if self._is_system_active(system, shot, raw_data):
                requirements.append(
                    Requirement(f'.TS.{self.REVISION}.{system}:{quantity}', shot, 'ELECTRONS')
                )
                requirements.append(
                    Requirement(f'.TS.{self.REVISION}.{system}:{quantity}_E', shot, 'ELECTRONS')
                )
        return requirements
    
    # Synthesis functions
    
    def _synthesize_channel_name(self, shot: int, raw_data: dict) -> np.ndarray:
        """Generate channel names: TS_{system}_r{lens}_{channel}"""
        names = []
        
        for system in self.SYSTEMS:
            if not self._is_system_active(system, shot, raw_data):
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
    
    def _synthesize_channel_identifier(self, shot: int, raw_data: dict) -> np.ndarray:
        """Generate short identifiers: {T|D|C}{channel:02d}"""
        identifiers = []
        
        for system in self.SYSTEMS:
            if not self._is_system_active(system, shot, raw_data):
                continue
            
            nc = self._get_system_channel_count(system, shot, raw_data)
            system_letter = system[0]
            
            for j in range(nc):
                identifiers.append(f'{system_letter}{j:02d}')
        
        return np.array(identifiers)
    
    def _synthesize_position(self, shot: int, raw_data: dict, coord: str) -> np.ndarray:
        """Synthesize position coordinate (r, z, or phi) across all active systems"""
        positions = []
        mds_coord = coord.upper()
        
        for system in self.SYSTEMS:
            if not self._is_system_active(system, shot, raw_data):
                continue
            
            coord_key = Requirement(
                f'.TS.{self.REVISION}.{system}:{mds_coord}', shot, 'ELECTRONS'
            ).as_key()
            coord_data = raw_data[coord_key]
            
            if coord == 'phi':
                coord_data = -coord_data * np.pi / 180.0
            
            positions.extend(coord_data)
        
        return np.array(positions)
    
    # Helper functions
    
    def _is_system_active(self, system: str, shot: int, raw_data: dict) -> bool:
        """Check if a Thomson system has active channels"""
        r_key = Requirement(f'.TS.{self.REVISION}.{system}:R', shot, 'ELECTRONS').as_key()
        
        if r_key not in raw_data:
            return False
        
        r_data = raw_data[r_key]
        if isinstance(r_data, Exception):
            return False
        
        return len(r_data) > 0
    
    def _get_system_channel_count(self, system: str, shot: int, raw_data: dict) -> int:
        """Get number of channels in a system"""
        r_key = Requirement(f'.TS.{self.REVISION}.{system}:R', shot, 'ELECTRONS').as_key()
        return len(raw_data[r_key])
    
    def _get_hwmap_key(self, system: str, shot: int, raw_data: dict) -> Tuple:
        """Get the raw_data key for a system's hardware map"""
        calib_key = Requirement(
            f'.ts.{self.REVISION}.header.calib_nums', shot, 'ELECTRONS'
        ).as_key()
        cal_set = raw_data[calib_key][0]
        return Requirement(f'.{system}.hwmapints', cal_set, 'TSCAL').as_key()
    
    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        """Return all IDS entry specifications"""
        return self.specs