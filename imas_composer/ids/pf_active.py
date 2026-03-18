"""
PF Active (Poloidal Field) IDS Mapping for DIII-D

Maps DIII-D poloidal field coil hardware geometry and current data to IMAS pf_active IDS.
See OMAS: omas/machine_mappings/d3d.py::pf_active_hardware and pf_active_coil_current_data

Hardware geometry is embedded in pf_active.yaml to avoid dependency on OMAS installation.
"""

from typing import Dict, List
import numpy as np

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class PfActiveMapper(IDSMapper):
    """Maps DIII-D PF coil hardware and current data to IMAS pf_active IDS."""

    CONFIG_PATH = "pf_active.yaml"

    def __init__(self, **kwargs):
        """Initialize PF Active mapper."""
        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Load coil names and hardware data from config
        self._coil_names = self.config.get('coil_names', [])
        self._hardware_data = self.config.get('hardware', {})

        # Build IDS specs
        self._build_specs()

    def _build_specs(self):
        """Build all IDS entry specifications"""

        # Current data - auxiliary nodes (DERIVED stage for TDI expressions)
        for coil_name in self._coil_names:
            # Data
            self.specs[f"pf_active._current_data_{coil_name}"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw, cn=coil_name:
                    self._derive_current_data_requirements(shot, raw, cn),
                ids_path=f"pf_active._current_data_{coil_name}",
                docs_file=self.CONFIG_PATH
            )

            # Time
            self.specs[f"pf_active._current_time_{coil_name}"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw, cn=coil_name:
                    self._derive_current_time_requirements(shot, raw, cn),
                ids_path=f"pf_active._current_time_{coil_name}",
                docs_file=self.CONFIG_PATH
            )

            # Header (for error calculation)
            self.specs[f"pf_active._current_header_{coil_name}"] = IDSEntrySpec(
                stage=RequirementStage.DERIVED,
                derive_requirements=lambda shot, raw, cn=coil_name:
                    self._derive_current_header_requirements(shot, raw, cn),
                ids_path=f"pf_active._current_header_{coil_name}",
                docs_file=self.CONFIG_PATH
            )

        # User-facing fields - hardware geometry (COMPUTED stage)
        # Coil-level fields
        for field in ['identifier', 'name']:
            self.specs[f"pf_active.coil.{field}"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[],  # No dependencies - reads from config
                compose=lambda shot, raw, fld=field: self._compose_coil_field(shot, raw, fld),
                ids_path=f"pf_active.coil.{field}",
                docs_file=self.CONFIG_PATH
            )

        # Function index
        self.specs["pf_active.coil.function.index"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_function_index,
            ids_path="pf_active.coil.function.index",
            docs_file=self.CONFIG_PATH
        )

        # Element-level fields
        for field in ['identifier', 'name', 'turns_with_sign']:
            self.specs[f"pf_active.coil.element.{field}"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[],
                compose=lambda shot, raw, fld=field:
                    self._compose_element_field(shot, raw, fld),
                ids_path=f"pf_active.coil.element.{field}",
                docs_file=self.CONFIG_PATH
            )

        # Geometry fields
        self.specs["pf_active.coil.element.geometry.geometry_type"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_geometry_type,
            ids_path="pf_active.coil.element.geometry.geometry_type",
            docs_file=self.CONFIG_PATH
        )

        for field in ['r', 'z', 'width', 'height']:
            self.specs[f"pf_active.coil.element.geometry.rectangle.{field}"] = IDSEntrySpec(
                stage=RequirementStage.COMPUTED,
                depends_on=[],
                compose=lambda shot, raw, fld=field:
                    self._compose_rectangle_field(shot, raw, fld),
                ids_path=f"pf_active.coil.element.geometry.rectangle.{field}",
                docs_file=self.CONFIG_PATH
            )

        # Current data fields (COMPUTED stage)
        current_deps = [f"pf_active._current_data_{cn}" for cn in self._coil_names]

        self.specs["pf_active.coil.current.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=current_deps,
            compose=self._compose_current_data,
            ids_path="pf_active.coil.current.data",
            docs_file=self.CONFIG_PATH
        )

        time_deps = [f"pf_active._current_time_{cn}" for cn in self._coil_names]

        self.specs["pf_active.coil.current.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=time_deps,
            compose=self._compose_current_time,
            ids_path="pf_active.coil.current.time",
            docs_file=self.CONFIG_PATH
        )

        error_deps = current_deps + [f"pf_active._current_header_{cn}" for cn in self._coil_names]

        self.specs["pf_active.coil.current.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=error_deps,
            compose=self._compose_current_data_error_upper,
            ids_path="pf_active.coil.current.data_error_upper",
            docs_file=self.CONFIG_PATH
        )

    # Requirement derivation functions
    def _derive_current_data_requirements(self, shot: int, _raw_data: dict,
                                         coil_name: str) -> List[Requirement]:
        """Derive requirements for current data for a specific coil."""
        return [Requirement(f'ptdata2("{coil_name}",{shot})', shot, None)]

    def _derive_current_time_requirements(self, shot: int, _raw_data: dict,
                                         coil_name: str) -> List[Requirement]:
        """Derive requirements for current time for a specific coil."""
        return [Requirement(f'dim_of(ptdata2("{coil_name}",{shot}),0)', shot, None)]

    def _derive_current_header_requirements(self, shot: int, _raw_data: dict,
                                           coil_name: str) -> List[Requirement]:
        """Derive requirements for current header for a specific coil."""
        return [Requirement(f'pthead2("{coil_name}",{shot}), __rarray', shot, None)]

    # Compose functions - Hardware geometry
    def _compose_coil_field(self, shot: int, raw_data: dict, field: str) -> list:
        """
        Compose coil-level field (identifier or name).

        Returns list of values, one per coil.
        """
        coils = self._hardware_data.get('coil', [])

        result = []
        for coil in coils:
            result.append(coil.get(field, ''))

        return result

    def _compose_function_index(self, shot: int, raw_data: dict) -> list:
        """
        Compose function index for each coil.

        Returns nested list: outer list is coils, inner list is functions.
        First 6 coils (ECOIL*) have function index 0 (flux).
        Remaining coils (F*) have function index 1 (shaping).
        """
        coils = self._hardware_data.get('coil', [])

        result = []
        for k, coil in enumerate(coils):
            if k < 6:
                # Flux function
                result.append([0])
            else:
                # Shaping function
                result.append([1])

        return result

    def _compose_element_field(self, shot: int, raw_data: dict, field: str) -> list:
        """
        Compose element-level field (identifier, name, or turns_with_sign).

        Returns nested list: outer list is coils, inner list is elements per coil.
        """
        coils = self._hardware_data.get('coil', [])

        result = []
        for coil in coils:
            elements = coil.get('element', [])
            coil_elements = []
            for element in elements:
                coil_elements.append(element.get(field, ''))
            result.append(coil_elements)

        return result

    def _compose_geometry_type(self, shot: int, raw_data: dict) -> list:
        """
        Compose geometry type for each element.

        Returns nested list: outer list is coils, inner list is elements per coil.
        Geometry type 2 = rectangle, type 1 = outline.
        """
        coils = self._hardware_data.get('coil', [])

        result = []
        for coil in coils:
            elements = coil.get('element', [])
            coil_elements = []
            for element in elements:
                geometry = element.get('geometry', {})
                coil_elements.append(geometry.get('geometry_type', 2))
            result.append(coil_elements)

        return result

    def _compose_rectangle_field(self, shot: int, raw_data: dict, field: str) -> list:
        """
        Compose rectangle geometry field (r, z, width, or height).

        Returns nested list: outer list is coils, inner list is elements per coil.
        Note: Some coils use outline geometry (type 1) instead of rectangle (type 2).
        For those elements, rectangle fields will be empty/zero.
        """
        coils = self._hardware_data.get('coil', [])

        result = []
        for coil in coils:
            elements = coil.get('element', [])
            coil_elements = []
            for element in elements:
                geometry = element.get('geometry', {})
                rectangle = geometry.get('rectangle', {})
                coil_elements.append(rectangle.get(field, 0.0))
            result.append(coil_elements)

        return result

    # Compose functions - Current data
    def _compose_current_data(self, shot: int, raw_data: dict) -> list:
        """
        Compose current data for all coils.

        Returns nested list: outer list is coils, inner list is time points per coil.
        IMAS convention: F-coils (indices 6-23) are divided by turns_with_sign.
        Non-homogeneous time: each coil has its own timebase.
        """
        coils = self._hardware_data.get('coil', [])

        result = []
        for k, coil_name in enumerate(self._coil_names):
            # Get current data
            data_key = Requirement(f'ptdata2("{coil_name}",{shot})', shot, None).as_key()

            if data_key not in raw_data:
                # Coil data not available
                result.append([])
                continue

            current = raw_data[data_key]

            # Apply F-coil correction (divide by turns_with_sign)
            if 'F' in coil_name:
                # Get turns_with_sign from hardware data
                if k < len(coils):
                    elements = coils[k].get('element', [])
                    if elements:
                        turns = elements[0].get('turns_with_sign', 1.0)
                        current = current / turns

            result.append(current)

        return result

    def _compose_current_time(self, shot: int, raw_data: dict) -> list:
        """
        Compose current time for all coils.

        Returns nested list: outer list is coils, inner list is time points per coil.
        Unit conversion: ms to s (divide by 1000).
        Non-homogeneous time: each coil has its own timebase.
        """
        result = []
        for coil_name in self._coil_names:
            # Get time data
            time_key = Requirement(
                f'dim_of(ptdata2("{coil_name}",{shot}),0)', shot, None
            ).as_key()

            if time_key not in raw_data:
                # Coil time not available
                result.append([])
                continue

            time = raw_data[time_key] / 1000.0  # ms to s
            result.append(time)

        return result

    def _compose_current_data_error_upper(self, shot: int, raw_data: dict) -> list:
        """
        Compose uncertainty for current data.

        From OMAS: abs(header[3] * header[4]) * ones(nt) * 10.0
        Returns nested list: outer list is coils, inner list is time points per coil.
        F-coils also divided by turns_with_sign.
        """
        coils = self._hardware_data.get('coil', [])

        result = []
        for k, coil_name in enumerate(self._coil_names):
            # Get data to determine time length
            data_key = Requirement(f'ptdata2("{coil_name}",{shot})', shot, None).as_key()

            if data_key not in raw_data:
                # Coil data not available
                result.append([])
                continue

            nt = len(raw_data[data_key])

            # Get header information
            header_key = Requirement(
                f'pthead2("{coil_name}",{shot}), __rarray', shot, None
            ).as_key()

            if header_key not in raw_data:
                # Header not available - use zeros
                result.append(np.zeros(nt))
                continue

            header = raw_data[header_key]

            # OMAS formula: abs(header[3] * header[4]) * ones(nt) * 10.0
            error = np.abs(header[3] * header[4]) * np.ones(nt) * 10.0

            # Apply F-coil correction (divide by turns_with_sign)
            if 'F' in coil_name:
                # Get turns_with_sign from hardware data
                if k < len(coils):
                    elements = coils[k].get('element', [])
                    if elements:
                        turns = elements[0].get('turns_with_sign', 1.0)
                        error = error / turns

            result.append(error)

        return result

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
