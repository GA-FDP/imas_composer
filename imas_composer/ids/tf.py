"""
Toroidal Field (TF) IDS Mapping for DIII-D

Maps toroidal magnetic field data to IMAS tf IDS.
See OMAS: omas/machine_mappings/d3d.py::ip_bt_dflux_data
"""

from typing import Dict, List
import numpy as np

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class TfMapper(IDSMapper):
    """Maps DIII-D toroidal field data to IMAS tf IDS."""

    CONFIG_PATH = "tf.yaml"

    def __init__(self, **kwargs):
        """Initialize TF mapper."""
        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Build IDS specs
        self._build_specs()

    def _build_specs(self):
        """Build all IDS entry specifications"""

        # Toroidal field - auxiliary nodes
        self.specs["tf._bt_data"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_bt_data_requirements,
            ids_path="tf._bt_data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["tf._bt_time"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_bt_time_requirements,
            ids_path="tf._bt_time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["tf._bt_header"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_bt_header_requirements,
            ids_path="tf._bt_header",
            docs_file=self.CONFIG_PATH
        )

        # User-facing fields - COMPUTED stage
        self.specs["tf.b_field_tor_vacuum_r.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["tf._bt_data"],
            compose=self._compose_bt_data,
            ids_path="tf.b_field_tor_vacuum_r.data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["tf.b_field_tor_vacuum_r.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["tf._bt_time"],
            compose=self._compose_bt_time,
            ids_path="tf.b_field_tor_vacuum_r.time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["tf.b_field_tor_vacuum_r.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["tf._bt_data", "tf._bt_header"],
            compose=self._compose_bt_data_error_upper,
            ids_path="tf.b_field_tor_vacuum_r.data_error_upper",
            docs_file=self.CONFIG_PATH
        )

    # Requirement derivation functions
    def _derive_bt_data_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for BT (data, time, and header bundled under __ptdata__ key)."""
        return [Requirement("BT", shot, "__ptdata__")]

    def _derive_bt_time_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for BT time (same key as data — deduplication handles it)."""
        return [Requirement("BT", shot, "__ptdata__")]

    def _derive_bt_header_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for BT header (same key as data — deduplication handles it)."""
        return [Requirement("BT", shot, "__ptdata__")]

    # Compose functions
    def _compose_bt_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get toroidal field data with DIII-D specific scaling.

        From OMAS: data * vacuum_r
        This converts from the measured value to the field at the vacuum vessel radius.
        """
        key = Requirement("BT", shot, "__ptdata__").as_key()
        vacuum_r = self.static_values['vacuum_r']
        return raw_data[key]['data'] * vacuum_r

    def _compose_bt_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get toroidal field time base with unit conversion.

        From OMAS: dim_of(...,0)/1000. (convert ms to s)
        """
        key = Requirement("BT", shot, "__ptdata__").as_key()
        return raw_data[key]['times'] / 1000.0

    def _compose_bt_data_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compute uncertainty for toroidal field.

        From OMAS: abs(header[3] * header[4]) * ones(nt) * 10.0 * vacuum_r
        where header rarray comes from PtDataHeader.
        """
        key = Requirement("BT", shot, "__ptdata__").as_key()
        nt = len(raw_data[key]['data'])
        header = raw_data[key]['rarray']
        # OMAS formula: abs(header[3] * header[4]) * ones(nt) * 10.0
        return np.abs(header[3] * header[4]) * np.ones(nt) * 10.0

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
