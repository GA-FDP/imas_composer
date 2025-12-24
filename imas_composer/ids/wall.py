"""
Wall IDS Mapping for DIII-D

See OMAS: omas/machine_mappings/d3d.py::wall
"""

from typing import Dict
import numpy as np

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class WallMapper(IDSMapper):
    """Maps DIII-D wall/limiter data to IMAS wall IDS."""

    # Configuration YAML file contains static values and field ledger
    DOCS_PATH = "wall.yaml"
    CONFIG_PATH = "wall.yaml"

    def __init__(self, efit_tree: str = "EFIT01", **kwargs):
        """
        Initialize wall mapper.

        Args:
            efit_tree: EFIT tree to use (default: "EFIT01")
        """
        self.efit_tree = efit_tree

        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Build IDS specs
        self._build_specs()

    def _build_specs(self):
        """Build all IDS entry specifications"""

        # Internal dependency - fetch limiter data from EFIT
        self.specs["wall._limiter_data"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('\\TOP.RESULTS.GEQDSK.LIM', 0, self.efit_tree)
            ],
            ids_path="wall._limiter_data",
            docs_file=self.DOCS_PATH
        )

        # Public IDS fields

        self.specs["wall.description_2d.0.limiter.unit.0.outline.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["wall._limiter_data"],
            compose=self._compose_limiter_r,
            ids_path="wall.description_2d.0.limiter.unit.0.outline.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["wall.description_2d.0.limiter.unit.0.outline.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["wall._limiter_data"],
            compose=self._compose_limiter_z,
            ids_path="wall.description_2d.0.limiter.unit.0.outline.z",
            docs_file=self.DOCS_PATH
        )

        self.specs["wall.description_2d.0.limiter.type.index"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, raw: self.static_values["description_2d.0.limiter.type.index"],
            ids_path="wall.description_2d.0.limiter.type.index",
            docs_file=self.DOCS_PATH
        )

        self.specs["wall.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, raw: np.array(self.static_values["time"]),
            ids_path="wall.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["wall.ids_properties.homogeneous_time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, raw: self.static_values["ids_properties.homogeneous_time"],
            ids_path="wall.ids_properties.homogeneous_time",
            docs_file=self.DOCS_PATH
        )

    # Compose functions
    def _compose_limiter_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose limiter R coordinates (first column of LIM array)."""
        lim_key = Requirement('\\TOP.RESULTS.GEQDSK.LIM', shot, self.efit_tree).as_key()
        lim = raw_data[lim_key]
        return lim[:, 0]

    def _compose_limiter_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose limiter Z coordinates (second column of LIM array)."""
        lim_key = Requirement('\\TOP.RESULTS.GEQDSK.LIM', shot, self.efit_tree).as_key()
        lim = raw_data[lim_key]
        return lim[:, 1]

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
