"""
Wall IDS Mapping for DIII-D

See OMAS: omas/machine_mappings/d3d.py::wall
"""

from typing import Dict, Optional
import awkward as ak
import numpy as np

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class WallMapper(IDSMapper):
    """Maps DIII-D wall/limiter data to IMAS wall IDS."""

    # Configuration YAML file contains static values and field ledger
    DOCS_PATH = "wall.yaml"
    CONFIG_PATH = "wall.yaml"

    def __init__(self, efit_tree: str = "EFIT01", efit_run_id: Optional[str] = None, **kwargs):
        """
        Initialize wall mapper.

        Args:
            efit_tree: EFIT tree to use (default: "EFIT01")
            efit_run_id: Run ID to append to shot for EFIT tree (e.g., '01', '02')
        """
        self.efit_tree = efit_tree
        self.efit_run_id = efit_run_id

        if efit_run_id is not None:
            assert efit_tree == 'EFIT', (
                f"efit_tree must be 'EFIT' when efit_run_id is set, got '{efit_tree}'"
            )

        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Build IDS specs
        self._build_specs()

    def resolve_shot(self, shot: int) -> int:
        """
        Override base class to append efit_run_id to shot number.

        Args:
            shot: Base shot number

        Returns:
            Combined shot number with run_id appended if efit_run_id is not None

        Example:
            >>> mapper = WallMapper(efit_run_id='01')
            >>> mapper.resolve_shot(200000)
            20000001
            >>> mapper = WallMapper(efit_run_id=None)
            >>> mapper.resolve_shot(200000)
            200000
        """
        if self.efit_run_id is not None:
            return int(str(shot) + self.efit_run_id)
        return shot

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

        self.specs["wall.description_2d.limiter.unit.outline.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["wall._limiter_data"],
            compose=self._compose_limiter_r,
            ids_path="wall.description_2d.limiter.unit.outline.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["wall.description_2d.limiter.unit.outline.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["wall._limiter_data"],
            compose=self._compose_limiter_z,
            ids_path="wall.description_2d.limiter.unit.outline.z",
            docs_file=self.DOCS_PATH
        )

        self.specs["wall.description_2d.limiter.type.index"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, raw: np.array([self.static_values["description_2d.limiter.type.index"]]),
            ids_path="wall.description_2d.limiter.type.index",
            docs_file=self.DOCS_PATH
        )

        self.specs["wall.description_2d.limiter.type.name"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=self._compose_limiter_type_name,
            ids_path="wall.description_2d.limiter.type.name",
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
    def _compose_limiter_type_name(self, _shot: int, _raw_data: dict) -> np.ndarray:
        """Compose limiter type name as array of shape (n_desc2d,)."""
        if self.efit_run_id is None:
            name = self.efit_tree
        else:
            name = self.efit_run_id + self.efit_tree
        return np.array([name])

    def _compose_limiter_r(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose limiter R coordinates as ak.Array with shape (n_desc2d, n_units, n_points)."""
        lim_key = Requirement('\\TOP.RESULTS.GEQDSK.LIM', self.resolve_shot(shot), self.efit_tree).as_key()
        lim = raw_data[lim_key]
        return ak.Array([[lim[:, 0]]])

    def _compose_limiter_z(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose limiter Z coordinates as ak.Array with shape (n_desc2d, n_units, n_points)."""
        lim_key = Requirement('\\TOP.RESULTS.GEQDSK.LIM', self.resolve_shot(shot), self.efit_tree).as_key()
        lim = raw_data[lim_key]
        return ak.Array([[lim[:, 1]]])

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
