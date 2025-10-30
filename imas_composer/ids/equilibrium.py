"""
Equilibrium IDS Mapping for DIII-D

Based on OMAS: omas/machine_mappings/_efit.json

Implements EFIT equilibrium reconstruction data mapping.
"""

from typing import Dict, Any
import numpy as np

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class EquilibriumMapper(IDSMapper):
    """Maps DIII-D EFIT equilibrium data to IMAS equilibrium IDS."""

    DOCS_PATH = "equilibrium.yaml"
    CONFIG_PATH = "equilibrium.yaml"

    def __init__(self, efit_tree: str = 'EFIT01'):
        """
        Initialize Equilibrium mapper.

        Args:
            efit_tree: EFIT tree name (e.g., 'EFIT01', 'EFIT02')
        """
        self.efit_tree = efit_tree

        # MDS+ path prefixes
        self.geqdsk_node = f'\\{efit_tree}::TOP.RESULTS.GEQDSK'
        self.aeqdsk_node = f'\\{efit_tree}::TOP.RESULTS.AEQDSK'

        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Build IDS specs
        self._build_specs()

    def _build_specs(self):
        """Build all IDS entry specifications"""

        # Internal dependency: GTIME (time base)
        self.specs["equilibrium._gtime"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.geqdsk_node}.GTIME', 0, self.efit_tree),
            ],
            ids_path="equilibrium._gtime",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: RBBBS (boundary R, used for masking)
        self.specs["equilibrium._rbbbs"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.geqdsk_node}.RBBBS', 0, self.efit_tree),
            ],
            ids_path="equilibrium._rbbbs",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: ZBBBS (boundary Z)
        self.specs["equilibrium._zbbbs"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.geqdsk_node}.ZBBBS', 0, self.efit_tree),
            ],
            ids_path="equilibrium._zbbbs",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: X-point 1 R
        self.specs["equilibrium._rxpt1"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.RXPT1', 0, self.efit_tree),
            ],
            ids_path="equilibrium._rxpt1",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: X-point 1 Z
        self.specs["equilibrium._zxpt1"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.ZXPT1', 0, self.efit_tree),
            ],
            ids_path="equilibrium._zxpt1",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: X-point 2 R
        self.specs["equilibrium._rxpt2"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.RXPT2', 0, self.efit_tree),
            ],
            ids_path="equilibrium._rxpt2",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: X-point 2 Z
        self.specs["equilibrium._zxpt2"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.ZXPT2', 0, self.efit_tree),
            ],
            ids_path="equilibrium._zxpt2",
            docs_file=self.DOCS_PATH
        )

        # Code metadata
        self.specs["equilibrium.code.name"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, raw: self.efit_tree,
            ids_path="equilibrium.code.name",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.code.version"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, raw: self.efit_tree,
            ids_path="equilibrium.code.version",
            docs_file=self.DOCS_PATH
        )

        # IDS properties
        self.specs["equilibrium.ids_properties.homogeneous_time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, raw: self.static_values['ids_properties.homogeneous_time'],
            ids_path="equilibrium.ids_properties.homogeneous_time",
            docs_file=self.DOCS_PATH
        )

        # Time arrays
        self.specs["equilibrium.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._gtime"],
            compose=self._compose_time,
            ids_path="equilibrium.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._gtime"],
            compose=self._compose_time,
            ids_path="equilibrium.time_slice.time",
            docs_file=self.DOCS_PATH
        )

        # Boundary outline
        self.specs["equilibrium.time_slice.boundary.outline.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._rbbbs"],
            compose=self._compose_boundary_outline_r,
            ids_path="equilibrium.time_slice.boundary.outline.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary.outline.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._rbbbs", "equilibrium._zbbbs"],
            compose=self._compose_boundary_outline_z,
            ids_path="equilibrium.time_slice.boundary.outline.z",
            docs_file=self.DOCS_PATH
        )

        # X-point coordinates
        # Note: IMAS uses array index for X-point number (0, 1, ...)
        # DIII-D has primary (XPT1) and secondary (XPT2) X-points

        self.specs["equilibrium.time_slice.boundary.x_point.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._rxpt1", "equilibrium._rxpt2"],
            compose=self._compose_xpoint_r,
            ids_path="equilibrium.time_slice.boundary.x_point.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary.x_point.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._zxpt1", "equilibrium._zxpt2"],
            compose=self._compose_xpoint_z,
            ids_path="equilibrium.time_slice.boundary.x_point.z",
            docs_file=self.DOCS_PATH
        )

    # Compose functions

    def _compose_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose time array (convert ms to seconds).

        OMAS: \\EFIT::TOP.RESULTS.GEQDSK.GTIME / 1000.
        """
        gtime_key = Requirement(f'{self.geqdsk_node}.GTIME', shot, self.efit_tree).as_key()
        gtime_ms = raw_data[gtime_key]
        return gtime_ms / 1000.0  # Convert milliseconds to seconds

    def _compose_boundary_outline_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose boundary outline R coordinates with NaN filtering.

        OMAS: nan_where(RBBBS, RBBBS, 0)
        Replaces 0 values with NaN.
        """
        rbbbs_key = Requirement(f'{self.geqdsk_node}.RBBBS', shot, self.efit_tree).as_key()
        rbbbs = raw_data[rbbbs_key].copy()

        # Replace 0 values with NaN
        rbbbs[rbbbs == 0] = np.nan

        return rbbbs

    def _compose_boundary_outline_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose boundary outline Z coordinates with NaN filtering.

        OMAS: nan_where(ZBBBS, RBBBS, 0)
        Uses RBBBS as mask - where R==0, set Z to NaN.
        """
        rbbbs_key = Requirement(f'{self.geqdsk_node}.RBBBS', shot, self.efit_tree).as_key()
        zbbbs_key = Requirement(f'{self.geqdsk_node}.ZBBBS', shot, self.efit_tree).as_key()

        rbbbs = raw_data[rbbbs_key]
        zbbbs = raw_data[zbbbs_key].copy()

        # Where R==0, set Z to NaN
        zbbbs[rbbbs == 0] = np.nan

        return zbbbs

    def _compose_xpoint_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose X-point R coordinates (both X-points).

        Returns array of shape (n_time, 2) with [RXPT1, RXPT2].
        OMAS: nan_where for each X-point separately.
        """
        rxpt1_key = Requirement(f'{self.aeqdsk_node}.RXPT1', shot, self.efit_tree).as_key()
        rxpt2_key = Requirement(f'{self.aeqdsk_node}.RXPT2', shot, self.efit_tree).as_key()

        rxpt1 = raw_data[rxpt1_key].copy()
        rxpt2 = raw_data[rxpt2_key].copy()

        # Replace 0 values with NaN
        rxpt1[rxpt1 == 0] = np.nan
        rxpt2[rxpt2 == 0] = np.nan

        # Stack into (n_time, 2) array
        return np.column_stack([rxpt1, rxpt2])

    def _compose_xpoint_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose X-point Z coordinates (both X-points).

        Returns array of shape (n_time, 2) with [ZXPT1, ZXPT2].
        """
        zxpt1_key = Requirement(f'{self.aeqdsk_node}.ZXPT1', shot, self.efit_tree).as_key()
        zxpt2_key = Requirement(f'{self.aeqdsk_node}.ZXPT2', shot, self.efit_tree).as_key()

        zxpt1 = raw_data[zxpt1_key].copy()
        zxpt2 = raw_data[zxpt2_key].copy()

        # Replace 0 values with NaN
        zxpt1[zxpt1 == 0] = np.nan
        zxpt2[zxpt2 == 0] = np.nan

        # Stack into (n_time, 2) array
        return np.column_stack([zxpt1, zxpt2])
