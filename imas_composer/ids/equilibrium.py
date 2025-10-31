"""
Equilibrium IDS Mapping for DIII-D

Based on OMAS: omas/machine_mappings/_efit.json

Implements EFIT equilibrium reconstruction data mapping.
"""

from typing import Dict, Any
import numpy as np
import awkward as ak

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


def filter_padding(arr: np.ndarray, mask: np.ndarray) -> ak.Array:
    """
    Remove padding from 2D array using boolean mask, returning ragged awkward array.

    Directly filters using mask without intermediate NaN conversion.

    Args:
        arr: 2D numpy array with shape (n_outer, n_max_inner) - data to filter
        mask: 2D boolean array - True where data is valid, False where padding

    Returns:
        Awkward array with ragged inner dimension where mask is True

    Example:
        >>> arr = np.array([[1, 2, 0, 0], [3, 0, 0, 0], [4, 5, 6, 0]])
        >>> mask = arr != 0
        >>> filter_padding(arr, mask)
        <Array [[1, 2], [3], [4, 5, 6]] type='3 * var * float64'>
    """
    filtered_rows = []
    for row, row_mask in zip(arr, mask):
        filtered_row = row[row_mask]
        filtered_rows.append(filtered_row)

    return ak.Array(filtered_rows)


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

        # Boundary separatrix outline (same as boundary.outline for EFIT)
        self.specs["equilibrium.time_slice.boundary_separatrix.outline.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._rbbbs"],
            compose=self._compose_boundary_outline_r,
            ids_path="equilibrium.time_slice.boundary_separatrix.outline.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary_separatrix.outline.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._rbbbs", "equilibrium._zbbbs"],
            compose=self._compose_boundary_outline_z,
            ids_path="equilibrium.time_slice.boundary_separatrix.outline.z",
            docs_file=self.DOCS_PATH
        )

        # Boundary separatrix x_points (same as boundary.x_point for EFIT)
        self.specs["equilibrium.time_slice.boundary_separatrix.x_point.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._rxpt1", "equilibrium._rxpt2"],
            compose=self._compose_xpoint_r,
            ids_path="equilibrium.time_slice.boundary_separatrix.x_point.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary_separatrix.x_point.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._zxpt1", "equilibrium._zxpt2"],
            compose=self._compose_xpoint_z,
            ids_path="equilibrium.time_slice.boundary_separatrix.x_point.z",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: RSURF (geometric axis R)
        self.specs["equilibrium._rsurf"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.RSURF', 0, self.efit_tree),
            ],
            ids_path="equilibrium._rsurf",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: ZSURF (geometric axis Z)
        self.specs["equilibrium._zsurf"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.ZSURF', 0, self.efit_tree),
            ],
            ids_path="equilibrium._zsurf",
            docs_file=self.DOCS_PATH
        )

        # Boundary separatrix geometric axis
        self.specs["equilibrium.time_slice.boundary_separatrix.geometric_axis.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._rsurf"],
            compose=self._compose_geometric_axis_r,
            ids_path="equilibrium.time_slice.boundary_separatrix.geometric_axis.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary_separatrix.geometric_axis.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._zsurf"],
            compose=self._compose_geometric_axis_z,
            ids_path="equilibrium.time_slice.boundary_separatrix.geometric_axis.z",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: SEPLIM (closest wall point distance)
        self.specs["equilibrium._seplim"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.SEPLIM', 0, self.efit_tree),
            ],
            ids_path="equilibrium._seplim",
            docs_file=self.DOCS_PATH
        )

        # Boundary separatrix closest wall point distance
        self.specs["equilibrium.time_slice.boundary_separatrix.closest_wall_point.distance"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._seplim"],
            compose=self._compose_closest_wall_distance,
            ids_path="equilibrium.time_slice.boundary_separatrix.closest_wall_point.distance",
            docs_file=self.DOCS_PATH
        )

        # Internal dependencies: Gap values
        self.specs["equilibrium._gapin"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.GAPIN', 0, self.efit_tree),
            ],
            ids_path="equilibrium._gapin",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._gapout"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.GAPOUT', 0, self.efit_tree),
            ],
            ids_path="equilibrium._gapout",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._gaptop"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.GAPTOP', 0, self.efit_tree),
            ],
            ids_path="equilibrium._gaptop",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._gapbot"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.GAPBOT', 0, self.efit_tree),
            ],
            ids_path="equilibrium._gapbot",
            docs_file=self.DOCS_PATH
        )

        # Boundary separatrix gaps (4 gaps per time: inboard, outboard, top, bottom)
        # Need time dimension to tile gap names across time slices
        self.specs["equilibrium.time_slice.boundary_separatrix.gap.name"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._gtime"],
            compose=self._compose_gap_names,
            ids_path="equilibrium.time_slice.boundary_separatrix.gap.name",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary_separatrix.gap.value"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._gapin", "equilibrium._gapout", "equilibrium._gaptop", "equilibrium._gapbot"],
            compose=self._compose_gap_values,
            ids_path="equilibrium.time_slice.boundary_separatrix.gap.value",
            docs_file=self.DOCS_PATH
        )

        # Internal dependencies: Strike point coordinates (4 strike points)
        self.specs["equilibrium._rvsid"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.RVSID', 0, self.efit_tree),
            ],
            ids_path="equilibrium._rvsid",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._zvsid"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.ZVSID', 0, self.efit_tree),
            ],
            ids_path="equilibrium._zvsid",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._rvsod"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.RVSOD', 0, self.efit_tree),
            ],
            ids_path="equilibrium._rvsod",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._zvsod"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.ZVSOD', 0, self.efit_tree),
            ],
            ids_path="equilibrium._zvsod",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._rvsiu"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.RVSIU', 0, self.efit_tree),
            ],
            ids_path="equilibrium._rvsiu",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._zvsiu"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.ZVSIU', 0, self.efit_tree),
            ],
            ids_path="equilibrium._zvsiu",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._rvsou"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.RVSOU', 0, self.efit_tree),
            ],
            ids_path="equilibrium._rvsou",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._zvsou"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.ZVSOU', 0, self.efit_tree),
            ],
            ids_path="equilibrium._zvsou",
            docs_file=self.DOCS_PATH
        )

        # Boundary separatrix strike points (4 points: vsid, vsod, vsiu, vsou)
        self.specs["equilibrium.time_slice.boundary_separatrix.strike_point.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._rvsid", "equilibrium._rvsod", "equilibrium._rvsiu", "equilibrium._rvsou"],
            compose=self._compose_strike_point_r,
            ids_path="equilibrium.time_slice.boundary_separatrix.strike_point.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary_separatrix.strike_point.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._rvsid", "equilibrium._zvsid", "equilibrium._rvsod", "equilibrium._zvsod",
                       "equilibrium._rvsiu", "equilibrium._zvsiu", "equilibrium._rvsou", "equilibrium._zvsou"],
            compose=self._compose_strike_point_z,
            ids_path="equilibrium.time_slice.boundary_separatrix.strike_point.z",
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

    def _compose_boundary_outline_r(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compose boundary outline R coordinates with padding removal.

        Removes 0-padded values to create ragged array.

        Returns:
            Ragged awkward array (n_time, var) where each time slice has different number of points
        """
        rbbbs_key = Requirement(f'{self.geqdsk_node}.RBBBS', shot, self.efit_tree).as_key()
        rbbbs = raw_data[rbbbs_key]

        # Filter out padding (where R==0)
        mask = rbbbs != 0
        return filter_padding(rbbbs, mask)

    def _compose_boundary_outline_z(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compose boundary outline Z coordinates with padding removal.

        Uses R coordinates as mask - filters Z where R==0 (padding).
        This preserves valid Z=0 values (at midplane) while removing padding.

        Returns:
            Ragged awkward array (n_time, var) where each time slice has different number of points
        """
        rbbbs_key = Requirement(f'{self.geqdsk_node}.RBBBS', shot, self.efit_tree).as_key()
        zbbbs_key = Requirement(f'{self.geqdsk_node}.ZBBBS', shot, self.efit_tree).as_key()

        rbbbs = raw_data[rbbbs_key]
        zbbbs = raw_data[zbbbs_key]

        # Filter using R as mask: where R==0 indicates padding, not valid data
        mask = rbbbs != 0
        return filter_padding(zbbbs, mask)

    def _compose_xpoint_r(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compose X-point R coordinates (both X-points).

        Removes 0-padded X-points. Result can have 0, 1, or 2 X-points per time.

        Returns:
            Ragged awkward array (n_time, var) where each time slice has 0-2 X-points
        """
        rxpt1_key = Requirement(f'{self.aeqdsk_node}.RXPT1', shot, self.efit_tree).as_key()
        rxpt2_key = Requirement(f'{self.aeqdsk_node}.RXPT2', shot, self.efit_tree).as_key()

        rxpt1 = raw_data[rxpt1_key]
        rxpt2 = raw_data[rxpt2_key]

        # Stack into (n_time, 2) array
        xpoints = np.column_stack([rxpt1, rxpt2])

        # Filter out padding (where X-point R==0)
        mask = xpoints != 0
        return filter_padding(xpoints, mask)

    def _compose_xpoint_z(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compose X-point Z coordinates (both X-points).

        X-points are never at Z=0 (physics constraint: X-points form at top/bottom,
        never at midplane). DIII-D Z range: ~-1.2 to ~1.2 m, X-points typically
        at Z ≈ ±0.3 to ±1.0 m. Therefore Z==0 directly indicates padding.

        Returns:
            Ragged awkward array (n_time, var) where each time slice has 0-2 X-points
        """
        zxpt1_key = Requirement(f'{self.aeqdsk_node}.ZXPT1', shot, self.efit_tree).as_key()
        zxpt2_key = Requirement(f'{self.aeqdsk_node}.ZXPT2', shot, self.efit_tree).as_key()

        zxpt1 = raw_data[zxpt1_key]
        zxpt2 = raw_data[zxpt2_key]

        # Stack Z coordinates
        xpoints_z = np.column_stack([zxpt1, zxpt2])

        # Z==0 means padding (X-points are never at midplane due to physics)
        mask = xpoints_z != 0

        return filter_padding(xpoints_z, mask)

    def _compose_geometric_axis_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose geometric axis R coordinate (convert cm to meters).

        OMAS: data(\\EFIT::TOP.RESULTS.AEQDSK.RSURF) / 100.
        """
        rsurf_key = Requirement(f'{self.aeqdsk_node}.RSURF', shot, self.efit_tree).as_key()
        rsurf_cm = raw_data[rsurf_key]
        return rsurf_cm / 100.0  # Convert cm to meters

    def _compose_geometric_axis_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose geometric axis Z coordinate (convert cm to meters).

        OMAS: data(\\EFIT::TOP.RESULTS.AEQDSK.ZSURF) / 100.
        """
        zsurf_key = Requirement(f'{self.aeqdsk_node}.ZSURF', shot, self.efit_tree).as_key()
        zsurf_cm = raw_data[zsurf_key]
        return zsurf_cm / 100.0  # Convert cm to meters

    def _compose_closest_wall_distance(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose closest wall point distance (convert cm to meters).

        OMAS: data(\\EFIT::TOP.RESULTS.AEQDSK.SEPLIM) / 100.
        """
        seplim_key = Requirement(f'{self.aeqdsk_node}.SEPLIM', shot, self.efit_tree).as_key()
        seplim_cm = raw_data[seplim_key]
        return seplim_cm / 100.0  # Convert cm to meters

    def _compose_gap_names(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose gap names tiled across time dimension.

        Returns (n_time, 4) array where each time slice has the same 4 names:
        ["inboard", "outboard", "top", "bottom"]

        OMAS deviation: OMAS only defines names once, but IMAS schema expects
        names per time_slice, so we tile across time dimension.
        """
        gtime_key = Requirement(f'{self.geqdsk_node}.GTIME', shot, self.efit_tree).as_key()
        gtime_ms = raw_data[gtime_key]
        n_time = len(gtime_ms)

        # Tile gap names across time dimension
        gap_names = np.array(["inboard", "outboard", "top", "bottom"])
        return np.tile(gap_names, (n_time, 1))

    def _compose_gap_values(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose gap values for all 4 gaps (convert cm to meters).

        Returns (n_time, 4) array with [inboard, outboard, top, bottom] per time.

        OMAS: data(\\EFIT::TOP.RESULTS.AEQDSK.GAPIN/GAPOUT/GAPTOP/GAPBOT) / 100.
        """
        gapin_key = Requirement(f'{self.aeqdsk_node}.GAPIN', shot, self.efit_tree).as_key()
        gapout_key = Requirement(f'{self.aeqdsk_node}.GAPOUT', shot, self.efit_tree).as_key()
        gaptop_key = Requirement(f'{self.aeqdsk_node}.GAPTOP', shot, self.efit_tree).as_key()
        gapbot_key = Requirement(f'{self.aeqdsk_node}.GAPBOT', shot, self.efit_tree).as_key()

        gapin_cm = raw_data[gapin_key]
        gapout_cm = raw_data[gapout_key]
        gaptop_cm = raw_data[gaptop_key]
        gapbot_cm = raw_data[gapbot_key]

        # Stack into (n_time, 4) array
        gaps = np.column_stack([gapin_cm, gapout_cm, gaptop_cm, gapbot_cm])
        return gaps / 100.0  # Convert cm to meters

    def _compose_strike_point_r(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compose strike point R coordinates for all 4 strike points (convert cm to meters).

        Removes invalid strike points (R == -0.89 in OMAS convention).
        Result can have 0-4 strike points per time.

        Returns:
            Ragged awkward array (n_time, var) where each time slice has 0-4 strike points
        """
        rvsid_key = Requirement(f'{self.aeqdsk_node}.RVSID', shot, self.efit_tree).as_key()
        rvsod_key = Requirement(f'{self.aeqdsk_node}.RVSOD', shot, self.efit_tree).as_key()
        rvsiu_key = Requirement(f'{self.aeqdsk_node}.RVSIU', shot, self.efit_tree).as_key()
        rvsou_key = Requirement(f'{self.aeqdsk_node}.RVSOU', shot, self.efit_tree).as_key()

        rvsid_cm = raw_data[rvsid_key]
        rvsod_cm = raw_data[rvsod_key]
        rvsiu_cm = raw_data[rvsiu_key]
        rvsou_cm = raw_data[rvsou_key]

        # Stack into (n_time, 4) array and convert to meters
        strike_points_m = np.column_stack([rvsid_cm, rvsod_cm, rvsiu_cm, rvsou_cm]) / 100.0

        # Filter out invalid strike points (OMAS uses -0.89 cm as sentinel, which becomes -0.0089 m)
        mask = strike_points_m != -0.0089
        return filter_padding(strike_points_m, mask)

    def _compose_strike_point_z(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compose strike point Z coordinates for all 4 strike points (convert cm to meters).

        Uses corresponding R coordinates as mask to handle Z=0 correctly.
        Result can have 0-4 strike points per time.

        Returns:
            Ragged awkward array (n_time, var) where each time slice has 0-4 strike points
        """
        rvsid_key = Requirement(f'{self.aeqdsk_node}.RVSID', shot, self.efit_tree).as_key()
        rvsod_key = Requirement(f'{self.aeqdsk_node}.RVSOD', shot, self.efit_tree).as_key()
        rvsiu_key = Requirement(f'{self.aeqdsk_node}.RVSIU', shot, self.efit_tree).as_key()
        rvsou_key = Requirement(f'{self.aeqdsk_node}.RVSOU', shot, self.efit_tree).as_key()
        zvsid_key = Requirement(f'{self.aeqdsk_node}.ZVSID', shot, self.efit_tree).as_key()
        zvsod_key = Requirement(f'{self.aeqdsk_node}.ZVSOD', shot, self.efit_tree).as_key()
        zvsiu_key = Requirement(f'{self.aeqdsk_node}.ZVSIU', shot, self.efit_tree).as_key()
        zvsou_key = Requirement(f'{self.aeqdsk_node}.ZVSOU', shot, self.efit_tree).as_key()

        rvsid_cm = raw_data[rvsid_key]
        rvsod_cm = raw_data[rvsod_key]
        rvsiu_cm = raw_data[rvsiu_key]
        rvsou_cm = raw_data[rvsou_key]
        zvsid_cm = raw_data[zvsid_key]
        zvsod_cm = raw_data[zvsod_key]
        zvsiu_cm = raw_data[zvsiu_key]
        zvsou_cm = raw_data[zvsou_key]

        # Stack Z coordinates and convert to meters
        strike_points_z_m = np.column_stack([zvsid_cm, zvsod_cm, zvsiu_cm, zvsou_cm]) / 100.0

        # Use R coordinates as mask (R == -0.0089 m means invalid)
        strike_points_r_m = np.column_stack([rvsid_cm, rvsod_cm, rvsiu_cm, rvsou_cm]) / 100.0
        mask = strike_points_r_m != -0.0089

        return filter_padding(strike_points_z_m, mask)
