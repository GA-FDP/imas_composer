"""
Equilibrium IDS Mapping for DIII-D

Based on OMAS: omas/machine_mappings/_efit.json

Implements EFIT equilibrium reconstruction data mapping.
"""

from typing import Dict, Any, Optional
import xml.etree.ElementTree as ET
import f90nml
import numpy as np
import awkward as ak

from ..core import RequirementStage, Requirement, IDSEntrySpec, NoData
from .base import IDSMapper
from ..cocos import (
    COCOSTransform,
    get_cocos_transform_type,
    identify_cocos_from_signals,
    apply_cocos_transform,
)
from ..efit_basis import (
    BASIS_KEYS,
    EFIT_MU0,
    RECOVERY_TOLERANCE,
    basis_settings,
    format_floats,
    resolve_basis,
    split_coefficients,
    write_basis,
)
from scipy.constants import mu_0
from scipy.interpolate import interp1d
from contourpy import contour_generator

# XRSP entries beyond kppcur + kffcur hold the netCDF fill value 9.97e36
XRSP_FILL_THRESHOLD = 1e30
# a k-file belongs to the equilibrium slice at the same time (KTIME and GTIME in ms)
KTIME_TOLERANCE = 1e-3
FITDELZ_KEYS = ("fitdelz", "ifitdelz")
# namelist group of the basis and of the fitdelz settings in a k-file and in a snap file
BASIS_GROUPS = {"kfile": "in1", "snap": "efitin"}
FITDELZ_GROUPS = {"kfile": "inwant", "snap": "efitin"}
# EFIT nodes <xrsp> and <basis> of equilibrium.code.parameters are composed from
XRSP_INPUTS = frozenset({"GTIME", "MTIME", "XRSP"})
BASIS_INPUTS = XRSP_INPUTS | {"PLASMA", "PSIN", "PPRIME", "FFPRIM", "BCENTR", "CPASMA"}
# an a-file belongs to the equilibrium slice at the same time (ATIME and GTIME in ms, spaced by >= 20 ms)
ATIME_TOLERANCE = 0.1


def filter_padding(arr: np.ndarray, mask: np.ndarray) -> ak.Array:
    """
    Remove padding from 2D array using boolean mask, returning ragged awkward array.

    Invalid entries are dropped wherever they are in the row, so rows without
    any valid entry become empty lists.

    Args:
        arr: 2D numpy array with shape (n_outer, n_max_inner) - data to filter
        mask: 2D boolean array - True where data is valid, False where padding

    Returns:
        Awkward array with ragged inner dimension

    Example:
        >>> arr = np.array([[1, 2, 0, 0], [3, 0, 0, 0], [4, 5, 6, 0]])
        >>> filter_padding(arr, arr != 0)
        <Array [[1, 2], [3], [4, 5, 6]] type='3 * var * int64'>
        >>> arr = np.array([[0, 2], [3, 0], [0, 0]])
        >>> filter_padding(arr, arr != 0)
        <Array [[2], [3], []] type='3 * var * int64'>
    """
    arr = np.asarray(arr)
    mask = np.asarray(mask, dtype=bool)
    return ak.unflatten(arr[mask], mask.sum(axis=1))


def parabola_extremum(x: np.ndarray, y: np.ndarray, i: int) -> tuple[float, float]:
    """
    Extremum of y(x) near index i of a closed curve, from the parabola through the points i-1, i, i+1.

    Same as OMFIT parabolaMaxCycle(bounded=True): falls back to (x[i], y[i]) when the parabola is
    degenerate or its vertex lies outside the three points.

    Args:
        x: Abscissae along the closed curve, without a repeated end point
        y: Ordinates along the closed curve
        i: Index of the grid extremum of y

    Returns:
        (x, y) of the extremum
    """
    x0, x1, x2 = x[[i - 1, i, (i + 1) % len(x)]]
    y0, y1, y2 = y[[i - 1, i, (i + 1) % len(x)]]
    denom = (x0 - x1) * (x0 - x2) * (x1 - x2)
    if denom == 0:
        return x1, y1
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom
    b = (x2**2 * (y0 - y1) + x1**2 * (y2 - y0) + x0**2 * (y1 - y2)) / denom
    if a == 0:
        return x1, y1
    x_vertex = -b / (2 * a)
    if not min(x0, x1, x2) <= x_vertex <= max(x0, x1, x2):
        return x1, y1
    # Taylor expansion of the parabola around x1
    return x_vertex, y1 + (2 * a * x1 + b) * (x_vertex - x1) + a * (x_vertex - x1) ** 2


def flux_surface_extrema(r_grid: np.ndarray, z_grid: np.ndarray, psirz: np.ndarray, psi_levels: np.ndarray,
                         r_axis: np.ndarray, z_axis: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Trace closed flux surfaces around the magnetic axis and return their extrema.

    Per time slice and level, the surface is the closed contour of psirz whose R and Z ranges contain the
    magnetic axis. The extrema are refined with parabola_extremum, as in OMFIT fluxGeo.

    Args:
        r_grid: (n_r,) R grid of psirz
        z_grid: (n_z,) Z grid of psirz
        psirz: (n_time, n_z, n_r) poloidal flux, as stored in GEQDSK.PSIRZ
        psi_levels: (n_time, n_level) poloidal flux of the surfaces, in the convention of psirz
        r_axis: (n_time,) R of the magnetic axis
        z_axis: (n_time,) Z of the magnetic axis

    Returns:
        Dict of (n_time, n_level) arrays: r_max, r_min, z_max, z_min and r_at_z_max, r_at_z_min (R of the
        highest and lowest point). Levels without a closed contour around the axis (e.g. failed slices) are NaN.
    """
    names = ['r_max', 'r_min', 'z_max', 'z_min', 'r_at_z_max', 'r_at_z_min']
    extrema = {name: np.full(psi_levels.shape, np.nan) for name in names}
    for i_time in range(psi_levels.shape[0]):
        contours = contour_generator(x=r_grid, y=z_grid, z=psirz[i_time], line_type="Separate")
        for i_level, level in enumerate(psi_levels[i_time]):
            surfaces = [line for line in contours.lines(level)
                        if np.array_equal(line[0], line[-1])
                        and line[:, 0].min() < r_axis[i_time] < line[:, 0].max()
                        and line[:, 1].min() < z_axis[i_time] < line[:, 1].max()]
            if not surfaces:
                continue
            r, z = surfaces[0][:-1, 0], surfaces[0][:-1, 1]
            _, extrema['r_max'][i_time, i_level] = parabola_extremum(z, r, np.argmax(r))
            _, extrema['r_min'][i_time, i_level] = parabola_extremum(z, r, np.argmin(r))
            extrema['r_at_z_max'][i_time, i_level], extrema['z_max'][i_time, i_level] = parabola_extremum(r, z, np.argmax(z))
            extrema['r_at_z_min'][i_time, i_level], extrema['z_min'][i_time, i_level] = parabola_extremum(r, z, np.argmin(z))
    return extrema


class EquilibriumMapper(IDSMapper):
    """Maps DIII-D EFIT equilibrium data to IMAS equilibrium IDS."""

    DOCS_PATH = "equilibrium.yaml"
    CONFIG_PATH = "equilibrium.yaml"

    # Constraints are always sourced from COCOS 7, regardless of the
    # Bt/Ip-identified COCOS used for the rest of the equilibrium IDS.
    CONSTRAINTS_SOURCE_COCOS = 7

    def __init__(self, efit_tree: str = 'EFIT01', efit_run_id: Optional[str] = None, **kwargs):
        """
        Initialize Equilibrium mapper.

        Args:
            efit_tree: EFIT tree name (e.g., 'EFIT01', 'EFIT02')
            efit_run_id: Run ID to append to shot for EFIT tree (e.g., '01', '02')
        """
        self.efit_tree = efit_tree
        self.efit_run_id = efit_run_id
        # The run-id runs of the EFIT tree (CAKE runs) store the k-file of every slice, the standard trees (EFIT01,
        # EFIT02) the snap file of the run. Only the namelist of the tree is a requirement.
        self.namelist_source = "kfile" if efit_tree == "EFIT" else "snap"

        # MDSplus path prefixes
        self.geqdsk_node = f'\\{efit_tree}::TOP.RESULTS.GEQDSK'
        self.aeqdsk_node = f'\\{efit_tree}::TOP.RESULTS.AEQDSK'
        self.measurements_node = f'\\{efit_tree}::TOP.MEASUREMENTS'
        self.namelists_node = f'\\{efit_tree}::TOP.NAMELISTS'

        # COCOS transformer
        self.cocos = COCOSTransform()
        self._cocos_cache: Dict[int, int] = {}  # shot -> cocos mapping

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
            >>> mapper = EquilibriumMapper(efit_run_id='01')
            >>> mapper.resolve_shot(200000)
            20000001
            >>> mapper = EquilibriumMapper(efit_run_id=None)
            >>> mapper.resolve_shot(200000)
            200000
        """
        if self.efit_run_id is not None:
            return int(str(shot) + self.efit_run_id)
        return shot

    def _build_specs(self):
        """Build all IDS entry specifications"""

        # Internal dependency: BCENTR (for COCOS identification)
        self.specs["equilibrium._bcentr"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.geqdsk_node}.BCENTR', 0, self.efit_tree),
            ],
            ids_path="equilibrium._bcentr",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: CPASMA (for COCOS identification)
        self.specs["equilibrium._cpasma_cocos"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.geqdsk_node}.CPASMA', 0, self.efit_tree),
            ],
            ids_path="equilibrium._cpasma_cocos",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: GTIME (time base)
        self.specs["equilibrium._gtime"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.geqdsk_node}.GTIME', 0, self.efit_tree),
            ],
            ids_path="equilibrium._gtime",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: MTIME (measurements time base)
        # Required for constraint time filtering (MEASUREMENTS have different time dimension)
        self.specs["equilibrium._mtime"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.measurements_node}.MTIME', 0, self.efit_tree),
            ],
            ids_path="equilibrium._mtime",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: Time indices for constraint filtering
        # Maps GTIME (equilibrium time base) to MTIME (measurements time base)
        # Needed because EFIT filters are applied to RESULTS but not MEASUREMENTS
        self.specs["equilibrium._constraint_time_indices"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._gtime", "equilibrium._mtime"],
            compose=self._compose_constraint_time_indices,
            ids_path="equilibrium._constraint_time_indices",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: ATIME (AEQDSK time base)
        self.specs["equilibrium._atime"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.ATIME', 0, self.efit_tree),
            ],
            ids_path="equilibrium._atime",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: Time indices for AEQDSK quantities
        # Maps GTIME (equilibrium time base) to ATIME, which lacks the slices that have no a-file
        self.specs["equilibrium._aeqdsk_time_indices"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._gtime", "equilibrium._atime"],
            compose=self._compose_aeqdsk_time_indices,
            ids_path="equilibrium._aeqdsk_time_indices",
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

        # Internal dependency: XRSP (p' and FF' basis function coefficients, MEASUREMENTS time base)
        self.specs["equilibrium._xrsp"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.measurements_node}.XRSP', 0, self.efit_tree),
            ],
            ids_path="equilibrium._xrsp",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: the k-file namelists EFIT ran with, one per KTIME. Only some runs store them
        # (CAKE runs do, the standard EFIT01/EFIT02 trees do not).
        self.specs["equilibrium._keqdsks"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.namelists_node}:KEQDSKS', 0, self.efit_tree, isolated=True),
                Requirement(f'{self.namelists_node}:KEQDSKS:KTIME', 0, self.efit_tree),
            ],
            ids_path="equilibrium._keqdsks",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: the snap file (&EFITIN) the run was set up with. The standard EFIT01/EFIT02 trees
        # store it, and their slices have no k-files.
        self.specs["equilibrium._snap"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'\\{self.efit_tree}::TOP:NAMELIST', 0, self.efit_tree),
            ],
            ids_path="equilibrium._snap",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.code.parameters"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "equilibrium._xrsp", "equilibrium._constraint_time_indices", "equilibrium._gtime",
                "equilibrium._keqdsks" if self.namelist_source == "kfile" else "equilibrium._snap",
                "equilibrium._psin", "equilibrium._pprime", "equilibrium._ffprim", "equilibrium._plasma",
                "equilibrium._bcentr", "equilibrium._cpasma_cocos",
            ],
            compose=self._compose_code_parameters,
            ids_path="equilibrium.code.parameters",
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
            depends_on=["equilibrium._rxpt1", "equilibrium._rxpt2", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_xpoint_r,
            ids_path="equilibrium.time_slice.boundary.x_point.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary.x_point.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._rxpt1", "equilibrium._rxpt2",
                        "equilibrium._zxpt1", "equilibrium._zxpt2", "equilibrium._aeqdsk_time_indices"],
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
            depends_on=["equilibrium._rxpt1", "equilibrium._rxpt2", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_xpoint_r,
            ids_path="equilibrium.time_slice.boundary_separatrix.x_point.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary_separatrix.x_point.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._zxpt1", "equilibrium._zxpt2",
                        "equilibrium._rxpt1", "equilibrium._rxpt2", "equilibrium._aeqdsk_time_indices"],
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
            depends_on=["equilibrium._rsurf", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_geometric_axis_r,
            ids_path="equilibrium.time_slice.boundary_separatrix.geometric_axis.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary_separatrix.geometric_axis.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._zsurf", "equilibrium._aeqdsk_time_indices"],
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
            depends_on=["equilibrium._seplim", "equilibrium._aeqdsk_time_indices"],
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
            depends_on=["equilibrium._gapin", "equilibrium._gapout", "equilibrium._gaptop", "equilibrium._gapbot", "equilibrium._aeqdsk_time_indices"],
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
            depends_on=["equilibrium._rvsid", "equilibrium._rvsod", "equilibrium._rvsiu", "equilibrium._rvsou", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_strike_point_r,
            ids_path="equilibrium.time_slice.boundary_separatrix.strike_point.r",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary_separatrix.strike_point.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._rvsid", "equilibrium._zvsid", "equilibrium._rvsod", "equilibrium._zvsod",
                       "equilibrium._rvsiu", "equilibrium._zvsiu", "equilibrium._rvsou", "equilibrium._zvsou", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_strike_point_z,
            ids_path="equilibrium.time_slice.boundary_separatrix.strike_point.z",
            docs_file=self.DOCS_PATH
        )

        # Boundary separatrix shape parameters
        # Internal dependencies for dimensionless parameters
        self.specs["equilibrium._tritop"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.TRITOP', 0, self.efit_tree),
            ],
            ids_path="equilibrium._tritop",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary_separatrix.triangularity_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._tritop", "equilibrium._aeqdsk_time_indices"],
            compose=lambda shot, raw: self._aeqdsk_on_gtime(
                shot, raw, raw[Requirement(f'{self.aeqdsk_node}.TRITOP', self.resolve_shot(shot), self.efit_tree).as_key()]
            ),
            ids_path="equilibrium.time_slice.boundary_separatrix.triangularity_upper",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._tribot"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.TRIBOT', 0, self.efit_tree),
            ],
            ids_path="equilibrium._tribot",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary_separatrix.triangularity_lower"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._tribot", "equilibrium._aeqdsk_time_indices"],
            compose=lambda shot, raw: self._aeqdsk_on_gtime(
                shot, raw, raw[Requirement(f'{self.aeqdsk_node}.TRIBOT', self.resolve_shot(shot), self.efit_tree).as_key()]
            ),
            ids_path="equilibrium.time_slice.boundary_separatrix.triangularity_lower",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._kappa"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.KAPPA', 0, self.efit_tree),
            ],
            ids_path="equilibrium._kappa",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary_separatrix.elongation"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._kappa", "equilibrium._aeqdsk_time_indices"],
            compose=lambda shot, raw: self._aeqdsk_on_gtime(
                shot, raw, raw[Requirement(f'{self.aeqdsk_node}.KAPPA', self.resolve_shot(shot), self.efit_tree).as_key()]
            ),
            ids_path="equilibrium.time_slice.boundary_separatrix.elongation",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: AMINOR (minor radius)
        self.specs["equilibrium._aminor"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.AMINOR', 0, self.efit_tree),
            ],
            ids_path="equilibrium._aminor",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary_separatrix.minor_radius"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._aminor", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_minor_radius,
            ids_path="equilibrium.time_slice.boundary_separatrix.minor_radius",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._ssibry"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.geqdsk_node}.SSIBRY', 0, self.efit_tree),
            ],
            ids_path="equilibrium._ssibry",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.boundary_separatrix.psi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._ssibry", "equilibrium._bcentr", "equilibrium._cpasma_cocos"],
            compose=self._compose_psi,
            ids_path="equilibrium.time_slice.boundary_separatrix.psi",
            docs_file=self.DOCS_PATH
        )

        # Constraints - Plasma current (ip)
        self.specs["equilibrium._plasma"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.PLASMA', 0, self.efit_tree)],
            ids_path="equilibrium._plasma",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.ip.measured"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._plasma", "equilibrium._bcentr", "equilibrium._cpasma_cocos", "equilibrium._constraint_time_indices"],
            compose=self._compose_ip_measured,
            ids_path="equilibrium.time_slice.constraints.ip.measured",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._sigpasma"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.SIGPASMA', 0, self.efit_tree)],
            ids_path="equilibrium._sigpasma",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.ip.measured_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._sigpasma", "equilibrium._constraint_time_indices"],
            compose=self._compose_ip_measured_error_upper,
            ids_path="equilibrium.time_slice.constraints.ip.measured_error_upper",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._fwtpasma"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.FWTPASMA', 0, self.efit_tree)],
            ids_path="equilibrium._fwtpasma",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.ip.weight"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._fwtpasma", "equilibrium._constraint_time_indices"],
            compose=self._compose_ip_weight,
            ids_path="equilibrium.time_slice.constraints.ip.weight",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._cpasma"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.CPASMA', 0, self.efit_tree)],
            ids_path="equilibrium._cpasma",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.ip.reconstructed"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._cpasma", "equilibrium._bcentr", "equilibrium._cpasma_cocos", "equilibrium._constraint_time_indices"],
            compose=self._compose_ip_reconstructed,
            ids_path="equilibrium.time_slice.constraints.ip.reconstructed",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._chipasma"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.CHIPASMA', 0, self.efit_tree)],
            ids_path="equilibrium._chipasma",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.ip.chi_squared"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._chipasma", "equilibrium._constraint_time_indices"],
            compose=self._compose_ip_chi_squared,
            ids_path="equilibrium.time_slice.constraints.ip.chi_squared",
            docs_file=self.DOCS_PATH
        )

        # Constraints - Poloidal field probe array (bpol_probe)
        self.specs["equilibrium._expmpi"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.EXPMPI', 0, self.efit_tree)],
            ids_path="equilibrium._expmpi",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.bpol_probe.measured"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._expmpi", "equilibrium._constraint_time_indices"],
            compose=self._compose_bpol_probe_measured,
            ids_path="equilibrium.time_slice.constraints.bpol_probe.measured",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._sigmpi"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.SIGMPI', 0, self.efit_tree)],
            ids_path="equilibrium._sigmpi",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.bpol_probe.measured_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._sigmpi", "equilibrium._constraint_time_indices"],
            compose=self._compose_bpol_probe_measured_error_upper,
            ids_path="equilibrium.time_slice.constraints.bpol_probe.measured_error_upper",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._fwtmp2"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.FWTMP2', 0, self.efit_tree)],
            ids_path="equilibrium._fwtmp2",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.bpol_probe.weight"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._fwtmp2", "equilibrium._constraint_time_indices"],
            compose=self._compose_bpol_probe_weight,
            ids_path="equilibrium.time_slice.constraints.bpol_probe.weight",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._cmpr2"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.CMPR2', 0, self.efit_tree)],
            ids_path="equilibrium._cmpr2",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.bpol_probe.reconstructed"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._cmpr2", "equilibrium._constraint_time_indices"],
            compose=self._compose_bpol_probe_reconstructed,
            ids_path="equilibrium.time_slice.constraints.bpol_probe.reconstructed",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._saimpi"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.SAIMPI', 0, self.efit_tree)],
            ids_path="equilibrium._saimpi",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.bpol_probe.chi_squared"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._saimpi", "equilibrium._constraint_time_indices"],
            compose=self._compose_bpol_probe_chi_squared,
            ids_path="equilibrium.time_slice.constraints.bpol_probe.chi_squared",
            docs_file=self.DOCS_PATH
        )

        # Constraints - Diamagnetic flux (diamagnetic_flux)
        self.specs["equilibrium._diamag"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.DIAMAG', 0, self.efit_tree)],
            ids_path="equilibrium._diamag",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.diamagnetic_flux.measured"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._diamag", "equilibrium._constraint_time_indices"],
            compose=self._compose_diamagnetic_flux_measured,
            ids_path="equilibrium.time_slice.constraints.diamagnetic_flux.measured",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._sigdia"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.SIGDIA', 0, self.efit_tree)],
            ids_path="equilibrium._sigdia",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.diamagnetic_flux.measured_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._sigdia", "equilibrium._constraint_time_indices"],
            compose=self._compose_diamagnetic_flux_measured_error_upper,
            ids_path="equilibrium.time_slice.constraints.diamagnetic_flux.measured_error_upper",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._fwtdia"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.FWTDIA', 0, self.efit_tree)],
            ids_path="equilibrium._fwtdia",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.diamagnetic_flux.weight"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._fwtdia", "equilibrium._constraint_time_indices"],
            compose=self._compose_diamagnetic_flux_weight,
            ids_path="equilibrium.time_slice.constraints.diamagnetic_flux.weight",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._cdflux"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.CDFLUX', 0, self.efit_tree)],
            ids_path="equilibrium._cdflux",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.diamagnetic_flux.reconstructed"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._cdflux", "equilibrium._constraint_time_indices"],
            compose=self._compose_diamagnetic_flux_reconstructed,
            ids_path="equilibrium.time_slice.constraints.diamagnetic_flux.reconstructed",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._chidflux"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.CHIDFLUX', 0, self.efit_tree)],
            ids_path="equilibrium._chidflux",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.diamagnetic_flux.chi_squared"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._chidflux", "equilibrium._constraint_time_indices"],
            compose=self._compose_diamagnetic_flux_chi_squared,
            ids_path="equilibrium.time_slice.constraints.diamagnetic_flux.chi_squared",
            docs_file=self.DOCS_PATH
        )

        # Constraints - Flux loop array (flux_loop)
        self.specs["equilibrium._silopt"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.SILOPT', 0, self.efit_tree)],
            ids_path="equilibrium._silopt",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.flux_loop.measured"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._silopt", "equilibrium._bcentr", "equilibrium._cpasma_cocos", "equilibrium._constraint_time_indices"],
            compose=self._compose_flux_loop_measured,
            ids_path="equilibrium.time_slice.constraints.flux_loop.measured",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._sigsil"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.SIGSIL', 0, self.efit_tree)],
            ids_path="equilibrium._sigsil",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.flux_loop.measured_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._sigsil", "equilibrium._bcentr", "equilibrium._cpasma_cocos", "equilibrium._constraint_time_indices"],
            compose=self._compose_flux_loop_measured_error_upper,
            ids_path="equilibrium.time_slice.constraints.flux_loop.measured_error_upper",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._fwtsi"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.FWTSI', 0, self.efit_tree)],
            ids_path="equilibrium._fwtsi",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.flux_loop.weight"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._fwtsi", "equilibrium._constraint_time_indices"],
            compose=self._compose_flux_loop_weight,
            ids_path="equilibrium.time_slice.constraints.flux_loop.weight",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._csilop"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.CSILOP', 0, self.efit_tree)],
            ids_path="equilibrium._csilop",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.flux_loop.reconstructed"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._csilop", "equilibrium._bcentr", "equilibrium._cpasma_cocos", "equilibrium._constraint_time_indices"],
            compose=self._compose_flux_loop_reconstructed,
            ids_path="equilibrium.time_slice.constraints.flux_loop.reconstructed",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._saisil"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.SAISIL', 0, self.efit_tree)],
            ids_path="equilibrium._saisil",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.flux_loop.chi_squared"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._saisil", "equilibrium._constraint_time_indices"],
            compose=self._compose_flux_loop_chi_squared,
            ids_path="equilibrium.time_slice.constraints.flux_loop.chi_squared",
            docs_file=self.DOCS_PATH
        )

        # Constraints - MSE polarisation angle array (mse_polarisation_angle)
        # Internal dependencies for ATAN transformations
        self.specs["equilibrium._tangam"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.measurements_node}.TANGAM', 0, self.efit_tree),
            ],
            ids_path="equilibrium._tangam",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.constraints.mse_polarisation_angle.measured"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._tangam", "equilibrium._constraint_time_indices"],
            compose=self._compose_mse_measured,
            ids_path="equilibrium.time_slice.constraints.mse_polarisation_angle.measured",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._siggam"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.measurements_node}.SIGGAM', 0, self.efit_tree),
            ],
            ids_path="equilibrium._siggam",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.constraints.mse_polarisation_angle.measured_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._siggam", "equilibrium._constraint_time_indices"],
            compose=self._compose_mse_error,
            ids_path="equilibrium.time_slice.constraints.mse_polarisation_angle.measured_error_upper",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._fwtgam"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.FWTGAM', 0, self.efit_tree)],
            ids_path="equilibrium._fwtgam",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.mse_polarisation_angle.weight"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._fwtgam", "equilibrium._constraint_time_indices"],
            compose=self._compose_mse_weight,
            ids_path="equilibrium.time_slice.constraints.mse_polarisation_angle.weight",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._cmgam"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.CMGAM', 0, self.efit_tree)],
            ids_path="equilibrium._cmgam",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.mse_polarisation_angle.reconstructed"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._cmgam", "equilibrium._constraint_time_indices"],
            compose=self._compose_mse_reconstructed,
            ids_path="equilibrium.time_slice.constraints.mse_polarisation_angle.reconstructed",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._chigam"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.CHIGAM', 0, self.efit_tree)],
            ids_path="equilibrium._chigam",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.mse_polarisation_angle.chi_squared"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._chigam", "equilibrium._constraint_time_indices"],
            compose=self._compose_mse_chi_squared,
            ids_path="equilibrium.time_slice.constraints.mse_polarisation_angle.chi_squared",
            docs_file=self.DOCS_PATH
        )

        # Constraints - PF current array (pf_current)
        # Internal dependencies for stack_outer_2 transformations
        self.specs["equilibrium._eccurt"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.measurements_node}.ECCURT', 0, self.efit_tree),
            ],
            ids_path="equilibrium._eccurt",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._fccurt"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.measurements_node}.FCCURT', 0, self.efit_tree),
            ],
            ids_path="equilibrium._fccurt",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.constraints.pf_current.measured"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._eccurt", "equilibrium._fccurt", "equilibrium._constraint_time_indices"],
            compose=self._compose_pf_current_measured,
            ids_path="equilibrium.time_slice.constraints.pf_current.measured",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._sigecc"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.measurements_node}.SIGECC', 0, self.efit_tree),
            ],
            ids_path="equilibrium._sigecc",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._sigfcc"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.measurements_node}.SIGFCC', 0, self.efit_tree),
            ],
            ids_path="equilibrium._sigfcc",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.constraints.pf_current.measured_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._sigecc", "equilibrium._sigfcc", "equilibrium._constraint_time_indices"],
            compose=self._compose_pf_current_error,
            ids_path="equilibrium.time_slice.constraints.pf_current.measured_error_upper",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._fwtec"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.measurements_node}.FWTEC', 0, self.efit_tree),
            ],
            ids_path="equilibrium._fwtec",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._fwtfc"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.measurements_node}.FWTFC', 0, self.efit_tree),
            ],
            ids_path="equilibrium._fwtfc",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.constraints.pf_current.weight"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._fwtec", "equilibrium._fwtfc", "equilibrium._constraint_time_indices"],
            compose=self._compose_pf_current_weight,
            ids_path="equilibrium.time_slice.constraints.pf_current.weight",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._cecurr"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.measurements_node}.CECURR', 0, self.efit_tree),
            ],
            ids_path="equilibrium._cecurr",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._ccbrsp"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.measurements_node}.CCBRSP', 0, self.efit_tree),
            ],
            ids_path="equilibrium._ccbrsp",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.constraints.pf_current.reconstructed"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._cecurr", "equilibrium._ccbrsp", "equilibrium._constraint_time_indices"],
            compose=self._compose_pf_current_reconstructed,
            ids_path="equilibrium.time_slice.constraints.pf_current.reconstructed",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._chiecc"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.measurements_node}.CHIECC', 0, self.efit_tree),
            ],
            ids_path="equilibrium._chiecc",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._chifcc"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.measurements_node}.CHIFCC', 0, self.efit_tree),
            ],
            ids_path="equilibrium._chifcc",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.constraints.pf_current.chi_squared"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._chiecc", "equilibrium._chifcc", "equilibrium._constraint_time_indices"],
            compose=self._compose_pf_current_chi_squared,
            ids_path="equilibrium.time_slice.constraints.pf_current.chi_squared",
            docs_file=self.DOCS_PATH
        )

        # Constraints - Pressure array (pressure)
        self.specs["equilibrium._rpress"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.RPRESS', 0, self.efit_tree)],
            ids_path="equilibrium._rpress",
            docs_file=self.DOCS_PATH
        )
        # Need SSIMAG and SSIBRY for psi transformation
        self.specs["equilibrium._ssimag"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.geqdsk_node}.SSIMAG', 0, self.efit_tree)],
            ids_path="equilibrium._ssimag",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium._ssibry"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.geqdsk_node}.SSIBRY', 0, self.efit_tree)],
            ids_path="equilibrium._ssibry",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.pressure.position.psi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._rpress", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._bcentr", "equilibrium._cpasma_cocos", "equilibrium._constraint_time_indices"],
            compose=self._compose_pressure_position_psi,
            ids_path="equilibrium.time_slice.constraints.pressure.position.psi",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._pressr"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.PRESSR', 0, self.efit_tree)],
            ids_path="equilibrium._pressr",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.pressure.measured"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._pressr", "equilibrium._constraint_time_indices"],
            compose=self._compose_pressure_measured,
            ids_path="equilibrium.time_slice.constraints.pressure.measured",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._sigpre"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.SIGPRE', 0, self.efit_tree)],
            ids_path="equilibrium._sigpre",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.pressure.measured_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._sigpre", "equilibrium._constraint_time_indices"],
            compose=self._compose_pressure_error,
            ids_path="equilibrium.time_slice.constraints.pressure.measured_error_upper",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._fwtpre"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.FWTPRE', 0, self.efit_tree)],
            ids_path="equilibrium._fwtpre",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.pressure.weight"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._fwtpre", "equilibrium._constraint_time_indices"],
            compose=self._compose_pressure_weight,
            ids_path="equilibrium.time_slice.constraints.pressure.weight",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._cpress"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.CPRESS', 0, self.efit_tree)],
            ids_path="equilibrium._cpress",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.pressure.reconstructed"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._cpress", "equilibrium._constraint_time_indices"],
            compose=self._compose_pressure_reconstructed,
            ids_path="equilibrium.time_slice.constraints.pressure.reconstructed",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._saipre"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.SAIPRE', 0, self.efit_tree)],
            ids_path="equilibrium._saipre",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.pressure.chi_squared"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._saipre", "equilibrium._constraint_time_indices"],
            compose=self._compose_pressure_chi_squared,
            ids_path="equilibrium.time_slice.constraints.pressure.chi_squared",
            docs_file=self.DOCS_PATH
        )

        # Constraints - Toroidal current density array (j_tor)
        # Note: j_tor only has position.psi and measured (no error, weight, reconstructed, chi_squared)
        self.specs["equilibrium._sizeroj"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.SIZEROJ', 0, self.efit_tree)],
            ids_path="equilibrium._sizeroj",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.j_tor.position.psi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._sizeroj", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._bcentr", "equilibrium._cpasma_cocos", "equilibrium._constraint_time_indices"],
            compose=self._compose_j_tor_position_psi,
            ids_path="equilibrium.time_slice.constraints.j_tor.position.psi",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium._vzeroj"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.measurements_node}.VZEROJ', 0, self.efit_tree)],
            ids_path="equilibrium._vzeroj",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.constraints.j_tor.measured"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._vzeroj", "equilibrium._bcentr", "equilibrium._cpasma_cocos", "equilibrium._constraint_time_indices"],
            compose=self._compose_j_tor_measured,
            ids_path="equilibrium.time_slice.constraints.j_tor.measured",
            docs_file=self.DOCS_PATH
        )

        # Global quantities
        # ip - plasma current (needs COCOS TOR transformation)
        self.specs["equilibrium.time_slice.global_quantities.ip"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._cpasma", "equilibrium._bcentr", "equilibrium._cpasma_cocos"],
            compose=self._compose_global_ip,
            ids_path="equilibrium.time_slice.global_quantities.ip",
            docs_file=self.DOCS_PATH
        )

        # li_3 - internal inductance
        self.specs["equilibrium._li3"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.aeqdsk_node}.LI3', 0, self.efit_tree)],
            ids_path="equilibrium._li3",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.global_quantities.li_3"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._li3", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_global_li_3,
            ids_path="equilibrium.time_slice.global_quantities.li_3",
            docs_file=self.DOCS_PATH
        )

        # magnetic_axis.r - R position
        self.specs["equilibrium._rmaxis"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.geqdsk_node}.RMAXIS', 0, self.efit_tree)],
            ids_path="equilibrium._rmaxis",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.global_quantities.magnetic_axis.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._rmaxis"],
            compose=self._compose_global_magnetic_axis_r,
            ids_path="equilibrium.time_slice.global_quantities.magnetic_axis.r",
            docs_file=self.DOCS_PATH
        )

        # magnetic_axis.z - Z position
        self.specs["equilibrium._zmaxis"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.geqdsk_node}.ZMAXIS', 0, self.efit_tree)],
            ids_path="equilibrium._zmaxis",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.global_quantities.magnetic_axis.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._zmaxis"],
            compose=self._compose_global_magnetic_axis_z,
            ids_path="equilibrium.time_slice.global_quantities.magnetic_axis.z",
            docs_file=self.DOCS_PATH
        )

        # magnetic_axis.b_field_tor - toroidal field at axis
        self.specs["equilibrium._bt0"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.aeqdsk_node}.BT0', 0, self.efit_tree)],
            ids_path="equilibrium._bt0",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.global_quantities.magnetic_axis.b_field_tor"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._bt0", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_global_magnetic_axis_b_field_tor,
            ids_path="equilibrium.time_slice.global_quantities.magnetic_axis.b_field_tor",
            docs_file=self.DOCS_PATH
        )

        # psi_axis - psi at magnetic axis (needs COCOS PSI transformation)
        self.specs["equilibrium.time_slice.global_quantities.psi_axis"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._ssimag", "equilibrium._bcentr", "equilibrium._cpasma_cocos"],
            compose=self._compose_global_psi_axis,
            ids_path="equilibrium.time_slice.global_quantities.psi_axis",
            docs_file=self.DOCS_PATH
        )

        # psi_boundary - psi at boundary (needs COCOS PSI transformation)
        self.specs["equilibrium.time_slice.global_quantities.psi_boundary"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._ssibry", "equilibrium._bcentr", "equilibrium._cpasma_cocos"],
            compose=self._compose_global_psi_boundary,
            ids_path="equilibrium.time_slice.global_quantities.psi_boundary",
            docs_file=self.DOCS_PATH
        )

        # area - plasma cross-sectional area
        self.specs["equilibrium._area"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.aeqdsk_node}.AREA', 0, self.efit_tree)],
            ids_path="equilibrium._area",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.global_quantities.area"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._area", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_global_area,
            ids_path="equilibrium.time_slice.global_quantities.area",
            docs_file=self.DOCS_PATH
        )

        # surface - plasma surface area
        self.specs["equilibrium._psurfa"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.aeqdsk_node}.PSURFA', 0, self.efit_tree)],
            ids_path="equilibrium._psurfa",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.global_quantities.surface"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._psurfa", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_global_surface,
            ids_path="equilibrium.time_slice.global_quantities.surface",
            docs_file=self.DOCS_PATH
        )

        # volume - plasma volume
        self.specs["equilibrium._volume"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.aeqdsk_node}.VOLUME', 0, self.efit_tree)],
            ids_path="equilibrium._volume",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.global_quantities.volume"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._volume", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_global_volume,
            ids_path="equilibrium.time_slice.global_quantities.volume",
            docs_file=self.DOCS_PATH
        )

        # beta_pol - poloidal beta
        self.specs["equilibrium._betap"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.aeqdsk_node}.BETAP', 0, self.efit_tree)],
            ids_path="equilibrium._betap",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.global_quantities.beta_pol"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._betap", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_global_beta_pol,
            ids_path="equilibrium.time_slice.global_quantities.beta_pol",
            docs_file=self.DOCS_PATH
        )

        # beta_tor - toroidal beta
        self.specs["equilibrium._betat"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.aeqdsk_node}.BETAT', 0, self.efit_tree)],
            ids_path="equilibrium._betat",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.global_quantities.beta_tor"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._betat", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_global_beta_tor,
            ids_path="equilibrium.time_slice.global_quantities.beta_tor",
            docs_file=self.DOCS_PATH
        )

        # beta_normal - normalized beta
        self.specs["equilibrium._betan"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.aeqdsk_node}.BETAN', 0, self.efit_tree)],
            ids_path="equilibrium._betan",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.global_quantities.beta_normal"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._betan", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_global_beta_normal,
            ids_path="equilibrium.time_slice.global_quantities.beta_normal",
            docs_file=self.DOCS_PATH
        )

        # q_95 - safety factor at 95% flux (needs COCOS Q transformation)
        self.specs["equilibrium._q95"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.aeqdsk_node}.Q95', 0, self.efit_tree)],
            ids_path="equilibrium._q95",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.global_quantities.q_95"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._q95", "equilibrium._bcentr", "equilibrium._cpasma_cocos", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_global_q_95,
            ids_path="equilibrium.time_slice.global_quantities.q_95",
            docs_file=self.DOCS_PATH
        )

        # q_axis - safety factor at axis (needs COCOS Q transformation)
        self.specs["equilibrium._q0"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.aeqdsk_node}.Q0', 0, self.efit_tree)],
            ids_path="equilibrium._q0",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.global_quantities.q_axis"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._q0", "equilibrium._bcentr", "equilibrium._cpasma_cocos", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_global_q_axis,
            ids_path="equilibrium.time_slice.global_quantities.q_axis",
            docs_file=self.DOCS_PATH
        )

        # q_min.value - minimum safety factor (needs COCOS Q transformation)
        self.specs["equilibrium._qmin"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.aeqdsk_node}.QMIN', 0, self.efit_tree)],
            ids_path="equilibrium._qmin",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.global_quantities.q_min.value"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._qmin", "equilibrium._bcentr", "equilibrium._cpasma_cocos", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_global_q_min_value,
            ids_path="equilibrium.time_slice.global_quantities.q_min.value",
            docs_file=self.DOCS_PATH
        )

        # 1D Profiles
        # dpressure_dpsi - pressure derivative (needs COCOS dPSI transformation)
        self.specs["equilibrium._pprime"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.geqdsk_node}.PPRIME', 0, self.efit_tree)],
            ids_path="equilibrium._pprime",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.dpressure_dpsi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._pprime", "equilibrium._bcentr", "equilibrium._cpasma_cocos"],
            compose=self._compose_profiles_1d_dpressure_dpsi,
            ids_path="equilibrium.time_slice.profiles_1d.dpressure_dpsi",
            docs_file=self.DOCS_PATH
        )

        # f - poloidal current function
        self.specs["equilibrium._fpol"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.geqdsk_node}.FPOL', 0, self.efit_tree)],
            ids_path="equilibrium._fpol",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.f"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._fpol"],
            compose=self._compose_profiles_1d_f,
            ids_path="equilibrium.time_slice.profiles_1d.f",
            docs_file=self.DOCS_PATH
        )

        # f_df_dpsi - derivative of F (needs COCOS dPSI transformation)
        self.specs["equilibrium._ffprim"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.geqdsk_node}.FFPRIM', 0, self.efit_tree)],
            ids_path="equilibrium._ffprim",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.f_df_dpsi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._ffprim", "equilibrium._bcentr", "equilibrium._cpasma_cocos"],
            compose=self._compose_profiles_1d_f_df_dpsi,
            ids_path="equilibrium.time_slice.profiles_1d.f_df_dpsi",
            docs_file=self.DOCS_PATH
        )

        # pressure - pressure profile
        self.specs["equilibrium._pres"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.geqdsk_node}.PRES', 0, self.efit_tree)],
            ids_path="equilibrium._pres",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.pressure"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._pres"],
            compose=self._compose_profiles_1d_pressure,
            ids_path="equilibrium.time_slice.profiles_1d.pressure",
            docs_file=self.DOCS_PATH
        )

        # psi - poloidal flux profile (needs geqdsk_psi transformation)
        self.specs["equilibrium._psin"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.geqdsk_node}.PSIN', 0, self.efit_tree)],
            ids_path="equilibrium._psin",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.psi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._psin", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._bcentr", "equilibrium._cpasma_cocos"],
            compose=self._compose_profiles_1d_psi,
            ids_path="equilibrium.time_slice.profiles_1d.psi",
            docs_file=self.DOCS_PATH
        )

        # q - safety factor profile (needs COCOS Q transformation)
        self.specs["equilibrium._qpsi"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.geqdsk_node}.QPSI', 0, self.efit_tree)],
            ids_path="equilibrium._qpsi",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.q"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._qpsi", "equilibrium._bcentr", "equilibrium._cpasma_cocos"],
            compose=self._compose_profiles_1d_q,
            ids_path="equilibrium.time_slice.profiles_1d.q",
            docs_file=self.DOCS_PATH
        )

        # rho_tor_norm - normalized toroidal flux coordinate
        self.specs["equilibrium._rhovn"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.geqdsk_node}.RHOVN', 0, self.efit_tree)],
            ids_path="equilibrium._rhovn",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.rho_tor_norm"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._rhovn"],
            compose=self._compose_profiles_1d_rho_tor_norm,
            ids_path="equilibrium.time_slice.profiles_1d.rho_tor_norm",
            docs_file=self.DOCS_PATH
        )

        # j_tor - toroidal current density profile (needs interpolate_psi_1d transformation)
        self.specs["equilibrium._fluxfun_psi"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'\\{self.efit_tree}::TOP.RESULTS.FLUXFUN.PSI', 0, self.efit_tree)],
            ids_path="equilibrium._fluxfun_psi",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium._fluxfun_jeff"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'\\{self.efit_tree}::TOP.RESULTS.FLUXFUN.JEFF', 0, self.efit_tree)],
            ids_path="equilibrium._fluxfun_jeff",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.j_tor"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._fluxfun_psi", "equilibrium._fluxfun_jeff", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._psin"],
            compose=self._compose_profiles_1d_j_tor,
            ids_path="equilibrium.time_slice.profiles_1d.j_tor",
            docs_file=self.DOCS_PATH
        )

        # j_parallel - parallel current density profile (needs interpolate_psi_1d transformation)
        self.specs["equilibrium._fluxfun_jll"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'\\{self.efit_tree}::TOP.RESULTS.FLUXFUN.JLL', 0, self.efit_tree)],
            ids_path="equilibrium._fluxfun_jll",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.j_parallel"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._fluxfun_psi", "equilibrium._fluxfun_jll", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._psin"],
            compose=self._compose_profiles_1d_j_parallel,
            ids_path="equilibrium.time_slice.profiles_1d.j_parallel",
            docs_file=self.DOCS_PATH
        )

        # volume - volume profile (needs interpolate_psi_1d transformation)
        self.specs["equilibrium._fluxfun_vol"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'\\{self.efit_tree}::TOP.RESULTS.FLUXFUN.VOL', 0, self.efit_tree)],
            ids_path="equilibrium._fluxfun_vol",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.volume"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._fluxfun_psi", "equilibrium._fluxfun_vol", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._psin"],
            compose=self._compose_profiles_1d_volume,
            ids_path="equilibrium.time_slice.profiles_1d.volume",
            docs_file=self.DOCS_PATH
        )

        # trapped_fraction, gm5, b_field_average - FLUXFUN flux-surface averages interpolated like volume (not in OMAS)
        self.specs["equilibrium._fluxfun_fc"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'\\{self.efit_tree}::TOP.RESULTS.FLUXFUN.FC', 0, self.efit_tree)],
            ids_path="equilibrium._fluxfun_fc",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium._fluxfun_b2ave"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'\\{self.efit_tree}::TOP.RESULTS.FLUXFUN.B2AVE', 0, self.efit_tree)],
            ids_path="equilibrium._fluxfun_b2ave",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium._fluxfun_bave"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'\\{self.efit_tree}::TOP.RESULTS.FLUXFUN.BAVE', 0, self.efit_tree)],
            ids_path="equilibrium._fluxfun_bave",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.trapped_fraction"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._fluxfun_psi", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._psin", "equilibrium._fluxfun_fc"],
            compose=self._compose_profiles_1d_trapped_fraction,
            ids_path="equilibrium.time_slice.profiles_1d.trapped_fraction",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.gm5"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._fluxfun_psi", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._psin", "equilibrium._fluxfun_b2ave", "equilibrium._bt0", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_profiles_1d_gm5,
            ids_path="equilibrium.time_slice.profiles_1d.gm5",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.b_field_average"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._fluxfun_psi", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._psin", "equilibrium._fluxfun_bave", "equilibrium._bt0", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_profiles_1d_b_field_average,
            ids_path="equilibrium.time_slice.profiles_1d.b_field_average",
            docs_file=self.DOCS_PATH
        )

        # psi_norm - the GEQDSK.PSIN grid
        self.specs["equilibrium.time_slice.profiles_1d.psi_norm"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._psin", "equilibrium._gtime"],
            compose=self._compose_profiles_1d_psi_norm,
            ids_path="equilibrium.time_slice.profiles_1d.psi_norm",
            docs_file=self.DOCS_PATH
        )

        # Flux surface geometry: surfaces traced in GEQDSK.PSIRZ between axis and boundary; the axis value comes from
        # GEQDSK.RMAXIS/ZMAXIS or AEQDSK.KAPPA0, the boundary value from the AEQDSK boundary scalars (not in OMAS)
        self.specs["equilibrium._kappa0"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(f'{self.aeqdsk_node}.KAPPA0', 0, self.efit_tree)],
            ids_path="equilibrium._kappa0",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.elongation"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._r_grid", "equilibrium._z_grid", "equilibrium._psirz", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._psin", "equilibrium._rmaxis", "equilibrium._zmaxis", "equilibrium._aeqdsk_time_indices", "equilibrium._kappa0", "equilibrium._kappa"],
            compose=self._compose_profiles_1d_elongation,
            ids_path="equilibrium.time_slice.profiles_1d.elongation",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.triangularity_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._r_grid", "equilibrium._z_grid", "equilibrium._psirz", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._psin", "equilibrium._rmaxis", "equilibrium._zmaxis", "equilibrium._aeqdsk_time_indices", "equilibrium._tritop"],
            compose=self._compose_profiles_1d_triangularity_upper,
            ids_path="equilibrium.time_slice.profiles_1d.triangularity_upper",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.triangularity_lower"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._r_grid", "equilibrium._z_grid", "equilibrium._psirz", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._psin", "equilibrium._rmaxis", "equilibrium._zmaxis", "equilibrium._aeqdsk_time_indices", "equilibrium._tribot"],
            compose=self._compose_profiles_1d_triangularity_lower,
            ids_path="equilibrium.time_slice.profiles_1d.triangularity_lower",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.r_inboard"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._r_grid", "equilibrium._z_grid", "equilibrium._psirz", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._psin", "equilibrium._rmaxis", "equilibrium._zmaxis", "equilibrium._aeqdsk_time_indices", "equilibrium._rsurf", "equilibrium._aminor"],
            compose=self._compose_profiles_1d_r_inboard,
            ids_path="equilibrium.time_slice.profiles_1d.r_inboard",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.r_outboard"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._r_grid", "equilibrium._z_grid", "equilibrium._psirz", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._psin", "equilibrium._rmaxis", "equilibrium._zmaxis", "equilibrium._aeqdsk_time_indices", "equilibrium._rsurf", "equilibrium._aminor"],
            compose=self._compose_profiles_1d_r_outboard,
            ids_path="equilibrium.time_slice.profiles_1d.r_outboard",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.geometric_axis.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._r_grid", "equilibrium._z_grid", "equilibrium._psirz", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._psin", "equilibrium._rmaxis", "equilibrium._zmaxis", "equilibrium._aeqdsk_time_indices", "equilibrium._rsurf"],
            compose=self._compose_profiles_1d_geometric_axis_r,
            ids_path="equilibrium.time_slice.profiles_1d.geometric_axis.r",
            docs_file=self.DOCS_PATH
        )
        self.specs["equilibrium.time_slice.profiles_1d.geometric_axis.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._r_grid", "equilibrium._z_grid", "equilibrium._psirz", "equilibrium._ssimag", "equilibrium._ssibry", "equilibrium._psin", "equilibrium._rmaxis", "equilibrium._zmaxis", "equilibrium._aeqdsk_time_indices", "equilibrium._zsurf"],
            compose=self._compose_profiles_1d_geometric_axis_z,
            ids_path="equilibrium.time_slice.profiles_1d.geometric_axis.z",
            docs_file=self.DOCS_PATH
        )

        # === Auxiliary nodes for profiles_2d ===

        # R grid from GEQDSK
        self.specs["equilibrium._r_grid"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.geqdsk_node}.R', 0, self.efit_tree)
            ],
            ids_path="equilibrium._r_grid",
            docs_file=self.DOCS_PATH
        )

        # Z grid from GEQDSK
        self.specs["equilibrium._z_grid"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.geqdsk_node}.Z', 0, self.efit_tree)
            ],
            ids_path="equilibrium._z_grid",
            docs_file=self.DOCS_PATH
        )

        # PSIRZ from GEQDSK
        self.specs["equilibrium._psirz"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.geqdsk_node}.PSIRZ', 0, self.efit_tree)
            ],
            ids_path="equilibrium._psirz",
            docs_file=self.DOCS_PATH
        )

        # === profiles_2d ===

        self.specs["equilibrium.time_slice.profiles_2d.grid.dim1"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._r_grid", "equilibrium._bcentr"],
            compose=self._compose_profiles_2d_grid_dim1,
            ids_path="equilibrium.time_slice.profiles_2d.grid.dim1",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.profiles_2d.grid.dim2"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._z_grid", "equilibrium._bcentr"],
            compose=self._compose_profiles_2d_grid_dim2,
            ids_path="equilibrium.time_slice.profiles_2d.grid.dim2",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.profiles_2d.grid_type.index"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._bcentr"],
            compose=self._compose_profiles_2d_grid_type_index,
            ids_path="equilibrium.time_slice.profiles_2d.grid_type.index",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.profiles_2d.type.index"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._bcentr"],
            compose=self._compose_profiles_2d_type_index,
            ids_path="equilibrium.time_slice.profiles_2d.type.index",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.profiles_2d.psi"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._psirz", "equilibrium._bcentr", "equilibrium._cpasma_cocos"],
            compose=self._compose_profiles_2d_psi,
            ids_path="equilibrium.time_slice.profiles_2d.psi",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.profiles_2d.b_field_tor"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "equilibrium._r_grid", "equilibrium._z_grid", "equilibrium._bcentr",
                "equilibrium._psirz", "equilibrium._cpasma_cocos",
                "equilibrium._psin", "equilibrium._ssimag", "equilibrium._ssibry",
                "equilibrium._fpol"
            ],
            compose=self._compose_profiles_2d_b_field_tor,
            ids_path="equilibrium.time_slice.profiles_2d.b_field_tor",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.profiles_2d.b_field_r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "equilibrium._r_grid", "equilibrium._z_grid", "equilibrium._bcentr",
                "equilibrium._psirz", "equilibrium._cpasma_cocos"
            ],
            compose=self._compose_profiles_2d_b_field_r,
            ids_path="equilibrium.time_slice.profiles_2d.b_field_r",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.profiles_2d.b_field_z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "equilibrium._r_grid", "equilibrium._z_grid", "equilibrium._bcentr",
                "equilibrium._psirz", "equilibrium._cpasma_cocos"
            ],
            compose=self._compose_profiles_2d_b_field_z,
            ids_path="equilibrium.time_slice.profiles_2d.b_field_z",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.profiles_2d.j_tor"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "equilibrium._r_grid", "equilibrium._z_grid", "equilibrium._bcentr",
                "equilibrium._psirz", "equilibrium._cpasma_cocos",
                "equilibrium._ssimag", "equilibrium._ssibry",
                "equilibrium._rbbbs", "equilibrium._zbbbs"
            ],
            compose=self._compose_profiles_2d_j_tor,
            ids_path="equilibrium.time_slice.profiles_2d.j_tor",
            docs_file=self.DOCS_PATH
        )

        # === Auxiliary nodes for vacuum_toroidal_field ===

        # RZERO from GEQDSK
        self.specs["equilibrium._rzero"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.geqdsk_node}.RZERO', 0, self.efit_tree)
            ],
            ids_path="equilibrium._rzero",
            docs_file=self.DOCS_PATH
        )

        # === vacuum_toroidal_field ===

        self.specs["equilibrium.vacuum_toroidal_field.b0"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._bcentr"],
            compose=self._compose_vacuum_b0,
            ids_path="equilibrium.vacuum_toroidal_field.b0",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.vacuum_toroidal_field.r0"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._rzero"],
            compose=self._compose_vacuum_r0,
            ids_path="equilibrium.vacuum_toroidal_field.r0",
            docs_file=self.DOCS_PATH
        )

        # === Auxiliary nodes for convergence ===

        # CERROR from MEASUREMENTS
        self.specs["equilibrium._cerror"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'\\{self.efit_tree}::TOP.MEASUREMENTS.CERROR', 0, self.efit_tree)
            ],
            ids_path="equilibrium._cerror",
            docs_file=self.DOCS_PATH
        )

        # AEQDSK ERROR
        self.specs["equilibrium._aeqdsk_error"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.aeqdsk_node}.ERROR', 0, self.efit_tree)
            ],
            ids_path="equilibrium._aeqdsk_error",
            docs_file=self.DOCS_PATH
        )

        # === convergence ===

        self.specs["equilibrium.time_slice.convergence.iterations_n"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._cerror", "equilibrium._constraint_time_indices"],
            compose=self._compose_convergence_iterations_n,
            ids_path="equilibrium.time_slice.convergence.iterations_n",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.convergence.grad_shafranov_deviation_value"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._aeqdsk_error", "equilibrium._aeqdsk_time_indices"],
            compose=self._compose_convergence_grad_shafranov_deviation_value,
            ids_path="equilibrium.time_slice.convergence.grad_shafranov_deviation_value",
            docs_file=self.DOCS_PATH
        )

        self.specs["equilibrium.time_slice.convergence.grad_shafranov_deviation_expression.index"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._gtime"],
            compose=self._compose_convergence_grad_shafranov_deviation_expression_index,
            ids_path="equilibrium.time_slice.convergence.grad_shafranov_deviation_expression.index",
            docs_file=self.DOCS_PATH
        )

    # Compose functions

    def _compose_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose time array (convert ms to seconds).

        OMAS: \\EFIT::TOP.RESULTS.GEQDSK.GTIME / 1000.
        """
        gtime_key = Requirement(f'{self.geqdsk_node}.GTIME', self.resolve_shot(shot), self.efit_tree).as_key()
        gtime_ms = raw_data[gtime_key]
        return gtime_ms / 1000.0  # Convert milliseconds to seconds

    def _compose_code_parameters(self, shot: int, raw_data: dict) -> str:
        """
        Compose the EFIT code parameters as XML: the basis function coefficients of p' and FF' (XRSP), the basis
        functions of p' and FF' of every slice and EFIT's rigid vertical shift settings (fitdelz, ifitdelz).

        XRSP is brsp/darea of the last EFIT iteration (write_m.F90), the kppcur coefficients of p' followed by
        the kffcur coefficients of FF'. With the profile basis functions of the run they give the profiles
        EFIT evaluates on every grid node (pressure.F90 ppcurr, current.f90 fpcurr):

            pprime(psi_n) = sum_j xrsp_j bsppel_j(psi_n),   ffprim(psi_n) = mu0 sum_j xrsp_(kppcur+j) bsffel_j(psi_n)

        in g-file units, before EFIT's output sign: the g-file PPRIME and FFPRIM are these sums times sign_out = -1
        where the measured Ip of the slice (MEASUREMENTS.PLASMA) is > 0 and 1 otherwise (write_g.f90), while XRSP is
        written without it. The reconstructed CPASMA does not decide it; it keeps its sign where the measured Ip is 0.
        One <time_slice> per equilibrium time slice (XRSP is on the MEASUREMENTS time base), the unused entries
        (netCDF fill values) are dropped and the float32 values written in their shortest exact form.

        The basis (kppfnc, kfffnc, knots, tensions) is an input of the run, set in the namelist of the slice
        (`_compose_slice_namelists`), with EFIT's defaults for what it does not set (efit_basis.basis_settings).
        <basis> holds one <time_slice> per equilibrium time slice with the `source` of its settings (kfile, snap or
        defaults) and the bases of p' and FF' (efit_basis.write_basis). The coefficients are scaled to the units and
        COCOS of profiles_1d.dpressure_dpsi and f_df_dpsi, and every basis is checked against these stored profiles;
        where EFIT moved the knots of a spline (kautoknt) they are fitted to them (efit_basis.resolve_basis). A slice
        whose settings are not supported or whose basis misses the stored profiles has no bases but an `error`.
        efit_basis.efit_profile_basis evaluates a slice of the XML.

        Every part is written when its inputs exist: <xrsp> needs XRSP, <basis> also the g-file profiles and the
        measured Ip, <fitdelz> a namelist. The field is NoData only when none of them can be written.

        With fitdelz EFIT fits a rigid vertical shift of the plasma: ifitdelz = 1 translates the total flux by it
        (pflux.F90), ifitdelz = 3 the plasma current (current.f90). The settings come from the same namelists, &INWANT
        of a k-file and &EFITIN of a snap file. <fitdelz> and <ifitdelz> hold one <time_slice> per equilibrium time
        slice whose namelist sets the value. A value the namelist does not set is left out (EFIT's default applies,
        set_defaults.f90: fitdelz = .false., ifitdelz = 1), and both are absent for runs whose namelists set neither.

        Returns:
            XML string

        Raises:
            ValueError: If none of the parts can be written, which compose() reports as the NoData of an input.
        """
        inputs = {
            name: raw_data[Requirement(f'{node}.{name}', self.resolve_shot(shot), self.efit_tree).as_key()]
            for node, names in (
                (self.geqdsk_node, ("GTIME", "PSIN", "PPRIME", "FFPRIM", "BCENTR", "CPASMA")),
                (self.measurements_node, ("MTIME", "XRSP", "PLASMA")),
            )
            for name in names
        }
        missing = {name for name, value in inputs.items() if isinstance(value, NoData)}
        namelists = [] if "GTIME" in missing else self._compose_slice_namelists(shot, raw_data)

        root = ET.Element("parameters")
        if not missing & XRSP_INPUTS:
            time_indices = self._compose_constraint_time_indices(shot, raw_data)
            xrsp = np.asarray(inputs["XRSP"], dtype=np.float32)[time_indices]
            coefficients = [values[np.abs(values) < XRSP_FILL_THRESHOLD] for values in xrsp]
            xrsp_element = ET.SubElement(root, "xrsp")
            for index, used in enumerate(coefficients):
                ET.SubElement(xrsp_element, "time_slice", index=str(index)).text = format_floats(used)

        if not missing & BASIS_INPUTS:
            sign_out = np.where(np.asarray(inputs["PLASMA"])[time_indices] > 0, -1.0, 1.0)
            psin = np.asarray(inputs["PSIN"], dtype=np.float64)
            profiles = {
                "pprime": np.asarray(self._compose_profiles_1d_dpressure_dpsi(shot, raw_data), dtype=np.float64),
                "ffprime": np.asarray(self._compose_profiles_1d_f_df_dpsi(shot, raw_data), dtype=np.float64),
            }
            # from the g-file sums of the coefficients to these profiles, before sign_out
            factors = {
                "pprime": self._apply_cocos_transform(
                    1.0, shot, raw_data, "equilibrium.time_slice.profiles_1d.dpressure_dpsi"),
                "ffprime": EFIT_MU0 * self._apply_cocos_transform(
                    1.0, shot, raw_data, "equilibrium.time_slice.profiles_1d.f_df_dpsi"),
            }
            basis_element = ET.SubElement(root, "basis")
            for index, ((source, namelist), used) in enumerate(zip(namelists, coefficients)):
                slice_element = ET.SubElement(basis_element, "time_slice", index=str(index), source=source)
                values = {} if namelist is None else self._parse_namelist(namelist, BASIS_GROUPS[source], BASIS_KEYS)
                try:
                    settings = basis_settings(values)
                    shares = split_coefficients(used.astype(np.float64), settings)
                except ValueError as error:
                    # settings efit_basis does not support, e.g. kinds 1-5 or rotation coefficients in XRSP
                    slice_element.set("error", str(error))
                    continue
                resolved = {
                    profile: resolve_basis(
                        profile_settings, sign_out[index] * factors[profile] * shares[profile], psin,
                        profiles[profile][index]
                    )
                    for profile, profile_settings in settings.items()
                }
                misses = [
                    f"{profile} ({settings[profile]['kind']}) by a relative rms of {rms:.1e}"
                    for profile, (_, rms, _) in resolved.items()
                    if rms > RECOVERY_TOLERANCE
                ]
                if misses:
                    slice_element.set(
                        "error",
                        f"EFIT's basis (settings from {source}) misses the stored {' and '.join(misses)} "
                        f"(tolerance {RECOVERY_TOLERANCE:.0e}).",
                    )
                    continue
                for profile, (basis, _, fitted) in resolved.items():
                    write_basis(slice_element, profile, basis, fitted)

        # fitdelz and ifitdelz by equilibrium time slice index
        fitdelz = {
            index: self._parse_namelist(namelist, FITDELZ_GROUPS[source], FITDELZ_KEYS)
            for index, (source, namelist) in enumerate(namelists)
            if namelist is not None
        }
        if any(fitdelz.values()):
            elements = {key: ET.SubElement(root, key) for key in FITDELZ_KEYS}
            for index, values in fitdelz.items():
                for key, value in values.items():
                    ET.SubElement(elements[key], "time_slice", index=str(index)).text = str(value).lower()
        if len(root) == 0:
            raise ValueError(
                f"No EFIT code parameters for shot {shot}, tree {self.efit_tree} run {self.efit_run_id!r}: "
                f"{sorted(missing)} are missing and no namelist sets fitdelz."
            )
        return ET.tostring(root, encoding="unicode")

    def _compose_slice_namelists(self, shot: int, raw_data: dict) -> list:
        """
        The namelist of every equilibrium time slice and its source.

        For the run-id runs of the EFIT tree (CAKE runs) that is the k-file of the slice ("kfile", NAMELISTS:KEQDSKS,
        matched to GTIME by KTIME), for the standard trees the snap file of the run ("snap", TOP:NAMELIST), which holds
        for every slice. A slice without a namelist, e.g. of a run that stores none, has None ("defaults").

        Returns:
            (source, namelist text or None) per equilibrium time slice
        """
        gtime_key = Requirement(f'{self.geqdsk_node}.GTIME', self.resolve_shot(shot), self.efit_tree).as_key()
        gtime = np.asarray(raw_data[gtime_key], dtype=np.float64)
        if self.namelist_source == "snap":
            snap = raw_data[Requirement(f'\\{self.efit_tree}::TOP:NAMELIST', self.resolve_shot(shot), self.efit_tree).as_key()]
            if isinstance(snap, NoData):
                return [("defaults", None)] * len(gtime)
            return [("snap", str(np.asarray(snap)))] * len(gtime)

        keqdsks = raw_data[Requirement(f'{self.namelists_node}:KEQDSKS', self.resolve_shot(shot), self.efit_tree).as_key()]
        ktime = raw_data[Requirement(f'{self.namelists_node}:KEQDSKS:KTIME', self.resolve_shot(shot), self.efit_tree).as_key()]
        if isinstance(keqdsks, NoData) or isinstance(ktime, NoData):
            return [("defaults", None)] * len(gtime)
        namelists = np.asarray(keqdsks).astype(str)
        ktime = np.asarray(ktime, dtype=np.float64)
        if namelists.ndim != 2 or len(namelists) != len(ktime):
            raise ValueError(
                f"KEQDSKS holds {namelists.shape} namelist lines for {len(ktime)} KTIME entries, expected one row "
                "of lines per k-file"
            )
        slices = []
        for time in gtime:
            matches = np.flatnonzero(np.abs(ktime - time) < KTIME_TOLERANCE)
            slices.append(("kfile", "\n".join(namelists[matches[0]])) if len(matches) else ("defaults", None))
        return slices

    @staticmethod
    def _parse_namelist(namelist: str, group: str, keys: tuple) -> dict:
        """The values of `keys` a namelist sets in `group`, e.g. &IN1 or &INWANT of a k-file (one line per record) or
        &EFITIN of a snap file. f90nml skips the text around the groups, such as the comments that follow the
        &EFITIN group of the stored snap files."""
        values = f90nml.reads(namelist).get(group, {})
        return {key: values[key] for key in keys if key in values}

    def _compose_constraint_time_indices(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose time index mapping for constraints.

        Constraints are stored in MEASUREMENTS which have a different time dimension (MTIME)
        than the equilibrium time base (GTIME). This is because between-shot EFIT filters
        are applied to RESULTS.GEQDSK but not to MEASUREMENTS.

        GTIME is therefore a subset of MTIME. Each GTIME is matched to the MTIME at the same time,
        and a GTIME without a matching MTIME raises. OMAS (_common.py) instead clamps searchsorted
        to len(GTIME) - 1, which maps late slices to the wrong MEASUREMENTS row when MTIME is longer.

        Returns:
            Array of indices into MTIME for each GTIME value

        Raises:
            ValueError: If a GTIME has no MTIME within 0.1 ms
        """
        gtime_key = Requirement(f'{self.geqdsk_node}.GTIME', self.resolve_shot(shot), self.efit_tree).as_key()
        mtime_key = Requirement(f'{self.measurements_node}.MTIME', self.resolve_shot(shot), self.efit_tree).as_key()

        gtime = raw_data[gtime_key]
        mtime = raw_data[mtime_key]

        indices = np.abs(mtime[:, None] - gtime[None, :]).argmin(axis=0)
        # Times are in ms and MTIME is spaced by >= 20 ms
        unmatched = ~np.isclose(mtime[indices], gtime, rtol=0, atol=0.1)
        if unmatched.any():
            raise ValueError(
                f"GTIME {gtime[unmatched]} ms of shot {shot} have no matching MTIME in {self.measurements_node}"
            )
        return indices

    def _compose_aeqdsk_time_indices(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose time index mapping for AEQDSK quantities.

        EFIT writes no a-file for some slices that have a g-file, so ATIME can be shorter than GTIME.
        Each GTIME is matched to the ATIME at the same time. Unlike MTIME, a GTIME without a matching
        ATIME is expected and does not raise.

        Returns:
            Array of indices into ATIME for each GTIME value, -1 where the slice has no a-file
        """
        gtime_key = Requirement(f'{self.geqdsk_node}.GTIME', self.resolve_shot(shot), self.efit_tree).as_key()
        atime_key = Requirement(f'{self.aeqdsk_node}.ATIME', self.resolve_shot(shot), self.efit_tree).as_key()

        gtime = raw_data[gtime_key]
        atime = raw_data[atime_key]

        indices = np.abs(atime[:, None] - gtime[None, :]).argmin(axis=0)
        matched = np.isclose(atime[indices], gtime, rtol=0, atol=ATIME_TOLERANCE)
        return np.where(matched, indices, -1)

    def _aeqdsk_on_gtime(self, shot: int, raw_data: dict, values: np.ndarray) -> np.ndarray | ak.Array:
        """
        Put AEQDSK values (first axis on ATIME) on the equilibrium time base GTIME.

        Slices without an a-file are missing (None). The result is only an option type when such slices exist.
        """
        indices = self._compose_aeqdsk_time_indices(shot, raw_data)
        values = np.asarray(values)[indices]
        if (indices >= 0).all():
            return values
        values = np.ma.masked_array(values)
        values[indices < 0] = np.ma.masked
        return ak.from_numpy(values)

    def _compose_boundary_outline_r(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compose boundary outline R coordinates with padding removal.

        Removes 0-padded values to create ragged array.

        Returns:
            Ragged awkward array (n_time, var) where each time slice has different number of points
        """
        rbbbs_key = Requirement(f'{self.geqdsk_node}.RBBBS', self.resolve_shot(shot), self.efit_tree).as_key()
        rbbbs = raw_data[rbbbs_key]

        # Filter out padding (where R==0)
        mask = ~np.isclose(rbbbs, 0)
        return filter_padding(rbbbs, mask)

    def _compose_boundary_outline_z(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compose boundary outline Z coordinates with padding removal.

        Uses R coordinates as mask - filters Z where R==0 (padding).
        This preserves valid Z=0 values (at midplane) while removing padding.

        Returns:
            Ragged awkward array (n_time, var) where each time slice has different number of points
        """
        rbbbs_key = Requirement(f'{self.geqdsk_node}.RBBBS', self.resolve_shot(shot), self.efit_tree).as_key()
        zbbbs_key = Requirement(f'{self.geqdsk_node}.ZBBBS', self.resolve_shot(shot), self.efit_tree).as_key()

        rbbbs = raw_data[rbbbs_key]
        zbbbs = raw_data[zbbbs_key]

        # Filter using R as mask: where R==0 indicates padding, not valid data
        mask = ~np.isclose(rbbbs, 0)
        return filter_padding(zbbbs, mask)

    def _compose_xpoint_r(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compose X-point R coordinates (both X-points).

        Removes 0-padded X-points. Result can have 0, 1, or 2 X-points per time.

        Returns:
            Ragged awkward array (n_time, var) where each time slice has 0-2 X-points
        """
        rxpt1_key = Requirement(f'{self.aeqdsk_node}.RXPT1', self.resolve_shot(shot), self.efit_tree).as_key()
        rxpt2_key = Requirement(f'{self.aeqdsk_node}.RXPT2', self.resolve_shot(shot), self.efit_tree).as_key()

        rxpt1 = raw_data[rxpt1_key]
        rxpt2 = raw_data[rxpt2_key]

        # Stack into (n_time, 2) array on GTIME, slices without an a-file (index -1) have no X-points
        indices = self._compose_aeqdsk_time_indices(shot, raw_data)
        xpoints = np.column_stack([rxpt1, rxpt2])[indices]

        # Filter out padding (where X-point R==0)
        mask = (xpoints > 0) & (indices >= 0)[:, None]
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
        zxpt1_key = Requirement(f'{self.aeqdsk_node}.ZXPT1', self.resolve_shot(shot),
                                 self.efit_tree).as_key()
        zxpt2_key = Requirement(f'{self.aeqdsk_node}.ZXPT2', self.resolve_shot(shot), 
                                self.efit_tree).as_key()
        rxpt1_key = Requirement(f'{self.aeqdsk_node}.RXPT1', self.resolve_shot(shot), 
                                self.efit_tree).as_key()
        rxpt2_key = Requirement(f'{self.aeqdsk_node}.RXPT2', self.resolve_shot(shot),
                                self.efit_tree).as_key()

        rxpt1 = raw_data[rxpt1_key]
        rxpt2 = raw_data[rxpt2_key]

        zxpt1 = raw_data[zxpt1_key]
        zxpt2 = raw_data[zxpt2_key]

        # Stack Z coordinates on GTIME, slices without an a-file (index -1) have no X-points
        indices = self._compose_aeqdsk_time_indices(shot, raw_data)
        xpoints_r = np.column_stack([rxpt1, rxpt2])[indices]
        xpoints_z = np.column_stack([zxpt1, zxpt2])[indices]

        mask = (xpoints_r > 0) & (indices >= 0)[:, None]
        return filter_padding(xpoints_z, mask)

    def _compose_geometric_axis_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose geometric axis R coordinate.

        All equilibrium lenghts are stored in meters in DIII-D MDSplus (unlike EFIT output files)
        """
        rsurf_key = Requirement(f'{self.aeqdsk_node}.RSURF', self.resolve_shot(shot), self.efit_tree).as_key()
        return self._aeqdsk_on_gtime(shot, raw_data, raw_data[rsurf_key])

    def _compose_geometric_axis_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose geometric axis Z coordinate.

        All equilibrium lenghts are stored in meters in DIII-D MDSplus (unlike EFIT output files)
        """
        zsurf_key = Requirement(f'{self.aeqdsk_node}.ZSURF', self.resolve_shot(shot), self.efit_tree).as_key()
        return self._aeqdsk_on_gtime(shot, raw_data, raw_data[zsurf_key])

    def _compose_closest_wall_distance(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose closest wall point distance (meters).

        OMAS: data(\\EFIT::TOP.RESULTS.AEQDSK.SEPLIM)
        """
        seplim_key = Requirement(f'{self.aeqdsk_node}.SEPLIM', self.resolve_shot(shot), self.efit_tree).as_key()
        return self._aeqdsk_on_gtime(shot, raw_data, raw_data[seplim_key])

    def _compose_gap_names(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose gap names tiled across time dimension.

        Returns (n_time, 4) array where each time slice has the same 4 names:
        ["inboard", "outboard", "top", "bottom"]

        OMAS deviation: OMAS only defines names once, but IMAS schema expects
        names per time_slice, so we tile across time dimension.
        """
        gtime_key = Requirement(f'{self.geqdsk_node}.GTIME', self.resolve_shot(shot), self.efit_tree).as_key()
        gtime_ms = raw_data[gtime_key]
        n_time = len(gtime_ms)

        # Tile gap names across time dimension
        gap_names = np.array(["inboard", "outboard", "top", "bottom"])
        return np.tile(gap_names, (n_time, 1))

    def _compose_gap_values(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose gap values for all 4 gaps (meters).

        Returns (n_time, 4) array with [inboard, outboard, top, bottom] per time.

        OMAS: data(\\EFIT::TOP.RESULTS.AEQDSK.GAPIN/GAPOUT/GAPTOP/GAPBOT)
        """
        gapin_key = Requirement(f'{self.aeqdsk_node}.GAPIN', self.resolve_shot(shot), self.efit_tree).as_key()
        gapout_key = Requirement(f'{self.aeqdsk_node}.GAPOUT', self.resolve_shot(shot), self.efit_tree).as_key()
        gaptop_key = Requirement(f'{self.aeqdsk_node}.GAPTOP', self.resolve_shot(shot), self.efit_tree).as_key()
        gapbot_key = Requirement(f'{self.aeqdsk_node}.GAPBOT', self.resolve_shot(shot), self.efit_tree).as_key()

        gapin = raw_data[gapin_key]
        gapout = raw_data[gapout_key]
        gaptop = raw_data[gaptop_key]
        gapbot = raw_data[gapbot_key]

        # Stack into (n_time, 4) array on GTIME
        return self._aeqdsk_on_gtime(shot, raw_data, np.column_stack([gapin, gapout, gaptop, gapbot]))

    def _compose_strike_point_r(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compose strike point R coordinates for all 4 strike points.

        All equilibrium lenghts are stored in meters in DIII-D MDSplus (unlike EFIT output files).

        Removes invalid strike points (R <= 0, which includes the -0.89 EFIT sentinel).
        Result can have 0-4 strike points per time.

        Returns:
            Ragged awkward array (n_time, var) where each time slice has 0-4 strike points
        """
        rvsid_key = Requirement(f'{self.aeqdsk_node}.RVSID', self.resolve_shot(shot), self.efit_tree).as_key()
        rvsod_key = Requirement(f'{self.aeqdsk_node}.RVSOD', self.resolve_shot(shot), self.efit_tree).as_key()
        rvsiu_key = Requirement(f'{self.aeqdsk_node}.RVSIU', self.resolve_shot(shot), self.efit_tree).as_key()
        rvsou_key = Requirement(f'{self.aeqdsk_node}.RVSOU', self.resolve_shot(shot), self.efit_tree).as_key()

        rvsid = raw_data[rvsid_key]
        rvsod = raw_data[rvsod_key]
        rvsiu = raw_data[rvsiu_key]
        rvsou = raw_data[rvsou_key]

        # Stack into (n_time, 4) array on GTIME and convert to meters, slices without an a-file (index -1)
        # have no strike points
        indices = self._compose_aeqdsk_time_indices(shot, raw_data)
        strike_points = np.column_stack([rvsid, rvsod, rvsiu, rvsou])[indices]

        # Filter out invalid strike points (R <= 0, which includes the -0.89 EFIT sentinel)
        mask = (strike_points > 0.0) & (indices >= 0)[:, None]
        return filter_padding(strike_points, mask)

    def _compose_strike_point_z(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compose strike point Z coordinates for all 4 strike points.

        All equilibrium lenghts are stored in meters in DIII-D MDSplus (unlike EFIT output files).

        Uses corresponding R coordinates as mask to handle Z=0 correctly.
        Result can have 0-4 strike points per time.

        Returns:
            Ragged awkward array (n_time, var) where each time slice has 0-4 strike points
        """
        rvsid_key = Requirement(f'{self.aeqdsk_node}.RVSID', self.resolve_shot(shot), self.efit_tree).as_key()
        rvsod_key = Requirement(f'{self.aeqdsk_node}.RVSOD', self.resolve_shot(shot), self.efit_tree).as_key()
        rvsiu_key = Requirement(f'{self.aeqdsk_node}.RVSIU', self.resolve_shot(shot), self.efit_tree).as_key()
        rvsou_key = Requirement(f'{self.aeqdsk_node}.RVSOU', self.resolve_shot(shot), self.efit_tree).as_key()
        zvsid_key = Requirement(f'{self.aeqdsk_node}.ZVSID', self.resolve_shot(shot), self.efit_tree).as_key()
        zvsod_key = Requirement(f'{self.aeqdsk_node}.ZVSOD', self.resolve_shot(shot), self.efit_tree).as_key()
        zvsiu_key = Requirement(f'{self.aeqdsk_node}.ZVSIU', self.resolve_shot(shot), self.efit_tree).as_key()
        zvsou_key = Requirement(f'{self.aeqdsk_node}.ZVSOU', self.resolve_shot(shot), self.efit_tree).as_key()

        rvsid = raw_data[rvsid_key]
        rvsod = raw_data[rvsod_key]
        rvsiu = raw_data[rvsiu_key]
        rvsou = raw_data[rvsou_key]
        zvsid = raw_data[zvsid_key]
        zvsod = raw_data[zvsod_key]
        zvsiu = raw_data[zvsiu_key]
        zvsou = raw_data[zvsou_key]

        # Stack Z coordinates on GTIME and convert to meters, slices without an a-file (index -1)
        # have no strike points
        indices = self._compose_aeqdsk_time_indices(shot, raw_data)
        strike_points_z = np.column_stack([zvsid, zvsod, zvsiu, zvsou])[indices]

        # Use R coordinates as mask (R <= 0 means invalid)
        strike_points_r = np.column_stack([rvsid, rvsod, rvsiu, rvsou])[indices]
        mask = (strike_points_r > 0.0) & (indices >= 0)[:, None]
        return filter_padding(strike_points_z, mask)

    def _compose_minor_radius(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose minor radius.

        All equilibrium lenghts are stored in meters in DIII-D MDSplus (unlike EFIT output files).
        """
        aminor_key = Requirement(f'{self.aeqdsk_node}.AMINOR', self.resolve_shot(shot), self.efit_tree).as_key()
        return self._aeqdsk_on_gtime(shot, raw_data, raw_data[aminor_key])

    def _compose_mse_measured(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose MSE polarisation angle measured values (apply ATAN and time filtering).

        OMAS: ATAN(data(\\EFIT::TOP.MEASUREMENTS.TANGAM))
        """
        tangam_key = Requirement(f'{self.measurements_node}.TANGAM', self.resolve_shot(shot), self.efit_tree).as_key()
        tangam_all_times = raw_data[tangam_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        tangam = tangam_all_times[time_indices]

        return np.arctan(tangam)

    def _compose_mse_error(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose MSE polarisation angle error values (apply ATAN and time filtering).

        OMAS: ATAN(data(\\EFIT::TOP.MEASUREMENTS.SIGGAM))
        """
        siggam_key = Requirement(f'{self.measurements_node}.SIGGAM', self.resolve_shot(shot), self.efit_tree).as_key()
        siggam_all_times = raw_data[siggam_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        siggam = siggam_all_times[time_indices]

        return np.arctan(siggam)

    def _compose_pf_current_measured(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose PF current measured values using stack_outer_2 and time filtering.

        OMAS: py2tdi(stack_outer_2,'\\EFIT::TOP.MEASUREMENTS.ECCURT','\\EFIT::TOP.MEASUREMENTS.FCCURT')

        Stacks ECCURT and FCCURT along outer dimension.
        """
        eccurt_key = Requirement(f'{self.measurements_node}.ECCURT', self.resolve_shot(shot), self.efit_tree).as_key()
        fccurt_key = Requirement(f'{self.measurements_node}.FCCURT', self.resolve_shot(shot), self.efit_tree).as_key()
        eccurt_all_times = raw_data[eccurt_key]
        fccurt_all_times = raw_data[fccurt_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        eccurt = eccurt_all_times[time_indices]
        fccurt = fccurt_all_times[time_indices]

        # Stack along the outer dimension (concatenate arrays)
        # If eccurt is (n_time, n_ec) and fccurt is (n_time, n_fc), result is (n_time, n_ec + n_fc)
        return np.concatenate([eccurt, fccurt], axis=1)

    def _compose_pf_current_error(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose PF current error values using stack_outer_2 and time filtering.

        OMAS: py2tdi(stack_outer_2,'\\EFIT::TOP.MEASUREMENTS.SIGECC','\\EFIT::TOP.MEASUREMENTS.SIGFCC')
        """
        sigecc_key = Requirement(f'{self.measurements_node}.SIGECC', self.resolve_shot(shot), self.efit_tree).as_key()
        sigfcc_key = Requirement(f'{self.measurements_node}.SIGFCC', self.resolve_shot(shot), self.efit_tree).as_key()
        sigecc_all_times = raw_data[sigecc_key]
        sigfcc_all_times = raw_data[sigfcc_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        sigecc = sigecc_all_times[time_indices]
        sigfcc = sigfcc_all_times[time_indices]

        return np.concatenate([sigecc, sigfcc], axis=1)

    def _compose_pf_current_weight(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose PF current weight values using stack_outer_2 and time filtering.

        OMAS: py2tdi(stack_outer_2,'\\EFIT::TOP.MEASUREMENTS.FWTEC','\\EFIT::TOP.MEASUREMENTS.FWTFC')
        """
        fwtec_key = Requirement(f'{self.measurements_node}.FWTEC', self.resolve_shot(shot), self.efit_tree).as_key()
        fwtfc_key = Requirement(f'{self.measurements_node}.FWTFC', self.resolve_shot(shot), self.efit_tree).as_key()
        fwtec_all_times = raw_data[fwtec_key]
        fwtfc_all_times = raw_data[fwtfc_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        fwtec = fwtec_all_times[time_indices]
        fwtfc = fwtfc_all_times[time_indices]

        return np.concatenate([fwtec, fwtfc], axis=1)

    def _compose_pf_current_reconstructed(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose PF current reconstructed values using stack_outer_2 and time filtering.

        OMAS: py2tdi(stack_outer_2,'\\EFIT::TOP.MEASUREMENTS.CECURR','\\EFIT::TOP.MEASUREMENTS.CCBRSP')
        """
        cecurr_key = Requirement(f'{self.measurements_node}.CECURR', self.resolve_shot(shot), self.efit_tree).as_key()
        ccbrsp_key = Requirement(f'{self.measurements_node}.CCBRSP', self.resolve_shot(shot), self.efit_tree).as_key()
        cecurr_all_times = raw_data[cecurr_key]
        ccbrsp_all_times = raw_data[ccbrsp_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        cecurr = cecurr_all_times[time_indices]
        ccbrsp = ccbrsp_all_times[time_indices]

        return np.concatenate([cecurr, ccbrsp], axis=1)

    def _compose_pf_current_chi_squared(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose PF current chi squared values using stack_outer_2 and time filtering.

        OMAS: py2tdi(stack_outer_2,'\\EFIT::TOP.MEASUREMENTS.CHIECC','\\EFIT::TOP.MEASUREMENTS.CHIFCC')
        """
        chiecc_key = Requirement(f'{self.measurements_node}.CHIECC', self.resolve_shot(shot), self.efit_tree).as_key()
        chifcc_key = Requirement(f'{self.measurements_node}.CHIFCC', self.resolve_shot(shot), self.efit_tree).as_key()
        chiecc_all_times = raw_data[chiecc_key]
        chifcc_all_times = raw_data[chifcc_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        chiecc = chiecc_all_times[time_indices]
        chifcc = chifcc_all_times[time_indices]

        return np.concatenate([chiecc, chifcc], axis=1)

    def _compose_psi(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose boundary separatrix psi with COCOS transformation.

        OMAS: data(\\EFIT::TOP.RESULTS.GEQDSK.SSIBRY)
        Transform: PSI (requires COCOS conversion)
        """
        ssibry_key = Requirement(f'{self.geqdsk_node}.SSIBRY', self.resolve_shot(shot), self.efit_tree).as_key()
        psi = raw_data[ssibry_key]
        return self._apply_cocos_transform(psi, shot, raw_data, "equilibrium.time_slice.boundary_separatrix.psi")

    def _compose_ip_measured(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose measured plasma current with COCOS transformation and time filtering.

        OMAS: data(\\EFIT::TOP.MEASUREMENTS.PLASMA)
        Transform: TOR (requires COCOS conversion)

        Time filtering: MEASUREMENTS have different time dimension (MTIME) than
        equilibrium time base (GTIME). Apply index mapping to align with GTIME.
        """
        plasma_key = Requirement(f'{self.measurements_node}.PLASMA', self.resolve_shot(shot), self.efit_tree).as_key()
        ip_all_times = raw_data[plasma_key]

        # Apply time filtering: get indices mapping GTIME to MTIME
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        ip = ip_all_times[time_indices]

        return self._apply_cocos_transform(ip, shot, raw_data, "equilibrium.time_slice.constraints.ip.measured")

    def _compose_ip_reconstructed(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose reconstructed plasma current with COCOS transformation and time filtering.

        OMAS: data(\\EFIT::TOP.MEASUREMENTS.CPASMA)
        Transform: TOR (requires COCOS conversion)
        """
        cpasma_key = Requirement(f'{self.measurements_node}.CPASMA', self.resolve_shot(shot), self.efit_tree).as_key()
        ip_all_times = raw_data[cpasma_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        ip = ip_all_times[time_indices]

        return self._apply_cocos_transform(ip, shot, raw_data, "equilibrium.time_slice.constraints.ip.reconstructed")

    def _compose_flux_loop_measured(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose measured flux loop data with COCOS transformation and time filtering.

        OMAS: data(\\EFIT::TOP.MEASUREMENTS.SILOPT)
        Transform: PSI (magnetic flux requires COCOS conversion, factor of 2π for COCOS 1→11)
        """
        silopt_key = Requirement(f'{self.measurements_node}.SILOPT', self.resolve_shot(shot), self.efit_tree).as_key()
        flux_all_times = raw_data[silopt_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        flux = flux_all_times[time_indices]

        return self._apply_cocos_transform(flux, shot, raw_data, "equilibrium.time_slice.constraints.flux_loop.measured")

    def _compose_flux_loop_measured_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose flux loop measurement error with COCOS transformation and time filtering.

        OMAS: data(\\EFIT::TOP.MEASUREMENTS.SIGSIL)
        Transform: PSI (error on magnetic flux requires same COCOS conversion as flux)
        """
        sigsil_key = Requirement(f'{self.measurements_node}.SIGSIL', self.resolve_shot(shot), self.efit_tree).as_key()
        error_all_times = raw_data[sigsil_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        error = error_all_times[time_indices]

        return self._apply_cocos_transform(error, shot, raw_data,
                                           "equilibrium.time_slice.constraints.flux_loop.measured_error_upper", no_sign=True)

    def _compose_flux_loop_reconstructed(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose reconstructed flux loop data with COCOS transformation and time filtering.

        OMAS: data(\\EFIT::TOP.MEASUREMENTS.CSILOP)
        Transform: PSI (magnetic flux requires COCOS conversion, factor of 2π for COCOS 1→11)
        """
        csilop_key = Requirement(f'{self.measurements_node}.CSILOP', self.resolve_shot(shot), self.efit_tree).as_key()
        flux_all_times = raw_data[csilop_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        flux = flux_all_times[time_indices]

        return self._apply_cocos_transform(flux, shot, raw_data, "equilibrium.time_slice.constraints.flux_loop.reconstructed")

    def _compose_pressure_position_psi(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose pressure position psi using efit_psi_to_real_psi_2d transformation with COCOS and time filtering.

        OMAS: py2tdi(efit_psi_to_real_psi_2d,'-\\EFIT::TOP.MEASUREMENTS.RPRESS',
                     '\\EFIT::TOP.RESULTS.GEQDSK.SSIMAG','\\EFIT::TOP.RESULTS.GEQDSK.SSIBRY')
        Transform: (-RPRESS.T * (SSIBRY - SSIMAG) + SSIMAG).T with COCOS PSI conversion
        """
        rpress_key = Requirement(f'{self.measurements_node}.RPRESS', self.resolve_shot(shot), self.efit_tree).as_key()
        ssimag_key = Requirement(f'{self.geqdsk_node}.SSIMAG', self.resolve_shot(shot), self.efit_tree).as_key()
        ssibry_key = Requirement(f'{self.geqdsk_node}.SSIBRY', self.resolve_shot(shot), self.efit_tree).as_key()

        rpress_all_times = -raw_data[rpress_key]
        ssimag = raw_data[ssimag_key]
        ssibry = raw_data[ssibry_key]

        # Apply time filtering to MEASUREMENTS data
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        rpress = rpress_all_times[time_indices]

        # Convert normalized psi to real psi, then apply COCOS transform
        if len(np.shape(rpress)) < 2:
            rpress_2d = np.atleast_2d(rpress).T
        else:
            rpress_2d = rpress
        psi = (rpress_2d.T * (ssibry - ssimag) + ssimag).T
        return self._apply_cocos_transform(psi, shot, raw_data, "equilibrium.time_slice.constraints.pressure.position.psi")

    def _compose_pressure_measured(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose measured pressure with ensure_2d transformation and time filtering.

        OMAS: py2tdi(ensure_2d,'\\EFIT::TOP.MEASUREMENTS.PRESSR')
        Transform: ensure_2d (adds dimension if 1D)
        """
        pressr_key = Requirement(f'{self.measurements_node}.PRESSR', self.resolve_shot(shot), self.efit_tree).as_key()
        pressure_all_times = raw_data[pressr_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        pressure = pressure_all_times[time_indices]

        if len(np.shape(pressure)) < 2:
            pressure_2d = np.atleast_2d(pressure).T
        else:
            pressure_2d = pressure

        return pressure_2d

    def _compose_pressure_error(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose pressure measurement error with ensure_2d transformation and time filtering.

        OMAS: py2tdi(ensure_2d,'\\EFIT::TOP.MEASUREMENTS.SIGPRE')
        Transform: ensure_2d (adds dimension if 1D)
        """
        sigpre_key = Requirement(f'{self.measurements_node}.SIGPRE', self.resolve_shot(shot), self.efit_tree).as_key()
        error_all_times = raw_data[sigpre_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        error = error_all_times[time_indices]

        if len(np.shape(error)) < 2:
            error_2d = np.atleast_2d(error).T
        else:
            error_2d = error

        return error_2d

    def _compose_pressure_weight(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose pressure weight with ensure_2d transformation and time filtering.

        OMAS: py2tdi(ensure_2d,'\\EFIT::TOP.MEASUREMENTS.FWTPRE')
        Transform: ensure_2d (adds dimension if 1D)
        """
        fwtpre_key = Requirement(f'{self.measurements_node}.FWTPRE', self.resolve_shot(shot), self.efit_tree).as_key()
        weight_all_times = raw_data[fwtpre_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        weight = weight_all_times[time_indices]

        if len(np.shape(weight)) < 2:
            weight_2d = np.atleast_2d(weight).T
        else:
            weight_2d = weight

        return weight_2d

    def _compose_pressure_reconstructed(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose reconstructed pressure with ensure_2d transformation and time filtering.

        OMAS: py2tdi(ensure_2d,'\\EFIT::TOP.MEASUREMENTS.CPRESS')
        Transform: ensure_2d (adds dimension if 1D)
        """
        cpress_key = Requirement(f'{self.measurements_node}.CPRESS', self.resolve_shot(shot), self.efit_tree).as_key()
        pressure_all_times = raw_data[cpress_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        pressure = pressure_all_times[time_indices]

        if len(np.shape(pressure)) < 2:
            pressure_2d = np.atleast_2d(pressure).T
        else:
            pressure_2d = pressure

        return pressure_2d

    def _compose_pressure_chi_squared(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose pressure chi squared with ensure_2d, negative sign, and time filtering.

        OMAS: py2tdi(ensure_2d,'-\\EFIT::TOP.MEASUREMENTS.SAIPRE')
        Transform: ensure_2d + negate
        """
        saipre_key = Requirement(f'{self.measurements_node}.SAIPRE', self.resolve_shot(shot), self.efit_tree).as_key()
        chi_sq_all_times = raw_data[saipre_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        chi_sq = chi_sq_all_times[time_indices]

        if len(np.shape(chi_sq)) < 2:
            chi_sq_2d = np.atleast_2d(chi_sq).T
        else:
            chi_sq_2d = chi_sq

        return -chi_sq_2d

    def _compose_j_tor_position_psi(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose j_tor position psi using efit_psi_to_real_psi_2d transformation with COCOS and time filtering.

        OMAS: py2tdi(efit_psi_to_real_psi_2d,'\\EFIT::TOP.MEASUREMENTS.SIZEROJ',
                     '\\EFIT::TOP.RESULTS.GEQDSK.SSIMAG','\\EFIT::TOP.RESULTS.GEQDSK.SSIBRY')
        Transform: (SIZEROJ.T * (SSIBRY - SSIMAG) + SSIMAG).T with COCOS PSI conversion
        """
        sizeroj_key = Requirement(f'{self.measurements_node}.SIZEROJ', self.resolve_shot(shot), self.efit_tree).as_key()
        ssimag_key = Requirement(f'{self.geqdsk_node}.SSIMAG', self.resolve_shot(shot), self.efit_tree).as_key()
        ssibry_key = Requirement(f'{self.geqdsk_node}.SSIBRY', self.resolve_shot(shot), self.efit_tree).as_key()

        sizeroj_all_times = raw_data[sizeroj_key]
        ssimag = raw_data[ssimag_key]
        ssibry = raw_data[ssibry_key]

        # Apply time filtering to MEASUREMENTS data
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        sizeroj = sizeroj_all_times[time_indices]

        # Convert normalized psi to real psi, then apply COCOS transform
        if len(np.shape(sizeroj)) < 2:
            sizeroj_2d = np.atleast_2d(sizeroj).T
        else:
            sizeroj_2d = sizeroj
        psi = (abs(sizeroj_2d).T * (ssibry - ssimag) + ssimag).T
        return self._apply_cocos_transform(psi, shot, raw_data, "equilibrium.time_slice.constraints.j_tor.position.psi")

    def _compose_j_tor_measured(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose j_tor measured values with ensure_2d transformation and time filtering.

        OMAS: py2tdi(ensure_2d,'\\EFIT::TOP.MEASUREMENTS.VZEROJ')
        Transform: ensure_2d (convert to 2D array)

        VZEROJ is stored as a magnitude without the plasma-current direction, so it is
        multiplied by sign(Ip) (a single per-shot sign from mean GEQDSK.CPASMA, matching
        the sign of global_quantities.ip) before the COCOS transform.
        """
        vzeroj_key = Requirement(f'{self.measurements_node}.VZEROJ', self.resolve_shot(shot), self.efit_tree).as_key()
        j_tor_all_times = raw_data[vzeroj_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        j_tor = j_tor_all_times[time_indices]

        if len(np.shape(j_tor)) < 2:
            j_tor_2d = np.atleast_2d(j_tor).T
        else:
            j_tor_2d = j_tor

        # Restore the plasma-current direction onto the unsigned VZEROJ magnitude
        cpasma_key = Requirement(f'{self.geqdsk_node}.CPASMA', self.resolve_shot(shot), self.efit_tree).as_key()
        ip_sign = np.sign(np.mean(raw_data[cpasma_key]))
        j_tor_2d = j_tor_2d * ip_sign

        return self._apply_cocos_transform(
            j_tor_2d, shot, raw_data,
            "equilibrium.time_slice.constraints.j_tor.measured",
        )

    # Simple constraint compose functions with time filtering (converted from lambdas)

    def _compose_ip_measured_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose IP measured error with time filtering."""
        sigpasma_key = Requirement(f'{self.measurements_node}.SIGPASMA', self.resolve_shot(shot), self.efit_tree).as_key()
        error_all_times = raw_data[sigpasma_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return error_all_times[time_indices]

    def _compose_ip_weight(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose IP weight with time filtering."""
        fwtpasma_key = Requirement(f'{self.measurements_node}.FWTPASMA', self.resolve_shot(shot), self.efit_tree).as_key()
        weight_all_times = raw_data[fwtpasma_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return weight_all_times[time_indices]

    def _compose_ip_chi_squared(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose IP chi squared with time filtering."""
        chipasma_key = Requirement(f'{self.measurements_node}.CHIPASMA', self.resolve_shot(shot), self.efit_tree).as_key()
        chi_sq_all_times = raw_data[chipasma_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return chi_sq_all_times[time_indices]

    def _compose_bpol_probe_measured(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose bpol probe measured with time filtering."""
        expmpi_key = Requirement(f'{self.measurements_node}.EXPMPI', self.resolve_shot(shot), self.efit_tree).as_key()
        measured_all_times = raw_data[expmpi_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return measured_all_times[time_indices]

    def _compose_bpol_probe_measured_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose bpol probe measured error with time filtering."""
        sigmpi_key = Requirement(f'{self.measurements_node}.SIGMPI', self.resolve_shot(shot), self.efit_tree).as_key()
        error_all_times = raw_data[sigmpi_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return error_all_times[time_indices]

    def _compose_bpol_probe_weight(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose bpol probe weight with time filtering."""
        fwtmp2_key = Requirement(f'{self.measurements_node}.FWTMP2', self.resolve_shot(shot), self.efit_tree).as_key()
        weight_all_times = raw_data[fwtmp2_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return weight_all_times[time_indices]

    def _compose_bpol_probe_reconstructed(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose bpol probe reconstructed with time filtering."""
        cmpr2_key = Requirement(f'{self.measurements_node}.CMPR2', self.resolve_shot(shot), self.efit_tree).as_key()
        reconstructed_all_times = raw_data[cmpr2_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return reconstructed_all_times[time_indices]

    def _compose_bpol_probe_chi_squared(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose bpol probe chi squared with time filtering."""
        saimpi_key = Requirement(f'{self.measurements_node}.SAIMPI', self.resolve_shot(shot), self.efit_tree).as_key()
        chi_sq_all_times = raw_data[saimpi_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return chi_sq_all_times[time_indices]

    def _compose_diamagnetic_flux_measured(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose diamagnetic flux measured with time filtering."""
        diamag_key = Requirement(f'{self.measurements_node}.DIAMAG', self.resolve_shot(shot), self.efit_tree).as_key()
        measured_all_times = raw_data[diamag_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return measured_all_times[time_indices]

    def _compose_diamagnetic_flux_measured_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose diamagnetic flux measured error with time filtering."""
        sigdia_key = Requirement(f'{self.measurements_node}.SIGDIA', self.resolve_shot(shot), self.efit_tree).as_key()
        error_all_times = raw_data[sigdia_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return error_all_times[time_indices]

    def _compose_diamagnetic_flux_weight(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose diamagnetic flux weight with time filtering."""
        fwtdia_key = Requirement(f'{self.measurements_node}.FWTDIA', self.resolve_shot(shot), self.efit_tree).as_key()
        weight_all_times = raw_data[fwtdia_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return weight_all_times[time_indices]

    def _compose_diamagnetic_flux_reconstructed(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose diamagnetic flux reconstructed with time filtering."""
        cdflux_key = Requirement(f'{self.measurements_node}.CDFLUX', self.resolve_shot(shot), self.efit_tree).as_key()
        reconstructed_all_times = raw_data[cdflux_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return reconstructed_all_times[time_indices]

    def _compose_diamagnetic_flux_chi_squared(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose diamagnetic flux chi squared with time filtering."""
        chidflux_key = Requirement(f'{self.measurements_node}.CHIDFLUX', self.resolve_shot(shot), self.efit_tree).as_key()
        chi_sq_all_times = raw_data[chidflux_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return chi_sq_all_times[time_indices]

    def _compose_flux_loop_weight(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose flux loop weight with time filtering."""
        fwtsi_key = Requirement(f'{self.measurements_node}.FWTSI', self.resolve_shot(shot), self.efit_tree).as_key()
        weight_all_times = raw_data[fwtsi_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return weight_all_times[time_indices]

    def _compose_flux_loop_chi_squared(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose flux loop chi squared with time filtering."""
        saisil_key = Requirement(f'{self.measurements_node}.SAISIL', self.resolve_shot(shot), self.efit_tree).as_key()
        chi_sq_all_times = raw_data[saisil_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return chi_sq_all_times[time_indices]

    def _compose_mse_weight(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose MSE weight with time filtering."""
        fwtgam_key = Requirement(f'{self.measurements_node}.FWTGAM', self.resolve_shot(shot), self.efit_tree).as_key()
        weight_all_times = raw_data[fwtgam_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return weight_all_times[time_indices]

    def _compose_mse_reconstructed(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose MSE reconstructed with time filtering."""
        cmgam_key = Requirement(f'{self.measurements_node}.CMGAM', self.resolve_shot(shot), self.efit_tree).as_key()
        reconstructed_all_times = raw_data[cmgam_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return reconstructed_all_times[time_indices]

    def _compose_mse_chi_squared(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose MSE chi squared with time filtering."""
        chigam_key = Requirement(f'{self.measurements_node}.CHIGAM', self.resolve_shot(shot), self.efit_tree).as_key()
        chi_sq_all_times = raw_data[chigam_key]
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        return chi_sq_all_times[time_indices]

    def _compose_global_ip(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose global plasma current with COCOS TOR transformation.

        OMAS: data(\\EFIT::TOP.RESULTS.GEQDSK.CPASMA)
        Transform: COCOS TOR (toroidal current)
        """
        cpasma_key = Requirement(f'{self.geqdsk_node}.CPASMA', self.resolve_shot(shot), self.efit_tree).as_key()
        ip = raw_data[cpasma_key]
        return self._apply_cocos_transform(ip, shot, raw_data, "equilibrium.time_slice.global_quantities.ip")

    def _compose_global_psi_axis(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose psi at magnetic axis with COCOS PSI transformation.

        OMAS: data(\\EFIT::TOP.RESULTS.GEQDSK.SSIMAG)
        Transform: COCOS PSI (magnetic flux)
        """
        ssimag_key = Requirement(f'{self.geqdsk_node}.SSIMAG', self.resolve_shot(shot), self.efit_tree).as_key()
        psi_axis = raw_data[ssimag_key]
        return self._apply_cocos_transform(psi_axis, shot, raw_data, "equilibrium.time_slice.global_quantities.psi_axis")

    def _compose_global_psi_boundary(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose psi at boundary with COCOS PSI transformation.

        OMAS: data(\\EFIT::TOP.RESULTS.GEQDSK.SSIBRY)
        Transform: COCOS PSI (magnetic flux)
        """
        ssibry_key = Requirement(f'{self.geqdsk_node}.SSIBRY', self.resolve_shot(shot), self.efit_tree).as_key()
        psi_boundary = raw_data[ssibry_key]
        return self._apply_cocos_transform(psi_boundary, shot, raw_data, "equilibrium.time_slice.global_quantities.psi_boundary")

    def _compose_global_area(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose plasma cross-sectional area.

        OMAS: data(\\EFIT::TOP.RESULTS.AEQDSK.AREA)
        """
        area_key = Requirement(f'{self.aeqdsk_node}.AREA', self.resolve_shot(shot), self.efit_tree).as_key()
        return self._aeqdsk_on_gtime(shot, raw_data, raw_data[area_key])

    def _compose_global_li_3(self, shot: int, raw_data: dict) -> np.ndarray:
        """Trivial pass-through for li_3."""
        li3_key = Requirement(f'{self.aeqdsk_node}.LI3', self.resolve_shot(shot), self.efit_tree).as_key()
        return self._aeqdsk_on_gtime(shot, raw_data, raw_data[li3_key])

    def _compose_global_magnetic_axis_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """Trivial pass-through for magnetic_axis.r."""
        rmaxis_key = Requirement(f'{self.geqdsk_node}.RMAXIS', self.resolve_shot(shot), self.efit_tree).as_key()
        return raw_data[rmaxis_key]

    def _compose_global_magnetic_axis_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """Trivial pass-through for magnetic_axis.z."""
        zmaxis_key = Requirement(f'{self.geqdsk_node}.ZMAXIS', self.resolve_shot(shot), self.efit_tree).as_key()
        return raw_data[zmaxis_key]

    def _compose_global_magnetic_axis_b_field_tor(self, shot: int, raw_data: dict) -> np.ndarray:
        """Trivial pass-through for magnetic_axis.b_field_tor."""
        bt0_key = Requirement(f'{self.aeqdsk_node}.BT0', self.resolve_shot(shot), self.efit_tree).as_key()
        return self._aeqdsk_on_gtime(shot, raw_data, raw_data[bt0_key])

    def _compose_global_surface(self, shot: int, raw_data: dict) -> np.ndarray:
        """Trivial pass-through for surface."""
        psurfa_key = Requirement(f'{self.aeqdsk_node}.PSURFA', self.resolve_shot(shot), self.efit_tree).as_key()
        return self._aeqdsk_on_gtime(shot, raw_data, raw_data[psurfa_key])

    def _compose_global_volume(self, shot: int, raw_data: dict) -> np.ndarray:
        """Trivial pass-through for volume."""
        volume_key = Requirement(f'{self.aeqdsk_node}.VOLUME', self.resolve_shot(shot), self.efit_tree).as_key()
        return self._aeqdsk_on_gtime(shot, raw_data, raw_data[volume_key])

    def _compose_global_beta_pol(self, shot: int, raw_data: dict) -> np.ndarray:
        """Trivial pass-through for beta_pol."""
        betap_key = Requirement(f'{self.aeqdsk_node}.BETAP', self.resolve_shot(shot), self.efit_tree).as_key()
        return self._aeqdsk_on_gtime(shot, raw_data, raw_data[betap_key])

    def _compose_global_beta_tor(self, shot: int, raw_data: dict) -> np.ndarray:
        """Trivial pass-through for beta_tor."""
        betat_key = Requirement(f'{self.aeqdsk_node}.BETAT', self.resolve_shot(shot), self.efit_tree).as_key()
        return self._aeqdsk_on_gtime(shot, raw_data, raw_data[betat_key])

    def _compose_global_beta_normal(self, shot: int, raw_data: dict) -> np.ndarray:
        """Trivial pass-through for beta_normal."""
        betan_key = Requirement(f'{self.aeqdsk_node}.BETAN', self.resolve_shot(shot), self.efit_tree).as_key()
        return self._aeqdsk_on_gtime(shot, raw_data, raw_data[betan_key])

    def _compose_global_q_95(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose q_95 with COCOS Q transformation.

        OMAS: data(\\EFIT::TOP.RESULTS.AEQDSK.Q95)
        Transform: COCOS Q (safety factor)
        """
        q95_key = Requirement(f'{self.aeqdsk_node}.Q95', self.resolve_shot(shot), self.efit_tree).as_key()
        q95 = self._aeqdsk_on_gtime(shot, raw_data, raw_data[q95_key])
        return self._apply_cocos_transform(q95, shot, raw_data, "equilibrium.time_slice.global_quantities.q_95")

    def _compose_global_q_axis(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose q_axis with COCOS Q transformation.

        OMAS: data(\\EFIT::TOP.RESULTS.AEQDSK.Q0)
        Transform: COCOS Q (safety factor)
        """
        q0_key = Requirement(f'{self.aeqdsk_node}.Q0', self.resolve_shot(shot), self.efit_tree).as_key()
        q_axis = self._aeqdsk_on_gtime(shot, raw_data, raw_data[q0_key])
        return self._apply_cocos_transform(q_axis, shot, raw_data, "equilibrium.time_slice.global_quantities.q_axis")

    def _compose_global_q_min_value(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose q_min.value with COCOS Q transformation.

        OMAS: data(\\EFIT::TOP.RESULTS.AEQDSK.QMIN)
        Transform: COCOS Q (safety factor)
        """
        qmin_key = Requirement(f'{self.aeqdsk_node}.QMIN', self.resolve_shot(shot), self.efit_tree).as_key()
        q_min = self._aeqdsk_on_gtime(shot, raw_data, raw_data[qmin_key])
        return self._apply_cocos_transform(q_min, shot, raw_data, "equilibrium.time_slice.global_quantities.q_min.value")

    def _compose_profiles_1d_dpressure_dpsi(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose dpressure_dpsi with COCOS dPSI transformation.

        OMAS: data(\\EFIT::TOP.RESULTS.GEQDSK.PPRIME)
        Transform: COCOS dPSI (inverse PSI, 1/(2π) factor)
        """
        pprime_key = Requirement(f'{self.geqdsk_node}.PPRIME', self.resolve_shot(shot), self.efit_tree).as_key()
        dpressure_dpsi = raw_data[pprime_key]
        return self._apply_cocos_transform(dpressure_dpsi, shot, raw_data, "equilibrium.time_slice.profiles_1d.dpressure_dpsi")

    def _compose_profiles_1d_f(self, shot: int, raw_data: dict) -> np.ndarray:
        """Trivial pass-through for f (poloidal current function)."""
        fpol_key = Requirement(f'{self.geqdsk_node}.FPOL', self.resolve_shot(shot), self.efit_tree).as_key()
        return raw_data[fpol_key]

    def _compose_profiles_1d_f_df_dpsi(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose f_df_dpsi with COCOS dPSI transformation.

        OMAS: data(\\EFIT::TOP.RESULTS.GEQDSK.FFPRIM)
        Transform: COCOS dPSI (inverse PSI, 1/(2π) factor)
        """
        ffprim_key = Requirement(f'{self.geqdsk_node}.FFPRIM', self.resolve_shot(shot), self.efit_tree).as_key()
        f_df_dpsi = raw_data[ffprim_key]
        return self._apply_cocos_transform(f_df_dpsi, shot, raw_data, "equilibrium.time_slice.profiles_1d.f_df_dpsi")

    def _compose_profiles_1d_pressure(self, shot: int, raw_data: dict) -> np.ndarray:
        """Trivial pass-through for pressure profile."""
        pres_key = Requirement(f'{self.geqdsk_node}.PRES', self.resolve_shot(shot), self.efit_tree).as_key()
        return raw_data[pres_key]

    def _compose_profiles_1d_psi(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose psi profile using geqdsk_psi transformation with COCOS.

        OMAS: py2tdi(geqdsk_psi,'\\EFIT::TOP.RESULTS.GEQDSK.SSIMAG','\\EFIT::TOP.RESULTS.GEQDSK.SSIBRY','\\EFIT::TOP.RESULTS.GEQDSK.PSIN')
        Transform: PSIN * (SSIBRY - SSIMAG) + SSIMAG, then apply COCOS PSI
        """
        psin_key = Requirement(f'{self.geqdsk_node}.PSIN', self.resolve_shot(shot), self.efit_tree).as_key()
        ssimag_key = Requirement(f'{self.geqdsk_node}.SSIMAG', self.resolve_shot(shot), self.efit_tree).as_key()
        ssibry_key = Requirement(f'{self.geqdsk_node}.SSIBRY', self.resolve_shot(shot), self.efit_tree).as_key()

        psin = raw_data[psin_key]
        ssimag = raw_data[ssimag_key]
        ssibry = raw_data[ssibry_key]

        # Convert normalized psi to real psi and fix the shape of psi which is constant in time
        psi = psin[None,:] * (ssibry[:,None] - ssimag[:,None]) + ssimag[:,None]
        return self._apply_cocos_transform(psi, shot, raw_data, "equilibrium.time_slice.profiles_1d.psi")

    def _compose_profiles_1d_q(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose q profile with COCOS Q transformation.

        OMAS: data(\\EFIT::TOP.RESULTS.GEQDSK.QPSI)
        Transform: COCOS Q (safety factor)
        """
        qpsi_key = Requirement(f'{self.geqdsk_node}.QPSI', self.resolve_shot(shot), self.efit_tree).as_key()
        q = raw_data[qpsi_key]
        return self._apply_cocos_transform(q, shot, raw_data, "equilibrium.time_slice.profiles_1d.q")

    def _compose_profiles_1d_rho_tor_norm(self, shot: int, raw_data: dict) -> np.ndarray:
        """Trivial pass-through for rho_tor_norm."""
        rhovn_key = Requirement(f'{self.geqdsk_node}.RHOVN', self.resolve_shot(shot), self.efit_tree).as_key()
        return raw_data[rhovn_key]

    def _interpolate_fluxfun(self, shot: int, raw_data: dict, node: str) -> np.ndarray:
        """
        Interpolate FLUXFUN.<node> from the FLUXFUN psi grid onto the GEQDSK.PSIN grid.

        OMAS: py2tdi(interpolate_psi_1d,'\\EFIT::TOP.RESULTS.FLUXFUN.PSI','\\EFIT::TOP.RESULTS.FLUXFUN.<node>',
                     '\\EFIT::TOP.RESULTS.GEQDSK.SSIMAG','\\EFIT::TOP.RESULTS.GEQDSK.SSIBRY','\\EFIT::TOP.RESULTS.GEQDSK.PSIN')
        Slices whose FLUXFUN.PSI has repeated values (failed fits) cannot be interpolated and stay zero.
        """
        fluxfun_psi_key = Requirement(f'\\{self.efit_tree}::TOP.RESULTS.FLUXFUN.PSI', self.resolve_shot(shot), self.efit_tree).as_key()
        fluxfun_key = Requirement(f'\\{self.efit_tree}::TOP.RESULTS.FLUXFUN.{node}', self.resolve_shot(shot), self.efit_tree).as_key()
        ssimag_key = Requirement(f'{self.geqdsk_node}.SSIMAG', self.resolve_shot(shot), self.efit_tree).as_key()
        ssibry_key = Requirement(f'{self.geqdsk_node}.SSIBRY', self.resolve_shot(shot), self.efit_tree).as_key()
        psin_key = Requirement(f'{self.geqdsk_node}.PSIN', self.resolve_shot(shot), self.efit_tree).as_key()
        # Fluxfun quantities come as (radial direction, time), but we need (time, radial direction/psi)
        fluxfun_psi = raw_data[fluxfun_psi_key].T
        fluxfun_values = raw_data[fluxfun_key].T
        ssimag = raw_data[ssimag_key]
        ssibry = raw_data[ssibry_key]
        psin = raw_data[psin_key]

        # Normalize FLUXFUN.PSI
        fluxfun_psi_norm = (fluxfun_psi - ssimag[:, None]) / (ssibry[:, None] - ssimag[:, None])

        # Interpolate from FLUXFUN grid to GEQDSK grid
        values = np.zeros(ssimag.shape + psin.shape)
        for i in range(fluxfun_psi.shape[0]):
            # Use constant extrapolation: first and last values for out-of-bounds
            fill_value = (fluxfun_values[i][0], fluxfun_values[i][-1])
            if np.all(np.unique(fluxfun_psi_norm[i], return_counts=True)[1] == 1):
                values[i] = interp1d(fluxfun_psi_norm[i], fluxfun_values[i], kind='cubic',
                                     bounds_error=False, fill_value=fill_value)(psin)
        return values

    def _compose_profiles_1d_j_tor(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose j_tor profile using interpolate_psi_1d transformation.

        OMAS: py2tdi(interpolate_psi_1d,'\\EFIT::TOP.RESULTS.FLUXFUN.PSI','\\EFIT::TOP.RESULTS.FLUXFUN.JEFF',
                     '\\EFIT::TOP.RESULTS.GEQDSK.SSIMAG','\\EFIT::TOP.RESULTS.GEQDSK.SSIBRY','\\EFIT::TOP.RESULTS.GEQDSK.PSIN')
        """
        return self._interpolate_fluxfun(shot, raw_data, 'JEFF')

    def _compose_profiles_1d_j_parallel(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose j_parallel profile using interpolate_psi_1d transformation.

        OMAS: py2tdi(interpolate_psi_1d,'\\EFIT::TOP.RESULTS.FLUXFUN.PSI','\\EFIT::TOP.RESULTS.FLUXFUN.JLL',
                     '\\EFIT::TOP.RESULTS.GEQDSK.SSIMAG','\\EFIT::TOP.RESULTS.GEQDSK.SSIBRY','\\EFIT::TOP.RESULTS.GEQDSK.PSIN')
        """
        return self._interpolate_fluxfun(shot, raw_data, 'JLL')

    def _compose_profiles_1d_volume(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose volume profile using interpolate_psi_1d transformation.

        OMAS: py2tdi(interpolate_psi_1d,'\\EFIT::TOP.RESULTS.FLUXFUN.PSI','\\EFIT::TOP.RESULTS.FLUXFUN.VOL',
                     '\\EFIT::TOP.RESULTS.GEQDSK.SSIMAG','\\EFIT::TOP.RESULTS.GEQDSK.SSIBRY','\\EFIT::TOP.RESULTS.GEQDSK.PSIN')
        """
        return self._interpolate_fluxfun(shot, raw_data, 'VOL')

    def _compose_profiles_1d_trapped_fraction(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose trapped_fraction as 1 - FLUXFUN.FC (circulating fraction) on the GEQDSK.PSIN grid.

        Not in OMAS. Matches OMFIT fluxSurfaces (1 - avg['fc']). No particle is trapped on the axis.
        """
        trapped_fraction = 1.0 - self._interpolate_fluxfun(shot, raw_data, 'FC')
        trapped_fraction[:, 0] = 0.0
        return trapped_fraction

    def _abs_b_on_axis(self, shot: int, raw_data: dict) -> np.ndarray:
        """|B| on the magnetic axis, where B_pol = 0, so |B| = |AEQDSK.BT0|. NaN for slices without an a-file."""
        return np.abs(self._aeqdsk_profile_end(shot, raw_data, 'BT0'))

    def _compose_profiles_1d_gm5(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose gm5 = <B^2> from FLUXFUN.B2AVE on the GEQDSK.PSIN grid, with BT0^2 on the axis.

        Not in OMAS. Matches OMFIT fluxSurfaces avg['Btot**2'].
        """
        gm5 = self._interpolate_fluxfun(shot, raw_data, 'B2AVE')
        gm5[:, 0] = self._abs_b_on_axis(shot, raw_data) ** 2
        return gm5

    def _compose_profiles_1d_b_field_average(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose b_field_average = <|B|> from FLUXFUN.BAVE on the GEQDSK.PSIN grid, with |BT0| on the axis.

        Not in OMAS. Matches OMFIT fluxSurfaces avg['Btot'] (OMFIT's own b_field_average is the plain mean
        over the contour points, not a flux-surface average).
        """
        b_field_average = self._interpolate_fluxfun(shot, raw_data, 'BAVE')
        b_field_average[:, 0] = self._abs_b_on_axis(shot, raw_data)
        return b_field_average

    def _compose_profiles_1d_psi_norm(self, shot: int, raw_data: dict) -> np.ndarray:
        """GEQDSK.PSIN repeated on every time slice."""
        psin_key = Requirement(f'{self.geqdsk_node}.PSIN', self.resolve_shot(shot), self.efit_tree).as_key()
        gtime_key = Requirement(f'{self.geqdsk_node}.GTIME', self.resolve_shot(shot), self.efit_tree).as_key()
        return np.tile(raw_data[psin_key], (len(raw_data[gtime_key]), 1))

    def _profiles_1d_flux_surface_extrema(self, shot: int, raw_data: dict) -> Dict[str, np.ndarray]:
        """Extrema of the flux surfaces traced in GEQDSK.PSIRZ at the PSIN levels between axis and boundary."""
        def get(node):
            return raw_data[Requirement(f'{self.geqdsk_node}.{node}', self.resolve_shot(shot), self.efit_tree).as_key()]

        ssimag = get('SSIMAG')
        ssibry = get('SSIBRY')
        psi_levels = ssimag[:, None] + get('PSIN')[None, 1:-1] * (ssibry - ssimag)[:, None]
        return flux_surface_extrema(get('R'), get('Z'), get('PSIRZ'), psi_levels, get('RMAXIS'), get('ZMAXIS'))

    def _aeqdsk_profile_end(self, shot: int, raw_data: dict, node: str) -> np.ndarray:
        """
        AEQDSK.<node> on GTIME as the axis or boundary column of a profile.

        Unlike _aeqdsk_on_gtime, slices without an a-file are NaN rather than missing.
        """
        key = Requirement(f'{self.aeqdsk_node}.{node}', self.resolve_shot(shot), self.efit_tree).as_key()
        indices = self._compose_aeqdsk_time_indices(shot, raw_data)
        return np.where(indices >= 0, np.asarray(raw_data[key], dtype=float)[indices], np.nan)

    def _compose_profiles_1d_elongation(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose elongation (zmax - zmin) / (rmax - rmin) of each flux surface.

        Not in OMAS. Interior surfaces match OMFIT fluxSurfaces (0.5 * (kapu + kapl)). The axis is AEQDSK.KAPPA0
        and the boundary AEQDSK.KAPPA, so the last point equals boundary_separatrix.elongation.
        """
        ext = self._profiles_1d_flux_surface_extrema(shot, raw_data)
        interior = (ext['z_max'] - ext['z_min']) / (ext['r_max'] - ext['r_min'])
        return np.column_stack([self._aeqdsk_profile_end(shot, raw_data, 'KAPPA0'), interior,
                                self._aeqdsk_profile_end(shot, raw_data, 'KAPPA')])

    def _compose_profiles_1d_triangularity_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose upper triangularity (R_geo - R(zmax)) / a of each flux surface.

        Not in OMAS. Interior surfaces match OMFIT fluxSurfaces (delu). Zero on the axis and AEQDSK.TRITOP at the
        boundary, so the last point equals boundary_separatrix.triangularity_upper.
        """
        ext = self._profiles_1d_flux_surface_extrema(shot, raw_data)
        interior = (ext['r_max'] + ext['r_min'] - 2 * ext['r_at_z_max']) / (ext['r_max'] - ext['r_min'])
        return np.column_stack([np.zeros(len(interior)), interior,
                                self._aeqdsk_profile_end(shot, raw_data, 'TRITOP')])

    def _compose_profiles_1d_triangularity_lower(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose lower triangularity (R_geo - R(zmin)) / a of each flux surface.

        Not in OMAS. Interior surfaces match OMFIT fluxSurfaces (dell). Zero on the axis and AEQDSK.TRIBOT at the
        boundary, so the last point equals boundary_separatrix.triangularity_lower.
        """
        ext = self._profiles_1d_flux_surface_extrema(shot, raw_data)
        interior = (ext['r_max'] + ext['r_min'] - 2 * ext['r_at_z_min']) / (ext['r_max'] - ext['r_min'])
        return np.column_stack([np.zeros(len(interior)), interior,
                                self._aeqdsk_profile_end(shot, raw_data, 'TRIBOT')])

    def _compose_profiles_1d_r_inboard(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose r_inboard, the smallest major radius of each flux surface.

        Not in OMAS. RMAXIS on the axis and AEQDSK.RSURF - AEQDSK.AMINOR at the boundary.
        """
        ext = self._profiles_1d_flux_surface_extrema(shot, raw_data)
        boundary = self._aeqdsk_profile_end(shot, raw_data, 'RSURF') - self._aeqdsk_profile_end(shot, raw_data, 'AMINOR')
        return np.column_stack([self._compose_global_magnetic_axis_r(shot, raw_data), ext['r_min'], boundary])

    def _compose_profiles_1d_r_outboard(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose r_outboard, the largest major radius of each flux surface.

        Not in OMAS. RMAXIS on the axis and AEQDSK.RSURF + AEQDSK.AMINOR at the boundary.
        """
        ext = self._profiles_1d_flux_surface_extrema(shot, raw_data)
        boundary = self._aeqdsk_profile_end(shot, raw_data, 'RSURF') + self._aeqdsk_profile_end(shot, raw_data, 'AMINOR')
        return np.column_stack([self._compose_global_magnetic_axis_r(shot, raw_data), ext['r_max'], boundary])

    def _compose_profiles_1d_geometric_axis_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose geometric_axis.r = (rmin + rmax) / 2 of each flux surface.

        Not in OMAS. RMAXIS on the axis and AEQDSK.RSURF at the boundary (= boundary_separatrix.geometric_axis.r).
        """
        ext = self._profiles_1d_flux_surface_extrema(shot, raw_data)
        return np.column_stack([self._compose_global_magnetic_axis_r(shot, raw_data), 0.5 * (ext['r_max'] + ext['r_min']),
                                self._aeqdsk_profile_end(shot, raw_data, 'RSURF')])

    def _compose_profiles_1d_geometric_axis_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose geometric_axis.z = (zmin + zmax) / 2 of each flux surface.

        Not in OMAS. ZMAXIS on the axis and AEQDSK.ZSURF at the boundary (= boundary_separatrix.geometric_axis.z).
        """
        ext = self._profiles_1d_flux_surface_extrema(shot, raw_data)
        return np.column_stack([self._compose_global_magnetic_axis_z(shot, raw_data), 0.5 * (ext['z_max'] + ext['z_min']),
                                self._aeqdsk_profile_end(shot, raw_data, 'ZSURF')])

    def _compose_profiles_2d_grid_dim1(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose profiles_2d R grid using tile transformation.

        OMAS: py2tdi(tile,'\\EFIT::TOP.RESULTS.GEQDSK.R','size(\\EFIT::TOP.RESULTS.GEQDSK.BCENTR)')
        Transform: Tile R grid array to match time dimension
        TRANSPOSE: [1,0,2] - swap time and radial axes
        """
        r_grid_key = Requirement(f'{self.geqdsk_node}.R', self.resolve_shot(shot), self.efit_tree).as_key()
        bcentr_key = Requirement(f'{self.geqdsk_node}.BCENTR', self.resolve_shot(shot), self.efit_tree).as_key()

        r_grid = raw_data[r_grid_key]
        bcentr = raw_data[bcentr_key]
        n_time = len(bcentr)

        # Tile R grid: create (n_time, n_r) array
        # OMAS: np.array([r_grid for k in range(n_time)])
        # Then transpose [1,0,2] swaps time and radial dimensions
        tiled = np.array([r_grid for _ in range(n_time)])
        # TRANSPOSE [1,0,2]: (time, r) -> (r, time) for 2D case, but this is actually (time, r, z) -> (r, time, z)
        # Add grid dimension
        return tiled[:, None, :]

    def _compose_profiles_2d_grid_dim2(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose profiles_2d Z grid using tile transformation.

        OMAS: py2tdi(tile,'\\EFIT::TOP.RESULTS.GEQDSK.Z','size(\\EFIT::TOP.RESULTS.GEQDSK.BCENTR)')
        Transform: Tile Z grid array to match time dimension
        TRANSPOSE: [1,0,2] - swap time and vertical axes
        """
        z_grid_key = Requirement(f'{self.geqdsk_node}.Z', self.resolve_shot(shot), self.efit_tree).as_key()
        bcentr_key = Requirement(f'{self.geqdsk_node}.BCENTR', self.resolve_shot(shot), self.efit_tree).as_key()

        z_grid = raw_data[z_grid_key]
        bcentr = raw_data[bcentr_key]
        n_time = len(bcentr)

        # Tile Z grid: create (n_time, n_z) array then transpose
        tiled = np.array([z_grid for _ in range(n_time)])
        # Add grid dimension
        return tiled[:, None, :]

    def _compose_profiles_2d_grid_type_index(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose grid_type.index - constant value of 1 for rectangular grid.

        OMAS: py2tdi(tile,1,'size(\\EFIT::TOP.RESULTS.GEQDSK.BCENTR)')
        Transform: Create array of 1s matching time dimension
        TRANSPOSE: [1,0] - but result is 1D so no transpose needed
        """
        bcentr_key = Requirement(f'{self.geqdsk_node}.BCENTR', self.resolve_shot(shot), self.efit_tree).as_key()
        bcentr = raw_data[bcentr_key]
        n_time = len(bcentr)

        # Return array of [1]s for each time slice and grid type 1 = rectangular R-Z grid
        return np.ones(n_time, dtype=int)[:,None]

    def _compose_profiles_2d_type_index(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose type.index - constant value of 0 for data type.

        Transform: Create array of 0s matching time dimension with extra grid-axes dimension
        Shape: (n_time, 1) where 1 is the grid-axes dimension for profiles_2d
        """
        bcentr_key = Requirement(f'{self.geqdsk_node}.BCENTR', self.resolve_shot(shot), self.efit_tree).as_key()
        bcentr = raw_data[bcentr_key]
        n_time = len(bcentr)

        # Return array of [0]s for each time slice with extra dimension for grid-axes
        # Shape: (n_time, 1) - always 0 indicating the data type
        return np.zeros(n_time, dtype=int)[:,None]

    def _compose_profiles_2d_psi(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose profiles_2d PSI with COCOS transformation.

        OMAS: [data(\\EFIT::TOP.RESULTS.GEQDSK.PSIRZ)]
        TRANSPOSE: [1, 0, 3, 2] - swap axes for (time, r, z) ordering
        """
        psirz_key = Requirement(f'{self.geqdsk_node}.PSIRZ', self.resolve_shot(shot), self.efit_tree).as_key()
        # Add grid dimension
        psirz = np.swapaxes(raw_data[psirz_key], 1,2)[:,None,:]

        # Apply COCOS transformation
        return self._apply_cocos_transform(psirz, shot, raw_data, "equilibrium.time_slice.profiles_2d.psi")

    def _compose_profiles_2d_b_field_tor(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose toroidal magnetic field on profiles_2d grid.

        Based on OMAS derive_equilibrium_profiles_2d_quantity for b_field_tor.
        Uses f(psi) interpolated onto the 2D grid and divided by R.

        Algorithm:
        1. Get psi on 2D grid and profiles_1d.psi, profiles_1d.f
        2. For regions within psi range, interpolate f(psi)
        3. For regions outside, use f[-1] (boundary value)
        4. Compute b_field_tor = f(psi) / R

        Note: Uses numpy.gradient for broadcasting over time axis.
        """
        # Get grid coordinates
        r_grid_key = Requirement(f'{self.geqdsk_node}.R', self.resolve_shot(shot), self.efit_tree).as_key()
        z_grid_key = Requirement(f'{self.geqdsk_node}.Z', self.resolve_shot(shot), self.efit_tree).as_key()
        bcentr_key = Requirement(f'{self.geqdsk_node}.BCENTR', self.resolve_shot(shot), self.efit_tree).as_key()

        r_grid = raw_data[r_grid_key]
        z_grid = raw_data[z_grid_key]
        bcentr = raw_data[bcentr_key]
        n_time = len(bcentr)

        # Create R meshgrid: shape (n_time, 1, n_r, n_z)
        # Need to broadcast over time, then add grid dimension
        r_mesh = np.tile(r_grid[None, None, :, None], (n_time, 1, 1, len(z_grid)))

        # Get psi on 2D grid (already has COCOS applied)
        psi_2d = self._compose_profiles_2d_psi(shot, raw_data)  # shape: (n_time, 1, n_r, n_z)

        # Get profiles_1d.psi and profiles_1d.f (both have COCOS applied where needed)
        psi_1d = self._compose_profiles_1d_psi(shot, raw_data)  # shape: (n_time, n_psi)
        f_1d = self._compose_profiles_1d_f(shot, raw_data)      # shape: (n_time, n_psi)

        # Initialize result
        b_field_tor = np.zeros_like(psi_2d)

        # Process each time slice
        for i_time in range(n_time):
            psi_2d_t = psi_2d[i_time, 0, :, :]  # shape: (n_r, n_z)
            psi_1d_t = psi_1d[i_time, :]
            f_1d_t = f_1d[i_time, :]
            r_mesh_t = r_mesh[i_time, 0, :, :]

            # Create mask for points within psi range
            psi_min = np.min(psi_1d_t)
            psi_max = np.max(psi_1d_t)
            mask = np.logical_and(psi_2d_t >= psi_min, psi_2d_t <= psi_max)

            # Initialize with boundary value for outside regions
            b_field_tor[i_time, 0, :, :] = f_1d_t[-1] / r_mesh_t

            # Interpolate f(psi) for inside regions
            if np.any(mask):
                from scipy.interpolate import interp1d
                f_interp = interp1d(psi_1d_t, f_1d_t, kind='linear',
                                   bounds_error=False, fill_value=f_1d_t[-1])
                b_field_tor[i_time, 0, :, :][mask] = f_interp(psi_2d_t[mask]) / r_mesh_t[mask]

        return b_field_tor

    def _compose_profiles_2d_b_field_r(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose radial magnetic field on profiles_2d grid.

        Based on OMAS derive_equilibrium_profiles_2d_quantity for b_field_r.
        Uses gradient of psi with respect to Z divided by R.

        Formula: B_r = (1/R) * dPsi/dZ * sigma_RpZ * sigma_Bp / (2*pi)^exp_Bp

        Note: Uses numpy.gradient for broadcasting over time axis instead of
        RectBivariateSpline to avoid iteration over time slices.
        """
        # Get grid coordinates
        r_grid_key = Requirement(f'{self.geqdsk_node}.R', self.resolve_shot(shot), self.efit_tree).as_key()
        z_grid_key = Requirement(f'{self.geqdsk_node}.Z', self.resolve_shot(shot), self.efit_tree).as_key()
        bcentr_key = Requirement(f'{self.geqdsk_node}.BCENTR', self.resolve_shot(shot), self.efit_tree).as_key()

        r_grid = raw_data[r_grid_key]
        z_grid = raw_data[z_grid_key]
        bcentr = raw_data[bcentr_key]
        n_time = len(bcentr)

        # Get grid spacing
        dr = r_grid[1] - r_grid[0]
        dz = z_grid[1] - z_grid[0]

        # Create R meshgrid: shape (n_time, 1, n_r, n_z)
        r_mesh = np.tile(r_grid[None, None, :, None], (n_time, 1, 1, len(z_grid)))

        # Get psi on 2D grid (already has COCOS 11 applied)
        psi_2d = self._compose_profiles_2d_psi(shot, raw_data)  # shape: (n_time, 1, n_r, n_z)

        # Compute gradient: axis 2 is R, axis 3 is Z
        # numpy.gradient returns [d/daxis0, d/daxis1, d/daxis2, d/daxis3]
        # We want d/dZ (axis 3) and d/dR (axis 2)
        gradients = np.gradient(psi_2d, dr, dz, axis=(2, 3))
        dPSIdZ = gradients[1]  # derivative with respect to Z

        # Apply COCOS 11 formula: B_r = dPsi/dZ / (2*pi * R)
        # Since psi is already in COCOS 11, and COCOS 11 has:
        # sigma_RpZ = +1, sigma_Bp = +1, exp_Bp = +1
        # The formula simplifies to: dPsi/dZ / (2*pi * R)
        b_field_r = dPSIdZ / (2.0 * np.pi * r_mesh)

        return b_field_r

    def _compose_profiles_2d_b_field_z(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose vertical magnetic field on profiles_2d grid.

        Based on OMAS derive_equilibrium_profiles_2d_quantity for b_field_z.
        Uses negative gradient of psi with respect to R divided by R.

        Formula: B_z = -(1/R) * dPsi/dR * sigma_RpZ * sigma_Bp / (2*pi)^exp_Bp

        Note: Uses numpy.gradient for broadcasting over time axis instead of
        RectBivariateSpline to avoid iteration over time slices.
        """
        # Get grid coordinates
        r_grid_key = Requirement(f'{self.geqdsk_node}.R', self.resolve_shot(shot), self.efit_tree).as_key()
        z_grid_key = Requirement(f'{self.geqdsk_node}.Z', self.resolve_shot(shot), self.efit_tree).as_key()
        bcentr_key = Requirement(f'{self.geqdsk_node}.BCENTR', self.resolve_shot(shot), self.efit_tree).as_key()

        r_grid = raw_data[r_grid_key]
        z_grid = raw_data[z_grid_key]
        bcentr = raw_data[bcentr_key]
        n_time = len(bcentr)

        # Get grid spacing
        dr = r_grid[1] - r_grid[0]
        dz = z_grid[1] - z_grid[0]

        # Create R meshgrid: shape (n_time, 1, n_r, n_z)
        r_mesh = np.tile(r_grid[None, None, :, None], (n_time, 1, 1, len(z_grid)))

        # Get psi on 2D grid (already has COCOS 11 applied)
        psi_2d = self._compose_profiles_2d_psi(shot, raw_data)  # shape: (n_time, 1, n_r, n_z)

        # Compute gradient: axis 2 is R, axis 3 is Z
        gradients = np.gradient(psi_2d, dr, dz, axis=(2, 3))
        dPSIdR = gradients[0]  # derivative with respect to R

        # Apply COCOS 11 formula: B_z = -dPsi/dR / (2*pi * R)
        # Since psi is already in COCOS 11, and COCOS 11 has:
        # sigma_RpZ = +1, sigma_Bp = +1, exp_Bp = +1
        # The formula simplifies to: -dPsi/dR / (2*pi * R)
        b_field_z = -dPSIdR / (2.0 * np.pi * r_mesh)

        return b_field_z

    def _compose_profiles_2d_j_tor(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose the toroidal plasma current density on the profiles_2d grid.

        EFIT does not store its current (pcurrt) in MDSplus, so it is recovered from the flux with the
        5-point finite difference Grad-Shafranov operator whose inverse EFIT uses for the interior flux
        (cyclic.F90) and which it evaluates forward only as a plotted diagnostic (shapesurf.F90, "delstar"):

            Delta* psi = d2psi/dR2 - (1/R) dpsi/dR + d2psi/dZ2 = 2*pi * mu0 * R * j_tor

        with psi in COCOS 11 (Sauter & Medvedev 2013, eqs. 7 and 9 with sigma_Bp = +1, exp_Bp = 1).
        This is the current implied by the stored flux, not EFIT's pcurrt itself. The two agree up to
        the float32 storage of PSIRZ if EFIT's last flux update was the finite difference solve
        (ibunmn=1); if it was the direct Green's function sum (ibunmn=0, or ibunmn=2 once converged
        below errcut), they differ by the O(h^2) truncation of the stencil. Neither is R p' + FF'/(mu0 R)
        on the final flux: EFIT evaluates its current on the previous iterate, so that differs by EFIT's
        convergence error.

        The current is kept where EFIT lets it flow (update_parameters.F90, weight): 0 <= psi_n <= 1
        inside the bounding box of the boundary outline, which also excludes the private flux region.
        Outside, Delta* psi only holds the truncation error of the vacuum field and conductors inside the
        grid. EFIT took this mask from the previous iterate as well, which can only matter on the ring of
        nodes at psi_n = 1. The grid edge, EFIT's Dirichlet boundary, is 0.

        Returns:
            (n_time, 1, n_r, n_z) toroidal current density [A/m^2]
        """
        r_grid_key = Requirement(f'{self.geqdsk_node}.R', self.resolve_shot(shot), self.efit_tree).as_key()
        z_grid_key = Requirement(f'{self.geqdsk_node}.Z', self.resolve_shot(shot), self.efit_tree).as_key()
        psirz_key = Requirement(f'{self.geqdsk_node}.PSIRZ', self.resolve_shot(shot), self.efit_tree).as_key()
        ssimag_key = Requirement(f'{self.geqdsk_node}.SSIMAG', self.resolve_shot(shot), self.efit_tree).as_key()
        ssibry_key = Requirement(f'{self.geqdsk_node}.SSIBRY', self.resolve_shot(shot), self.efit_tree).as_key()
        rbbbs_key = Requirement(f'{self.geqdsk_node}.RBBBS', self.resolve_shot(shot), self.efit_tree).as_key()
        zbbbs_key = Requirement(f'{self.geqdsk_node}.ZBBBS', self.resolve_shot(shot), self.efit_tree).as_key()

        r_grid = raw_data[r_grid_key]
        z_grid = raw_data[z_grid_key]
        dr = r_grid[1] - r_grid[0]
        dz = z_grid[1] - z_grid[0]
        r_inner = r_grid[None, None, 1:-1, None]

        # psi on 2D grid (already has COCOS 11 applied), shape: (n_time, 1, n_r, n_z)
        psi_2d = self._compose_profiles_2d_psi(shot, raw_data).astype(np.float64)
        center = psi_2d[:, :, 1:-1, 1:-1]
        r_plus = psi_2d[:, :, 2:, 1:-1]
        r_minus = psi_2d[:, :, :-2, 1:-1]
        z_plus = psi_2d[:, :, 1:-1, 2:]
        z_minus = psi_2d[:, :, 1:-1, :-2]
        delta_star = (
            (r_plus - 2 * center + r_minus) / dr**2
            - (r_plus - r_minus) / (2 * dr * r_inner)
            + (z_plus - 2 * center + z_minus) / dz**2
        )
        j_tor = np.zeros_like(psi_2d)
        j_tor[:, :, 1:-1, 1:-1] = delta_star / (2 * np.pi * mu_0 * r_inner)

        # EFIT's mask; psi_n is COCOS invariant, so it is taken on the raw g-file flux
        psirz = np.swapaxes(raw_data[psirz_key], 1, 2)[:, None]
        ssimag = raw_data[ssimag_key][:, None, None, None]
        ssibry = raw_data[ssibry_key][:, None, None, None]
        psi_n = (psirz - ssimag) / (ssibry - ssimag)

        # RBBBS == 0 is padding, as in _compose_boundary_outline_r
        rbbbs = raw_data[rbbbs_key]
        valid = rbbbs != 0
        r_bbbs = np.where(valid, rbbbs, np.nan)
        z_bbbs = np.where(valid, raw_data[zbbbs_key], np.nan)
        in_r = (r_grid[None, :] >= np.nanmin(r_bbbs, axis=1)[:, None]) & (
            r_grid[None, :] <= np.nanmax(r_bbbs, axis=1)[:, None])
        in_z = (z_grid[None, :] >= np.nanmin(z_bbbs, axis=1)[:, None]) & (
            z_grid[None, :] <= np.nanmax(z_bbbs, axis=1)[:, None])
        in_box = in_r[:, None, :, None] & in_z[:, None, None, :]

        return np.where(in_box & (psi_n >= 0) & (psi_n <= 1), j_tor, 0.0)

    def _compose_vacuum_b0(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose vacuum toroidal field at reference radius (trivial pass-through).

        OMAS: data(\\EFIT::TOP.RESULTS.GEQDSK.BCENTR)
        """
        bcentr_key = Requirement(f'{self.geqdsk_node}.BCENTR', self.resolve_shot(shot), self.efit_tree).as_key()
        return raw_data[bcentr_key]

    def _compose_vacuum_r0(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose vacuum toroidal field reference radius (first element of RZERO).

        OMAS: data(\\EFIT::TOP.RESULTS.GEQDSK.RZERO)[0]
        """
        rzero_key = Requirement(f'{self.geqdsk_node}.RZERO', self.resolve_shot(shot), self.efit_tree).as_key()
        rzero = raw_data[rzero_key]
        # RZERO is scalar per time slice, but OMAS takes [0] suggesting it might be an array
        # If it's already scalar, just return it; if array, take first element
        if isinstance(rzero, np.ndarray) and rzero.ndim > 0:
            return rzero if rzero.shape == () else rzero[0] if rzero.ndim == 1 else rzero[:, 0]
        return rzero

    def _compose_convergence_iterations_n(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose number of iterations by finding the largest non-zero iteration index per time slice.

        OMAS: py2tdi(get_largest_axis_value,'dim_of(\\EFIT::TOP.MEASUREMENTS.CERROR,0)','\\EFIT::TOP.MEASUREMENTS.CERROR')
        Transform: Find maximum non-zero iteration index per time slice

        Time filtering: CERROR is from MEASUREMENTS which has different time dimension (MTIME)
        than equilibrium time base (GTIME). Apply index mapping to align with GTIME.
        """
        cerror_key = Requirement(f'\\{self.efit_tree}::TOP.MEASUREMENTS.CERROR', self.resolve_shot(shot), self.efit_tree).as_key()
        cerror_all_times = raw_data[cerror_key]

        # Apply time filtering
        time_indices = self._compose_constraint_time_indices(shot, raw_data)
        cerror = cerror_all_times[time_indices]

        # CERROR shape: (n_time, n_iter)
        # For each time slice, find the index of the last non-zero error value

        # Create indices array: same shape as cerror, filled with iteration indices
        n_time, n_iter = cerror.shape
        indices = np.arange(n_iter)[np.newaxis, :] * np.ones((n_time, 1), dtype=int)

        # Set indices to -1 where cerror is zero, so they won't be selected as max
        indices = np.where(cerror != 0, indices, -1)

        # Find the maximum index (last non-zero iteration) along iteration axis
        iterations_n = np.max(indices, axis=1)

        # Replace -1 (no non-zero values) with 0
        iterations_n = np.maximum(iterations_n, 0)

        return iterations_n

    def _compose_convergence_grad_shafranov_deviation_value(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose Grad-Shafranov deviation (trivial pass-through).

        OMAS: data(\\EFIT::TOP.RESULTS.AEQDSK.ERROR)
        """
        error_key = Requirement(f'{self.aeqdsk_node}.ERROR', self.resolve_shot(shot), self.efit_tree).as_key()
        return self._aeqdsk_on_gtime(shot, raw_data, raw_data[error_key])

    def _compose_convergence_grad_shafranov_deviation_expression_index(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose Grad-Shafranov deviation expression index (constant value 3).

        OMAS: EVAL 3
        Expression index 3 indicates a specific formulation of the GS equation residual.
        """
        gtime_key = Requirement(f'{self.geqdsk_node}.GTIME', self.resolve_shot(shot), self.efit_tree).as_key()

        # Return array of 3s for each time slice
        return np.full(len(raw_data[gtime_key]), 3, dtype=int)

    # COCOS transformation methods

    def _get_cocos_for_shot(self, shot: int, raw_data: dict) -> int:
        """
        Identify COCOS convention for a given shot.

        Uses Bt and Ip signs from GEQDSK to determine COCOS, following OMAS logic.
        Results are cached per shot.

        Args:
            shot: Shot number
            raw_data: Dictionary containing fetched MDSplus data

        Returns:
            COCOS number (1, 3, 5, or 7 typically)

        Note:
            This implements the same logic as OMAS MDS_gEQDSK_COCOS_identify
        """
        # Get Bt and Ip from GEQDSK data (mean over time, like OMAS)
        bcentr_key = Requirement(f'{self.geqdsk_node}.BCENTR', self.resolve_shot(shot), self.efit_tree).as_key()
        cpasma_key = Requirement(f'{self.geqdsk_node}.CPASMA', self.resolve_shot(shot), self.efit_tree).as_key()

        # Delegate to the shared helper (with per-shot caching)
        return identify_cocos_from_signals(
            raw_data[bcentr_key], raw_data[cpasma_key],
            cocos=self.cocos, cache=self._cocos_cache, cache_key=shot,
        )

    def _apply_cocos_transform(self, data: np.ndarray, shot: int, raw_data: dict,
                               ids_path: str, no_sign:bool = False) -> np.ndarray:
        """
        Apply COCOS transformation to data if needed.

        Constraints (equilibrium.time_slice.constraints.*) always transform from
        COCOS 7, since the constraint measurements are not identified from the
        Bt/Ip signs like the rest of the equilibrium IDS.

        Args:
            data: Data to transform
            shot: Shot number
            raw_data: Dictionary containing fetched MDSplus data (for COCOS identification)
            ids_path: IDS path to determine transformation type

        Returns:
            Transformed data (or original if no transform needed)
        """
        transform_type = get_cocos_transform_type(ids_path)
        if transform_type is None:
            return data

        if ids_path.startswith("equilibrium.time_slice.constraints.") and "position" not in ids_path:
            return self.cocos.transform(data, self.CONSTRAINTS_SOURCE_COCOS, transform_type, no_sign)

        bcentr_key = Requirement(f'{self.geqdsk_node}.BCENTR', self.resolve_shot(shot), self.efit_tree).as_key()
        cpasma_key = Requirement(f'{self.geqdsk_node}.CPASMA', self.resolve_shot(shot), self.efit_tree).as_key()
        return apply_cocos_transform(
            data, raw_data[bcentr_key], raw_data[cpasma_key], ids_path,
            cocos=self.cocos, cache=self._cocos_cache, cache_key=shot, no_sign=no_sign,
        )
