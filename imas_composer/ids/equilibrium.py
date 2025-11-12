"""
Equilibrium IDS Mapping for DIII-D

Based on OMAS: omas/machine_mappings/_efit.json

Implements EFIT equilibrium reconstruction data mapping.
"""

from typing import Dict, Any, Optional
import numpy as np
import awkward as ak

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper
from ..cocos import COCOSTransform, get_cocos_transform_type


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
        self.measurements_node = f'\\{efit_tree}::TOP.MEASUREMENTS'

        # COCOS transformer
        self.cocos = COCOSTransform()
        self._cocos_cache: Dict[int, int] = {}  # shot -> cocos mapping

        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Build IDS specs
        self._build_specs()

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
            depends_on=["equilibrium._tritop"],
            compose=lambda shot, raw: raw[Requirement(f'{self.aeqdsk_node}.TRITOP', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._tribot"],
            compose=lambda shot, raw: raw[Requirement(f'{self.aeqdsk_node}.TRIBOT', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._kappa"],
            compose=lambda shot, raw: raw[Requirement(f'{self.aeqdsk_node}.KAPPA', shot, self.efit_tree).as_key()],
            ids_path="equilibrium.time_slice.boundary_separatrix.elongation",
            docs_file=self.DOCS_PATH
        )

        # Internal dependency: AMINOR (minor radius, needs cm to m conversion)
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
            depends_on=["equilibrium._aminor"],
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
            depends_on=["equilibrium._plasma", "equilibrium._bcentr", "equilibrium._cpasma_cocos"],
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
            depends_on=["equilibrium._sigpasma"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.SIGPASMA', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._fwtpasma"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.FWTPASMA', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._cpasma", "equilibrium._bcentr", "equilibrium._cpasma_cocos"],
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
            depends_on=["equilibrium._chipasma"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.CHIPASMA', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._expmpi"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.EXPMPI', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._sigmpi"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.SIGMPI', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._fwtmp2"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.FWTMP2', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._cmpr2"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.CMPR2', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._saimpi"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.SAIMPI', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._diamag"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.DIAMAG', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._sigdia"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.SIGDIA', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._fwtdia"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.FWTDIA', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._cdflux"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.CDFLUX', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._chidflux"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.CHIDFLUX', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._silopt", "equilibrium._bcentr", "equilibrium._cpasma_cocos"],
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
            depends_on=["equilibrium._sigsil", "equilibrium._bcentr", "equilibrium._cpasma_cocos"],
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
            depends_on=["equilibrium._fwtsi"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.FWTSI', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._csilop", "equilibrium._bcentr", "equilibrium._cpasma_cocos"],
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
            depends_on=["equilibrium._saisil"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.SAISIL', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._tangam"],
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
            depends_on=["equilibrium._siggam"],
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
            depends_on=["equilibrium._fwtgam"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.FWTGAM', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._cmgam"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.CMGAM', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._chigam"],
            compose=lambda shot, raw: raw[Requirement(f'{self.measurements_node}.CHIGAM', shot, self.efit_tree).as_key()],
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
            depends_on=["equilibrium._eccurt", "equilibrium._fccurt"],
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
            depends_on=["equilibrium._sigecc", "equilibrium._sigfcc"],
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
            depends_on=["equilibrium._fwtec", "equilibrium._fwtfc"],
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
            depends_on=["equilibrium._cecurr", "equilibrium._ccbrsp"],
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
            depends_on=["equilibrium._chiecc", "equilibrium._chifcc"],
            compose=self._compose_pf_current_chi_squared,
            ids_path="equilibrium.time_slice.constraints.pf_current.chi_squared",
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

    def _compose_minor_radius(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose minor radius (convert cm to meters).

        OMAS: \\EFIT::TOP.RESULTS.AEQDSK.AMINOR / 100.
        """
        aminor_key = Requirement(f'{self.aeqdsk_node}.AMINOR', shot, self.efit_tree).as_key()
        aminor_cm = raw_data[aminor_key]
        return aminor_cm / 100.0  # Convert cm to meters

    def _compose_mse_measured(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose MSE polarisation angle measured values (apply ATAN).

        OMAS: ATAN(data(\\EFIT::TOP.MEASUREMENTS.TANGAM))
        """
        tangam_key = Requirement(f'{self.measurements_node}.TANGAM', shot, self.efit_tree).as_key()
        tangam = raw_data[tangam_key]
        return np.arctan(tangam)

    def _compose_mse_error(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose MSE polarisation angle error values (apply ATAN).

        OMAS: ATAN(data(\\EFIT::TOP.MEASUREMENTS.SIGGAM))
        """
        siggam_key = Requirement(f'{self.measurements_node}.SIGGAM', shot, self.efit_tree).as_key()
        siggam = raw_data[siggam_key]
        return np.arctan(siggam)

    def _compose_pf_current_measured(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose PF current measured values using stack_outer_2.

        OMAS: py2tdi(stack_outer_2,'\\EFIT::TOP.MEASUREMENTS.ECCURT','\\EFIT::TOP.MEASUREMENTS.FCCURT')

        Stacks ECCURT and FCCURT along outer dimension.
        """
        eccurt_key = Requirement(f'{self.measurements_node}.ECCURT', shot, self.efit_tree).as_key()
        fccurt_key = Requirement(f'{self.measurements_node}.FCCURT', shot, self.efit_tree).as_key()
        eccurt = raw_data[eccurt_key]
        fccurt = raw_data[fccurt_key]

        # Stack along the outer dimension (concatenate arrays)
        # If eccurt is (n_time, n_ec) and fccurt is (n_time, n_fc), result is (n_time, n_ec + n_fc)
        return np.concatenate([eccurt, fccurt], axis=1)

    def _compose_pf_current_error(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose PF current error values using stack_outer_2.

        OMAS: py2tdi(stack_outer_2,'\\EFIT::TOP.MEASUREMENTS.SIGECC','\\EFIT::TOP.MEASUREMENTS.SIGFCC')
        """
        sigecc_key = Requirement(f'{self.measurements_node}.SIGECC', shot, self.efit_tree).as_key()
        sigfcc_key = Requirement(f'{self.measurements_node}.SIGFCC', shot, self.efit_tree).as_key()
        sigecc = raw_data[sigecc_key]
        sigfcc = raw_data[sigfcc_key]
        return np.concatenate([sigecc, sigfcc], axis=1)

    def _compose_pf_current_weight(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose PF current weight values using stack_outer_2.

        OMAS: py2tdi(stack_outer_2,'\\EFIT::TOP.MEASUREMENTS.FWTEC','\\EFIT::TOP.MEASUREMENTS.FWTFC')
        """
        fwtec_key = Requirement(f'{self.measurements_node}.FWTEC', shot, self.efit_tree).as_key()
        fwtfc_key = Requirement(f'{self.measurements_node}.FWTFC', shot, self.efit_tree).as_key()
        fwtec = raw_data[fwtec_key]
        fwtfc = raw_data[fwtfc_key]
        return np.concatenate([fwtec, fwtfc], axis=1)

    def _compose_pf_current_reconstructed(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose PF current reconstructed values using stack_outer_2.

        OMAS: py2tdi(stack_outer_2,'\\EFIT::TOP.MEASUREMENTS.CECURR','\\EFIT::TOP.MEASUREMENTS.CCBRSP')
        """
        cecurr_key = Requirement(f'{self.measurements_node}.CECURR', shot, self.efit_tree).as_key()
        ccbrsp_key = Requirement(f'{self.measurements_node}.CCBRSP', shot, self.efit_tree).as_key()
        cecurr = raw_data[cecurr_key]
        ccbrsp = raw_data[ccbrsp_key]
        return np.concatenate([cecurr, ccbrsp], axis=1)

    def _compose_pf_current_chi_squared(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose PF current chi squared values using stack_outer_2.

        OMAS: py2tdi(stack_outer_2,'\\EFIT::TOP.MEASUREMENTS.CHIECC','\\EFIT::TOP.MEASUREMENTS.CHIFCC')
        """
        chiecc_key = Requirement(f'{self.measurements_node}.CHIECC', shot, self.efit_tree).as_key()
        chifcc_key = Requirement(f'{self.measurements_node}.CHIFCC', shot, self.efit_tree).as_key()
        chiecc = raw_data[chiecc_key]
        chifcc = raw_data[chifcc_key]
        return np.concatenate([chiecc, chifcc], axis=1)

    def _compose_psi(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose boundary separatrix psi with COCOS transformation.

        OMAS: data(\\EFIT::TOP.RESULTS.GEQDSK.SSIBRY)
        Transform: PSI (requires COCOS conversion)
        """
        ssibry_key = Requirement(f'{self.geqdsk_node}.SSIBRY', shot, self.efit_tree).as_key()
        psi = raw_data[ssibry_key]
        return self._apply_cocos_transform(psi, shot, raw_data, "equilibrium.time_slice.boundary_separatrix.psi")

    def _compose_ip_measured(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose measured plasma current with COCOS transformation.

        OMAS: data(\\EFIT::TOP.MEASUREMENTS.PLASMA)
        Transform: TOR (requires COCOS conversion)
        """
        plasma_key = Requirement(f'{self.measurements_node}.PLASMA', shot, self.efit_tree).as_key()
        ip = raw_data[plasma_key]
        return self._apply_cocos_transform(ip, shot, raw_data, "equilibrium.time_slice.constraints.ip.measured")

    def _compose_ip_reconstructed(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose reconstructed plasma current with COCOS transformation.

        OMAS: data(\\EFIT::TOP.MEASUREMENTS.CPASMA)
        Transform: TOR (requires COCOS conversion)
        """
        cpasma_key = Requirement(f'{self.measurements_node}.CPASMA', shot, self.efit_tree).as_key()
        ip = raw_data[cpasma_key]
        return self._apply_cocos_transform(ip, shot, raw_data, "equilibrium.time_slice.constraints.ip.reconstructed")

    def _compose_flux_loop_measured(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose measured flux loop data with COCOS transformation.

        OMAS: data(\\EFIT::TOP.MEASUREMENTS.SILOPT)
        Transform: PSI (magnetic flux requires COCOS conversion, factor of 2π for COCOS 1→11)
        """
        silopt_key = Requirement(f'{self.measurements_node}.SILOPT', shot, self.efit_tree).as_key()
        flux = raw_data[silopt_key]
        return self._apply_cocos_transform(flux, shot, raw_data, "equilibrium.time_slice.constraints.flux_loop.measured")

    def _compose_flux_loop_measured_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose flux loop measurement error with COCOS transformation.

        OMAS: data(\\EFIT::TOP.MEASUREMENTS.SIGSIL)
        Transform: PSI (error on magnetic flux requires same COCOS conversion as flux)
        """
        sigsil_key = Requirement(f'{self.measurements_node}.SIGSIL', shot, self.efit_tree).as_key()
        error = raw_data[sigsil_key]
        return self._apply_cocos_transform(error, shot, raw_data, "equilibrium.time_slice.constraints.flux_loop.measured_error_upper")

    def _compose_flux_loop_reconstructed(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose reconstructed flux loop data with COCOS transformation.

        OMAS: data(\\EFIT::TOP.MEASUREMENTS.CSILOP)
        Transform: PSI (magnetic flux requires COCOS conversion, factor of 2π for COCOS 1→11)
        """
        csilop_key = Requirement(f'{self.measurements_node}.CSILOP', shot, self.efit_tree).as_key()
        flux = raw_data[csilop_key]
        return self._apply_cocos_transform(flux, shot, raw_data, "equilibrium.time_slice.constraints.flux_loop.reconstructed")

    # COCOS transformation methods

    def _get_cocos_for_shot(self, shot: int, raw_data: dict) -> int:
        """
        Identify COCOS convention for a given shot.

        Uses Bt and Ip signs from GEQDSK to determine COCOS, following OMAS logic.
        Results are cached per shot.

        Args:
            shot: Shot number
            raw_data: Dictionary containing fetched MDS+ data

        Returns:
            COCOS number (1, 3, 5, or 7 typically)

        Note:
            This implements the same logic as OMAS MDS_gEQDSK_COCOS_identify
        """
        if shot in self._cocos_cache:
            return self._cocos_cache[shot]

        # Get Bt and Ip from GEQDSK data
        # Use mean values like OMAS does
        bcentr_key = Requirement(f'{self.geqdsk_node}.BCENTR', shot, self.efit_tree).as_key()
        cpasma_key = Requirement(f'{self.geqdsk_node}.CPASMA', shot, self.efit_tree).as_key()

        bt = raw_data[bcentr_key]
        ip = raw_data[cpasma_key]

        # Take mean if arrays (OMAS does this for time-dependent data)
        bt_mean = np.mean(bt) if hasattr(bt, '__len__') else bt
        ip_mean = np.mean(ip) if hasattr(ip, '__len__') else ip

        # Identify COCOS
        cocos = self.cocos.identify_cocos(bt_mean, ip_mean)
        self._cocos_cache[shot] = cocos

        return cocos

    def _apply_cocos_transform(self, data: np.ndarray, shot: int, raw_data: dict,
                               ids_path: str) -> np.ndarray:
        """
        Apply COCOS transformation to data if needed.

        Args:
            data: Data to transform
            shot: Shot number
            raw_data: Dictionary containing fetched MDS+ data (for COCOS identification)
            ids_path: IDS path to determine transformation type

        Returns:
            Transformed data (or original if no transform needed)
        """
        transform_type = get_cocos_transform_type(ids_path)
        if transform_type is None:
            return data

        source_cocos = self._get_cocos_for_shot(shot, raw_data)
        return self.cocos.transform(data, source_cocos, transform_type)
