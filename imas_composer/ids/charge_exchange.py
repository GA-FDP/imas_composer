"""
Charge Exchange IDS Mapping for DIII-D

Maps DIII-D CER (Charge Exchange Recombination) diagnostic data to the
IMAS charge_exchange IDS.

See OMAS: omas/machine_mappings/d3d.py::charge_exchange_data

MDSplus tree: IONS
Subsystems: TANGENTIAL, VERTICAL
Analysis type: CERQUICK (default, configurable)

Channel discovery: getnci("CER.{analysis_type}.{sub}.CHANNEL*:TIME","LENGTH") is
fetched per subsystem; channels with LENGTH > 0 are active and fetched individually.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import awkward as ak

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class ChargeExchangeMapper(IDSMapper):
    """Maps DIII-D CER data to IMAS charge_exchange IDS."""

    DOCS_PATH = "charge_exchange.yaml"
    CONFIG_PATH = "charge_exchange.yaml"

    def __init__(self, analysis_type: str = 'CERAUTO', **kwargs):
        """
        Initialize charge exchange mapper.

        Args:
            analysis_type: CER analysis quality level ('CERAUTO', 'CERQUICK', 'CERFIT')
        """
        self.analysis_type = analysis_type
        super().__init__()
        # Load subsystem config from YAML
        self.SUBSYSTEMS = self._load_config().get('subsystems', ['TANGENTIAL', 'VERTICAL'])
        self._build_specs()

    # -------------------------------------------------------------------------
    # MDSplus path helpers
    # -------------------------------------------------------------------------

    def _handle_ROTC(self, sub:str, node: str):
        new_node = node
        if node == "ROTC" and sub == "VERTICAL":
            new_node = "ROT"
        return new_node

    def _cer_path(self, sub: str, ch: int, node: str) -> str:
        """Full MDSplus path for a CER channel node."""
        new_node = self._handle_ROTC(sub, node)
        return f'\\IONS::TOP.CER.{self.analysis_type}.{sub}.CHANNEL{ch:02d}.{new_node}'

    def _cer_time_path(self, sub: str, ch: int, node: str) -> str:
        """MDSplus dim_of expression for CER channel node time (returns seconds)."""
        new_node = self._handle_ROTC(sub, node)
        return f'dim_of({self._cer_path(sub, ch, new_node)}, 0)/1000'

    def _zeff_path(self) -> str:
        """MDSplus path for bulk ZEFF data — flattened array covering all channels."""
        return f'\\IONS::TOP.IMPDENS.{self.analysis_type}.ZEFF'
    
    def _zeff_time_path(self) -> str:
        """MDSplus dim_of for ZEFF time axis (seconds)."""
        return f'dim_of({self._zeff_path()}, 0)/1000'

    def _zimp_path(self) -> str:
        """MDSplus path for charge of the impurity measured — flattened array covering all channels."""
        return f'\\IONS::TOP.IMPDENS.{self.analysis_type}.ZIMP'

    def _zimp_time_path(self) -> str:
        """MDSplus dim_of for ZIMP time axis (seconds)."""
        return f'dim_of({self._zimp_path()}, 0)/1000'

    def _concen_path(self) -> str:
        """MDSplus path for bulk CONCEN (ion fraction) data — flattened array covering all channels."""
        return f'\\IONS::TOP.IMPDENS.{self.analysis_type}.CONCEN'

    def _concen_time_path(self) -> str:
        """MDSplus dim_of for CONCEN time axis (seconds)."""
        return f'dim_of({self._concen_path()}, 0)/1000'

    def _concen_err_path(self) -> str:
        """MDSplus path for bulk CONCEN_ERR (ion fraction uncertainty) — flattened array covering all channels."""
        return f'\\IONS::TOP.IMPDENS.{self.analysis_type}.ERR_CONCEN'

    def _impdens_path(self) -> str:
        """MDSplus path for bulk IMPDENS (ion density) data — flattened array covering all channels."""
        return f'\\IONS::TOP.IMPDENS.{self.analysis_type}.IMPDENS'

    def _impdens_time_path(self) -> str:
        """MDSplus dim_of for IMPDENS time axis (seconds)."""
        return f'dim_of({self._impdens_path()}, 0)/1000'

    def _impdens_err_path(self) -> str:
        """MDSplus path for bulk ERR_IMPDENS (ion density uncertainty) — flattened array covering all channels."""
        return f'\\IONS::TOP.IMPDENS.{self.analysis_type}.ERR_IMPDENS'

    def _impdens_indices_path(self) -> str:
        """MDSplus path for INDECIES — maps bulk array columns to CER channel numbers."""
        return f'\\IONS::TOP.IMPDENS.{self.analysis_type}.INDECIES'

    def _get_active_path(self, sub: str) -> str:
        """MDSplus path returning TIME node LENGTH for each channel in a subsystem."""
        return f'getnci("CER.{self.analysis_type}.{sub}.CHANNEL*:TIME","LENGTH")'

    # -------------------------------------------------------------------------
    # Requirement building
    # -------------------------------------------------------------------------

    def _make_derive_fn(self, path_fn, channel_getter=None):
        """Create a derive_requirements function for a per-channel path function.

        Called by the resolver after the getnci active-channel data is available.
        Returns Requirements only for channels where LENGTH > 0.

        Args:
            path_fn: callable(sub, ch) -> mds_path string
            channel_getter: callable(shot, raw_data) -> List[Tuple[str, int]],
                defaults to _get_active_channels
        """
        if channel_getter is None:
            channel_getter = self._get_active_channels
        def derive(shot: int, raw_data: dict) -> List[Requirement]:
            return [Requirement(path_fn(sub, ch), shot, 'IONS')
                    for sub, ch in channel_getter(shot, raw_data)]
        return derive

    def _build_specs(self):
        """Build all IDS entry specifications."""

        # ---- Phase 1 DIRECT specs: getnci TIME LENGTH arrays ----
        # Fetched first; LENGTH > 0 identifies which channels have data.
        _active_deps = [
            "charge_exchange._tangential_active",
            "charge_exchange._vertical_active",
        ]

        # ---- Phase 2 DERIVED specs: per-channel CER data for active channels only ----
        self.specs["charge_exchange._position_time"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=_active_deps,
            derive_requirements=self._make_derive_fn(lambda s, c: self._cer_path(s, c, 'TIME')),
            ids_path="charge_exchange._position_time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._position_r"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=_active_deps,
            derive_requirements=self._make_derive_fn(lambda s, c: self._cer_path(s, c, 'R')),
            ids_path="charge_exchange._position_r",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._position_z"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=_active_deps,
            derive_requirements=self._make_derive_fn(lambda s, c: self._cer_path(s, c, 'Z')),
            ids_path="charge_exchange._position_z",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._position_phi"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=_active_deps,
            derive_requirements=self._make_derive_fn(lambda s, c: self._cer_path(s, c, 'VIEW_PHI')),
            ids_path="charge_exchange._position_phi",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._t_i_data"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=_active_deps,
            derive_requirements=self._make_derive_fn(lambda s, c: self._cer_path(s, c, 'TEMP')),
            ids_path="charge_exchange._t_i_data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._t_i_error"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=_active_deps,
            derive_requirements=self._make_derive_fn(lambda s, c: self._cer_path(s, c, 'TEMP_ERR')),
            ids_path="charge_exchange._t_i_error",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._t_i_error_statistical"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=_active_deps,
            derive_requirements=self._make_derive_fn(lambda s, c: self._cer_path(s, c, 'TEMP_ERR_PS')),
            ids_path="charge_exchange._t_i_error_statistical",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._t_i_time"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=_active_deps,
            derive_requirements=self._make_derive_fn(lambda s, c: self._cer_time_path(s, c, 'TEMP')),
            ids_path="charge_exchange._t_i_time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._velocity_data"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=_active_deps,
            derive_requirements=self._make_derive_fn(lambda s, c: self._cer_path(s, c, 'ROTC'), self._get_active_channels),
            ids_path="charge_exchange._velocity_data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._velocity_error"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=_active_deps,
            derive_requirements=self._make_derive_fn(lambda s, c: self._cer_path(s, c, 'ROT_ERR'), self._get_active_channels),
            ids_path="charge_exchange._velocity_error",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._velocity_error_statistical"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=_active_deps,
            derive_requirements=self._make_derive_fn(lambda s, c: self._cer_path(s, c, 'ROT_ERR_PS'), self._get_active_channels),
            ids_path="charge_exchange._velocity_error_statistical",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._velocity_time"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=_active_deps,
            derive_requirements=self._make_derive_fn(lambda s, c: self._cer_time_path(s, c, 'ROTC'), 
                                                     self._get_active_channels),
            ids_path="charge_exchange._velocity_time",
            docs_file=self.DOCS_PATH
        )
        # Bulk IMPDENS data: ZEFF, ZIMP (charge of measured impurity) and CONCEN (ion fraction) are flattened arrays
        # covering all channels. INDECIES maps array columns to CER channel numbers.
        # ARRAY_ORDER from CALIBRATION describes the subsystem/channel ordering.

        self.specs["charge_exchange._zeff"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(self._zeff_path(), 0, 'IONS')],
            ids_path="charge_exchange._zeff",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._zeff_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(self._zeff_time_path(), 0, 'IONS')],
            ids_path="charge_exchange._zeff_time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._zimp"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(self._zimp_path(), 0, 'IONS')],
            ids_path="charge_exchange._zimp",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._zimp_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(self._zimp_time_path(), 0, 'IONS')],
            ids_path="charge_exchange._zimp_time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._concen"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(self._concen_path(), 0, 'IONS')],
            ids_path="charge_exchange._concen",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._concen_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(self._concen_time_path(), 0, 'IONS')],
            ids_path="charge_exchange._concen_time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._concen_err"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(self._concen_err_path(), 0, 'IONS')],
            ids_path="charge_exchange._concen_err",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._impdens"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(self._impdens_path(), 0, 'IONS')],
            ids_path="charge_exchange._impdens",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._impdens_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(self._impdens_time_path(), 0, 'IONS')],
            ids_path="charge_exchange._impdens_time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._impdens_err"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(self._impdens_err_path(), 0, 'IONS')],
            ids_path="charge_exchange._impdens_err",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._impdens_indices"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(self._impdens_indices_path(), 0, 'IONS')],
            ids_path="charge_exchange._impdens_indices",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._array_order"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement('\\IONS::TOP.CER.CALIBRATION.ARRAY_ORDER', 0, 'IONS')],
            ids_path="charge_exchange._array_order",
            docs_file=self.DOCS_PATH
        )

        # get_active_channels DIRECT specs: TIME node LENGTH array, one entry per channel node.
        # LENGTH > 0 means the channel has data for this analysis type.
        # These drive _get_active_channels and must appear in depends_on of every channel spec.
        self.specs["charge_exchange._tangential_active"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(self._get_active_path('TANGENTIAL'), 0, 'IONS')],
            ids_path="charge_exchange._tangential_active",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._vertical_active"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[Requirement(self._get_active_path('VERTICAL'), 0, 'IONS')],
            ids_path="charge_exchange._vertical_active",
            docs_file=self.DOCS_PATH
        )

        # ---- Public COMPUTED specs ----
        self.specs["charge_exchange.channel.identifier"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=_active_deps,
            compose=self._compose_identifier,
            ids_path="charge_exchange.channel.identifier",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.name"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=_active_deps,
            compose=self._compose_name,
            ids_path="charge_exchange.channel.name",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.position.r.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_r"],
            compose=self._compose_position_r_data,
            ids_path="charge_exchange.channel.position.r.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.position.r.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time"],
            compose=self._compose_position_time,
            ids_path="charge_exchange.channel.position.r.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.position.z.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_z"],
            compose=self._compose_position_z_data,
            ids_path="charge_exchange.channel.position.z.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.position.z.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time"],
            compose=self._compose_position_time,
            ids_path="charge_exchange.channel.position.z.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.position.phi.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_phi"],
            compose=self._compose_position_phi_data,
            ids_path="charge_exchange.channel.position.phi.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.position.phi.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time"],
            compose=self._compose_position_time,
            ids_path="charge_exchange.channel.position.phi.time",
            docs_file=self.DOCS_PATH
        )

        _impdens_deps = [
            "charge_exchange._impdens_indices",
            "charge_exchange._array_order",
        ]

        self.specs["charge_exchange.channel.zeff.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=_active_deps + ["charge_exchange._zeff"] + _impdens_deps,
            compose=self._compose_zeff_data,
            ids_path="charge_exchange.channel.zeff.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.zeff.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=_active_deps + ["charge_exchange._zeff_time"] + _impdens_deps,
            compose=self._compose_zeff_time,
            ids_path="charge_exchange.channel.zeff.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.n_i_over_n_e.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=_active_deps + ["charge_exchange._concen"] + _impdens_deps,
            compose=self._compose_n_i_over_n_e_data,
            ids_path="charge_exchange.channel.ion.n_i_over_n_e.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.n_i_over_n_e.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=_active_deps + ["charge_exchange._concen_time"] + _impdens_deps,
            compose=self._compose_n_i_over_n_e_time,
            ids_path="charge_exchange.channel.ion.n_i_over_n_e.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.n_i_over_n_e.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=_active_deps + ["charge_exchange._concen_err"] + _impdens_deps,
            compose=self._compose_n_i_over_n_e_error,
            ids_path="charge_exchange.channel.ion.n_i_over_n_e.data_error_upper",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.n_i.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=_active_deps + ["charge_exchange._impdens"] + _impdens_deps,
            compose=self._compose_n_i_data,
            ids_path="charge_exchange.channel.ion.n_i.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.n_i.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=_active_deps + ["charge_exchange._impdens_time"] + _impdens_deps,
            compose=self._compose_n_i_time,
            ids_path="charge_exchange.channel.ion.n_i.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.n_i.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=_active_deps + ["charge_exchange._impdens_err"] + _impdens_deps,
            compose=self._compose_n_i_error,
            ids_path="charge_exchange.channel.ion.n_i.data_error_upper",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.t_i.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._t_i_data"],
            compose=self._compose_t_i_data,
            ids_path="charge_exchange.channel.ion.t_i.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.t_i.error"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._t_i_error", "charge_exchange._t_i_error_statistical"],
            compose=self._compose_t_i_error,
            ids_path="charge_exchange.channel.ion.t_i.error",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.t_i.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._t_i_error", "charge_exchange._t_i_error_statistical"],
            compose=self._compose_t_i_data_error_upper,
            ids_path="charge_exchange.channel.ion.t_i.data_error_upper",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.t_i.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._t_i_time"],
            compose=self._compose_t_i_time,
            ids_path="charge_exchange.channel.ion.t_i.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.velocity.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=_active_deps + ["charge_exchange._velocity_data"],
            compose=self._compose_velocity_data,
            ids_path="charge_exchange.channel.ion.velocity.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.velocity.error"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=_active_deps + ["charge_exchange._velocity_data", 
                                       "charge_exchange._velocity_error", 
                                       "charge_exchange._velocity_error_statistical"],
            compose=self._compose_velocity_error,
            ids_path="charge_exchange.channel.ion.velocity.error",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.velocity.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=_active_deps + ["charge_exchange._velocity_data", 
                                       "charge_exchange._velocity_error", 
                                       "charge_exchange._velocity_error_statistical"],
            compose=self._compose_velocity_data_error_upper,
            ids_path="charge_exchange.channel.ion.velocity.data_error_upper",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.velocity.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=_active_deps + ["charge_exchange._velocity_time"],
            compose=self._compose_velocity_time,
            ids_path="charge_exchange.channel.ion.velocity.time",
            docs_file=self.DOCS_PATH
        )

        # Total installed channels: count nodes with data in the CALIBRATION tree (analysis-type independent)
        self.specs["charge_exchange._tangential_installed"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(
                    'getnci("CER.CALIBRATION.TANGENTIAL.CHANNEL*:BEAMGEOMETRY","LENGTH")',
                    0, 'IONS'
                )
            ],
            ids_path="charge_exchange._tangential_installed",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._vertical_installed"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(
                    'getnci("CER.CALIBRATION.VERTICAL.CHANNEL*:BEAMGEOMETRY","LENGTH")',
                    0, 'IONS'
                )
            ],
            ids_path="charge_exchange._vertical_installed",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.code.parameters.total_installed_channels"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[
                "charge_exchange._tangential_installed",
                "charge_exchange._vertical_installed",
            ],
            compose=self._compose_total_installed_channels,
            ids_path="charge_exchange.code.parameters.total_installed_channels",
            docs_file=self.DOCS_PATH
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_active_by_node(self, shot: int, raw_data: dict, path_fn) -> List[Tuple[str, int]]:
        """Return (sub, ch) pairs for channels where getnci LENGTH > 0.

        Uses getnci LENGTH arrays: index i corresponds to channel i+1.

        Args:
            path_fn: callable(sub) -> getnci mds_path string
        """
        active = []
        for sub in self.SUBSYSTEMS:
            lengths = self._lookup(raw_data, shot, path_fn(sub))
            if lengths is None:
                continue
            for i, length in enumerate(np.atleast_1d(lengths)):
                if length > 0:
                    active.append((sub, i + 1))
        return active

    def _get_active_channels(self, shot: int, raw_data: dict) -> List[Tuple[str, int]]:
        """Return (sub, ch) pairs for channels that have TIME data."""
        return self._get_active_by_node(shot, raw_data, self._get_active_path)

    def _get_array_order(self, shot: int, raw_data: dict) -> List[str]:
        """Return the names of CER systems/channels in the order stored."""
        array_order = self._lookup(raw_data, shot, '\\IONS::TOP.CER.CALIBRATION.ARRAY_ORDER')
        return [a.decode().strip() for a in array_order]

    def _lookup(self, raw_data: dict, shot: int, mds_path: str) -> Optional[np.ndarray]:
        """Look up a value in raw_data by MDS path.

        Returns None if the key is missing or the value is an Exception.
        """
        key = Requirement(mds_path, shot, 'IONS').as_key()
        val = raw_data.get(key)
        if val is None or isinstance(val, Exception):
            return None
        return val

    # -------------------------------------------------------------------------
    # Compose functions
    # -------------------------------------------------------------------------

    def _compose_identifier(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose channel identifiers: '{sub[0]}{ch:02d}' (e.g., 'T01', 'V05').

        Matches OMAS: ch['identifier'] = '{}{:02d}'.format(sub[0], channel)
        """
        active = self._get_active_channels(shot, raw_data)
        return np.array([f'{sub[0]}{ch:02d}' for sub, ch in active])

    def _compose_name(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose channel names: 'impCER_{sub}{ch:02d}' (e.g., 'impCER_TANGENTIAL01').

        Matches OMAS: ch['name'] = 'impCER_{}{:02d}'.format(sub, channel)
        """
        active = self._get_active_channels(shot, raw_data)
        return np.array([f'impCER_{sub}{ch:02d}' for sub, ch in active])

    def _compose_position_time(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose position time arrays (TIME / 1000, in seconds).

        Used for position.r.time, position.z.time, position.phi.time.
        Matches OMAS: chpos['time'] = postime  (postime = data[TIME] / 1000.0)
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'TIME'))
            result.append(np.atleast_1d(val) / 1000.0)
        return ak.Array(result)

    def _compose_position_r_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose position R values (scalar per channel, in meters).

        Matches OMAS: chpos['data'] = posdat  (posdat = data[R], scalar)
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'R'))
            result.append([] if val is None else np.atleast_1d(val))
        return ak.Array(result)

    def _compose_position_z_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose position Z values (scalar per channel, in meters).

        Matches OMAS: chpos['data'] = posdat  (posdat = data[Z], scalar)
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'Z'))
            result.append([] if val is None else np.atleast_1d(val))
        return ak.Array(result)

    def _compose_position_phi_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose position phi values (scalar per channel, in radians, COCOS 11).

        Matches OMAS: chpos['data'] = posdat * -np.pi / 180.0
        (VIEW_PHI in degrees → radians with sign convention)
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'VIEW_PHI'))
            result.append(np.atleast_1d(val) * -np.pi / 180.0)
        return ak.Array(result)

    def _compose_t_i_data(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose ion temperature time series per channel (in eV).

        Matches OMAS: ch['ion.:.t_i.data'] = unumpy.uarray(TEMP, TEMP_ERR) → .n (nominal)
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'TEMP'))
            result.append(np.atleast_1d(val) if val is not None else np.array([]))
        return ak.Array(result)[:, None, ...]

    def _compose_t_i_error(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose ion temperature errors per channel (in eV), shape (2, n_time) per channel.

        Element 0: statistical uncertainty (TEMP_ERR_PS)
        Element 1: systematic uncertainty (TEMP_ERR)
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            stat = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'TEMP_ERR_PS'))
            sys = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'TEMP_ERR'))
            stat_arr = np.atleast_1d(stat) if stat is not None else np.array([])
            sys_arr = np.atleast_1d(sys) if sys is not None else np.array([])
            result.append(np.stack([stat_arr, sys_arr]))
        return ak.Array(result)[:, None,...]

    def _compose_t_i_data_error_upper(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose total ion temperature uncertainty per channel (sum of statistical and systematic, in eV)."""
        return ak.sum(self._compose_t_i_error(shot, raw_data), axis=-2)

    def _compose_t_i_time(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose ion temperature time arrays per channel (in seconds).

        Matches OMAS: ch['ion.:.t_i.time'] = dim_of(TEMP, 0)/1000
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._cer_time_path(sub, ch, 'TEMP'))
            result.append(np.atleast_1d(val) if val is not None else np.array([]))
        return ak.Array(result)[:, None,...]

    def _compose_velocity_data(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose velocity time series per channel (in m/s).

        TANGENTIAL uses ROTC node; VERTICAL uses ROT node (because ROTC is missing).

        This is not part of the IMAS schema, which only has velocity_tor and velocity_pol.
        The direction will depend on the beam orientation and should be extracted accordingly.
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'ROTC'))
            if val is not None:
                result.append(np.atleast_1d(val) * 1000.0)  # km/s → m/s
            else:
                result.append([])
        return ak.Array(result)[:, None,...]


    def _compose_velocity_error(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose velocity errors per channel (in m/s), shape (2, n_time) per channel.

        Element 0: statistical uncertainty (ROT_ERR_PS)
        Element 1: systematic uncertainty (ROT_ERR)

        This is not part of the IMAS schema, which only has velocity_tor and velocity_pol.
        The direction will depend on the beam orientation and should be extracted accordingly.
        """
        active = self._get_active_channels(shot, raw_data)
        active_velocity = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'ROTC'))
            stat = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'ROT_ERR_PS'))
            sys = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'ROT_ERR'))
            if val is not None:
                stat_arr = np.atleast_1d(stat) * 1000.0 if stat is not None else np.array([])
                sys_arr = np.atleast_1d(sys) * 1000.0 if sys is not None else np.array([])
                result.append(np.stack([stat_arr, sys_arr]))
            else:
                result.append(np.stack([[], []]))
        return ak.Array(result)[:, None,...]

    def _compose_velocity_data_error_upper(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose total velocity uncertainty per channel (sum of statistical and systematic, in m/s)."""
        return ak.sum(self._compose_velocity_error(shot, raw_data), axis=-2)

    def _compose_velocity_time(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose velocity time arrays per channel (in seconds).

        This is not part of the IMAS schema, which only has velocity_tor and velocity_pol.
        The direction will depend on the beam orientation and should be extracted accordingly.
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._cer_time_path(sub, ch, 'ROTC'))
            result.append(np.atleast_1d(val) if val is not None else np.array([]))
        return ak.Array(result)[:, None,...]

    def _compose_n_i_over_n_e_data(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose ion fraction time series per channel (dimensionless, 0-1).

        Data source: CONCEN bulk flattened array (all channels), indexed via INDECIES and ARRAY_ORDER.
        Matches OMAS: ch['ion.:.n_i_over_n_e.data'] = CONCEN * 0.01  (percent → fraction)
        """
        active = self._get_active_channels(shot, raw_data)
        concen = self._lookup(raw_data, shot, self._concen_path())
        indices = self._lookup(raw_data, shot, self._impdens_indices_path())
        array_order = self._get_array_order(shot, raw_data)
        result = []
        for sub, ch in active:
            ich = array_order.index(f'{sub[0:4]}{ch}')
            ind = slice(indices[ich],indices[ich+1])
            val = np.atleast_1d(concen[ind])
            result.append(val * 1.e-2 if len(val) > 0 else np.array([])) # remove %
        return ak.Array(result)[:, None,...]

    def _compose_n_i_over_n_e_time(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose ion fraction time arrays per channel (in seconds).

        Data source: dim_of(CONCEN) time axis, indexed via INDECIES and ARRAY_ORDER.
        Matches OMAS: ch['ion.:.n_i_over_n_e.time'] = dim_of(CONCEN, 0)/1000
        """
        active = self._get_active_channels(shot, raw_data)
        concen_time = self._lookup(raw_data, shot, self._concen_time_path())
        indices = self._lookup(raw_data, shot, self._impdens_indices_path())
        array_order = self._get_array_order(shot, raw_data)
        result = []
        for sub, ch in active:
            ich = array_order.index(f'{sub[0:4]}{ch}')
            ind = slice(indices[ich],indices[ich+1])
            val = np.atleast_1d(concen_time[ind])
            result.append(val if len(val) > 0 else np.array([]))
        return ak.Array(result)[:, None,...]

    def _compose_n_i_over_n_e_error(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose ion fraction upper uncertainty per channel (dimensionless, 0-1).

        Data source: ERR_CONCEN bulk flattened array, indexed via INDECIES and ARRAY_ORDER.
        """
        active = self._get_active_channels(shot, raw_data)
        concen_err = self._lookup(raw_data, shot, self._concen_err_path())
        indices = self._lookup(raw_data, shot, self._impdens_indices_path())
        array_order = self._get_array_order(shot, raw_data)
        result = []
        for sub, ch in active:
            ich = array_order.index(f'{sub[0:4]}{ch}')
            ind = slice(indices[ich],indices[ich+1])
            val = np.atleast_1d(concen_err[ind])
            result.append(val * 1.e-2  if len(val) > 0 else np.array([]))
        return ak.Array(result)[:, None,...]

    def _compose_n_i_data(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose ion density time series per channel.

        Data source: IMPDENS bulk flattened array (all channels), indexed via INDECIES and ARRAY_ORDER.
        """
        active = self._get_active_channels(shot, raw_data)
        impdens = self._lookup(raw_data, shot, self._impdens_path())
        indices = self._lookup(raw_data, shot, self._impdens_indices_path())
        array_order = self._get_array_order(shot, raw_data)
        result = []
        for sub, ch in active:
            ich = array_order.index(f'{sub[0:4]}{ch}')
            ind = slice(indices[ich], indices[ich+1])
            val = np.atleast_1d(impdens[ind])
            result.append(val if len(val) > 0 else np.array([]))
        return ak.Array(result)[:, None,...]

    def _compose_n_i_time(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose ion density time arrays per channel (in seconds).

        Data source: dim_of(IMPDENS) time axis, indexed via INDECIES and ARRAY_ORDER.
        """
        active = self._get_active_channels(shot, raw_data)
        impdens_time = self._lookup(raw_data, shot, self._impdens_time_path())
        indices = self._lookup(raw_data, shot, self._impdens_indices_path())
        array_order = self._get_array_order(shot, raw_data)
        result = []
        for sub, ch in active:
            ich = array_order.index(f'{sub[0:4]}{ch}')
            ind = slice(indices[ich], indices[ich+1])
            val = np.atleast_1d(impdens_time[ind])
            result.append(val if len(val) > 0 else np.array([]))
        return ak.Array(result)[:, None,...]

    def _compose_n_i_error(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose ion density upper uncertainty per channel.

        Data source: ERR_IMPDENS bulk flattened array, indexed via INDECIES and ARRAY_ORDER.
        """
        active = self._get_active_channels(shot, raw_data)
        impdens_err = self._lookup(raw_data, shot, self._impdens_err_path())
        indices = self._lookup(raw_data, shot, self._impdens_indices_path())
        array_order = self._get_array_order(shot, raw_data)
        result = []
        for sub, ch in active:
            ich = array_order.index(f'{sub[0:4]}{ch}')
            ind = slice(indices[ich], indices[ich+1])
            val = np.atleast_1d(impdens_err[ind])
            result.append(val if len(val) > 0 else np.array([]))
        return ak.Array(result)[:, None,...]

    def _compose_zeff_data(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose effective charge time series per channel (dimensionless).

        Data source: ZEFF bulk flattened array (all channels), indexed via INDECIES and ARRAY_ORDER.
        Matches OMAS: ch['zeff.data'] = ZEFF
        """
        active = self._get_active_channels(shot, raw_data)
        zeff = self._lookup(raw_data, shot, self._zeff_path())
        indices = self._lookup(raw_data, shot, self._impdens_indices_path())
        array_order = self._get_array_order(shot, raw_data)
        result = []
        for sub, ch in active:
            ich = array_order.index(f'{sub[0:4]}{ch}')
            ind = slice(indices[ich], indices[ich+1])
            val = np.atleast_1d(zeff[ind])
            result.append(val if len(val) > 0 else np.array([]))
        return ak.Array(result)

    def _compose_zeff_time(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose effective charge time arrays per channel (in seconds).

        Data source: dim_of(ZIMP) time axis, indexed via INDECIES and ARRAY_ORDER.
        Matches OMAS: ch['zeff.time'] = dim_of(ZIMP, 0)/1000
        """
        active = self._get_active_channels(shot, raw_data)
        zeff_time = self._lookup(raw_data, shot, self._zeff_time_path())
        indices = self._lookup(raw_data, shot, self._impdens_indices_path())
        array_order = self._get_array_order(shot, raw_data)
        result = []
        for sub, ch in active:
            ich = array_order.index(f'{sub[0:4]}{ch}')
            ind = slice(indices[ich],indices[ich+1])
            val = np.atleast_1d(zeff_time[ind])
            result.append(val if len(val) > 0 else np.array([]))
        return ak.Array(result)

    def _compose_total_installed_channels(self, shot: int, raw_data: dict) -> int:
        """Count total installed CER channels from CALIBRATION tree.

        Uses getnci to count BEAMGEOMETRY nodes that have data across TANGENTIAL and VERTICAL
        subsystems. The CALIBRATION tree is analysis-type independent.

        getnci returns one LENGTH value per matching channel node, so
        len() of the result equals the number of installed channels.
        """
        tang_path = 'getnci("CER.CALIBRATION.TANGENTIAL.CHANNEL*:BEAMGEOMETRY","LENGTH")'
        vert_path = 'getnci("CER.CALIBRATION.VERTICAL.CHANNEL*:BEAMGEOMETRY","LENGTH")'
        tang_data = self._lookup(raw_data, shot, tang_path)
        vert_data = self._lookup(raw_data, shot, vert_path)
        tang_count = int(sum(tang_data > 0)) if tang_data is not None else 0
        vert_count = int(sum(vert_data > 0)) if vert_data is not None else 0
        return tang_count + vert_count

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
