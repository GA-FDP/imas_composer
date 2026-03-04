"""
Charge Exchange IDS Mapping for DIII-D

Maps DIII-D CER (Charge Exchange Recombination) diagnostic data to the
IMAS charge_exchange IDS.

See OMAS: omas/machine_mappings/d3d.py::charge_exchange_data

MDSplus tree: IONS
Subsystems: TANGENTIAL, VERTICAL
Analysis type: CERQUICK (default, configurable)

Channel discovery: channels 1..MAX_CHANNELS per subsystem are probed by fetching
TIME. Channels where TIME is an Exception or empty are skipped.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import awkward as ak

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class ChargeExchangeMapper(IDSMapper):
    """Maps DIII-D CER data to IMAS charge_exchange IDS."""

    SUBSYSTEMS = ['TANGENTIAL', 'VERTICAL']
    # Per-subsystem channel probe limit.
    # DIII-D CER has ~32 tangential and ~20 vertical channels.
    # Channels above this limit are not probed.
    MAX_CHANNELS = 40

    DOCS_PATH = "charge_exchange.yaml"
    CONFIG_PATH = "charge_exchange.yaml"

    def __init__(self, analysis_type: str = 'CERQUICK', **kwargs):
        """
        Initialize charge exchange mapper.

        Args:
            analysis_type: CER analysis quality level ('CERQUICK', 'CERAUTO', 'CERFIT')
        """
        self.analysis_type = analysis_type
        super().__init__()
        self._build_specs()

    # -------------------------------------------------------------------------
    # MDSplus path helpers
    # -------------------------------------------------------------------------

    def _cer_path(self, sub: str, ch: int, node: str) -> str:
        """Full MDSplus path for a CER channel node."""
        return f'\\IONS::TOP.CER.{self.analysis_type}.{sub}.CHANNEL{ch:02d}.{node}'

    def _cer_time_path(self, sub: str, ch: int, node: str) -> str:
        """MDSplus dim_of expression for CER channel node time (returns seconds)."""
        return f'dim_of({self._cer_path(sub, ch, node)}, 0)/1000'

    def _impdens_path(self, sub: str, ch: int, quantity: str) -> str:
        """Full MDSplus path for an IMPDENS channel node.

        Note: ch is not zero-padded in IMPDENS paths (e.g., FZT5, not FZT05).
        """
        return f'\\IONS::TOP.IMPDENS.{self.analysis_type}.{quantity}{sub[0]}{ch}'

    def _impdens_time_path(self, sub: str, ch: int, quantity: str) -> str:
        """MDSplus dim_of expression for IMPDENS channel time (returns seconds)."""
        return f'dim_of({self._impdens_path(sub, ch, quantity)}, 0)/1000'

    def _rot_node(self, sub: str) -> str:
        """MDSplus node name for toroidal rotation (differs by subsystem)."""
        return 'ROTC' if sub == 'TANGENTIAL' else 'ROT'

    # -------------------------------------------------------------------------
    # Requirement building
    # -------------------------------------------------------------------------

    def _build_reqs(self, path_fn) -> List[Requirement]:
        """Build DIRECT requirements for all (subsystem, channel) combinations.

        Args:
            path_fn: callable(sub, ch) -> mds_path string

        Returns:
            List of Requirements with shot=0 (replaced by actual shot at resolve time)
        """
        return [
            Requirement(path_fn(sub, ch), 0, 'IONS')
            for sub in self.SUBSYSTEMS
            for ch in range(1, self.MAX_CHANNELS + 1)
        ]

    def _build_specs(self):
        """Build all IDS entry specifications."""

        # ---- Internal DIRECT specs (one per data type, isolated requirements) ----

        # Position TIME: used for channel discovery and as position time base
        self.specs["charge_exchange._position_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._build_reqs(
                lambda s, c: self._cer_path(s, c, 'TIME')
            ),
            ids_path="charge_exchange._position_time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._position_r"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._build_reqs(
                lambda s, c: self._cer_path(s, c, 'R')
            ),
            ids_path="charge_exchange._position_r",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._position_z"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._build_reqs(
                lambda s, c: self._cer_path(s, c, 'Z')
            ),
            ids_path="charge_exchange._position_z",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._position_phi"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._build_reqs(
                lambda s, c: self._cer_path(s, c, 'VIEW_PHI')
            ),
            ids_path="charge_exchange._position_phi",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._t_i_data"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._build_reqs(
                lambda s, c: self._cer_path(s, c, 'TEMP')
            ),
            ids_path="charge_exchange._t_i_data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._t_i_error"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._build_reqs(
                lambda s, c: self._cer_path(s, c, 'TEMP_ERR')
            ),
            ids_path="charge_exchange._t_i_error",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._t_i_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._build_reqs(
                lambda s, c: self._cer_time_path(s, c, 'TEMP')
            ),
            ids_path="charge_exchange._t_i_time",
            docs_file=self.DOCS_PATH
        )

        # Toroidal rotation: ROTC for TANGENTIAL, ROT for VERTICAL
        self.specs["charge_exchange._velocity_tor_data"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._build_reqs(
                lambda s, c: self._cer_path(s, c, self._rot_node(s))
            ),
            ids_path="charge_exchange._velocity_tor_data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._velocity_tor_error"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._build_reqs(
                lambda s, c: self._cer_path(s, c, 'ROT_ERR')
            ),
            ids_path="charge_exchange._velocity_tor_error",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._velocity_tor_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._build_reqs(
                lambda s, c: self._cer_time_path(s, c, self._rot_node(s))
            ),
            ids_path="charge_exchange._velocity_tor_time",
            docs_file=self.DOCS_PATH
        )

        # Ion fraction (FZ) from IMPDENS tree
        self.specs["charge_exchange._n_i_over_n_e_data"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._build_reqs(
                lambda s, c: self._impdens_path(s, c, 'FZ')
            ),
            ids_path="charge_exchange._n_i_over_n_e_data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._n_i_over_n_e_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._build_reqs(
                lambda s, c: self._impdens_time_path(s, c, 'FZ')
            ),
            ids_path="charge_exchange._n_i_over_n_e_time",
            docs_file=self.DOCS_PATH
        )

        # Effective charge from IMPDENS tree
        self.specs["charge_exchange._zeff_data"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._build_reqs(
                lambda s, c: self._impdens_path(s, c, 'ZEFF')
            ),
            ids_path="charge_exchange._zeff_data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange._zeff_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=self._build_reqs(
                lambda s, c: self._impdens_time_path(s, c, 'ZEFF')
            ),
            ids_path="charge_exchange._zeff_time",
            docs_file=self.DOCS_PATH
        )

        # ---- Public COMPUTED specs ----

        self.specs["charge_exchange.channel.identifier"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time"],
            compose=self._compose_identifier,
            ids_path="charge_exchange.channel.identifier",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.name"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time"],
            compose=self._compose_name,
            ids_path="charge_exchange.channel.name",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.position.r.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time", "charge_exchange._position_r"],
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
            depends_on=["charge_exchange._position_time", "charge_exchange._position_z"],
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
            depends_on=["charge_exchange._position_time", "charge_exchange._position_phi"],
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

        self.specs["charge_exchange.channel.zeff.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time", "charge_exchange._zeff_data"],
            compose=self._compose_zeff_data,
            ids_path="charge_exchange.channel.zeff.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.zeff.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time", "charge_exchange._zeff_time"],
            compose=self._compose_zeff_time,
            ids_path="charge_exchange.channel.zeff.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.n_i_over_n_e.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time", "charge_exchange._n_i_over_n_e_data"],
            compose=self._compose_n_i_over_n_e_data,
            ids_path="charge_exchange.channel.ion.n_i_over_n_e.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.n_i_over_n_e.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time", "charge_exchange._n_i_over_n_e_time"],
            compose=self._compose_n_i_over_n_e_time,
            ids_path="charge_exchange.channel.ion.n_i_over_n_e.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.t_i.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time", "charge_exchange._t_i_data"],
            compose=self._compose_t_i_data,
            ids_path="charge_exchange.channel.ion.t_i.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.t_i.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time", "charge_exchange._t_i_error"],
            compose=self._compose_t_i_error,
            ids_path="charge_exchange.channel.ion.t_i.data_error_upper",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.t_i.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time", "charge_exchange._t_i_time"],
            compose=self._compose_t_i_time,
            ids_path="charge_exchange.channel.ion.t_i.time",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.velocity_tor.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time", "charge_exchange._velocity_tor_data"],
            compose=self._compose_velocity_tor_data,
            ids_path="charge_exchange.channel.ion.velocity_tor.data",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.velocity_tor.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time", "charge_exchange._velocity_tor_error"],
            compose=self._compose_velocity_tor_error,
            ids_path="charge_exchange.channel.ion.velocity_tor.data_error_upper",
            docs_file=self.DOCS_PATH
        )

        self.specs["charge_exchange.channel.ion.velocity_tor.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["charge_exchange._position_time", "charge_exchange._velocity_tor_time"],
            compose=self._compose_velocity_tor_time,
            ids_path="charge_exchange.channel.ion.velocity_tor.time",
            docs_file=self.DOCS_PATH
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _get_active_channels(self, shot: int, raw_data: dict) -> List[Tuple[str, int]]:
        """Return (sub, ch) pairs for channels with valid TIME data.

        A channel is active if its TIME fetch succeeded and returned a non-empty array.
        Matches OMAS: `if isinstance(postime, Exception): continue`.
        """
        active = []
        for sub in self.SUBSYSTEMS:
            for ch in range(1, self.MAX_CHANNELS + 1):
                key = Requirement(self._cer_path(sub, ch, 'TIME'), shot, 'IONS').as_key()
                val = raw_data.get(key)
                if val is None or isinstance(val, Exception):
                    continue
                time_arr = np.atleast_1d(val)
                if len(time_arr) == 0:
                    continue
                active.append((sub, ch))
        return active

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
            result.append(np.nan if val is None else float(np.atleast_1d(val)[0]))
        return np.array(result)

    def _compose_position_z_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose position Z values (scalar per channel, in meters).

        Matches OMAS: chpos['data'] = posdat  (posdat = data[Z], scalar)
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'Z'))
            result.append(np.nan if val is None else float(np.atleast_1d(val)[0]))
        return np.array(result)

    def _compose_position_phi_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """Compose position phi values (scalar per channel, in radians, COCOS 11).

        Matches OMAS: chpos['data'] = posdat * -np.pi / 180.0
        (VIEW_PHI in degrees → radians with sign convention)
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'VIEW_PHI'))
            if val is None:
                result.append(np.nan)
            else:
                result.append(float(np.atleast_1d(val)[0]) * -np.pi / 180.0)
        return np.array(result)

    def _compose_t_i_data(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose ion temperature time series per channel (in eV).

        Matches OMAS: ch['ion.0.t_i.data'] = unumpy.uarray(TEMP, TEMP_ERR) → .n (nominal)
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'TEMP'))
            result.append(np.atleast_1d(val) if val is not None else np.array([np.nan]))
        return ak.Array(result)

    def _compose_t_i_error(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose ion temperature upper uncertainty per channel (in eV).

        Matches OMAS: ch['ion.0.t_i.data'] = unumpy.uarray(TEMP, TEMP_ERR) → .std_devs
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'TEMP_ERR'))
            result.append(np.atleast_1d(val) if val is not None else np.array([np.nan]))
        return ak.Array(result)

    def _compose_t_i_time(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose ion temperature time arrays per channel (in seconds).

        Matches OMAS: ch['ion.0.t_i.time'] = dim_of(TEMP, 0)/1000
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._cer_time_path(sub, ch, 'TEMP'))
            result.append(np.atleast_1d(val) if val is not None else np.array([np.nan]))
        return ak.Array(result)

    def _compose_velocity_tor_data(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose toroidal velocity time series per channel (in m/s).

        TANGENTIAL uses ROTC node; VERTICAL uses ROT node.
        Matches OMAS: ch['ion.0.velocity_tor.data'] = ROT * 1000  (km/s → m/s)
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            node = self._rot_node(sub)
            val = self._lookup(raw_data, shot, self._cer_path(sub, ch, node))
            if val is not None:
                result.append(np.atleast_1d(val) * 1000.0)  # km/s → m/s
            else:
                result.append(np.array([np.nan]))
        return ak.Array(result)

    def _compose_velocity_tor_error(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose toroidal velocity upper uncertainty per channel (in m/s).

        Matches OMAS: ch['ion.0.velocity_tor.data'] = unumpy.uarray(ROT*1e3, ROT_ERR*1e3)
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._cer_path(sub, ch, 'ROT_ERR'))
            if val is not None:
                result.append(np.atleast_1d(val) * 1000.0)  # km/s → m/s
            else:
                result.append(np.array([np.nan]))
        return ak.Array(result)

    def _compose_velocity_tor_time(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose toroidal velocity time arrays per channel (in seconds).

        Matches OMAS: ch['ion.0.velocity_tor.time'] = dim_of(ROTC|ROT, 0)/1000
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            node = self._rot_node(sub)
            val = self._lookup(raw_data, shot, self._cer_time_path(sub, ch, node))
            result.append(np.atleast_1d(val) if val is not None else np.array([np.nan]))
        return ak.Array(result)

    def _compose_n_i_over_n_e_data(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose ion fraction time series per channel (dimensionless, 0-1).

        Matches OMAS: ch['ion.0.n_i_over_n_e.data'] = FZ * 0.01  (percent → fraction)
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._impdens_path(sub, ch, 'FZ'))
            if val is not None:
                result.append(np.atleast_1d(val) * 0.01)  # percent → fraction
            else:
                result.append(np.array([np.nan]))
        return ak.Array(result)

    def _compose_n_i_over_n_e_time(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose ion fraction time arrays per channel (in seconds).

        Matches OMAS: ch['ion.0.n_i_over_n_e.time'] = dim_of(FZ, 0)/1000
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._impdens_time_path(sub, ch, 'FZ'))
            result.append(np.atleast_1d(val) if val is not None else np.array([np.nan]))
        return ak.Array(result)

    def _compose_zeff_data(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose effective charge time series per channel (dimensionless).

        Matches OMAS: ch['zeff.data'] = ZEFF
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._impdens_path(sub, ch, 'ZEFF'))
            result.append(np.atleast_1d(val) if val is not None else np.array([np.nan]))
        return ak.Array(result)

    def _compose_zeff_time(self, shot: int, raw_data: dict) -> ak.Array:
        """Compose effective charge time arrays per channel (in seconds).

        Matches OMAS: ch['zeff.time'] = dim_of(ZEFF, 0)/1000
        """
        active = self._get_active_channels(shot, raw_data)
        result = []
        for sub, ch in active:
            val = self._lookup(raw_data, shot, self._impdens_time_path(sub, ch, 'ZEFF'))
            result.append(np.atleast_1d(val) if val is not None else np.array([np.nan]))
        return ak.Array(result)

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
