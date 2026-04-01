"""
Magnetics IDS Mapping for DIII-D

Maps plasma current, diamagnetic flux, and poloidal field probe measurements
to IMAS magnetics IDS.
See OMAS: omas/machine_mappings/d3d.py::ip_bt_dflux_data, magnetics_hardware,
          magnetics_probes_data, magnetics_floops_data
"""

from pathlib import Path
import json
from typing import Dict, List
import numpy as np
import awkward as ak

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class MagneticsMapper(IDSMapper):
    """Maps DIII-D magnetics data to IMAS magnetics IDS."""

    CONFIG_PATH = "magnetics.yaml"

    def __init__(self, **kwargs):
        """Initialize Magnetics mapper."""
        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Build IDS specs
        self._build_specs()

    def _load_mhdin(self, shot: int, key: str) -> list:
        """
        Load a magnetics hardware list from mhdin_ods.json.

        Selects the file for the nearest shot boundary <= shot.

        Args:
            shot: Shot number
            key: Key under magnetics (e.g. 'b_field_pol_probe', 'flux_loop')
        """
        machine_desc_dir = Path(__file__).parent.parent / 'machine_description' / 'D3D'
        available = sorted(
            int(d.name) for d in machine_desc_dir.iterdir()
            if d.is_dir() and d.name.isdigit()
        )
        shot_dir = available[0]
        for s in available:
            if s <= shot:
                shot_dir = s
            else:
                break
        json_path = machine_desc_dir / str(shot_dir) / 'mhdin_ods.json'
        with open(json_path) as f:
            data = json.load(f)
        return data['magnetics'][key]

    def _load_probes(self, shot: int) -> list:
        """Load b_field_pol_probe hardware geometry from mhdin_ods.json."""
        return self._load_mhdin(shot, 'b_field_pol_probe')

    def _load_flux_loops(self, shot: int) -> list:
        """Load flux_loop hardware geometry from mhdin_ods.json."""
        return self._load_mhdin(shot, 'flux_loop')

    def _load_fitweights(self, shot: int, name: str) -> list:
        """
        Load fitweight data for the given shot and weight name.

        Parses fitweight.dat using the same logic as D3DFitweight from OMAS _common.py.
        Selects the nearest shot boundary <= shot.

        Args:
            shot: Shot number
            name: Weight name, either 'fwtsi' (flux_loop) or 'fwtmp2' (b_field_pol_probe)
        """
        fitweight_path = Path(__file__).parent.parent / 'machine_description' / 'D3D' / 'fitweight.dat'

        magpri67 = 29
        magpri322 = 31
        magprirdp = 8
        magudom = 5
        maglds = 3
        nsilds = 3
        nsilol = 41

        with open(fitweight_path) as f:
            data = f.read()

        tokens = data.strip().split()
        weights_by_shot = {}
        ishot = None
        for token in tokens:
            ifloat = float(token)
            if ifloat > 100:
                ishot = int(ifloat)
                weights_by_shot[ishot] = []
            else:
                weights_by_shot[ishot].append(ifloat)

        processed = {}
        for irshot, vals in weights_by_shot.items():
            if irshot < 124985:
                mloop = nsilol
            else:
                mloop = nsilol + nsilds

            if irshot < 59350:
                mprobe = magpri67
            elif irshot < 91000:
                mprobe = magpri67 + magpri322
            elif irshot < 100771:
                mprobe = magpri67 + magpri322 + magprirdp
            elif irshot < 124985:
                mprobe = magpri67 + magpri322 + magprirdp + magudom
            else:
                mprobe = magpri67 + magpri322 + magprirdp + magudom + maglds

            processed[irshot] = {
                'fwtsi': vals[0:mloop],
                'fwtmp2': vals[mloop:mloop + mprobe],
            }

        available = sorted(processed.keys())
        selected = available[0]
        for s in available:
            if s <= shot:
                selected = s
            else:
                break

        return processed[selected][name]

    def _load_compfile(self, path: Path) -> dict:
        """
        Parse a compensation file (btcomp, ccomp, or icomp).

        Replicates D3DCompfile.load() from OMAS _common.py.
        Returns {compshot: {compsig: {identifier: coeff}}}.
        """
        with open(path) as f:
            lines = f.readlines()

        result = {}
        compshot = None
        for line in lines:
            linesplit = line.split()
            if not linesplit:
                continue
            try:
                compshot = int(eval(linesplit[0]))
                result[compshot] = {}
                for compsig in linesplit[1:]:
                    result[compshot][compsig.strip("'")] = {}
            except Exception:
                sig = linesplit[0][1:].strip()
                comps = [float(x) for x in linesplit[2:]]
                for comp_val, compsig in zip(comps, result[compshot]):
                    result[compshot][compsig][sig] = comp_val
        return result

    def _get_comp_corrections(self, shot: int) -> list:
        """
        Load all compensation corrections active for the given shot.

        Reads btcomp, ccomp, and icomp; selects the nearest compshot <= shot
        for each. Excludes N1COIL for shots > 112962.

        Returns list of (compsig, coeff_map) tuples where coeff_map is
        {identifier: coeff}. Multiple entries may share the same compsig
        (from different comp files) and are applied sequentially.
        """
        comp_dir = Path(__file__).parent.parent / 'machine_description' / 'D3D'
        corrections = []
        for compfile in ['btcomp', 'ccomp', 'icomp']:
            comp = self._load_compfile(comp_dir / f'{compfile}.dat')
            available = sorted(comp.keys())
            compshot = available[0]
            for s in available:
                if s <= shot:
                    compshot = s
                else:
                    break
            for compsig, coeff_map in comp[compshot].items():
                if compsig == 'N1COIL' and shot > 112962:
                    continue
                corrections.append((compsig, coeff_map))
        return corrections

    def _apply_compensation(
        self,
        corrections: list,
        shot: int,
        raw_data: dict,
        identifier: str,
        channel_data: np.ndarray,
        channel_time: np.ndarray,
    ) -> np.ndarray:
        """
        Apply all compensation corrections to one channel's data array.

        Args:
            corrections: Output of _get_comp_corrections(shot)
            shot: Shot number
            raw_data: Fetched MDS+ data dict
            identifier: Channel identifier (e.g. 'MPI11M067' or 'PSF1A')
            channel_data: Raw data array for the channel
            channel_time: Time array for the channel (seconds)
        """
        result = channel_data.copy()
        for compsig, coeff_map in corrections:
            coeff = coeff_map.get(identifier)
            if coeff is None:
                continue
            data_key = Requirement(f'ptdata2("{compsig}",{shot})', shot, None).as_key()
            time_key = Requirement(f'dim_of(ptdata2("{compsig}",{shot}),0)', shot, None).as_key()
            compsig_data = raw_data[data_key]
            compsig_time = raw_data[time_key] / 1000.0
            compsig_interp = np.interp(channel_time, compsig_time, compsig_data)
            result = result - coeff * compsig_interp
        return result

    def _build_specs(self):
        """Build all IDS entry specifications"""

        # Plasma current (Ip) - auxiliary nodes
        self.specs["magnetics._ip_data"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_ip_data_requirements,
            ids_path="magnetics._ip_data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics._ip_time"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_ip_time_requirements,
            ids_path="magnetics._ip_time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics._ip_header"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_ip_header_requirements,
            ids_path="magnetics._ip_header",
            docs_file=self.CONFIG_PATH
        )

        # Diamagnetic flux - auxiliary nodes
        self.specs["magnetics._diamag_data"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_diamag_data_requirements,
            ids_path="magnetics._diamag_data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics._diamag_time"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_diamag_time_requirements,
            ids_path="magnetics._diamag_time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics._diamag_header"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            derive_requirements=self._derive_diamag_header_requirements,
            ids_path="magnetics._diamag_header",
            docs_file=self.CONFIG_PATH
        )

        # User-facing fields - COMPUTED stage
        # Note: ip and diamagnetic_flux are arrays (dimension 0 is measurement index)
        self.specs["magnetics.ip.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._ip_data"],
            compose=self._compose_ip_data,
            ids_path="magnetics.ip.data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.ip.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._ip_time"],
            compose=self._compose_ip_time,
            ids_path="magnetics.ip.time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.ip.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._ip_data", "magnetics._ip_header"],
            compose=self._compose_ip_data_error_upper,
            ids_path="magnetics.ip.data_error_upper",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.diamagnetic_flux.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._diamag_data"],
            compose=self._compose_diamagnetic_flux_data,
            ids_path="magnetics.diamagnetic_flux.data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.diamagnetic_flux.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._diamag_time"],
            compose=self._compose_diamagnetic_flux_time,
            ids_path="magnetics.diamagnetic_flux.time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.diamagnetic_flux.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._diamag_data", "magnetics._diamag_header"],
            compose=self._compose_diamagnetic_flux_data_error_upper,
            ids_path="magnetics.diamagnetic_flux.data_error_upper",
            docs_file=self.CONFIG_PATH
        )

        # Compensation signals - auxiliary nodes (shared by bpol and floop)
        self.specs["magnetics._comp_data"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=[],
            derive_requirements=self._derive_comp_data_requirements,
            ids_path="magnetics._comp_data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics._comp_time"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=[],
            derive_requirements=self._derive_comp_time_requirements,
            ids_path="magnetics._comp_time",
            docs_file=self.CONFIG_PATH
        )

        # b_field_pol_probe - auxiliary nodes (hardware-derived MDSplus requirements)
        self.specs["magnetics._bpol_data"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=[],
            derive_requirements=self._derive_bpol_data_requirements,
            ids_path="magnetics._bpol_data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics._bpol_time"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=[],
            derive_requirements=self._derive_bpol_time_requirements,
            ids_path="magnetics._bpol_time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics._bpol_header"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=[],
            derive_requirements=self._derive_bpol_header_requirements,
            ids_path="magnetics._bpol_header",
            docs_file=self.CONFIG_PATH
        )

        # b_field_pol_probe - hardware geometry (from mhdin_ods.json)
        self.specs["magnetics.b_field_pol_probe.identifier"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, _: np.array([p['identifier'] for p in self._load_probes(shot)]),
            ids_path="magnetics.b_field_pol_probe.identifier",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.b_field_pol_probe.name"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, _: np.array([p['name'] for p in self._load_probes(shot)]),
            ids_path="magnetics.b_field_pol_probe.name",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.b_field_pol_probe.length"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, _: np.array([p['length'] for p in self._load_probes(shot)]),
            ids_path="magnetics.b_field_pol_probe.length",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.b_field_pol_probe.poloidal_angle"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, _: np.array([p['poloidal_angle'] for p in self._load_probes(shot)]),
            ids_path="magnetics.b_field_pol_probe.poloidal_angle",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.b_field_pol_probe.position.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, _: np.array([p['position']['r'] for p in self._load_probes(shot)]),
            ids_path="magnetics.b_field_pol_probe.position.r",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.b_field_pol_probe.position.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, _: np.array([p['position']['z'] for p in self._load_probes(shot)]),
            ids_path="magnetics.b_field_pol_probe.position.z",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.b_field_pol_probe.toroidal_angle"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, _: np.array([p['toroidal_angle'] for p in self._load_probes(shot)]),
            ids_path="magnetics.b_field_pol_probe.toroidal_angle",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.b_field_pol_probe.turns"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, _: np.array([p['turns'] for p in self._load_probes(shot)]),
            ids_path="magnetics.b_field_pol_probe.turns",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.b_field_pol_probe.type.index"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, _: np.array([p['type']['index'] for p in self._load_probes(shot)]),
            ids_path="magnetics.b_field_pol_probe.type.index",
            docs_file=self.CONFIG_PATH
        )

        # b_field_pol_probe - time-series data (from MDSplus)
        self.specs["magnetics.b_field_pol_probe.field.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._bpol_data", "magnetics._bpol_time", "magnetics._comp_data", "magnetics._comp_time"],
            compose=self._compose_bpol_field_data,
            ids_path="magnetics.b_field_pol_probe.field.data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.b_field_pol_probe.field.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._bpol_time"],
            compose=self._compose_bpol_field_time,
            ids_path="magnetics.b_field_pol_probe.field.time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.b_field_pol_probe.field.validity"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._bpol_data"],
            compose=self._compose_bpol_field_validity,
            ids_path="magnetics.b_field_pol_probe.field.validity",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.b_field_pol_probe.field.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._bpol_data", "magnetics._bpol_time", "magnetics._bpol_header", "magnetics._comp_data", "magnetics._comp_time"],
            compose=self._compose_bpol_field_data_error_upper,
            ids_path="magnetics.b_field_pol_probe.field.data_error_upper",
            docs_file=self.CONFIG_PATH
        )

        # flux_loop - auxiliary nodes
        self.specs["magnetics._floop_data"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=[],
            derive_requirements=self._derive_floop_data_requirements,
            ids_path="magnetics._floop_data",
            docs_file=self.CONFIG_PATH
        )

        # flux_loop uses homogeneous_time=True: all loops share one time base.
        # Only one dim_of requirement is needed (from the first loop identifier).
        self.specs["magnetics._floop_time"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=[],
            derive_requirements=self._derive_floop_time_requirements,
            ids_path="magnetics._floop_time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics._floop_header"] = IDSEntrySpec(
            stage=RequirementStage.DERIVED,
            depends_on=[],
            derive_requirements=self._derive_floop_header_requirements,
            ids_path="magnetics._floop_header",
            docs_file=self.CONFIG_PATH
        )

        # flux_loop - hardware geometry (from mhdin_ods.json)
        self.specs["magnetics.flux_loop.identifier"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, _: np.array([f['identifier'] for f in self._load_flux_loops(shot)]),
            ids_path="magnetics.flux_loop.identifier",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.flux_loop.name"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, _: np.array([f['name'] for f in self._load_flux_loops(shot)]),
            ids_path="magnetics.flux_loop.name",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.flux_loop.position.r"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, _: np.array([f['position'][0]['r'] for f in self._load_flux_loops(shot)]),
            ids_path="magnetics.flux_loop.position.r",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.flux_loop.position.z"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, _: np.array([f['position'][0]['z'] for f in self._load_flux_loops(shot)]),
            ids_path="magnetics.flux_loop.position.z",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.flux_loop.type.index"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=[],
            compose=lambda shot, _: np.ones(len(self._load_flux_loops(shot)), dtype=int),
            ids_path="magnetics.flux_loop.type.index",
            docs_file=self.CONFIG_PATH
        )

        # flux_loop - time-series data (from MDSplus)
        self.specs["magnetics.flux_loop.flux.data"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._floop_data", "magnetics._floop_time", "magnetics._comp_data", "magnetics._comp_time"],
            compose=self._compose_floop_flux_data,
            ids_path="magnetics.flux_loop.flux.data",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.flux_loop.flux.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._floop_time"],
            compose=self._compose_floop_flux_time,
            ids_path="magnetics.flux_loop.flux.time",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.flux_loop.flux.validity"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._floop_data"],
            compose=self._compose_floop_flux_validity,
            ids_path="magnetics.flux_loop.flux.validity",
            docs_file=self.CONFIG_PATH
        )

        self.specs["magnetics.flux_loop.flux.data_error_upper"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["magnetics._floop_data", "magnetics._floop_time", "magnetics._floop_header", "magnetics._comp_data", "magnetics._comp_time", "magnetics._ip_data", "magnetics._ip_time"],
            compose=self._compose_floop_flux_data_error_upper,
            ids_path="magnetics.flux_loop.flux.data_error_upper",
            docs_file=self.CONFIG_PATH
        )

    # Requirement derivation functions
    def _derive_ip_data_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for IP data (needs shot number in TDI expression)."""
        return [Requirement(f'ptdata2("IP",{shot})', shot, None)]

    def _derive_ip_time_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for IP time dimension (needs shot number in TDI expression)."""
        return [Requirement(f'dim_of(ptdata2("IP",{shot}),0)', shot, None)]

    def _derive_ip_header_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for IP header (needs shot number in TDI expression)."""
        return [Requirement(f'pthead2("IP",{shot}), __rarray', shot, None)]

    def _derive_diamag_data_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for DIAMAG3 data (needs shot number in TDI expression)."""
        return [Requirement(f'ptdata2("DIAMAG3",{shot})', shot, None)]

    def _derive_diamag_time_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for DIAMAG3 time dimension (needs shot number in TDI expression)."""
        return [Requirement(f'dim_of(ptdata2("DIAMAG3",{shot}),0)', shot, None)]

    def _derive_diamag_header_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive requirements for DIAMAG3 header (needs shot number in TDI expression)."""
        return [Requirement(f'pthead2("DIAMAG3",{shot}), __rarray', shot, None)]

    def _derive_comp_data_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive ptdata2 requirements for all compensation signals (btcomp, ccomp, icomp)."""
        corrections = self._get_comp_corrections(shot)
        seen = set()
        reqs = []
        for compsig, _ in corrections:
            if compsig not in seen:
                seen.add(compsig)
                reqs.append(Requirement(f'ptdata2("{compsig}",{shot})', shot, None))
        return reqs

    def _derive_comp_time_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive dim_of requirements for all compensation signals (btcomp, ccomp, icomp)."""
        corrections = self._get_comp_corrections(shot)
        seen = set()
        reqs = []
        for compsig, _ in corrections:
            if compsig not in seen:
                seen.add(compsig)
                reqs.append(Requirement(f'dim_of(ptdata2("{compsig}",{shot}),0)', shot, None))
        return reqs

    def _derive_bpol_data_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive ptdata2 requirements for all b_field_pol_probe channels."""
        probes = self._load_probes(shot)
        return [Requirement(f'ptdata2("{p["identifier"]}",{shot})', shot, None) for p in probes]

    def _derive_bpol_time_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive dim_of requirements for all b_field_pol_probe channels."""
        probes = self._load_probes(shot)
        return [Requirement(f'dim_of(ptdata2("{p["identifier"]}",{shot}),0)', shot, None) for p in probes]

    def _derive_bpol_header_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive pthead2 requirements for all b_field_pol_probe channels."""
        probes = self._load_probes(shot)
        return [Requirement(f'pthead2("{p["identifier"].upper()}",{shot}), __rarray', shot, None) for p in probes]

    def _derive_floop_data_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive ptdata2 requirements for all flux_loop channels."""
        loops = self._load_flux_loops(shot)
        return [Requirement(f'ptdata2("{f["identifier"]}",{shot})', shot, None) for f in loops]

    def _derive_floop_time_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """
        Derive dim_of requirement for flux_loop time base.

        flux_loop uses homogeneous_time=True in OMAS: all loops share one time axis.
        Only one dim_of is needed, fetched from the first loop identifier.
        """
        loops = self._load_flux_loops(shot)
        return [Requirement(f'dim_of(ptdata2("{f["identifier"]}",{shot}),0)', shot, None) for f in loops]

    def _derive_floop_header_requirements(self, shot: int, _raw_data: dict) -> List[Requirement]:
        """Derive pthead2 requirements for all flux_loop channels."""
        loops = self._load_flux_loops(shot)
        return [Requirement(f'pthead2("{f["identifier"].upper()}",{shot}), __rarray', shot, None) for f in loops]

    # Compose functions
    def _compose_ip_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get plasma current data.

        Returns array of shape (n_measurements, n_time).
        For DIII-D: (1, n_time) - single Ip measurement.
        """
        data_key = Requirement(f'ptdata2("IP",{shot})', shot, None).as_key()
        # Add measurement dimension: (n_time,) -> (1, n_time)
        return raw_data[data_key][np.newaxis, :]

    def _compose_ip_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get plasma current time base with unit conversion.

        From OMAS: dim_of(...,0)/1000. (convert ms to s)
        Returns array of shape (n_measurements, n_time).
        For DIII-D: (1, n_time) - single Ip measurement.
        """
        time_key = Requirement(f'dim_of(ptdata2("IP",{shot}),0)', shot, None).as_key()
        # Add measurement dimension: (n_time,) -> (1, n_time)
        return (raw_data[time_key] / 1000.0)[np.newaxis, :]

    def _compose_ip_data_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compute uncertainty for plasma current.

        From OMAS: abs(header[3] * header[4]) * ones(nt) * 10.0
        where header is from pthead2("IP", shot)
        Returns array of shape (n_measurements, n_time).
        For DIII-D: (1, n_time) - single Ip measurement.
        """
        # Get the data to determine time length
        data_key = Requirement(f'ptdata2("IP",{shot})', shot, None).as_key()
        nt = len(raw_data[data_key])

        # Get header information
        header_key = Requirement(f'pthead2("IP",{shot}), __rarray', shot, None).as_key()
        header = raw_data[header_key]

        # OMAS formula: abs(header[3] * header[4]) * ones(nt) * 10.0
        # Add measurement dimension: (n_time,) -> (1, n_time)
        return (np.abs(header[3] * header[4]) * np.ones(nt) * 10.0)[np.newaxis, :]

    def _compose_diamagnetic_flux_data(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get diamagnetic flux data with unit conversion.

        From OMAS: data * 1e-3 (convert to Weber)
        Returns array of shape (n_measurements, n_time).
        For DIII-D: (1, n_time) - single diamagnetic flux measurement.
        """
        data_key = Requirement(f'ptdata2("DIAMAG3",{shot})', shot, None).as_key()
        # Add measurement dimension: (n_time,) -> (1, n_time)
        return (raw_data[data_key] * 1e-3)[np.newaxis, :]

    def _compose_diamagnetic_flux_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get diamagnetic flux time base with unit conversion.

        From OMAS: dim_of(...,0)/1000. (convert ms to s)
        Returns array of shape (n_measurements, n_time).
        For DIII-D: (1, n_time) - single diamagnetic flux measurement.
        """
        time_key = Requirement(f'dim_of(ptdata2("DIAMAG3",{shot}),0)', shot, None).as_key()
        # Add measurement dimension: (n_time,) -> (1, n_time)
        return (raw_data[time_key] / 1000.0)[np.newaxis, :]

    def _compose_diamagnetic_flux_data_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compute uncertainty for diamagnetic flux.

        From OMAS: abs(header[3] * header[4]) * ones(nt) * 10.0 / 1000.0
        where header is from pthead2("DIAMAG3", shot)
        Returns array of shape (n_measurements, n_time).
        For DIII-D: (1, n_time) - single diamagnetic flux measurement.
        """
        # Get the data to determine time length
        data_key = Requirement(f'ptdata2("DIAMAG3",{shot})', shot, None).as_key()
        nt = len(raw_data[data_key])

        # Get header information
        header_key = Requirement(f'pthead2("DIAMAG3",{shot}), __rarray', shot, None).as_key()
        header = raw_data[header_key]

        # OMAS formula: abs(header[3] * header[4]) * ones(nt) / 100.0
        # Add measurement dimension: (n_time,) -> (1, n_time)
        return (np.abs(header[3] * header[4]) * np.ones(nt) / 100.0)[np.newaxis, :]

    def _compose_bpol_field_data(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Get b_field_pol_probe field data per channel with compensation applied.

        From OMAS: ptdata2("{identifier}", shot) minus compensation corrections
        from btcomp, ccomp, icomp (applied only to valid channels).
        Returns ragged awkward array of shape (n_probes, n_time).
        """
        probes = self._load_probes(shot)
        corrections = self._get_comp_corrections(shot)
        result = []
        for p in probes:
            data_key = Requirement(f'ptdata2("{p["identifier"]}",{shot})', shot, None).as_key()
            time_key = Requirement(f'dim_of(ptdata2("{p["identifier"]}",{shot}),0)', shot, None).as_key()
            data = raw_data[data_key]
            time_s = raw_data[time_key] / 1000.0
            result.append(self._apply_compensation(corrections, shot, raw_data, p['identifier'], data, time_s))
        return ak.Array(result)

    def _compose_bpol_field_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get b_field_pol_probe time base per channel with unit conversion.

        From OMAS: dim_of(...,0) / 1000 (convert ms to s).
        Returns ragged awkward array of shape (n_probes, n_time).
        """
        probes = self._load_probes(shot)
        return ak.Array([
            raw_data[Requirement(f'dim_of(ptdata2("{p["identifier"]}",{shot}),0)', shot, None).as_key()] / 1000.0
            for p in probes
        ])

    def _compose_bpol_field_validity(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get validity flags for b_field_pol_probe channels.

        0 = valid, -2 = invalid (data fetch failed or fitweight < 0.5).
        Returns array of shape (n_probes,).
        """
        probes = self._load_probes(shot)
        weights = self._load_fitweights(shot, 'fwtmp2')
        result = []
        for k, p in enumerate(probes):
            data_key = Requirement(f'ptdata2("{p["identifier"]}",{shot})', shot, None).as_key()
            data_missing = data_key not in raw_data or len(raw_data[data_key]) <= 1
            weight_invalid = k < len(weights) and weights[k] < 0.5
            result.append(-2 if data_missing or weight_invalid else 0)
        return np.array(result, dtype=int)

    def _compose_bpol_field_data_error_upper(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compute uncertainty for b_field_pol_probe channels.

        From OMAS: max(abs(header[3]*header[4]) * ones(nt) * 10.0, 0.03 * abs(compensated_data))
        where header is from pthead2("{identifier}", shot).
        Returns 1e30 array for channels with invalid data or fitweight < 0.5.
        rel_error uses compensated data (matching OMAS which computes uncertainty after compensation).
        Returns ragged awkward array of shape (n_probes, n_time).
        """
        probes = self._load_probes(shot)
        weights = self._load_fitweights(shot, 'fwtmp2')
        corrections = self._get_comp_corrections(shot)
        result = []
        for k, p in enumerate(probes):
            data_key = Requirement(f'ptdata2("{p["identifier"]}",{shot})', shot, None).as_key()
            header_key = Requirement(f'pthead2("{p["identifier"].upper()}",{shot}), __rarray', shot, None).as_key()
            time_key = Requirement(f'dim_of(ptdata2("{p["identifier"]}",{shot}),0)', shot, None).as_key()
            data = raw_data[data_key]
            nt = len(data)
            data_missing = nt <= 1
            weight_invalid = k < len(weights) and weights[k] < 0.5
            if data_missing or weight_invalid:
                result.append(1.e30 * np.ones(nt))
            else:
                time_s = raw_data[time_key] / 1000.0
                compensated = self._apply_compensation(corrections, shot, raw_data, p['identifier'], data, time_s)
                header = raw_data[header_key]
                digi_error = np.abs(header[3] * header[4]) * np.ones(nt) * 10.0
                rel_error = 0.03 * np.abs(compensated)
                result.append(np.fmax(digi_error, rel_error))
        return ak.Array(result)

    def _compose_floop_flux_data(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Get flux_loop flux data per channel with compensation and differential-to-total conversion.

        From OMAS magnetics_floops_data (store_differential=False, nref=0):
          1. Fetch ptdata2 per loop
          2. Apply compensation corrections (btcomp, ccomp, icomp) for valid channels
          3. Convert differential fluxes to total by adding reference loop (index 0) data
        Returns ragged awkward array of shape (n_loops, n_time).
        """
        loops = self._load_flux_loops(shot)
        corrections = self._get_comp_corrections(shot)
        result = []
        all_time = []
        for f in loops:
            data_key = Requirement(f'ptdata2("{f["identifier"]}",{shot})', shot, None).as_key()
            time_key = Requirement(f'dim_of(ptdata2("{f["identifier"]}",{shot}),0)', shot, None).as_key()
            data = raw_data[data_key] * -2 * np.pi # COCOS transformation
            time_s = raw_data[time_key] / 1000.0
            result.append(self._apply_compensation(corrections, shot, raw_data, f['identifier'], data, time_s))
            all_time.append(time_s)

        # Differential-to-total conversion: add reference loop (nref=0) to all other valid loops
        ref_data = np.asarray(result[0])
        for k in range(1, len(loops)):
            data_k = np.asarray(result[k])
            if len(data_k) < 2:
                continue
            elif len(data_k) == len(ref_data):
                result[k] = data_k + ref_data
            else:
                ref_time = np.asarray(all_time[0])
                time_k = np.asarray(all_time[k])
                ref_interp = np.interp(time_k, ref_time, ref_data,
                                       left=0.0, right=0.0)
                result[k] = data_k + ref_interp
        return ak.Array(result)

    def _compose_floop_flux_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get flux_loop time base with unit conversion.

        From OMAS: dim_of(...,0) / 1000 (convert ms to s).
        Returns ragged awkward array of shape (n_loops, n_time).
        """
        loops = self._load_flux_loops(shot)
        return ak.Array([
            raw_data[Requirement(f'dim_of(ptdata2("{f["identifier"]}",{shot}),0)', shot, None).as_key()] / 1000.0
            for f in loops
        ])

    def _compose_floop_flux_validity(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Get validity flags for flux_loop channels.

        0 = valid, -2 = invalid (data fetch failed or fitweight < 0.5).
        Returns array of shape (n_loops,).
        """
        loops = self._load_flux_loops(shot)
        weights = self._load_fitweights(shot, 'fwtsi')
        result = []
        for k, f in enumerate(loops):
            data_key = Requirement(f'ptdata2("{f["identifier"]}",{shot})', shot, None).as_key()
            data_missing = data_key not in raw_data or len(raw_data[data_key]) <= 1
            weight_invalid = k < len(weights) and weights[k] < 0.5
            result.append(-2 if data_missing or weight_invalid else 0)
        return np.array(result, dtype=int)

    def _compose_floop_flux_data_error_upper(self, shot: int, raw_data: dict) -> ak.Array:
        """
        Compute uncertainty for flux_loop channels with differential-to-total error propagation.

        From OMAS magnetics_floops_data (store_differential=False, nref=0):
          1. Per-loop uncertainty: max(digi_error, rel_error, position_error)
               digi_error = 10 * abs(header[3]*header[4]) * ones(nt)
               rel_error  = 0.03 * abs(compensated_data)
               position_error = 1e-9 * position.r * abs(Ip)
          2. Invalid channels (data missing or fitweight < 0.5): 1e30
          3. Propagate reference loop (index 0) uncertainty into all other valid loops:
             data_error_upper[k] = sqrt(uncertainty[k]^2 + ref_uncertainty^2)
        Returns ragged awkward array of shape (n_loops, n_time).
        """
        loops = self._load_flux_loops(shot)
        weights = self._load_fitweights(shot, 'fwtsi')
        corrections = self._get_comp_corrections(shot)

        result = []
        all_time = []
        for k, f in enumerate(loops):
            data_key = Requirement(f'ptdata2("{f["identifier"]}",{shot})', shot, None).as_key()
            time_key = Requirement(f'dim_of(ptdata2("{f["identifier"]}",{shot}),0)', shot, None).as_key()
            header_key = Requirement(f'pthead2("{f["identifier"].upper()}",{shot}), __rarray', shot, None).as_key()
            data = raw_data[data_key] * -2 * np.pi # COCOS transformation
            time_s = raw_data[time_key] / 1000.0
            nt = len(data)
            data_missing = nt <= 1
            weight_invalid = k < len(weights) and weights[k] < 0.5
            if data_missing or weight_invalid:
                result.append(1.e30 * np.ones(nt))
            else:
                compensated = self._apply_compensation(corrections, shot, raw_data, f['identifier'], data, time_s)
                header = raw_data[header_key]
                digi_error = 10 * np.abs(header[3] * header[4]) * np.ones(nt)
                rel_error = 0.03 * np.abs(compensated)
                # Interpolate Ip to flux loop time base for position_error term
                ip_data_key = Requirement(f'ptdata2("IP",{shot})', shot, None).as_key()
                ip_time_key = Requirement(f'dim_of(ptdata2("IP",{shot}),0)', shot, None).as_key()
                Ip = np.interp(time_s, np.ravel(raw_data[ip_time_key]) / 1000.0,
                            np.ravel(raw_data[ip_data_key]), left=0.0, right=0.0)
                position_r = f['position'][0]['r']
                position_error = 1.e-9 * position_r * np.abs(Ip)
                result.append(np.fmax.reduce([digi_error, rel_error, position_error]))
            all_time.append(time_s)

        # Error propagation from differential-to-total conversion: add ref loop uncertainty in quadrature
        ref_uncertainty = np.asarray(result[0])
        ref_time = np.asarray(all_time[0])
        for k in range(1, len(loops)):
            err_k = np.asarray(result[k])
            if len(err_k) < 2:
                continue
            elif len(err_k) == len(ref_uncertainty):
                result[k] = np.sqrt(err_k**2 + ref_uncertainty**2)
            else:
                time_k = np.asarray(all_time[k])
                ref_un_interp = np.interp(time_k, ref_time,
                                          ref_uncertainty, left=0.0, right=0.0)
                result[k] = np.sqrt(err_k**2 + ref_un_interp**2)
        return ak.Array(result)

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
