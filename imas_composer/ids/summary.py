"""
Summary IDS Mapping for DIII-D


Maps the DIII-D one-line shot comment (\\D3D::TOP.COMMENTS:BRIEF) to the IMAS and
global scalar quantities from the DIII-D TRANSPORT MDSplus tree
to the IMAS summary IDS.

The TRANSPORT tree stores time-traced global quantities computed by
transport analysis codes (e.g. TRANSP, ONETWO).  The GLOBAL subtree
contains quantities derived from the full shot analysis, including
confinement times, stored energies, and power balance terms.

Currently implemented:
  - summary.global_quantities.tau_energy.value  ← TRANSPORT.GLOBAL.TIMES.TAUE
  - summary.global_quantities.tau_energy.time   ← dim_of(TRANSPORT.GLOBAL.TIMES.TAUE)
"""

from typing import Dict, List

import numpy as np

from ..core import IDSEntrySpec, Requirement, RequirementStage

from .base import IDSMapper


class SummaryMapper(IDSMapper):
    """Maps DIII-D COMMENTS/TRANSPORT tree data to the IMAS summary IDS."""

    CONFIG_PATH = "summary.yaml"
    DOCS_PATH = "summary.yaml"

    def __init__(self, **kwargs):
        """Initialize Summary mapper."""
        super().__init__()
        self._build_specs()

    def _build_specs(self):
        """Build all IDS entry specifications."""

        # --- internal: fetch TAUE signal from TRANSPORT tree ---
        self.specs["summary._taue"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(r"\TRANSPORT::TOP.GLOBAL.TIMES.TAUE", 0, "TRANSPORT"),
            ],
            ids_path="summary._taue",
            docs_file=self.CONFIG_PATH,
        )
        
        self.specs["summary._taue_time"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(
                    r"dim_of(\TRANSPORT::TOP.GLOBAL.TIMES.TAUE, 0)",
                    0,
                    "TRANSPORT",
                ),
            ],
            ids_path="summary._taue_time",
            docs_file=self.CONFIG_PATH,
        )

        # Internal dependency - fetch the brief shot comment
        self.specs["summary._description"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('\\D3D::TOP.COMMENTS:BRIEF', 0, 'D3D')
            ],
            ids_path="summary._description",
            docs_file=self.DOCS_PATH
        )

        # --- public: summary.global_quantities.tau_energy.value ---
        self.specs["summary.global_quantities.tau_energy.value"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["summary._taue"],
            compose=self._compose_tau_energy_value,
            ids_path="summary.global_quantities.tau_energy.value",
            docs_file=self.CONFIG_PATH,
        )

        # --- public: summary.global_quantities.tau_energy.time ---
        # Depends on both _taue_time (for the raw time array) and _taue (for
        # the valid-point mask, so that value and time stay the same length).
        self.specs["summary.global_quantities.tau_energy.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["summary._taue_time", "summary._taue"],
            compose=self._compose_tau_energy_time,
            ids_path="summary.global_quantities.tau_energy.time",
            docs_file=self.CONFIG_PATH,
        )

        # Public summary IDS field
        self.specs["summary.description"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["summary._description"],
            compose=self._compose_description,
            ids_path="summary.description",
            docs_file=self.DOCS_PATH
        )

    def _compose_tau_energy_value(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose energy confinement time values (seconds).

        TRANSPORT.GLOBAL.TIMES.TAUE is stored in seconds.  Non-positive and
        non-finite values (analysis artifacts or L-mode ramp sentinels) are
        removed; the corresponding time points are also removed by
        _compose_tau_energy_time via the same mask stored on this array.
        """
        key = Requirement(
            r"\TRANSPORT::TOP.GLOBAL.TIMES.TAUE", shot, "TRANSPORT"
        ).as_key()
        tau_e = np.asarray(raw_data[key], dtype=float)
        mask = np.isfinite(tau_e) & (tau_e > 0)
        return tau_e[mask]

    def _compose_tau_energy_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose time base for tau_energy (seconds).

        dim_of(..., 0) returns milliseconds; divide by 1e3 to convert to seconds.
        Applies the same valid-point mask as _compose_tau_energy_value so that
        value and time arrays have the same length.
        """
        time_key = Requirement(
            r"dim_of(\TRANSPORT::TOP.GLOBAL.TIMES.TAUE, 0)",
            shot,
            "TRANSPORT",
        ).as_key()
        value_key = Requirement(
            r"\TRANSPORT::TOP.GLOBAL.TIMES.TAUE", shot, "TRANSPORT"
        ).as_key()
        tau_e = np.asarray(raw_data[value_key], dtype=float)
        mask = np.isfinite(tau_e) & (tau_e > 0)
        t = np.asarray(raw_data[time_key], dtype=float) / 1e3
        return t[mask]

    def _compose_description(self, shot: int, raw_data: dict) -> str:
        """Compose the brief shot comment as a string."""
        key = Requirement('\\D3D::TOP.COMMENTS:BRIEF', shot, 'D3D').as_key()
        value = raw_data[key]
        if isinstance(value, bytes):
            value = value.decode()
        return str(value)


    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
