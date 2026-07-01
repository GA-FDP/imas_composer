"""
Summary IDS Mapping for DIII-D

Maps global scalar quantities from the DIII-D TRANSPORT MDSplus tree
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
    """Maps DIII-D TRANSPORT tree data to the IMAS summary IDS."""

    CONFIG_PATH = "summary.yaml"

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
                    r"dim_of(\TRANSPORT::TOP.GLOBAL.TIMES.TAUE, 0) / 1e3",
                    0,
                    "TRANSPORT",
                ),
            ],
            ids_path="summary._taue_time",
            docs_file=self.CONFIG_PATH,
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
        self.specs["summary.global_quantities.tau_energy.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["summary._taue_time"],
            compose=self._compose_tau_energy_time,
            ids_path="summary.global_quantities.tau_energy.time",
            docs_file=self.CONFIG_PATH,
        )

    def _compose_tau_energy_value(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose energy confinement time values (seconds).

        TRANSPORT.GLOBAL.TIMES.TAUE is stored in seconds.
        """
        key = Requirement(
            r"\TRANSPORT::TOP.GLOBAL.TIMES.TAUE", shot, "TRANSPORT"
        ).as_key()
        return np.asarray(raw_data[key], dtype=float)

    def _compose_tau_energy_time(self, shot: int, raw_data: dict) -> np.ndarray:
        """
        Compose time base for tau_energy (seconds).

        dim_of(..., 0) returns milliseconds; divide by 1e3 to convert to seconds.
        """
        key = Requirement(
            r"dim_of(\TRANSPORT::TOP.GLOBAL.TIMES.TAUE, 0) / 1e3",
            shot,
            "TRANSPORT",
        ).as_key()
        return np.asarray(raw_data[key], dtype=float)

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
