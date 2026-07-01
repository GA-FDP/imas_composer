"""
Summary IDS Mapping for DIII-D

Maps the DIII-D one-line shot comment (\\D3D::TOP.COMMENTS:BRIEF) to the IMAS
summary.description scalar string.
"""

from typing import Dict

from ..core import RequirementStage, Requirement, IDSEntrySpec
from .base import IDSMapper


class SummaryMapper(IDSMapper):
    """Maps the DIII-D shot comment to the IMAS summary IDS."""

    CONFIG_PATH = "summary.yaml"
    DOCS_PATH = "summary.yaml"

    def __init__(self, **kwargs):
        """Initialize summary mapper."""
        # Initialize base class (loads config, static_values, supported_fields)
        super().__init__()

        # Build IDS specs
        self._build_specs()

    def _build_specs(self):
        """Build all IDS entry specifications"""

        # Internal dependency - fetch the brief shot comment
        self.specs["summary._description"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement('\\D3D::TOP.COMMENTS:BRIEF', 0, 'D3D')
            ],
            ids_path="summary._description",
            docs_file=self.DOCS_PATH
        )

        # Public IDS field
        self.specs["summary.description"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["summary._description"],
            compose=self._compose_description,
            ids_path="summary.description",
            docs_file=self.DOCS_PATH
        )

    def _compose_description(self, shot: int, raw_data: dict) -> str:
        """Compose the brief shot comment as a string."""
        key = Requirement('\\D3D::TOP.COMMENTS:BRIEF', shot, 'D3D').as_key()
        value = raw_data[key]
        if isinstance(value, bytes):
            value = value.decode()
        return str(value)

    def get_specs(self) -> Dict[str, IDSEntrySpec]:
        return self.specs
