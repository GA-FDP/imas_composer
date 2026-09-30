"""
IMAS Composer - Compose IMAS-compliant data from MDSplus sources.

Public API:
    ImasComposer: Main interface for resolving and composing IDS data
    Requirement: Data requirement specification
    NoData: Placeholder value for requested data that is not available
    simple_load: Simple utility for loading IDS data in one call (requires OMAS)
"""

from .composer import ImasComposer
from .fetchers import simple_load, fetch_requirements
from .core import Requirement, NoData

__all__ = ['ImasComposer', 'Requirement', 'NoData', 'simple_load', 'fetch_requirements']
