"""
IMAS Composer - Compose IMAS-compliant data from MDSplus sources.

Public API:
    ImasComposer: Main interface for resolving and composing IDS data
    Requirement: Data requirement specification
    simple_load: Simple utility for loading IDS data in one call (requires OMAS)
"""

from .composer import ImasComposer
from .fetchers import simple_load, fetch_requirements
from .core import Requirement
from . import _version
__version__ = _version.get_versions()['version']

__all__ = ['ImasComposer', 'Requirement', 'simple_load', 'fetch_requirements']
