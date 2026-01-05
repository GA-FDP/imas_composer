"""
IMAS Composer - Compose IMAS-compliant data from MDSplus sources.

Public API:
    ImasComposer: Main interface for resolving and composing IDS data
    Requirement: Data requirement specification
    simple_load: Simple utility for loading IDS data in one call (requires OMAS)
"""

from .composer import ImasComposer, simple_load
from .core import Requirement

__all__ = ['ImasComposer', 'Requirement', 'simple_load']
