"""
IMAS Composer - Compose IMAS-compliant data from MDSplus sources.

Public API:
    ImasComposer: Main interface for resolving and composing IDS data
    Requirement: Data requirement specification
"""

from .composer import ImasComposer
from .core import Requirement

__all__ = ['ImasComposer', 'Requirement']
