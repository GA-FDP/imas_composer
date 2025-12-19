"""
Thomson Scattering IDS Mapper Factory for DIII-D

Factory function to create Thomson scattering mappers.
"""

from .base import IDSMapper
from .thomson_scattering import ThomsonScatteringMapper


def create_thomson_scattering_mapper() -> IDSMapper:
    """
    Factory function to create a Thomson scattering mapper.

    Returns:
        ThomsonScatteringMapper instance

    Examples:
        >>> mapper = create_thomson_scattering_mapper()
        >>> isinstance(mapper, ThomsonScatteringMapper)
        True
    """
    return ThomsonScatteringMapper()
