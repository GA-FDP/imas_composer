"""
EC Launchers IDS Mapper Factory for DIII-D

Factory function to create EC launchers mappers.
"""

from .base import IDSMapper
from .ec_launchers import ECLaunchersMapper


def create_ec_launchers_mapper() -> IDSMapper:
    """
    Factory function to create an EC launchers mapper.

    Returns:
        ECLaunchersMapper instance

    Examples:
        >>> mapper = create_ec_launchers_mapper()
        >>> isinstance(mapper, ECLaunchersMapper)
        True
    """
    return ECLaunchersMapper()
