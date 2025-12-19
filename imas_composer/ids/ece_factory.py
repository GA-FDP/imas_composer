"""
ECE IDS Mapper Factory for DIII-D

Factory function to create electron cyclotron emission mappers.
"""

from .base import IDSMapper
from .ece import ElectronCyclotronEmissionMapper


def create_ece_mapper(fast_ece: bool = False) -> IDSMapper:
    """
    Factory function to create an ECE mapper.

    Args:
        fast_ece: If True, use high-frequency sampling data (TECEF nodes).
                 If False, use standard ECE data (TECESM nodes). Default: False

    Returns:
        ElectronCyclotronEmissionMapper instance

    Examples:
        >>> mapper = create_ece_mapper(fast_ece=False)
        >>> isinstance(mapper, ElectronCyclotronEmissionMapper)
        True

        >>> mapper = create_ece_mapper(fast_ece=True)
        >>> # Uses TECEF nodes for high-frequency sampling
    """
    return ElectronCyclotronEmissionMapper(fast_ece=fast_ece)
