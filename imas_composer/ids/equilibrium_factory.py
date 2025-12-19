"""
Equilibrium IDS Mapper Factory for DIII-D

Factory function to create equilibrium mappers for different EFIT tree sources.
"""

from .base import IDSMapper
from .equilibrium import EquilibriumMapper


def create_equilibrium_mapper(efit_tree: str = 'EFIT01') -> IDSMapper:
    """
    Factory function to create an equilibrium mapper.

    Args:
        efit_tree: EFIT tree identifier (e.g., 'EFIT01', 'EFIT02', 'EFIT03')

    Returns:
        EquilibriumMapper instance configured for the specified EFIT tree

    Examples:
        >>> mapper = create_equilibrium_mapper('EFIT01')
        >>> isinstance(mapper, EquilibriumMapper)
        True

        >>> mapper = create_equilibrium_mapper('EFIT02')
        >>> mapper.efit_tree
        'EFIT02'
    """
    return EquilibriumMapper(efit_tree=efit_tree)
