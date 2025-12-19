"""
Core Profiles IDS Mapper Factory for DIII-D

This module provides a factory function to create the appropriate core profiles mapper
based on the specified tree type (ZIPFIT or OMFIT_PROFS).

Factory Pattern:
    mapper = create_core_profiles_mapper(profiles_tree='ZIPFIT01')
    # Returns CoreProfilesZIPFITMapper instance

    mapper = create_core_profiles_mapper(profiles_tree='OMFIT_PROFS', run_id='001')
    # Returns CoreProfilesOMFITMapper instance
"""

from .base import IDSMapper
from .core_profiles_zipfit import CoreProfilesZIPFITMapper
from .core_profiles_omfit import CoreProfilesOMFITMapper


def create_core_profiles_mapper(profiles_tree: str = 'ZIPFIT01', run_id: str = '001') -> IDSMapper:
    """
    Factory function to create the appropriate core profiles mapper.

    Args:
        profiles_tree: Profile tree identifier (e.g., 'ZIPFIT01', 'ZIPFIT02', 'OMFIT_PROFS')
        run_id: Run ID to append to pulse for OMFIT_PROFS tree (default: '001')

    Returns:
        CoreProfilesZIPFITMapper or CoreProfilesOMFITMapper instance

    Raises:
        ValueError: If tree type is not recognized

    Examples:
        >>> # ZIPFIT tree
        >>> mapper = create_core_profiles_mapper('ZIPFIT01')
        >>> isinstance(mapper, CoreProfilesZIPFITMapper)
        True

        >>> # OMFIT_PROFS tree
        >>> mapper = create_core_profiles_mapper('OMFIT_PROFS', run_id='001')
        >>> isinstance(mapper, CoreProfilesOMFITMapper)
        True
    """
    if 'OMFIT_PROFS' in profiles_tree:
        return CoreProfilesOMFITMapper(omfit_tree=profiles_tree, run_id=run_id)
    elif 'ZIPFIT' in profiles_tree:
        return CoreProfilesZIPFITMapper(zipfit_tree=profiles_tree)
    else:
        raise ValueError(
            f"Unknown profiles tree type: '{profiles_tree}'. "
            f"Expected tree name containing 'ZIPFIT' or 'OMFIT_PROFS'"
        )
