"""
COCOS (COordinate COnventionS) transformations for fusion data.

This module handles coordinate convention transformations to convert data
from various COCOS systems to COCOS 11, which is used by IMAS.

COCOS Background:
-----------------
COCOS is a standard for coordinate conventions in tokamak physics data.
Different conventions affect the signs and factors of magnetic field and
current quantities. OMAS converts all data to COCOS 11 during data ingestion.

For DIII-D EFIT data:
- Source COCOS: 1, 3, 5, or 7 (determined by signs of Bt and Ip)
- Target COCOS: 11 (IMAS standard)

References:
- OMAS omas_cocos.py
- OMAS omas_physics.py (omas_environment context manager)
- O. Sauter & S.Yu. Medvedev, Comp. Phys. Comm. 184 (2013) 293-302
"""

import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Tuple, Optional


class COCOSTransform:
    """
    Handles COCOS coordinate convention transformations.

    Converts tokamak physics data from arbitrary COCOS to COCOS 11 (IMAS standard).
    """

    # Target COCOS for IMAS
    TARGET_COCOS = 11

    # Transformation types from OMAS
    # These are the valid transform keys used in _cocos_signals
    TRANSFORMS = {
        'PSI': 'PSI',        # Poloidal flux
        'dPSI': 'dPSI',      # Poloidal flux derivative
        '1/PSI': '1/PSI',    # Inverse poloidal flux
        'invPSI': 'invPSI',  # Same as 1/PSI
        'F_FPRIME': 'F_FPRIME',  # F and F' (flux function derivatives)
        'PPRIME': 'PPRIME',  # Pressure derivative
        'Q': 'Q',            # Safety factor
        'TOR': 'TOR',        # Toroidal quantities (Bt, Ip, etc.)
        'BT': 'BT',          # Same as TOR
        'IP': 'IP',          # Same as TOR
        'F': 'F',            # Same as TOR
        'POL': 'POL',        # Poloidal quantities
        'BP': 'BP',          # Same as POL
        None: None,          # No transformation
    }

    def __init__(self):
        """Initialize COCOS transformer."""
        self._cocos_cache: Dict[Tuple[int, int], int] = {}

    def identify_cocos(self, bt: float, ip: float) -> int:
        """
        Identify COCOS convention from Bt and Ip signs.

        For gEQDSK data (like DIII-D EFIT), the COCOS is determined by:
        - Sign of toroidal field (Bt)
        - Sign of plasma current (Ip)

        Args:
            bt: Toroidal magnetic field at magnetic axis (Tesla)
            ip: Plasma current (Amperes)

        Returns:
            COCOS number (1, 3, 5, or 7 for typical cases)

        Note:
            This is the same logic as OMAS MDS_gEQDSK_COCOS_identify
        """
        sign_bt = int(np.sign(bt))
        sign_ip = int(np.sign(ip))

        # Mapping from (sign_Bt, sign_Ip) to COCOS
        # Based on OMAS _common.py line 199
        g_cocos = {
            (+1, +1): 1,
            (+1, -1): 3,
            (-1, +1): 5,
            (-1, -1): 7,
            (+1,  0): 1,  # Zero Ip defaults to +1
            (-1,  0): 3,
        }

        cocos = g_cocos.get((sign_bt, sign_ip), None)
        if cocos is None:
            raise ValueError(
                f"Could not identify COCOS from Bt={bt}, Ip={ip}. "
                f"Sign combination ({sign_bt}, {sign_ip}) not recognized."
            )

        return cocos

    def get_transform_factor(self, source_cocos: int, target_cocos: int,
                            transform_type: str) -> float:
        """
        Calculate transformation factor between two COCOS systems.

        Args:
            source_cocos: Source COCOS number (e.g., 1, 3, 5, 7)
            target_cocos: Target COCOS number (typically 11 for IMAS)
            transform_type: Type of transformation ('PSI', 'TOR', 'POL', etc.)

        Returns:
            Multiplicative factor to convert from source to target COCOS

        Note:
            This implements the transformations defined in OMAS cocos_transform()
            function in omas_physics.py
        """
        if source_cocos == target_cocos:
            return 1.0

        if transform_type is None or transform_type not in self.TRANSFORMS:
            return 1.0

        # Extract COCOS parameters
        source_params = self._decode_cocos(source_cocos)
        target_params = self._decode_cocos(target_cocos)

        # Calculate effective signs and exponents
        sigma_Ip_src, sigma_Bp_src, exp_Bp_src, sigma_rhotp_src = source_params
        sigma_Ip_tgt, sigma_Bp_tgt, exp_Bp_tgt, sigma_rhotp_tgt = target_params

        # Effective signs after transformation (multiply, not divide!)
        # From OMAS cocos_transform: sigma_Ip_eff = cocosin['sigma_RpZ'] * cocosout['sigma_RpZ']
        # NOTE: In OMAS, sigma_B0_eff uses sigma_RpZ (which is sigma_Ip in our notation)
        sigma_Ip_eff = sigma_Ip_src * sigma_Ip_tgt
        sigma_Bp_eff = sigma_Bp_src * sigma_Bp_tgt
        sigma_B0_eff = sigma_Ip_src * sigma_Ip_tgt  # Uses sigma_RpZ (same as sigma_Ip)!
        sigma_rhotp_eff = sigma_rhotp_src * sigma_rhotp_tgt
        # exp_Bp_eff = target - source (not source - target!)
        exp_Bp_eff = exp_Bp_tgt - exp_Bp_src

        # Apply transformation based on type
        # From OMAS omas_physics.py cocos_transform()
        if transform_type in ['PSI']:
            factor = sigma_Ip_eff * sigma_Bp_eff * (2 * np.pi) ** exp_Bp_eff
        elif transform_type in ['1/PSI', 'invPSI', 'dPSI', 'F_FPRIME', 'PPRIME']:
            factor = sigma_Ip_eff * sigma_Bp_eff / (2 * np.pi) ** exp_Bp_eff
        elif transform_type in ['Q']:
            factor = sigma_Ip_eff * sigma_B0_eff * sigma_rhotp_eff
        elif transform_type in ['TOR', 'BT', 'IP', 'F']:
            factor = sigma_B0_eff
        elif transform_type in ['POL', 'BP']:
            factor = sigma_B0_eff * sigma_rhotp_eff
        else:
            factor = 1.0

        return factor

    def _decode_cocos(self, cocos: int) -> Tuple[int, int, int, int]:
        """
        Decode COCOS number into constituent parameters.

        Args:
            cocos: COCOS number (1-16)

        Returns:
            Tuple of (sigma_Ip, sigma_Bp, exp_Bp, sigma_rhotp)

            Note: sigma_B0 is not returned because OMAS uses sigma_RpZ (sigma_Ip)
            for sigma_B0_eff in transformations.

        Note:
            Based on Table 1 in Sauter & Medvedev 2013
        """
        # COCOS 1-8 have exp_Bp = 0, COCOS 11-18 have exp_Bp = 1
        if 1 <= cocos <= 8:
            exp_Bp = 0
            base = cocos
        elif 11 <= cocos <= 18:
            exp_Bp = 1
            base = cocos - 10
        else:
            raise ValueError(f"COCOS {cocos} not in valid range (1-8, 11-18)")

        # Decode base (1-8) into signs
        # sigma_Ip (same as sigma_RpZ): 1 for odd, -1 for even
        sigma_Ip = 1 if base % 2 == 1 else -1

        # sigma_Bp: determined by bits
        sigma_Bp = 1 if base in [1, 2, 5, 6] else -1

        # sigma_rhotp: determined by bits
        sigma_rhotp = 1 if base in [1, 2, 7, 8] else -1

        return (sigma_Ip, sigma_Bp, exp_Bp, sigma_rhotp)

    def transform(self, data: np.ndarray, source_cocos: int,
                 transform_type: str, no_sign: bool = False) -> np.ndarray:
        """
        Transform data from source COCOS to target COCOS (11).

        Args:
            data: Input data array
            source_cocos: Source COCOS number
            transform_type: Type of transformation ('PSI', 'TOR', etc.)

        Returns:
            Transformed data array
        """
        factor = self.get_transform_factor(source_cocos, self.TARGET_COCOS, transform_type)
        if no_sign:
            factor = np.abs(factor)
        return data * factor


# Cache for loaded COCOS mappings
_COCOS_MAP_CACHE: Optional[Dict[str, str]] = None


def _load_cocos_mappings() -> Dict[str, str]:
    """
    Load COCOS transformation mappings from cocos.yaml.

    Returns:
        Dictionary mapping IDS paths to transformation types
    """
    global _COCOS_MAP_CACHE

    if _COCOS_MAP_CACHE is not None:
        return _COCOS_MAP_CACHE

    # Load YAML file from same directory as this module
    cocos_yaml_path = Path(__file__).parent / 'cocos.yaml'

    with open(cocos_yaml_path, 'r') as f:
        cocos_config = yaml.safe_load(f)

    # Flatten the nested YAML structure into dot-notation paths
    # e.g., {'equilibrium': {'time_slice.psi': 'PSI'}} -> {'equilibrium.time_slice.psi': 'PSI'}
    flat_map = {}
    for ids_name, field_map in cocos_config.items():
        for field_path, transform_type in field_map.items():
            full_path = f"{ids_name}.{field_path}"
            flat_map[full_path] = transform_type

    _COCOS_MAP_CACHE = flat_map
    return flat_map


def get_cocos_transform_type(ids_path: str) -> Optional[str]:
    """
    Get COCOS transformation type for a given IDS path.

    Args:
        ids_path: Full IDS path (e.g., 'equilibrium.time_slice.boundary_separatrix.psi')

    Returns:
        Transform type string ('PSI', 'TOR', etc.) or None if no transform needed
    """
    cocos_map = _load_cocos_mappings()
    return cocos_map.get(ids_path, None)
