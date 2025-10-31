# Equilibrium IDS Implementation - First 10 Fields

## Summary

Implemented first 10 fields from OMAS `_efit.json` to validate our approach for equilibrium data mapping.

## Implemented Fields

1. **equilibrium.code.name** - EFIT tree name (static)
2. **equilibrium.code.version** - EFIT tree name (static)
3. **equilibrium.ids_properties.homogeneous_time** - Always 1 (static)
4. **equilibrium.time** - Time array in seconds
5. **equilibrium.time_slice.:.time** - Same as equilibrium.time
6. **equilibrium.time_slice.:.boundary.outline.r** - Boundary R coordinates (ragged)
7. **equilibrium.time_slice.:.boundary.outline.z** - Boundary Z coordinates (ragged)
8. **equilibrium.time_slice.:.boundary.x_point.:.r** - X-point R (2 per time)
9. **equilibrium.time_slice.:.boundary.x_point.:.z** - X-point Z (2 per time)

## Architecture

### Mapper Class: `EquilibriumMapper`

```python
class EquilibriumMapper(IDSMapper):
    def __init__(self, efit_tree: str = 'EFIT01'):
        self.efit_tree = efit_tree
        self.geqdsk_node = f'\\{efit_tree}::TOP.RESULTS.GEQDSK'
        self.aeqdsk_node = f'\\{efit_tree}::TOP.RESULTS.AEQDSK'
```

### Internal Dependencies (7 specs)

- `equilibrium._gtime` - GEQDSK.GTIME (time base)
- `equilibrium._rbbbs` - GEQDSK.RBBBS (boundary R)
- `equilibrium._zbbbs` - GEQDSK.ZBBBS (boundary Z)
- `equilibrium._rxpt1` - AEQDSK.RXPT1 (X-point 1 R)
- `equilibrium._zxpt1` - AEQDSK.ZXPT1 (X-point 1 Z)
- `equilibrium._rxpt2` - AEQDSK.RXPT2 (X-point 2 R)
- `equilibrium._zxpt2` - AEQDSK.ZXPT2 (X-point 2 Z)

All DIRECT stage - simple fetch from MDSplus.

### Public Fields (9 specs)

All COMPUTED stage - transform data from internal dependencies.

## Key Implementation Patterns

### 1. NaN Filtering

OMAS uses TDI expression: `py2tdi(nan_where, data, mask, value)`

Our Python implementation:
```python
def _compose_boundary_outline_r(self, shot, raw_data):
    rbbbs = raw_data[rbbbs_key].copy()
    rbbbs[rbbbs == 0] = np.nan  # Replace 0 with NaN
    return rbbbs

def _compose_boundary_outline_z(self, shot, raw_data):
    rbbbs = raw_data[rbbbs_key]
    zbbbs = raw_data[zbbbs_key].copy()
    zbbbs[rbbbs == 0] = np.nan  # Use R as mask
    return zbbbs
```

**Requirement isolation**:
- `outline.r` only needs RBBBS
- `outline.z` needs both RBBBS and ZBBBS (for masking)

### 2. Unit Conversion

```python
def _compose_time(self, shot, raw_data):
    gtime_ms = raw_data[gtime_key]
    return gtime_ms / 1000.0  # ms → seconds
```

### 3. Multi-Dimensional Arrays

X-point coordinates stack two signals:
```python
def _compose_xpoint_r(self, shot, raw_data):
    rxpt1 = raw_data[rxpt1_key].copy()
    rxpt2 = raw_data[rxpt2_key].copy()

    rxpt1[rxpt1 == 0] = np.nan
    rxpt2[rxpt2 == 0] = np.nan

    return np.column_stack([rxpt1, rxpt2])  # Shape: (n_time, 2)
```

## Data Shapes

- **time**: (n_time,)
- **boundary.outline.r/z**: (n_time, n_boundary_points) - ragged, NaN-padded
- **x_point.:.r/z**: (n_time, 2) - fixed size array

## MDSplus Tree Structure

```
EFIT01 (tree)
└── TOP
    └── RESULTS
        ├── GEQDSK (G-EQDSK format)
        │   ├── GTIME    - Time base [ms]
        │   ├── RBBBS    - Boundary R coordinates
        │   └── ZBBBS    - Boundary Z coordinates
        └── AEQDSK (A-EQDSK format)
            ├── RXPT1    - Primary X-point R
            ├── ZXPT1    - Primary X-point Z
            ├── RXPT2    - Secondary X-point R
            └── ZXPT2    - Secondary X-point Z
```

## Testing

Created standard test structure:
- `test_equilibrium_requirements.py` - Requirement resolution test
- `test_equilibrium_composition.py` - OMAS validation test

Both use generic test functions from `conftest.py`.

## Next Steps

Once these 10 fields are validated:
1. Add more boundary fields (strike points, separatrix)
2. Add 1D profiles (pressure, q, psi)
3. Add 2D grid data (psi_grid, coordinate grids)
4. Add global quantities (plasma current, stored energy, etc.)

## Differences from ECE/Thomson

1. **Fixed tree structure**: EFIT tree is simpler than Thomson's system-based structure
2. **Ragged arrays**: Boundary points vary per time slice (similar to Thomson)
3. **2D data**: Will need to handle grid quantities (coming in later fields)
4. **NaN filtering**: Common pattern in equilibrium data (0 means invalid)
5. **COCOS**: Not implemented yet - all DIII-D data should be consistent

## Files Created

- `imas_composer/ids/equilibrium.yaml` - Configuration
- `imas_composer/ids/equilibrium.py` - Mapper implementation
- `tests/test_equilibrium_requirements.py` - Requirements test
- `tests/test_equilibrium_composition.py` - Composition test
- Updated `composer.py` - Registered equilibrium mapper

Total: ~280 lines of implementation code for 9 public fields + 7 internal dependencies.
