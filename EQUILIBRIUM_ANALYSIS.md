# Equilibrium IDS Analysis

Analysis of OMAS equilibrium mappings from `_efit.json` to guide imas_composer implementation.

## Key Observations

### EFIT Tree Variable
- Uses template variable `{EFIT_tree}` throughout (e.g., 'EFIT01', 'EFIT02')
- Similar to how we handle different Thomson systems
- Need to make EFIT tree configurable

### COCOS Detection (Lines 2-6)
```json
"__cocos_rules__": {
  "EFIT_tree": {
    "eval2TDI": "py2tdi(MDS_gEQDSK_COCOS_identify, ...)",
    "PYTHON": "MDS_gEQDSK_COCOS_identify({machine!r}, {pulse}, {EFIT_tree!r}, {EFIT_run_id!r})"
  }
}
```
- COCOS = COordinate COnvention System for tokamaks
- Determines sign conventions based on Bt (toroidal field) and Ip (plasma current)
- Function in `python_tdi.py:31-39`: returns COCOS number (1,3,5,7)
- We'll need to fetch BCENTR and CPASMA to determine COCOS

## First 10 IDS Fields Analysis

### 1-2. Code Metadata (Lines 8-13)
```json
"equilibrium.code.name": {"EVAL": "{EFIT_tree!r}"},
"equilibrium.code.version": {"EVAL": "{EFIT_tree!r}"}
```
**Mapping**:
- Stage: COMPUTED (static based on config)
- Value: Just return the EFIT tree name (e.g., "EFIT01")
- No MDSplus requirements

### 3. Homogeneous Time (Lines 15-17)
```json
"equilibrium.ids_properties.homogeneous_time": {"VALUE": 1}
```
**Mapping**:
- Stage: COMPUTED (static)
- Value: Always 1
- Same pattern as ECE/Thomson

### 4. Time Array (Lines 18-21)
```json
"equilibrium.time": {
  "TDI": "\\{EFIT_tree}::TOP.RESULTS.GEQDSK.GTIME/1000.",
  "treename": "{EFIT_tree}"
}
```
**Mapping**:
- Stage: DIRECT
- MDSplus: `\\{EFIT_tree}::TOP.RESULTS.GEQDSK.GTIME` from EFIT tree
- Transform: Divide by 1000 (convert ms to seconds)
- Pattern: `Requirement(path, shot, efit_tree)`

### 5. Time Slice Time (Lines 22-25)
```json
"equilibrium.time_slice.:.time": {
  "TDI": "\\{EFIT_tree}::TOP.RESULTS.GEQDSK.GTIME/1000.",
  "treename": "{EFIT_tree}"
}
```
**Mapping**:
- Same as `equilibrium.time`
- Note the `.:` indicates array of time slices
- In IMAS, each time slice has its own time value

### 6. Time Slice Count (Lines 26-29)
```json
"equilibrium.time_slice.:": {
  "TDI": "size(\\{EFIT_tree}::TOP.RESULTS.GEQDSK.GTIME)",
  "treename": "{EFIT_tree}"
}
```
**Mapping**:
- Stage: DERIVED (needs GTIME fetched first)
- Derive: `len(raw_data[GTIME_key])`
- Used to determine array dimensions

### 7. Boundary Outline R (Lines 30-34)
```json
"equilibrium.time_slice.:.boundary.outline.r": {
  "NANFILTER": true,
  "eval2TDI": "py2tdi(nan_where,'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.RBBBS','\\{EFIT_tree}::TOP.RESULTS.GEQDSK.RBBBS',0)",
  "treename": "{EFIT_tree}"
}
```
**Mapping**:
- Stage: DIRECT (fetch RBBBS)
- MDSplus: `\\{EFIT_tree}::TOP.RESULTS.GEQDSK.RBBBS`
- Transform: `nan_where(RBBBS, RBBBS, 0)` - replace 0 values with NaN
  - From `python_tdi.py:14-20`: `a[b == n] = np.nan`
  - So: `rbbbs[rbbbs == 0] = np.nan`
- Shape: (n_time, n_boundary_points) - ragged array

### 8. Boundary Outline Z (Lines 35-39)
```json
"equilibrium.time_slice.:.boundary.outline.z": {
  "NANFILTER": true,
  "eval2TDI": "py2tdi(nan_where,'\\{EFIT_tree}::TOP.RESULTS.GEQDSK.ZBBBS','\\{EFIT_tree}::TOP.RESULTS.GEQDSK.RBBBS',0)",
  "treename": "{EFIT_tree}"
}
```
**Mapping**:
- Stage: DIRECT (fetch both ZBBBS and RBBBS)
- MDSplus: `\\{EFIT_tree}::TOP.RESULTS.GEQDSK.ZBBBS` and `RBBBS`
- Transform: `zbbbs[rbbbs == 0] = np.nan`
- Uses RBBBS as the mask (where R==0, set Z to NaN)
- **Note**: Needs both signals, but only for the compose step

### 9. X-Point Count (Lines 40-42)
```json
"equilibrium.time_slice.:.boundary.x_point.:": {"VALUE": 2}
```
**Mapping**:
- Stage: COMPUTED (static)
- Value: Always 2 (DIII-D has up to 2 X-points)
- Array sizing information

### 10. X-Point 0 R Coordinate (Lines 43-47)
```json
"equilibrium.time_slice.:.boundary.x_point.0.r": {
  "NANFILTER": true,
  "eval2TDI": "py2tdi(nan_where,'\\{EFIT_tree}::TOP.RESULTS.AEQDSK.RXPT1','\\{EFIT_tree}::TOP.RESULTS.AEQDSK.RXPT1',0)",
  "treename": "{EFIT_tree}"
}
```
**Mapping**:
- Stage: DIRECT
- MDSplus: `\\{EFIT_tree}::TOP.RESULTS.AEQDSK.RXPT1`
- Transform: `rxpt1[rxpt1 == 0] = np.nan`
- First X-point R coordinate

## Pattern Recognition

### Common Patterns

1. **Static Values**: `{"VALUE": ...}` → COMPUTED stage, return constant
2. **Direct Fetch**: `{"TDI": "...", "treename": "..."}` → DIRECT stage, fetch and optionally transform
3. **Array Sizing**: `{"TDI": "size(...)"}` → DERIVED stage, compute from fetched data
4. **String Evaluation**: `{"EVAL": "..."}` → COMPUTED stage, template substitution

### Transform Patterns

1. **Unit Conversion**: `/1000.` for time (ms → s)
2. **NaN Filtering**: `nan_where(data, mask, value)` → `data[mask == value] = np.nan`
3. **Ensure 2D**: Many fields use `ensure_2d()` or `atleast_2d()`

### Tree Structure

```
EFIT tree structure:
  TOP.RESULTS.GEQDSK.*  - G-EQDSK format outputs (boundary, grid, psi, etc.)
  TOP.RESULTS.AEQDSK.*  - A-EQDSK format outputs (X-points, separatrix, profiles)
```

## Implementation Strategy

### Configuration
Create `equilibrium.yaml`:
```yaml
# Static values
static_values:
  ids_properties.homogeneous_time: 1
  time_slice.boundary.x_point.: 2  # DIII-D has up to 2 X-points

# EFIT tree options
efit_trees:
  - EFIT01
  - EFIT02
  - EFIT03
  # etc.

# Default EFIT tree
default_efit_tree: EFIT01

# Supported fields
fields:
  - code.name
  - code.version
  - ids_properties.homogeneous_time
  - time
  - time_slice.:.time
  - time_slice.:.boundary.outline.r
  - time_slice.:.boundary.outline.z
  - time_slice.:.boundary.x_point.:.r
  - time_slice.:.boundary.x_point.:.z
  # ... more fields
```

### Mapper Structure

```python
class EquilibriumMapper(IDSMapper):
    CONFIG_PATH = "equilibrium.yaml"
    DOCS_PATH = "equilibrium.yaml"

    def __init__(self, efit_tree: str = 'EFIT01'):
        self.efit_tree = efit_tree
        self.geqdsk_node = f'\\{efit_tree}::TOP.RESULTS.GEQDSK'
        self.aeqdsk_node = f'\\{efit_tree}::TOP.RESULTS.AEQDSK'
        super().__init__()
        self._build_specs()

    def _build_specs(self):
        # Time base
        self.specs["equilibrium._gtime"] = IDSEntrySpec(
            stage=RequirementStage.DIRECT,
            static_requirements=[
                Requirement(f'{self.geqdsk_node}.GTIME', 0, self.efit_tree)
            ]
        )

        # Time array (convert ms to s)
        self.specs["equilibrium.time"] = IDSEntrySpec(
            stage=RequirementStage.COMPUTED,
            depends_on=["equilibrium._gtime"],
            compose=lambda shot, raw: self._compose_time(shot, raw)
        )

    def _compose_time(self, shot, raw_data):
        gtime_key = Requirement(f'{self.geqdsk_node}.GTIME', shot, self.efit_tree).as_key()
        gtime_ms = raw_data[gtime_key]
        return gtime_ms / 1000.0  # Convert ms to seconds
```

### Helper Functions

We'll need Python equivalents of the TDI functions:

```python
def nan_where(data, mask, value):
    """Replace values in data where mask equals value with NaN."""
    result = data.copy()
    result[mask == value] = np.nan
    return result

def ensure_2d(data):
    """Ensure data is at least 2D."""
    if data.ndim < 2:
        return np.atleast_2d(data)
    return data
```

## Key Differences from ECE/Thomson

1. **Template Variables**: EFIT tree is configurable (like Thomson systems but more extensive)
2. **More Complex Transforms**: Interpolation, psi calculations, COCOS conversions
3. **Interdependencies**: Many fields reference same base signals (RBBBS, ZBBBS, etc.)
4. **2D Data**: Grid quantities (psi, profiles) are 2D arrays
5. **Ragged Arrays**: Boundary points vary per time slice

## Next Steps

1. Create `equilibrium.yaml` configuration
2. Implement `EquilibriumMapper` class
3. Start with simple fields (metadata, time arrays)
4. Add boundary outline fields (demonstrates nan_where pattern)
5. Test requirement isolation (RBBBS used by both outline.r and outline.z)
6. Gradually add more complex fields (psi grid, profiles, etc.)

## Questions for User

1. Which EFIT tree should be default? (EFIT01, EFIT02, ...)
2. Do we need to support multiple EFIT trees simultaneously?
3. Should COCOS detection be part of requirements or just for validation?
