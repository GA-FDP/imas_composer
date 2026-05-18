# Implementing New IDS Fields

This guide covers the workflow for implementing new IDS fields by transcribing OMAS mappings to imas_composer.

## Overview

The implementation workflow typically happens in a separate session from testing. You analyze OMAS mappings, create mapper code and YAML config, then move to testing in a later session.

## Step 1: Analyze OMAS Mapping

OMAS uses two mapping styles:

### JSON Mappings (Equilibrium fields)

Located in `omas/machine_mappings/_efit.json`:

```json
{
  "equilibrium.time_slice.profiles_1d.psi": {
    "COMMENT": "Poloidal flux",
    "mdsvalue": "\\EFIT::TOP.RESULTS.GEQDSK:PSIRZ",
    "TRANSPOSE": [2, 1, 0]
  }
}
```

**Key patterns:**
- `mdsvalue`: MDSplus path → maps to DIRECT requirement
- `TRANSPOSE`: Dimension reordering (watch for Fortran column-major)
- `python_tdi`: Custom TDI functions → translate to compose functions
- Static values: Use COMPUTED stage with constant returns

### Python Mappings (Diagnostic fields)

Located in `omas/machine_mappings/d3d.py`:

```python
def thomson_scattering_data(ods, toplevel, shot):
    # Static query for known paths
    query = [
        '\\TS_BEST:R',
        '\\TS_BEST:Z',
    ]
    data = mdsvalue(device, treename, shot, query).raw()

    # Loop over channels - becomes DERIVED requirements
    for k in range(len(data['R'])):
        ods[f'channel.{k}.position.r'] = data['R'][k]
        ods[f'channel.{k}.position.z'] = data['Z'][k]
```

**Key patterns:**
- `query` list → maps to DIRECT requirements
- Loops (`for k in range(...)`) → maps to DERIVED requirements
- `unumpy.uarray(nominal, error)` → separate `data` and `data_error_upper` fields
- Unit conversions (e.g., `/1e3` for ms→s) → apply in compose functions
- Calibration shots → DERIVED stage, document with `allow_different_shot` in test config

## Step 2: Create Auxiliary Nodes

Auxiliary nodes store raw MDSplus data before final composition. Name them with leading underscore.

### For DIRECT Stage (Static Paths)

```python
self.specs["_position_r"] = IDSEntrySpec(
    stage=RequirementStage.DIRECT,
    static_requirements=[Requirement('\\TS_BEST:R', 0, 'ELECTRONS')]
)
```

### For DERIVED Stage (Dynamic Paths)

```python
self.specs["_channel_ids"] = IDSEntrySpec(
    stage=RequirementStage.DERIVED,
    depends_on=["_hwmap"],  # Needs hwmap data first
    derive_requirements=lambda shot, raw: [
        Requirement(f'\\TREE::CHANNEL_{ch}', shot, 'TREE')
        for ch in get_channel_ids(raw)
    ]
)
```

**Dependency Chain:**
- Use `depends_on` to specify what data must be fetched first
- Can have multi-level dependencies (A depends on B, B depends on C)
- System automatically resolves dependencies in correct order

## Step 3: Create Compose Functions

Compose functions transform raw MDSplus data into final IDS values.

### Simple Transform

```python
def _compose_position_r(self, shot: int, raw_data: dict) -> np.ndarray:
    """R coordinate of measurement position."""
    return raw_data[('\\TS_BEST:R', shot, 'ELECTRONS')]
```

### With Unit Conversion

```python
def _compose_time(self, shot: int, raw_data: dict) -> np.ndarray:
    """Time in seconds (MDSplus stores milliseconds)."""
    time_ms = raw_data[('\\TS_BEST:TIME', shot, 'ELECTRONS')]
    return time_ms / 1000.0  # Convert ms → s
```

### Handling Uncertainty

```python
def _compose_n_e_data(self, shot: int, raw_data: dict) -> np.ndarray:
    """Electron density nominal values."""
    return raw_data[('\\TS_BEST:NE', shot, 'ELECTRONS')]

def _compose_n_e_data_error_upper(self, shot: int, raw_data: dict) -> np.ndarray:
    """Electron density uncertainty."""
    return raw_data[('\\TS_BEST:NE_ERR', shot, 'ELECTRONS')]
```

**Pattern:** Separate compose functions for `data` and `data_error_upper`.

### Array Construction (from loops)

```python
def _compose_channel_positions_r(self, shot: int, raw_data: dict) -> np.ndarray:
    """R coordinates for all channels."""
    # OMAS loops over channels - we build array
    positions = []
    for k in range(self._get_n_channels(shot, raw_data)):
        r_key = (f'\\TS_BEST:R[{k}]', shot, 'ELECTRONS')
        positions.append(raw_data[r_key])
    return np.array(positions)
```

### COCOS Transformations (Equilibrium Only)

```python
def _compose_psi_axis(self, shot: int, raw_data: dict) -> np.ndarray:
    """Poloidal flux at magnetic axis with COCOS transformation."""
    ssimag = raw_data[('\\EFIT::SSIMAG', shot, self.efit_tree)]
    return self._apply_cocos_transform(
        ssimag, shot, raw_data,
        "equilibrium.time_slice.global_quantities.psi_axis"
    )
```

COCOS mappings defined in `cocos.py`:
```python
IDS_COCOS_MAP = {
    'equilibrium.time_slice.global_quantities.psi_axis': 'PSI',
    'equilibrium.time_slice.global_quantities.ip': 'TOR',
    'equilibrium.time_slice.profiles_1d.q': 'Q',
}
```

### Ragged Arrays (Variable Length per Time Slice)

```python
def _compose_boundary_outline_r(self, shot: int, raw_data: dict) -> ak.Array:
    """Boundary outline R (ragged - variable points per time slice)."""
    rbbbs = raw_data[('\\EFIT::RBBBS', shot, self.efit_tree)]
    mask = rbbbs != 0  # Filter zero padding
    return filter_padding(rbbbs, mask)  # Returns awkward array
```

**Pattern:** Use `awkward` arrays for ragged data, filter zero padding.

## Step 4: Register COMPUTED Specs

Final IDS fields are COMPUTED stage - they depend on auxiliary nodes and have compose functions.

```python
self.specs["thomson_scattering.channel.position.r"] = IDSEntrySpec(
    stage=RequirementStage.COMPUTED,
    depends_on=["_position_r"],  # Auxiliary node
    compose=self._compose_position_r
)
```

**Naming:**
- Auxiliary nodes: `_position_r` (leading underscore)
- Final fields: `thomson_scattering.channel.position.r` (full IDS path)

## Step 5: Update YAML Configuration

Add field to `ids/<ids_name>.yaml`:

```yaml
fields:
  - thomson_scattering.ids_properties.homogeneous_time
  - thomson_scattering.channel.name
  - thomson_scattering.channel.position.r
  - thomson_scattering.channel.position.z
  - thomson_scattering.channel.n_e.data
  - thomson_scattering.channel.n_e.data_error_upper
```

YAML files define which fields are supported. Tests use this list for parametric testing.

## Common Patterns

### Calibration Data (Different Shot Numbers)

Some fields use calibration data from a different shot:

```python
def _derive_hwmap_requirements(self, shot: int, raw_data: dict) -> List[Requirement]:
    """Hardware map uses calibration shot, not requested shot."""
    calib_shot = raw_data[('\\TS_CALIB_SHOT', shot, 'ELECTRONS')]
    return [Requirement('\\TS_HWMAP', calib_shot, 'ELECTRONS')]
```

**Important:** Document this in test config YAML (see TESTING_GUIDE.md).

### Channel Loops → Array Dimensions

OMAS loops over channels sequentially:
```python
for k in range(n_channels):
    ods[f'channel.{k}.position.r'] = data[k]
```

imas_composer uses array dimensions:
```python
# Returns shape (n_channels,)
return np.array([data[k] for k in range(n_channels)])
```

### Coordinate Transformations

Match OMAS transformations exactly:
```python
# OMAS: phi_out = -phi_in * np.pi / 180
phi_deg = raw_data[phi_key]
phi_rad = -phi_deg * np.pi / 180  # Degrees → radians, sign flip
```

### Fortran vs C Ordering (Transpose)

OMAS `TRANSPOSE` operations often fix Fortran column-major ordering:
```python
# MDSplus returns (65, 65, 100) - Fortran order
# Need (100, 65, 65) - time first
psi_2d = raw_data[psi_key]
return np.transpose(psi_2d, (2, 1, 0))  # Matches OMAS TRANSPOSE
```

## Adding a New IDS

To add support for a complete new IDS:

1. **Create mapper class:** `ids/<ids_name>.py`
   ```python
   from .base import IDSMapper

   class MagneticsMapper(IDSMapper):
       def __init__(self, **kwargs):
           super().__init__()
           self._register_fields()

       def _register_fields(self):
           # Add specs here
           pass
   ```

2. **Create YAML config:** `ids/<ids_name>.yaml`
   ```yaml
   fields:
     - magnetics.ids_properties.homogeneous_time
     - magnetics.bpol_probe.field.data
   ```

3. **Register in factory:** `ids/ids_factory.py`
   ```python
   if ids_name == 'magnetics':
       return MagneticsMapper(**kwargs)
   ```

4. **Create tests** (see TESTING_GUIDE.md)

## Tips for Successful Implementation

1. **Start with simple fields** - time arrays, coordinates, metadata
2. **Match OMAS exactly** - same units, same transformations, same ordering
3. **Use auxiliary nodes** - keep raw data separate from computed fields
4. **Document special cases** - calibration shots, coordinate transforms, etc.
5. **One field at a time** - easier to debug when tests fail
6. **Check OMAS reference** - when stuck, read the OMAS implementation

## Next Steps

After implementing fields, move to testing (see TESTING_GUIDE.md).
