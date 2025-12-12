# IMAS Composer

A Python library for converting DIII-D MDSplus data to IMAS (ITER Integrated Modelling & Analysis Suite) format.

## Overview

IMAS Composer provides a clean, declarative API for mapping DIII-D diagnostic and analysis data (stored in MDSplus) to standardized IMAS v3.41.0 data structures (IDS - Interface Data Structures). It handles:

- **Requirement identification**: Determining what MDSplus data needs to be fetched
- **Data transformation**: Converting units, coordinate systems, and data layouts
- **COCOS conversions**: Handling magnetic coordinate conventions
- **Composition**: Assembling final IMAS-compliant data structures

## VSCode Setup

**Requirements:**
- Python extension for VSCode
- Python environment with `pytest` and `omas` (installed from the `omas_mds_convert_to_64bit` branch)
- MDSplus access configured (connection to DIII-D data servers)

**Test Explorer Setup:**
1. Open VSCode settings (Cmd/Ctrl + ,)
2. Search for "Python Testing"
3. Enable: `Python › Testing: Pytest Enabled`
4. Set `Python › Testing: Pytest Args` to include `-v` and `-m` markers if desired

**Running Tests:**
- Click the flask/beaker icon in the left sidebar
- Navigate the test tree to find specific tests
- Click ▶️ next to any test to run it
- View output in the "Test Results" panel

**Note:** All tests require MDSplus access to DIII-D data servers.

## Quick Start

```python
from imas_composer import ImasComposer

# Create composer instance
composer = ImasComposer()

# For equilibrium with specific EFIT tree
composer = ImasComposer(efit_tree='EFIT01')

# Resolve requirements iteratively (batch API)
shot = 200000
ids_paths = ['equilibrium.time', 'equilibrium.time_slice.profiles_1d.psi']
raw_data = {}

while True:
    status, requirements = composer.resolve(ids_paths, shot, raw_data)
    if all(status.values()):
        break

    # Fetch requirements from MDSplus
    for req in requirements:
        mds_data = fetch_from_mdsplus(req)  # Your fetching logic
        raw_data[req.as_key()] = mds_data

# Compose final IMAS data for all paths at once
results = composer.compose(ids_paths, shot, raw_data)
# results = {'equilibrium.time': array(...), 'equilibrium.time_slice.profiles_1d.psi': array(...)}
```

## Working with Claude Code to Add New Fields

This project is designed to be extended by working with Claude Code (AI assistant). The `.claude/` directory contains comprehensive documentation that Claude uses to understand the project architecture.

### Starting a Session with Claude

At the beginning of each Claude Code session, tell Claude:

```
Read `.claude/README.md` to understand the project context
```

This gives Claude access to:
- Development principles and architecture
- Data privacy policies (never execute code - data is restricted)
- Three-stage requirement system
- Testing patterns and configuration
- IDS-specific implementation guides

### Adding New Fields: Quick Workflow

You know OMAS and DIII-D MDSplus data structures. Here's the direct workflow for working with Claude:

#### 1. Tell Claude what to implement

OMAS uses two mapping styles - point Claude to the right one:

**Equilibrium fields** (JSON mapping):
```
Implement equilibrium.time_slice.profiles_2d.psi from omas/machine_mappings/_efit.json
```
or
```
Implement these equilibrium.time_slice.profiles_1d fields from _efit.json:
dpressure_dpsi, f, f_df_dpsi, pressure, psi, q, rho_tor_norm, j_tor, volume
```

**Diagnostic fields** (Python mapping):
```
Implement thomson_scattering.channel.n_e.data from the thomson_scattering_data
function in omas/machine_mappings/d3d.py
```
or
```
Implement thomson_scattering.channel fields from thomson_scattering_data in d3d.py:
position.r, position.z, position.phi, n_e.data, n_e.data_error_upper, t_e.data, t_e.data_error_upper
```

Claude will read the OMAS implementation and create:
- Auxiliary nodes for raw MDSplus data
- `IDSEntrySpec` with proper stage (DIRECT/DERIVED/COMPUTED)
- Compose functions with transformations
- YAML field list entries
- Test configuration mappings

#### 2. Test and iterate

Use VSCode Test Explorer to run tests (see [VSCode Setup](#vscode-setup) above):
- Click the test flask icon in VSCode sidebar
- Navigate to the specific test
- Click ▶️ to run

Or use pytest directly:
```bash
pytest tests/test_equilibrium_composition.py::test_composition_matches_omas[equilibrium.time_slice.profiles_2d.psi] -v
```

When tests fail, share the output with Claude:
```
Test failed: shapes don't match. Composer gives (100, 65, 65) but values differ by 1e-3.
```

Claude will debug common issues:
- **Dimension mismatches**: Fortran column-major ordering, missing transposes
- **COCOS transformations**: Missing or incorrect coordinate conversions
- **Precision differences**: Interpolation methods, tolerance adjustments
- **Ragged arrays**: Zero-padding, awkward array conversion

**Key patterns Claude handles automatically:**

For **d3d.py mappings**:
- **unumpy arrays**: `unumpy.uarray(nominal, error)` → separate `data` and `data_error_upper` fields
- **Calibration shots**: DERIVED stage for calibration shot number, document with `allow_different_shot` in test config
- **Unit conversions**: Apply same transformations (e.g., `/1e3` for ms→s, `-angle * pi/180` for coordinate conversions)
- **Loop structures**: Map channel loops to array indices in IMAS

For **JSON mappings**:
- **python_tdi functions**: Translate OMAS helper functions to compose functions
- **TRANSPOSE operations**: Handle dimension reordering (watch for Fortran column-major)
- **COCOS**: Add transformations to cocos.py map
- **Static values**: Use COMPUTED stage with constant returns

### Common Scenarios

**Precision issues after implementation:**
```
Test passes but values differ by 1e-3. The interpolation might need adjustments.
```
Claude will check: interpolation method (linear vs cubic), extrapolation mode, COCOS factors.

**Shape mismatches:**
```
Shapes don't match: composer has (65, 65, 100) but OMAS has (100, 65, 65)
```
Claude will fix: transpose operations, Fortran vs C ordering, time axis location.

**Different shot numbers in requirements:**
```
Test fails: requirement has shot 123456 but expected 200000
```
Claude will add: DERIVED stage for calibration data, `allow_different_shot` in test config.

**Ragged array handling:**
```
Boundary data has variable length per time slice with zero padding
```
Claude will implement: zero filtering with awkward arrays.

### What Claude WON'T Do

Due to data privacy restrictions:
- ❌ Execute Python code or run pytest
- ❌ Access MDSplus servers
- ❌ Import modules

Claude will:
- ✅ Read and write code
- ✅ Debug from error messages you provide
- ✅ Use static analysis tools

## Architecture Overview

### Three-Stage Requirement System

1. **DIRECT**: Static MDSplus paths known at initialization
   ```python
   static_requirements=[Requirement('\\EFIT::TOP.RESULTS.GEQDSK.BCENTR', 0, 'EFIT01')]
   ```

2. **DERIVED**: Requirements that depend on fetched data
   ```python
   derive_requirements=lambda shot, raw: [
       Requirement(f'\\TREE::PATH.{channel_id}', shot, 'TREE')
       for channel_id in get_channel_ids(raw)
   ]
   ```

3. **COMPUTED**: Final composition from raw_data
   ```python
   compose=lambda shot, raw: transform(raw[req_key])
   ```

### File Structure

```
imas_composer/
├── imas_composer/
│   ├── core.py              # Core requirement system
│   ├── composer.py          # Public ImasComposer API
│   ├── cocos.py             # COCOS transformations
│   └── ids/
│       ├── base.py          # IDSMapper base class
│       ├── equilibrium.py   # Equilibrium IDS mapper
│       ├── equilibrium.yaml # Field list
│       └── ...
├── tests/
│   ├── conftest.py          # Shared test infrastructure
│   ├── test_equilibrium_requirements.py
│   ├── test_equilibrium_composition.py
│   ├── test_config_equilibrium.yaml  # Test configuration
│   └── ...
└── .claude/                 # Claude Code documentation
    ├── README.md            # Index of documentation
    ├── .claudecontext       # Critical policies & quick reference
    ├── DEVELOPMENT_PRINCIPLES.md  # Architecture guide
    └── ...
```

### Key Principles

1. **DRY (Don't Repeat Yourself)**: Common logic lives in shared utilities
2. **Requirement Isolation**: Each field fetches only what it needs
3. **Declarative Configuration**: Use YAML for field lists and static values
4. **Test-Driven**: Every field has requirement and composition tests
5. **OMAS Compatibility**: Output exactly matches OMAS reference implementation

## Advanced Topics

### COCOS Transformations

Equilibrium fields require coordinate convention conversions (COCOS):

```python
# In equilibrium.py
def _compose_psi_axis(self, shot: int, raw_data: dict) -> np.ndarray:
    ssimag = raw_data[ssimag_key]
    return self._apply_cocos_transform(ssimag, shot, raw_data, "equilibrium.time_slice.global_quantities.psi_axis")
```

COCOS mappings are defined in `cocos.py`:
```python
IDS_COCOS_MAP = {
    'equilibrium.time_slice.global_quantities.psi_axis': 'PSI',
    'equilibrium.time_slice.global_quantities.ip': 'TOR',
    'equilibrium.time_slice.profiles_1d.q': 'Q',
}
```

### Ragged Arrays

Some fields have variable length per time slice (e.g., boundary outline):

```python
def _compose_boundary_outline_r(self, shot: int, raw_data: dict) -> ak.Array:
    rbbbs = raw_data[rbbbs_key]
    mask = rbbbs != 0
    return filter_padding(rbbbs, mask)  # Returns awkward array
```

### Field-Specific Test Tolerances

For fields with expected precision differences:

```yaml
# test_config_equilibrium.yaml
field_tolerances:
  equilibrium.time_slice.profiles_1d.j_tor:
    rtol: 1.0e-5
    atol: 1.0
```

## Contributing

When contributing new IDS mappers or fields:

1. Follow the three-stage requirement pattern
2. Write compose functions that match OMAS transformations exactly
3. Add fields to the YAML configuration
4. Include both requirement and composition tests
5. Document any special cases or transformations
6. Consider working with Claude Code for guidance on architecture

## References

- **IMAS Documentation**: https://www.iter.org/
- **OMAS Repository**: https://github.com/gafusion/omas
- **DIII-D MDSplus Documentation**: Internal DIII-D documentation https://nomos.gat.com/DIII-D/documentation/
- **Claude Code Context**: See `.claude/README.md` for development guides

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or contributions, please open an issue on the project repository.
