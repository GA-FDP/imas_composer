# IMAS Composer Development Principles

This document outlines the core principles and patterns for developing the IMAS Composer project.

## Core Principle: DRY (Don't Repeat Yourself)

**Dryness is a key principle for this project.** Code should NEVER be duplicated. Common patterns must be extracted and shared.

### Application in Practice

1. **Tests**: All test logic lives in `tests/conftest.py` as reusable functions
   - Generic test functions work across all IDS types
   - IDS-specific test files are minimal (~17 lines each)
   - Adding a new IDS requires only 2 small test files

2. **Mappers**: Common mapper functionality lives in base classes
   - `IDSMapper` base class in `ids/base.py` handles common patterns
   - Subclasses only implement IDS-specific logic

3. **Configuration**: Use YAML files, not hardcoded values
   - Field lists loaded from `ids/<ids_name>.yaml`
   - Static values defined in YAML, not in code
   - Tests use `load_ids_fields()` to get field lists dynamically

4. **Utilities**: Shared utilities in `conftest.py` or appropriate modules
   - `resolve_and_compose()` - full resolve/fetch/compose cycle
   - `get_omas_value()` - navigate OMAS structures
   - `compare_values()` - type-aware value comparison
   - `test_requirements_resolution()` - generic requirement testing
   - `test_composition_against_omas()` - generic OMAS validation

## Requirement Isolation Principle

**Each IDS field should only fetch exactly what it needs.** Never bundle requirements together.

### Anti-Pattern: Bundled Requirements

```python
# BAD - bundles R, Z, PHI together
self.specs["_geometry_setup"] = IDSEntrySpec(
    stage=RequirementStage.DIRECT,
    static_requirements=[
        Requirement('...R', 0, 'ELECTRONS'),
        Requirement('...Z', 0, 'ELECTRONS'),
        Requirement('...PHI', 0, 'ELECTRONS'),
    ]
)
```

### Correct Pattern: Isolated Requirements

```python
# GOOD - separate spec for each coordinate
self.specs["_position_r"] = IDSEntrySpec(
    stage=RequirementStage.DIRECT,
    static_requirements=[Requirement('...R', 0, 'ELECTRONS')]
)

self.specs["_position_z"] = IDSEntrySpec(
    stage=RequirementStage.DIRECT,
    static_requirements=[Requirement('...Z', 0, 'ELECTRONS')]
)
```

**Rationale**: When requesting `channel.position.r`, we should ONLY fetch R data, not Z and PHI.

## Public API Pattern

Users interact with `ImasComposer` public API, never with mapper internals.

```python
# Basic usage
composer = ImasComposer()

# With configuration options
composer = ImasComposer(efit_tree='EFIT01')  # Specify equilibrium tree

# Resolve requirements iteratively
raw_data = {}
while True:
    fully_resolved, requirements = composer.resolve(ids_path, shot, raw_data)
    if fully_resolved:
        break
    # Fetch requirements...
    raw_data.update(fetched)

# Compose final data
result = composer.compose(ids_path, shot, raw_data)
```

### Tree/Source Selection Pattern

Some IDS types have multiple data sources (e.g., different equilibrium reconstructions):

**Equilibrium**: Multiple EFIT trees (`EFIT01`, `EFIT02`, etc.)
```python
# Use EFIT01 (default)
composer = ImasComposer()

# Use specific tree
composer = ImasComposer(efit_tree='EFIT02')
```

**Pattern**: Tree/source selection happens at `ImasComposer` initialization, not per-field.

**Testing**: Tests use default configuration (e.g., `EFIT01`) for consistency.

## Test Configuration System

### Per-IDS Test Customization

**Only create test config files when needed.** If an IDS follows all standard test rules, no config file is required.

When needed, create `tests/test_config_<ids>.yaml` with custom test behavior:

```yaml
# Thomson Scattering Test Configuration
field_exceptions:
  thomson_scattering._hwmap:
    allow_different_shot: true
    reason: "Uses calibration shot from calib_nums, not requested shot"

requirement_validation:
  allow_different_shot:
    - thomson_scattering._hwmap
```

**Purpose**: Handle valid exceptions to standard test assertions (e.g., calibration data from different shots)

**When to create a config file**:
- Field uses calibration/reference data from a different shot
- Field has valid reason to violate standard requirement rules
- Document the `reason` in `field_exceptions` for clarity

**Default behavior**: If no config file exists, `load_test_config()` returns empty exceptions and standard validation rules apply

## OMAS Integration Special Cases

### Equilibrium IDS - Field-by-Field Fetching

**Problem**: Equilibrium IDS contains huge 2D grids. Fetching `equilibrium.*` loads thousands of points unnecessarily.

**Solution**: For equilibrium only, fetch each field individually in tests.

```python
# conftest.py - omas_data fixture supports optional ids_path parameter
def _fetch_omas_data(ids_name, ids_path=None):
    # For equilibrium, pass specific field path
    # For others, use wildcard '{ids_name}.*'
    if ids_path is None:
        ids_path = f'{ids_name}.*'
    machine_to_omas(ods, 'd3d', shot, ids_path, options={'EFIT_tree': 'EFIT01'})
```

**Usage**:
```python
# ECE/Thomson - fetch entire IDS
omas_ece = omas_data('ece')

# Equilibrium - fetch specific field
omas_eq_time = omas_data('equilibrium', 'equilibrium.time')
```

**Implementation**: `run_composition_against_omas` automatically uses field-by-field fetching for equilibrium:
```python
if ids_name == 'equilibrium':
    omas_value = get_omas_value(omas_data(ids_name, ids_path), ids_path)
else:
    omas_value = get_omas_value(omas_data(ids_name), ids_path)
```

**Note**: This is the ONLY IDS that requires field-by-field fetching. All others use wildcard.

## Test Structure

### Two Test Files Per IDS

1. **`test_<ids>_requirements.py`** - Tests requirement resolution
   - Single parametric test using `load_ids_fields()`
   - Calls `test_requirements_resolution()` from conftest.py
   - Verifies all fields can be fully resolved

2. **`test_<ids>_composition.py`** - Tests composition against OMAS
   - Single parametric test using `load_ids_fields()`
   - Calls `test_composition_against_omas()` from conftest.py
   - Verifies output matches OMAS reference

### Test File Template

```python
"""Test <IDS> requirement resolution."""

import pytest
from conftest import REFERENCE_SHOT, load_ids_fields, test_requirements_resolution

pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus]

@pytest.mark.parametrize('ids_path', load_ids_fields('<ids_name>'))
def test_can_resolve_requirements(ids_path, composer):
    """Test that resolve() can fully resolve requirements."""
    resolution_steps = test_requirements_resolution(ids_path, composer, REFERENCE_SHOT)
    print(f"\n{ids_path}: resolved in {resolution_steps} steps")
```

**Total lines per test file**: ~17 lines

## Naming Conventions

### Consistency Between Public API and Internals

- Public API: `compose()` method
- Internal methods: `_compose_*()` functions
- **Never** use different terms (like `synthesize`) for the same concept

### Method Naming

- `_compose_*()` - Creates final IDS data
- `_derive_*_requirements()` - Creates DERIVED stage requirements
- `_get_*()` - Retrieves values from raw_data

## Three-Stage Requirement System

1. **DIRECT** - Static requirements (shot number may vary)
   ```python
   static_requirements=[Requirement('\\PATH', 0, 'TREE')]
   ```

2. **DERIVED** - Requirements that depend on fetched data
   ```python
   derive_requirements=lambda shot, raw: [...]
   ```

3. **COMPUTED** - No requirements, synthesizes from raw_data
   ```python
   compose=lambda shot, raw: ...
   ```

## Key Implementation Details

### Requirement Keys

Always use tuple keys matching `Requirement.as_key()`:
```python
# Correct
raw_data[(req.mds_path, req.shot, req.treename)] = value

# Wrong
raw_data[req.mds_path] = value
```

### OMAS Integration

- Use `mdsvalue('d3d', treename, shot, mds_path).raw()` to fetch data
- Use `machine_to_omas(ods, 'd3d', shot, '<ids>.*')` to fetch reference data
- Use OMAS's list indexing for nested access: `ods[ids_name][field1, field2, ...]`

## Backwards Compatibility

**Not a concern.** We can make breaking changes freely during development. Focus on getting the architecture right, not preserving compatibility with earlier versions.

## Adding a New IDS

To add support for a new IDS (e.g., `magnetics`):

1. Create `ids/magnetics.yaml` with field list and static values
2. Create `ids/magnetics.py` with `MagneticsMapper(IDSMapper)` class
3. Register mapper in `composer.py`
4. Create `tests/test_magnetics_requirements.py` (~17 lines)
5. Create `tests/test_magnetics_composition.py` (~17 lines)
6. (Optional) Create `tests/test_config_magnetics.yaml` if special test rules needed

That's it! All test infrastructure is already in place.

### When to Create Test Config YAML

Create `tests/test_config_<ids>.yaml` when:
- Fields use calibration/reference data from different shots
- Requirements need custom validation rules
- Standard test assertions don't apply to specific fields

Otherwise, the default config (all requirements must match requested shot) is used.

## Questions to Ask

When implementing new features, always ask:

1. **Is this duplicating existing code?** → Extract to shared utility
2. **Am I bundling requirements?** → Split into isolated specs
3. **Is this IDS-specific or generic?** → Put in right place
4. **Can tests reuse existing functions?** → Use conftest.py functions
5. **Are field lists hardcoded?** → Load from YAML instead

## References

- OMAS implementation: `/home/denks/DIIID_IMAS/omas/omas/machine_mappings/d3d.py`
- IMAS Data Dictionary: See YAML `docs_file` paths
- Test patterns: `tests/conftest.py` generic functions
