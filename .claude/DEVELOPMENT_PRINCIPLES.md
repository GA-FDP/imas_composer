# IMAS Composer Development Principles

Core principles and quick reference for developing IMAS Composer. For detailed information, see specialized guides in this directory.

## Core Principles

### 1. DRY (Don't Repeat Yourself)
- All test logic in `tests/conftest.py` as reusable functions
- Common mapper functionality in `ids/base.py` base class
- Configuration in YAML files, never hardcoded
- IDS-specific test files are ~18 lines each

### 2. Requirement Isolation
**Each field fetches only what it needs.** Never bundle requirements.

```python
# GOOD - isolated
self.specs["_position_r"] = IDSEntrySpec(
    static_requirements=[Requirement('...R', 0, 'TREE')]
)

# BAD - bundled
self.specs["_geometry"] = IDSEntrySpec(
    static_requirements=[
        Requirement('...R', 0, 'TREE'),
        Requirement('...Z', 0, 'TREE'),  # Don't bundle!
    ]
)
```

### 3. Batch API
Both `resolve()` and `compose()` accept **lists of paths**:

```python
# Resolve multiple paths together
status, requirements = composer.resolve(ids_paths, shot, raw_data)  # ids_paths is a list

# Compose multiple paths together
results = composer.compose(ids_paths, shot, raw_data)  # Returns dict
```

Benefits: Requirements automatically deduplicated, single fetch cycle.

## Three-Stage Requirement System

1. **DIRECT** - Static MDSplus paths known at init
2. **DERIVED** - Requirements depending on fetched data
3. **COMPUTED** - No requirements, synthesizes from raw_data

See `TEST_CONFIGURATION.md` for detailed patterns.

## ImasComposer Constructor

```python
composer = ImasComposer(
    efit_tree='EFIT01',           # Equilibrium tree
    profiles_tree='ZIPFIT01',     # Core profiles tree (ZIPFIT01 or OMFIT_PROFS)
    profiles_run_id='001',        # Run ID for OMFIT_PROFS
    fast_ece=False,               # ECE time resolution
    include_rip=False             # Interferometer RIP data
)
```

Tree/source selection at initialization, not per-field.

## Test Configuration Quick Reference

Create `tests/test_config_<ids>.yaml` when needed:

```yaml
# Shot selection
override_shots: [200000, 203321]  # Override TEST_SHOTS
exclude_shots: [202161]            # Filter from TEST_SHOTS

# Field validation
skip_fields:
  field.path: "Reason for skipping OMAS comparison"

field_tolerances:
  field.path:
    rtol: 1.0e-5
    atol: 1.0

field_shot_exclusions:
  field.path: [203321, 204602]

# Requirement validation
requirement_validation:
  allow_different_shot:
    - '.mds.path.to.calibration'

# Test variants (for multiple configurations)
test_variants:
  variant_name:
    composer_params: {fast_ece: true}
    omas_params: {fast_ece: true}
    exclude_fields: []

# OMAS path mapping (when needed)
omas_path_map:
  composer.path: omas.path.with.colons
```

**See `TEST_CONFIGURATION.md` for complete reference.**

## Adding a New IDS - Checklist

Required files:
1. `ids/<ids>.yaml` - Field list and static values
2. `ids/<ids>.py` - Mapper implementation
3. Register in `composer.py`
4. `tests/test_<ids>_requirements.py` (~18 lines)
5. `tests/test_<ids>_composition.py` (~18 lines)

Optional:
6. `tests/test_config_<ids>.yaml` - Only if deviating from defaults
7. Add to `scripts/generate_baseline_data.py`

**See `ADDING_NEW_IDS.md` for step-by-step guide.**

## Common Patterns

### Channel-Based Data
```python
# Auxiliary: Channel IDs (DERIVED)
self.specs["_channel_ids"] = IDSEntrySpec(
    derive_requirements=lambda shot, raw: [
        Requirement('.diagnostic.channels', shot, 'TREE')
    ]
)

# Final: Data per channel (DERIVED)
self.specs["diagnostic.channel.data"] = IDSEntrySpec(
    derive_requirements=lambda shot, raw: [
        Requirement(f'.diagnostic.ch{ch}.data', shot, 'TREE')
        for ch in self._get_channel_ids(raw)
    ]
)
```

### Calibration Data
```python
# Calibration shot (DIRECT from shot 0)
self.specs["_calib_shot"] = IDSEntrySpec(
    static_requirements=[Requirement('.calib_shot', 0, 'TREE')]
)

# Calibration data (DERIVED using calib shot)
self.specs["_calib_data"] = IDSEntrySpec(
    derive_requirements=lambda shot, raw: [
        Requirement('.calib', raw[('.calib_shot', 0, 'TREE')], 'TREE')
    ]
)
```

**Important:** Add to `allow_different_shot` in test config.

### Unit Conversions
```python
# COMPUTED - no requirements
self.specs["diagnostic.position.r"] = IDSEntrySpec(
    compose=lambda shot, raw: raw[aux_key] / 100.0  # cm -> m
)
```

### Static Values
```python
# COMPUTED - constant
self.specs["diagnostic.name"] = IDSEntrySpec(
    compose=lambda shot, raw: "Diagnostic Name"
)
```

## Key Implementation Rules

### Requirement Keys
Always use tuple keys matching `Requirement.as_key()`:

```python
# Correct
raw_data[(req.mds_path, req.shot, req.treename)] = value

# Wrong
raw_data[req.mds_path] = value
```

### Naming Conventions
- Public API: `compose()` method
- Internal methods: `_compose_*()` functions
- Auxiliary nodes: `_underscore_prefix`
- Never mix terminology (compose vs synthesize)

### Field Lists in YAML
```yaml
fields:
  # Simple - available for all configs
  - field.path

  # With tree restriction
  - field: field.path
    trees: ['ZIPFIT']  # Only when profiles_tree='ZIPFIT'
```

## Conftest Utilities - Quick Reference

```python
# Configuration
load_test_config(ids_name)
load_ids_fields(ids_name, tree_filter=None)

# Test execution
run_requirements_resolution(ids_path, composer, shot)
run_composition_against_omas(ids_path, composer, omas_data, ids_name, shot)

# Data operations
resolve_and_compose(composer, ids_path, shot)

# Validation
check_skip_field(ids_name, ids_path)
check_field_shot_exclusion(ids_name, ids_path, shot)

# Comparison
compare_values(composer_val, omas_val, label, rtol, atol_float, atol_array)
```

## Questions to Ask

Before implementing:
1. **Is this duplicating code?** → Extract to shared utility
2. **Am I bundling requirements?** → Split into isolated specs
3. **Is this IDS-specific or generic?** → Right location (ids/ vs conftest.py)
4. **Can tests reuse existing functions?** → Use conftest.py
5. **Are field lists hardcoded?** → Load from YAML
6. **Does this need test config?** → Only if deviating from defaults
7. **Is batch API used?** → Pass lists to resolve()/compose()
8. **Are variant params consistent?** → composer_params must match omas_params

## Detailed Documentation

For comprehensive information, see:
- **`TEST_CONFIGURATION.md`** - Complete test config reference, parametrization, fixtures
- **`ADDING_NEW_IDS.md`** - Step-by-step guide for adding new IDS
- **`OMAS_INTEGRATION.md`** - OMAS path mapping, fetch ordering, tolerances
- **`COMMON_PATTERNS.md`** - COCOS, ragged arrays, multiple data sources, etc.

## References

- Test patterns: `tests/conftest.py`
- Example mappers: `ids/equilibrium.py`, `ids/thomson_scattering.py`
- OMAS implementation: Check d3d.py mappings
