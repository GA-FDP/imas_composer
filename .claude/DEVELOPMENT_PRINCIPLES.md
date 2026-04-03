# IMAS Composer Development Principles

This document outlines the core architectural principles for the imas_composer project.

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
   - `run_composition_against_omas()` - generic OMAS validation

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

## Separation of Concerns

**Data fetching is separate from composition logic.**

- `composer.py` - Requirement resolution and composition (NO data fetching)
- `fetchers.py` - Data retrieval via MDSplus/OMAS
- `ids/*.py` - IDS-specific mappers

This allows:
- Using imas_composer without MDSplus/OMAS (with pre-fetched data)
- Testing composition logic independently
- Swapping data sources without changing composition code

## Public API Pattern

Users interact with `ImasComposer` public API, never with mapper internals.

### Basic Usage

```python
from imas_composer import ImasComposer

composer = ImasComposer()

# Resolve requirements iteratively
raw_data = {}
while True:
    status, requirements = composer.resolve(ids_paths, shot, raw_data)
    if all(status.values()):
        break
    # Fetch requirements...
    raw_data.update(fetched)

# Compose final data
results = composer.compose(ids_paths, shot, raw_data)
```

### With Configuration

```python
# Specify data sources at initialization
composer = ImasComposer(
    efit_tree='EFIT01',           # Equilibrium reconstruction
    profiles_tree='ZIPFIT01',      # Kinetic profiles
    fast_ece=True                  # Use fast ECE data
)
```

### Tree/Source Selection Pattern

Some IDS types have multiple data sources:

**Equilibrium**: Multiple EFIT trees (`EFIT01`, `EFIT02`, etc.)
```python
composer = ImasComposer(efit_tree='EFIT02')
```

**Core Profiles**: Multiple profile trees (`ZIPFIT01`, `OMFIT_PROFS`, etc.)
```python
composer = ImasComposer(profiles_tree='OMFIT_PROFS')
```

**Pattern**: Tree/source selection happens at `ImasComposer` initialization, not per-field.

## Three-Stage Requirement System

The core of imas_composer's architecture.

### Stage 1: DIRECT

Static requirements - MDSplus paths known at initialization.

```python
self.specs["_bcentr"] = IDSEntrySpec(
    stage=RequirementStage.DIRECT,
    static_requirements=[
        Requirement('\\EFIT::TOP.RESULTS.GEQDSK.BCENTR', 0, 'EFIT01')
    ]
)
```

- Shot number may be 0 (placeholder) - will be filled at resolve time
- Treename specifies which MDSplus tree to use
- Can have multiple static requirements for one node

### Stage 2: DERIVED

Requirements that depend on previously fetched data.

```python
self.specs["_channel_data"] = IDSEntrySpec(
    stage=RequirementStage.DERIVED,
    depends_on=["_channel_ids"],  # Must fetch this first
    derive_requirements=lambda shot, raw: [
        Requirement(f'\\TREE::DATA_{ch}', shot, 'TREE')
        for ch in get_channel_ids(raw)
    ]
)
```

- `depends_on`: List of nodes that must be resolved first
- `derive_requirements`: Function that generates requirements from fetched data
- Can have multi-level dependencies (A → B → C)

### Stage 3: COMPUTED

Final composition from raw data - no new requirements.

```python
self.specs["thomson_scattering.channel.position.r"] = IDSEntrySpec(
    stage=RequirementStage.COMPUTED,
    depends_on=["_position_r"],
    compose=self._compose_position_r
)
```

- `compose`: Function that transforms raw data to final IDS value
- All COMPUTED nodes appear in YAML field list
- Auxiliary nodes (leading `_`) are DIRECT or DERIVED

### Dependency Resolution

The system automatically resolves dependencies:

1. Collect all DIRECT requirements
2. Fetch data
3. Use fetched data to derive DERIVED requirements
4. Fetch derived data
5. Repeat until all dependencies resolved
6. Compose final values from all fetched data

**Max depth**: 10 levels to prevent infinite loops

## Naming Conventions

### Consistency Between Public API and Internals

- Public API: `compose()` method
- Internal methods: `_compose_*()` functions
- **Never** use different terms (like `synthesize`) for the same concept

### Method Naming

- `_compose_*()` - Creates final IDS data from raw data
- `_derive_*_requirements()` - Creates DERIVED stage requirements
- `_get_*()` - Retrieves values from raw_data

### Node Naming

- Auxiliary nodes: `_position_r` (leading underscore, not in YAML)
- Final fields: `thomson_scattering.channel.position.r` (full IDS path, in YAML)

## Requirement Keys

Always use tuple keys matching `Requirement.as_key()`:

```python
# Correct
key = (req.mds_path, req.shot, req.treename)
raw_data[key] = value

# Wrong
raw_data[req.mds_path] = value
```

**Why:** Multiple requests can have same path but different shot/tree.

## Test Configuration System

### Per-IDS Test Customization

**Only create test config files when needed.** If an IDS follows all standard test rules, no config file is required.

When needed, create `tests/test_config_<ids>.yaml` with custom test behavior:

```yaml
# Field exceptions - document WHY
field_exceptions:
  thomson_scattering._hwmap:
    allow_different_shot: true
    reason: "Uses calibration shot from calib_nums, not requested shot"

# Requirement validation rules
requirement_validation:
  allow_different_shot:
    - thomson_scattering._hwmap

# Field-specific tolerances (if needed)
field_tolerances:
  equilibrium.time_slice.profiles_1d.j_tor:
    rtol: 1.0e-5
    atol: 1.0
```

**Purpose**: Handle valid exceptions to standard test assertions.

**When to create a config file**:
- Field uses calibration/reference data from a different shot
- Field has expected precision differences (document why)
- Field needs custom requirement validation

**Default behavior**: If no config file exists, standard rules apply:
- All requirements must match requested shot
- Default tolerances (rtol=1e-7, atol=0)
- No exceptions

## OMAS Integration

### Fetching Reference Data

```python
from omas import ODS
from omas.machine_mappings.d3d import machine_to_omas

ods = ODS()
machine_to_omas(ods, 'd3d', shot, 'equilibrium.*')
```

### Accessing Values

```python
# Nested navigation
value = ods['equilibrium']['time_slice'][0]['profiles_1d']['psi']

# Using list indexing
value = ods['equilibrium', 'time_slice', 0, 'profiles_1d', 'psi']
```

### Uncertainty Handling

- `ods['path']['data']` - nominal values
- `ods['path']['data_error_upper']` - uncertainties
- Ignore `unumpy.uarray()` in OMAS source - it auto-converts

### Field-by-Field Fetching (Equilibrium Only)

Equilibrium IDS is large - fetch fields individually in tests:

```python
# For equilibrium only
machine_to_omas(ods, 'd3d', shot, 'equilibrium.time')  # Single field

# For other IDS
machine_to_omas(ods, 'd3d', shot, 'thomson_scattering.*')  # All fields
```

This is handled automatically in test infrastructure.

## Backwards Compatibility

**Not a concern.** We can make breaking changes freely during development. Focus on getting the architecture right, not preserving compatibility.

- Let tests fail when behavior changes
- Explicit is better than implicit
- Clear errors are better than silent fallbacks

## Questions to Ask

When implementing new features, always ask:

1. **Is this duplicating existing code?** → Extract to shared utility
2. **Am I bundling requirements?** → Split into isolated specs
3. **Is this IDS-specific or generic?** → Put in right place
4. **Can tests reuse existing functions?** → Use conftest.py functions
5. **Are field lists hardcoded?** → Load from YAML instead
6. **Does composition depend on fetching?** → Keep separate (composer vs fetchers)

## Adding a New IDS

To add support for a new IDS (e.g., `magnetics`):

1. Create `ids/magnetics.yaml` with field list
2. Create `ids/magnetics.py` with `MagneticsMapper(IDSMapper)` class
3. Register mapper in `ids/ids_factory.py`
4. Create `tests/test_magnetics_requirements.py` (~17 lines)
5. Create `tests/test_magnetics_composition.py` (~17 lines)
6. (Optional) Create `tests/test_config_magnetics.yaml` if special rules needed

That's it! All test infrastructure is already in place.

See IMPLEMENTING_FIELDS.md for field implementation details and TESTING_GUIDE.md for testing workflow.

## References

- OMAS implementation: `/home/denks/DIIID_IMAS/omas/omas/machine_mappings/d3d.py`
- IMAS Data Dictionary: See YAML `docs_file` paths
- Test patterns: `tests/conftest.py` generic functions
