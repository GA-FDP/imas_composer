# Testing Guide

This guide covers creating and debugging tests for imas_composer fields. Testing typically happens in a separate session from implementation.

## Test Structure

Every IDS has two test files (both ~17 lines):

1. **`test_<ids>_requirements.py`** - Tests requirement resolution
2. **`test_<ids>_composition.py`** - Tests composition against OMAS

All test logic lives in `tests/conftest.py` (DRY principle).

## Creating Test Files

### Requirements Test Template

`tests/test_<ids>_requirements.py`:

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

### Composition Test Template

`tests/test_<ids>_composition.py`:

```python
"""Test <IDS> composition against OMAS."""

import pytest
from conftest import (
    REFERENCE_SHOT,
    load_ids_fields,
    run_composition_against_omas,
)

pytestmark = [pytest.mark.integration, pytest.mark.requires_mdsplus, pytest.mark.omas_validation]

@pytest.mark.parametrize('ids_path', load_ids_fields('<ids_name>'))
def test_composition_matches_omas(ids_path, composer, fetch_requirements_fn, omas_data):
    """Test that composed values match OMAS reference implementation."""
    run_composition_against_omas(
        ids_path, composer, fetch_requirements_fn, omas_data, REFERENCE_SHOT
    )
```

**That's it!** Total ~17 lines per test file. All logic is in conftest.py.

## Running Tests

### VSCode Test Explorer

1. Click the flask/beaker icon in sidebar
2. Navigate test tree to find specific test
3. Click ▶️ to run
4. View output in "Test Results" panel

### Command Line

```bash
# Run all tests for an IDS
pytest tests/test_equilibrium_composition.py -v

# Run specific field test
pytest tests/test_equilibrium_composition.py::test_composition_matches_omas[equilibrium.time] -v

# Run with markers
pytest -m "not slow"  # Skip slow tests
pytest -m omas_validation  # Only OMAS validation tests

# Stop on first failure
pytest -x
```

**Note:** All tests require MDSplus access to DIII-D servers.

## Test Configuration YAML

**Only create test config when needed.** If an IDS follows standard rules, no config is required.

### When to Create Config

Create `tests/test_config_<ids>.yaml` when fields have valid exceptions:
- Use calibration data from different shots
- Have field-specific precision tolerances
- Need custom requirement validation

### Config Structure

`tests/test_config_thomson_scattering.yaml`:

```yaml
# Field exceptions - document WHY fields break standard rules
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

**Always document the `reason` in `field_exceptions`** - future developers need to know why.

### Default Behavior

If no config file exists, `load_test_config()` returns:
- Empty field exceptions
- Standard validation rules (all requirements must match requested shot)
- Default tolerances (rtol=1e-7, atol=0)

## Common Test Failures and Solutions

### Shape Mismatches

```
AssertionError: Shape mismatch: composer (65, 65, 100) vs OMAS (100, 65, 65)
```

**Cause:** Fortran column-major vs C row-major ordering.

**Solution:** Add transpose in compose function:
```python
psi_2d = raw_data[key]
return np.transpose(psi_2d, (2, 1, 0))  # Match OMAS TRANSPOSE
```

### Precision Differences

```
AssertionError: Values differ by 1e-3 (tolerance: 1e-7)
```

**Cause:** Interpolation methods, COCOS factors, or numerical precision.

**Solutions:**
1. Check COCOS transformation is applied
2. Verify interpolation method matches OMAS (linear vs cubic)
3. Add field-specific tolerance in test config YAML if valid:
   ```yaml
   field_tolerances:
     equilibrium.time_slice.profiles_1d.j_tor:
       rtol: 1.0e-5
       atol: 1.0
   ```

### Different Shot Numbers

```
AssertionError: Requirement has shot 123456 but expected 200000
```

**Cause:** Field uses calibration/reference data from different shot.

**Solution:** Document in test config:
```yaml
field_exceptions:
  thomson_scattering._hwmap:
    allow_different_shot: true
    reason: "Uses calibration shot from calib_nums array"

requirement_validation:
  allow_different_shot:
    - thomson_scattering._hwmap
```

### Unresolved Requirements

```
AssertionError: Failed to resolve after 10 iterations
```

**Cause:** Missing dependency or circular dependency.

**Solution:**
1. Check `depends_on` chain is correct
2. Verify all auxiliary nodes are registered
3. Check `derive_requirements` functions for errors
4. Print resolution steps to see where it gets stuck

### Type Mismatches

```
AssertionError: Type mismatch: composer <class 'awkward.Array'> vs OMAS <class 'numpy.ndarray'>
```

**Cause:** Using awkward array for non-ragged data or vice versa.

**Solution:**
- Use `awkward.Array` only for truly ragged data (variable length per time slice)
- Use `numpy.ndarray` for fixed-size arrays
- Check OMAS reference to see what type it returns

## Equilibrium Special Case: Field-by-Field Fetching

Equilibrium IDS is large - fetching `equilibrium.*` loads thousands of points.

**Solution:** Test infrastructure automatically fetches equilibrium fields individually:

```python
if ids_name == 'equilibrium':
    omas_value = get_omas_value(omas_data(ids_name, ids_path), ids_path)
else:
    omas_value = get_omas_value(omas_data(ids_name), ids_path)
```

**You don't need to do anything** - this is handled in `conftest.py`.

## Debugging Workflow

When a test fails:

1. **Read the error message** - it tells you what's wrong:
   - Shape mismatch → transpose issue
   - Value difference → check units/transformations
   - Shot number mismatch → calibration data

2. **Check OMAS reference** - compare your implementation:
   ```python
   # In omas/machine_mappings/d3d.py or _efit.json
   # See exactly what OMAS does
   ```

3. **Add debug prints** (user will run):
   ```python
   def _compose_field(self, shot, raw_data):
       result = transform(raw_data[key])
       print(f"Debug: shape={result.shape}, min={result.min()}, max={result.max()}")
       return result
   ```

4. **Share output with user** - they'll run the test and provide output

5. **Iterate** - fix issue, user runs again

**Remember:** You CANNOT run tests yourself (data privacy policy). User runs tests and shares output.

## Test Infrastructure (conftest.py)

All shared test logic lives in `conftest.py`:

### Key Functions

- `load_ids_fields(ids_name)` - Loads field list from YAML for parametric tests
- `test_requirements_resolution()` - Generic requirement testing
- `run_composition_against_omas()` - Generic OMAS validation
- `get_omas_value()` - Navigate OMAS nested structures
- `compare_values()` - Type-aware value comparison
- `load_test_config()` - Load IDS-specific test config

### Key Fixtures

- `composer` - ImasComposer instance with default config
- `fetch_requirements_fn` - Function to fetch requirements from MDSplus
- `omas_data` - Factory to fetch OMAS reference data
- `REFERENCE_SHOT` - Shot number used for all tests (200000)

## Adding Tests for New IDS

1. Create `tests/test_<ids>_requirements.py` (use template above)
2. Create `tests/test_<ids>_composition.py` (use template above)
3. (Optional) Create `tests/test_config_<ids>.yaml` if special rules needed
4. Run tests: `pytest tests/test_<ids>_*.py -v`
5. Debug failures using workflow above

That's it! Each test file is ~17 lines. All complexity is in conftest.py (DRY principle).

## Tips for Effective Testing

1. **Test one field at a time** - easier to isolate issues
2. **Start with simple fields** - time, coordinates, metadata
3. **Check types match OMAS** - numpy array vs awkward array vs scalar
4. **Document exceptions** - always include `reason` in test config
5. **Match OMAS exactly** - same units, transforms, ordering
6. **Use default tolerances** - only increase if you understand why

## Next Steps

After tests pass:
1. Add IDS to `scripts/generate_baseline_data.py` if needed
2. Commit changes
3. Move to next field/IDS
