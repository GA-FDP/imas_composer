# IMAS Composer

A Python library for converting DIII-D MDSplus data to IMAS (ITER Integrated Modelling & Analysis Suite) format.

## Overview

IMAS Composer provides a clean, declarative API for mapping DIII-D diagnostic and analysis data (stored in MDSplus) to standardized IMAS v3.41.0 data structures (IDS - Interface Data Structures). It handles:

- **Requirement identification**: Determining what MDSplus data needs to be fetched
- **Data transformation**: Converting units, coordinate systems, and data layouts
- **COCOS conversions**: Handling magnetic coordinate conventions
- **Composition**: Assembling final IMAS-compliant data structures

## Installation
```bash
mamba install -c conda-forge imas_composer
```

For development setup, see [INSTALLATION.md](INSTALLATION.md).

## Quick Start

### Simple Usage (with automatic data fetching)

```python
from imas_composer.fetchers import simple_load

# Load IMAS data with automatic MDSplus fetching
shot = 200000
ids_paths = ['equilibrium.time', 'equilibrium.time_slice.profiles_1d.psi']

results = simple_load(ids_paths, shot, efit_tree='EFIT01')
# results = {'equilibrium.time': array(...), 'equilibrium.time_slice.profiles_1d.psi': array(...)}
```

### Advanced Usage (manual data fetching)

```python
from imas_composer import ImasComposer
from imas_composer.fetchers import fetch_requirements

# Create composer instance
composer = ImasComposer(efit_tree='EFIT01')

# Resolve requirements iteratively
shot = 200000
ids_paths = ['equilibrium.time', 'equilibrium.time_slice.profiles_1d.psi']
raw_data = {}

while True:
    status, requirements = composer.resolve(ids_paths, shot, raw_data)
    if all(status.values()):
        break

    # Fetch requirements from MDSplus via OMAS
    fetched = fetch_requirements(requirements)
    raw_data.update(fetched)

# Compose final IMAS data
results = composer.compose(ids_paths, shot, raw_data)
```

## Working with Claude Code

This project is designed to be extended with AI assistance.

**At the start of each Claude Code session, tell Claude:**
```
Read .claude/.claudecontext
```

This provides:
- Data privacy policy (NEVER execute code)
- Project architecture and three-stage requirement system
- OMAS integration patterns
- Links to detailed implementation and testing guides

### Adding New Fields

Tell Claude which OMAS mapping to implement:
```
Implement equilibrium.time_slice.profiles_2d.psi from omas/machine_mappings/_efit.json
```

Claude will create the necessary mapper code, YAML configuration, and test files. See [.claude/.claudecontext](.claude/.claudecontext) and [.claude/IMPLEMENTING_FIELDS.md](.claude/IMPLEMENTING_FIELDS.md) for details.

## Contributing

When contributing new fields or IDS mappers:
1. Follow the three-stage requirement pattern (DIRECT → DERIVED → COMPUTED)
2. Match OMAS transformations exactly
3. Add field configurations to YAML files
4. Include both requirement and composition tests
5. Consider using Claude Code with [.claude/](.claude/) documentation for guidance

## References

- **IMAS Documentation**: https://www.iter.org/
- **OMAS Repository**: https://github.com/gafusion/omas
- **DIII-D MDSplus Documentation**: Internal DIII-D documentation https://nomos.gat.com/DIII-D/documentation/
- **Claude Code Context**: See `.claude/README.md` for development guides

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or contributions, please open an issue on the project repository.
