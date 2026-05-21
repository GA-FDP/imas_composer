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

## Temporary setup to export to standard IDS (assumes Omega cluster)
Clone IMAS_composer:
```
git clone git@github.com:GA-FDP/imas_composer.git
cd imas_composer
```
Create a new environment:
```
module load conda
mamba env create -f environment.yaml
conda activate imas_composer
pip install --no-deps --no-build-isolation -e .
```
add imas_python dependencies
```
pip install --no-deps --no-build-isolation imas_data_dictionaries xxhash imas_core
```
Checkout dev version of imas_python
```
cd ../
git clone git@github.com:AreWeDreaming/IMAS-Python.git
cd IMAS-Python
git switch awkward_array_support
```
Add imas_python to the environment
```
pip install --no-deps --no-build-isolation -e  .
```
Patch `_version.py` missed by `pip install`
```
cat > imas/_version.py << 'EOF'
version="0.0.0"
version_tuple=[0,0,0]
EOF
```
Run the simple test:
```
cd ../imas_composer
python convert_to_ids.py
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

### Code Style Guidelines

**CRITICAL: Module-level imports only**
- **NEVER** import modules at the function level (e.g., `import awkward as ak` inside a function)
- All imports must be at the module/file level at the top of the file
- This ensures consistent performance and avoids import overhead in tight loops
- Violation of this rule can cause significant performance degradation in tests and production code

## References

- **IMAS Documentation**: https://www.iter.org/
- **OMAS Repository**: https://github.com/gafusion/omas
- **DIII-D MDSplus Documentation**: Internal DIII-D documentation https://nomos.gat.com/DIII-D/documentation/
- **Claude Code Context**: See `.claude/README.md` for development guides

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or contributions, please open an issue on the project repository.
