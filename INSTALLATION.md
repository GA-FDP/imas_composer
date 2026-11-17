# Installation

## Recommended: Install from conda-forge

```bash
mamba install -c conda-forge imas_composer
```

This installs the package with all core dependencies (numpy, pyyaml, awkward).

## For Development

```bash
# Clone and navigate to repository
cd imas_composer

# Create environment from environment.yaml
mamba env create -f environment.yaml
mamba activate imas_composer

# Install the submodules and the package in editable mode
pip install --no-deps --no-build-isolation -e ./omas_submodule
pip install --no-deps --no-build-isolation -e ./toksearch_submodule
pip install --no-deps --no-build-isolation -e ./toksearch_d3d_submodule
pip install --no-deps --no-build-isolation -e .
```

The [environment.yaml](environment.yaml) file includes all dependencies for development and testing, including OMAS and MDSplus.
