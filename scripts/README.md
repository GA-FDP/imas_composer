# Baseline Data Generation and Validation

This directory contains utilities for generating and validating baseline data across different conda environments and dependency versions.

## Overview

Primary goal of this is to verify data integrity when updating MDSplus versions

## Workflow

### 1. Generate Baseline Data (Reference Environment)


```bash
# Generate baseline for all IDS fields across all parametrized shots
# This creates baseline_data.pkl in the repository root
python scripts/generate_baseline_data.py

```

This will:
- Fetch all parametrized test fields using `simple_load`
- Test all shots from `TEST_SHOTS` in conftest.py (currently: 202161, 203321, 204602, 204601)
- Store them in a pickle file
- Print a summary of successful/failed/skipped fields per shot

### 2. Validate in New Environment

After setting up a new conda environment or updating dependencies, validate that data fetching still works correctly:

**IMPORTANT:** Baseline validation tests are opt-in only. You must use `-m baseline_validation` to run them.

```bash
# Run validation against baseline (MUST use -m baseline_validation)
# Automatically uses baseline_data.pkl in repo root
pytest -m baseline_validation -v
