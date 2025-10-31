# Claude Context Files

This directory contains context files for Claude Code sessions working on the imas_composer project.

## Quick Start for New Sessions

**At the start of each session, the user should tell Claude:**
> "Read `.claude/README.md` to understand the project context"

Claude will read this index and know where to find detailed information when needed.

## Context Files Index

### Core Guidelines

- **`.claudecontext`** - Critical rules and policies
  - **TESTING POLICY**: Never run tests that access restricted MDS+ data
  - OMAS usage patterns (fetching, uncertainty handling)
  - Development philosophy (no backwards compatibility needed)
  - OMAS reference files for debugging
  - IDS configuration YAML structure

- **`DEVELOPMENT_PRINCIPLES.md`** - Architecture and design patterns
  - Requirement resolution system
  - Stage-based dependency management (DIRECT → DERIVED → COMPUTED)
  - Tree/source selection pattern
  - OMAS integration patterns
  - Testing infrastructure

### IDS-Specific Documentation

- **`EQUILIBRIUM_ANALYSIS.md`** - Analysis of OMAS equilibrium implementation
  - Field patterns from `_efit.json`
  - Transform patterns (nan_where, unit conversion)
  - MDSplus tree structure (GEQDSK vs AEQDSK)
  - Implementation strategy

- **`EQUILIBRIUM_IMPLEMENTATION.md`** - Equilibrium mapper implementation details
  - Field-by-field mapping documentation
  - MDS+ paths and transforms
  - Ragged array handling
  - Test configuration

## When to Read What

- **Starting a new session**: Read this file to get oriented
- **Implementing a new IDS mapper**: Read `.claudecontext` (OMAS patterns), then IDS-specific analysis if it exists
- **Debugging tests**: Read `.claudecontext` (testing policy and OMAS reference files)
- **Architectural questions**: Read `DEVELOPMENT_PRINCIPLES.md`
- **Equilibrium-specific questions**: Read `EQUILIBRIUM_ANALYSIS.md` or `EQUILIBRIUM_IMPLEMENTATION.md`

## File Organization

Keep this directory minimal and well-organized:
- Core guidelines that apply to all IDS mappers in root-level files
- IDS-specific documentation clearly labeled
- Update this README when adding new files
