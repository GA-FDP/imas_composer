# Claude Context Files

This directory contains context files for Claude Code sessions working on the imas_composer project.

## Quick Start for New Sessions

**At the start of each session, the user should tell Claude:**
> "Read `.claude/README.md` to understand the project context"

Claude will read this index and know where to find detailed information when needed.

## Context Files Index

### Core Guidelines (Read First)

- **`.claudecontext`** (~66 lines) - Critical policies and quick reference
  - ⚠️ **NEVER EXECUTE CODE** - Data privacy policy
  - OMAS quick reference (fetching, uncertainty)
  - OMAS reference files for debugging
  - Links to detailed architecture docs

- **`DEVELOPMENT_PRINCIPLES.md`** (~297 lines) - Complete architecture reference
  - DRY principle and requirement isolation
  - Three-stage system (DIRECT → DERIVED → COMPUTED)
  - Public API and tree selection patterns
  - Test structure and configuration
  - YAML configuration system
  - Adding new IDS guide

### IDS-Specific Documentation

- **`EQUILIBRIUM_ANALYSIS.md`** - Analysis of OMAS equilibrium implementation
  - Field patterns from `_efit.json`
  - Transform patterns (nan_where, unit conversion)
  - MDSplus tree structure (GEQDSK vs AEQDSK)
  - Implementation strategy

- **`EQUILIBRIUM_IMPLEMENTATION.md`** - Equilibrium mapper implementation details
  - Field-by-field mapping documentation
  - MDSplus paths and transforms
  - Ragged array handling
  - Test configuration

## When to Read What

- **Starting a session**: Read `.claudecontext` (~66 lines) - gets you 80% of what you need
- **Implementing new IDS**: `.claudecontext` first, then `DEVELOPMENT_PRINCIPLES.md` for details
- **Debugging**: `.claudecontext` points to OMAS reference files
- **Architecture deep-dive**: `DEVELOPMENT_PRINCIPLES.md`
- **Equilibrium-specific**: `EQUILIBRIUM_IMPLEMENTATION.md` (COCOS, ragged arrays)

## File Organization

Keep this directory minimal and well-organized:
- Core guidelines that apply to all IDS mappers in root-level files
- IDS-specific documentation clearly labeled
- Update this README when adding new files
