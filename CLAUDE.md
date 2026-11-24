# imas_composer - Claude Code context

## Project Architecture at a Glance

**imas_composer** converts DIII-D MDSplus data to IMAS format using a three-stage requirement system:

1. **DIRECT**: Static MDSplus paths known at initialization
2. **DERIVED**: Requirements that depend on fetched data (e.g., channel IDs)
3. **COMPUTED**: Final composition from raw_data (no new requirements)

**Key modules:**
- `composer.py` - Requirement resolution and composition (NO data fetching)
- `fetchers.py` - Data retrieval via MDSplus/OMAS
- `ids/*.py` - IDS-specific mappers with YAML configurations

**Core principles:**
- **DRY**: All common logic in shared utilities (especially `tests/conftest.py`)
- **Requirement Isolation**: Each field fetches only what it needs
- **Declarative Config**: Field lists and static values in YAML files
- **OMAS Compatibility**: Output exactly matches OMAS reference implementation

## Quick Reference

**OMAS Integration:**
```python
machine_to_omas(ods, 'd3d', shot, 'ids_name.*')  # Fetch all fields
```

**OMAS → imas_composer mapping:**
- Static data (known paths in OMAS `query` dict) → **DIRECT** stage
- Dynamic data (depends on fetch, e.g., OMAS loops) → **DERIVED** stage
- Unit conversions must match OMAS exactly

**Development Philosophy:**
- No backwards compatibility needed - let tests fail when behavior changes
- Explicit is better than implicit
- Clear errors are better than silent fallbacks

## Documentation Index

**When to read what:**
- **Starting a session**: This file - quick orientation
- **Architecture details**: `.claude/DEVELOPMENT_PRINCIPLES.md` - three-stage system, DRY patterns, public API
- **Implementing fields**: `.claude/IMPLEMENTING_FIELDS.md` - transcription workflow from OMAS
- **Writing tests**: `.claude/TESTING_GUIDE.md` - test patterns and configuration
- **Reviewing changes**: `docs/review_hazards.md` - what a change can get silently wrong

## Branches and pull requests

### Risks
PR descriptions are organised around what the change could get silently wrong. The change classes,
the general risks and the hazard checklist are in `docs/review_hazards.md`. When a defect escapes
review, its fix adds an entry there.

### Pull request checks
The `open-pr` skill reads this section.

The full test suite fetches DIII-D data for every IDS and takes 10 hours or more when run locally and
serially, so it runs in CI, not in the session. `.github/workflows/tests.yml` runs on every pull
request: a quick job (`tests/test_fetchers.py`, `tests/test_get_supported_fields.py`), then one job
per IDS (`tests/test_<ids>_*.py`, one IDS per `tests/test_<ids>_requirements.py`) and an `other` job
that runs every remaining module. `baseline_validation` and `benchmark` tests are opt-in and run
nowhere by default; `requires_d3drdb` tests run only locally, because CI cannot reach D3DRDB.

- Locally, run at most the tests of the touched IDS, in the `imas_composer` environment:
  `mamba run -n imas_composer pytest tests/test_<ids>_*.py -n auto --maxfail=0`.
- Open the PR as a draft, then wait for CI on its head commit. Poll with a fresh token per call
  rather than `gh pr checks --watch`, which outlives the one-hour token:
  `GH_TOKEN=$(mamba run -n imas_composer bot-token) mamba run -n imas_composer gh pr checks <number> --repo GA-FDP/imas_composer`.
  Read the logs of a failed job with `gh run view <run id> --log-failed`.
- Fix failures before the PR is marked ready; a known pre-existing failure is reported, not hidden.
  The evidence in the risk entries is the CI run of the head commit: name the jobs and their result,
  and update the PR body once CI has finished.

Then check what CI cannot: a new IDS has a `tests/test_<ids>_requirements.py` (otherwise it gets
no job of its own and its tests run in the `other` job), a new field is in its `ids/<ids>.yaml` and has tests, unit conversions
and time bases are checked against OMAS or the MDSplus source, a new or loosened entry in a
`tests/test_config_<ids>.yaml` says why, and no requirement bundles data a field does not need.

## GitHub access
`gh`, `jq` and `openssl` are not on the plain PATH: they come from the `imas_composer` mamba
environment. Run `gh` with `mamba run -n imas_composer` and authenticate every call as the
imas_composer bot, never as the developer. Shell variables do not persist between commands, so
every call mints its own token. `bot-token` comes with the `session-to-pr-skills` plugin (see
"Claude Code" in `INSTALLATION.md`):
```
GH_TOKEN=$(mamba run -n imas_composer bot-token) mamba run -n imas_composer gh pr list --repo GA-FDP/imas_composer
```
The token is valid for one hour and only for GA-FDP/imas_composer; mint a new one when a call
returns 401. If `bot-token` is not found, the plugin is not installed; if `mamba` is not found, ask
the developer where it is. If it fails, the developer has not installed their bot key yet: point
them to "GitHub access" in `INSTALLATION.md`. Open and update PRs with the `open-pr` skill.

Pushes use the developer's own SSH key. The ruleset on `main` requires approval of the most recent
push by someone other than its pusher, so the developer's approval never counts for work they
pushed. The ruleset also covers every PR in a GitHub stack whose bottom PR targets `main`. The one
exception is propagating merges through a PR stack: conflict-free merges are committed and pushed as
the bot, so the reviewer can tell them from new work at a glance. They still need the reviewer's
re-approval (see the `propagate-stack` skill). Push through the bot only conflict-free merge
commits, never a conflict resolution or new work:
```
GH_TOKEN=$(mamba run -n imas_composer bot-token) git -c credential.helper= \
  -c credential.helper='!f() { echo username=x-access-token; echo "password=$GH_TOKEN"; }; f' \
  push https://github.com/GA-FDP/imas_composer.git <branch>
```
