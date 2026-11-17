# Review risks and hazards

What a change to imas_composer can get wrong, for the author writing a PR, the reviewer reading it
and the skills that audit it. The PR body is organised around these risks: each risk names its
cause in the code, the evidence that rules it out and what that evidence does not reach (see the
`open-pr` skill).

The worst outcome is a field that is served with the right shape and plausible values but is
wrong: a unit off by a factor, a slice that belongs to another time, a sign convention flipped.
Downstream codes such as IDA take the values as measurements and fit them without complaint.
Crashes and missing fields are secondary, because they announce themselves.

## General risks

A change often carries several of these; each asks for different evidence. They are not PR labels:
nearly every change to a mapper alters values or what is fetched, so a label would say nothing.

| Risk | The change | The PR has to show |
| --- | --- | --- |
| Numerics | can alter the values a field returns | Which fields moved, by how much, and why the new values are right: the comparison against OMAS or the MDSplus source, on which shots |
| Data source | alters what is fetched: MDSplus paths, trees, run ids, shots, the fetch backend | That the fetched data is the same signal for the same shot and run, on the shots CI covers and on the edge cases below |
| API | changes `ImasComposer` arguments, served IDS paths, YAML field lists or returned types | Every caller that relies on the old form, including pinned consumers such as IDA |

Risk entries in a PR are ordered by severity: silently wrong values first (units, time bases, signs,
the wrong shot or run), then missing or ragged data, then crashes, performance and API.

## Change classes

Every PR falls into one class. When unsure, pick the riskier class.

| Class | Covers |
| --- | --- |
| A. New field or IDS | New mapper, new IDS path, new requirement |
| B. Numerics change | Bug fix in a compose function, unit conversion, time base, dtype, changed default tree or run, fetch backend, dependency bump |
| C. No numerics change | Refactor, move, rename; must show nothing moved |
| D. Tooling | CI, test infrastructure, docs, plotting apps |

What to look for in each class, beyond the general risks:

- **A:** units against the MDSplus source and OMAS, the time base the field shares with its IDS,
  sign and COCOS conventions, behaviour when the signal is missing for a shot, and a test that
  compares against OMAS or the source rather than against the mapper itself.
- **B:** which fields move, by how much, and why the new values are right. For a bug fix the
  mitigation includes the test that fails on the base branch, and the post-mortem of the `open-pr`
  skill.
- **C:** one central risk, that something moved. The mitigation is the same fields composed on the
  base and on the head for the same shots, compared exactly, plus which hunks are pure moves
  (`git diff --color-moved`).
- **D:** whose workflow changes or breaks, and how it was tried.

## Hazard checklist

Known ways imas_composer has gone silently wrong. Reviewers hunt for these, and the audit skills
read this file. When a defect escapes review, its fix adds an entry here.

| Area | Hazard | Seen in |
| --- | --- | --- |
| Units | MDSplus data assumed in cm when it is stored in m, or the other way round | #113 (RSURF, ZSURF, AMINOR), 0f3f19b (SEPLIM, GAP), adddf50 |
| Units | Conversion done in float32: numpy >= 2 keeps float32 for `array / 1000.0`, so toksearch data loses precision | IDA EFIT verification, 203877 EFIT01 (open) |
| Time bases | Quantities on one time base served on another: GTIME, MTIME and ATIME differ in length for some shots | #119 (MTIME/GTIME), #130 (AEQDSK on ATIME, 118162 EFIT01) |
| Time bases | Profiles mapped onto the EFIT time base with missing times left empty | 1befa30 (ZIPFIT on EFIT01) |
| Missing data | Missing slices masked or padded so the array is ragged or shifted, instead of dropped consistently | #131 (X-points, strike points) |
| Signs | COCOS of constraints, psi positions or j_tor not consistent with EFIT | #97, 229a92a, b5c1744, 32fd14c |
| Requirements | Arithmetic or whitespace in a TDI string changes the cache key, so the fetched data is never found | #70 (`summary.tau_energy`, d5501d7), #108 |
| Requirements | A field fetches data it does not need, or two fields share a bundled requirement | `.claude/DEVELOPMENT_PRINCIPLES.md` |
| Data source | Data taken from a calibration or reference shot instead of the requested one, undeclared | `tests/test_config_thomson_scattering.yaml` |
| Physics edge cases | Hydrogen or helium plasmas, missing a-files for some g-file times, a diagnostic absent for a shot | dd7cad4, #130 |
| Tests | Tolerance loosened in a `tests/test_config_<ids>.yaml` without a stated reason | 895ccc1 (`v_loop`) |
| Tests | OMAS treated as ground truth where OMAS itself is wrong: matching OMAS hid a factor 100 because OMAS had the same bug; the comparison needs the MDSplus source and its units | #113, gafusion/omas#442 |
| Tests | Only checks shape or "it runs" | `.claude/TESTING_GUIDE.md` |
| Hygiene | Merge resolution that drops or duplicates code | #133 |
