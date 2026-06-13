# Quickstart: FM Signal Characterization

This guide describes the intended validation path for the FM Signal Characterization feature. It is a planning artifact for the first Spec Kit pass; no runtime implementation has been added yet in this feature directory.

## Prerequisites

- Current git branch: `006-fm-signal-characterization`
- Active Spec Kit feature directory: `specs/004-fm-signal-characterization`
- Baseline comparison branch remains `devControl`
- Existing FM stability reference artifacts are available under `specs/003-fm-detection-card-stability/`
- SDRwatch continues to be validated through the web UI and controller lifecycle, with scanner CLI checks used only as backend smoke coverage

## Test-First Work Before Any Runtime Change

Add or extend automated tests for the requested characterization cases before changing scanner, persistence, or diagnostic behavior.

Required cases:

1. Synthetic FM-like wide and spiky signal:
   stable card count remains bounded
   measured bandwidth is tracked separately from display width
2. Tiny FFT fragments:
   fragments do not become fake measured FM bandwidth on their own
3. Revisit refinement:
   revisit updates characterization fields without exploding cards
4. Nearby FM-like stations:
   stations remain separate
5. Narrow non-FM signal:
   signal remains narrow outside FM Validation and is not labeled as an FM candidate from bandplan context alone
6. Bandplan separation:
   contextual service or profile labels remain separate from measured characterization and classification evidence
7. Diagnostic bundle:
   export includes raw, measured, match, and display span summaries
8. Persistence invariant:
   `f_low_hz <= f_center_hz <= f_high_hz` remains true after revisit and hysteresis updates

Suggested new or expanded test files:

- `tests/test_fm_characterization.py`
- `tests/test_fm_characterization_diagnostics.py`
- `tests/test_fm_characterization_persistence.py`
- existing FM stability tests under `tests/test_fm_persistence_stability.py`
- existing diagnostics coverage under `tests/test_web_diagnostics_bundle.py`

Suggested Windows and Codex environment command shape:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/test_fm_persistence_stability.py tests/test_fm_persistence_diagnostics.py tests/test_non_fm_width_scope.py tests/test_fm_validation_profile.py tests/test_web_diagnostics_bundle.py -q --basetemp .pytest-tmp
```

Extend the command with any new characterization-focused tests before implementation.

## GUI And Controller Acceptance Path

1. Open the SDRwatch web UI.
2. Select or create a baseline.
3. Select an RTL-SDR Blog v4 device.
4. Select FM Validation, not Discovery.
5. Enable diagnostics mode.
6. Start the scan through the existing web and controller workflow.
7. Confirm stable station-scale cards still appear for FM Validation.
8. Export a diagnostic bundle from the GUI.
9. Verify the exported evidence separates:
   - raw detector segment span
   - measured center and occupied bandwidth
   - persistence or match span
   - display or card span
   - contextual bandplan or profile metadata
10. Confirm that any reported classification candidate includes evidence sources and is not based only on contextual labels.

## Discovery Regression Check

1. Return to the web UI control page.
2. Select RTL-SDR v4 Discovery.
3. Start a normal first-light scan through the controller-backed workflow.
4. Confirm Discovery remains available and distinct from FM Validation.
5. Confirm Discovery still behaves as the first-light preset rather than silently inheriting characterization-specific FM behavior.

## Diagnostics Acceptance Checklist

For a successful FM characterization run, the bundle should make these facts explicit:

- raw segment width is visible
- measured occupied bandwidth is visible
- match span is visible
- display span is visible
- contextual bandplan and profile labels are separate from measured evidence
- revisit-derived evidence is visible when revisit ran
- confidence explains its evidence sources
- bounded export limits still report missing or truncated evidence in `manifest.json`

## Backend Smoke Checks

Backend smoke checks are useful for plumbing, but they do not replace GUI acceptance.

Useful later checks:

- `python -m sdrwatch.cli --list-profiles`
- controller command construction for FM Validation
- effective tuning params in diagnostic window records

These checks confirm backend behavior only. The operator acceptance path remains the browser and controller lifecycle.

## Expected Outcome

- FM Validation continues to show stable station-scale cards.
- Measured RF characterization is available separately from display behavior.
- Nearby FM stations remain separate.
- Narrow non-FM signals are not forced into FM classification.
- Diagnostics become the first proof surface for characterization quality.
- Any later database expansion is justified by clear evidence rather than guessed up front.
