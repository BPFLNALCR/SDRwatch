# Quickstart: FM Signal Characterization

This guide describes the current validation path for the FM Signal Characterization feature. The first implementation pass now includes diagnostics-first characterization scaffolding, bounded export summaries, and no-hardware FM regression coverage without a schema migration.

## Prerequisites

- Current git branch: `006-fm-signal-characterization`
- Active Spec Kit feature directory: `specs/004-fm-signal-characterization`
- Baseline comparison branch remains `devControl`
- Existing FM stability reference artifacts are available under `specs/003-fm-detection-card-stability/`
- SDRwatch continues to be validated through the web UI and controller lifecycle, with scanner CLI checks used only as backend smoke coverage

## Test-First Work Before Any Runtime Change

Add or extend automated tests for the requested characterization cases before broadening scanner, persistence, or diagnostic behavior further.

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

Current focused US1 command and result on branch `006-fm-signal-characterization`:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/test_fm_characterization.py tests/test_fm_characterization_persistence.py tests/test_fm_persistence_stability.py tests/test_fm_persistence_diagnostics.py tests/test_web_diagnostics_bundle.py -q --basetemp .pytest-tmp-char-us1-doc
```

Observed result on June 12, 2026:

```text
21 passed in 0.84s
```

This focused run currently covers:

- explicit `raw_*`, `measured_*`, `match_*`, and `display_*` characterization record shape
- wide and spiky FM-like bounded-card behavior with measured-versus-display width separation
- tiny-fragment protection so narrow FFT fragments do not inflate measured FM bandwidth by themselves
- bounded diagnostic bundle export plus center-within-span regression scanning
- existing FM stability and persistence diagnostics regressions

US1 implementation notes:

- T012 is satisfied by the focused FM stability and characterization tests: FM-like wide/spiky signals still produce bounded station-scale persisted rows while characterization records keep measured bandwidth separate from display bandwidth.
- T013 is satisfied through the existing scanner logging route: `ScannerRunner` mirrors `ScanLogger` output to `diagnostic_jsonl`, `DetectionEngine` emits `characterization_record` events on coarse-pass persistence, and the diagnostic bundle writes a bounded `diagnostics/characterization-summary.json`.
- `build_window_record()` remains unchanged in this pass because per-window detector diagnostics and persistent characterization evidence have different lifetimes; combining them would risk making one FFT window look like the final measured signal.

Invariant fix validation on June 13, 2026:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/test_extent_hysteresis.py tests/test_fm_persistence_stability.py tests/test_fm_persistence_diagnostics.py tests/test_fm_characterization.py tests/test_fm_characterization_persistence.py tests/test_fm_characterization_diagnostics.py tests/test_non_fm_width_scope.py tests/test_web_diagnostics_bundle.py tests/test_control_fm_validation.py tests/test_control_page_scan_settings.py -q --basetemp .pytest-tmp-invariant-broader
```

Observed result:

```text
56 passed in 1.85s
```

US3 revisit and center-stability validation on June 13, 2026:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/test_fm_characterization.py tests/test_fm_characterization_persistence.py tests/test_fm_characterization_diagnostics.py tests/test_extent_hysteresis.py tests/test_fm_persistence_stability.py tests/test_fm_persistence_diagnostics.py tests/test_non_fm_width_scope.py tests/test_web_diagnostics_bundle.py tests/test_control_fm_validation.py tests/test_control_page_scan_settings.py -q --basetemp .pytest-tmp-char-us3-broader
```

Observed result:

```text
60 passed in 2.36s
```

US3 implementation notes:

- Characterization records now carry `measured_center_hz`, `stable_center_hz`, `display_center_hz`, `center_delta_hz`, `center_stability_hz`, and `source_pass` so center jitter is visible without changing the persistent schema.
- FM Validation center updates now use a bounded stable-center smoother in persistence so repeated coarse fragment wobble is damped instead of moving the persisted/card center one-for-one.
- Revisit confirmations now emit `characterization_record` events with `source_pass="revisit"` and increment `revisit_measurement_count`, so diagnostic bundles can show revisit contribution directly instead of only `revisit_apply`.

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
   - stable center and center delta
   - persistence or match span
   - display or card span
   - contextual bandplan or profile metadata
10. Confirm the bundle now includes either `source_pass="revisit"` characterization records or nonzero `revisit_measurement_count` for stations that received revisit confirmation.
11. Confirm that any reported classification candidate includes evidence sources and is not based only on contextual labels.

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
- stable center and center delta are visible
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
