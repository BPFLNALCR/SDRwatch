# Quickstart: Cross-Sweep Persistence and Telemetry

This guide describes the validation path for the Cross-Sweep Persistence and Telemetry feature. It is written for planning and task generation; implementation may add exact commands or update results as tasks land.

## Prerequisites

- Current git branch: `007-cross-sweep-persistence-and-telemetry`
- Behavioral baseline: `006-fm-signal-characterization`
- Active Spec Kit feature directory: `specs/007-cross-sweep-persistence-and-telemetry`
- Existing FM characterization artifacts remain under `specs/004-fm-signal-characterization`
- SDRwatch operator acceptance remains web UI -> controller job -> diagnostic bundle export
- Scanner CLI checks are backend smoke and parameter parity checks only

## Test-First Work Before Runtime Changes

Add tests before changing scanner, controller, diagnostics, or telemetry behavior.

Required first tests:

1. Cross-sweep promotion:
   simulate a stable signal that appears once per complete sweep loop in a non-overlapping window and assert promotion after the configured number of loop observations.
2. Same-loop non-promotion:
   assert repeated observations from one sweep loop do not satisfy a multi-loop threshold.
3. Nearby signal separation:
   assert bounded center/span/width matching keeps nearby incompatible signals separate.
4. FM control-band non-regression:
   assert two close but separable FM-like signals remain separate with bounded display widths and raw/measured/match/display fields preserved.
5. Controller passthrough:
   submit all supported characterization, revisit, persistence, and width params through controller/API fixtures and assert scanner invocation parity or documented unsupported mappings.
6. Effective-parameter manifest:
   assert in-band and out-of-band FM profile requests record requested/applied/skipped/fallback/final values.
7. Diagnostic summary:
   assert aggregate event counts exist for emitted, rejected, persistence match/no-match, width, revisit, and characterization events.
8. Device telemetry:
   assert requested and actual gain when available, gain mode, device identity/index when available, sample rate, FFT size, and bin width are represented.

## Suggested No-Hardware Test Files

- `tests/test_fm_persistence_stability.py`
- `tests/test_fm_persistence_diagnostics.py`
- `tests/test_fm_characterization.py`
- `tests/test_fm_characterization_persistence.py`
- `tests/test_fm_characterization_diagnostics.py`
- `tests/test_control_fm_validation.py`
- `tests/test_control_page_scan_settings.py`
- `tests/test_web_diagnostics_bundle.py`
- New focused files if clearer:
  - `tests/test_cross_sweep_persistence.py`
  - `tests/test_effective_parameter_manifest.py`
  - `tests/test_device_telemetry.py`

## Suggested Windows Test Command Shape

Use bundled Python and a fresh workspace-local `--basetemp`.

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/test_cross_sweep_persistence.py tests/test_fm_persistence_stability.py tests/test_fm_characterization.py tests/test_fm_characterization_persistence.py tests/test_fm_characterization_diagnostics.py tests/test_control_fm_validation.py tests/test_control_page_scan_settings.py tests/test_web_diagnostics_bundle.py -q --basetemp .pytest-tmp-cross-sweep
```

If new files have not been created yet, run the nearest existing subset after tasks are generated.

## Backend Smoke Checks

These checks verify scanner/controller plumbing only. They do not replace GUI acceptance.

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m sdrwatch.cli --list-profiles
```

Expected backend smoke outcomes:

- `fm_broadcast` is present.
- FM profile fields include stable-card controls such as step, revisit, center match, match/display bandwidth settings, and width caps.
- Controller command construction tests prove web/API params map to scanner flags.

## GUI And Controller Acceptance Path

1. Start the controller service.
2. Start the SDRwatch web UI.
3. Create or select a baseline.
4. Select an RTL-SDR device.
5. Select FM Validation.
6. Enable diagnostics mode.
7. Use Copy current scan settings and confirm the JSON keeps `{device_key, label, baseline_id, params}`.
8. Start the scan from the browser.
9. Let at least several complete sweep loops run.
10. Export a diagnostic bundle from the GUI.
11. Confirm the bundle includes:
    - effective-parameter manifest
      - archive path: `job/effective-parameters.json`
      - JSONL event: `effective_parameters`
    - requested and applied or skipped profile state
    - final effective scanner parameters
    - structured aggregate decision counts
      - archive path: `diagnostics/decision-summary.json`
    - raw/measured/match/display characterization summaries
    - requested gain, actual gain when available, gain mode, device metadata, sample rate, FFT, and bin width
      - archive path: `job/device-telemetry.json`
      - JSONL event: `device_telemetry`
12. Confirm FM cards remain stable, separated, and bounded.
13. Repeat Discovery from the GUI and confirm it remains available as first-light behavior.

## Out-Of-Band Profile Validation

1. Launch a scan through the web/API path with `profile=fm_broadcast` and a range outside 88-108 MHz.
2. Export diagnostics.
3. Confirm the manifest records:
   - requested profile `fm_broadcast`
   - `profile_applied=false`
   - a skipped reason for invalid range
   - fallback/default values
   - final effective scanner parameters

## Hardware Telemetry Validation

This step requires current SDR hardware and should remain open when not run.

1. Run FM Validation from the GUI on Raspberry Pi 5 with RTL-SDR Blog v4.
2. Use fixed manual gain and diagnostics mode.
3. Export a bundle.
4. Confirm requested gain and gain mode are present.
5. Confirm actual gain and supported gains are present when exposed by the driver.
6. Confirm device identity/index/serial/tuner fields are present when exposed, otherwise null or unavailable.
7. Confirm sample rate, FFT, and bin width are present.

## Expected Outcome

- A stable repeated signal is not missed solely because it appears once per complete sweep loop.
- Same-loop repeats do not falsely satisfy strict multi-loop persistence.
- FM Broadcast remains a reliable control band with stable, separated, bounded cards.
- Diagnostic bundles are sufficient to reconstruct important scanner/profile/controller decisions.
- Web/API scans and scanner smoke invocations use equivalent supported characterization parameters.
- Device and gain telemetry make receiver-state changes visible without failing scans when fields are unavailable.

## Implementation Validation Log

- 2026-06-14 T001 branch check: `git branch --show-current` returned `007-cross-sweep-persistence-and-telemetry`.
- 2026-06-14 T063 backend smoke: `python -m sdrwatch.cli --list-profiles` completed successfully and included `fm_broadcast` with `persistence_min_sweep_loops`, match/display bandwidth controls, width caps, revisit settings, and centroid controls.
- 2026-06-14 T064 hardware telemetry acceptance: not run in this no-hardware environment; Raspberry Pi 5 plus RTL-SDR Blog v4 validation remains the manual acceptance step above.
