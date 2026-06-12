# Quickstart: FM Detection Card Stability Validation

This guide validates FM Detection Card Stability through no-hardware tests first, then through the SDRwatch web GUI and controller job lifecycle. Scanner CLI checks are internal backend smoke tests only.

## Prerequisites

- Current baseline is `devControl`.
- The active feature directory is `specs/003-fm-detection-card-stability`.
- The diagnostic evidence report exists at `docs/FM_DETECTION_CARD_EXPLOSION_REPORT.md`.
- SDRwatch web app and controller are available for manual validation.
- For hardware acceptance, Raspberry Pi 5 with RTL-SDR Blog v4 is available in an RF environment with FM broadcast activity.

## Test-First Automated Checks

Add or extend tests before implementation. Target these behaviors:

1. A wide/spiky FM-like signal does not create dozens of persisted cards under FM Validation settings.
2. Multiple separated FM-like signals remain separate.
3. Narrow non-FM signals remain narrow outside FM-specific behavior.
4. Repeated nearby detections update existing baseline rows.
5. Width floor/cap behavior is bounded and visible.
6. Two-pass behavior is covered if FM Validation enables or exposes it.
7. GUI FM Validation submits correct `/api/jobs` params.
8. Discovery remains available and first-light card-producing.
9. Diagnostics export includes create/update/no-match/missing/width/revisit evidence.

Suggested focused command in this Windows/Codex environment:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/test_control_page_scan_settings.py tests/test_detection_diagnostics.py tests/test_extent_hysteresis.py tests/test_segment_splitting.py tests/test_control_diagnostics_mode.py -q --basetemp .pytest-tmp
```

Add new test files as needed for `BaselinePersistence` and FM Validation profile behavior.

Expected automated outcomes:

- FM Validation fixture produces a bounded card count for one FM-like station-scale signal.
- Separated FM-like signals are not merged.
- Non-FM narrow fixture remains narrow.
- Baseline upsert tests show updates instead of repeated inserts for nearby FM fragments.
- Width decisions show configured min/max behavior.
- GUI payload tests prove FM Validation includes `profile=fm_broadcast` and/or explicit FM-specific params.
- Discovery tests still pass unchanged.

## GUI Payload Validation

1. Open the scan/control page.
2. Select or create a baseline.
3. Enable an FM Broadcast monitoring zone covering 88-108 MHz.
4. Select the RTL-SDR Blog v4 device.
5. Select FM Validation.
6. Enable Diagnostics mode.
7. Use Copy current scan settings.
8. Confirm the copied JSON has `device_key`, `label`, `baseline_id`, and `params`.
9. Confirm `params.start=88000000` and `params.stop=108000000` for the FM zone.
10. Confirm FM Validation uses existing params and includes the planned FM behavior:
    - `profile: "fm_broadcast"` and/or explicit FM stabilization params.
    - `two_pass: true` if enabled explicitly.
    - bounded revisit settings if two-pass is enabled.
    - width cap/min-width or profile behavior that prevents sub-kHz card explosion.
11. Switch back to RTL-SDR v4 Discovery.
12. Copy settings again and confirm Discovery remains present with relaxed first-light values.

## Controller Command Validation

Start an FM Validation job from the web GUI.

Expected controller evidence:

- Job starts through `POST /api/jobs`.
- Job detail includes controller params and generated `cmd` when available.
- If FM Validation explicitly enables two-pass, command includes `--two-pass`.
- If FM Validation relies on profile-applied two-pass, diagnostic tuning params show effective `two_pass=true`.
- If FM Validation relies on `fm_broadcast`, command includes `--profile fm_broadcast`.
- Diagnostics mode uses a controller-generated diagnostic path; the operator does not type one.

## Hardware Acceptance

Run on Raspberry Pi 5 with RTL-SDR Blog v4 through the web GUI.

1. Open the SDRwatch web GUI.
2. Select the target baseline.
3. Enable the FM Broadcast monitoring zone.
4. Select RTL-SDR Blog v4.
5. Select FM Validation.
6. Enable Diagnostics mode.
7. Start the scan from the page.
8. Let multiple sweeps complete.
9. Confirm signal cards appear in the GUI and do not explode into hundreds of tiny cards.
10. Export a diagnostic bundle from the GUI.
11. Confirm `baseline-detections.json` is not dominated by 293-880 Hz widths.
12. Confirm active detections are not dominated by sub-5 kHz widths.
13. Confirm diagnostics show create/update/no-match/missing/width/revisit decisions.
14. Confirm `scan_updates` and diagnostic records explain whether two-pass ran.

Discovery regression check:

1. Select RTL-SDR v4 Discovery.
2. Start a diagnostics-mode scan from the GUI.
3. Confirm it still produces first-light cards.
4. Confirm Discovery is documented as noisier/first-light and not the FM Validation stability path.

## Backend Smoke Checks

Backend smoke checks may verify scanner/profile plumbing, but they do not replace GUI acceptance.

Useful smoke targets:

- `python -m sdrwatch.cli --list-profiles` lists `fm_broadcast`.
- Controller `_build_cmd` includes `--profile fm_broadcast` and `--two-pass` when FM Validation submits those params.
- Scanner diagnostic tuning params show effective FM profile values when `profile=fm_broadcast` is applied.

## Expected Outcome

- FM Validation through the GUI produces a reasonable, stable set of FM-band cards, not hundreds of tiny 293-880 Hz cards.
- Discovery remains available and card-producing.
- Multiple FM-like signals remain separate.
- Narrow non-FM behavior remains narrow.
- Width clamp and two-pass behavior are tested and observable.
- Diagnostics clearly show create/update/merge/missing/revisit decisions.
