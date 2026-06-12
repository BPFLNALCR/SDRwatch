# Quickstart: GUI Diagnostic Capture Validation

## Prerequisites

- Current branch: `004-gui-diagnostics-capture`
- Active feature plan: `specs/001-gui-diagnostics-capture/plan.md`
- Local SDRwatch web app and controller code available.
- No SDR hardware is required for the no-hardware validation path.

## No-Hardware Automated Checks

Run the focused test suite after implementation:

```powershell
python -m pytest -q tests/test_web_diagnostics_bundle.py tests/test_control_diagnostics_mode.py
```

In the current Windows Codex shell, use the bundled Python runtime and a workspace-local pytest temp directory if `python` is not on `PATH` or the default temp directory is inaccessible:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest -q --basetemp .pytest-tmp tests/test_web_diagnostics_bundle.py tests/test_control_diagnostics_mode.py
```

Expected outcomes:

- Bundle creation succeeds with a temporary SQLite database, temporary scanner log, and temporary diagnostic JSONL file.
- The zip contains manifest, notes template, job metadata, params, scanner command, bounded logs, bounded diagnostic JSONL, baseline rows, baseline detection rows, scan update rows, monitoring zones, and friendly signals when present.
- Missing optional evidence is recorded in `manifest.json` instead of failing the export.
- Diagnostics mode scan start preserves `/api/jobs` behavior and results in a generated `diagnostic_jsonl` path only when diagnostics are enabled.

## GUI/Controller Acceptance Flow

1. Start the controller service in the normal project workflow.
2. Start the SDRwatch web dashboard in the normal project workflow.
3. Open the scan/control page in a browser.
4. Select or create a baseline/location.
5. Enable one or more monitoring zones.
6. Turn on Diagnostics mode.
7. Start the scan from the web page.
8. Confirm the job starts through the controller and the UI does not ask for a diagnostic file path.
9. During or after the scan, use Export diagnostic bundle for the active or recent job.
10. Open the downloaded zip and verify the archive includes `manifest.json`, a notes template, job metadata, controller params, scanner command, scanner log evidence, diagnostic JSONL evidence when available, and recent baseline/monitoring context.

Expected outcome:

- The operator completes diagnostics capture and export from the GUI without running scanner CLI commands manually.

## Compatibility Checks

Verify existing behavior still works:

```powershell
python -m pytest -q tests/test_detection_diagnostics.py tests/test_extent_hysteresis.py tests/test_segment_splitting.py
```

Expected outcomes:

- Detection diagnostics shape tests still pass.
- Extent hysteresis and segment splitting tests still pass.
- No test depends on SDR hardware.

## Documentation Review

Review updated operator documentation and confirm it describes GUI-based steps:

- Enable Diagnostics mode on the control page.
- Start a scan from the web UI.
- Export the diagnostic bundle from the web UI.
- Fill in the included notes template.

The operator documentation must not require scanner CLI commands for normal diagnostic capture.
