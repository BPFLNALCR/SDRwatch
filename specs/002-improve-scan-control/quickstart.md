# Quickstart: Improve Scan Control Validation

This guide validates the cleaned-up scan/control page through the web GUI and controller job lifecycle. Scanner CLI commands are not part of operator acceptance for this feature.

## Prerequisites

- SDRwatch web app and controller are available in the local environment.
- At least one monitoring location or baseline exists, or the tester can create one from the web page.
- A controller device response is available. Real SDR hardware is preferred for full acceptance, but no-hardware UI and fake-controller tests should cover page rendering and payload behavior.
- Browser clipboard permission is available for the happy-path copy checks, or the tester is prepared to verify the visible failure message.

## Automated No-Hardware Checks

After implementation, run focused tests for the web control behavior and existing diagnostics compatibility:

```powershell
python -m pytest tests/test_control_page_scan_settings.py tests/test_web_diagnostics_bundle.py tests/test_control_diagnostics_mode.py -q
```

In the Codex Windows sandbox used for this implementation, `python` and `py` were not on `PATH`, and the default pytest temp directory under `AppData\Local\Temp` was not readable. The equivalent verified command was:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest tests/test_control_page_scan_settings.py tests/test_web_diagnostics_bundle.py tests/test_control_diagnostics_mode.py -q --basetemp .pytest-tmp
```

Verification result on June 10, 2026: `19 passed in 1.05s`.

Expected results:

- Rendered control page contains Basic, Tuning, and Expert sections.
- Required sliders, selects, reset, copy settings, diagnostics mode, start/stop, and live logs controls are present.
- Scan form and numeric controls suppress autocomplete.
- GUI-generated settings use the existing `/api/jobs` payload shape and parameter names.
- Diagnostics-mode tests continue to pass.

If `python` or `pytest` is unavailable in the local shell, record that limitation and complete the GUI manual verification below.

## Implementation Smoke Notes

- Server-render smoke completed with the local Flask app on `http://127.0.0.1:8769/control`: status `200`, with Basic, Tuning, and Expert sections present.
- A Playwright browser smoke through the Codex Node runtime was attempted, but the Windows sandbox blocked child-process setup for the Flask server. Use the GUI checklist below for final operator UAT in a normal browser session.
- Scanner CLI commands are not part of operator acceptance for this page; the generated scanner command copy action remains labeled internal/debug and depends on controller-provided job metadata.
- Final diff review: runtime implementation changes are limited to `templates/control.html`, with new no-hardware tests and Spec Kit artifacts. Scanner, controller, database, detection, DSP, baseline-helper, and dashboard implementation files were not changed.

## GUI Manual Verification

1. Open the SDRwatch web GUI in a browser and navigate to the scan/control page.
2. Confirm Basic controls show only the normal workflow: monitoring location or baseline, SDR device, monitoring zones, run mode, Diagnostics mode, start/stop, and live logs.
3. Select or create a monitoring location.
4. Select an SDR device or confirm the page shows a clear no-device state.
5. Enable one or more monitoring zones.
6. Choose each run mode and confirm timed mode shows its duration control only when relevant.
7. Enable Diagnostics mode and confirm no diagnostic path entry is required in Basic controls.
8. Open Tuning controls.
9. Move `threshold_db`, `guard_bins`, `min_width_bins`, `cfar_alpha_db`, and `cfar_quantile` sliders and confirm their visible numeric values update immediately.
10. Change CFAR mode, FFT, averaging, sample rate, and gain controls and confirm each uses bounded choices or an explicit auto/manual pattern.
11. Confirm each Tuning control has a short description or tooltip.
12. Open Expert controls and confirm less common settings are grouped there, including persistence, revisit, width/cluster, database path, JSONL path, and any retained raw diagnostic path override.
13. Change representative Basic, Tuning, and Expert settings.
14. Use "Copy current scan settings" and paste into a text editor. Confirm the result is valid JSON with `device_key`, `label`, `baseline_id`, and `params`.
15. Confirm the copied `params` names match the existing `/api/jobs` names, such as `threshold_db`, `guard_bins`, `min_width_bins`, `cfar`, `cfar_alpha_db`, `cfar_quantile`, `fft`, `avg`, `samp_rate`, and `gain`.
16. Use "Reset to safe defaults" and confirm all changed settings return to defaults and slider values update.
17. Start a scan from the GUI.
18. Confirm the page enters running state and live logs are visible or polling.
19. If a generated scanner command copy action appears, confirm it is labeled internal/debug and is available only for the active or recent job.
20. Stop the scan from the GUI and confirm the page reflects the stopped or idle state.

## Autofill Check

1. In the same browser profile, store or trigger unrelated autofill values if available.
2. Reload the scan/control page.
3. Confirm scan settings are not silently replaced by unrelated browser autofill values.
4. Submit only after explicitly reviewing the GUI-generated settings JSON.

## Expected Outcome

- The operator can complete the scan workflow from the web page without scanner CLI instructions.
- Common tuning controls are safer and easier to inspect.
- Expert controls remain available but separated from the normal workflow.
- Reset restores known safe defaults.
- Copied settings accurately represent the GUI-generated job payload.
- Existing controller job lifecycle and logs remain intact.
