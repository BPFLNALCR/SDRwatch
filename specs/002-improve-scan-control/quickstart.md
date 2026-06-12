# Quickstart: Improve Scan Control Validation

This guide validates the cleaned-up scan/control page and real-hardware tuning presets through the web GUI and controller job lifecycle. Scanner CLI commands are not part of operator acceptance for this feature.

## Prerequisites

- SDRwatch web app and controller are available in the local environment.
- At least one monitoring location or baseline exists, or the tester can create one from the web page.
- A controller device response is available. Real SDR hardware is preferred for full acceptance, but no-hardware UI and fake-controller tests should cover page rendering and payload behavior.
- For hardware acceptance, a Raspberry Pi 5 with RTL-SDR Blog v4 is available in an RF environment where SDR++ or prior diagnostics show visible peaks.
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

Implementation verification result on June 12, 2026, after the real-hardware preset additions: `27 passed in 1.39s` using the bundled Python command above.

Expected results:

- Rendered control page contains Basic, Tuning, and Expert sections.
- Required sliders, selects, reset, copy settings, diagnostics mode, start/stop, and live logs controls are present.
- GUI tuning presets render and apply documented defaults for RTL-SDR v4 Discovery, Stable Baseline, and Fast Wide Survey.
- Scan form and numeric controls suppress autocomplete.
- GUI-generated settings use the existing `/api/jobs` payload shape and parameter names.
- Copied settings after preset selection include expected values for `gain`, `step`, `fft`, `avg`, and persistence gates.
- Diagnostics-mode tests continue to pass.

If `python` or `pytest` is unavailable in the local shell, record that limitation and complete the GUI manual verification below.

## Implemented Preset Values

| Preset | Gain | Sample rate | Step | FFT | Avg | Persistence gates |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RTL-SDR v4 Discovery | manual `30 dB` | `2400000` | `2400000` | `8192` | `8` | `min_hits=1`, `min_windows=1` |
| Stable Baseline | manual `30 dB` | `2400000` | `1200000` | `8192` | `16` | `min_hits=2`, `min_windows=2` |
| Fast Wide Survey | manual `30 dB` | `2400000` | `2400000` | `4096` | `8` | `min_hits=1`, `min_windows=1` |

The Discovery preset is the reset/default first-light preset. FFT remains a speed/resolution and characterization knob; the zero-card diagnostic failure was promotion/persistence, not candidate starvation.

## Implementation Smoke Notes

- Server-render smoke completed with the local Flask app on `http://127.0.0.1:8773/control`: status `200`, with RTL-SDR v4 Discovery plus Basic, Tuning, and Expert sections present.
- An in-app Browser smoke through the Codex Node runtime was attempted, but the Windows sandbox blocked browser setup with `CreateProcessAsUserW failed: 5`. Use the GUI checklist below for final operator UAT in a normal browser session.
- Scanner CLI commands are not part of operator acceptance for this page; the generated scanner command copy action remains labeled internal/debug and depends on controller-provided job metadata.
- Final diff review: runtime implementation changes should stay centered on `templates/control.html`, with new no-hardware tests and Spec Kit artifacts. Scanner, controller, database schema, CFAR, detection algorithm, baseline-helper, and dashboard implementation files should not change unless implementation discovers an obvious small compatibility bug.
- Real Raspberry Pi 5 plus RTL-SDR Blog v4 validation was not executed in this Codex environment. The Discovery preset still requires web-GUI hardware acceptance that signal cards appear and exported diagnostics show nonempty promoted detections and `baseline_detections`.

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
14. Select RTL-SDR v4 Discovery and confirm the UI applies fixed manual gain, fast step, FFT/avg values, and relaxed persistence gates.
15. Select Stable Baseline and confirm the UI applies overlapping step, higher FFT/avg values, and stricter persistence gates.
16. Select Fast Wide Survey and confirm the UI applies lower FFT/avg values for scan speed.
17. Confirm preset text explains that FFT affects scan speed and signal characterization, not the diagnosed zero-card promotion failure.
18. Confirm fixed-gain text warns about overload and remains configurable.
19. Use "Copy current scan settings" and paste into a text editor. Confirm the result is valid JSON with `device_key`, `label`, `baseline_id`, and `params`.
20. Confirm the copied `params` names match the existing `/api/jobs` names, such as `threshold_db`, `guard_bins`, `min_width_bins`, `cfar`, `cfar_alpha_db`, `cfar_quantile`, `fft`, `avg`, `samp_rate`, `step`, `gain`, `persistence_min_hits`, and `persistence_min_windows`.
21. Use "Reset to safe defaults" and confirm all changed settings return to documented defaults and slider values update.
22. Start a scan from the GUI.
23. Confirm the page enters running state and live logs are visible or polling.
24. If a generated scanner command copy action appears, confirm it is labeled internal/debug and is available only for the active or recent job.
25. Stop the scan from the GUI and confirm the page reflects the stopped or idle state.

## Real Hardware Preset Acceptance

Run this validation on Raspberry Pi 5 with RTL-SDR Blog v4 through the web GUI.

1. Open the control page in the browser.
2. Select the target monitoring location and enable a zone with known visible RF activity.
3. Select the RTL-SDR Blog v4 device.
4. Select the RTL-SDR v4 Discovery preset.
5. Enable Diagnostics mode.
6. Use "Copy current scan settings" and confirm the copied JSON includes fixed manual gain, `step=2.4e6`, `avg=8`, and relaxed promotion gates.
7. Start monitoring from the web GUI.
8. Watch live logs until at least one full sweep completes.
9. Confirm one or more signal cards appear in the web UI.
10. Export a diagnostic bundle from the web GUI.
11. Confirm the exported evidence has nonzero accepted hits, nonzero promoted detections, and nonempty `baseline_detections`.
12. If no cards appear, keep the bundle and document whether raw candidates and accepted hits are present before changing algorithms.

Stable Baseline follow-up:

1. Select the Stable Baseline preset.
2. Confirm copied settings show overlapping step, higher FFT/avg, and stricter `2/2` persistence.
3. Start from the web GUI and verify signal cards still appear, accepting slower sweep cadence.

## Autofill Check

1. In the same browser profile, store or trigger unrelated autofill values if available.
2. Reload the scan/control page.
3. Confirm scan settings are not silently replaced by unrelated browser autofill values.
4. Submit only after explicitly reviewing the GUI-generated settings JSON.

## Expected Outcome

- The operator can complete the scan workflow from the web page without scanner CLI instructions.
- Common tuning controls are safer and easier to inspect.
- GUI presets make the FFT, gain, speed, and persistence tradeoffs explicit.
- The RTL-SDR v4 Discovery preset produces initial signal cards on real hardware through the web UI.
- Expert controls remain available but separated from the normal workflow.
- Reset restores known safe defaults.
- Copied settings accurately represent the GUI-generated job payload.
- Existing controller job lifecycle and logs remain intact.
