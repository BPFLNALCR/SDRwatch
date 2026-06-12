# UI Contract: Scan/Control Page

This contract defines the expected operator-facing behavior of the scan/control page after the cleanup.

## Page Scope

- Route: existing scan/control page.
- Primary user: SDRwatch operator using the browser.
- Primary workflow: select baseline/location, select device, select or enable monitoring zones, choose run mode, optionally enable Diagnostics mode, start/stop scan, view live logs.
- Out of scope: new dashboard pages, scanner DSP algorithm rewrites, database schema changes, simulation mode, blind CFAR/detection rewrites, or CLI-first operator instructions.

## Required Sections

### Basic Controls

Basic controls must contain the normal operator workflow:

- Monitoring location or baseline selection.
- New location action if already present.
- SDR device selection and refresh.
- Monitoring zone selection and enable/disable actions.
- Run mode selection and duration input for timed mode.
- Diagnostics mode toggle.
- Start and stop controls.
- Live logs access.

Basic controls must not expose tuning and expert-only raw numeric fields as part of the normal workflow.

### GUI Tuning Presets

The page must expose operator-facing presets that apply existing scan parameters through the GUI:

| Preset | Purpose | Expected Parameter Direction |
| --- | --- | --- |
| RTL-SDR v4 Discovery | First-light real-hardware cards | Fixed manual gain near `30 dB`, fast `2.4e6` step, relaxed `persistence_min_hits=1` and `persistence_min_windows=1`, FFT `4096` or `8192`, avg `8` |
| Stable Baseline | Slower cleaner baseline scans | Fixed manual gain around `25-30 dB`, overlapping `1.2e6` step, FFT `8192`, avg `16`, stricter `2/2` persistence |
| Fast Wide Survey | Broad scan speed | Fixed manual gain near `30 dB`, `2.4e6` step, FFT `4096`, avg `8`, relaxed promotion gates |

Preset rules:

- Applying a preset updates visible controls and the generated `/api/jobs` parameters.
- The preset must not submit a new backend-only preset identifier unless separately implemented and documented.
- Copy current scan settings must show the applied preset values using existing parameter names.
- Preset text must explain that FFT controls scan speed and characterization quality, while the diagnosed zero-card failure is promotion/persistence.
- Preset text must keep fixed-gain overload risk visible and configurable.

### Tuning Controls

Tuning controls must contain common RF adjustment controls:

| Parameter | Required UI | Value Visibility | Help Text |
| --- | --- | --- | --- |
| `threshold_db` | Slider | Live numeric value | Explains detection threshold above noise |
| `guard_bins` | Slider | Live numeric value | Explains tolerated below-threshold bins inside a detection |
| `min_width_bins` | Slider | Live numeric value | Explains minimum contiguous detection width |
| `cfar` | Select | Selected option | Explains CFAR mode |
| `cfar_alpha_db` | Slider | Live numeric value | Explains CFAR threshold scaling override |
| `cfar_quantile` | Slider | Live numeric value | Explains OS-CFAR quantile |
| `fft` | Select | Selected option | Explains FFT size tradeoff |
| `avg` | Select or segmented control | Selected option | Explains averaging speed/noise tradeoff |
| `samp_rate` | Select | Selected option | Explains sample rate effect |
| `gain` | Auto/manual control | Selected mode and manual value when relevant | Explains automatic versus manual gain, repeatability, and overload risk |

### Expert Controls

Expert controls must contain less common, compatibility-sensitive, or higher-risk controls:

- `cluster_merge_hz`
- `max_detection_width_hz`
- `max_detection_width_ratio`
- `new_ema_occ`
- `persistence_mode`
- `persistence_hit_ratio`
- `persistence_min_seconds`
- `persistence_min_hits`
- `persistence_min_windows`
- Revisit controls
- `db`
- `jsonl`
- Raw diagnostic path override only if still required for compatibility
- Existing compatibility controls not assigned to Basic or Tuning

Expert controls may be collapsed by default if they remain discoverable.

## Autofill Contract

- The scan settings form must set `autocomplete="off"`.
- Numeric scan controls must set `autocomplete="off"`.
- Path and text controls used for scan parameters should also suppress autocomplete when practical.
- Baseline saved in local storage is allowed because it is an intentional page behavior, not browser autofill.

## Reset Contract

The page must provide a visible "Reset to safe defaults" action.

When activated:

- All Basic, Tuning, and Expert controls return to documented safe defaults.
- Slider visible values update immediately.
- Gain returns to the documented first-light/default preset behavior. For RTL-SDR v4 detection presets this should be fixed manual gain unless implementation chooses another documented default.
- Optional expert values return to the page's default blank/auto/none semantics.
- The operator can still review settings before starting a scan.

## Copy Current Scan Settings Contract

The page must provide a visible "Copy current scan settings" action.

The copied value must be valid JSON with this shape:

```json
{
  "device_key": "rtl:0",
  "label": "web",
  "baseline_id": 1,
  "params": {
    "start": 88000000,
    "stop": 108000000,
    "samp_rate": 2400000,
    "fft": 4096,
    "avg": 8
  }
}
```

Rules:

- Values shown above are examples only.
- The JSON must use the same parameter names the GUI submits to `/api/jobs`.
- Enabled monitoring zones determine `params.start` and `params.stop`.
- Selected presets determine values such as `gain`, `step`, `fft`, `avg`, and persistence gates until the operator overrides them.
- Diagnostics mode adds the existing diagnostics parameter used by the current web/controller workflow.
- Copy failure must be visible to the operator if browser clipboard access is blocked.

## Internal/Debug Command Copy Contract

The page may provide "Copy generated scanner command (internal/debug)" only when an active or recent job detail already includes a generated command from the controller.

Rules:

- The action must not appear as the main scan workflow.
- The action must not ask the operator to run the command.
- Browser code must not construct a scanner command from form fields.
- If command metadata is missing, the action must be omitted or disabled with a clear unavailable state.

## Start/Stop Contract

- Start posts to the existing job endpoint with the existing payload shape.
- Stop uses the existing job stop workflow.
- Live logs continue to poll the existing logs endpoint for the active or recent job.
- Diagnostics export behavior, if present from prior work, must continue to work.
- Real-hardware validation for the RTL-SDR v4 Discovery preset must confirm signal cards appear through the web UI.

## Error and Empty-State Contract

- No baseline: start unavailable or blocked with a clear message.
- No zones enabled: start unavailable or blocked with a clear message.
- No devices: device selection shows unavailable state without exposing scanner CLI instructions.
- Clipboard blocked: copy status reports failure without changing settings.
- Logs unavailable: page remains usable and can retry polling.
