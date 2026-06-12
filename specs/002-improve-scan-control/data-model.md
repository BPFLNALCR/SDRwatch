# Data Model: Improve Scan Control

This feature does not introduce persistent database entities. The model below describes operator-visible state and existing job payload data that the page must preserve.

## Scan Configuration

**Purpose**: Complete operator-selected configuration that the GUI converts into `/api/jobs` parameters.

**Fields**:

- `device_key`: selected SDR device key.
- `label`: job label, defaulting to the existing web label.
- `baseline_id`: selected monitoring location or baseline identifier.
- `params`: JSON-compatible scan parameter object using existing parameter names.

**Relationships**:

- References one selected baseline/location.
- References one selected SDR device.
- Uses enabled monitoring zones to derive `params.start` and `params.stop`.
- May include diagnostics, tuning, expert, run-mode, and output parameters.

**Validation Rules**:

- `baseline_id` is required before start.
- `device_key` is required before start.
- At least one monitoring zone must be enabled before start.
- `params` must preserve existing `/api/jobs` parameter names.
- Empty optional controls must be omitted or handled the same way the current GUI handles them.

## Basic Controls

**Purpose**: Normal operator workflow controls for running scans.

**Fields**:

- Baseline/location selection.
- SDR device selection and refresh.
- Monitoring zones selection and enable/disable actions.
- Run mode and duration when timed mode is selected.
- Diagnostics mode toggle.
- Start and stop actions.
- Live log visibility.

**Validation Rules**:

- Start is disabled or blocked until baseline and at least one monitoring zone are selected.
- Diagnostics mode does not require a manual diagnostic path.
- Duration is relevant only for timed run mode.

## Tuning Controls

**Purpose**: Common RF tuning controls presented with safer bounded UI elements and short descriptions.

**Fields**:

- `threshold_db`: slider with live numeric value.
- `guard_bins`: slider with live numeric value.
- `min_width_bins`: slider with live numeric value.
- `cfar`: bounded selection.
- `cfar_alpha_db`: slider with live numeric value.
- `cfar_quantile`: slider with live numeric value.
- `fft`: bounded selection.
- `avg`: bounded selection or segmented control.
- `samp_rate`: bounded selection.
- `gain`: auto/manual choice with explicit manual value handling.

**Validation Rules**:

- Slider values stay within documented UI bounds.
- Visible numeric values update when sliders change.
- Manual gain value is submitted only when manual gain is explicitly selected.
- Control descriptions or tooltips are visible or discoverable.
- FFT help text describes scan speed versus frequency-resolution and characterization tradeoffs.
- Gain help text explains baseline/detection repeatability and overload risk.

## GUI Tuning Preset

**Purpose**: Named operator-facing configuration that applies a coherent set of scan parameters through the web GUI while preserving existing `/api/jobs` names.

**Fields**:

- `preset_id`: stable identifier such as `rtl_v4_discovery`, `stable_baseline`, or `fast_wide_survey`.
- `label`: operator-visible preset name.
- `description`: short explanation of when to use the preset.
- `tradeoff_summary`: concise speed, FFT, gain, and persistence notes.
- `params`: parameter values to apply to the scan configuration.
- `hardware_hint`: optional text such as RTL-SDR Blog v4.

**Recommended Presets**:

- `rtl_v4_discovery`: first-light RTL-SDR v4 preset using fixed manual gain around `30 dB`, fast wide-sweep settings, and relaxed promotion gates so initial signal cards can appear.
- `stable_baseline`: slower preset using overlapping windows, higher FFT/averaging, fixed manual gain, and stricter multi-window persistence for cleaner baseline work.
- `fast_wide_survey`: fast broad-scan preset using lower FFT/averaging and relaxed promotion gates for broad RF visibility.

**Validation Rules**:

- Applying a preset updates the same controls and generated job parameters used by manual edits.
- Preset values are visible through Copy current scan settings before the operator starts a scan.
- Presets do not introduce new `/api/jobs` parameter names unless a separate contract update is planned.
- Presets do not instruct the operator to run scanner CLI commands.
- The UI does not claim FFT alone fixes zero-card behavior.

## Expert Controls

**Purpose**: Less common, compatibility-sensitive, or higher-risk settings that should remain available outside the normal workflow.

**Fields**:

- `cluster_merge_hz`
- `max_detection_width_hz`
- `max_detection_width_ratio`
- `new_ema_occ`
- `persistence_mode`
- `persistence_hit_ratio`
- `persistence_min_seconds`
- `persistence_min_hits`
- `persistence_min_windows`
- Revisit controls: `two_pass`, `revisit_fft`, `revisit_avg`, `revisit_margin_hz`, `revisit_span_limit_hz`, `revisit_max_bands`, `revisit_floor_threshold_db`
- Path/output controls: `db`, `jsonl`, and `diagnostic_jsonl` only if retained for compatibility
- Existing compatibility controls not promoted to Basic or Tuning, such as profile, bandplan, driver, CFAR train/guard, step, sleep between sweeps, latitude/longitude, and spur calibration mode

**Validation Rules**:

- Expert controls do not appear as part of Basic controls.
- Existing parameter names are preserved when submitted.
- Path controls do not become required for normal diagnostics mode.

## Safe Defaults

**Purpose**: Known values restored by the visible reset action.

**Fields**:

- Default value for every Basic, Tuning, and Expert setting.
- Default selected preset or explicit "custom/default" state.
- Current database path default provided by the rendered page.
- Empty/default markers for optional expert values that currently mean "auto", "none", or "use scanner default".

**Validation Rules**:

- Reset restores all scan settings, not only visible controls.
- Reset does not erase required context that is intentionally selected by the operator unless the documented page default is empty.
- Reset leaves the page in a state where the operator can inspect and adjust before starting a scan.
- Reset restores the documented first-light/default preset behavior chosen for this feature, not the older auto-gain/non-promoting defaults.

## Generated Job Parameters

**Purpose**: JSON object produced by the GUI for current settings and used for both scan submission and copied settings.

**Fields**:

- `device_key`
- `label`
- `baseline_id`
- `params`

**Validation Rules**:

- Must be valid JSON when copied.
- Must use existing parameter names.
- Must match what `POST /api/jobs` would receive for the same page state.
- Must omit unavailable optional values consistently with current submission behavior.

## Generated Scanner Command

**Purpose**: Optional internal/debug context copied from existing controller job metadata.

**Fields**:

- `cmd`: command array or command text from active/recent job detail when available.
- Job identifier associated with the command.
- Visible internal/debug label.

**Validation Rules**:

- Command copy is unavailable before the controller exposes a command for a job.
- Browser code must not generate scanner commands independently.
- Command copy is not presented as the primary operator workflow.

## State Transitions

```text
No baseline selected
  -> baseline selected
  -> zones loaded
  -> one or more zones enabled
  -> ready to start
  -> starting job
  -> running job with logs
  -> stopped or finished job
```

Settings-related transitions:

```text
Safe defaults
  -> operator selects a GUI tuning preset
  -> operator changes Basic/Tuning/Expert controls
  -> copied settings JSON can be generated
  -> reset returns all controls to safe defaults
```

Diagnostics-related transitions:

```text
Diagnostics mode off
  -> operator enables Diagnostics mode
  -> submitted params include diagnostics mode
  -> controller may add generated diagnostic path
  -> raw path entry is not required for normal use
```
