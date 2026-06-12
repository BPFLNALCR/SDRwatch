# Data Model: FM Detection Card Stability

This feature does not introduce new persistent database tables. It describes operator-visible state, existing scanner/profile settings, existing baseline detections, and diagnostic evidence needed to stabilize FM-band cards.

## FM Validation Preset

**Purpose**: GUI-visible scan preset for validating the 88-108 MHz FM band without card explosion.

**Fields**:

- `preset_id`: stable GUI identifier, expected to be `fm_validation` or equivalent.
- `label`: operator-visible label, expected to include "FM Validation".
- `description`: concise explanation that this is a stable FM-band validation path, not first-light Discovery.
- `profile`: optional scanner profile name, preferably `fm_broadcast` if profile reuse remains the implementation path.
- `params`: existing `/api/jobs` parameter values applied by the GUI.
- `diagnostic_expectations`: expected evidence to look for in diagnostic bundles.

**Validation Rules**:

- Must be selectable and copyable from `templates/control.html`.
- Must submit existing `/api/jobs` names through `params`.
- Must not require the operator to run scanner CLI commands.
- Must not silently remove Discovery.
- Must make two-pass/profile behavior testable, either through submitted params or scanner diagnostic tuning params.

## Discovery Preset

**Purpose**: Existing RTL-SDR v4 first-light preset that proves end-to-end card production.

**Fields**:

- `preset_id`: existing Discovery identifier.
- `params`: relaxed first-light scan values, including relaxed promotion gates.
- `description`: states speed/first-light tradeoff.

**Validation Rules**:

- Remains available in the GUI.
- Copy current scan settings still shows card-producing first-light values.
- Does not inherit FM Validation-only constraints unless explicitly selected by the operator.

## FM Profile Application

**Purpose**: Scanner-side application of built-in FM-band defaults from `sdrwatch/io/profiles.py`.

**Fields**:

- `profile_name`: `fm_broadcast`.
- `frequency_bounds_hz`: `88000000` to `108000000`.
- `samp_rate`, `step_hz`, `fft`, `avg`, `gain_db`, `threshold_db`.
- CFAR fields: `cfar_train`, `cfar_guard`, `cfar_quantile`.
- Persistence fields: `persistence_min_hits`, `persistence_min_windows`, `persistence_hit_ratio`, `persistence_min_seconds`.
- Revisit fields: `two_pass`, `revisit_fft`, `revisit_avg`, `revisit_margin_hz`, `revisit_max_bands`, `revisit_floor_threshold_db`, `revisit_span_limit_hz`.
- Span and match fields: `cluster_merge_hz`, `center_match_hz`, `match_bandwidth_pad_hz`, `min_match_bandwidth_hz`, `display_bandwidth_pad_hz`, `min_display_bandwidth_hz`, `max_detection_width_ratio`, `max_detection_width_hz`.
- Segment shaping fields: `segment_center_mode`, `segment_centroid_span_hz`, `segment_centroid_drop_db`, `segment_centroid_floor_margin_db`.

**Validation Rules**:

- If FM Validation sets `profile=fm_broadcast`, tests must prove scanner-applied tuning params reflect the intended profile values.
- If the GUI sends explicit params that override profile values, tests must prove those overrides are intentional.
- Hidden profile-only fields must either remain intentionally profile-owned or gain narrow CLI/controller passthrough.

## Detection Fragment

**Purpose**: Narrow emitted segment from one sweep window.

**Fields**:

- `f_low_hz`
- `f_high_hz`
- `f_center_hz`
- `bandwidth_hz`
- `peak_db`
- `noise_db`
- `snr_db`
- `window_idx`
- `sweep_id`

**Validation Rules**:

- Raw fragments may remain narrow internally.
- FM Validation must prevent raw fragments from becoming unbounded numbers of user-visible cards.
- Diagnostic output should preserve enough fragment evidence for tuning without exporting huge raw arrays.

## Stable FM Card

**Purpose**: Existing `baseline_detections` row that represents a useful FM-band signal card.

**Fields**:

- Existing baseline fields: `id`, `baseline_id`, `f_low_hz`, `f_high_hz`, `f_center_hz`, `first_seen_utc`, `last_seen_utc`, `total_hits`, `total_windows`, `confidence`, `missing_since_utc`.
- Existing optional fields: `peak_db`, `noise_db`, `snr_db`, `service`, `region`, `bandplan_notes`, `classification`, `label`, `notes`, `selected`, `user_bw_hz`.

**Validation Rules**:

- FM Validation should produce cards whose active rows are not dominated by sub-5 kHz spans.
- Repeated nearby fragments should update an existing stable card when within match tolerances.
- Multiple separated FM-like signals must remain separate stable cards.
- Missing state should not immediately churn hundreds of tiny rows under normal FM Validation.

## Persistence Match Decision

**Purpose**: Decision record explaining how a promoted detection interacts with existing baseline rows.

**Fields**:

- `action`: one of `insert`, `update`, `no_match`, `width_reject`, `missing_marked`, `missing_cleared`.
- `baseline_id`
- `detection_id`: existing row id when applicable.
- `center_hz`
- `center_delta_hz`
- `seg_width_hz`
- `persisted_width_hz`
- `spans_overlap`
- `center_close`
- `max_detection_width_ratio`
- `max_detection_width_hz`
- `reason`

**Validation Rules**:

- Diagnostics should include compact counts and representative records.
- Tests should assert insert vs update behavior for repeated nearby FM fragments.
- Decision records must not require schema changes; they can be diagnostic JSONL/log/export artifacts.

## Width Clamp Decision

**Purpose**: Evidence that configured min/max widths are applied and bounded.

**Fields**:

- `stage`: `shape_match`, `shape_display`, `persistence_blend`, or `revisit_trim`.
- `input_width_hz`
- `output_width_hz`
- `min_width_hz`
- `pad_hz`
- `max_width_hz`
- `was_clamped`
- `was_floored`
- `center_hz`

**Validation Rules**:

- FM Validation tests must cover min display/match widths and max width caps.
- Width behavior must be observable in diagnostics without unbounded per-bin output.
- Non-FM narrow behavior must not be widened unless FM-specific settings are selected.

## Revisit Decision

**Purpose**: Existing two-pass confirmation/pruning behavior made visible for FM Validation.

**Fields**:

- `action`: `queued`, `skipped_overlap`, `confirmed`, `marked_missing`, `pruned`, `trimmed`, or `summary`.
- `tag_id`
- `detection_id`
- `reason`: `new` or `missing`.
- `center_hz`
- `width_hz`
- `matched`
- `measured_width_hz`
- `segment_count`

**Validation Rules**:

- If FM Validation enables two-pass, tests must cover revisit queue and result summaries.
- `scan_updates` should show nonzero revisit counts when revisits actually run.
- Diagnostic bundle should make it clear when two-pass was disabled.

## Diagnostic Bundle Evidence

**Purpose**: Bounded operator-shareable evidence package for future FM stability debugging.

**Fields**:

- Existing included files: job params, scanner command, log tail, diagnostic JSONL tail, baseline rows, scan updates, monitoring zones, friendly signals, manifest.
- New or improved evidence: persistence decision summary, width decision summary, revisit decision summary, profile/two-pass effective settings.
- Manifest entries for missing or truncated evidence.

**Validation Rules**:

- Must remain bounded.
- Must record missing/truncated evidence in `manifest.json`.
- Must make profile and two-pass application explicit enough to distinguish disabled, profile-enabled, and explicitly enabled behavior.

## State Transitions

Preset selection:

```text
Discovery selected
  -> operator selects FM Validation
  -> FM Validation values/profile are applied
  -> operator may copy settings
  -> operator starts GUI/controller job
  -> diagnostics explain card stability decisions
```

Persistence decision:

```text
Detection fragment promoted
  -> compare with existing baseline detections
  -> insert new row OR update existing row OR reject/skip by width
  -> mark unseen old rows missing after sweep
  -> optionally queue revisit
  -> optionally confirm/prune/mark after revisit
```

Width shaping:

```text
Raw segment width
  -> optional match span padding/floor/cap
  -> persisted/update span
  -> optional display span padding/floor/cap
  -> card display span
```
