# Contract: FM Stability Diagnostic Evidence

This contract describes the diagnostic evidence needed to analyze FM card stability without unbounded exports.

## Existing Evidence To Preserve

Diagnostic bundles must continue to include bounded versions of:

- `job/job.json`
- `job/params.json`
- `job/scanner-command.txt`
- `logs/scanner-log-tail.txt`
- `diagnostics/diagnostic-jsonl-tail.jsonl`
- `database/baseline.json`
- `database/baseline-detections.json`
- `database/scan-updates.json`
- `database/monitoring-zones.json`
- `database/friendly-signals.json`
- `manifest.json`
- `NOTES.md`

Missing or truncated evidence must be reported in `manifest.json`.

## Effective Settings Evidence

Each diagnostic bundle for an FM Validation job must make these effective settings clear:

- `start`, `stop`, `samp_rate`, `step`, `fft`, `avg`
- `threshold_db`, `guard_bins`, `min_width_bins`
- `cfar`, `cfar_train`, `cfar_guard`, `cfar_quantile`, `cfar_alpha_db`
- `gain`, driver, profile
- `persistence_mode`, `persistence_hit_ratio`, `persistence_min_seconds`, `persistence_min_hits`, `persistence_min_windows`
- `cluster_merge_hz`, `center_match_hz`
- `match_bandwidth_pad_hz`, `min_match_bandwidth_hz`
- `display_bandwidth_pad_hz`, `min_display_bandwidth_hz`
- `max_detection_width_ratio`, `max_detection_width_hz`
- `segment_center_mode` and centroid fields when active
- `two_pass` and revisit fields

Evidence may come from job params, scanner command, diagnostic window records, or a compact effective-settings summary.

## Persistence Decision Evidence

Bundles should expose compact decision evidence for:

- Inserts
- Updates/matches
- No-match outcomes
- Width rejects
- Missing marks
- Missing clears

Suggested JSONL event fields:

```json
{
  "event": "persistence_decision",
  "action": "update",
  "baseline_id": 1,
  "detection_id": 123,
  "center_hz": 101100000,
  "center_delta_hz": 2500,
  "seg_width_hz": 80000,
  "persisted_width_hz": 90000,
  "spans_overlap": true,
  "center_close": true,
  "reason": null
}
```

The exact event name may differ if it follows an existing logger convention, but the exported evidence must distinguish create/update/no-match/missing outcomes.

## Width Decision Evidence

Bundles should expose compact decision evidence for:

- Match span floor/padding/cap
- Display span floor/padding/cap
- Persistence width EMA/hysteresis
- Revisit span trimming

Suggested JSONL event fields:

```json
{
  "event": "width_decision",
  "stage": "shape_display",
  "center_hz": 101100000,
  "input_width_hz": 878.9,
  "output_width_hz": 200000.0,
  "min_width_hz": 200000.0,
  "max_width_hz": 270000.0,
  "was_floored": true,
  "was_clamped": false
}
```

## Revisit Evidence

When two-pass is enabled, bundles must make revisit behavior visible:

- Number of tags queued
- Number skipped by overlap or duplicate filtering
- Number confirmed
- Number marked missing
- Number pruned as false positive
- Number trimmed by revisit span limit

Existing `scan_updates` fields `num_revisits`, `num_confirmed`, and `num_false_positive` should remain populated. Diagnostic JSONL or summary evidence should explain why those counts are zero when two-pass is disabled.

## Bounded Export Rules

- Evidence must remain bounded by existing diagnostic export limits or by new explicit limits.
- Representative decision records are acceptable if full logs would exceed bounds.
- Summary counts should be preferred for high-volume decisions.
- `manifest.json` must record truncation.

## Acceptance Checks

For a successful FM Validation diagnostic bundle:

- `profile` and `two_pass` application is unambiguous.
- Persisted card count is not in the hundreds for a normal short FM validation run.
- Active cards are not dominated by sub-5 kHz widths.
- Diagnostics show whether detections inserted, updated, missed, clamped, or revisited.
- Discovery bundles can still show first-light card creation without requiring FM Validation behavior.
