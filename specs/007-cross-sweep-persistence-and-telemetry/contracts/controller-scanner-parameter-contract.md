# Contract: Controller To Scanner Parameter Parity

This contract preserves the existing web/controller request shape while requiring parity between web/API job params, controller command construction, and scanner-supported characterization behavior.

## Start Job Shape

`POST /api/jobs`

Request body remains:

```json
{
  "device_key": "rtl:0",
  "label": "web",
  "baseline_id": 1,
  "params": {
    "start": 88000000,
    "stop": 108000000,
    "profile": "fm_broadcast"
  }
}
```

Rules:

- The top-level shape remains `{device_key, label, baseline_id, params}`.
- The web UI remains the operator-facing workflow.
- Scanner command generation remains owned by `sdrwatch-control.py`.
- Direct scanner CLI use remains backend smoke coverage only.

## Required Passthrough Mapping

| Web/API param | Scanner flag | Scanner arg | Status |
| --- | --- | --- | --- |
| `segment_center_mode` | `--segment-center-mode` | `segment_center_mode` | Required if scanner flag exists |
| `segment_centroid_span_hz` | `--segment-centroid-span-hz` | `segment_centroid_span_hz` | Required if scanner flag exists |
| `segment_centroid_drop_db` | `--segment-centroid-drop-db` | `segment_centroid_drop_db` | Required if scanner flag exists |
| `segment_centroid_floor_margin_db` | `--segment-centroid-floor-margin-db` | `segment_centroid_floor_margin_db` | Required if scanner flag exists |
| `match_bandwidth_pad_hz` | `--match-bandwidth-pad-hz` | `match_bandwidth_pad_hz` | Required if scanner flag exists |
| `min_match_bandwidth_hz` | `--min-match-bandwidth-hz` | `min_match_bandwidth_hz` | Required if scanner flag exists |
| `display_bandwidth_pad_hz` | `--display-bandwidth-pad-hz` | `display_bandwidth_pad_hz` | Required if scanner flag exists |
| `min_display_bandwidth_hz` | `--min-display-bandwidth-hz` | `min_display_bandwidth_hz` | Required if scanner flag exists |
| `max_persist_width_hz` | `--max-detection-width-hz` | `max_detection_width_hz` | Mapped equivalent |
| `max_card_width_hz` | `--max-detection-width-hz` | `max_detection_width_hz` | Mapped equivalent |
| `center_match_hz` | `--center-match-hz` | `center_match_hz` | Required if scanner flag exists |
| `persistence_mode` | `--persistence-mode` | `persistence_mode` | Required |
| `persistence_hit_ratio` | `--persistence-hit-ratio` | `persistence_hit_ratio` | Required |
| `persistence_min_seconds` | `--persistence-min-seconds` | `persistence_min_seconds` | Required |
| `persistence_min_hits` | `--persistence-min-hits` | `persistence_min_hits` | Required |
| `persistence_min_windows` | `--persistence-min-windows` | `persistence_min_windows` | Required |
| `persistence_min_sweep_loops` | new or documented equivalent | new or documented equivalent | Required for cross-sweep if added |
| `two_pass` | `--two-pass` | `two_pass` | Required |
| `revisit_fft` | `--revisit-fft` | `revisit_fft` | Required |
| `revisit_avg` | `--revisit-avg` | `revisit_avg` | Required |
| `revisit_margin_hz` | `--revisit-margin-hz` | `revisit_margin_hz` | Required |
| `revisit_span_limit_hz` | `--revisit-span-limit-hz` | `revisit_span_limit_hz` | Required |
| `revisit_max_bands` | `--revisit-max-bands` | `revisit_max_bands` | Required |
| `revisit_floor_threshold_db` | `--revisit-floor-threshold-db` | `revisit_floor_threshold_db` | Required |

## Mapping Rules

- If the scanner currently supports a parameter only as a profile-hidden arg, add a scanner flag or document why direct passthrough is unsupported.
- `max_persist_width_hz` and `max_card_width_hz` intentionally map to current `max_detection_width_hz`; this preserves the existing scanner cap while giving controller/API callers contract names for persisted/card width limits.
- Characterization result fields such as `measured_center_hz`, `measured_bandwidth_hz`, `classification_candidate`, and `profile_context` are derived output fields, not operator-entered request params.

## Acceptance Checks

- Controller command tests cover every supported parameter in the mapping table.
- Unsupported or differently named parameters are documented and visible in feature artifacts.
- FM Validation and Discovery remain separate presets.
- Copy current scan settings remains parseable and uses the existing `/api/jobs` shape.
