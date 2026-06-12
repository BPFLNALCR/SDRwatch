# Contract: FM Validation `/api/jobs` Payload

This contract preserves the existing web-to-controller shape while adding an FM Validation preset that applies FM-stabilizing values through existing `params`.

## Start FM Validation Job

`POST /api/jobs`

Request body shape remains:

```json
{
  "device_key": "rtl:0",
  "label": "web",
  "baseline_id": 1,
  "params": {
    "start": 88000000,
    "stop": 108000000
  }
}
```

## Required Compatibility Rules

- Request body remains `{device_key, label, baseline_id, params}`.
- `params.start` and `params.stop` continue to come from enabled monitoring zones.
- Existing controller job lifecycle remains unchanged.
- Scanner command generation remains owned by `sdrwatch-control.py`.
- The web UI must not generate scanner commands independently.
- Scanner CLI checks are internal backend smoke tests, not operator acceptance.

## FM Validation Required Param Characteristics

The implementation may choose either profile-driven or explicit-param-driven wiring.

### Preferred profile-driven shape

```json
{
  "profile": "fm_broadcast",
  "two_pass": true,
  "diagnostics_mode": true
}
```

The GUI may also submit visible values aligned with the profile, such as:

```json
{
  "samp_rate": 2400000,
  "step": 1200000,
  "fft": 8192,
  "avg": 10,
  "gain": 20,
  "threshold_db": 6,
  "guard_bins": 3,
  "min_width_bins": 5,
  "cfar_train": 32,
  "cfar_guard": 6,
  "cfar_quantile": 0.6,
  "persistence_min_hits": 1,
  "persistence_min_windows": 1,
  "revisit_span_limit_hz": 420000,
  "max_detection_width_hz": 270000,
  "cluster_merge_hz": 12000
}
```

If explicit values differ from `fm_broadcast`, tests must document and assert the intentional override.

### Explicit-param shape, only if needed

If relying on `profile=fm_broadcast` is too opaque, the controller/CLI may add narrow passthrough for existing scanner/profile fields such as:

- `match_bandwidth_pad_hz`
- `min_match_bandwidth_hz`
- `display_bandwidth_pad_hz`
- `min_display_bandwidth_hz`
- `center_match_hz`
- `segment_center_mode`
- `segment_centroid_span_hz`
- `segment_centroid_drop_db`
- `segment_centroid_floor_margin_db`

Rules:

- Add passthrough only for existing fields already consumed by scanner/profile code.
- Preserve existing defaults for callers that do not submit these params.
- Do not add a separate endpoint for FM Validation.

## Discovery Preservation

RTL-SDR v4 Discovery remains a separate preset. Its payload must continue to include first-light characteristics, such as fast wide-sweep settings and relaxed `persistence_min_hits=1` / `persistence_min_windows=1`.

FM Validation must not silently replace Discovery defaults.

## Copy Current Scan Settings

When FM Validation is selected, Copy current scan settings must produce parseable JSON containing:

- `device_key`
- `label`
- `baseline_id`
- `params`
- expected FM Validation params using existing names

The copied settings are the review/debug artifact. They are not instructions for the operator to run scanner CLI commands.

## Controller Command Expectations

If FM Validation explicitly enables two-pass, the generated command should include `--two-pass`.

If FM Validation relies on profile-applied two-pass, diagnostics must show effective `two_pass=true` in scanner tuning params, even if the literal generated command only includes `--profile fm_broadcast`.

If FM Validation relies on `fm_broadcast`, the generated command should include `--profile fm_broadcast`.

## Response Expectations

Existing start response behavior remains unchanged:

```json
{
  "state": "running",
  "job": {
    "id": "532516344b4b",
    "status": "running",
    "params": {},
    "cmd": []
  }
}
```

Rules:

- `job.params` remains the controller-side parameter record.
- `job.cmd`, when present, can be copied as internal/debug context only.
- Missing `job.cmd` must not block FM Validation.
