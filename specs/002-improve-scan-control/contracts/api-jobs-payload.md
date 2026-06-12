# Contract: Existing `/api/jobs` Payload Compatibility

This feature preserves the existing web-to-controller job contract. It may reorganize controls, improve input safety, and apply GUI presets/defaults, but it must not change the request shape used to start scans.

## Start Job

`POST /api/jobs`

Request body:

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

Required fields:

- `device_key`: selected device key.
- `baseline_id`: selected monitoring location or baseline identifier.
- `params.start`: derived from enabled monitoring zones.
- `params.stop`: derived from enabled monitoring zones.

Existing optional `params` names that must remain compatible:

- SDR/runtime: `driver`, `samp_rate`, `step`, `gain`, `loop`, `duration`, `sleep_between_sweeps`
- FFT/tuning: `fft`, `avg`, `threshold_db`, `guard_bins`, `min_width_bins`
- CFAR: `cfar`, `cfar_train`, `cfar_guard`, `cfar_quantile`, `cfar_alpha_db`
- Detection/expert: `cluster_merge_hz`, `max_detection_width_hz`, `max_detection_width_ratio`, `new_ema_occ`
- Persistence: `persistence_mode`, `persistence_hit_ratio`, `persistence_min_seconds`, `persistence_min_hits`, `persistence_min_windows`
- Revisit: `two_pass`, `revisit_fft`, `revisit_avg`, `revisit_margin_hz`, `revisit_span_limit_hz`, `revisit_max_bands`, `revisit_floor_threshold_db`
- Context/output: `profile`, `bandplan`, `db`, `jsonl`, `diagnostics_mode`, `diagnostic_jsonl`, `latitude`, `longitude`, `spur_calibration`

Compatibility rules:

- GUI presets/defaults may intentionally submit different values from older page defaults to resolve the promotion/persistence mismatch documented in `docs/DETECTION_TUNING_REPORT.md`.
- Changed defaults must continue to submit the same parameter names and compatible value types as the current page.
- Optional blank values must keep current omission/default behavior.
- `diagnostics_mode` must enable diagnostics without requiring `diagnostic_jsonl` from the operator.
- `diagnostic_jsonl`, if retained, is an expert compatibility override only.
- Preset selection should normally be resolved client-side into existing `params` values such as `gain`, `step`, `fft`, `avg`, `persistence_min_hits`, and `persistence_min_windows`.
- The request body must not require operators or tests to provide scanner CLI commands.

Preset payload expectations:

| Preset | Required payload characteristics |
| --- | --- |
| RTL-SDR v4 Discovery | Includes fixed manual `gain`, fast `step` near `samp_rate`, `fft` `4096` or `8192`, `avg=8`, and relaxed `persistence_min_hits=1` / `persistence_min_windows=1` |
| Stable Baseline | Includes fixed manual `gain`, overlapping `step` such as `1.2e6` with `samp_rate=2.4e6`, `fft=8192`, `avg=16`, and stricter `persistence_min_hits=2` / `persistence_min_windows=2` |
| Fast Wide Survey | Includes fixed manual `gain`, `step=2.4e6`, `fft=4096`, `avg=8`, and relaxed promotion gates |

## Start Job Response

The web API returns:

```json
{
  "state": "running",
  "job": {
    "id": "abc123def456",
    "status": "running",
    "params": {},
    "cmd": []
  }
}
```

Rules:

- Existing fields may vary by controller state, but `job.params` remains the authoritative controller-side parameters.
- If `job.cmd` is present, the page may offer internal/debug command copy.
- Missing `job.cmd` must not block the operator workflow.

## Active Job

`GET /api/jobs/active`

Used to refresh running state and discover active job metadata.

Rules:

- Existing response behavior is preserved.
- If the active job includes a generated command, it may be used for internal/debug copy.

## Job Detail

`GET /api/jobs/<job_id>`

Used when the page needs details for a selected active or recent job.

Rules:

- Existing job detail data, including `params` and `cmd` when available, may be displayed or copied.
- No new command-preview endpoint is required for this feature.

## Logs

`GET /api/jobs/<job_id>/logs?tail=200`

Rules:

- Existing live log polling remains unchanged.
- Logs are operator-facing browser output, not CLI instructions.

## Stop Job

`DELETE /api/jobs/<job_id>`

Rules:

- Existing stop behavior remains unchanged.
- Stop remains available from the scan/control page.
