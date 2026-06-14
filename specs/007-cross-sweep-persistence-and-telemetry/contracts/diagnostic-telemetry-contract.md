# Contract: Structured Diagnostics And Device Telemetry

This contract defines the diagnostic bundle evidence needed to audit cross-sweep persistence, scan parameters, and receiver state.

## Existing Bundle Evidence To Preserve

Bundles must continue to include bounded versions of:

- `job/job.json`
- `job/params.json`
- `job/scanner-command.txt`
- `logs/scanner-log-tail.txt`
- `diagnostics/diagnostic-jsonl-tail.jsonl`
- `diagnostics/decision-summary.json`
- `diagnostics/characterization-summary.json`
- `database/baseline.json`
- `database/baseline-detections.json`
- `database/scan-updates.json`
- `manifest.json`
- `NOTES.md`

Missing or truncated evidence must continue to be recorded in `manifest.json`.

## Required Aggregate Counts

Diagnostic summaries must provide aggregate counts for:

- `segment_inventory`
- `cluster_emit`
- `cluster_reject`
- `persistence_decision` action `match`
- `persistence_decision` action `no_match`
- `persistence_decision` action `cross_sweep_promote`
- `width_decision`
- `revisit_queue`
- `revisit_result`
- `characterization_record`

Recommended shape:

```json
{
  "event_counts": {
    "segment_inventory": 12,
    "cluster_emit": 4,
    "cluster_reject": 8,
    "persistence_decision": 20,
    "width_decision": 10,
    "revisit_result": 2,
    "characterization_record": 4
  },
  "persistence_actions": {
    "match": 5,
    "no_match": 3,
    "cross_sweep_promote": 1
  },
  "revisit_events": {
    "revisit_queue": 2,
    "revisit_result": 2
  },
  "parse_errors": 0,
  "truncated": false
}
```

## Device Telemetry

Each diagnostic job must record a device telemetry snapshot when diagnostics are enabled.

Required fields:

```json
{
  "event": "device_telemetry",
  "device_key": "rtl:0",
  "device_index": 0,
  "device_serial": "00000001",
  "device_label": "RTL-SDR #0",
  "driver": "rtlsdr_native",
  "requested_gain": "30",
  "gain_mode": "manual",
  "actual_gain": 30.0,
  "supported_gains": [0.0, 9.9, 19.7, 29.7],
  "sample_rate_hz": 2400000,
  "actual_sample_rate_hz": 2400000,
  "fft": 8192,
  "bin_width_hz": 292.96875,
  "selected_profile": "fm_broadcast",
  "unavailable_fields": []
}
```

Rules:

- Unavailable values may be `null` and should be listed in `unavailable_fields`.
- Missing telemetry must not fail a scan.
- Telemetry is receiver state, not RF-environment evidence.

## Effective Parameter Summary

The bundle must expose the effective-parameter manifest described in [effective-parameter-manifest-contract.md](./effective-parameter-manifest-contract.md). Decision summaries may include a compact `effective_settings` field, but the manifest remains the authoritative job-level summary.

## Acceptance Checks

- Aggregate counts are available without reading scanner log text.
- Detailed JSONL tails may be truncated, but truncation is visible.
- Device telemetry is represented for both available and unavailable driver fields.
- A developer can distinguish receiver changes from RF-environment changes when comparing bundles.
