# Contract: Role-Aware Diagnostic Telemetry

## Purpose

Extend diagnostic JSONL and bundle summaries with enough receiver, role, job, timing, and resource provenance to benchmark one, two, and three RTL operation.

## Common Provenance Fields

Applicable diagnostic events should include these fields where feasible:

```json
{
  "role_run_id": "rr-20260614-200000",
  "job_id": "job-guard",
  "receiver_role": "GUARD",
  "role_lane": "guard_primary",
  "source_task": "guard_window",
  "source_profile": "fm_broadcast",
  "device_identity": "rtl:serial:00000001",
  "device_key": "rtl:0",
  "device_serial": "00000001",
  "device_index": 0,
  "device_kind": "rtlsdr",
  "backend": "rtlsdr_native",
  "active_device_count": 2,
  "active_role_count": 2
}
```

## Detection Window Event Additions

Existing `detection_window` records should add fields without removing current fields.

```json
{
  "event": "detection_window",
  "sweep_id": "20260614T200000Z",
  "window_idx": 0,
  "center_hz": 102300000,
  "sample_rate": 2400000,
  "fft": 4096,
  "avg": 8,
  "num_segments": 2,
  "samples_requested": 32768,
  "samples_read": 32768,
  "short_read": false,
  "dropped_reads": null,
  "timing": {
    "tune_ms": 12.5,
    "flush_ms": 8.2,
    "read_ms": 18.9,
    "fft_ms": 4.1,
    "detect_ms": 2.3,
    "db_update_ms": 6.8,
    "jsonl_ms": 0.7,
    "total_window_ms": 54.0
  },
  "unavailable_fields": ["dropped_reads"]
}
```

## Resource Telemetry Event

Emitted at job start, periodically during long runs where feasible, and at job end.

```json
{
  "event": "resource_telemetry",
  "timestamp": "2026-06-14T20:00:10Z",
  "pid": 4321,
  "role_run_id": "rr-20260614-200000",
  "job_id": "job-guard",
  "receiver_role": "GUARD",
  "active_device_count": 2,
  "active_role_count": 2,
  "sample_rate": 2400000,
  "cpu_load": 37.5,
  "rss_memory_bytes": 184000000,
  "unavailable_fields": []
}
```

## Device Telemetry Event Additions

Existing `device_telemetry` records should include:

- `device_identity`
- `device_key`
- `device_serial`
- `device_index`
- `receiver_role`
- `role_lane`
- `role_run_id`
- `job_id`
- `active_device_count`
- `active_role_count`

## Effective Parameter Manifest Additions

Existing effective-parameter manifest records should include:

- `source_task`
- `receiver_role`
- `role_lane`
- `role_run_id`
- `device_identity`
- `device_serial`
- `device_index`
- `runnable_backend`

## Bundle Summary Additions

Diagnostic bundles should summarize:

- Roles present.
- Devices present.
- Jobs present.
- Role-run IDs present.
- Timing field availability.
- Resource telemetry availability.
- Missing/unavailable telemetry fields.
- Whether provenance was diagnostic-only or durably stored.

## Unavailable Field Rules

- Missing platform metrics must not fail scans.
- Use `null` for unavailable field values.
- Add the field name to `unavailable_fields`.
- Short reads should be represented when detectable; otherwise `short_read=null` and `short_read` listed as unavailable.
