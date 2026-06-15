# Contract: Additive Job Status Fields

## Purpose

Preserve existing single-job contracts while allowing role-aware jobs to carry receiver role, identity, and grouping metadata.

## Existing Job Start Compatibility

Existing request shape remains valid:

```json
{
  "device_key": "rtl:0",
  "label": "web",
  "baseline_id": 12,
  "params": {
    "profile": "fm_broadcast"
  }
}
```

Role-aware grouped runs should use `/role-runs` or `/api/role-runs`. Existing `/api/jobs` may accept additive metadata fields only if they do not break current callers.

## Additive Job Fields

Any job object may include:

```json
{
  "id": "job-guard",
  "device_key": "rtl:0",
  "status": "running",
  "role_run_id": "rr-20260614-200000",
  "receiver_role": "GUARD",
  "role_lane": "guard_primary",
  "source_task": "guard_window",
  "device_identity": "rtl:serial:00000001",
  "device_serial": "00000001",
  "device_index": 0,
  "identity_confidence": "stable",
  "identity_warning": null,
  "active_device_count": 2,
  "active_role_count": 2,
  "last_update_ts": "2026-06-14T20:00:02Z",
  "error_message": null
}
```

## Compatibility Rules

- Existing fields must keep their names and semantics.
- Callers that ignore unknown fields must continue to work.
- Role-aware fields may be absent or null for legacy jobs.
- `/api/jobs/active` can remain backward-compatible for existing workflows, but role-aware UI must use role-run status and job lists rather than assuming only one active job.
- Stopping a child job through the existing job stop endpoint must update parent role-run status if the job belongs to a role run.

## Error Payload: Unsupported Backend Before Spawn

HTTP 400

```json
{
  "error": "unsupported_backend",
  "message": "Scanner execution is available only for rtlsdr_native devices in this build.",
  "requested_backend": "soapy",
  "supported_backends": ["rtlsdr_native"],
  "spawned": false
}
```

## Error Payload: Duplicate Assignment Before Spawn

HTTP 409

```json
{
  "error": "device_locked",
  "message": "Device is already locked by another running job.",
  "device_key": "rtl:0",
  "device_identity": "rtl:serial:00000001",
  "active_job_id": "job-existing",
  "spawned": false
}
```
