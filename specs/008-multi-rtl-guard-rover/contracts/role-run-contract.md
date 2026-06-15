# Contract: Grouped Role-Aware Runs

## Purpose

Start, inspect, and stop one operator run containing one or more role-aware child scanner jobs.

## Start Role Run

`POST /role-runs`

Web proxy: `POST /api/role-runs`

The request starts child jobs from existing role assignments. It may optionally override tasks for this run without changing persistent assignments.

```json
{
  "label": "evening-watch",
  "baseline_id": 12,
  "role_lanes": ["guard_primary", "rover"],
  "tasks": {
    "guard_primary": {
      "source_task": "guard_window",
      "start_hz": 101100000,
      "stop_hz": 103500000,
      "profile": "fm_broadcast"
    },
    "rover": {
      "source_task": "rover_sweep",
      "start_hz": 88000000,
      "stop_hz": 108000000,
      "profile": "fm_broadcast"
    }
  },
  "params": {
    "diagnostics_mode": true
  }
}
```

## Start Success

HTTP 201

```json
{
  "role_run": {
    "role_run_id": "rr-20260614-200000",
    "status": "running",
    "capability_tier_at_start": "2",
    "started_ts": "2026-06-14T20:00:00Z",
    "updated_ts": "2026-06-14T20:00:02Z",
    "finished_ts": null,
    "active_device_count": 2,
    "active_role_count": 2,
    "child_jobs": [
      {
        "job_id": "job-guard",
        "receiver_role": "GUARD",
        "role_lane": "guard_primary",
        "device_identity": "rtl:serial:00000001",
        "device_key": "rtl:0",
        "status": "running",
        "error_message": null
      },
      {
        "job_id": "job-rover",
        "receiver_role": "ROVER",
        "role_lane": "rover",
        "device_identity": "rtl:serial:00000002",
        "device_key": "rtl:1",
        "status": "running",
        "error_message": null
      }
    ],
    "error": null,
    "warnings": []
  }
}
```

## List Role Runs

`GET /role-runs`

Web proxy: `GET /api/role-runs`

Returns recent and active grouped runs.

## Role Run Detail

`GET /role-runs/{role_run_id}`

Web proxy: `GET /api/role-runs/{role_run_id}`

Returns the same `role_run` object as the start response with refreshed child job states.

## Stop Role Run

`DELETE /role-runs/{role_run_id}`

Web proxy: `DELETE /api/role-runs/{role_run_id}`

```json
{
  "role_run": {
    "role_run_id": "rr-20260614-200000",
    "status": "finished",
    "finished_ts": "2026-06-14T20:05:00Z",
    "child_jobs": [
      {
        "job_id": "job-guard",
        "status": "finished"
      },
      {
        "job_id": "job-rover",
        "status": "finished"
      }
    ]
  }
}
```

## Error Payloads

### Duplicate Device

HTTP 409

```json
{
  "error": "duplicate_device_assignment",
  "message": "Role run cannot start because the same physical receiver is requested by more than one active role.",
  "device_identity": "rtl:serial:00000001",
  "role_lanes": ["guard_primary", "rover"]
}
```

### Device Locked

HTTP 409

```json
{
  "error": "device_locked",
  "message": "Receiver is locked by another active job.",
  "device_identity": "rtl:serial:00000001",
  "active_job_id": "job-existing"
}
```

### Unsupported Backend

HTTP 400

```json
{
  "error": "unsupported_backend",
  "message": "This build supports scanner execution only for native RTL-SDR receivers.",
  "requested_backend": "hackrf",
  "supported_backends": ["rtlsdr_native"]
}
```

### Partial Startup Failure

HTTP 207 or HTTP 500 depending on existing error conventions.

```json
{
  "error": "role_run_degraded",
  "message": "One or more child jobs failed during startup.",
  "role_run_id": "rr-20260614-200000",
  "started_jobs": ["job-guard"],
  "failed_roles": [
    {
      "role_lane": "rover",
      "error": "device_locked"
    }
  ]
}
```
