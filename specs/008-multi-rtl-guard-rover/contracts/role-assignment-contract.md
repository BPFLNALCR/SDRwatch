# Contract: Manual Receiver Role Assignment

## Purpose

Allow the operator to assign detected runnable RTL receivers to role lanes while preserving identity warnings and preventing duplicate active use.

## Roles and Lanes

`role` values:

- `GUARD`
- `ROVER`
- `REFERENCE`

`role_lane` values:

- `guard_primary`
- `guard_secondary`
- `rover`
- `reference`

The UI may label `guard_primary` as Friendly Guard and `guard_secondary` as Watchlist Guard.

## List Assignments

`GET /role-assignments`

Web proxy: `GET /api/role-assignments`

```json
{
  "assignments": [
    {
      "assignment_id": "assign-guard-primary",
      "role": "GUARD",
      "role_lane": "guard_primary",
      "display_name": "Friendly Guard",
      "device_identity": "rtl:serial:00000001",
      "legacy_device_key": "rtl:0",
      "serial": "00000001",
      "runtime_index": 0,
      "identity_confidence": "stable",
      "assignment_scope": "persistent",
      "task": {
        "source_task": "guard_window",
        "start_hz": 101100000,
        "stop_hz": 103500000
      },
      "assigned_ts": "2026-06-14T20:00:00Z",
      "updated_ts": "2026-06-14T20:00:00Z",
      "warnings": []
    }
  ]
}
```

## Set Assignment

`PUT /role-assignments/{role_lane}`

Web proxy: `PUT /api/role-assignments/{role_lane}`

```json
{
  "device_identity": "rtl:serial:00000001",
  "legacy_device_key": "rtl:0",
  "role": "GUARD",
  "task": {
    "source_task": "guard_window",
    "start_hz": 101100000,
    "stop_hz": 103500000,
    "profile": "fm_broadcast"
  },
  "acknowledge_identity_warning": false
}
```

### Success Response

```json
{
  "assignment": {
    "assignment_id": "assign-guard-primary",
    "role": "GUARD",
    "role_lane": "guard_primary",
    "device_identity": "rtl:serial:00000001",
    "legacy_device_key": "rtl:0",
    "identity_confidence": "stable",
    "assignment_scope": "persistent",
    "warnings": []
  }
}
```

## Clear Assignment

`DELETE /role-assignments/{role_lane}`

Web proxy: `DELETE /api/role-assignments/{role_lane}`

```json
{
  "cleared": true,
  "role_lane": "guard_primary"
}
```

## Error Payloads

### Duplicate Active Assignment

HTTP 409

```json
{
  "error": "duplicate_device_assignment",
  "message": "Receiver is already assigned to an active role or running job.",
  "device_identity": "rtl:serial:00000001",
  "active_job_id": "job-123",
  "role_run_id": "rr-456",
  "role_lane": "rover"
}
```

### Identity Warning Not Acknowledged

HTTP 409

```json
{
  "error": "identity_warning_requires_acknowledgement",
  "message": "Receiver identity is index-only or ambiguous; assignment is session-scoped unless reconfirmed.",
  "warnings": ["index_only_identity"]
}
```

### Unsupported Device

HTTP 400

```json
{
  "error": "unsupported_device",
  "message": "Only native RTL-SDR receivers are assignable to scanner roles in this feature.",
  "support_state": "unsupported",
  "runnable": false
}
```
