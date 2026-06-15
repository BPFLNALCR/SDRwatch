# Contract: Hardware Inventory and Capability Tier

## Purpose

Expose detected hardware, runnable scanner capability, identity confidence, lock/job state, and manual role assignment state without implying unsupported hardware can run.

## Controller Endpoint

`GET /hardware/inventory`

Returns the authoritative controller inventory.

## Web Proxy Endpoint

`GET /api/hardware/inventory`

Returns the same logical payload through the web app, honoring existing controller token behavior.

## Compatibility

Existing `/devices` and `/ctl/devices` behavior must remain compatible for current single-device workflows. They may either continue returning the legacy list or include additive fields, but new multi-RTL UI should use the inventory contract.

## Response

```json
{
  "capability": {
    "tier": "2",
    "label": "Tier 2 - Multi-RTL guard + rover",
    "runnable_rtl_count": 2,
    "max_role_count": 2,
    "supported_roles": ["GUARD", "ROVER"],
    "warnings": [],
    "generated_ts": "2026-06-14T20:00:00Z"
  },
  "devices": [
    {
      "device_identity": "rtl:serial:00000001",
      "legacy_device_key": "rtl:0",
      "device_kind": "rtlsdr",
      "label": "RTL-SDR #0 SN 00000001",
      "runtime_index": 0,
      "serial": "00000001",
      "identity_confidence": "stable",
      "identity_scope": "persistent",
      "warnings": [],
      "detected": true,
      "runnable": true,
      "support_state": "runnable",
      "runnable_backend": "rtlsdr_native",
      "backend_status": "supported",
      "busy": false,
      "locked": false,
      "lock_owner": null,
      "active_job_id": null,
      "assigned_role": "GUARD",
      "role_lane": "guard_primary",
      "assignment_id": "assign-guard-primary",
      "last_seen_ts": "2026-06-14T20:00:00Z"
    }
  ],
  "unsupported_hardware": [
    {
      "hardware_kind": "hackrf",
      "detected": false,
      "support_state": "planned",
      "runnable": false,
      "runnable_backend": null,
      "message": "HackRF scanner execution is planned/future and is not runnable in this build."
    }
  ]
}
```

## Warning Codes

- `missing_serial`: receiver serial is unavailable.
- `duplicate_serial`: receiver serial is not unique among detected RTLs.
- `index_only_identity`: assignment can only bind to runtime index.
- `stale_lock`: lock existed for a job that is no longer running.
- `unsupported_backend`: hardware is not runnable by the scanner.

## Capability Tier Rules

- `tier="0"` when `runnable_rtl_count` is 0.
- `tier="1"` when `runnable_rtl_count` is 1.
- `tier="2"` when `runnable_rtl_count` is 2.
- `tier="2_plus"` when `runnable_rtl_count` is 3 or more.

## Error Payload

Inventory retrieval should normally return HTTP 200 even when no hardware exists. Controller communication failures through the web proxy use existing web error behavior plus:

```json
{
  "error": "controller_unavailable",
  "message": "Controller is unavailable; hardware inventory cannot be refreshed."
}
```
