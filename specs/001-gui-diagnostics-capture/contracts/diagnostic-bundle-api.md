# Contract: Diagnostic Bundle API

## Existing Job Start Compatibility

### `POST /api/jobs`

Starts a controller-managed scan job.

**Existing request shape remains valid**:

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

**New optional diagnostics request shape**:

```json
{
  "device_key": "rtl:0",
  "label": "web",
  "baseline_id": 1,
  "params": {
    "start": 88000000,
    "stop": 108000000,
    "diagnostics_mode": true
  }
}
```

**Compatibility rules**:

- `params.diagnostics_mode` is optional.
- Existing callers that omit `params.diagnostics_mode` behave as they do today.
- Existing internal/backend callers that provide `params.diagnostic_jsonl` remain supported.
- GUI callers must not provide a manually typed diagnostic path.
- When `params.diagnostics_mode` is true and no `params.diagnostic_jsonl` is supplied, the controller generates a safe local per-job diagnostic JSONL path and includes it in the job params/metadata.

**Success response**:

```json
{
  "state": "running",
  "job": {
    "id": "abc123def456",
    "status": "running",
    "baseline_id": 1,
    "params": {
      "start": 88000000,
      "stop": 108000000,
      "diagnostics_mode": true,
      "diagnostic_jsonl": "/controller-base/diagnostics/abc123def456.diagnostic.jsonl"
    },
    "cmd": ["python", "-m", "sdrwatch.cli", "--baseline-id", "1", "--diagnostic-jsonl", "..."],
    "log_path": "/controller-base/logs/abc123def456.log"
  }
}
```

**Error behavior**:

- Existing validation errors remain unchanged.
- Diagnostics mode must not cause a different validation contract for required `device_key` or `baseline_id`.

## Export Diagnostic Bundle

### `GET /api/jobs/{job_id}/diagnostic-bundle`

Downloads a local diagnostic bundle archive for a selected job.

**Query parameters**:

- `log_tail_lines`: Optional positive integer. Defaults to the configured safe bound.
- `diagnostic_tail_lines`: Optional positive integer. Defaults to the configured safe bound.
- `row_limit`: Optional positive integer for recent SQLite evidence. Defaults to the configured safe bound.

**Response on success**:

- Status: `200 OK`
- Content type: `application/zip`
- Content disposition: attachment filename such as `sdrwatch-diagnostics-{job_id}.zip`
- Body: Zip archive containing the manifest, notes template, job/controller evidence, logs, diagnostic JSONL subset, and SQLite evidence.

**Required archive entries when evidence is available**:

- `manifest.json`
- `README.md` or `NOTES.md`
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

**Manifest requirements**:

```json
{
  "bundle_version": 1,
  "created_at": "2026-06-09T12:00:00Z",
  "job_id": "abc123def456",
  "job_status_at_export": "running",
  "bounds": {
    "log_tail_lines": 2000,
    "diagnostic_tail_lines": 5000,
    "row_limit": 500
  },
  "included": ["job/job.json"],
  "missing": [
    {"category": "friendly_signals", "reason": "no records configured"}
  ],
  "truncated": [
    {"category": "scanner_log", "reason": "tail limited", "included_lines": 2000}
  ]
}
```

**Error behavior**:

- `404 Not Found` if the job ID cannot be resolved by the controller.
- `401 Unauthorized` if existing web API auth is enabled and the request is unauthenticated.
- `502 Bad Gateway` if controller communication fails before job metadata can be resolved.
- Missing optional files or database rows do not fail the request; they are recorded in the manifest.

## Active/Recent Job Export Support

The control page may call `GET /api/jobs/active` or `GET /api/jobs` to select the current or most recent job before calling the export endpoint.

**Compatibility rules**:

- Existing `/api/jobs`, `/api/jobs/active`, `/api/jobs/{job_id}`, and `/api/jobs/{job_id}/logs` contracts remain available.
- The export action uses existing job metadata and logs where possible instead of adding a separate scanner process.
