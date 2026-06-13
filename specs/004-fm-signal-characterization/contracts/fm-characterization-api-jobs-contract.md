# Contract: FM Characterization `/api/jobs` Payload

This contract preserves the existing web-to-controller request shape while allowing FM Validation to remain the operator-facing entry point for characterization work.

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

## Compatibility Rules

- Request body remains `{device_key, label, baseline_id, params}`.
- The web UI remains the operator-facing workflow surface.
- `params.start` and `params.stop` continue to come from the selected monitoring zone or preset context.
- Scanner command generation remains owned by `sdrwatch-control.py`.
- The web UI must not require operators to type raw characterization fields manually.
- Scanner CLI runs remain backend smoke coverage only.

## Characterization-Specific Rules

- FM characterization must run through the existing FM Validation workflow rather than a new operator-facing endpoint.
- Measured characterization values such as `measured_center_hz` or `measured_bandwidth_hz` are derived by the scanner path after the job starts; they are not user-entered request fields.
- If FM Validation continues to rely on `profile=fm_broadcast`, that profile remains contextual job input, not proof of modulation.
- Any explicit characterization-related tuning that is later exposed through `params` must remain optional, additive, and backward-compatible for callers that do not provide it.

## Discovery Preservation

RTL-SDR v4 Discovery remains a separate first-light preset.

Rules:

- FM Validation must not silently replace Discovery.
- Discovery payload behavior remains first-light oriented and distinct from FM characterization behavior.
- Characterization work must not require Discovery to submit new measurement-specific fields.

## Copy Current Scan Settings

When FM Validation is selected, Copy current scan settings must continue to produce parseable JSON containing:

- `device_key`
- `label`
- `baseline_id`
- `params`

The copied payload may include contextual tuning such as `profile`, revisit settings, or diagnostics mode, but it must not imply that the operator is expected to provide measured characterization evidence manually.

## Response Expectations

Existing response behavior remains unchanged:

```json
{
  "state": "running",
  "job": {
    "id": "job-id",
    "status": "running",
    "params": {},
    "cmd": []
  }
}
```

Rules:

- `job.params` remains the controller-side record of submitted and normalized parameters.
- `job.cmd`, when present, remains internal or debug context only.
- Missing `job.cmd` must not block characterization validation.

## Acceptance Checks

- The existing request shape is unchanged.
- FM Validation remains GUI-first.
- Discovery remains distinct.
- Operators are not asked to type measured characterization fields into the UI.
- Characterization results are produced by the scanner and diagnostics path, not by manual request payload construction.
