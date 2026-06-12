# Research: GUI Diagnostic Capture

## Decision: Generate diagnostic JSONL paths in the controller per job

**Rationale**: The controller already creates the job ID, owns the controller base directory, persists job params, builds the scanner command, and invokes the scanner internally. Generating the diagnostic path there avoids asking the operator for a path and produces a stable link between job ID, job metadata, command, log path, and diagnostic evidence.

**Alternatives considered**:

- Web-generated timestamp path: rejected because the web layer should not own controller filesystem layout or scanner invocation details.
- Operator-entered path: rejected because the feature goal is GUI-first capture without manual diagnostic file paths.
- Scanner-generated path: rejected because the controller would not reliably know which diagnostic file belongs to which job without adding more scanner/controller coupling.

## Decision: Preserve existing scanner diagnostic flag and detection behavior

**Rationale**: The scanner already accepts diagnostic JSONL configuration and emits per-window diagnostics when requested. This feature should only decide when and where that output is written, then package it for analysis.

**Alternatives considered**:

- Add new scanner diagnostic modes: rejected as unnecessary and out of scope.
- Change detection thresholds or CFAR settings while diagnostics are enabled: rejected because diagnostics must not change detection behavior.

## Decision: Add optional diagnostic mode request metadata while preserving `/api/jobs`

**Rationale**: Current `/api/jobs` clients post `{device_key, label, baseline_id, params}` and the web proxy forwards params to the controller. Adding an optional boolean such as `params.diagnostics_mode` lets the GUI request automatic diagnostic capture without breaking callers that omit it. Existing internal callers that already pass `params.diagnostic_jsonl` remain compatible.

**Alternatives considered**:

- Replace `diagnostic_jsonl` with a new field: rejected because existing controller-to-scanner mapping already supports `diagnostic_jsonl`.
- Require a separate pre-start diagnostics endpoint: rejected because it splits the normal operator start workflow.

## Decision: Export a local zip with manifest, notes, and bounded evidence files

**Rationale**: A single archive is browser-friendly, works offline, is easy to attach to an issue or hand to Codex, and can include a manifest that records bounds, missing evidence, truncation, and job activity state.

**Alternatives considered**:

- Export a folder only: rejected for the default path because it is less convenient from a browser and harder to share intact.
- Export unbounded raw logs and JSONL: rejected because long-running scans can produce huge artifacts.
- Upload to a cloud service: rejected by the offline/local requirement.

## Decision: Use tail-limited or count-limited evidence by default

**Rationale**: Diagnostic bundles must be safe to create during or after long scans. Tail limits keep the most recent reproduction context while preventing very large downloads. The manifest and README must make limits explicit so analysis can request a wider export if needed.

**Alternatives considered**:

- Full file export by default: rejected because logs and diagnostic JSONL can grow without bound.
- Time-window-only export: deferred because job timestamps and scanner event timestamps may be incomplete or inconsistent across historical artifacts.

## Decision: Query existing SQLite tables without schema changes for v1

**Rationale**: Required evidence already exists in current tables: `baselines`, `baseline_detections`, `scan_updates`, `monitoring_zones`, and `friendly_signals`. The feature can package those rows without introducing migration risk.

**Alternatives considered**:

- Add diagnostic bundle history tables: rejected as unnecessary for the first workflow.
- Copy all database tables into the bundle: rejected because it is larger than needed and less focused for detection-quality triage.

## Decision: Include a generated operator notes template in every bundle

**Rationale**: Detection-quality analysis needs human-observed expected/actual behavior and a problem type. A template in the bundle captures that without adding a separate database workflow or cloud issue tracker.

**Alternatives considered**:

- Require a web form before export: rejected for v1 because it could block urgent evidence collection.
- Omit notes: rejected because raw evidence alone may not explain false positives, false negatives, bandwidth errors, center-frequency errors, merged signals, split signals, or unstable baseline behavior.

## Decision: Test bundle creation with fake controller data and temporary SQLite

**Rationale**: The feature must work and be verifiable without SDR hardware. A fake controller client plus temp files and a temporary SQLite database can exercise bundle contents, bounds, missing evidence, and API download behavior.

**Alternatives considered**:

- Hardware-dependent acceptance only: rejected by the no-hardware requirement.
- Scanner CLI integration tests as the primary acceptance test: rejected because operator-facing workflows must use the web UI/controller lifecycle.
