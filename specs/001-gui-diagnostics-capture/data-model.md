# Data Model: GUI Diagnostic Capture

## Diagnostic Mode Setting

Represents the operator's opt-in choice for the next scan job.

**Fields**:

- `enabled`: Boolean. True when the next scan should capture diagnostic JSONL automatically.
- `requested_from`: Text label for provenance, defaulting to the web control page.
- `requested_at`: Timestamp when the operator started the diagnostic job, if available.

**Validation rules**:

- Default is disabled.
- Enabling diagnostics must not require or expose a filesystem path to the operator.
- Disabling diagnostics must preserve the current non-diagnostic scan workflow.

**Relationships**:

- Applied to one scan-start request.
- Produces a controller-managed diagnostic event file path when enabled.

## Scan Job Diagnostic Metadata

Extends the existing controller-managed scan job evidence without replacing current job behavior.

**Fields**:

- `job_id`: Existing controller job identifier.
- `status`: Existing job state such as running, stopped, finished, or error.
- `created_ts`: Existing creation timestamp.
- `finished_ts`: Existing finish timestamp when available.
- `baseline_id`: Existing baseline associated with the job.
- `device_key`: Existing SDR device key.
- `params`: Existing controller job params, including optional diagnostic mode metadata and generated `diagnostic_jsonl`.
- `cmd`: Existing generated scanner command.
- `log_path`: Existing scanner log path.
- `diagnostic_jsonl`: Generated or existing diagnostic event file path when available.

**Validation rules**:

- Generated diagnostic file paths must be unique for the job and safe for local filesystem use.
- Existing callers that omit diagnostic fields must receive compatible job behavior.
- Existing internal callers that provide `diagnostic_jsonl` remain supported.

**Relationships**:

- References one active baseline when `baseline_id` is present.
- References zero or one diagnostic event file.
- Supplies metadata to one or more exported diagnostic bundles.

## Diagnostic Event File

Represents scanner-emitted JSONL diagnostic events for detection-quality analysis.

**Fields**:

- `path`: Local file path known to the controller/job metadata.
- `line_count_included`: Number of diagnostic records included in the bundle.
- `truncated`: Boolean indicating whether earlier content was omitted.
- `read_error`: Error text when the file was expected but could not be read.
- `contents`: Bounded JSONL subset included in the bundle.

**Validation rules**:

- Default export must apply bounded limits.
- Missing or unreadable files must be recorded in the bundle manifest instead of failing the entire export.

**Relationships**:

- Belongs to a scan job when diagnostics were enabled or explicitly configured.

## Diagnostic Bundle

The local downloadable evidence artifact for a selected or recent scan job.

**Fields**:

- `bundle_id`: Generated export identifier.
- `created_at`: Export timestamp.
- `job_id`: Selected job identifier.
- `job_status_at_export`: Job state when the bundle was created.
- `bounds`: Log, diagnostic JSONL, and database row limits applied.
- `files`: Manifest of files included in the archive.
- `missing`: Evidence categories unavailable at export time.
- `truncated`: Evidence categories included with bounded subsets.

**Expected archive contents**:

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

**Validation rules**:

- Must be downloadable from the web UI as a single archive by default.
- Must record missing or truncated evidence.
- Must not require SDR hardware to create when local evidence exists.

## Evidence Bounds

Represents default export limits used to prevent huge bundles.

**Fields**:

- `log_tail_lines`: Maximum scanner log lines included by default.
- `diagnostic_jsonl_tail_lines`: Maximum diagnostic JSONL lines included by default.
- `baseline_detections_limit`: Maximum recent baseline detection rows.
- `scan_updates_limit`: Maximum recent scan update rows.
- `zone_scope`: Selected or enabled monitoring zones for the job baseline.
- `friendly_signal_scope`: Known/friendly signals for the job baseline.

**Validation rules**:

- Values must be positive and bounded.
- Applied values must be recorded in `manifest.json`.

## Baseline Context

Existing database context relevant to the selected scan.

**Fields**:

- `baseline`: Current row from `baselines`.
- `baseline_detections`: Recent rows from `baseline_detections`.
- `scan_updates`: Recent rows from `scan_updates`.

**Validation rules**:

- Missing baseline or missing tables must be recorded in the manifest.
- Recent rows should be ordered from most recent evidence back within the applied limit.

## Monitoring Context

Existing operator-defined context relevant to interpreting the selected scan.

**Fields**:

- `monitoring_zones`: Selected or enabled rows from `monitoring_zones`.
- `friendly_signals`: Rows from `friendly_signals`.

**Validation rules**:

- Missing tables or no configured records must not block bundle creation.
- The bundle must distinguish "table missing" from "no records configured" when possible.

## Operator Notes Template

A Markdown template included in the bundle.

**Fields**:

- `expected_behavior`
- `actual_behavior`
- `frequency_or_band_affected`
- `problem_type`
- `additional_notes`

**Validation rules**:

- Problem type list must include false positive, false negative, wrong bandwidth, wrong center, merged signals, split signals, unstable baseline, and other.
