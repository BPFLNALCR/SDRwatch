# Feature Specification: GUI Diagnostic Capture

**Feature Branch**: `004-gui-diagnostics-capture`

**Created**: 2026-06-09

**Status**: Draft

**Input**: User description: "Add a GUI-first diagnostic capture workflow for SDRwatch."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Enable Diagnostics Before a Scan (Priority: P1)

As an SDRwatch operator on the scan/control page, I can turn on Diagnostics mode before starting a scan so the next scan job automatically captures diagnostic evidence without me typing file paths or running commands.

**Why this priority**: Diagnostic capture is only useful if operators can enable it in the normal workflow before they reproduce a detection-quality problem.

**Independent Test**: Can be fully tested through the browser by enabling Diagnostics mode, starting a scan through the web control workflow, and confirming the resulting job carries a safe diagnostic capture destination selected by the system.

**Acceptance Scenarios**:

1. **Given** the operator is on the scan/control page, **When** they enable Diagnostics mode and start a scan, **Then** the scan starts through the controller job lifecycle with diagnostic capture configured automatically.
2. **Given** Diagnostics mode is enabled, **When** the operator starts a scan, **Then** the operator is not asked to enter or edit a diagnostic file path.
3. **Given** Diagnostics mode is disabled, **When** the operator starts a scan, **Then** the scan follows the existing non-diagnostic start workflow.

---

### User Story 2 - Export a Diagnostic Bundle (Priority: P2)

As an SDRwatch operator investigating a detection problem, I can export a diagnostic bundle for the selected or recent job from the web UI during or after the scan so I can share the relevant context for analysis.

**Why this priority**: The captured data must be easy to package with surrounding job, baseline, and monitoring context or the evidence remains fragmented and hard to use.

**Independent Test**: Can be tested through the browser by selecting a recent job, using the Export diagnostic bundle action, downloading the resulting local artifact, and inspecting that available evidence and notes are present.

**Acceptance Scenarios**:

1. **Given** a scan job has diagnostic evidence available, **When** the operator exports a diagnostic bundle from the web UI, **Then** a downloadable local bundle is produced for that job.
2. **Given** a scan job is still running, **When** the operator exports a diagnostic bundle, **Then** the bundle contains the evidence available so far and clearly indicates that the job was still active at export time.
3. **Given** optional evidence is missing for a job, **When** the operator exports a diagnostic bundle, **Then** the bundle is still created and the missing items are clearly noted.

---

### User Story 3 - Provide Problem Notes with the Evidence (Priority: P3)

As an operator preparing a bundle for analysis, I receive a notes template in the bundle so I can describe what I expected, what I observed, and which RF problem type best matches the issue.

**Why this priority**: Human observation is required to distinguish false positives, false negatives, wrong bandwidth, wrong center frequency, merged signals, split signals, unstable baseline behavior, and other detection-quality problems.

**Independent Test**: Can be tested by exporting a bundle and confirming it contains a README or NOTES template with prompts for expected behavior, actual behavior, affected frequency or band, and problem type.

**Acceptance Scenarios**:

1. **Given** a diagnostic bundle has been exported, **When** the operator opens its notes template, **Then** it prompts for expected behavior, actual behavior, frequency or band affected, and the problem type.
2. **Given** the operator is reporting an issue outside the predefined problem types, **When** they use the notes template, **Then** they can mark the problem as "other" and add details.

---

### User Story 4 - Create Bundles Without SDR Hardware (Priority: P4)

As a maintainer or operator without hardware attached, I can still create a diagnostic bundle from existing logs, local job data, and diagnostic files so software-level export behavior can be tested and evidence can be packaged after the fact.

**Why this priority**: Bundle creation should be verifiable and useful even when a real SDR scan cannot be run in the current environment.

**Independent Test**: Can be tested with temporary local files and a temporary local data store containing representative job, baseline, monitoring-zone, friendly-signal, detection, and scan-update records.

**Acceptance Scenarios**:

1. **Given** representative local job data, logs, and diagnostic event files already exist, **When** the export workflow is invoked without SDR hardware, **Then** the bundle is created from those artifacts.
2. **Given** no SDR device is available, **When** bundle creation is tested using prepared local artifacts, **Then** the test does not require the operator or tester to run scanner CLI commands manually.

### Edge Cases

- A job has no diagnostic event file because Diagnostics mode was not enabled.
- A diagnostic event file or scanner log is very large.
- A job is still running while the export is requested.
- The selected job has partial metadata, missing scanner log, missing baseline metadata, or no active monitoring zones.
- The diagnostic event file referenced by the job cannot be found or is no longer readable.
- Recent baseline detection or scan update evidence exceeds the default export limit.
- Known or friendly signal data has not been configured.
- Multiple exports are requested for the same job.
- Existing job-control clients continue to use the current job workflow without diagnostics.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The scan/control page MUST include a visible Diagnostics mode toggle before the operator starts a scan.
- **FR-002**: Diagnostics mode MUST be opt-in per scan job and MUST not require the operator to type, paste, browse for, or otherwise manage diagnostic file paths.
- **FR-003**: When Diagnostics mode is enabled, the web and controller workflow MUST automatically configure the scanner diagnostic event output, including the existing `diagnostic_jsonl` capture setting, with a safe per-job or timestamped local filename.
- **FR-004**: The diagnostic capture filename MUST avoid unsafe path characters, accidental overwrite of unrelated files, and ambiguity between different scan jobs.
- **FR-005**: Diagnostics mode MUST preserve the existing scan start, status, stop, recent-job, and `/api/jobs` behavior for operators and existing job-control integrations.
- **FR-006**: The controller MUST remain the layer that creates scanner jobs and invokes the scanner internally; the operator-facing workflow MUST stay in the web UI.
- **FR-007**: Enabling or disabling Diagnostics mode MUST NOT change detection behavior, detection thresholds, baseline logic, RF processing, or signal classification outcomes.
- **FR-008**: The web UI MUST expose an Export diagnostic bundle action for the selected or most recent relevant job during and after a scan.
- **FR-009**: The exported diagnostic bundle MUST be downloadable from the web UI as a single archive by default.
- **FR-010**: The export workflow MUST operate locally and offline without requiring cloud services, remote storage, or internet connectivity.
- **FR-011**: The diagnostic bundle MUST include job metadata when available.
- **FR-012**: The diagnostic bundle MUST include the controller job parameters when available.
- **FR-013**: The diagnostic bundle MUST include the scanner command generated by the controller when available.
- **FR-014**: The diagnostic bundle MUST include the scanner log when available.
- **FR-015**: The diagnostic bundle MUST include diagnostic event contents when available, using either a bounded full subset or a tail-limited subset by default.
- **FR-016**: The diagnostic bundle MUST include recent baseline detection records, including current `baseline_detections` rows, when available.
- **FR-017**: The diagnostic bundle MUST include recent scan update records, including current `scan_updates` rows, when available.
- **FR-018**: The diagnostic bundle MUST include active baseline metadata when available.
- **FR-019**: The diagnostic bundle MUST include selected monitoring zones when available.
- **FR-020**: The diagnostic bundle MUST include known or friendly signal context when available.
- **FR-021**: The diagnostic bundle MUST include a README or NOTES template that prompts for expected behavior, actual behavior, frequency or band affected, and problem type.
- **FR-022**: The README or NOTES template MUST list the problem types false positive, false negative, wrong bandwidth, wrong center, merged signals, split signals, unstable baseline, and other.
- **FR-023**: The bundle MUST clearly identify evidence that was unavailable, omitted, unreadable, or truncated.
- **FR-024**: The export workflow MUST avoid huge exports by default by applying bounded limits to logs, diagnostic event records, and recent local records, or by requiring an explicit bounded export window.
- **FR-025**: The export workflow MUST make the applied bounds visible in the bundle so analysis can account for missing earlier evidence.
- **FR-026**: Bundle creation MUST work without SDR hardware when existing local logs, job metadata, local records, and diagnostic event files are present.
- **FR-027**: Automated or reproducible no-hardware tests MUST cover bundle creation with temporary local files and a temporary SQLite database.
- **FR-028**: Automated or reproducible web/API tests MUST cover the Diagnostics mode scan-start path and the diagnostic bundle export path without requiring SDR hardware.
- **FR-029**: Documentation MUST explain the GUI-based diagnostic capture and export steps and MUST NOT require operators to run scanner CLI commands manually.
- **FR-030**: If existing access controls are enabled for job control or export actions, Diagnostics mode and bundle export MUST follow the same protections.

### Key Entities *(include if feature involves data)*

- **Diagnostic Mode Setting**: The operator-visible choice that determines whether the next scan job should capture diagnostic evidence.
- **Scan Job**: A controller-managed scan lifecycle item with job metadata, job parameters, generated scanner command, status, timestamps, and log references.
- **Diagnostic Event File**: The per-job or timestamped local event capture created when Diagnostics mode is enabled.
- **Diagnostic Bundle**: The exported local artifact containing available job, scanner, diagnostic event, baseline, monitoring, and friendly-signal context.
- **Evidence Bounds**: The limits applied to logs, diagnostic event records, and recent local records to keep default exports manageable.
- **Operator Notes Template**: A README or NOTES file included in each bundle for the operator to describe expected behavior, actual behavior, affected frequency or band, and problem type.
- **Baseline Context**: Active baseline metadata and recent baseline detection evidence relevant to the selected job.
- **Monitoring Context**: Selected monitoring zones and known or friendly signals relevant to interpreting the selected job.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An operator can enable Diagnostics mode, start a scan, and export a diagnostic bundle from the web UI in under 3 minutes without using scanner CLI commands.
- **SC-002**: For a job with representative available evidence, the exported bundle contains all required evidence categories or an explicit unavailable/truncated note for each category.
- **SC-003**: Default exports remain bounded so a job with large logs or diagnostic event files can still produce a downloadable bundle in under 10 seconds on a typical local development system.
- **SC-004**: Existing scan start, status, stop, and recent-job workflows continue to work for users who do not enable Diagnostics mode.
- **SC-005**: No-hardware verification can create a diagnostic bundle from prepared local artifacts and representative local records.
- **SC-006**: Review of the updated operator documentation allows a tester to perform the diagnostic capture workflow through the GUI without being instructed to run scanner CLI commands.
- **SC-007**: Detection outputs for equivalent scan inputs are unchanged by the presence of Diagnostics mode except for the additional diagnostic artifacts it requests.

## Assumptions

- Diagnostics mode is off by default and applies to the next scan job started from the web control workflow.
- A single downloadable archive is the default bundle format because it is the clearest browser-based export for local sharing.
- Large logs and diagnostic event files are tail-limited or window-limited by default, with truncation made visible in the bundle.
- Missing optional evidence should not block bundle creation if enough job identity is available to create a meaningful export.
- Existing operator permissions and token behavior apply to diagnostic capture and export actions.
- The feature adds diagnostic capture, packaging, and documentation only; it does not tune detection parameters or add simulation mode.
