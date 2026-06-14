# Feature Specification: Cross-Sweep Persistence and Telemetry

**Feature Branch**: `007-cross-sweep-persistence-and-telemetry`

**Feature Directory**: `specs/007-cross-sweep-persistence-and-telemetry`

**Created**: 2026-06-14

**Status**: Draft

**Input**: User description: "Improve SDRwatch's lawful passive RF monitoring pipeline by making signal persistence reliable across repeated sweep loops and making scanner/controller telemetry auditable, while preserving FM Broadcast as the stable control band and avoiding demodulation, content interception, private communications decoding, or offensive SIGINT capabilities."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Promote Signals Across Sweep Loops (Priority: P1)

As an SDRwatch operator, I want a stable signal that appears once per complete sweep loop to become a persistent baseline card after the configured number of repeated observations, even when it does not appear in adjacent same-sweep windows.

**Why this priority**: SDRwatch should not miss real passive RF observations solely because the signal falls into one non-overlapping window per sweep. Reliable persistence across loops is the core trust gap this feature closes.

**Independent Test**: Can be tested with a deterministic scan fixture launched through the controller/web job path, using strict persistence settings and diagnostics export to confirm that one stable observation per full sweep promotes only after the configured number of sweep-loop observations.

**Acceptance Scenarios**:

1. **Given** strict persistence settings requiring multiple observations, **When** the same stable signal appears once in each complete sweep loop and never in adjacent windows during the same loop, **Then** it becomes a persistent baseline card after the configured number of sweep-loop observations.
2. **Given** the same strict persistence settings, **When** a signal appears only once and is not observed again in later complete sweep loops, **Then** it does not become a persistent baseline card.
3. **Given** two nearby signals with centers or widths outside the configured compatibility rules, **When** both appear across repeated sweep loops, **Then** they remain separate candidates or cards rather than merging into one broad card.

---

### User Story 2 - Keep FM Broadcast Stable As Control Band (Priority: P1)

As an operator, I want the FM Broadcast profile to continue producing stable, bounded, separated station cards so FM remains a reliable control band for future tuning.

**Why this priority**: FM Broadcast is the regression-control band for SDRwatch's current signal-characterization pipeline. Cross-sweep persistence must improve reliability without undoing bounded FM card behavior.

**Independent Test**: Can be tested through the web UI by selecting FM Validation, starting a controller-backed scan or deterministic fixture run, exporting diagnostics, and confirming that nearby FM-like signals remain separate with raw, measured, match, display, and contextual evidence preserved.

**Acceptance Scenarios**:

1. **Given** the FM Broadcast profile and two FM-like signals close but separable under the profile's matching rules, **When** the scan completes, **Then** the resulting baseline cards remain separate.
2. **Given** FM Validation is active, **When** diagnostics are exported, **Then** raw detector span, measured center and bandwidth, match span, display span, persisted baseline span, and contextual metadata are separately identifiable.
3. **Given** revisit or refinement sees one bad wide observation, **When** later cards are displayed and matched, **Then** the card is not permanently ratcheted wider solely because of that observation.

---

### User Story 3 - Audit Effective Scan Parameters (Priority: P1)

As a developer, I want each scan or job to record the requested profile, whether it was applied or skipped, operator overrides, and final effective scanner parameters so scan behavior can be reproduced.

**Why this priority**: SDRwatch cannot be calibrated or debugged if the diagnostic bundle does not show which profile, overrides, device settings, and derived parameters actually governed a job.

**Independent Test**: Can be tested by launching jobs through the web/API path with an in-band FM profile request and an out-of-band FM profile request, then inspecting the diagnostic bundle manifest for requested, applied, skipped, fallback, and final effective parameter records.

**Acceptance Scenarios**:

1. **Given** a scan launched with the FM Broadcast profile and a valid FM range, **When** diagnostics are exported, **Then** the manifest shows the requested profile, applied profile, successful application status, operator overrides, and final effective scanner parameters.
2. **Given** a scan launched with the FM Broadcast profile outside the profile's valid band, **When** diagnostics are exported, **Then** the manifest records the requested profile, that the profile was skipped, the skipped reason, fallback behavior, and final effective parameters.
3. **Given** a job uses operator/API overrides, **When** the manifest is inspected, **Then** it distinguishes requested values, profile-derived values, overrides, and final effective values.

---

### User Story 4 - Match Web/API Parameters To Scanner Behavior (Priority: P2)

As a developer, I want characterization-related API and controller parameters to pass through to the scanner invocation so web/API scans behave the same as direct scanner smoke checks.

**Why this priority**: FM Validation and future replay tests need one auditable parameter contract across operator-facing jobs and internal scanner execution.

**Independent Test**: Can be tested with a controller command-construction check that submits all supported characterization and persistence parameters through the web/API path and verifies that the resulting scanner invocation contains the corresponding effective settings or a documented unsupported mapping.

**Acceptance Scenarios**:

1. **Given** a web/API scan request includes all supported characterization parameters, **When** the controller constructs the scanner invocation, **Then** every supported parameter is present in the scanner invocation using the project naming convention.
2. **Given** a parameter is not supported by the current scanner, **When** passthrough coverage is reviewed, **Then** the unsupported parameter is explicitly documented with the reason and any equivalent mapping.

---

### User Story 5 - Debug With Structured Diagnostic Summaries (Priority: P2)

As a developer, I want diagnostic bundles to summarize candidate emission, rejection, persistence matching, width decisions, revisit decisions, and characterization records without scraping human-readable logs.

**Why this priority**: Reliable RF monitoring needs reproducible evidence. Structured diagnostic summaries make failures visible without relying on fragile log text.

**Independent Test**: Can be tested by running deterministic diagnostic fixtures that emit, reject, match, fail to match, revisit, and characterize signals, then verifying detailed machine-readable events and compact aggregate counts in the exported bundle.

**Acceptance Scenarios**:

1. **Given** diagnostics are enabled for a completed scan, **When** the bundle is inspected, **Then** it includes structured aggregate counts for emitted clusters, rejected clusters, persistence matches, persistence no-matches, width decisions, revisit results, and characterization records.
2. **Given** detailed diagnostic events exceed bundle limits, **When** the bundle is exported, **Then** aggregate counts remain available and truncation or omission is recorded in the manifest.

---

### User Story 6 - Record SDR Device And Gain Telemetry (Priority: P2)

As an operator, I want basic SDR device telemetry recorded with each job so receiver-state changes are not confused with RF-environment changes.

**Why this priority**: Gain mode, requested gain, actual gain, sample rate, FFT size, bin width, and device identity can materially change observed RF evidence. The job record should make those differences visible.

**Independent Test**: Can be tested with current SDR hardware when available, plus no-hardware driver fixtures for unavailable fields, by verifying that the diagnostic manifest records requested values, actual values when exposed, and null or unavailable values without failing the scan.

**Acceptance Scenarios**:

1. **Given** a scan runs on current SDR hardware, **When** diagnostics are exported, **Then** requested gain, actual gain when available, gain mode, device identity or index when available, sample rate, FFT size, bin width, and selected profile are recorded.
2. **Given** the driver does not expose actual gain, supported gain list, serial, tuner, or another telemetry field, **When** diagnostics are exported, **Then** the field is represented as unavailable or null rather than causing scan failure.

### Edge Cases

- A real signal appears once per complete sweep loop in a non-overlapping window and never appears in adjacent same-sweep windows.
- Several observations occur in the same sweep loop and should not falsely satisfy a multi-loop persistence threshold unless the configured rules explicitly allow same-loop evidence.
- A signal's center is stable but its measured bandwidth varies within the profile's compatibility tolerance.
- A nearby signal appears between sweep loops and should not be merged with an existing candidate when center, span, or width rules say it is incompatible.
- FM Broadcast is requested outside its valid range and must be reported as skipped rather than silently applied.
- Operator overrides conflict with profile-derived values and the final effective value must be auditable.
- Device telemetry is partially unavailable from the current driver.
- Diagnostic detail exceeds bundle size limits.
- Revisit or refinement produces a single anomalously wide observation.
- Prior FM characterization fields exist and must remain semantically separate from any new persistence or telemetry data.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The feature MUST preserve GUI-first operation through the web UI and controller job lifecycle; direct scanner CLI checks remain internal backend smoke tests only.
- **FR-002**: The feature MUST preserve the existing `POST /api/jobs` request shape `{device_key, label, baseline_id, params}`.
- **FR-003**: The feature MUST define cross-sweep persistence as repeated observations of a likely same signal across multiple complete sweep loops.
- **FR-004**: A signal observed once per complete sweep loop at approximately the same center frequency and compatible bandwidth MUST be able to satisfy configured persistence thresholds.
- **FR-005**: Cross-sweep persistence MUST work when sweep windows are non-overlapping and the signal does not appear in adjacent same-sweep windows.
- **FR-006**: Cross-sweep state MUST respect existing profile persistence parameters where appropriate and MUST NOT require weakening FM control-band defaults to one hit in one window merely to produce cards.
- **FR-007**: Cross-sweep matching MUST use bounded center, span, and width compatibility criteria so unrelated nearby signals do not merge.
- **FR-008**: The feature MUST preserve the distinct meanings of raw detector span, measured center and bandwidth, match span, display span, persisted baseline span, and contextual or bandplan metadata.
- **FR-009**: FM Broadcast characterization MUST remain conservative and evidence-based; bandplan or profile context alone MUST NOT create a confident non-unknown classification candidate.
- **FR-010**: Nearby FM-like signals MUST remain separate when their centers are sufficiently separated by the active profile's matching rules.
- **FR-011**: Display bandwidth MAY be wider than measured bandwidth, but persistence matching MUST remain tighter than broad display presentation.
- **FR-012**: Revisit and refinement behavior MUST NOT permanently ratchet cards wider because of one anomalously wide observation.
- **FR-013**: Each scan or job diagnostic bundle MUST include a structured effective-parameter manifest.
- **FR-014**: The effective-parameter manifest MUST include requested profile, applied profile, profile application status, skipped reason when skipped, operator/API overrides, final effective scanner parameters, frequency range, step size, sample rate, FFT size, bin width, persistence thresholds, revisit settings, segment center mode, match/display bandwidth settings, width caps, gain settings, and driver/device metadata when available.
- **FR-015**: The manifest MUST distinguish requested values, profile-derived values, override values, skipped or fallback values, and final effective values.
- **FR-016**: The web/controller command builder MUST pass all scanner-supported characterization, revisit, and persistence parameters through to the scanner invocation using existing project naming conventions.
- **FR-017**: Passthrough coverage MUST include `segment_center_mode`, `segment_centroid_span_hz`, `segment_centroid_drop_db`, `segment_centroid_floor_margin_db`, `match_bandwidth_pad_hz`, `min_match_bandwidth_hz`, `display_bandwidth_pad_hz`, `min_display_bandwidth_hz`, `max_persist_width_hz`, `max_card_width_hz`, revisit-related parameters, and persistence-related parameters when supported by the current scanner.
- **FR-018**: If any requested parameter has a different current project name or is unsupported, the mapping, omission, or unsupported status MUST be documented in the feature artifacts or diagnostic manifest.
- **FR-019**: Diagnostic output MUST include machine-readable summaries for segment inventory, cluster emitted, cluster rejected, persistence match, persistence no-match, width decision, revisit queued, revisit result, and characterization record events.
- **FR-020**: Diagnostic bundles MUST include compact aggregate counts in addition to detailed event records where detailed records are already supported.
- **FR-021**: The UI, tests, and diagnostic review workflow MUST NOT depend on parsing human-readable scanner log text for state that is required as structured diagnostic data.
- **FR-022**: Device and gain telemetry MUST record requested gain, gain mode, actual applied gain when available, supported gain list when available, device index or identity metadata when available, sample rate requested or set, FFT size, bin width, and selected profile.
- **FR-023**: Unavailable driver telemetry MUST be represented as unavailable or null and MUST NOT fail an otherwise valid scan.
- **FR-024**: Schema and data changes MUST be additive or sidecar-oriented unless later planning proves that a migration is necessary.
- **FR-025**: The feature MUST preserve existing tests and behavior for FM characterization, raw/measured/match/display separation, bounded card width, revisit refinement, stable display center, FM Validation, and Discovery first-light behavior.
- **FR-026**: The feature MUST remain limited to lawful passive RF monitoring and signal characterization and MUST NOT add demodulation, content interception, decoding of private communications, offensive SIGINT workflows, a broad classifier, multi-SDR coordination, alert rules, long-term ML training, major UI redesign, or wholesale database replacement.

### Key Entities *(include if feature involves data)*

- **Sweep Loop Observation**: Evidence that a candidate signal was seen during one complete sweep loop, including loop identity, center estimate, compatible width information, and provenance for whether it came from coarse or revisit evidence.
- **Cross-Sweep Candidate State**: Temporary or persisted state that accumulates compatible observations across complete sweep loops until persistence thresholds are satisfied or the candidate expires.
- **Persistent Baseline Card**: The operator-facing stable signal record shown in the web UI, with persisted center and span kept separate from measured and display spans.
- **Measured Characterization Record**: Evidence-backed signal characterization containing raw detector span, measured center and bandwidth, match span, display span, confidence or evidence notes, and contextual metadata references.
- **Effective-Parameter Manifest**: A structured job-level record of requested, applied, skipped, overridden, fallback, and final effective scan parameters.
- **Profile Application Record**: The requested profile, applied profile, success or skipped state, skipped reason, valid range decision, and fallback behavior for a scan.
- **Controller Parameter Mapping**: The auditable relationship between web/API parameter names, controller-held job parameters, and scanner-supported invocation parameters.
- **Diagnostic Event Summary**: Detailed machine-readable events and compact aggregate counts for emitted, rejected, matched, unmatched, width, revisit, and characterization decisions.
- **Device Telemetry Snapshot**: Job-level receiver-state data, including gain, gain mode, actual gain when available, supported gains when available, device identity, sample rate, FFT size, bin width, and selected profile.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In a strict persistence validation scenario requiring multiple sweep observations, a stable signal appearing once per complete sweep loop is promoted after the configured number of sweep-loop observations and not before.
- **SC-002**: In the same scenario, repeated observations from only one sweep loop do not satisfy a threshold that requires multiple complete sweep-loop observations.
- **SC-003**: In nearby-signal validation scenarios, signals whose centers or compatible widths fall outside the active matching rules remain separate in at least 95% of deterministic fixture runs and in all accepted FM control-band regression fixtures.
- **SC-004**: FM Broadcast regression fixtures keep close but separable FM-like signals as separate baseline cards with bounded display widths and preserved raw, measured, match, display, persisted, and contextual fields.
- **SC-005**: For an in-band FM Broadcast scan with diagnostics enabled, the exported manifest includes 100% of the required requested, applied, override, final effective parameter, profile, frequency, sample, FFT, bin-width, persistence, revisit, width, gain, and available device fields.
- **SC-006**: For an out-of-band FM Broadcast profile request, the exported manifest records the requested profile, skipped state, skipped reason, fallback behavior, and final effective parameters.
- **SC-007**: Controller passthrough tests cover 100% of scanner-supported characterization, revisit, and persistence parameters listed in this spec, either as present in the invocation or explicitly documented as unsupported or mapped.
- **SC-008**: Diagnostic bundle validation reports aggregate counts for emitted clusters, rejected clusters, persistence matches, persistence no-matches, width decisions, revisit results, and characterization records without parsing human-readable logs.
- **SC-009**: Device telemetry validation represents requested gain, actual gain when available, gain mode, device identity or index when available, sample rate, FFT size, bin width, and selected profile for every completed diagnostic job.
- **SC-010**: Existing FM characterization, card-stability, revisit, stable-center, Discovery, and web/controller regression tests continue to pass after the feature is implemented.

## Assumptions

- FM Broadcast remains the stable control band for this feature because it provides a continuous, well-understood, locally observable validation environment.
- The current tested `006-fm-signal-characterization` branch is the behavioral baseline, even though its Spec Kit artifact directory remains `specs/004-fm-signal-characterization`.
- Operators will validate user-facing behavior through the web UI, controller job lifecycle, and diagnostic bundle export.
- Direct scanner invocation remains useful for backend smoke and parity checks, but it is not the operator workflow.
- Sweep-loop boundaries can be identified or represented well enough during planning to count one compatible observation per complete sweep loop.
- Driver-specific telemetry availability will vary; unavailable values are acceptable when recorded explicitly as unavailable or null.
- Diagnostic bundles may remain bounded, provided they include aggregate counts and explicit truncation or omission metadata.
- Later implementation should prefer small, test-driven, additive changes and should avoid destructive migrations unless planning produces a clear justification.
