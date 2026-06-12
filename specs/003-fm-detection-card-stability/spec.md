# Feature Specification: FM Detection Card Stability

**Feature Branch**: `devControl`

**Feature Directory**: `specs/003-fm-detection-card-stability`

**Created**: 2026-06-12

**Status**: Draft

**Input**: User description: "Plan the SDRwatch feature `fm-detection-card-stability` from the current `devControl` baseline. Use the FM diagnostic bundle as evidence. Preserve GUI-first operation, preserve Discovery as a first-light preset, add or improve an FM Validation path that avoids hundreds of tiny FM-band cards, identify the likely failing stage, prefer small testable changes, and avoid solving this by simply raising thresholds until cards disappear."

## User Scenarios & Testing

### User Story 1 - Validate FM Band Without Card Explosion (Priority: P1)

As an SDRwatch operator, I can start an FM Validation scan for 88-108 MHz from the web GUI and get a smaller set of stable, useful FM-band signal cards instead of hundreds of tiny sub-kHz cards.

**Why this priority**: The current Discovery path proves the GUI can produce cards, but the FM-band bundle shows the next blocking failure is card overproduction and instability.

**Independent Test**: Can be tested through the browser by selecting the FM Validation preset, enabling Diagnostics mode, starting a controller-backed job, exporting a diagnostic bundle, and confirming persisted detections are not dominated by 293-880 Hz rows.

**Acceptance Scenarios**:

1. **Given** an operator has selected an FM monitoring zone and RTL-SDR device, **When** they choose FM Validation and start a scan through the web GUI, **Then** the submitted job uses the existing `/api/jobs` lifecycle and applies FM-specific stabilization settings.
2. **Given** the FM Validation scan completes multiple sweeps, **When** the operator reviews cards or exported `baseline_detections`, **Then** the result is not hundreds of tiny 293-880 Hz cards.
3. **Given** diagnostics are enabled, **When** the operator exports a bundle, **Then** the bundle contains enough create/update/merge/missing/revisit evidence to explain card stability decisions.

---

### User Story 2 - Keep Discovery as First-Light (Priority: P1)

As an SDRwatch operator, I can still use RTL-SDR v4 Discovery to prove end-to-end operation and produce visible cards from the GUI.

**Why this priority**: The completed scan-control work intentionally made first-light scans produce cards. This feature must not regress back to the earlier zero-card failure.

**Independent Test**: Can be tested by rendering the control page, selecting Discovery, copying current scan settings, and verifying the payload still uses relaxed promotion and first-light values.

**Acceptance Scenarios**:

1. **Given** the operator selects RTL-SDR v4 Discovery, **When** they copy or start a scan, **Then** the preset remains available and continues to submit first-light card-producing settings.
2. **Given** implementation adds FM Validation, **When** Discovery is selected, **Then** Discovery is not silently replaced by stricter FM Validation behavior.

---

### User Story 3 - Preserve Signal Separation and Narrow Non-FM Behavior (Priority: P2)

As a maintainer, I can trust that FM-specific stabilization groups fragments from one FM-like signal without merging separated FM-like signals, and without widening unrelated narrow signals outside the FM-specific behavior.

**Why this priority**: A fix that blindly widens or merges all detections would hide cards but damage core scanner behavior.

**Independent Test**: Can be tested with deterministic detector/persistence fixtures before implementation: one wide/spiky FM-like signal should produce a small bounded number of cards, separated FM-like signals should remain separate, and narrow non-FM signals should remain narrow outside FM Validation/profile behavior.

**Acceptance Scenarios**:

1. **Given** one synthetic FM-like signal with several narrow spikes inside one station-scale span, **When** the FM Validation settings are applied, **Then** the system updates or merges into a small bounded number of persisted detections rather than dozens.
2. **Given** multiple separated FM-like signals, **When** the FM Validation settings are applied, **Then** the signals remain separate persistent detections.
3. **Given** a narrow non-FM signal outside FM-specific settings, **When** default detection behavior runs, **Then** the persisted width remains narrow and is not forced to FM scale.

---

### User Story 4 - Make Persistence Decisions Observable (Priority: P2)

As a maintainer or operator sharing a diagnostic bundle, I can see why cards were created, updated, missed, clamped, merged, or revisited.

**Why this priority**: The current bundle proves the symptom but still requires inference about insert/update/no-match and width decisions.

**Independent Test**: Can be tested without hardware by running persistence/diagnostics fixtures and verifying exported diagnostics contain bounded decision summaries.

**Acceptance Scenarios**:

1. **Given** a detection updates an existing baseline row, **When** diagnostics are enabled, **Then** the bundle includes an update/match decision with center delta and width information.
2. **Given** a detection creates a new row, **When** diagnostics are enabled, **Then** the bundle includes a create/no-match decision.
3. **Given** a width is expanded, clamped, or rejected, **When** diagnostics are enabled, **Then** the bundle includes the configured cap and resulting width.
4. **Given** two-pass is enabled, **When** revisit decisions occur, **Then** diagnostics include queued, confirmed, pruned, or missed revisit counts.

### Edge Cases

- FM Validation is selected for a non-FM monitoring zone.
- FM Validation is selected but the operator manually edits advanced values.
- The GUI selects `fm_broadcast` profile but also submits explicit values that override profile defaults.
- The built-in `fm_broadcast` profile applies hidden span-shaping fields that are not separately surfaced as CLI flags.
- `max_detection_width_hz` is empty in the GUI, making the scanner default `0.0` disable hard width caps.
- Two-pass is enabled through profile defaults but the generated command does not visibly include `--two-pass`.
- Strong FM activity creates many narrow segments in one sweep window.
- Multiple stations are separated by normal FM channel spacing and must not be merged.
- Existing persisted rows are tiny or missing from prior runs.
- Repeated nearby detections should update an existing row instead of creating a new one.
- Card stability improves for FM but Discovery still needs to produce first-light cards.
- Diagnostics are truncated by bundle bounds; the manifest must report truncation.

## Requirements

### Functional Requirements

- **FR-001**: The feature MUST preserve GUI-first operation through the web UI and controller job lifecycle.
- **FR-002**: The feature MUST preserve the existing `POST /api/jobs` request shape `{device_key, label, baseline_id, params}`.
- **FR-003**: The feature MUST preserve RTL-SDR v4 Discovery as a first-light GUI preset that can produce cards.
- **FR-004**: The feature MUST add or improve a GUI-accessible FM Validation path for 88-108 MHz.
- **FR-005**: FM Validation MUST avoid producing hundreds of tiny 293-880 Hz persisted cards for normal FM-band validation.
- **FR-006**: FM Validation MUST use small, testable changes to existing segmentation, grouping, profile, controller, persistence, or diagnostics behavior.
- **FR-007**: The feature MUST NOT solve the problem only by raising thresholds until cards disappear.
- **FR-008**: The feature MUST distinguish the Discovery first-light path from the FM Validation stability path in UI labels, settings, and tests.
- **FR-009**: FM Validation MUST use existing scanner/controller parameters where practical, especially the existing `fm_broadcast` profile behavior if it remains the smallest compatible path.
- **FR-010**: If FM Validation relies on `profile=fm_broadcast`, tests MUST prove the GUI payload and controller command apply or preserve the expected profile behavior.
- **FR-011**: If FM Validation needs profile fields that are not reachable through existing controller/CLI flags, the implementation MUST either rely on `--profile fm_broadcast` intentionally or add narrow passthrough flags for those existing fields.
- **FR-012**: FM-like spiky signals in tests MUST be grouped, matched, widened for display, or otherwise stabilized so they do not create dozens of cards from one station-scale signal.
- **FR-013**: Multiple separated FM-like signals MUST remain separate.
- **FR-014**: Narrow non-FM signals MUST remain narrow outside FM-specific behavior.
- **FR-015**: Repeated nearby detections MUST update existing baseline detections when they match configured FM Validation tolerances.
- **FR-016**: Width clamp behavior MUST be bounded and observable in tests and diagnostics.
- **FR-017**: Two-pass behavior MUST be covered if enabled or exposed by FM Validation.
- **FR-018**: The GUI FM Validation preset MUST submit correct `/api/jobs` params and remain copyable through Copy current scan settings.
- **FR-019**: Diagnostics MUST show create/update/no-match/missing/width/revisit decisions clearly enough to analyze future bundles without inferring solely from final rows.
- **FR-020**: The feature MUST NOT introduce a detector rewrite, database schema rewrite, new frontend framework, cloud dependency, simulation requirement, or CLI-first operator workflow.

### Key Entities

- **FM Validation Preset**: GUI-applied scan configuration for the 88-108 MHz FM band, separate from first-light Discovery.
- **Discovery Preset**: Existing first-light GUI scan configuration that remains available and card-producing.
- **FM Profile Application**: The scanner/controller path that applies built-in `fm_broadcast` profile defaults, including FM-specific hidden span-shaping behavior if used.
- **Detection Fragment**: A narrow emitted segment from a sweep window, often one to several FFT bins wide.
- **Stable FM Card**: A persisted baseline detection that represents a useful FM-band signal span and remains stable across sweeps.
- **Persistence Match Decision**: The create/update/no-match outcome when a promoted detection is compared to existing `baseline_detections`.
- **Width Clamp Decision**: The result of applying configured minimum display/match widths or maximum width caps.
- **Revisit Decision**: Two-pass queue, confirmation, prune, miss, or skip outcome.
- **Diagnostic Bundle Evidence**: Bounded exported records that explain scan settings, detections, persistence decisions, and manifest truncation.

## Success Criteria

### Measurable Outcomes

- **SC-001**: A synthetic wide/spiky FM-like fixture produces a small bounded number of persisted detections under FM Validation instead of dozens.
- **SC-002**: A synthetic multiple-station FM-like fixture preserves separated signals as separate persisted detections.
- **SC-003**: A narrow non-FM fixture remains narrow when FM-specific behavior is not selected.
- **SC-004**: Repeated nearby detections update existing baseline rows rather than creating new rows under the configured matching tolerances.
- **SC-005**: Width cap and width floor behavior are covered by automated tests and are visible in diagnostic output.
- **SC-006**: If FM Validation enables two-pass, tests verify the generated controller command or scanner-applied settings and diagnostic revisit counts.
- **SC-007**: The GUI FM Validation preset copy output includes the expected existing `/api/jobs` params, including `profile` and/or explicit FM-specific settings chosen by the implementation.
- **SC-008**: RTL-SDR v4 Discovery remains present and its test coverage continues to prove first-light card-producing settings.
- **SC-009**: A GUI FM Validation diagnostic bundle no longer shows hundreds of tiny 293-880 Hz FM cards; active persisted detections are not dominated by sub-5 kHz widths.
- **SC-010**: Exported diagnostics show enough create/update/merge/missing/revisit decisions to explain why cards were or were not stabilized.

## Assumptions

- The FM diagnostic bundle `C:/Users/User/Downloads/sdrwatch-diagnostics-532516344b4b/` is representative of the current FM-band failure mode.
- The existing `fm_broadcast` profile is intended to encode FM-band stabilization behavior and should be reused or made visible before adding a new detector path.
- The Discovery preset should remain noisy enough to prove end-to-end card creation; FM Validation can be more band-specific and stable.
- Real RF environments vary, so automated synthetic fixtures should define deterministic correctness while hardware acceptance confirms the operator-facing result.
- Existing compatibility with scanner CLI flags and controller params remains required, but scanner CLI runs are internal backend smoke tests only.
