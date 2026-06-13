# Feature Specification: FM Signal Characterization

**Feature Branch**: `006-fm-signal-characterization`

**Feature Directory**: `specs/004-fm-signal-characterization`

**Created**: 2026-06-12

**Status**: Draft

**Input**: User description: "Add an evidence-based FM signal characterization layer that keeps stable station-scale FM cards while separately tracking measured RF properties for FM Validation, without regressing card stability, Discovery behavior, GUI-first workflow, or the existing `/api/jobs` contract."

## User Scenarios & Testing

### User Story 1 - Validate Stable FM Cards Without Confusing Widths (Priority: P1)

As an SDRwatch operator, I can run FM Validation from the web GUI and keep stable station-scale FM cards while separately seeing measured RF values that are not confused with the card's display width.

**Why this priority**: The current FM Validation path fixed the card explosion problem. The next step must preserve that stable operator view while making the measured RF evidence more technically honest.

**Independent Test**: Can be tested through the browser by selecting FM Validation, enabling diagnostics mode, starting a controller-backed job, exporting a diagnostic bundle, and confirming that raw span, measured bandwidth, match span, and display span are separately visible for each characterized signal.

**Acceptance Scenarios**:

1. **Given** FM Validation is selected in the web UI, **When** the operator starts a scan through the existing controller job lifecycle, **Then** the resulting cards remain station-scale and do not revert to fragment-scale FM card explosion.
2. **Given** one FM station produces several narrow or spiky fragments across sweeps, **When** the operator reviews the resulting signal evidence, **Then** the displayed card span is not presented as the measured occupied bandwidth.
3. **Given** a characterized FM signal appears in diagnostics or exported evidence, **When** the operator inspects it, **Then** raw detector width, measured occupied bandwidth, match span, and display span are separately identifiable.

---

### User Story 2 - Inspect Characterization Evidence Without Overclaiming FM (Priority: P1)

As a maintainer or operator, I can inspect characterization confidence and evidence sources without treating bandplan or profile labels as proof that a signal is truly FM.

**Why this priority**: Later automatic classification work depends on preserving the distinction between contextual labels and measured RF evidence.

**Independent Test**: Can be tested through diagnostic export and no-hardware fixtures by confirming that contextual metadata remains separate from measured characterization and that low-evidence signals can stay unclassified.

**Acceptance Scenarios**:

1. **Given** a signal falls inside an FM Broadcast bandplan range, **When** the system records contextual metadata, **Then** the bandplan/profile label remains separate from measured characterization evidence.
2. **Given** a signal does not have enough measured evidence to support an FM conclusion, **When** characterization results are exported, **Then** the signal may remain unknown or tentative rather than being forced into an FM classification.
3. **Given** characterization confidence is reported, **When** a maintainer inspects the evidence, **Then** the evidence sources explain why the confidence is high, low, or unchanged.

---

### User Story 3 - Refine Measurements Through Revisit Without Multiplying Cards (Priority: P2)

As a maintainer, I can use revisit-derived evidence to improve center and bandwidth measurements without creating extra cards for the same station-scale signal.

**Why this priority**: Revisit is the most plausible path to better FM measurement quality, but it must not undo the card-stability gains from the previous feature.

**Independent Test**: Can be tested with deterministic fixtures by comparing coarse-pass and revisit-refined characterization fields while keeping the card count bounded and the persistence invariants intact.

**Acceptance Scenarios**:

1. **Given** revisit-capable FM Validation is active, **When** repeated sweeps gather more evidence for the same signal, **Then** measured center, measured bandwidth, or confidence can improve without increasing the number of operator-facing cards for that signal.
2. **Given** revisit updates an existing signal, **When** persistence and hysteresis logic apply, **Then** the stored span invariants remain valid.
3. **Given** optional FM-specific evidence such as pilot or RDS indicators is practical to detect, **When** it is found, **Then** it strengthens characterization evidence without being required for basic FM characterization.

---

### User Story 4 - Keep Nearby Stations Separate And Protect Non-FM Signals (Priority: P2)

As a maintainer, I can trust that nearby FM stations remain separate while narrow or weak non-FM signals are not widened or labeled as FM just because they happen to be near the FM band.

**Why this priority**: The feature loses value if it stabilizes FM by merging neighbors or by over-assigning FM labels to unrelated signals.

**Independent Test**: Can be tested with no-hardware fixtures containing multiple nearby FM-like stations plus narrow non-FM signals and by verifying that contextual labels do not override measured evidence.

**Acceptance Scenarios**:

1. **Given** multiple nearby FM stations are present, **When** characterization runs across repeated sweeps, **Then** they remain separate signals in raw evidence, persistence behavior, and display behavior.
2. **Given** a narrow or weak non-FM signal appears, **When** it lacks FM-like evidence, **Then** it remains narrow and is not labeled as an FM candidate based only on contextual metadata.
3. **Given** diagnostics are exported for mixed-signal scenarios, **When** a maintainer reviews them, **Then** the evidence clearly distinguishes contextual labels from measured classification evidence.

### Edge Cases

- An FM card remains stable while the measured occupied bandwidth varies sweep to sweep.
- The raw detector reports only tiny fragments for a strong FM station.
- Revisit confirms a different center estimate than the coarse pass.
- Two nearby FM stations briefly appear to overlap in coarse fragments but should remain separate cards.
- A narrow carrier appears inside the FM broadcast band but does not behave like FM broadcast.
- Bandplan data is missing, incomplete, or inconsistent with measured evidence.
- Optional FM-specific indicators are absent even for a valid FM station.
- Diagnostic bundle size limits truncate detailed decision evidence.
- Existing stable cards from FM Validation must remain compatible with later characterization updates.
- Persistence updates or revisit refinement must not violate `f_low_hz <= f_center_hz <= f_high_hz`.

## Requirements

### Functional Requirements

- **FR-001**: The feature MUST preserve GUI-first operation through the web UI and controller job lifecycle.
- **FR-002**: The feature MUST preserve the existing `POST /api/jobs` request shape `{device_key, label, baseline_id, params}`.
- **FR-003**: The feature MUST preserve RTL-SDR v4 Discovery as a separate first-light preset and preserve the current FM Validation stable-card behavior as the control baseline for this feature.
- **FR-004**: FM Validation MUST continue to show stable station-scale FM cards rather than reverting to fragment-scale FM card explosion.
- **FR-005**: The feature MUST represent raw detection segment evidence separately from measured characterization, persistence or match span, display or card span, and contextual metadata.
- **FR-006**: The system MUST retain or export the raw detector span that triggered characterization, including raw segment width, separately from any later refined measurement.
- **FR-007**: The system MUST derive and retain or export a best available measured center estimate and occupied bandwidth estimate that are distinct from raw segment width, persistence span, and display span.
- **FR-008**: The system MUST retain or export the span used to decide whether later detections are the same signal, separately from the measured occupied bandwidth and the display span.
- **FR-009**: The system MUST retain or export the stable operator-facing card span separately from the measured occupied bandwidth.
- **FR-010**: The system MUST retain or export SNR or CNR style metrics with provenance sufficient to explain whether the evidence came from coarse sweeps, revisit sweeps, stability over time, or optional FM-specific indicators.
- **FR-011**: The system MUST retain or export center-stability and bandwidth-stability evidence across sweeps for characterized FM Validation signals.
- **FR-012**: Revisit passes, when available, MUST refine characterization measurements or confidence without multiplying operator-facing cards for the same station-scale signal.
- **FR-013**: Multiple nearby FM stations MUST remain separate signals throughout raw measurement, persistence matching, and display behavior.
- **FR-014**: Weak, narrow, or otherwise non-FM signals MUST NOT be forced into FM characterization or labeled as FM candidates solely because bandplan or profile context points to the FM band.
- **FR-015**: Bandplan, region, service, and profile metadata MUST remain contextual annotations, not proof of measured modulation or service classification.
- **FR-016**: If the system proposes a classification candidate, it MUST keep the candidate and its supporting evidence separate from contextual bandplan or profile labels and it MUST be allowed to remain unknown when evidence is insufficient.
- **FR-017**: Characterization confidence MUST be accompanied by enough evidence detail to explain why the confidence increased, decreased, or remained low.
- **FR-018**: Diagnostics and exported troubleshooting evidence MUST expose raw, measured, match, and display span summaries separately for characterized signals.
- **FR-019**: Diagnostic evidence MUST identify whether characterization values came from coarse sweeps, revisit sweeps, stability aggregation across sweeps, or optional FM-specific indicators.
- **FR-020**: The persistence and update path MUST preserve the invariant `f_low_hz <= f_center_hz <= f_high_hz` after characterization-driven updates, revisit refinement, or hysteresis behavior.
- **FR-021**: The feature MUST preserve separation between Discovery and FM Validation presets and MUST NOT reintroduce the previous behavior where every tiny FM fragment becomes its own card.
- **FR-022**: The feature MUST remain narrow, test-first, and compatible with the existing FM stability work; it MUST NOT require a broad detector rewrite, CLI-first workflow, or schema migration unless later planning proves a migration is necessary.
- **FR-023**: Optional FM-specific evidence such as 19 kHz pilot or 57 kHz RDS or RBDS indicators MAY be added when practical, but the absence of those indicators MUST NOT block base characterization results.
- **FR-024**: If persistent storage of new characterization fields is deferred, the system MUST still expose characterization evidence through diagnostics and other non-destructive export paths well enough to validate behavior and support later classification planning.

### Key Entities

- **Raw Detection Segment**: The direct detector output from a sweep window, including the measured raw span and power metrics before later refinement.
- **Measured Characterization Record**: The best available estimate of signal center, occupied bandwidth, signal quality, stability, and confidence derived from coarse and revisit evidence.
- **Persistence Match Span**: The span used to decide whether later detections belong to the same long-lived signal record.
- **Display Card Span**: The stable operator-facing span used to render cards in the UI.
- **Contextual Signal Metadata**: Bandplan, service, region, notes, and selected profile information that provide context but do not prove modulation or service identity.
- **Classification Evidence**: The candidate label, supporting observations, and confidence explanation that may support later automatic classification while remaining separate from contextual labels.
- **Revisit Measurement**: Follow-up evidence used to refine characterization without exploding card count.

## Success Criteria

### Measurable Outcomes

- **SC-001**: In a validation scenario containing one station-scale FM signal composed of multiple narrow fragments, the system produces no more than two operator-facing cards for that signal while separately exposing raw, measured, match, and display width values.
- **SC-002**: In validation scenarios with multiple nearby FM-like stations, all station pairs separated by normal FM channel spacing remain separate operator-facing cards and separate characterization records.
- **SC-003**: In narrow non-FM acceptance scenarios outside FM Validation, the signal remains narrow and is never labeled as an FM candidate from contextual metadata alone.
- **SC-004**: Exported diagnostic evidence for an FM Validation run separately reports raw segment width, measured occupied bandwidth, match span, and display span for every characterized signal included in the export sample.
- **SC-005**: Revisit-enabled characterization updates measured center, measured bandwidth, or confidence for repeated signals without increasing the number of operator-facing cards for the same underlying stations.
- **SC-006**: Any non-unknown classification candidate in acceptance tests includes evidence sources explaining the candidate, and no candidate is created from bandplan or profile context alone.
- **SC-007**: Automated persistence and revisit acceptance tests preserve the invariant `f_low_hz <= f_center_hz <= f_high_hz` for 100% of updated signals.
- **SC-008**: FM Validation continues to present stable station-scale cards instead of hundreds of fragment-scale cards in the accepted validation scenarios derived from the FM stability control case.

## Assumptions

- The existing `fm_broadcast` profile and current FM Validation stable-card behavior remain the control path for this feature.
- FM broadcast remains the first target for characterization because it is a stable, continuous, high-confidence validation environment.
- Operators will validate this feature through the web UI, controller job lifecycle, and diagnostic bundle export rather than through direct scanner CLI runs.
- Optional FM-specific indicators such as 19 kHz pilot or 57 kHz RDS or RBDS evidence are opportunistic enhancements, not prerequisites for the first implementation.
- Planning will prefer the narrowest storage path that preserves compatibility, starting from current tables and diagnostic exports before adding persistent schema surface.
