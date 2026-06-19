# Feature Specification: Profile-Governed Signal Identity Span and Revisit Authority

**Feature Branch**: `008-multi-rtl-guard-rover`

**Created**: 2026-06-19

**Status**: Draft

**Input**: User description: "Revise the feature specification so the current Spec Kit artifacts align around Profile-Governed Signal Identity Span and Revisit Authority. Keep the pass planning/specification-only, generic and profile-driven, preserve raw fragment evidence, prevent raw fragment width from becoming identity/persist/display width by accident, and keep stale multi-RTL tasks out of implementation."

**Artifact Alignment Note**: This specification and the regenerated `tasks.md` are now aligned around the focused span-policy update in `specs/008-multi-rtl-guard-rover`. Do not continue implementation from the earlier hardware-aware multi-RTL task list.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Preserve Honest Raw Evidence While Showing Policy-Shaped Identity (Priority: P1)

As an SDRwatch operator reviewing a diagnostic bundle or scan results, I need raw detector and revisit fragments to stay visible as raw evidence while identity, persisted card, and display spans follow the active profile policy, so I can understand why a raw fragment may be tiny without mistaking it for the signal's identity width.

**Why this priority**: This is the core confusion observed in the Pi 5 canary. The system already applies profile audit/export correctly, but tiny raw fragments can still be misread as identity or card width. The first viable slice must make these semantics explicit and enforce the correct width floors at the identity and persistence layers.

**Independent Test**: Can be tested with a normal single-device web/controller diagnostic scan or no-hardware diagnostic fixture that emits a tiny raw segment under a broad continuous profile, then verifies that raw fragment width remains tiny while identity, persisted/card, and display spans are distinct and policy-shaped.

**Acceptance Scenarios**:

1. **Given** a profile with `min_match_bandwidth_hz`, `min_identity_bandwidth_hz`, `min_persist_bandwidth_hz`, and `min_display_bandwidth_hz`, **When** a tiny raw detector fragment is characterized, **Then** SDRwatch preserves the tiny raw fragment width and also reports an identity/match span that does not fall below the active identity floor except documented scan-edge clipping.
2. **Given** a broad continuous canary profile, **When** raw characterization evidence is only hundreds of Hz to a few kHz wide, **Then** SDRwatch labels that width as raw fragment or threshold-fragment evidence instead of presenting it as persisted card width or display width.
3. **Given** a persisted card is updated repeatedly from narrow fragments, **When** the active profile has a persist floor, **Then** the persisted/card span does not shrink below that floor except documented scan-edge clipping.
4. **Given** a display span is shown to an operator, **When** raw and measured bandwidths are narrower than the display floor, **Then** SDRwatch keeps display bandwidth separate and does not present it as measured occupied bandwidth.

---

### User Story 2 - Gate Revisit Authority Separately from Confirmation (Priority: P1)

As an SDRwatch maintainer or operator diagnosing revisit results, I need revisit evidence to be able to confirm presence without automatically moving center or changing identity/persisted width, so a tiny or offset revisit sub-peak cannot destabilize cards or split/duplicate tracks.

**Why this priority**: Revisit refinement is useful, but narrow or far-offset revisit fragments can be too weak to control identity. Splitting confirmation from identity update protects stability without hiding revisit evidence.

**Independent Test**: Can be tested by running or simulating revisit confirmations where revisit bandwidth is below the active identity-update floor or center delta exceeds the configured gate, then verifying confirmation-only diagnostics and unchanged identity/persisted span.

**Acceptance Scenarios**:

1. **Given** a revisit segment whose bandwidth is below `min_revisit_bandwidth_for_identity_update_hz`, **When** the policy forbids narrow revisit identity updates, **Then** SDRwatch records revisit confirmation evidence but marks the revisit as `confirmation_only` or rejected for identity update.
2. **Given** a revisit segment whose center delta exceeds `max_revisit_center_delta_for_identity_update_hz`, **When** the policy forbids that center movement, **Then** SDRwatch does not move identity center and records the center-gate decision.
3. **Given** a fragmented or ambiguous revisit result, **When** the active `fragmented_revisit_policy` is confirmation-only, **Then** SDRwatch records presence evidence without shrinking or moving identity.
4. **Given** `allow_revisit_to_shrink_identity` is false, **When** revisit evidence is narrower than the current identity or persist floor, **Then** revisit evidence cannot shrink identity or persisted/card width below the policy floor.

---

### User Story 3 - Support Profile-Neutral Width Policies (Priority: P2)

As a maintainer defining scan profiles, I need broad continuous, narrowband watchlist, unknown discovery, and guard/event profiles to express different identity, persistence, display, and revisit authority policies, so SDRwatch does not bake FM-like assumptions into generic signal behavior.

**Why this priority**: FM Broadcast is only the live RF canary. SDRwatch must remain useful across profiles with very different bandwidth expectations and must not force 200 kHz display or persistence semantics onto narrowband or discovery workflows.

**Independent Test**: Can be tested with multiple profile fixtures: a broad continuous canary profile, a narrowband voice/watchlist profile, an unknown discovery profile, and a guard/event profile. Each fixture verifies that its width floors and revisit gates are honored independently.

**Acceptance Scenarios**:

1. **Given** a broad continuous canary profile, **When** tiny raw fragments are observed, **Then** identity, persisted/card, and display spans may use larger profile floors while raw evidence remains tiny and visible.
2. **Given** a narrowband voice or watchlist profile, **When** narrow signals are observed, **Then** SDRwatch allows much smaller identity, persist, and display floors than a broad continuous profile.
3. **Given** an unknown discovery profile with no broad floor configured, **When** raw fragments are detected, **Then** SDRwatch preserves narrow discovery behavior and does not inherit broad display or persist widths.
4. **Given** a guard/event profile, **When** a fast candidate event is emitted, **Then** SDRwatch can report event evidence without immediately implying stable baseline-card identity.

---

### User Story 4 - Preserve Close-Signal Separation (Priority: P2)

As an SDRwatch operator monitoring crowded spectrum, I need width floors to stabilize identity and cards without over-merging close but distinct signals, so profile policy improves clarity without hiding nearby activity.

**Why this priority**: Applying a width floor too early can cause nearby signals to overlap and merge. The policy must act at semantic boundaries, not at raw candidate formation.

**Independent Test**: Can be tested with fixtures containing close signals whose raw clusters or centers should remain distinct, then verifying that applying identity/persist/display floors after candidate formation does not merge them.

**Acceptance Scenarios**:

1. **Given** two close signals that are distinct under the active center and cluster policy, **When** a width floor is applied, **Then** SDRwatch keeps the signals separate.
2. **Given** raw candidate or cluster formation is in progress, **When** the active profile has a broad identity or persist floor, **Then** SDRwatch does not widen raw cluster extents before close-signal matching decisions.
3. **Given** max width caps and width-ratio rejection are configured, **When** one observation is much wider than the persisted/card span, **Then** SDRwatch prevents that observation from absorbing unrelated nearby signals.

---

### User Story 5 - Audit Effective Policy and Diagnostics End-to-End (Priority: P3)

As an operator or maintainer validating a scan, I need effective parameters and diagnostics to expose the active signal span policy and each width/revisit decision, so I can prove whether the scanner, controller export, and diagnostic bundle agree.

**Why this priority**: Recent Pi 5 canary work fixed profile audit/export contradictions. This feature must preserve that trust while adding the derived span policy and clearer bandwidth interpretation.

**Independent Test**: Can be tested by exporting a diagnostic bundle for a normal single-device RTL scan and confirming that effective parameters, decision summaries, and characterization records agree on the active policy, raw fragment interpretation, and revisit authority.

**Acceptance Scenarios**:

1. **Given** a normal single-device RTL scan with `rtlsdr_native`, `device_key=rtl:0`, `requested_profile=fm_broadcast`, `applied_profile=fm_broadcast`, and no receiver role, **When** diagnostics are exported, **Then** effective parameters expose the derived signal span policy while role-run fields remain null.
2. **Given** a characterization record is exported, **When** it includes old bandwidth fields, **Then** it also includes additive fields clarifying raw fragment, measured, identity, persisted/card, display, and bandwidth interpretation semantics.
3. **Given** a revisit confirmation occurs, **When** diagnostics are exported, **Then** the record states whether the revisit was confirmation-only or identity-update authority and why.
4. **Given** effective-parameter export falls back through available sources, **When** the bundle is inspected, **Then** scanner-owned profile audit data remains authoritative when present and the derived policy is not contradicted by controller fallback data.

### Edge Cases

- A raw detector fragment is only hundreds of Hz wide under a profile with much larger identity and display floors.
- A revisit segment is tiny but correctly confirms signal presence.
- A revisit segment is strong but too far from the stable center to update identity.
- A fragmented revisit produces multiple plausible sub-peaks.
- A profile intentionally leaves identity and persist floors unset for unknown discovery.
- A narrowband profile sets much smaller floors than a broad continuous profile.
- A guard/event profile emits fast candidates without baseline-learning recurrence.
- A persisted/card span is clipped by the scan or baseline edge and becomes narrower than the requested policy floor.
- Two close signals remain distinct under active center/cluster policy but would overlap if raw extents were widened too early.
- Existing diagnostic consumers read old bandwidth fields and ignore new additive fields.
- Existing cross-sweep persistence, effective-parameter export, and Slice 1 multi-RTL inventory/backend-gating behavior must remain valid.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: SDRwatch MUST define an active signal span policy for each scan from profile defaults, operator overrides, and existing compatibility aliases.
- **FR-002**: The signal span policy MUST support `min_identity_bandwidth_hz`, `min_persist_bandwidth_hz`, `max_persist_bandwidth_hz`, `min_match_bandwidth_hz`, `min_display_bandwidth_hz`, `allow_revisit_to_shrink_identity`, `allow_revisit_to_move_center`, `min_revisit_bandwidth_for_identity_update_hz`, `max_revisit_center_delta_for_identity_update_hz`, `fragmented_revisit_policy`, `raw_fragment_interpretation`, and either `center_smoothing_enabled` or `center_stability_mode`.
- **FR-003**: When `min_identity_bandwidth_hz` is unset and `min_match_bandwidth_hz` is set, SDRwatch MUST use `min_match_bandwidth_hz` as the default identity floor.
- **FR-004**: When `min_persist_bandwidth_hz` is unset and `min_match_bandwidth_hz` is set, SDRwatch MUST use `min_match_bandwidth_hz` as the default persisted/card floor.
- **FR-005**: SDRwatch MUST resolve `max_persist_bandwidth_hz` from existing `max_persist_width_hz`, `max_card_width_hz`, or `max_detection_width_hz` aliases when an explicit max persist policy is not set.
- **FR-006**: If no profile or policy floor is set, SDRwatch MUST preserve current narrow discovery behavior and MUST NOT apply broad continuous-profile widths by default.
- **FR-007**: SDRwatch MUST preserve raw detector and revisit fragment center and width as honest raw evidence, even when those widths are tiny.
- **FR-008**: SDRwatch MUST label raw fragment evidence clearly with fields such as `raw_fragment_bandwidth_hz`, `raw_fragment_center_hz`, and a bandwidth interpretation such as `threshold_fragment`.
- **FR-009**: SDRwatch MUST keep measured characterization evidence separate from identity/match span, persisted/card span, and operator display span.
- **FR-010**: SDRwatch MUST NOT present displayed card width as measured occupied bandwidth.
- **FR-011**: SDRwatch MUST derive identity/match span from the active profile policy and MUST NOT allow it to shrink below the active identity floor except documented scan-edge clipping.
- **FR-012**: SDRwatch MUST keep identity/match span separate from raw fragment width and display width in diagnostics and summaries.
- **FR-013**: SDRwatch MUST NOT widen live raw cluster extents before raw candidate formation or close-signal matching decisions solely because a width floor exists.
- **FR-014**: SDRwatch MUST apply `min_persist_bandwidth_hz` and `max_persist_bandwidth_hz` to persisted/card span semantics.
- **FR-015**: Persistence width smoothing MUST NOT shrink persisted/card width below the active persist floor except documented scan-edge clipping.
- **FR-016**: Persisted/card width MUST be reported as policy-shaped card identity evidence, not as true occupied bandwidth.
- **FR-017**: Display span MUST remain operator-facing and governed by `min_display_bandwidth_hz`.
- **FR-018**: Narrowband and discovery profiles MUST be able to keep display spans narrow when their policies do not request broad display floors.
- **FR-019**: SDRwatch MUST distinguish revisit `confirmation_only` from revisit `identity_update`.
- **FR-020**: Revisit confirmation MUST be able to record evidence, confirm presence, and clear missing state without being allowed to move center or update identity width.
- **FR-021**: Revisit identity update MUST be allowed only when configured bandwidth, center-delta, fragmented-revisit, and shrink-authority gates pass.
- **FR-022**: Revisit bandwidth gates MUST be configurable per profile and MUST NOT globally force `min_revisit_bandwidth_for_identity_update_hz` to equal `min_identity_bandwidth_hz` for every profile.
- **FR-023**: Revisit center movement MUST obey `allow_revisit_to_move_center` and `max_revisit_center_delta_for_identity_update_hz` when configured.
- **FR-024**: When `allow_revisit_to_shrink_identity` is false, revisit evidence MUST NOT shrink identity or persisted/card width below the active policy floor.
- **FR-025**: Fragmented or ambiguous revisit evidence MUST obey `fragmented_revisit_policy`.
- **FR-026**: Diagnostics MUST be additive and preserve existing bandwidth and characterization fields for compatibility.
- **FR-027**: Diagnostics MUST expose `raw_fragment_bandwidth_hz`, `raw_fragment_center_hz`, `measured_occupied_bandwidth_hz`, `identity_match_bandwidth_hz`, `persisted_card_bandwidth_hz`, `display_bandwidth_hz`, `bandwidth_interpretation`, `width_floor_applied_hz`, `persist_width_floor_applied_hz`, `baseline_clipped`, and `clip_reason` where applicable.
- **FR-028**: Revisit diagnostics MUST expose `revisit_authority`, `identity_update_allowed`, `confirmation_recorded`, `revisit_center_delta_hz`, `revisit_bandwidth_hz`, `revisit_bandwidth_policy_result`, and `revisit_center_policy_result`.
- **FR-029**: Effective parameters MUST expose the derived signal span policy under `signal_span_policy` or as additive fields under `span_controls`.
- **FR-030**: Effective parameters MUST include `min_identity_bandwidth_hz`, `min_persist_bandwidth_hz`, `max_persist_bandwidth_hz`, `min_revisit_bandwidth_for_identity_update_hz`, `max_revisit_center_delta_for_identity_update_hz`, `allow_revisit_to_shrink_identity`, `allow_revisit_to_move_center`, `fragmented_revisit_policy`, `raw_fragment_interpretation`, and `center_smoothing_enabled` or equivalent active center-stability mode.
- **FR-031**: SDRwatch MUST preserve profile audit/export trust for normal single-device RTL scans, including the ability to show `rtlsdr_native`, `rtl:0`, requested/applied profile agreement, and null role-run fields for legacy jobs.
- **FR-032**: SDRwatch MUST preserve existing `/api/jobs` compatibility for current single-device workflows.
- **FR-033**: SDRwatch MUST preserve Slice 1 multi-RTL hardware inventory, capability reporting, backend gating, and RTL-only runtime boundaries.
- **FR-034**: SDRwatch MUST NOT add database migrations for this feature.
- **FR-035**: SDRwatch MUST NOT add new multi-RTL role assignment, grouped role-run, Airspy, HackRF, Soapy, UI redesign, signal fusion, continuous raw IQ capture, Rust DSP rewrite, or FM-specific tuning scope as part of this feature.

### Scope Boundaries

- This feature is a generic profile-policy update for signal span semantics and revisit authority.
- FM Broadcast is only a live canary/control profile; the feature must not hard-code 88-108 MHz behavior or FM-specific logic in generic detection or persistence behavior.
- Broad continuous, narrowband, unknown discovery, and guard/event profiles must be able to express different policies.
- Width floors apply after raw candidate and cluster formation at identity, persistence, and display semantic boundaries.
- No database migration is required.
- Existing diagnostic fields remain available; new fields are additive.
- The current web UI and controller job lifecycle remain the operator acceptance path. Scanner CLI checks are internal backend smoke only.
- `tasks.md` has been regenerated for this focused span-policy update; implementation must follow the current task list rather than the earlier multi-RTL role-run task list.

### Key Entities *(include if feature involves data)*

- **SignalSpanPolicy**: Represents the active profile-governed rules for identity, persistence, display, and revisit authority spans.
- **Raw Fragment Evidence**: Represents raw detector or revisit segment center, width, and interpretation as threshold-fragment evidence.
- **Measured Characterization Evidence**: Represents measured center and occupied-bandwidth-like observations with explicit confidence and interpretation, without implying identity or display width.
- **Identity/Match Span**: Represents the profile-shaped span used for signal identity and matching semantics.
- **Persisted/Card Span**: Represents the policy-shaped span stored and shown as card/baseline identity, bounded by persist floors and caps.
- **Display Span**: Represents the operator-facing span governed by display policy.
- **Revisit Authority Decision**: Represents whether revisit evidence is confirmation-only or allowed to update identity, including bandwidth, center-delta, fragmented-evidence, and shrink-authority decisions.
- **Bandwidth Interpretation**: Represents diagnostic explanation of whether a width is raw fragment, measured evidence, identity policy, persisted card, or display presentation.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In no-hardware tests, a tiny raw segment remains visible as raw fragment evidence with its original narrow bandwidth.
- **SC-002**: In no-hardware tests, a tiny raw segment under a profile with an identity floor does not shrink identity/match span below that floor except documented scan-edge clipping.
- **SC-003**: In no-hardware tests, persisted/card span does not shrink below `min_persist_bandwidth_hz` except documented scan-edge clipping.
- **SC-004**: In no-hardware tests, display span remains independently governed by `min_display_bandwidth_hz` and is not reported as measured occupied bandwidth.
- **SC-005**: In no-hardware revisit tests, revisit evidence below the configured bandwidth gate is recorded as confirmation-only or rejected for identity update.
- **SC-006**: In no-hardware revisit tests, revisit evidence outside the configured center-delta gate is confirmation-only or rejected for identity update and does not move identity center.
- **SC-007**: In close-signal tests, distinct nearby signals remain separate when center/cluster policy says they are distinct, even when width floors exist.
- **SC-008**: In profile variation tests, a narrowband profile can keep identity, persisted/card, and display spans narrow.
- **SC-009**: In profile variation tests, unknown discovery without broad floors is not forced into broad display or persisted/card widths.
- **SC-010**: Effective-parameter exports include the active signal span policy and do not contradict scanner-owned profile audit data when present.
- **SC-011**: Diagnostic records include bandwidth interpretation and revisit authority decisions while preserving existing fields.
- **SC-012**: Existing cross-sweep persistence tests continue to pass.
- **SC-013**: Existing effective-parameter/profile export tests continue to pass.
- **SC-014**: Existing Slice 1 multi-RTL inventory and backend-gating tests continue to pass.
- **SC-015**: On a Pi 5 canary run, ordinary persisted cards do not fall below the active persist floor except documented scan-edge clipping, while raw/revisit fragments may remain small and labeled.

## Assumptions

- The active Pi 5 canary is a normal single-device RTL scan using `rtlsdr_native`, `device_key=rtl:0`, requested/applied `fm_broadcast`, and null role-run metadata.
- Current profile audit/export behavior is trustworthy and should be preserved.
- Existing `min_match_bandwidth_hz`, `min_display_bandwidth_hz`, `max_persist_width_hz`, `max_card_width_hz`, and `max_detection_width_hz` semantics remain compatibility inputs to the policy.
- Broad continuous profiles may choose larger identity/display floors, but those values are profile policy, not global SDRwatch assumptions.
- Narrowband, discovery, guard/event, and future profiles may choose smaller or unset floors and different revisit gates.
- A first implementation may be smaller than the full policy surface if it covers policy/effective-parameter plumbing, identity/persist floor enforcement, diagnostics clarity, and enough revisit authority gating to prevent tiny or offset revisits from shrinking or moving identity.
- Hardware-only confidence requires a later Pi 5 web/controller diagnostic-bundle run; no hardware validation is required during this specification pass.
