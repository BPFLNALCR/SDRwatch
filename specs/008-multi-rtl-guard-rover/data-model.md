# Data Model: Profile-Governed Signal Identity Span and Revisit Authority

## Entity: SignalSpanPolicy

Derived runtime policy object built from scanner args after profile application.

### Fields

- `profile_name`: active requested/applied profile context when available.
- `min_identity_bandwidth_hz`: minimum width for identity/match semantics.
- `min_persist_bandwidth_hz`: minimum width for persisted/card span semantics.
- `max_persist_bandwidth_hz`: maximum persisted/card width, derived from existing max width aliases.
- `min_match_bandwidth_hz`: existing match floor retained for compatibility.
- `min_display_bandwidth_hz`: existing display floor retained for operator presentation.
- `allow_revisit_to_shrink_identity`: whether revisit can reduce identity width.
- `allow_revisit_to_move_center`: whether revisit can move stable center within policy gates.
- `min_revisit_bandwidth_for_identity_update_hz`: revisit bandwidth floor for identity updates.
- `max_revisit_center_delta_for_identity_update_hz`: center delta gate for revisit identity updates.
- `fragmented_revisit_policy`: policy for ambiguous revisit fragments.
- `raw_fragment_interpretation`: label for raw detector evidence.
- `center_smoothing_enabled` or `center_stability_mode`: profile-governed center smoothing behavior.

### Validation Rules

- Negative bandwidth or delta values are invalid and should resolve to unset/disabled.
- `min_persist_bandwidth_hz` defaults to `min_match_bandwidth_hz` when unset.
- `min_identity_bandwidth_hz` defaults to `min_match_bandwidth_hz` when unset.
- `max_persist_bandwidth_hz` must not be below the effective persist floor; if it is, diagnostics should report invalid policy and use the safer max of the two or skip the max cap.
- Discovery/no-profile behavior must preserve current narrow widths when no policy floor is set.

## Entity: RawFragmentEvidence

Raw detector or revisit segment evidence.

### Fields

- `raw_fragment_center_hz`
- `raw_fragment_low_hz`
- `raw_fragment_high_hz`
- `raw_fragment_bandwidth_hz`
- `peak_db`
- `noise_db`
- `snr_db`
- `source_pass`: `coarse` or `revisit`

### Validation Rules

- Raw fragment bandwidth may be tiny.
- Raw fragment evidence must not be overwritten by policy-shaped identity, persisted, or display widths.

## Entity: IdentityMatchSpan

Signal identity span used for matching and characterization identity.

### Fields

- `identity_match_center_hz`
- `identity_match_low_hz`
- `identity_match_high_hz`
- `identity_match_bandwidth_hz`
- `width_floor_applied_hz`
- `max_width_clamped_hz`
- `baseline_clipped`

### Validation Rules

- Width should not fall below `min_identity_bandwidth_hz` except baseline/scan-edge clipping.
- Identity span must remain separate from raw fragment and display span.
- Identity span must not be used to widen live cluster extents before close-signal matching decisions.

## Entity: PersistedCardSpan

Stored card span in `baseline_detections.f_low_hz/f_high_hz/f_center_hz`.

### Fields

- `persisted_card_center_hz`
- `persisted_card_low_hz`
- `persisted_card_high_hz`
- `persisted_card_bandwidth_hz`
- `persist_width_floor_applied_hz`
- `max_persist_width_clamped_hz`
- `baseline_clipped`

### Validation Rules

- `f_low_hz <= f_center_hz <= f_high_hz`.
- Width should not fall below `min_persist_bandwidth_hz` except baseline/scan-edge clipping.
- Width should not exceed `max_persist_bandwidth_hz` when configured.
- Store schema remains unchanged for this slice.

## Entity: DisplaySpan

Operator-facing span used in emitted records and UI summaries.

### Fields

- `display_center_hz`
- `display_low_hz`
- `display_high_hz`
- `display_bandwidth_hz`
- `display_width_floor_applied_hz`

### Validation Rules

- Display width is governed by `min_display_bandwidth_hz`.
- Display width must not be treated as measured occupied bandwidth.
- Narrowband and discovery profiles must be able to keep display widths narrow.

## Entity: RevisitAuthorityDecision

Decision record describing what a revisit confirmation is allowed to update.

### Fields

- `revisit_authority`: `identity_update`, `confirmation_only`, `rejected_for_center_delta`, `rejected_for_bandwidth_floor`, or `fragmented_or_ambiguous`.
- `identity_update_allowed`
- `confirmation_recorded`
- `revisit_center_delta_hz`
- `revisit_bandwidth_hz`
- `max_revisit_center_delta_for_identity_update_hz`
- `min_revisit_bandwidth_for_identity_update_hz`
- `revisit_center_policy_result`
- `revisit_bandwidth_policy_result`
- `reason`

### Validation Rules

- Confirmation can be recorded even when identity update is not allowed.
- Revisit cannot move identity center when the center delta gate fails.
- Revisit cannot shrink or update identity/persisted width when the bandwidth floor gate fails.

## Entity: BandwidthInterpretation

Diagnostic explanation for how bandwidth fields should be read.

### Fields

- `raw_fragment_interpretation`
- `measured_bandwidth_interpretation`
- `identity_bandwidth_interpretation`
- `persisted_bandwidth_interpretation`
- `display_bandwidth_interpretation`

### Validation Rules

- Raw and measured values may be narrow and low-confidence.
- Identity and persisted values may be policy-shaped.
- Display values are presentation-oriented.

## State Transitions

1. Raw detector segment becomes `RawFragmentEvidence`.
2. Clustered evidence becomes `IdentityMatchSpan` through `SignalSpanPolicy`.
3. Persistence insert/update stores `PersistedCardSpan`.
4. Display emission creates `DisplaySpan`.
5. Revisit segment creates `RawFragmentEvidence`.
6. Revisit policy creates `RevisitAuthorityDecision`.
7. Revisit either records confirmation-only evidence or updates identity/persisted span through policy gates.
