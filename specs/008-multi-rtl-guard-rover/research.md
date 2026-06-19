# Research: Profile-Governed Signal Identity Span and Revisit Authority

## Decision 1: Use an internal SignalSpanPolicy derived from profiles and args

**Decision**: Add optional profile fields and derive a small internal `SignalSpanPolicy` in detection/persistence code.

**Rationale**: The current system already has scanner profile plumbing, controller pass-through, effective-parameter export, match/display span shaping, and width-decision diagnostics. A derived policy object lets the implementation reuse those paths instead of adding a schema rewrite or a separate profile engine.

**Alternatives considered**:

- Hard-code FM Broadcast behavior: rejected because FM Broadcast is only the live canary.
- Add database columns for each width concept: rejected for this slice because the problem is runtime policy enforcement and diagnostics, not durable schema.
- Rewrite the detector to estimate true occupied bandwidth: rejected because RTL-SDR PSD fragments are valid raw evidence but not always true occupied bandwidth.

## Decision 2: Keep raw detector/revisit fragments policy-free

**Decision**: Preserve `detect_segments()` output as raw fragment evidence and apply policy only when deriving identity/match, persisted/card, and display spans.

**Rationale**: The raw detector is allowed to emit tiny thresholded fragments. Widening raw clusters too early risks merging close signals and hiding useful narrowband detections.

**Alternatives considered**:

- Increase detector thresholds or minimum bins globally: rejected because it suppresses evidence and does not solve semantic confusion.
- Inflate every segment width before clustering: rejected because it increases over-merge risk.

## Decision 3: Default identity and persist floors from match policy

**Decision**: If `min_identity_bandwidth_hz` or `min_persist_bandwidth_hz` is unset, default each to `min_match_bandwidth_hz`; if no match floor exists, preserve existing narrow discovery behavior.

**Rationale**: Current profiles already use `min_match_bandwidth_hz` to express persistence/matching intent. Reusing it as the default identity/persist floor is the smallest compatible step.

**Alternatives considered**:

- Default persist floor to display floor: rejected because display width is operator presentation, not persistence identity.
- Default all profiles to a broad floor: rejected because narrowband and discovery profiles must remain narrow.

## Decision 4: Revisit confirmation is separate from revisit identity authority

**Decision**: Revisit evidence can confirm presence even when it is too narrow or too offset to update identity center or persisted width.

**Rationale**: The observed bug is not that revisit detects fragments; it is that narrow or offset revisit fragments can become strong identity evidence. Separating confirmation from identity update keeps useful revisit evidence while preventing unstable cards.

**Alternatives considered**:

- Reject narrow revisit detections entirely: rejected because they can still prove presence.
- Allow all revisit matches to update center/width: rejected because that is the current failure mode.

## Decision 5: Replace profile-name checks with policy flags

**Decision**: Move existing center smoothing behavior behind a profile-derived policy field such as `center_smoothing_enabled` or `center_stability_mode`.

**Rationale**: `BaselinePersistence._center_smoothing_enabled()` currently checks `profile == "fm_broadcast"`. That is exactly the kind of profile-specific branch the update should remove from generic persistence code.

**Alternatives considered**:

- Leave the FM-specific check in place: rejected because this plan is profile-neutral.
- Disable smoothing globally: rejected because broad continuous profiles still need bounded center stability.

## Decision 6: Keep `/api/jobs` stable

**Decision**: New policy fields flow through existing job `params` and scanner CLI args. Do not change the `/api/jobs` top-level payload.

**Rationale**: Existing controller/web tests and operator workflows depend on the stable payload shape `{device_key, label, baseline_id, params}`.

**Alternatives considered**:

- Add new top-level API fields: rejected as unnecessary contract churn.
- Make policy scanner-only: rejected because operator-facing GUI/controller runs must be able to apply and audit the policy.

## Decision 7: Event detection and baseline learning remain configurable modes

**Decision**: Do not globally tighten `persistence_min_hits`, `persistence_min_windows`, or `persistence_min_sweep_loops`. Instead, document and later formalize mode/profile distinction for event/first-light, baseline-learning, guard, and characterization behavior.

**Rationale**: The live canary can use 1/1/1 for first-light/event behavior, while baseline-learning can require recurrence. A global stricter persistence change would regress discovery and guard workflows.

**Alternatives considered**:

- Force stricter persistence globally: rejected because it changes workflow semantics beyond this span policy fix.
- Leave modes undocumented: rejected because it would keep event and baseline-learning behavior ambiguous.
