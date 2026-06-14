# Contract: Cross-Sweep Persistence

This contract defines how SDRwatch should promote a stable signal that appears once per complete sweep loop without weakening FM defaults or merging nearby signals.

## Scope

Applies to scanner-side candidate accumulation and promotion for a running job. It does not create demodulation, content decoding, or a new operator-facing workflow.

## Observation Semantics

A cross-sweep observation is a compatible signal observation associated with one complete sweep loop.

Required fields:

```json
{
  "event": "cross_sweep_observation",
  "baseline_id": 1,
  "sweep_loop_id": 3,
  "window_idx": 7,
  "candidate_id": "cs-100100000",
  "center_hz": 100100000,
  "match_low_hz": 100060000,
  "match_high_hz": 100140000,
  "measured_bandwidth_hz": 80000,
  "source_pass": "coarse"
}
```

Rules:

- A candidate may count at most one observation for a given `sweep_loop_id` toward a multi-loop threshold.
- Observations from the same sweep loop may update candidate evidence, but they must not pretend to be multiple complete-loop observations.
- Observation records should carry enough raw, measured, match, display, and context references to preserve 006 characterization boundaries.

## Candidate Compatibility

Candidates match when active profile rules consider them compatible.

Required compatibility inputs:

- center frequency difference
- match span overlap or bounded center closeness
- measured or match bandwidth compatibility
- active profile matching values such as `center_match_hz`, `min_match_bandwidth_hz`, `max_persist_width_hz`, and width ratio limits where supported

Rules:

- Display span must not be used as the primary matching span.
- Bandplan/profile context must not be used as proof that two detections are the same signal.
- Incompatible nearby candidates must remain separate.

## Promotion Semantics

Promotion occurs only when configured persistence gates are satisfied by cross-sweep evidence.

Required decision record:

```json
{
  "event": "persistence_decision",
  "action": "cross_sweep_promote",
  "baseline_id": 1,
  "candidate_id": "cs-100100000",
  "observation_count": 3,
  "observation_loop_count": 3,
  "required_loop_count": 3,
  "center_hz": 100100000,
  "match_width_hz": 80000
}
```

Rules:

- Promotion must not occur before the configured number of distinct sweep-loop observations.
- A single one-off observation must not create a persistent card when strict settings require multiple observations.
- The inserted or updated persistent detection must satisfy `f_low_hz <= f_center_hz <= f_high_hz`.

## Rejection And No-Match Semantics

Rejected or unmatched candidate decisions must be structured.

Recommended records:

```json
{
  "event": "persistence_decision",
  "action": "cross_sweep_no_match",
  "baseline_id": 1,
  "center_hz": 100350000,
  "reason": "center outside compatibility"
}
```

```json
{
  "event": "persistence_decision",
  "action": "cross_sweep_reject",
  "candidate_id": "cs-100350000",
  "reason": "expired before required loop count"
}
```

## Acceptance Checks

- A signal observed once per complete sweep loop promotes after the configured number of loop observations.
- Same-loop repeated observations do not satisfy a multi-loop threshold.
- Nearby FM-like signals remain separate under profile rules.
- Existing FM characterization fields remain distinct in diagnostics.
- Existing FM bounded width and stable center tests continue to pass.
