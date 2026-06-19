# Contract: Signal Span Policy Diagnostics and Effective Parameters

## Scope

This contract covers additive scanner diagnostics and effective-parameter fields for profile-governed signal identity span, persisted card span, display span, and revisit authority.

It does not change `/api/jobs`, SQLite schema, or existing diagnostic field names.

## Effective Parameters

`effective_parameters` records MUST continue to include current profile audit fields:

- `requested_profile`
- `applied_profile`
- `profile_applied`
- `profile_skip_reason`
- `profile_application_source`
- `profile_audit_complete`
- `profile_defaults`
- `operator_overrides`
- `fallback_defaults`

They MUST also expose the derived signal span policy either under `signal_span_policy` or as additive keys in `span_controls`:

```json
{
  "signal_span_policy": {
    "min_identity_bandwidth_hz": 80000.0,
    "min_persist_bandwidth_hz": 80000.0,
    "max_persist_bandwidth_hz": 270000.0,
    "min_match_bandwidth_hz": 80000.0,
    "min_display_bandwidth_hz": 200000.0,
    "allow_revisit_to_shrink_identity": false,
    "allow_revisit_to_move_center": true,
    "min_revisit_bandwidth_for_identity_update_hz": 80000.0,
    "max_revisit_center_delta_for_identity_update_hz": 60000.0,
    "fragmented_revisit_policy": "confirmation_only",
    "raw_fragment_interpretation": "threshold_fragment",
    "center_smoothing_enabled": true
  }
}
```

Unset fields MAY be `null` when a profile intentionally preserves raw discovery behavior.

## Characterization Records

Existing fields MUST remain:

- `raw_bandwidth_hz`
- `measured_bandwidth_hz`
- `match_bandwidth_hz`
- `display_bandwidth_hz`
- `raw_segment`
- `measured_span`
- `match_span`
- `display_span`
- `source_pass`
- `stable_center_hz`
- `center_delta_hz`

Additive aliases SHOULD be emitted:

- `raw_fragment_bandwidth_hz`
- `raw_fragment_center_hz`
- `measured_occupied_bandwidth_hz`
- `identity_match_bandwidth_hz`
- `persisted_card_bandwidth_hz`
- `bandwidth_interpretation`
- `width_floor_applied_hz`
- `persist_width_floor_applied_hz`

## Width Decision Events

`width_decision` events SHOULD include:

- `stage`
- `input_width_hz`
- `padded_width_hz`
- `output_width_hz`
- `min_width_hz`
- `min_identity_bandwidth_hz`
- `min_persist_bandwidth_hz`
- `max_width_hz`
- `was_floored`
- `was_clamped`
- `baseline_clipped`
- `width_floor_applied_hz`
- `persist_width_floor_applied_hz`

Existing consumers MAY continue reading `min_width_hz`, `output_width_hz`, `was_floored`, and `was_clamped`.

## Revisit Authority Events

Every revisit confirmation SHOULD emit or attach a decision with:

- `revisit_authority`
- `identity_update_allowed`
- `confirmation_recorded`
- `revisit_center_delta_hz`
- `revisit_bandwidth_hz`
- `min_revisit_bandwidth_for_identity_update_hz`
- `max_revisit_center_delta_for_identity_update_hz`
- `revisit_bandwidth_policy_result`
- `revisit_center_policy_result`
- `reason`

Allowed `revisit_authority` values:

- `identity_update`
- `confirmation_only`
- `rejected_for_center_delta`
- `rejected_for_bandwidth_floor`
- `fragmented_or_ambiguous`

## Compatibility Requirements

- Existing diagnostic bundles MUST remain readable when new fields are absent.
- New fields MUST be additive.
- `/api/jobs` MUST remain `{device_key, label, baseline_id, params}`.
- Scanner CLI checks remain backend smoke tests only; operator validation is through web UI and controller lifecycle.
