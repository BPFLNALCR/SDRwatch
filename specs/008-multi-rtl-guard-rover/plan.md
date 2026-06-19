# Implementation Plan: Profile-Governed Signal Identity Span and Revisit Authority

**Branch**: `008-multi-rtl-guard-rover` | **Date**: 2026-06-19 | **Spec**: [spec.md](./spec.md)

**Input**: Focused technical planning request for a small, generic update that makes signal identity span, persisted card span, operator display span, and revisit authority obey profile-defined policy.

**Spec Alignment Note**: `spec.md` and `tasks.md` in this directory have been reconciled around the current focused span-policy request. Do not resume the earlier hardware-aware multi-RTL role-run tasking from this feature directory.

## A. Executive Summary

The root design problem is width semantics leaking across layers. SDRwatch correctly captures tiny thresholded FFT fragments as raw evidence, but parts of the current detection, characterization, persistence, and revisit pipeline can still let that tiny fragment width act like the signal's identity width or persisted card width. That creates narrow duplicate cards, split tracks, unstable centers/spans, and misleading "occupied bandwidth" interpretation.

The fix should be profile-policy enforcement, not FM Broadcast tuning. FM Broadcast is only the live RF canary. The update must stay generic and profile-driven so broad continuous signals, narrowband watchlist signals, unknown discovery, and guard/event workflows can each express different floors and revisit authority without hard-coded 88-108 MHz behavior or `profile == fm_broadcast` branches in generic persistence logic.

The lowest-risk approach is to formalize a derived internal `SignalSpanPolicy` from existing and new optional profile fields. Keep raw detector output unchanged. Apply policy at the points where raw evidence becomes identity/match span, persisted/card span, display span, and revisit authority. Preserve backward-compatible diagnostic fields while adding clearer names and decision records.

This planning pass does not implement code. It intentionally excludes FM-specific hard-coding, new multi-RTL role assignment work, Airspy/HackRF/Soapy runtime support, UI redesign, signal fusion schema work, Rust DSP rewrites, continuous IQ capture, and broad detector threshold retuning based only on the FM canary.

## Constitution Check

- **I. Raspberry Pi First Reliability**: PASS. The plan keeps the scanner local, avoids raw IQ capture, and uses bounded diagnostics plus no-hardware and Pi 5 canary validation.
- **II. Minimal Local Stack**: PASS. The plan stays in Python, Flask/controller pass-through, SQLite persistence, and JSONL diagnostics. No new framework or service is introduced.
- **III. Stable Interfaces and Clean Layering**: PASS. The web UI/controller job lifecycle remains the operator workflow. The scanner CLI remains an internal backend surface. `/api/jobs` stays compatible.
- **IV. Adapter-Based Hardware and Honest RF Claims**: PASS. The plan does not expand runnable hardware. It separates raw evidence, measured values, profile context, and display/persisted policy decisions.
- **V. Migration-Safe, Verifiable Change**: PASS. No database migration is required. New fields are optional, derived, and diagnostics-first. Existing tests remain in scope.
- **Operator Acceptance Gate**: PASS. Hardware acceptance is through browser -> controller -> scanner diagnostic bundle. CLI checks are backend smoke only.

## B. Code-Path Map

| Width/center concept | Current source/file/function | Current meaning | Problem | Planned policy owner | Planned fix |
| --- | --- | --- | --- | --- | --- |
| Raw detector segment width/center | `sdrwatch/dsp/detection.py::detect_segments`; `Segment` in `sdrwatch/detection/types.py` | Contiguous or estimated thresholded PSD fragment with center from midpoint, peak, or centroid mode | Can be hundreds of Hz to a few kHz and is valid raw evidence, but too narrow to act as identity/card width for broad profiles | Raw detector remains policy-free | Preserve raw output. Rename/alias diagnostics as `raw_fragment_*` so small width is not interpreted as card identity. |
| Cluster extent and center | `sdrwatch/detection/engine.py::_record_hit`, `_cluster_center_hz`, `_segments_overlap`, `_cluster_gate_status` | Aggregates raw segments within a sweep window sequence, with power-weighted center and raw cluster low/high | Cluster low/high can still be fragment-shaped or split nearby evidence before policy shaping | `DetectionEngine` plus derived `SignalSpanPolicy` | Keep cluster extents tight for neighbor separation. Derive identity/match span from policy after clustering, not by widening the live cluster. |
| Measured characterization occupied bandwidth | `DetectionEngine._emit_detection`, `DetectionEngine.apply_revisit_confirmation`, `CharacterizationEvidence` | Existing `measured_span` and `measured_bandwidth_hz` record raw/coarse or revisit measured width | The field can read like true occupied bandwidth even when it is only a thresholded fragment | Diagnostics contract owned by detection/util layer | Preserve old fields. Add interpretation fields and clearer aliases: `measured_occupied_bandwidth_hz` only with `bandwidth_interpretation`, and `raw_fragment_bandwidth_hz` for raw evidence. |
| Match/identity span | `DetectionEngine._shape_span`, `_shape_match_span` | Adds match padding, applies `min_match_bandwidth_hz`, applies `max_detection_width_hz`, clips to baseline | This is the closest current identity span, but it is not explicitly named as identity policy and does not expose a separate identity floor | `SignalSpanPolicy` derived from profile/effective args | Add internal `min_identity_bandwidth_hz`, defaulting to `min_match_bandwidth_hz`. Continue to use `_shape_match_span` but log `identity_match_bandwidth_hz` and policy floor application. |
| Display span | `DetectionEngine._shape_display_span`; emitted `display_span` | Operator-facing span shaped by display padding and `min_display_bandwidth_hz` | Generally correct, but summaries can still be confused with measured bandwidth | `SignalSpanPolicy.display` | Keep display separate. Add diagnostics that explicitly state display width was policy-shaped. Never use display span as measured occupied bandwidth. |
| Persistence width EMA input | `sdrwatch/baseline/persistence.py::_upsert_detection`, `_blend_width_ema` | Blends previous stored width with current cluster width and clamps to `min_detection_width_hz`/`max_detection_width_hz` | The EMA floor is currently detector-bin/min-width based, not profile persist policy based, so persisted rows can converge below profile intent | `BaselinePersistence` using `SignalSpanPolicy.persist` | Replace/augment EMA floor with `min_persist_bandwidth_hz`, defaulting to `min_match_bandwidth_hz`. Keep max width cap. Log `persist_width_floor_applied_hz`. |
| Final persisted/card span | `BaselinePersistence._upsert_detection`, `apply_revisit_confirmation`, `_enforce_persisted_span_invariant`; `Store.insert_baseline_detection`, `Store.update_baseline_detection` | SQLite `baseline_detections.f_low_hz/f_high_hz/f_center_hz` used by cards and baseline persistence | Stored span can shrink through EMA or revisit update and become narrower than active profile intent | `BaselinePersistence` | Add a persist-span clamp before store insert/update and after hysteresis, with documented scan-edge clipping. Store rows remain current schema. |
| Revisit target and selected segment | `sdrwatch/sweep/sweeper.py::_run_revisit_pass`, `_select_revisit_segment` | Finds a matching revisit segment around queued target and calls `apply_revisit_confirmation` | Any tiny or offset sub-peak can confirm and then update center/width | `SignalSpanPolicy.revisit` in `BaselinePersistence`/`DetectionEngine` | Revisit can confirm presence separately from identity updates. Gate center and width authority before changing stable center or persisted span. |
| Revisit center smoothing | `BaselinePersistence._center_smoothing_enabled` | Current generic persistence code enables smoothing only when `profile == "fm_broadcast"` | This is the existing profile-specific branch that should become policy configuration | `SignalSpanPolicy.center` | Replace with a policy flag such as `center_smoothing_enabled` or `center_stability_mode`, populated by profile defaults. |
| Effective parameter export | `sdrwatch/util/detection_diagnostics.py::build_effective_parameter_manifest`; `Sweeper._sweep_params` | Emits applied profile, persistence, revisit, and `span_controls` | New policy floors/gates would be invisible without manifest additions | Diagnostics manifest | Add policy fields under `span_controls` or `signal_span_policy`, preserving current keys. |
| Profile definitions and pass-through | `sdrwatch/io/profiles.py`, `sdrwatch/cli.py::_apply_scan_profile`, `sdrwatch-control.py::_build_cmd` | Profiles set scanner args; controller maps `/api/jobs` params to scanner CLI flags | Current profile has match/display floors but no explicit identity/persist/revisit authority policy | Profile definitions plus internal policy object | Add optional profile/CLI fields. Controller passes them through inside existing `params` without changing `/api/jobs`. |

## C. Proposed Signal Identity/Span Policy

### Policy Owner

Use a small internal policy object rather than a broad schema rewrite:

- Add optional fields to `ScanProfile` in `sdrwatch/io/profiles.py`.
- Add scanner CLI args only for fields that need operator/controller override.
- Derive an internal `SignalSpanPolicy` from `args` inside detection/persistence code.
- Export derived values through effective parameters and diagnostic decisions.
- Do not add SQLite columns for this slice.

This gives profiles a durable contract while keeping enforcement close to the code that already shapes spans and updates persistence.

### Candidate Fields

Existing fields to keep:

- `min_match_bandwidth_hz`
- `match_bandwidth_pad_hz`
- `min_display_bandwidth_hz`
- `display_bandwidth_pad_hz`
- `center_match_hz`
- `max_detection_width_hz`
- `max_persist_width_hz` and `max_card_width_hz` as compatibility aliases
- `revisit_span_limit_hz`

New optional fields:

- `min_identity_bandwidth_hz`
- `min_persist_bandwidth_hz`
- `max_persist_bandwidth_hz` as internal resolved value, derived from existing max aliases unless explicitly added later
- `allow_revisit_to_shrink_identity`
- `allow_revisit_to_move_center`
- `min_revisit_bandwidth_for_identity_update_hz`
- `max_revisit_center_delta_for_identity_update_hz`
- `fragmented_revisit_policy`
- `raw_fragment_interpretation`
- `center_smoothing_enabled` or `center_stability_mode`

### Defaults

- `min_identity_bandwidth_hz`: if unset, use `min_match_bandwidth_hz`; if that is unset, use 0 or bin width to preserve discovery behavior.
- `min_persist_bandwidth_hz`: if unset, use `min_match_bandwidth_hz`; if that is unset, use `min_identity_bandwidth_hz`; if both are unset, preserve existing `min_detection_width_hz` behavior.
- `max_persist_bandwidth_hz`: use `max_persist_width_hz`, then `max_card_width_hz`, then `max_detection_width_hz`, then unlimited.
- `min_display_bandwidth_hz`: keep existing behavior.
- `min_revisit_bandwidth_for_identity_update_hz`: if unset and `min_identity_bandwidth_hz > 0`, use `min_identity_bandwidth_hz`; otherwise no bandwidth gate.
- `max_revisit_center_delta_for_identity_update_hz`: if unset and `center_match_hz` is set, use `center_match_hz`; otherwise no additional delta gate.
- `allow_revisit_to_shrink_identity`: default false when an identity floor exists; otherwise preserve current behavior.
- `allow_revisit_to_move_center`: default true only within the center delta gate.
- `fragmented_revisit_policy`: default `confirmation_only`.
- `raw_fragment_interpretation`: default `threshold_fragment`.
- `center_smoothing_enabled`: default false, with broad continuous profiles enabling it via profile configuration.

### Span Formulas

The implementation should keep raw evidence intact and apply policy only at semantic transitions:

```text
raw_fragment_bandwidth_hz = detector/revisit segment width

measured_occupied_bandwidth_hz = raw/coarse/revisit measurement, with explicit interpretation

identity_match_bandwidth_hz =
  clamp(max(raw_or_measured_width + match_padding, min_identity_bandwidth_hz, min_match_bandwidth_hz),
        lower=min_identity_bandwidth_hz,
        upper=max_persist_bandwidth_hz when configured)

persisted_card_bandwidth_hz =
  clamp(width_ema_input_or_existing_card_width,
        lower=min_persist_bandwidth_hz,
        upper=max_persist_bandwidth_hz)

display_bandwidth_hz =
  max(display_shaped_width, min_display_bandwidth_hz)
```

The exact enforcement should reuse existing `_shape_span()` and `_blend_width_ema()` paths where possible. Do not widen live cluster extents used for raw neighbor separation.

### Profile Examples

| Profile family | Example policy | Expected behavior |
| --- | --- | --- |
| Broad continuous canary | FM Broadcast canary may use `min_identity_bandwidth_hz=80000`, `min_persist_bandwidth_hz=80000`, `min_display_bandwidth_hz=200000`, `max_persist_bandwidth_hz=270000`, `min_revisit_bandwidth_for_identity_update_hz=80000`, `max_revisit_center_delta_for_identity_update_hz=60000`, `center_smoothing_enabled=true` | Raw/revisit fragments can remain tiny, but identity and persisted cards do not shrink below policy floor except scan-edge clipping. Display remains broad. |
| Narrowband voice/watchlist | Example values might be `min_identity_bandwidth_hz=6000-12500`, `min_persist_bandwidth_hz=6000-25000`, `min_display_bandwidth_hz=12500-25000`, low revisit bandwidth floor, tighter center delta | Narrowband cards stay narrow and are not forced into FM-like widths. |
| Unknown discovery | `min_identity_bandwidth_hz` unset, `min_persist_bandwidth_hz` unset, `min_display_bandwidth_hz` unset or low, raw interpretation `raw_uncertain` | Preserve raw measurements and low-confidence discovery. Do not inherit 200 kHz display behavior. |
| Guard/event mode | Fast event mode can use low persistence gates and emit candidate events; baseline-learning mode can require recurrence | Guard can report fast candidates without immediately declaring stable baseline cards. Baseline learning can use stricter recurrence without globally changing first-light behavior. |

### Effective Parameters and Backward Compatibility

Add a `signal_span_policy` section or extend `span_controls` in `effective_parameters` with:

- `min_identity_bandwidth_hz`
- `min_persist_bandwidth_hz`
- `max_persist_bandwidth_hz`
- `min_revisit_bandwidth_for_identity_update_hz`
- `max_revisit_center_delta_for_identity_update_hz`
- `allow_revisit_to_shrink_identity`
- `allow_revisit_to_move_center`
- `fragmented_revisit_policy`
- `raw_fragment_interpretation`
- `center_smoothing_enabled`

Keep all existing fields and aliases. Existing diagnostic readers should still understand `raw_bandwidth_hz`, `measured_bandwidth_hz`, `match_bandwidth_hz`, and `display_bandwidth_hz`.

## D. Revisit Authority Plan

Revisit needs two separate outcomes:

- **Confirmation-only**: the revisit found energy consistent enough to mark a detection present, clear missing state, increase confirmation evidence, and record diagnostics, but it cannot move identity center or shrink/expand persisted width.
- **Identity update**: the revisit may update stable center, identity/match span, and persisted/card width because it passes profile gates.

### Gates

For every revisit confirmation:

1. Compute `revisit_center_delta_hz = abs(revisit_center_hz - current_stable_or_persisted_center_hz)`.
2. Compute `revisit_bandwidth_hz` from raw revisit segment.
3. If `max_revisit_center_delta_for_identity_update_hz` is configured and delta exceeds it, mark identity update rejected for center delta.
4. If `min_revisit_bandwidth_for_identity_update_hz` is configured and revisit width is below it, mark identity update rejected for bandwidth floor.
5. If multiple revisit segments are fragmented or ambiguous, prefer `confirmation_only` unless policy explicitly allows fragmented identity update.
6. If `allow_revisit_to_shrink_identity` is false, do not let a narrower revisit segment reduce identity or persisted width below the current policy floor.
7. If `allow_revisit_to_move_center` is false or the delta gate fails, preserve current stable center.

### Diagnostics

Emit a revisit authority decision on every revisit confirmation:

- `revisit_authority`: `identity_update`, `confirmation_only`, `rejected_for_center_delta`, `rejected_for_bandwidth_floor`, `fragmented_or_ambiguous`
- `identity_update_allowed`
- `revisit_center_delta_hz`
- `revisit_bandwidth_hz`
- `min_revisit_bandwidth_for_identity_update_hz`
- `max_revisit_center_delta_for_identity_update_hz`
- `revisit_bandwidth_policy_result`
- `revisit_center_policy_result`
- `confirmation_recorded`

Existing `revisit_apply` and `characterization_record` events should remain, with additive fields.

## E. Persistence/Card Span Plan

### Persist Width Floor

`BaselinePersistence._blend_width_ema()` should use a policy persist floor rather than only `min_detection_width_hz`:

```text
effective_min_persist_width =
  max(bin_hz, min_detection_width_hz, min_persist_bandwidth_hz when configured)
```

The EMA measurement should be floored to this value before blending, and the final blended result should be floored again after blending. Diagnostics should distinguish:

- `input_width_hz`
- `measurement_width_hz`
- `min_width_hz`
- `min_persist_bandwidth_hz`
- `persist_width_floor_applied_hz`
- `output_width_hz`

### Store/Update Clamp

Before `Store.insert_baseline_detection()` and `Store.update_baseline_detection()`, apply a persist-card span clamp centered on the current stable center:

```text
width = clamp(width, min_persist_bandwidth_hz, max_persist_bandwidth_hz)
low = center - width / 2
high = center + width / 2
clip to baseline start/stop
if edge clipped, record baseline_clipped=true
```

`_enforce_persisted_span_invariant()` should continue to ensure `f_low_hz <= f_center_hz <= f_high_hz`, but it should not be the only place policy width is enforced.

### Max Width Caps

Keep existing max width semantics:

- `max_detection_width_hz` remains the scanner's current broad cap.
- `max_persist_width_hz` and `max_card_width_hz` remain compatibility aliases.
- Internal `max_persist_bandwidth_hz` resolves from those fields.
- Existing width ratio rejection remains useful to prevent one wide observation from absorbing unrelated signals.

### Scan-Edge Clipping

Persisted/card width may be narrower than the policy floor only when the policy-centered span is clipped by the active baseline or scan boundary. Diagnostics must say:

- `baseline_clipped=true`
- `requested_persist_bandwidth_hz`
- `persisted_card_bandwidth_hz`
- `clip_reason=scan_edge`

### Avoid Over-Merging Close Signals

Do not widen the live cluster extent or raw detector segments just because a floor exists. Use the floor only for identity/persist/display spans after candidate formation. Keep:

- `cluster_merge_hz`
- `center_match_hz`
- width ratio rejection
- max persist width cap
- existing close-signal regression tests

Add tests where two close but distinct signals remain separate even with a broad profile floor.

## F. Test Plan

Add or update no-hardware tests before implementation.

### New focused tests

Create `tests/test_signal_span_policy.py` if that keeps policy coverage clearer:

- `test_policy_defaults_identity_and_persist_floor_from_min_match_bandwidth`
- `test_tiny_raw_segment_does_not_shrink_identity_below_policy_floor`
- `test_persisted_card_width_cannot_shrink_below_min_persist_bandwidth`
- `test_display_span_still_uses_min_display_bandwidth_independently`
- `test_raw_fragment_bandwidth_remains_available_and_tiny`
- `test_narrowband_profile_can_use_small_identity_and_display_floors`
- `test_unknown_discovery_profile_does_not_inherit_broad_display_floor`

### Existing persistence and characterization tests

Update or extend `tests/test_fm_characterization_persistence.py`:

- `test_tiny_fft_fragment_does_not_become_fake_measured_fm_bandwidth` should continue proving raw/measured can be tiny while identity/display are policy-shaped.
- Add `test_tiny_revisit_is_confirmation_only_when_below_identity_floor`.
- Add `test_large_delta_revisit_is_confirmation_only_when_center_gate_fails`.
- Add `test_revisit_authority_diagnostics_explain_confirmation_only`.

Update `tests/test_extent_hysteresis.py`:

- Rename or add `test_width_ema_applies_min_persist_bandwidth_floor`.
- Add live-path assertions that stored rows remain above persist floor after `_upsert_detection()` and `apply_revisit_confirmation()`.
- Keep existing invariant tests for center within low/high.

Update `tests/test_fm_persistence_stability.py`:

- Keep close-signal separation tests.
- Add `test_policy_floor_does_not_merge_nearby_close_signals`.

Update `tests/test_non_fm_width_scope.py`:

- Keep `test_narrow_non_fm_signal_remains_narrow_without_fm_profile`.
- Add a narrowband profile/policy fixture proving small floors stay small.
- Add an unknown/discovery fixture proving no FM-like display/persist floor is applied.

### Profile, CLI, controller, and diagnostics tests

Update `tests/test_fm_validation_profile.py`:

- Add serialization checks for new policy fields.
- Add CLI profile application checks for derived policy fields.
- Add override preservation checks.

Update `tests/test_effective_parameter_manifest.py`:

- Add `test_effective_parameter_manifest_records_signal_span_policy`.
- Ensure `max_persist_width_hz`/`max_card_width_hz` aliases remain present.

Update `tests/test_control_fm_validation.py`:

- Add controller command pass-through for new numeric/boolean/string policy params.
- Ensure `/api/jobs` payload remains `{device_key, label, baseline_id, params}`.

Update `tests/test_fm_characterization_diagnostics.py` and `tests/test_web_diagnostics_bundle.py`:

- Add summary coverage for `raw_fragment_bandwidth_hz`, `identity_match_bandwidth_hz`, `persisted_card_bandwidth_hz`, `bandwidth_interpretation`, and `revisit_authority`.
- Preserve old summary fields.

### Existing regression suites to keep green

- `tests/test_cross_sweep_persistence.py`
- `tests/test_effective_parameter_manifest.py`
- `tests/test_device_telemetry.py`
- `tests/test_multi_rtl_inventory.py`
- `tests/test_multi_rtl_backend_gating.py`
- `tests/test_multi_rtl_guard.py`
- `tests/test_multi_rtl_telemetry.py`
- `tests/test_legacy_job_compatibility.py`
- Existing FM Validation and control-page tests

### Optional hardware acceptance

After implementation, run the same Pi 5 live canary through the web GUI/controller diagnostic flow:

- Use a normal single-device RTL scan with `rtlsdr_native`, `device_key=rtl:0`, and `profile=fm_broadcast`.
- Confirm requested/applied profile and effective parameters still agree.
- Expect fewer misleading narrow persisted cards, but do not require an exact FM station count.
- Confirm no ordinary persisted card falls below active persist floor except documented scan-edge clipping.
- Confirm raw/revisit fragment widths can remain small and are labeled as raw fragments.
- Confirm effective parameters still agree with scanner log and decision summary.

## G. Implementation Slices

### Slice A: Policy and Effective-Parameter Plumbing

- Extend `ScanProfile` with optional signal span policy fields.
- Add scanner CLI args for policy fields.
- Add controller pass-through mapping inside existing `params`.
- Build internal `SignalSpanPolicy` from args.
- Export derived policy through effective parameters.
- Tests: profile serialization, CLI application, controller pass-through, manifest contract.

### Slice B: Identity/Persist Width Floor Enforcement

- Add `min_identity_bandwidth_hz` use to match/identity span diagnostics.
- Add `min_persist_bandwidth_hz` to persistence EMA and final store/update clamp.
- Preserve max width caps and scan-edge clipping diagnostics.
- Tests: tiny raw segment identity floor, persisted card floor, display floor independence, close-signal non-merge.

### Slice C: Revisit Authority Gating

- Split revisit confirmation from identity update.
- Gate center movement and width update by policy.
- Replace `profile == fm_broadcast` center smoothing with policy.
- Emit revisit authority diagnostics.
- Tests: tiny revisit confirmation-only, large center delta confirmation-only/rejected for identity update, diagnostics decision fields.

### Slice D: Diagnostics Clarity

- Add clearer aliases while preserving old fields.
- Add `bandwidth_interpretation`, `width_floor_applied_hz`, `persist_width_floor_applied_hz`, `identity_update_allowed`, and revisit policy result fields.
- Update bundle summaries to surface bounded samples.
- Tests: diagnostics JSONL and bundle compatibility.

### Slice E: Regression and Pi 5 Canary Validation

- Run targeted no-hardware tests first.
- Run broader FM, cross-sweep, effective-parameter, and multi-RTL backend gating suites.
- Run GUI/controller Pi 5 canary when hardware is available.
- Leave hardware-only acceptance visibly open if not run in the current environment.

## H. Risk and Rollback

### Risks

- **Over-merging close signals**: A width floor used too early could make separate nearby signals overlap. Mitigation: keep raw cluster extents tight, apply floors only to identity/persist/display spans, preserve `center_match_hz`, `cluster_merge_hz`, width ratio rejection, and max width caps.
- **Hiding useful narrowband signals**: Bad defaults could force narrowband profiles into broad widths. Mitigation: defaults derive from profile fields; unprofiled discovery keeps current narrow behavior; narrowband fixtures prove small floors remain possible.
- **Breaking existing FM/cross-sweep behavior**: Persistence and revisit changes touch shared paths. Mitigation: implement in small slices with existing FM characterization, persistence, cross-sweep, and diagnostics tests green after each slice.
- **Changing operator contract accidentally**: New policy fields could leak into a new API shape. Mitigation: keep `/api/jobs` unchanged and pass new values through existing `params`.
- **Misleading diagnostics during transition**: New names might imply old fields disappeared. Mitigation: additive aliases only, old fields retained.

### Rollback Strategy

- All new profile-policy fields are optional.
- If a slice regresses behavior, set new fields to unset/0/compat defaults and the system falls back to current match/display/min-detection behavior.
- Revisit authority gates can be disabled by leaving bandwidth and center delta gates unset for a profile.
- Keep old diagnostic field names so bundle readers can ignore new fields.
- Avoid database migrations so rollback is code/config only.

## Post-Design Constitution Check

- **Raspberry Pi reliability**: PASS. The implementation slices are small, bounded, and hardware acceptance is explicit.
- **Minimal local stack**: PASS. No new services, frameworks, or storage systems.
- **Layering**: PASS. Profiles/CLI/controller only carry policy; detection and persistence enforce it; web remains the operator surface.
- **Honest RF claims**: PASS. Raw fragments, measured values, identity policy, persisted card spans, and display spans remain separate.
- **Migration safety**: PASS. No schema changes are required for this plan.
- **Operator validation**: PASS. Pi 5 canary validation remains GUI/controller diagnostic-bundle based.
