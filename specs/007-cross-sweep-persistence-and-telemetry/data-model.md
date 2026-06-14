# Data Model: Cross-Sweep Persistence and Telemetry

This feature adds sweep-loop-aware persistence state and auditable telemetry around the existing scanner/controller job lifecycle. It preserves the 006 characterization model: raw detector evidence, measured signal characterization, match span, display span, persisted baseline span, and contextual metadata remain separate.

## Naming Guidance

Use explicit prefixes for new diagnostic or sidecar fields so they do not collide with existing persistent detection fields:

- `sweep_loop_id`
- `observation_count`
- `observation_loop_count`
- `first_observed_sweep_id`
- `last_observed_sweep_id`
- `cross_sweep_candidate_id`
- `requested_profile`
- `applied_profile`
- `profile_applied`
- `profile_skip_reason`
- `operator_overrides`
- `profile_defaults`
- `final_effective_params`
- `requested_gain`
- `actual_gain`
- `gain_mode`
- `supported_gains`
- `device_index`
- `device_serial`
- `device_tuner`
- `bin_width_hz`

Existing characterization names remain unchanged:

- `raw_*`
- `measured_*`
- `match_*`
- `display_*`
- `stable_center_hz`
- `bandplan_*`
- `profile_context`

## Sweep Loop Observation

**Purpose**: Represents one compatible observation of a candidate signal during one complete sweep loop.

**Fields**:

- `sweep_loop_id`: complete sweep identifier, aligned with scanner `sweep_seq`
- `window_idx`: window inside the sweep where the observation appeared
- `candidate_center_hz`
- `candidate_low_hz`
- `candidate_high_hz`
- `candidate_bandwidth_hz`
- `raw_low_hz`, `raw_high_hz`, `raw_center_hz`, `raw_bandwidth_hz`
- `measured_center_hz`, `measured_bandwidth_hz` when available
- `source_pass`: coarse or revisit
- `snr_db`, `peak_db`, `noise_db`
- `observed_at_utc`

**Validation Rules**:

- One complete sweep loop may contribute at most one loop observation to thresholds for the same candidate unless a later plan explicitly allows otherwise.
- The raw detector span must remain distinct from match and display spans.
- Observations must include enough center/span/width data for compatibility checks.

## Cross-Sweep Candidate State

**Purpose**: Accumulates compatible observations across complete sweep loops before a persistent baseline card is created or updated.

**Fields**:

- `cross_sweep_candidate_id`
- `baseline_id`
- `first_observed_sweep_id`
- `last_observed_sweep_id`
- `observation_count`
- `observation_loop_count`
- `observed_sweep_ids`
- `stable_center_hz`
- `candidate_center_hz`
- `match_low_hz`
- `match_high_hz`
- `match_bandwidth_hz`
- `measured_bandwidth_hz`
- `last_raw_segment`
- `last_seen_utc`
- `promotion_ready`: boolean
- `rejection_reason` when expired or incompatible

**Validation Rules**:

- `observation_loop_count` must count distinct complete sweep loops.
- Compatibility checks must use bounded center/span/width criteria.
- Candidate state must expire or be pruned so it cannot grow unbounded.
- Candidate state must not be shown as a persistent baseline card before promotion.

## Persistent Baseline Card

**Purpose**: Existing operator-facing stable signal record shown in the web UI.

**Fields**:

- Existing `baseline_detections` fields such as `f_low_hz`, `f_high_hz`, `f_center_hz`, `total_hits`, `total_windows`, `confidence`, `service`, `region`, `bandplan_notes`
- Existing characterization diagnostics that reference `detection_id`
- Optional future additive fields only if later planning justifies persistence

**Validation Rules**:

- `f_low_hz <= f_center_hz <= f_high_hz` must hold after insert, update, revisit, and hysteresis paths.
- Persisted span remains separate from measured and display spans.
- Nearby FM stations must remain separate under profile matching rules.

## Effective-Parameter Manifest

**Purpose**: Job-level structured record of requested, profile-derived, overridden, skipped, fallback, and final effective settings.

**Fields**:

- `job_id`
- `requested_profile`
- `applied_profile`
- `profile_applied`
- `profile_skip_reason`
- `operator_overrides`
- `profile_defaults`
- `fallback_defaults`
- `final_effective_params`
- `frequency_range_hz`
- `step_hz`
- `sample_rate_hz`
- `fft`
- `bin_width_hz`
- `persistence`
- `revisit`
- `segment_center`
- `match_bandwidth`
- `display_bandwidth`
- `width_caps`
- `gain`
- `device`

**Validation Rules**:

- Requested, profile-derived, override, and final effective values must be distinguishable.
- Out-of-band profile skips must be visible without parsing logs.
- Missing optional fields must be explicit nulls or include unavailable reasons.

## Profile Application Record

**Purpose**: Explains profile request/application decisions for a scan.

**Fields**:

- `requested_profile`
- `requested_range_hz`
- `profile_range_hz`
- `applied_profile`
- `profile_applied`
- `profile_skip_reason`
- `profile_defaults_applied`
- `operator_overrides_preserved`

**Validation Rules**:

- A requested but skipped profile must not be represented as applied.
- Operator overrides must be preserved and reported.
- Profile context must not be treated as classification proof.

## Controller Parameter Mapping

**Purpose**: Documents the relation between web/API params, controller command builder keys, scanner CLI flags, and scanner effective args.

**Fields**:

- `api_param`
- `controller_key`
- `scanner_flag`
- `scanner_arg`
- `supported`: boolean
- `mapping_note`

**Validation Rules**:

- Every scanner-supported characterization, revisit, persistence, and width parameter from the spec is mapped or explicitly marked unsupported.
- Unsupported or differently named parameters are documented rather than silently dropped.

## Diagnostic Event Summary

**Purpose**: Compact aggregate and sample evidence for scanner/controller decisions.

**Fields**:

- `event_counts`
- `segment_inventory_count`
- `cluster_emitted_count`
- `cluster_rejected_count`
- `persistence_match_count`
- `persistence_no_match_count`
- `width_decision_count`
- `revisit_queued_count`
- `revisit_result_count`
- `characterization_record_count`
- `parse_errors`
- `truncated`

**Validation Rules**:

- Aggregate counts must remain available even when detailed diagnostic tails are truncated.
- The summary must be derived from structured events, not scanner log text.
- Missing or truncated evidence must appear in `manifest.json`.

## Device Telemetry Snapshot

**Purpose**: Records receiver state for a job so RF changes are not confused with receiver configuration changes.

**Fields**:

- `device_key`
- `device_kind`
- `device_index`
- `device_serial`
- `device_label`
- `device_tuner`
- `driver`
- `requested_gain`
- `actual_gain`
- `gain_mode`
- `supported_gains`
- `sample_rate_hz`
- `actual_sample_rate_hz`
- `fft`
- `bin_width_hz`
- `selected_profile`
- `unavailable_fields`

**Validation Rules**:

- Unavailable fields must be null or listed in `unavailable_fields`.
- Missing optional telemetry must not fail a scan.
- Hardware validation must confirm real RTL-SDR values when available.

## Storage Strategy

### Option A: Running-Job Cross-Sweep State Plus Diagnostics

**Description**: Keep cross-sweep candidate state in scanner memory during a running job, promote into existing persistence only after thresholds are satisfied, and export structured decisions through diagnostics.

**Advantages**:

- Minimal migration risk.
- Solves the primary once-per-sweep promotion gap for active scans.
- Keeps unpromoted candidates out of persistent baseline cards.

**Tradeoffs**:

- Candidate state is not restart durable.
- Long-run replay beyond one job depends on diagnostic exports.

### Option B: Additive Sidecar Persistence

**Description**: Add a narrow sidecar table or additive state record for cross-sweep candidates if implementation proves restart durability is required.

**Advantages**:

- Supports restart recovery and longer-running candidate audit trails.
- Can feed later replay and classifier work.

**Tradeoffs**:

- Requires migration planning and backward compatibility tests.
- Increases risk of confusing candidate state with promoted cards.

**Current Recommendation**: Start with Option A. Add Option B only if tests or hardware validation show running-job state is insufficient.

## State Flow

```text
web/API job params
  -> controller command mapping
  -> scanner effective args and profile decision
  -> complete sweep loop
  -> raw detector segments
  -> per-loop compatible observations
  -> cross-sweep candidate state
  -> persistence promotion/update
  -> characterization and revisit evidence
  -> diagnostic summaries and effective-parameter manifest
  -> operator diagnostic bundle export
```
