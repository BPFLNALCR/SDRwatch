# Data Model: FM Signal Characterization

This feature adds a conceptual characterization layer on top of the existing FM Validation stability work. The goal is to preserve the current operator-facing card behavior while explicitly separating raw detector evidence, measured characterization, persistence matching behavior, display behavior, and contextual metadata.

## Naming Guidance

The current repository already uses `f_low_hz`, `f_high_hz`, `f_center_hz`, and `bandwidth_hz` for detector segments and persistent detections. To avoid collapsing concepts, characterization fields should use explicit prefixes when they are emitted alongside existing detection fields.

Recommended field names for new characterization artifacts:

- `raw_low_hz`
- `raw_high_hz`
- `raw_center_hz`
- `raw_bandwidth_hz`
- `measured_center_hz`
- `measured_bandwidth_hz`
- `measured_bandwidth_confidence`
- `match_low_hz`
- `match_high_hz`
- `match_bandwidth_hz`
- `display_low_hz`
- `display_high_hz`
- `display_bandwidth_hz`
- `center_stability_hz`
- `bandwidth_stability_hz`
- `characterization_confidence`
- `characterization_method`
- `classification_candidate`
- `classification_evidence`
- `bandplan_service`
- `bandplan_region`
- `bandplan_notes`
- `profile_context`

## Raw Detection Segment

**Purpose**: Represents what the FFT and CFAR detector actually measured in a window before any persistence or display shaping is applied.

**Fields**:

- `raw_low_hz`
- `raw_high_hz`
- `raw_center_hz`
- `raw_bandwidth_hz`
- `peak_db`
- `noise_db`
- `snr_db`
- `sweep_id`
- `window_idx`
- `source_pass`: coarse or revisit

**Validation Rules**:

- Raw detector width must remain inspectable even when later spans are widened or refined.
- Raw segment width must not be treated as the final occupied-bandwidth estimate by default.
- Multiple raw segments may contribute to one later characterization record.

## Measured Characterization Record

**Purpose**: Represents the best available measured estimate of the signal itself, independent of how it is matched or displayed.

**Fields**:

- `measured_center_hz`
- `measured_bandwidth_hz`
- `measured_bandwidth_confidence`
- `characterization_confidence`
- `characterization_method`
- `peak_db`
- `noise_db`
- `snr_db`
- `center_stability_hz`
- `bandwidth_stability_hz`
- `revisit_measurement_count`
- `coarse_measurement_count`

**Validation Rules**:

- Measured center and measured bandwidth must remain distinct from raw width, match span, and display span.
- Confidence must reflect actual evidence rather than bandplan or profile context.
- Stability metrics must reflect variation across sweeps rather than one isolated segment.

## Classification Evidence

**Purpose**: Represents a future-facing classification layer built from measured evidence while remaining separate from contextual metadata.

**Fields**:

- `classification_candidate`
- `classification_evidence`
- `characterization_confidence`
- `evidence_sources`
- `context_only`: boolean marker for contextual hints that are not measured proof

**Validation Rules**:

- The candidate may remain empty, unknown, or tentative when evidence is insufficient.
- Contextual labels must never be the only reason a candidate is assigned.
- Any non-unknown candidate should carry enough evidence detail to explain it.

## Optional FM-Specific Evidence

**Purpose**: Captures supporting indicators that may strengthen FM characterization without being mandatory.

**Fields**:

- `pilot_19khz_present`
- `pilot_19khz_confidence`
- `rds_57khz_present`
- `rds_57khz_confidence`
- `stereo_indicator_present`

**Validation Rules**:

- These indicators are optional enhancements for the first implementation.
- Their absence must not block basic FM characterization.
- Their presence should raise confidence only as supporting evidence, not as sole proof.

## Persistence Match Span

**Purpose**: Defines how the system decides that later detections belong to the same long-lived signal.

**Fields**:

- Existing compatibility fields: `f_low_hz`, `f_high_hz`, `f_center_hz`
- Derived or exported aliases: `match_low_hz`, `match_high_hz`, `match_bandwidth_hz`
- `match_reason`
- `center_match_hz`

**Validation Rules**:

- Persistence span must remain distinct from measured occupied bandwidth.
- Nearby FM stations must remain separate under the chosen match-span rules.
- The invariant `f_low_hz <= f_center_hz <= f_high_hz` must always hold after updates and revisit refinement.

## Display Card Span

**Purpose**: Defines the stable operator-facing span used for card rendering and summaries.

**Fields**:

- `display_low_hz`
- `display_high_hz`
- `display_bandwidth_hz`
- `display_reason`

**Validation Rules**:

- Display span must stay stable and operator-friendly.
- Display bandwidth must not be presented as the measured occupied bandwidth.
- Display behavior must preserve the stable FM Validation card count achieved by the previous feature.

## Contextual Signal Metadata

**Purpose**: Carries externally supplied or operator-selected metadata that helps interpret signals without claiming measured truth.

**Fields**:

- `bandplan_service`
- `bandplan_region`
- `bandplan_notes`
- `profile_context`
- Existing operator-side fields such as `classification`, `label`, `notes`, and `selected` when relevant to presentation

**Validation Rules**:

- Contextual metadata must remain visible and useful.
- Contextual metadata must not be confused with measured characterization or classification evidence.
- Existing web-layer operator classification remains separate from scanner-side measured evidence.

## Characterization State Across Sweeps

**Purpose**: Tracks how characterization evolves over time for a persistent signal.

**Fields**:

- `detection_id` or other stable signal key
- `first_characterized_utc`
- `last_characterized_utc`
- `center_stability_hz`
- `bandwidth_stability_hz`
- `characterization_confidence`
- `last_characterization_method`
- `revisit_measurement_count`
- `coarse_measurement_count`

**Validation Rules**:

- Cross-sweep state must support the requested center and bandwidth stability metrics.
- The design must work whether the first implementation stores this state durably or emits it through diagnostics-first summaries.

## Storage Strategy Candidates

### Option A: Diagnostics-First Characterization Artifacts

**Description**: Emit characterization records through diagnostic JSONL and bounded diagnostic bundle summaries while preserving the current persistent schema.

**Advantages**:

- Lowest migration risk.
- Fastest way to validate field separation and confidence reporting.
- Compatible with the user's request to avoid schema migration unless needed.

**Tradeoffs**:

- Long-run stability metrics may be harder to compare across runs unless summaries or stable signal identifiers are carried forward.

### Option B: Additive Persistent Characterization Fields

**Description**: Add new persistent fields to existing signal records or a narrow sidecar structure after the model is proven.

**Advantages**:

- Stronger support for long-run center and bandwidth stability tracking.
- Easier foundation for later automatic classification features.

**Tradeoffs**:

- Requires explicit migration planning and compatibility validation.
- Raises the risk of collapsing current match or display semantics if field ownership is not kept clear.

**Current Recommendation**: Start with Option A unless implementation planning proves that cross-sweep stability metrics cannot be supported cleanly enough without additive persistent state.

## State Flow

```text
raw detector segment
  -> measured characterization record
  -> persistence match decision
  -> display card span
  -> contextual metadata attachment
  -> optional classification evidence
  -> diagnostic export and, if justified later, additive persistence
```
