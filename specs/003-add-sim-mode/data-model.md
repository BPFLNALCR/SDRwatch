# Data Model: No-Hardware Simulation Mode

## Overview

This feature does not require new persistent schema. Its data model is a conceptual model for the synthetic scan input, the deterministic controls that govern it, and the existing persisted outputs that prove the normal scan pipeline was exercised.

## Entities

### 1. Simulation Scan Request

- **Purpose**: A scan request that intentionally selects synthetic input while preserving the standard baseline-oriented scan contract.
- **Fields**:
  - `driver_key`: explicit simulation selector used by the CLI.
  - `baseline_id`: required target baseline.
  - `frequency_span`: start/stop frequencies and step for sweep scheduling.
  - `sweep_controls`: FFT, averaging, repeat/loop, threshold, and optional profile parameters shared with real scans.
  - `output_targets`: DB path, optional JSONL output, and diagnostic paths.
- **Validation rules**:
  - Must require the same baseline selection semantics as real scans.
  - Must not silently fall back from a requested physical driver to simulation.
  - Must reuse the current scan argument contract wherever possible.

### 2. Synthetic Signal Pattern

- **Purpose**: The deterministic RF-like pattern emitted by the simulation source.
- **Fields**:
  - `noise_floor_profile`: repeatable background noise characteristics.
  - `stable_carriers`: fixed center frequencies and amplitudes that persist across sweeps.
  - `intermittent_signal`: a gated emission with a deterministic presence schedule.
  - `power_shifting_signal`: a signal whose amplitude changes according to a deterministic sequence.
  - `seed`: deterministic state used to reproduce outputs across runs.
  - `window_sequence`: sweep/window counters used to vary intermittent and power-shift behavior predictably.
- **Validation rules**:
  - Must include each required signal category from the specification.
  - Must generate identical outputs for identical inputs and seed.
  - Must remain bounded enough to run quickly in CI and developer workflows.

### 3. Simulation Source State

- **Purpose**: The in-memory state needed by the synthetic driver to produce the next set of complex samples.
- **Fields**:
  - `current_center_hz`: tuned center frequency.
  - `sample_rate_hz`: active sample rate.
  - `rng_state`: deterministic random state.
  - `sweep_index`: current sweep count.
  - `window_index`: current window count within the scan.
- **Validation rules**:
  - Must be isolated per scan instance.
  - Must reset deterministically between fresh runs unless explicitly extended later.
  - Must not leak state into physical drivers or global process behavior.

### 4. Persisted Simulation Evidence

- **Purpose**: The existing persisted outputs that demonstrate the simulation exercised the real scan workflow.
- **Fields**:
  - `baseline_noise_rows`: updated per-bin noise and power EMA records.
  - `baseline_occupancy_rows`: updated occupancy counts and observed durations.
  - `baseline_detection_rows`: persistent signal records for stable, intermittent, or power-shifting detections that meet existing thresholds.
  - `scan_update_rows`: per-sweep counters and timing summaries.
  - `baseline_snapshot`: aggregate recent activity and persistent detection counts.
- **Validation rules**:
  - All rows must use the selected baseline ID.
  - The persisted outputs must remain readable by current dashboard helpers without simulation-only logic.
  - Repeat runs against fresh baselines must yield deterministic expected outcomes for automated tests.

## Relationships

- One **Simulation Scan Request** drives one synthetic source instance for the duration of a scan run.
- One **Synthetic Signal Pattern** defines the content emitted by the **Simulation Source State** across windows and sweeps.
- Each scan run produces many **Persisted Simulation Evidence** rows through the existing baseline and scan update tables.
- The existing web helpers and dashboard views consume **Persisted Simulation Evidence** without knowing whether the input came from hardware or simulation.

## State Transitions

### Simulation Source Lifecycle

- `initialized` -> `tuned`: after the runner selects the source and the sweeper requests a center frequency.
- `tuned` -> `samples_emitted`: after the source produces complex samples for a window.
- `samples_emitted` -> `advanced`: after sweep/window counters update for the next deterministic pattern state.
- `advanced` -> `closed`: after the scan runner completes or aborts and closes the source.

### Persisted Evidence Lifecycle

- `absent` -> `observed`: when a simulated sweep writes baseline noise, occupancy, and scan update rows.
- `observed` -> `persistent`: when repeated windows/sweeps satisfy the existing persistence rules and produce or update `baseline_detections`.
- `persistent` -> `displayed`: when the existing dashboard reads the baseline tables and renders summaries or signal cards.