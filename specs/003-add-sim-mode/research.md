# Research: No-Hardware Simulation Mode

## Decision 1: Implement simulation as a new scanner source that satisfies the existing `tune`/`read`/`close` contract

- **Decision**: Add a synthetic SDR source under `sdrwatch/drivers/` and dispatch it from `ScannerRunner._select_source()` when the user explicitly selects a simulation driver value.
- **Rationale**: The current sweep pipeline already treats the source abstractly: the sweeper calls `src.tune()` and `src.read()` to obtain complex samples, then all PSD, detection, baseline, and persistence logic runs downstream. Reusing that seam keeps the change small and exercises the real scan pipeline.
- **Alternatives considered**:
  - Inject synthetic detections directly after DSP. Rejected because it would skip the PSD and detection path that the feature is meant to validate.
  - Add a separate simulation-only scanner command. Rejected because it would fragment the CLI and create a second scan workflow.

## Decision 2: Generate deterministic synthetic IQ samples rather than synthetic database rows

- **Decision**: Build deterministic complex sample output that includes a configurable noise floor, multiple stable carriers, an intermittent emission gated by sweep/window sequence, and a power-shifting emission whose level changes predictably across sweeps.
- **Rationale**: Generating samples preserves the same DSP and thresholding behavior as real scans as much as possible while allowing repeatable no-hardware tests. Seeded synthetic input keeps CI deterministic.
- **Alternatives considered**:
  - Write synthetic PSD arrays directly into detection logic. Rejected because it bypasses source-side sample handling and makes the simulation less representative of a real scan.
  - Use randomized, non-seeded synthetic signals. Rejected because CI and regression checks require identical outputs for identical inputs.

## Decision 3: Keep persistence schema and dashboard reads unchanged

- **Decision**: Reuse the existing `Store` tables and current dashboard queries without adding simulation-specific tables, columns, or special-case read paths.
- **Rationale**: The dashboard and baseline helpers already derive signal cards, recent activity, and summaries from `baseline_detections`, `scan_updates`, `baseline_noise`, `baseline_occupancy`, and `baseline_snapshot`. Writing simulated runs through the same tables proves end-to-end compatibility.
- **Alternatives considered**:
  - Add a flag column to mark simulated rows. Rejected for this feature because it is not required for the acceptance criteria and would expand schema and dashboard scope unnecessarily.
  - Use a separate simulation database schema. Rejected because the acceptance criteria require the web dashboard to show simulated detections from the generated database.

## Decision 4: Make simulation CLI-first and treat controller device discovery support as optional follow-up

- **Decision**: The required slice is an explicit CLI simulation mode, documented with an FM-band example. Controller and web support for listing a synthetic device are optional enhancements, not required for feature acceptance.
- **Rationale**: The controller and web control panel currently require a `device_key` discovered through `/devices`. Adding a synthetic discovered device is feasible, but the acceptance criteria can be satisfied by CLI-driven generation of a database that the existing web dashboard then reads.
- **Alternatives considered**:
  - Require controller `/devices` to return a simulated device in the first slice. Rejected because it increases API/control-panel scope without being necessary to satisfy the stated goal.
  - Leave simulation undocumented outside tests. Rejected because developers need a supported manual entrypoint.

## Decision 5: Verify simulation through deterministic DB-write tests plus a manual dashboard validation guide

- **Decision**: Add automated tests that create temp baselines and databases, run simulated sweeps, and assert deterministic contents in `scan_updates`, `baseline_detections`, and baseline stats tables. Document a manual simulated FM-band scan followed by opening the web app against the resulting DB.
- **Rationale**: This covers the feature's two core promises: hardware-free automated validation and visible simulated results in the existing dashboard.
- **Alternatives considered**:
  - Limit tests to unit-testing the synthetic source only. Rejected because that would not prove persistence or dashboard compatibility.
  - Depend on browser automation for dashboard proof. Rejected because the acceptance criteria only require the dashboard to show the generated data, which can be demonstrated by documented manual validation and DB contracts.