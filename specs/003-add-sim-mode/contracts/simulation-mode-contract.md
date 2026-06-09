# Contract: No-Hardware Simulation Mode

## Purpose

Define the minimum contract for SDRwatch's additive no-hardware simulation mode so implementation, tests, and documentation stay aligned.

## Output

- **Scanner capability**: explicit CLI-accessible simulation mode
- **Driver boundary**: synthetic source implementing the same source contract as real scanners
- **Persistence**: writes to the existing baseline and scan summary tables
- **Documentation**: runnable simulated FM-band workflow
- **Automation**: no-hardware tests covering deterministic output and persisted results

## Required Capabilities

1. **Explicit Scan Selection**
   - A developer can intentionally select simulation mode from the scanner CLI.
   - Baseline selection remains mandatory exactly as it is for real scans.
   - Choosing a physical driver on a hardware-free machine must continue to surface the normal hardware error path rather than silently switching to simulation.

2. **Source Interface Compatibility**
   - The simulation source must satisfy the same operational contract the sweeper already uses for SDR sources:
     - `tune(center_hz)`
     - `read(count)` returning complex sample data compatible with the existing PSD pipeline
     - `close()`
   - The source must be isolated from real driver imports and must not interfere with RTL-SDR or Soapy-backed operation.

3. **Deterministic Synthetic Spectrum**
   - Synthetic output must include a noise floor, multiple stable carriers, at least one intermittent signal, and at least one power-shifting signal.
   - Identical documented inputs on fresh state must yield identical expected persisted outcomes.
   - Determinism must be strong enough for CI to assert specific detection and persistence behavior.

4. **Pipeline Reuse**
   - Simulated scans must flow through the existing PSD, detection, baseline, persistence, and scan summary logic rather than bypassing those stages.
   - Existing SQLite tables remain authoritative for simulated results.
   - Optional JSONL output, if enabled, continues to use the current emission path.

5. **Dashboard Compatibility**
   - A database produced by simulated scans must be readable by the existing dashboard and baseline helper queries.
   - No simulation-specific schema or separate read path is required for the feature to be considered complete.

6. **Validation Contract**
   - Automated no-hardware tests must verify deterministic simulated behavior.
   - Automated no-hardware tests must verify required writes into the standard baseline and scan summary tables.
   - Documentation must include a simulated FM-band scan example and a reproducible way to inspect the generated database through the existing web UI.

## Evidence Rules

- CLI support must be traceable to committed scanner code and documentation.
- Deterministic behavior must be traceable to automated tests that can run without SDR hardware.
- Persistence compatibility must be traceable to assertions against current baseline tables.
- Dashboard compatibility must be traceable to existing web/database contracts rather than a simulation-only rendering path.

## Acceptance Conditions

- A developer can run a scan without SDR hardware by explicitly selecting simulation mode.
- Repeating the same simulated scan on fresh state produces the same expected persistence results.
- The generated SQLite database can be opened through the current web dashboard and show simulated detections and recent activity.
- The no-hardware test suite can run in CI without attached SDR hardware.