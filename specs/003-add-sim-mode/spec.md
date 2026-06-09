# Feature Specification: No-Hardware Simulation Mode

**Feature Branch**: `[003-add-sim-mode]`

**Created**: 2026-06-09

**Status**: Draft

**Input**: User description: "Add a no-hardware simulation mode for SDRwatch."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run a Simulated Scan (Priority: P1)

As a developer, I can run a scan in a no-hardware simulation mode that follows the same baseline-oriented workflow as a physical scan, so I can develop and debug SDRwatch without attaching an SDR device.

**Why this priority**: This is the core capability that unlocks local development, repeatable debugging, and hardware-free validation.

**Independent Test**: Create or select a baseline, run the documented simulated FM-band scan on a machine with no attached SDR, and confirm that detections, scan updates, and baseline changes are written to the standard data store.

**Acceptance Scenarios**:

1. **Given** no SDR hardware is attached and a baseline is available, **When** a developer starts a scan in simulation mode using the documented FM-band workflow, **Then** the scan completes without hardware access errors and writes detections and scan updates to the same data store used by physical scans.
2. **Given** the same baseline and the same simulation inputs on a fresh data store, **When** the developer reruns the same simulated scan, **Then** the stable, intermittent, and power-shifting synthetic signals produce the same expected persisted outcomes needed for repeatable development and testing.
3. **Given** no baseline is selected, **When** the developer attempts to run a simulated scan, **Then** the system rejects the request using the same baseline requirement applied to physical scans.

---

### User Story 2 - Review Simulated Results in the Dashboard (Priority: P2)

As a developer or maintainer, I can open the existing dashboard against a database populated by simulated scans and inspect detections, baseline summaries, and recent activity, so I can verify end-to-end behavior without RF hardware.

**Why this priority**: The simulator is only useful if it drives the same persisted outputs that the rest of SDRwatch already depends on.

**Independent Test**: Run one or more simulated scans, start the web UI against the generated database, and confirm that the existing views show simulated detections and scan activity without requiring any hardware-specific changes.

**Acceptance Scenarios**:

1. **Given** a database populated by simulated scans, **When** the user opens the dashboard, **Then** detections, baseline summaries, and recent scan activity appear through the existing views without requiring a separate storage format or a hardware device.
2. **Given** repeated simulated sweeps over the same baseline, **When** the user refreshes the dashboard views, **Then** intermittent and power-shifting signal behavior is reflected through the stored scan history and baseline updates already used by the product.

---

### User Story 3 - Run Hardware-Free Automated Validation (Priority: P3)

As a maintainer, I can run automated validation for simulated scans on a machine with no SDR hardware, so the project can verify deterministic scan behavior and persistence in continuous integration.

**Why this priority**: Deterministic no-hardware validation reduces regressions in detection and persistence behavior while keeping CI independent of attached devices.

**Independent Test**: Execute the documented automated no-hardware test suite on a hardware-free environment and verify that deterministic simulated outputs and required data-store writes match the expected results.

**Acceptance Scenarios**:

1. **Given** a hardware-free validation environment, **When** maintainers run the automated simulation tests, **Then** the expected deterministic detections and required writes to the standard scan and baseline tables succeed.
2. **Given** simulation tests are executed without selecting a physical driver, **When** the validation suite runs, **Then** it completes without depending on attached SDR devices or optional hardware-only libraries.

---

### Edge Cases

- A workstation has no discoverable SDR devices at all; simulation mode must still start when explicitly selected.
- A user accidentally selects a physical driver on a hardware-free machine; the system must preserve the current hardware error behavior rather than silently falling back to simulation.
- Simulated scans run against a database that already contains real baseline history; writes must stay isolated to the chosen baseline and must not alter unrelated baselines.
- Long-running simulated loops must keep the intermittent and power-shifting signals repeatable by sweep order so automated checks remain deterministic.
- Existing dashboard views must read simulated scan outputs from the current tables without requiring simulation-specific schema changes or filters.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a no-hardware scan mode that a developer can intentionally select at scan start without attaching an SDR device.
- **FR-002**: System MUST generate deterministic synthetic spectrum observations in simulation mode that include a noise floor, multiple stable carriers, at least one intermittent signal, and at least one signal whose power changes across sweeps.
- **FR-003**: System MUST route simulated observations through the same detection, baseline update, persistence, and scan summary workflows used by physical scans wherever those workflows do not require direct hardware access.
- **FR-004**: System MUST preserve the existing baseline-first scan contract in simulation mode, including baseline selection requirements and writes to the standard scan and baseline data stores.
- **FR-005**: System MUST keep existing RTL-SDR and Soapy-backed workflows unchanged unless simulation mode is explicitly selected.
- **FR-006**: System MUST expose a documented command-line way to start a simulated scan, including a documented FM-band example for development use.
- **FR-007**: System MUST produce repeatable simulated results for the same documented inputs so automated tests can verify detections and persisted scan outcomes without hardware.
- **FR-008**: System MUST allow the existing dashboard experience to display detections, baseline summaries, and recent scan activity produced by simulated scans from the generated data store.
- **FR-009**: System MUST provide automated no-hardware tests that verify deterministic simulated output and successful writes to the standard scan and baseline persistence tables.
- **FR-010**: System MUST allow hardware-free validation environments to run the simulation tests without requiring attached SDR devices or optional dependencies that are only needed for real hardware paths.

### Key Entities *(include if feature involves data)*

- **Simulation Scan Request**: A scan request that explicitly selects synthetic input rather than a physical SDR while still targeting a chosen baseline and the normal scan workflow.
- **Synthetic Signal Pattern**: The deterministic set of simulated spectrum features, including the noise floor, stable carriers, intermittent emission, and power-shifting emission used to exercise detection behavior.
- **Simulated Sweep Result**: The stored output of a simulated sweep, including detections, scan activity, and baseline changes that can be inspected through existing tools.
- **Baseline Observation History**: The accumulated per-baseline state updated by simulated sweeps in the same persistence paths used for physical scans.

## Constitution Alignment *(mandatory)*

- **Pi 5 / Field Reliability Impact**: The feature adds a development and validation path without changing the default operator workflow for Raspberry Pi deployments. Real scans remain explicit, and simulated scans must stay bounded and diagnosable so they do not burden long-running field systems.
- **Stack / Dependency Impact**: No new service layer, cloud dependency, or mandatory frontend technology is expected. The feature should rely on local deterministic data generation and the existing local stack.
- **Layer Ownership**: The change affects scan input selection, synthetic observation generation, documentation, and automated validation. Persistence, controller behavior, and the web dashboard remain consumers of the same scan and database contracts rather than new sources of scan logic.
- **Compatibility / Migration Impact**: The feature is expected to be additive. Existing command-line workflows continue to work, current database tables remain authoritative, and no destructive schema or API transition is required for the dashboard to read simulated results.
- **Security / Offline Impact**: Token behavior and offline workflows remain unchanged. Developers must be able to run simulated scans and inspect results without network access or attached SDR hardware.
- **Verification Plan**: Validate the feature with deterministic automated simulation tests, data-store write checks, a documented simulated FM-band scan, and a reproducible dashboard check against the generated database. Include a compatibility smoke check to confirm existing real-hardware paths are unaffected when simulation mode is not selected.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a machine with no attached SDR, a developer can follow the documented simulation steps and produce a populated scan dataset in 10 minutes or less.
- **SC-002**: In automated validation, repeating the same documented simulated scan against fresh baselines produces the same expected detection and scan-summary outcomes in 100% of runs.
- **SC-003**: The standard dashboard can display detections, baseline summaries, and recent activity from a dataset generated entirely by simulated scans.
- **SC-004**: The automated hardware-free validation suite completes successfully on runners with no attached SDR device while covering deterministic simulated behavior and persisted scan results.

## Assumptions

- The simulation capability is intended for development, testing, demonstrations, and CI confidence rather than RF-accurate modeling of every real-world condition.
- Existing baseline and scan storage tables are sufficient for simulated results, and the dashboard reads those outputs through its current database-driven views.
- The documented FM-band simulation example favors repeatability and representative behavior over exact physical realism.
- Real-device verification remains a separate workflow on supported Linux or Raspberry Pi environments and is not replaced by simulation mode.