# Feature Specification: Hardware-Aware Multi-RTL Guard/Rover Mode

**Feature Branch**: `007-cross-sweep-persistence-and-telemetry`

**Created**: 2026-06-14

**Status**: Draft

**Input**: User description: "Hardware-aware multi-RTL guard/rover mode for RTL-SDR devices using the currently runnable native RTL-SDR scanner path, preserving current FM broadcast behavior, cross-sweep persistence, diagnostics, controller job lifecycle, and GUI-first operation."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Understand Attached Receiver Capability (Priority: P1)

As an SDRwatch operator, I need the control surface to show which attached receivers are detected, which are runnable today, and what capability tier the station can support, so I do not start scans based on misleading hardware claims.

**Why this priority**: Hardware-aware operation is not trustworthy unless the operator can distinguish runnable RTL receivers from unsupported or future hardware classes before starting work.

**Independent Test**: Can be fully tested through the web UI and controller job lifecycle by simulating or attaching zero, one, two, and three RTL receivers, opening the control surface, and confirming the displayed inventory, warnings, tier, and start availability.

**Acceptance Scenarios**:

1. **Given** no runnable RTL receivers are detected, **When** the operator opens the control surface, **Then** SDRwatch reports Tier 0 and does not offer a runnable scanner start that requires RTL hardware.
2. **Given** one detected runnable RTL receiver, **When** the operator opens the control surface, **Then** SDRwatch reports Tier 1 and shows the receiver label, current index, serial when available, runnable backend, busy state, role assignment state, and identity warning state.
3. **Given** two detected runnable RTL receivers, **When** the operator opens the control surface, **Then** SDRwatch reports Tier 2 and shows that manual GUARD plus ROVER operation is available.
4. **Given** three or more detected runnable RTL receivers, **When** the operator opens the control surface, **Then** SDRwatch reports Tier 2+ and shows that two GUARD roles plus a REFERENCE or ROVER role can be assigned manually.
5. **Given** two devices have missing or duplicate serial values, **When** the inventory is displayed, **Then** SDRwatch warns that durable identity is ambiguous and marks any index-based identity as unstable.

---

### User Story 2 - Run One RTL as a Guarded Receiver (Priority: P1)

As an operator with one RTL receiver, I need to keep the existing single-device scanning workflow and also assign that receiver to a GUARD role for one priority frequency window, so the system is useful before I add more hardware.

**Why this priority**: Tier 1 must remain a complete and safe workflow, and the feature must not regress current FM broadcast scanning, characterization, cross-sweep persistence, or diagnostic behavior.

**Independent Test**: Can be fully tested by assigning the single detected receiver to GUARD in the web UI, starting a guard-style job through the controller lifecycle, observing individual job status, stopping it, and confirming existing single-device scan presets still work.

**Acceptance Scenarios**:

1. **Given** one runnable RTL receiver and no active job on it, **When** the operator assigns GUARD and starts one guarded window, **Then** the job starts with that receiver, shows the GUARD role, and produces the same class of detections and diagnostics as existing scanner workflows.
2. **Given** the receiver is already locked by an active job, **When** the operator tries to start another role-aware job using that same receiver, **Then** SDRwatch rejects the start and explains which role or job is using the device.
3. **Given** an existing FM broadcast workflow, **When** the operator starts it using the existing controls, **Then** FM broadcast behavior, profile behavior, and cross-sweep promotion remain stable.

---

### User Story 3 - Run Two RTLs as Guard Plus Rover (Priority: P2)

As an operator with two RTL receivers, I need to manually assign one receiver as GUARD and one as ROVER, then run both at the same time, so high-priority windows can be watched while lower-priority ranges continue to be swept.

**Why this priority**: This is the first true multi-receiver value slice and proves that the controller, locks, status, diagnostics, and UI can handle parallel role-aware jobs.

**Independent Test**: Can be fully tested by assigning two distinct detected RTL receivers to GUARD and ROVER, starting a grouped role run through the web UI, observing both child jobs, then stopping the group and verifying both devices become available.

**Acceptance Scenarios**:

1. **Given** two runnable RTL receivers with distinct identities or acknowledged index warnings, **When** the operator assigns GUARD to one and ROVER to the other, **Then** the role assignment view shows both assignments without allowing either device to be assigned to another active role.
2. **Given** GUARD and ROVER are assigned to different receivers, **When** the operator starts the role-aware run, **Then** SDRwatch starts one GUARD job and one ROVER job, shows their child job IDs, device identities, role names, and running states, and records grouped status.
3. **Given** the operator stops the grouped role run, **When** stop completes, **Then** both child jobs are stopped or marked terminal, both device locks are released, and the grouped status no longer appears healthy if any child stop failed.
4. **Given** one child job exits unexpectedly, **When** the operator views grouped status, **Then** the grouped role run clearly shows the failed or stopped child and does not imply that all roles are still healthy.

---

### User Story 4 - Run Three RTLs with Two Guards and Reference or Rover (Priority: P3)

As an operator with three RTL receivers, I need to manually assign a friendly GUARD, a watchlist GUARD, and either a REFERENCE or ROVER receiver, so SDRwatch can monitor known and priority activity while collecting supporting context.

**Why this priority**: Tier 2+ validates the first deployment shape for a Pi-based hardware-aware watch node while keeping automatic scheduling and full signal fusion out of scope.

**Independent Test**: Can be tested by assigning three detected RTL receivers to two GUARD roles and one REFERENCE or ROVER role, starting the role-aware run, and confirming each physical receiver has only one active role and child job.

**Acceptance Scenarios**:

1. **Given** three runnable RTL receivers, **When** the operator assigns friendly GUARD, watchlist GUARD, and REFERENCE, **Then** the role view shows all three roles, their devices, identity confidence, and current job states.
2. **Given** the third receiver is assigned as ROVER instead of REFERENCE, **When** the operator starts the run, **Then** SDRwatch runs two guard-style jobs and one rover-style job without assigning the same physical receiver twice.
3. **Given** a REFERENCE role is running, **When** diagnostics are exported, **Then** reference records include receiver, role, job, frequency-window, and power/noise context suitable for later confidence scoring without applying automatic correction.

---

### User Story 5 - Benchmark and Diagnose Multi-RTL Operation (Priority: P3)

As an operator or maintainer, I need role-aware diagnostics, timing, and resource telemetry for each active receiver, so I can benchmark one, two, and three RTL operation on a Raspberry Pi 5 without guessing where bottlenecks are.

**Why this priority**: Multi-receiver operation adds CPU, USB, memory, and I/O risk. Diagnostics must make those risks visible before deeper scheduler or DSP optimization is attempted.

**Independent Test**: Can be tested by running one, two, and three role-aware jobs through the web UI and exporting diagnostics that show per-window timing, samples, device/role/job provenance, and resource summaries.

**Acceptance Scenarios**:

1. **Given** any role-aware job emits scan-window diagnostics, **When** the operator exports diagnostics, **Then** records include device identity, current index, serial when available, backend, assigned role, job or role-run identity, source task/profile, window frequency, sample rate, FFT size, averaging, segment count, samples read, and short-read or dropped-read representation when detectable.
2. **Given** timing data is available during a scan window, **When** diagnostics are exported, **Then** records include tune, flush, read, transform, detection, persistence/update, logging, and total window timing where feasible, with unavailable fields represented explicitly.
3. **Given** a role-aware run is active, **When** diagnostics or status are viewed, **Then** resource telemetry includes process identity, active device count, active role count, CPU load when available, and resident memory when available.

### Edge Cases

- No runnable RTL devices are detected, but unsupported hardware or planned hardware classes are present.
- A receiver has no serial value, a duplicate serial value, or a serial that cannot be read consistently.
- Device indexes change after unplug, replug, or restart.
- A receiver is locked by a job that has already exited.
- Startup reconciliation finds running job records whose processes no longer exist.
- Two operators or browser actions attempt to assign or start the same receiver at nearly the same time.
- A role-aware group starts one child job successfully and another child job fails before becoming healthy.
- A child job is stopped outside the grouped stop workflow.
- Timing, CPU, memory, or short-read telemetry is unavailable on the current platform.
- The operator attempts to start Airspy, HackRF, or Soapy-backed scanner execution before those runners exist.
- Existing single-device FM broadcast workflows are used while role assignment data exists in the controller.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: SDRwatch MUST expose a hardware inventory that lists detected receiver hardware and clearly distinguishes detected hardware, runnable scanner hardware, and planned or unsupported hardware classes.
- **FR-002**: SDRwatch MUST report an inferred capability tier: Tier 0 for no runnable RTL, Tier 1 for one runnable RTL, Tier 2 for two runnable RTLs, and Tier 2+ for three or more runnable RTLs.
- **FR-003**: For this feature, SDRwatch MUST treat native RTL-SDR receivers as the only runnable scanner hardware class.
- **FR-004**: SDRwatch MUST NOT expose Airspy, HackRF, Soapy-backed devices, or other planned hardware classes as runnable scanner options unless scanner execution support exists.
- **FR-005**: Attempts to start unsupported scanner hardware MUST fail before scan execution begins and MUST present a clear operator-facing reason.
- **FR-006**: Each detected RTL inventory entry MUST include an operator-facing label, current index, serial when available, stable device key when available, backend support status, runnable scanner backend when supported, busy or locked state, active job identity when any, assigned role when any, and warning state.
- **FR-007**: SDRwatch MUST prefer a unique serial-based identity when binding manual role assignments to RTL receivers.
- **FR-008**: SDRwatch MUST mark missing, duplicate, or index-only receiver identity as unstable or ambiguous and avoid presenting it as durable across unplug, replug, reboot, or controller restart.
- **FR-009**: Operators MUST be able to manually assign a detected runnable RTL receiver to GUARD, ROVER, or REFERENCE, and MUST be able to clear that assignment.
- **FR-010**: Persistent role assignment MUST be allowed only when the receiver has a unique stable identity; index-only assignments MUST be treated as session-scoped unless the operator reconfirms them after restart.
- **FR-011**: SDRwatch MUST prevent the same active physical receiver from being assigned to more than one simultaneously running role or job.
- **FR-012**: SDRwatch MUST show role assignments, busy or locked state, and active jobs together so the operator can understand why a receiver cannot be used.
- **FR-013**: Existing single-job start, stop, status, and diagnostic workflows MUST remain available and compatible for current single-device workflows.
- **FR-014**: Role-aware operation MUST be additive to the existing job lifecycle and MUST retain individual child job visibility.
- **FR-015**: A grouped role-aware run, if introduced, MUST expose group identity, role name, receiver identity, child job identity, running or terminal state, start time, last update time, and error message when any.
- **FR-016**: Stopping a grouped role-aware run MUST stop the relevant child jobs cleanly and release their receiver reservations when the jobs are terminal.
- **FR-017**: If one child job stops or fails independently, grouped role status MUST reflect the degraded or terminal child state.
- **FR-018**: Startup reconciliation, stale reservation cleanup, process reaping, and stop behavior MUST continue to work for role-aware jobs.
- **FR-019**: Concurrent attempts to assign or start the same receiver MUST NOT both succeed.
- **FR-020**: With one runnable RTL, SDRwatch MUST support existing single-device scanning behavior and one GUARD assignment for a priority frequency window.
- **FR-021**: GUARD behavior MUST keep attention on one configured priority frequency window and avoid roaming away from that task unless the operator changes the configuration.
- **FR-022**: ROVER behavior MUST sweep configured lower-priority ranges and may accept a slower detection cadence than GUARD behavior.
- **FR-023**: REFERENCE behavior MUST keep attention on a configured stable, noise, or reference window and record power, noise, or signal-context telemetry where available.
- **FR-024**: GUARD, ROVER, and REFERENCE roles MUST use existing detection, baseline, event, and cross-sweep persistence behavior where applicable without regressing current single-device behavior.
- **FR-025**: Multi-device operation MUST NOT silently merge observations from different receivers as if they were equivalent.
- **FR-026**: Diagnostics and persisted records where feasible MUST include receiver role, device key or identity, serial when available, runtime index, job identity, role-run identity when any, and source profile or task.
- **FR-027**: The feature MUST preserve enough provenance to support future signal tracks and per-receiver observations without implementing full fusion in this feature.
- **FR-028**: Per-window diagnostic records MUST include receiver identity, runtime index, serial when available, device kind, runnable backend, assigned role, job identity, source profile or task, window frequency, sample rate, FFT size, averaging, number of detected segments, samples read, and dropped or short-read representation when detectable.
- **FR-029**: Timing telemetry MUST include tune, flush, read, transform, detection, persistence or database update, diagnostic logging, and total window timing where feasible.
- **FR-030**: Resource telemetry MUST include process identity, active device count, active role count, sample rate, CPU load when available, and resident memory when available.
- **FR-031**: Telemetry MUST make it practical to compare one, two, and three RTL operation on a Raspberry Pi 5 without relying on raw IQ capture from every receiver.
- **FR-032**: Persistence changes MUST be additive and migration-safe, with existing baseline, detection, scan update, diagnostic, and profile behavior remaining compatible.
- **FR-033**: If any provenance is initially diagnostic-only rather than stored in the main persistence model, the planning artifacts MUST call out that boundary and identify what is deferred.
- **FR-034**: SDRwatch MUST NOT add continuous raw IQ archive behavior as part of this feature.
- **FR-035**: SDRwatch MUST favor metadata, power summaries, event records, and bounded future evidence capture over unbounded receiver recording.
- **FR-036**: Operator-facing documentation MUST state that current scanner execution is RTL-native only and that Airspy, HackRF, and Soapy support are planned or future unless implemented later.
- **FR-037**: Documentation MUST describe capability tiers, manual role assignment, GUARD, ROVER, and REFERENCE semantics, identity warnings, Raspberry Pi 5 resource expectations, and the no-continuous-raw-IQ default.
- **FR-038**: Tests MUST cover hardware inventory tiers, missing and duplicate serials, index-only warnings, backend gating, manual role assignment, grouped and child job lifecycle, stale reservation cleanup, process reaping, startup reconciliation, concurrency around duplicate receiver assignment, telemetry provenance, timing fields, and existing FM, cross-sweep, diagnostic, and job-lifecycle regressions.

### Scope Boundaries

- Runnable scanner execution in this feature is limited to native RTL-SDR receivers, including RTL-SDR Blog V3 and RTL-SDR Blog V4 devices supported by the existing native path.
- Airspy R2, HackRF One, Soapy-backed devices, automatic role assignment, scheduler optimization, full signal fusion, full signal track or observation schema redesign, continuous raw IQ archive, low-level DSP rewrite, broad UI redesign, high-band hazard monitoring, and Airspy or HackRF characterization workflows are out of scope.
- REFERENCE is included only as a manual parked role with contextual telemetry; automatic environmental correction and confidence scoring are deferred.
- The existing operator workflow remains web UI to controller lifecycle to scanner backend. Direct scanner command checks are internal smoke tests only.

### Key Entities *(include if feature involves data)*

- **Hardware Inventory Entry**: Represents one detected receiver or planned hardware class with label, hardware kind, current index, serial, identity confidence, runnable status, backend support, busy state, assigned role, active job, and warnings.
- **Capability Tier**: Represents what the station can safely offer based on runnable RTL receiver count: Tier 0, Tier 1, Tier 2, or Tier 2+.
- **Receiver Role Assignment**: Represents the operator's manual binding of one receiver identity to GUARD, ROVER, or REFERENCE, including whether the binding is stable or session-scoped.
- **Role-Aware Run**: Represents a grouped operator action that may own multiple child jobs, each tied to one role and one physical receiver.
- **Child Scan Job**: Represents an individual scanner job with one receiver, one role, one task or profile, status, timestamps, and error state.
- **Scan Task**: Represents the configured work for a role, such as one guarded priority window, a rover sweep range, or a reference window.
- **Telemetry Record**: Represents diagnostic evidence for a job, role, receiver, timing sample, scan window, resource sample, or detection event.
- **Detection Provenance**: Represents the receiver, role, job, task/profile, and runtime identity context associated with emitted detections or scan updates.
- **Unsupported Hardware Class**: Represents detected or documented hardware that may be supported in the future but is not runnable in this feature.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: In acceptance tests, zero, one, two, and three runnable RTL receiver scenarios produce the correct Tier 0, Tier 1, Tier 2, and Tier 2+ capability reports.
- **SC-002**: In acceptance tests, every detected RTL inventory entry shows identity, runnable status, busy state, role assignment state, and warning state without requiring direct scanner command use.
- **SC-003**: With one runnable RTL, an operator can assign GUARD, start a guarded window, see the role-aware job state, stop it, and return the receiver to available state through the web/controller workflow.
- **SC-004**: With two runnable RTLs, an operator can start GUARD plus ROVER concurrently with two distinct physical receivers, and duplicate active receiver assignment is rejected in all tested concurrent start scenarios.
- **SC-005**: With three runnable RTLs, an operator can configure and view two GUARD roles plus one REFERENCE or ROVER role, with one visible child job per active role.
- **SC-006**: Unsupported Airspy, HackRF, and Soapy scanner start attempts are rejected before scanner execution begins in all backend-gating tests.
- **SC-007**: Diagnostic exports for completed role-aware scan windows include receiver, role, job, task/profile, sample, segment, and timing provenance for every field that is available on the platform, and explicitly mark unavailable fields.
- **SC-008**: Resource telemetry for role-aware runs includes active role count, active receiver count, process identity, CPU load when available, and resident memory when available, sufficient to compare one, two, and three RTL runs on Raspberry Pi 5.
- **SC-009**: Existing FM broadcast profile tests, cross-sweep persistence tests, diagnostic JSONL tests, and current job lifecycle tests pass unchanged or with only additive-field expectations.
- **SC-010**: Operator-facing documentation accurately distinguishes current RTL-native scanner execution from planned Airspy, HackRF, and Soapy support.

## Assumptions

- Manual role assignment is the first user experience; automatic role assignment and scheduler optimization will be specified later.
- Unique serial identity is the preferred durable receiver identity. Index-only identity is allowed only with warnings and session-scoped behavior unless the operator reconfirms it after restart.
- Role assignment persistence may use existing controller state or another additive persistence mechanism; a broad database redesign is not part of this feature.
- A grouped role-aware run may introduce an additive group or run identity while retaining individual child job records and status.
- Provenance should be added first to diagnostics, controller job state, scan updates, and other migration-safe records where feasible; full signal tracks and per-observation fusion are deferred.
- The Raspberry Pi 5 target has 4 GB RAM, local NVMe storage, active cooling, and finite USB, CPU, memory, and I/O capacity.
- Default role tasks should remain conservative and should not run every receiver at maximum sample rate by default.
- The existing control-plane and scanner implementation remain the foundation for this feature; this specification does not require a low-level DSP rewrite.
