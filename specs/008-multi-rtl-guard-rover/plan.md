# Implementation Plan: Hardware-Aware Multi-RTL Guard/Rover Mode

**Branch**: `007-cross-sweep-persistence-and-telemetry` | **Date**: 2026-06-14 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/008-multi-rtl-guard-rover/spec.md`

**Branch Hygiene Note**: The feature directory is intentionally `specs/008-multi-rtl-guard-rover` while the current checkout remains on `007-cross-sweep-persistence-and-telemetry`. Before implementation, create or switch to a stacked branch named `008-multi-rtl-guard-rover` from the current branch so this hardware-aware feature does not blur with the already-completed cross-sweep work.

## Summary

Implement an RTL-only hardware-aware watch mode that lets the operator see receiver inventory and capability tier, manually assign detected RTL-SDR receivers to GUARD, ROVER, or REFERENCE roles, and start role-aware jobs through the existing controller/web lifecycle. The design preserves current native RTL-SDR scanner execution, FM broadcast behavior, cross-sweep persistence, diagnostics, and the existing single-job `/api/jobs` contract while adding inventory, role assignment, grouped role-run status, provenance, and benchmark telemetry.

The implementation approach is additive: controller-owned inventory/role state and atomic receiver reservations first, scanner job metadata and diagnostic provenance second, then minimal control-page UI updates. Airspy, HackRF, Soapy, automatic scheduling, full signal fusion, continuous raw IQ capture, and broad schema rewrites remain out of scope.

## Technical Context

**Language/Version**: Python 3 project; no committed project version pin on this branch.

**Primary Dependencies**: Flask web app, server-rendered Jinja templates, controller HTTP API, scanner modules under `sdrwatch/`, NumPy-based DSP, native RTL-SDR driver integration, SQLite baseline store, diagnostic JSONL and bundle export.

**Storage**: Existing SQLite baseline tables plus controller `state.json`, controller lock files, controller job logs, diagnostic JSONL, and diagnostic bundle exports. This feature uses controller JSON state for role assignments and role runs, diagnostic JSONL for first-line provenance, and only additive SQLite columns where planning shows durable provenance is needed.

**Testing**: Pytest no-hardware tests first with fake device discovery, fake processes, fake locks, and diagnostics fixtures; web/controller route tests for operator workflow; hardware acceptance on Raspberry Pi 5 with one, two, and three RTL-SDR receivers remains required before full field confidence.

**Target Platform**: Raspberry Pi 5 with 4 GB RAM, 1 TB NVMe, active cooling, finite USB/CPU/RAM/I/O capacity, and local offline operation.

**Project Type**: Local web dashboard plus controller service plus internal scanner backend.

**Operator Workflow Surface**: SDRwatch operator-facing features MUST use the web UI and controller job lifecycle. Treat scanner CLI work as internal backend tooling unless the feature is explicitly scanner-only.

**Performance Goals**: Keep single-device FM broadcast behavior stable; support two to three concurrent RTL jobs conservatively; expose timing/resource telemetry sufficient to compare one, two, and three receiver runs on Pi 5; avoid unbounded diagnostic, capture, or database growth.

**Constraints**: RTL-only runnable scanner support via `rtlsdr_native`; no Airspy/HackRF/Soapy scanner execution; no continuous raw IQ capture; no Rust DSP rewrite; no full signal fusion; no broad database rewrite; preserve `/api/jobs` compatibility and existing diagnostic JSONL usefulness.

**Scale/Scope**: One local watch node with zero to three primary RTL receivers for this feature, while inventory and tiering tolerate three or more detected RTLs. Concurrent child jobs are bounded by assigned roles, locks, and explicit operator action.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Raspberry Pi First Reliability**: PASS. The plan targets Pi 5, keeps defaults conservative, avoids continuous IQ capture, and requires resource telemetry plus real-device validation.
- **II. Minimal Local Stack**: PASS. The feature stays inside existing Python, Flask/Jinja, SQLite, controller JSON state, and diagnostic files. No new frontend framework, cloud service, or infrastructure is introduced.
- **III. Stable Interfaces and Clean Layering**: PASS. The web UI remains the operator surface, the controller owns discovery, locks, role state, and process spawning, and the scanner owns DSP/detection/persistence. `/api/jobs` remains compatible.
- **IV. Adapter-Based Hardware and Honest RF Claims**: PASS. Runnable support is explicitly native RTL only. Unsupported hardware can appear only as planned/not runnable, and role/provenance metadata avoids unsupported emitter claims.
- **V. Migration-Safe, Verifiable Change**: PASS. The plan uses additive state and columns only, documents diagnostic-only provenance where applicable, and requires no-hardware tests plus web/controller and hardware validation.
- **Operator Acceptance Gate**: PASS. Validation flows are browser -> web API -> controller -> scanner. CLI checks are limited to backend smoke and regression checks.

## Project Structure

### Documentation (this feature)

```text
specs/008-multi-rtl-guard-rover/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- checklists/
|   `-- requirements.md
|-- contracts/
|   |-- hardware-inventory-contract.md
|   |-- role-assignment-contract.md
|   |-- role-run-contract.md
|   |-- job-status-additions-contract.md
|   `-- diagnostic-telemetry-contract.md
`-- tasks.md
```

### Source Code (repository root)

```text
sdrwatch-control.py
  Extend controller device discovery into hardware inventory, capability tier,
  serial/index identity warnings, role assignment state, role-run orchestration,
  atomic receiver lock acquisition, stale-lock cleanup, process reaping, and
  pre-spawn backend gating.

sdrwatch/cli.py
  Preserve `rtlsdr_native` as the only runnable scanner backend. Add only
  scanner metadata flags needed for role/job provenance if they can remain
  backward-compatible.

sdrwatch/sweep/runner.py
  Preserve native RTL source selection. Pass role/job/source-task provenance
  and active-role metadata to scanner telemetry without enabling other hardware.

sdrwatch/sweep/sweeper.py
  Keep existing sequential tune/read/FFT/detect/baseline/event loop. Add
  per-window timing, sample-count, short-read representation, and role/task
  provenance in diagnostics. Use existing narrow sweep configuration for GUARD
  and REFERENCE parked windows.

sdrwatch/util/detection_diagnostics.py
  Extend device telemetry, effective-parameter, and detection-window records
  with device identity, receiver role, role-run/job metadata, timing, samples,
  and resource fields with explicit unavailable markers.

sdrwatch/util/scan_logger.py
  Preserve JSONL behavior while measuring logger/jsonl write cost where feasible.

sdrwatch/baseline/store.py
  Add migration-safe nullable provenance columns where durable storage is chosen,
  especially scan update provenance. Avoid broad signal-track/observation schema
  work.

sdrwatch/baseline/persistence.py
  Preserve cross-sweep and FM persistence behavior. Carry provenance into
  persistence calls only where needed for additive storage and diagnostics.

sdrwatch_web/controller.py
  Add client wrappers for inventory, capability tier, role assignment, and
  role-run endpoints while preserving existing job wrappers.

sdrwatch_web/blueprints/api_jobs.py
  Preserve existing `/api/jobs` routes. Add minimal web API proxies for hardware
  inventory, role assignments, and role runs. Adjust active-job helpers so
  multi-job status does not collapse to one misleading active job.

sdrwatch_web/diagnostics.py
  Include new role/device/resource/timing fields in diagnostic bundle summaries
  and manifests without requiring log scraping.

templates/control.html
  Add a compact inventory/tier/role-assignment surface and role-aware job status.
  Remove single-active-job assumptions only where required for multi-RTL role
  operation. Avoid a broad UI redesign.

README.md and operator docs
  Correct current hardware support claims, document RTL-native-only scanner
  execution, capability tiers, role semantics, identity warnings, Pi 5 resource
  expectations, and no continuous raw IQ capture default.

tests/
  Add focused no-hardware tests for inventory, tiering, identity warnings,
  backend gating, role assignment, atomic locks, grouped role runs, telemetry,
  persistence provenance, and web/controller regressions. Keep existing FM and
  cross-sweep tests passing.
```

**Structure Decision**: Keep the feature in the existing single-repository architecture. The controller becomes the role/inventory coordinator, the scanner remains the one-device job executor, and the web UI presents role-aware orchestration without taking ownership of hardware or DSP.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Baseline Starting Point

- `sdrwatch-control.py` already discovers RTL devices, represents `Device` and `Job`, owns controller state, lock files, stale-lock cleanup, process reaping, and single-job start/stop/status.
- `discover_devices()` currently returns native RTL devices only, despite adjacent code and docs that mention other hardware.
- Current `Device.key` values are index-first, such as `rtl:0`, while serial metadata may be available but is not used as durable role identity.
- `sdrwatch/cli.py` and `sdrwatch/sweep/runner.py` already reject non-`rtlsdr_native` scanner backends, making backend gating mostly an explicit reporting and pre-spawn-controller concern.
- `sdrwatch/sweep/sweeper.py` already executes the sequential tune/read/PSD/detect/baseline/event loop and can model a parked GUARD or REFERENCE role with a narrow configured range.
- Cross-sweep persistence is per scanner process before database persistence, while persisted baseline rows are baseline-first and lack enough device/role provenance for concurrent receivers.
- `sdrwatch/util/detection_diagnostics.py`, `scan_logger.py`, and `sdrwatch_web/diagnostics.py` already provide structured diagnostic JSONL and diagnostic bundles, but lack per-window timing/resource and role provenance.
- `templates/control.html` and web job APIs currently assume one selected device and a single active job in several places.

## Technical Architecture

### Controller Changes

- Introduce a controller-side hardware inventory builder that merges detected RTL devices with lock state, active job state, role assignment state, backend support, identity confidence, and warning messages.
- Keep hardware inventory RTL-runnable-only for execution. Unsupported/planned hardware classes may be represented only as `runnable=false` and `support_state=planned` or `unsupported`.
- Derive stable receiver identity as:
  - `rtl:serial:<serial>` when a unique serial is present.
  - `rtl:index:<index>` as an unstable runtime fallback when serial is missing or duplicated.
  - Existing `rtl:<index>` remains accepted for compatibility and maps to the runtime index identity.
- Add role assignment state to controller persisted state:
  - Stable serial assignments can persist across controller restart.
  - Index-only assignments are session-scoped and cleared or require reconfirmation after restart.
  - Assignment records include role, role lane, device identity, runtime index, serial, identity confidence, warnings, timestamps, and assignment scope.
- Introduce grouped role-run records because GUARD+ROVER and two-GUARD+REFERENCE are one operator action with multiple child jobs. Child jobs remain normal `Job` records and keep individual visibility.
- Harden lock acquisition by replacing the current check-then-write lock path with an atomic create/claim operation around lock files, while retaining stale-lock cleanup and startup reconciliation.
- Reject unsupported backend/device starts before process spawn, including controller paths that might otherwise map `hackrf:` or future keys into scanner commands.
- Add role/job metadata to child job records and scanner args where backward-compatible: receiver role, role lane, role-run ID, source task, stable identity, serial, runtime index, active role count, and active device count.

### Scanner Runner Changes

- Preserve `rtlsdr_native` as the only accepted scanner backend.
- Preserve current CLI defaults and existing single-device smoke behavior.
- Add optional metadata-only flags if needed for diagnostics, such as receiver role, role-run ID, source task, device serial, and runtime index. These flags must not change detection behavior.
- Keep source selection native RTL only. Serial-based opening may be used only after compatibility planning confirms the current driver can do it safely; otherwise controller identity maps to the runtime index passed to the existing runner.

### Scan Behavior Mapping

- **GUARD**: Use existing scanner loop with a narrow configured start/stop/step centered on the priority window. The runner still tunes and reads per window, but the configured range should produce one stable parked window and avoid rover-like broad sweeps.
- **ROVER**: Use existing sequential sweep configuration over lower-priority ranges.
- **REFERENCE**: Use the GUARD-style parked-window shape with a reference/noise/stable-signal task label and diagnostic context; no automatic correction, scoring, or fusion.
- Keep FM Broadcast profile behavior untouched. Role tasks may choose profile/task parameters, but FM Validation remains its current explicit operator path.

### Diagnostic and Telemetry Changes

- Extend `device_telemetry`, `effective_parameters`, and `detection_window` records with role-aware provenance.
- Add per-window timing measurements around tune, flush, read, transform, detect, baseline/persistence/database update, diagnostic JSONL logging, and total window duration.
- Include sample accounting: requested samples, samples read, short-read indicator when detectable, and dropped-read field when the driver can expose it. Use `null` plus `unavailable_fields` when not available.
- Add process/resource telemetry records or summary fields for PID, active device count, active role count, CPU load when available, RSS memory when available, sample rate, and role-run ID.
- Keep diagnostic JSONL bounded and bundle summaries aggregate-oriented so Pi 5 I/O is not dominated by logging.

### API/UI Changes

- Add web/controller surfaces for:
  - Hardware inventory with capability tier.
  - Role assignment list, set, and clear.
  - Role-run start, status/list, detail, and stop.
- Preserve existing `/api/jobs` payload compatibility: `{device_key, label, baseline_id, params}` remains valid for current workflows.
- Add only additive fields to job status objects, such as `receiver_role`, `role_lane`, `role_run_id`, `source_task`, `device_identity`, and `identity_warning`.
- Update the control page with a compact inventory/tier strip, role assignment controls, and role-run status. Keep existing scan controls and presets intact.
- Replace single-active-job assumptions only where multi-role status and stop/export controls require it. Existing active-job endpoints may continue to return a primary or most recent job for compatibility, but role-run views must not depend on them.

### Persistence Changes

- Controller role assignments and role-run records live in controller JSON state first.
- Diagnostic JSONL stores full role/device/job/task/timing/resource provenance in the first implementation.
- Add nullable SQLite columns only where low-risk and directly useful:
  - Prefer `scan_updates`: `receiver_role`, `device_key`, `device_serial`, `device_index`, `job_id`, `role_run_id`, `source_profile`, `source_task`.
  - Consider `baseline_detections` only for last-seen provenance fields, not as a full observation table.
- Do not add signal_tracks, observations, fusion tables, or destructive migrations in this feature.
- If baseline detections remain baseline-scoped for first implementation, document that multi-receiver provenance is diagnostic-first and that future fusion needs a separate model.

### Documentation Changes

- Correct README claims that imply current Soapy scanner execution or non-RTL runtime support.
- Document `rtlsdr_native` as the only runnable scanner backend for this feature.
- Mark Airspy, HackRF, and Soapy as planned/future unless implemented later.
- Add operator docs for capability tiers, manual roles, GUARD/ROVER/REFERENCE semantics, identity warnings, Pi 5 resource expectations, and no continuous raw IQ capture.

## Phase 0 Research

Research decisions are captured in [research.md](./research.md). Key outcomes:

- Treat grouped role runs as an additive orchestration layer over existing child jobs.
- Use controller state for role assignment durability and role-run coordination.
- Make receiver identity serial-first with index fallback warnings.
- Harden device locking atomically before relying on concurrent multi-RTL starts.
- Keep backend support explicit and RTL-only.
- Store full provenance in diagnostics first, with minimal additive database columns for scan updates.

## Phase 1 Design

Design artifacts:

- [data-model.md](./data-model.md)
- [contracts/hardware-inventory-contract.md](./contracts/hardware-inventory-contract.md)
- [contracts/role-assignment-contract.md](./contracts/role-assignment-contract.md)
- [contracts/role-run-contract.md](./contracts/role-run-contract.md)
- [contracts/job-status-additions-contract.md](./contracts/job-status-additions-contract.md)
- [contracts/diagnostic-telemetry-contract.md](./contracts/diagnostic-telemetry-contract.md)
- [quickstart.md](./quickstart.md)

## Migration Strategy

- Use additive-only migrations.
- Keep controller state backward-compatible by accepting older `state.json` files without role assignment or role-run keys.
- Add controller state keys:
  - `role_assignments`: map of role lane to assignment record.
  - `role_runs`: map of role-run ID to grouped run record.
  - `role_assignment_session_epoch` or equivalent session marker for clearing index-only assignments after restart.
- Add nullable scan provenance columns only if implementation tasks confirm scanner writes can populate them without disturbing existing readers:
  - `receiver_role TEXT`
  - `device_key TEXT`
  - `device_serial TEXT`
  - `device_index INTEGER`
  - `job_id TEXT`
  - `role_run_id TEXT`
  - `source_profile TEXT`
  - `source_task TEXT`
- Durable provenance in first implementation:
  - Controller role assignments and role runs in controller state.
  - Child job metadata in controller state.
  - Diagnostic JSONL and bundle manifests.
  - Scan update provenance if additive columns are implemented.
- Diagnostic-only provenance in first implementation:
  - Per-window timing breakdown.
  - Per-window samples/short-read/dropped-read details.
  - Resource telemetry fields.
  - Detection-level receiver provenance if adding it to `baseline_detections` is deferred.
- Deferred:
  - Full `signal_tracks` table.
  - Full per-observation table.
  - Cross-receiver fusion, confidence scoring, and automatic correction.

## Test Strategy

- **Inventory and capability**: zero/one/two/three RTL fixtures, missing serial, duplicate serial, index-only warning, unique serial identity, Tier 0/Tier 1/Tier 2/Tier 2+.
- **Backend gating**: `rtlsdr_native` accepted; Airspy/HackRF/Soapy keys rejected before spawn; existing scanner CLI rejection remains covered.
- **Role assignment**: assign GUARD, ROVER, REFERENCE; clear assignment; stable serial persistence; index-only session scoping; duplicate active assignment rejection.
- **Lock/lifecycle**: same-device concurrent starts do not both succeed; distinct-device starts can proceed; stale lock cleanup; startup reconciliation; process reaping releases locks; failed spawn releases lock or reports state clearly; stop status does not drift misleadingly after reaper completion.
- **Role runs**: one GUARD job; GUARD+ROVER jobs; two GUARD plus REFERENCE/ROVER with three devices; grouped stop; child failure reflected as degraded/failed group status.
- **Scanner behavior**: GUARD and REFERENCE narrow-window tasks use existing scanner loop; ROVER uses existing sweep; no FM Broadcast regression; no cross-sweep persistence regression.
- **Telemetry**: diagnostic JSONL includes role/device/job/task provenance, sample counts, unavailable fields, timing fields, process identity, active device/role counts, CPU/RSS when available.
- **Persistence**: additive migration against existing SQLite files; scan update provenance writes when enabled; baseline detection behavior remains compatible.
- **Web/API**: inventory, role assignment, role-run routes, auth behavior when `SDRWATCH_CONTROL_TOKEN` is enabled, existing `/api/jobs` payload compatibility, multi-job status UI, diagnostic export for active/recent/finished jobs.
- **Docs**: README no longer claims runnable Soapy/non-RTL scanner support; operator docs describe tiers, roles, warnings, and Pi 5 constraints.

## Implementation Sequence

1. **Slice 1: Inventory, Capability Tier, Backend Gating**
   - Add controller inventory builder and capability tier.
   - Expose web/controller inventory surface.
   - Make runnable backend reporting explicit.
   - Reject unsupported starts before spawn.
   - Tests: inventory tiers, backend gating, current job compatibility.

2. **Slice 2: Stable Identity, Warnings, Role Assignment State**
   - Add serial-first identity model and warning generation.
   - Add role assignment set/list/clear state.
   - Persist only stable serial assignments; session-scope index fallbacks.
   - Tests: missing/duplicate serials, index fallback, assignment persistence.

3. **Slice 3: Lock/Lifecycle Hardening and Concurrency Tests**
   - Make lock acquisition atomic.
   - Preserve stale-lock cleanup and startup reconciliation.
   - Add fake-process tests for reaper/stop/failure paths.
   - Tests: concurrent same-device start, distinct-device starts, stale locks.

4. **Slice 4: One-Device GUARD Role Path**
   - Start a single GUARD child job from a role assignment.
   - Pass role/task metadata to job and diagnostics.
   - Use narrow-window scanner params with existing scanner behavior.
   - Tests: one GUARD job, stop/release, FM regression.

5. **Slice 5: Two-Device GUARD + ROVER Grouped Run**
   - Add role-run state and start/stop/status endpoints.
   - Start one GUARD and one ROVER child job.
   - Reflect degraded state when a child exits or fails.
   - Tests: grouped start/stop, child failure, status and locks.

6. **Slice 6: Telemetry and Provenance Expansion**
   - Add per-window timing, sample accounting, resource telemetry, and role/device/job provenance to diagnostic JSONL.
   - Add minimal scan update provenance columns if implementation confirms low risk.
   - Tests: diagnostic JSONL fields, unavailable fields, bundle summaries.

7. **Slice 7: Three-Device Two-GUARD plus REFERENCE/ROVER Support**
   - Add role lanes for friendly GUARD, watchlist GUARD, and REFERENCE/ROVER.
   - Ensure three distinct physical receiver assignment.
   - Tests: three-device role run, duplicate prevention, reference telemetry.

8. **Slice 8: Minimal UI/Docs Polish and Regression Hardening**
   - Add compact control-page inventory/tier/role-run UI.
   - Fix README/operator docs drift.
   - Run no-hardware regression suite and document Pi 5 hardware acceptance path.

## Risks and Mitigations

- **Unstable RTL index identity**: Prefer unique serial identities; mark index-only identity unstable; session-scope index assignments.
- **Duplicate or missing serials**: Surface warnings, avoid durable assignment, and require operator reconfirmation after restart.
- **Non-atomic lock acquisition**: Harden lock files with atomic create/claim semantics before multi-job orchestration.
- **Stale process/job state**: Keep startup reconciliation, lazy job refresh, stale-lock cleanup, and reaper release tests.
- **SQLite/write contention**: Keep DB changes minimal, prefer diagnostic JSONL for high-frequency telemetry, and avoid writing every detail to baseline tables.
- **Pi 5 USB/CPU pressure**: Keep conservative defaults, no all-radios-max-rate mode, and add timing/resource telemetry before optimizing.
- **Diagnostic JSONL overhead**: Aggregate where possible, record logger timing, keep bundle tails bounded, and avoid raw IQ logging.
- **UI single-active-job assumptions**: Preserve compatibility endpoints, but add explicit role-run status and lists for multi-job operation.
- **Unsupported hardware confusion**: Make runnable/planned/unsupported status explicit in inventory and docs; pre-spawn reject unsupported starts.
- **Accidental FM/cross-sweep regression**: Keep detector behavior unchanged for role metadata; run FM Broadcast, cross-sweep, diagnostic, and `/api/jobs` regression tests after each slice.

## Post-Design Constitution Check

- **Raspberry Pi reliability remains central**: PASS. Resource telemetry, bounded logging, conservative defaults, and hardware acceptance are planned.
- **Minimal local stack remains intact**: PASS. No new service, cloud dependency, or frontend framework is introduced.
- **Layering remains clean**: PASS. Controller owns roles/locks/processes; scanner owns DSP; web owns operator presentation and proxies.
- **Hardware claims remain honest**: PASS. Only native RTL execution is runnable; Airspy/HackRF/Soapy are explicitly non-runnable future classes.
- **Migration remains safe**: PASS. Controller state and diagnostic-first provenance are primary; SQLite changes are nullable and additive only.
- **Operator validation remains GUI/controller focused**: PASS. Quickstart and tests prioritize web/controller lifecycle, with CLI only for backend smoke/regression.
