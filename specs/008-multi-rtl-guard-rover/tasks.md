# Tasks: Hardware-Aware Multi-RTL Guard/Rover Mode

**Input**: Design documents from `specs/008-multi-rtl-guard-rover/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Required. This feature must use no-hardware fake-device/fake-process tests before or alongside implementation, plus regression checks for existing `/api/jobs`, FM Broadcast, cross-sweep persistence, diagnostic JSONL, and web/controller behavior.

**Organization**: Tasks follow the plan's slice order. Story labels map to spec user stories: [US1] inventory/capability, [US2] one-RTL GUARD, [US3] two-RTL GUARD+ROVER, [US4] three-RTL two-GUARD plus REFERENCE/ROVER, [US5] benchmark telemetry.

**Branch Hygiene**: Feature artifacts were created on `007-cross-sweep-persistence-and-telemetry`. Before implementation code starts, create or switch to stacked branch `008-multi-rtl-guard-rover` from the current branch tip. Do not create a new spec and do not move this feature directory.

**Feature Boundary**: Runnable scanner execution remains RTL-only via `rtlsdr_native`. Airspy, HackRF, SoapySDR, and other non-RTL hardware may appear only as unsupported/planned classes and must not become runnable scanner choices.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Parallelizable only when the task touches different files and does not depend on an incomplete behavior.
- **[Story]**: Required for user-story slice tasks.
- All descriptions include concrete file paths.

---

## Phase 1: Setup and Branch Hygiene

**Purpose**: Prepare the implementation branch and confirm Spec Kit artifacts without changing runtime behavior.

- [X] T001 Create or switch to stacked branch `008-multi-rtl-guard-rover` from repository root `C:\Users\User\SDRwatch`
- [X] T002 Verify `.specify/feature.json` points to `specs/008-multi-rtl-guard-rover` in `C:\Users\User\SDRwatch\.specify\feature.json`
- [X] T003 Verify `AGENTS.md` points to `specs/008-multi-rtl-guard-rover/plan.md` in `C:\Users\User\SDRwatch\AGENTS.md`
- [X] T004 Record the partial role-run startup convention in `specs/008-multi-rtl-guard-rover/contracts/role-run-contract.md`: if at least one child starts and another fails, status is `degraded` with child errors; if no child starts, return a conventional error; do not use HTTP 207 unless existing project tests already establish it
- [X] T005 [P] Add `tests/helpers_multi_rtl.py` with fake RTL devices, duplicate serial fixtures, missing serial fixtures, fake lock owner helpers, and fake process/reaper stubs
- [X] T006 [P] Add `tests/test_multi_rtl_branch_hygiene.py` to assert the feature directory and plan references remain `specs/008-multi-rtl-guard-rover`

**Checkpoint**: Branch and Spec Kit artifact hygiene are explicit before code implementation begins.

---

## Phase 2: Foundational Controller State and Test Fixtures

**Purpose**: Add shared no-hardware fixtures and backward-compatible controller state support that every slice depends on.

- [X] T007 [P] Add no-hardware controller import helpers for `sdrwatch-control.py` in `tests/helpers_control.py` without changing existing helper behavior
- [X] T008 Add controller state migration tests for missing `role_assignments`, `role_runs`, and session marker keys in `tests/test_multi_rtl_state.py`
- [X] T009 Add child job metadata serialization tests for `receiver_role`, `role_lane`, `role_run_id`, `source_task`, `device_identity`, `device_serial`, `device_index`, `identity_confidence`, `active_device_count`, and `active_role_count` in `tests/test_multi_rtl_state.py`
- [X] T010 Implement backward-compatible default `role_assignments`, `role_runs`, and session marker loading in `sdrwatch-control.py`
- [X] T011 Add additive role-aware fields to the controller `Job` representation and persistence serialization in `sdrwatch-control.py`
- [X] T012 Add state persistence helpers for role assignments, role runs, and role-aware child job metadata in `sdrwatch-control.py`
- [X] T013 Run foundational state tests with `python -m pytest -q tests/test_multi_rtl_state.py tests/test_control_diagnostics_mode.py`

**Checkpoint**: Controller state can read older files, persist new role metadata, and keep existing diagnostics-mode state behavior.

---

## Phase 3: Slice 1 - Inventory, Capability Tier, Backend Gating (US1)

**Goal**: The operator can see detected RTL hardware, runnable capability tier, busy state, and unsupported hardware status without enabling unsupported backends.

**Independent Test**: Fake zero/one/two/three RTL inventories produce Tier 0/Tier 1/Tier 2/Tier 2+ through controller and web inventory surfaces; unsupported starts are rejected before spawn.

### Tests First

- [X] T014 [US1] Add zero/one/two/three RTL inventory tier tests in `tests/test_multi_rtl_inventory.py`
- [X] T015 [US1] Add missing serial, duplicate serial, and index-only warning inventory tests in `tests/test_multi_rtl_inventory.py`
- [X] T016 [US1] Add unsupported Airspy/HackRF/Soapy inventory class tests with `runnable=false` in `tests/test_multi_rtl_inventory.py`
- [X] T017 [P] [US1] Add backend gating tests that reject `hackrf:`, `airspy:`, and `soapy:` starts before `subprocess.Popen` in `tests/test_multi_rtl_backend_gating.py`
- [X] T018 [US1] Add controller route contract tests for `GET /hardware/inventory` and legacy `/devices` compatibility in `tests/test_multi_rtl_inventory.py`
- [X] T019 [P] [US1] Add web proxy tests for `GET /api/hardware/inventory` and existing `/ctl/devices` compatibility in `tests/test_multi_rtl_web_api.py`

### Implementation

- [X] T020 [US1] Implement hardware inventory entry construction with label, runtime index, serial, identity fields, runnable backend, busy/locked state, active job ID, assigned role, and warnings in `sdrwatch-control.py`
- [X] T021 [US1] Implement capability tier derivation for Tier 0, Tier 1, Tier 2, and Tier 2+ in `sdrwatch-control.py`
- [X] T022 [US1] Add unsupported/planned hardware class reporting with `runnable=false` and no runnable backend in `sdrwatch-control.py`
- [X] T023 [US1] Add explicit pre-spawn backend gating for unsupported device keys and backends in `sdrwatch-control.py`
- [X] T024 [US1] Add controller route `GET /hardware/inventory` in `sdrwatch-control.py`
- [X] T025 [US1] Preserve legacy `/devices` response compatibility while allowing additive inventory fields in `sdrwatch-control.py`
- [X] T026 [US1] Add inventory and capability wrapper methods in `sdrwatch_web/controller.py`
- [X] T027 [US1] Add web proxy route `GET /api/hardware/inventory` in `sdrwatch_web/blueprints/api_jobs.py`
- [X] T028 [US1] Add token-auth coverage for the web inventory proxy in `tests/test_multi_rtl_web_api.py`

### Slice 1 Acceptance and Regression

- [X] T029 [US1] Run `python -m pytest -q tests/test_multi_rtl_inventory.py tests/test_multi_rtl_backend_gating.py tests/test_multi_rtl_web_api.py`
- [X] T030 [US1] Run existing compatibility checks `python -m pytest -q tests/test_control_page_scan_settings.py tests/test_web_diagnostics_bundle.py`
- [X] T031 [US1] Confirm unsupported Airspy/HackRF/Soapy scanner starts report `spawned=false` in `tests/test_multi_rtl_backend_gating.py`

---

## Phase 4: Slice 2 - Stable Identity, Warnings, Role Assignment State (US1, US2)

**Goal**: The operator can manually assign GUARD, ROVER, or REFERENCE roles to runnable RTLs using stable serial identity where possible, with clear warnings for unstable identity.

**Independent Test**: Role assignment set/list/clear works through controller and web APIs, duplicate active assignment is rejected, serial assignments persist, and index-only assignments are session-scoped.

### Tests First

- [X] T032 [US1] Add serial-first identity resolution tests for unique serials in `tests/test_multi_rtl_identity.py`
- [X] T033 [US1] Add missing serial, duplicate serial, and index-only identity warning tests in `tests/test_multi_rtl_identity.py`
- [X] T034 [US2] Add role assignment set/list/clear controller tests for GUARD, ROVER, and REFERENCE in `tests/test_multi_rtl_roles.py`
- [X] T035 [P] [US2] Add role assignment web proxy tests for `GET`, `PUT`, and `DELETE /api/role-assignments/{role_lane}` in `tests/test_multi_rtl_web_api.py`
- [X] T036 [US2] Add persistence tests proving stable serial assignments survive restart and index-only assignments are invalidated or require reconfirmation after restart in `tests/test_multi_rtl_roles.py`
- [X] T037 [US2] Add duplicate active assignment rejection tests in `tests/test_multi_rtl_roles.py`

### Implementation

- [X] T038 [US1] Implement `rtl:serial:<serial>` stable identity and `rtl:index:<index>` unstable fallback generation in `sdrwatch-control.py`
- [X] T039 [US1] Implement duplicate serial and missing serial warning generation in `sdrwatch-control.py`
- [X] T040 [US2] Implement controller role assignment set/list/clear helpers with `role_assignments` state in `sdrwatch-control.py`
- [X] T041 [US2] Implement assignment scope rules for persistent stable serial assignments and session-scoped index-only assignments in `sdrwatch-control.py`
- [X] T042 [US2] Add controller routes `GET /role-assignments`, `PUT /role-assignments/{role_lane}`, and `DELETE /role-assignments/{role_lane}` in `sdrwatch-control.py`
- [X] T043 [US2] Add web client wrappers for role assignment list/set/clear in `sdrwatch_web/controller.py`
- [X] T044 [US2] Add web proxy routes `GET /api/role-assignments`, `PUT /api/role-assignments/{role_lane}`, and `DELETE /api/role-assignments/{role_lane}` in `sdrwatch_web/blueprints/api_jobs.py`
- [X] T045 [US2] Add duplicate active assignment and identity-warning error payload handling in `sdrwatch-control.py`

### Slice 2 Acceptance and Regression

- [X] T046 [US2] Run `python -m pytest -q tests/test_multi_rtl_identity.py tests/test_multi_rtl_roles.py tests/test_multi_rtl_web_api.py`
- [X] T047 [US2] Run existing job payload regression `python -m pytest -q tests/test_control_page_scan_settings.py tests/test_web_diagnostics_bundle.py`
- [X] T048 [US2] Confirm existing `/api/jobs` request shape remains `{device_key, label, baseline_id, params}` in `tests/test_control_page_scan_settings.py`

---

## Phase 5: Slice 3 - Lock/Lifecycle Hardening and Concurrency Tests (US2, US3)

**Goal**: Concurrent requests cannot claim the same physical receiver twice, and existing stale-lock cleanup, startup reconciliation, process reaping, and stop behavior remain intact.

**Independent Test**: Fake-process tests prove same-device concurrent starts race safely, distinct-device starts can proceed, stale locks are cleaned, reaping releases locks, and stop status stays accurate.

### Tests First

- [X] T049 [US2] Add atomic lock acquisition tests for concurrent same-device starts in `tests/test_multi_rtl_lifecycle.py`
- [X] T050 [US3] Add distinct-device concurrent start tests for `rtl:0` and `rtl:1` in `tests/test_multi_rtl_lifecycle.py`
- [X] T051 [US2] Add stale lock cleanup tests for known and unknown owners in `tests/test_multi_rtl_lifecycle.py`
- [X] T052 [US2] Add startup reconciliation tests for dead persisted job PIDs in `tests/test_multi_rtl_lifecycle.py`
- [X] T053 [US2] Add process reaper tests proving lock release and terminal status updates in `tests/test_multi_rtl_lifecycle.py`
- [X] T054 [US2] Add stop-job tests preventing misleading status drift after reaper completion in `tests/test_multi_rtl_lifecycle.py`

### Implementation

- [X] T055 [US2] Replace check-then-write device lock claiming with atomic create/claim behavior in `sdrwatch-control.py`
- [X] T056 [US2] Preserve stale-lock cleanup semantics while using atomic lock acquisition in `sdrwatch-control.py`
- [X] T057 [US2] Preserve startup reconciliation release behavior for role-aware and legacy jobs in `sdrwatch-control.py`
- [X] T058 [US2] Preserve process reaper lock release and terminal status update behavior for role-aware and legacy jobs in `sdrwatch-control.py`
- [X] T059 [US2] Preserve stop-job terminate/kill behavior while preventing role-aware group status drift in `sdrwatch-control.py`
- [X] T060 [US2] Add lock owner metadata for `device_identity`, runtime `device_key`, `job_id`, and `role_run_id` in `sdrwatch-control.py`

### Slice 3 Acceptance and Regression

- [X] T061 [US2] Run `python -m pytest -q tests/test_multi_rtl_lifecycle.py`
- [X] T062 [US2] Run existing single-job lifecycle regressions `python -m pytest -q tests/test_control_diagnostics_mode.py tests/test_web_diagnostics_bundle.py`
- [X] T063 [US2] Confirm failed pre-spawn starts release or never acquire locks in `tests/test_multi_rtl_lifecycle.py`

---

## Phase 6: Slice 4 - One-Device GUARD Role Path (US2)

**Goal**: With one runnable RTL, the operator can assign GUARD and start one guarded narrow-window job without changing existing scanner detection behavior.

**Independent Test**: A GUARD role run starts one child job with role metadata, narrow-window scanner parameters, lock ownership, diagnostics path, and clean stop/release behavior.

### Tests First

- [X] T064 [US2] Add one-device GUARD role-run start tests in `tests/test_multi_rtl_guard.py`
- [X] T065 [US2] Add GUARD narrow-window parameter mapping tests in `tests/test_multi_rtl_guard.py`
- [X] T066 [US2] Add child job metadata tests for one GUARD job in `tests/test_multi_rtl_guard.py`
- [X] T067 [P] [US2] Add scanner metadata flag passthrough tests in `tests/test_control_fm_validation.py`
- [X] T068 [US2] Add FM Broadcast and cross-sweep non-regression selection tests for GUARD metadata-only changes in `tests/test_multi_rtl_guard.py`

### Implementation

- [X] T069 [US2] Add optional scanner metadata arguments for role/job/source-task provenance in `sdrwatch/cli.py`
- [X] T070 [US2] Pass role/job/source-task metadata through runner setup without changing source selection in `sdrwatch/sweep/runner.py`
- [X] T071 [US2] Add role-aware child job command construction for GUARD narrow-window tasks in `sdrwatch-control.py`
- [X] T072 [US2] Add single-role-run start path for one GUARD child job in `sdrwatch-control.py`
- [X] T073 [US2] Ensure GUARD child jobs still use `rtlsdr_native` and reject unsupported backends before spawn in `sdrwatch-control.py`
- [X] T074 [US2] Include GUARD role metadata in job status responses in `sdrwatch-control.py`

### Slice 4 Acceptance and Regression

- [X] T075 [US2] Run `python -m pytest -q tests/test_multi_rtl_guard.py tests/test_control_fm_validation.py`
- [X] T076 [US2] Run FM and cross-sweep regressions `python -m pytest -q tests/test_cross_sweep_persistence.py tests/test_effective_parameter_manifest.py tests/test_device_telemetry.py`
- [X] T077 [US2] Confirm scanner source selection remains native RTL only in `sdrwatch/sweep/runner.py`

---

## Phase 7: Slice 5 - Two-Device GUARD + ROVER Grouped Run (US3)

**Goal**: With two runnable RTLs, the operator can start GUARD and ROVER child jobs concurrently through one grouped role run.

**Independent Test**: A grouped role run starts one GUARD and one ROVER child job on distinct physical receivers, stops both cleanly, and reports degraded status when a child fails.

### Tests First

- [X] T078 [US3] Add role-run contract tests for `POST /role-runs`, `GET /role-runs`, `GET /role-runs/{role_run_id}`, and `DELETE /role-runs/{role_run_id}` in `tests/test_multi_rtl_role_runs.py`
- [X] T079 [P] [US3] Add web proxy tests for `POST /api/role-runs`, `GET /api/role-runs`, `GET /api/role-runs/{role_run_id}`, and `DELETE /api/role-runs/{role_run_id}` in `tests/test_multi_rtl_web_api.py`
- [X] T080 [US3] Add GUARD+ROVER grouped start tests with two distinct serial identities in `tests/test_multi_rtl_role_runs.py`
- [X] T081 [US3] Add serial refresh-before-start tests resolving `rtl:serial:<serial>` assignments to current runtime indexes in `tests/test_multi_rtl_role_runs.py`
- [X] T082 [US3] Add rejected-start tests for missing serial, duplicated serial, ambiguous identity, and unreconfirmed index-only assignment after restart in `tests/test_multi_rtl_role_runs.py`
- [X] T083 [US3] Add partial startup tests: one child started plus one failed becomes `degraded`; no child started returns conventional error in `tests/test_multi_rtl_role_runs.py`
- [X] T084 [US3] Add grouped stop tests proving both child jobs are stopped and locks released in `tests/test_multi_rtl_role_runs.py`
- [X] T085 [US3] Add child-direct-stop tests proving parent role-run status becomes degraded or terminal in `tests/test_multi_rtl_role_runs.py`

### Implementation

- [X] T086 [US3] Implement role-run state creation, list, detail, status refresh, and terminal-state helpers in `sdrwatch-control.py`
- [X] T087 [US3] Implement refresh-inventory-before-role-run-start and serial-to-current-index resolution in `sdrwatch-control.py`
- [X] T088 [US3] Acquire receiver locks only after current physical receiver resolution in `sdrwatch-control.py`
- [X] T089 [US3] Implement GUARD+ROVER child job startup with distinct physical receiver validation in `sdrwatch-control.py`
- [X] T090 [US3] Implement partial startup convention with `degraded` role-run status and child-level errors in `sdrwatch-control.py`
- [X] T091 [US3] Implement grouped stop that stops all non-terminal child jobs and releases locks in `sdrwatch-control.py`
- [X] T092 [US3] Update child job stop/reaper paths to refresh parent role-run health in `sdrwatch-control.py`
- [X] T093 [US3] Add controller routes `POST /role-runs`, `GET /role-runs`, `GET /role-runs/{role_run_id}`, and `DELETE /role-runs/{role_run_id}` in `sdrwatch-control.py`
- [X] T094 [US3] Add web client wrappers for role-run start/list/detail/stop in `sdrwatch_web/controller.py`
- [X] T095 [US3] Add web proxy routes for `/api/role-runs` and `/api/role-runs/{role_run_id}` in `sdrwatch_web/blueprints/api_jobs.py`

### Slice 5 Acceptance and Regression

- [X] T096 [US3] Run `python -m pytest -q tests/test_multi_rtl_role_runs.py tests/test_multi_rtl_lifecycle.py tests/test_multi_rtl_web_api.py`
- [X] T097 [US3] Run existing job API regressions `python -m pytest -q tests/test_control_page_scan_settings.py tests/test_web_diagnostics_bundle.py`
- [X] T098 [US3] Confirm `/api/jobs/active` compatibility remains intact while role-run UI uses role-run status in `sdrwatch_web/blueprints/api_jobs.py`

---

## Phase 8: Slice 6 - Telemetry and Provenance Expansion (US5)

**Goal**: Diagnostic JSONL and bundles include role/device/job/task provenance, per-window timing, sample accounting, resource telemetry, and explicit unavailable fields.

**Independent Test**: Fake scanner and bundle tests show role-aware diagnostic records are sufficient to benchmark one, two, and three RTL jobs without continuous raw IQ capture.

### Tests First

- [X] T099 [US5] Add diagnostic record tests for common role/device/job/task provenance fields in `tests/test_multi_rtl_telemetry.py`
- [X] T100 [US5] Add timing field tests for `tune_ms`, `flush_ms`, `read_ms`, `fft_ms`, `detect_ms`, `db_update_ms`, `jsonl_ms`, and `total_window_ms` in `tests/test_multi_rtl_telemetry.py`
- [X] T101 [US5] Add sample accounting tests for `samples_requested`, `samples_read`, `short_read`, `dropped_reads`, and `unavailable_fields` in `tests/test_multi_rtl_telemetry.py`
- [X] T102 [US5] Add resource telemetry tests for `pid`, CPU load, RSS memory, active device count, and active role count in `tests/test_multi_rtl_telemetry.py`
- [X] T103 [P] [US5] Add diagnostic bundle summary tests for roles, devices, jobs, role-run IDs, timing availability, resource availability, and missing fields in `tests/test_web_diagnostics_bundle.py`
- [X] T104 [P] [US5] Add additive scan-update provenance migration tests in `tests/test_multi_rtl_persistence.py`

### Implementation

- [X] T105 [US5] Extend `build_device_telemetry_snapshot` with role/device/job/run provenance in `sdrwatch/util/detection_diagnostics.py`
- [X] T106 [US5] Extend `build_effective_parameter_manifest` with source task, receiver role, role lane, role-run ID, stable identity, serial, runtime index, and runnable backend in `sdrwatch/util/detection_diagnostics.py`
- [X] T107 [US5] Extend `build_window_record` with role/device/job/task provenance, sample accounting, timing fields, and unavailable field handling in `sdrwatch/util/detection_diagnostics.py`
- [X] T108 [US5] Measure logger/jsonl write timing without changing JSONL semantics in `sdrwatch/util/scan_logger.py`
- [X] T109 [US5] Add tune, flush, read, FFT, detect, DB update, logger, and total window timing collection in `sdrwatch/sweep/sweeper.py`
- [X] T110 [US5] Add sample requested/read and short-read representation in `sdrwatch/sweep/sweeper.py`
- [X] T111 [US5] Add process/resource telemetry records with unavailable field handling in `sdrwatch/sweep/runner.py`
- [X] T112 [US5] Update diagnostic bundle summaries for role/device/job/timing/resource fields in `sdrwatch_web/diagnostics.py`
- [X] T113 [US5] Add nullable scan update provenance columns only in `sdrwatch/baseline/store.py`: `receiver_role`, `device_key`, `device_serial`, `device_index`, `job_id`, `role_run_id`, `source_profile`, and `source_task`
- [X] T114 [US5] Populate scan update provenance from scanner args when available without changing baseline detection matching in `sdrwatch/baseline/events.py` and `sdrwatch/sweep/sweeper.py`
- [X] T115 [US5] Document any provenance that remains diagnostic-only in `specs/008-multi-rtl-guard-rover/quickstart.md`

### Slice 6 Acceptance and Regression

- [X] T116 [US5] Run `python -m pytest -q tests/test_multi_rtl_telemetry.py tests/test_multi_rtl_persistence.py tests/test_web_diagnostics_bundle.py`
- [X] T117 [US5] Run diagnostic regressions `python -m pytest -q tests/test_device_telemetry.py tests/test_effective_parameter_manifest.py tests/test_cross_sweep_persistence.py`
- [X] T118 [US5] Confirm no continuous raw IQ capture paths were added in `sdrwatch/sweep/runner.py`, `sdrwatch/sweep/sweeper.py`, and `README.md`

---

## Phase 9: Slice 7 - Three-Device Two-GUARD plus REFERENCE/ROVER Support (US4)

**Goal**: With three runnable RTLs, the operator can run friendly GUARD, watchlist GUARD, and REFERENCE or ROVER with one physical receiver per child job.

**Independent Test**: Three fake RTL devices can be assigned to two GUARD lanes and one REFERENCE/ROVER lane, started as a role run, stopped cleanly, and inspected with role-specific diagnostics.

### Tests First

- [X] T119 [US4] Add three-RTL capability and role lane tests for `guard_primary`, `guard_secondary`, and `reference` in `tests/test_multi_rtl_three_device.py`
- [X] T120 [US4] Add two-GUARD plus ROVER start tests in `tests/test_multi_rtl_three_device.py`
- [X] T121 [US4] Add two-GUARD plus REFERENCE start tests in `tests/test_multi_rtl_three_device.py`
- [X] T122 [US4] Add duplicate physical receiver rejection tests across three role lanes in `tests/test_multi_rtl_three_device.py`
- [X] T123 [US4] Add REFERENCE narrow-window telemetry tests in `tests/test_multi_rtl_three_device.py`

### Implementation

- [X] T124 [US4] Add `guard_secondary`, `reference`, and tier `2_plus` role lane support in `sdrwatch-control.py`
- [X] T125 [US4] Add friendly GUARD and watchlist GUARD display metadata in controller inventory and role assignment responses in `sdrwatch-control.py`
- [X] T126 [US4] Implement three-child role-run validation requiring distinct physical receivers in `sdrwatch-control.py`
- [X] T127 [US4] Implement REFERENCE task mapping as a parked narrow-window job with contextual telemetry only in `sdrwatch-control.py`
- [X] T128 [US4] Ensure REFERENCE does not trigger automatic environmental correction or signal fusion in `sdrwatch/sweep/runner.py`

### Slice 7 Acceptance and Regression

- [X] T129 [US4] Run `python -m pytest -q tests/test_multi_rtl_three_device.py tests/test_multi_rtl_role_runs.py`
- [X] T130 [US4] Run role and telemetry regressions `python -m pytest -q tests/test_multi_rtl_roles.py tests/test_multi_rtl_telemetry.py`
- [X] T131 [US4] Confirm Airspy/HackRF/Soapy remain non-runnable in three-device inventory tests in `tests/test_multi_rtl_inventory.py`

---

## Phase 10: Slice 8 - Minimal UI, Documentation, and Regression Hardening (US1, US2, US3, US4, US5)

**Goal**: Add the compact operator UI and docs needed for multi-RTL role operation while preserving Discovery, FM Validation, diagnostics, and current single-device workflows.

**Independent Test**: The control page shows inventory/tier, identity warnings, role assignment controls, role-run status, child job visibility, and grouped stop without breaking existing scan controls.

### Tests First

- [X] T132 [US1] Add control-page inventory/tier rendering tests in `tests/test_multi_rtl_ui.py`
- [X] T133 [US2] Add control-page identity warning and role assignment control tests in `tests/test_multi_rtl_ui.py`
- [X] T134 [US3] Add role-run status, child job visibility, and grouped stop UI tests in `tests/test_multi_rtl_ui.py`
- [X] T135 [P] [US1] Add regression tests proving Discovery, FM Validation, diagnostics mode, and existing device selector still work in `tests/test_control_page_scan_settings.py`

### Implementation

- [X] T136 [US1] Add compact hardware inventory and capability tier display in `templates/control.html`
- [X] T137 [US2] Add identity warning display and manual GUARD/ROVER/REFERENCE assignment controls in `templates/control.html`
- [X] T138 [US3] Add role-run status, child job visibility, degraded/error display, and grouped stop controls in `templates/control.html`
- [X] T139 [US3] Replace single-active-job assumptions only where role-run status and grouped stop require it in `templates/control.html`
- [X] T140 [US1] Preserve existing scan controls, Discovery preset, FM Validation preset, diagnostics mode, and diagnostic bundle export in `templates/control.html`
- [X] T141 [P] Update README hardware support claims to state scanner execution is currently RTL-native only and Airspy/HackRF/Soapy are planned or unsupported in `README.md`
- [X] T142 [P] Add operator documentation for capability tiers, manual roles, identity warnings, Pi 5 resource expectations, and no continuous raw IQ default in `docs/MULTI_RTL_GUARD_ROVER.md`
- [X] T143 [P] Update diagnostic capture documentation with role-aware bundle expectations in `docs/DIAGNOSTIC_CAPTURE.md`

### Slice 8 Acceptance and Regression

- [X] T144 [US1] Run `python -m pytest -q tests/test_multi_rtl_ui.py tests/test_control_page_scan_settings.py`
- [X] T145 [US3] Run web/controller regression suite `python -m pytest -q tests/test_web_diagnostics_bundle.py tests/test_multi_rtl_web_api.py tests/test_multi_rtl_role_runs.py`
- [ ] T146 [US2] Confirm manual browser workflow from `specs/008-multi-rtl-guard-rover/quickstart.md` can be performed with fake or real controller inventory

---

## Final Phase: Full Validation and Release Readiness

**Purpose**: Prove the whole feature remains inside scope and document any hardware acceptance gaps.

- [X] T147 Run focused feature suite `python -m pytest -q tests/test_multi_rtl_inventory.py tests/test_multi_rtl_identity.py tests/test_multi_rtl_roles.py tests/test_multi_rtl_lifecycle.py tests/test_multi_rtl_guard.py tests/test_multi_rtl_role_runs.py tests/test_multi_rtl_telemetry.py tests/test_multi_rtl_persistence.py tests/test_multi_rtl_three_device.py tests/test_multi_rtl_ui.py tests/test_multi_rtl_web_api.py`
- [X] T148 Run existing regression suite `python -m pytest -q tests/test_cross_sweep_persistence.py tests/test_device_telemetry.py tests/test_effective_parameter_manifest.py tests/test_control_fm_validation.py tests/test_control_page_scan_settings.py tests/test_web_diagnostics_bundle.py`
- [X] T149 Run scanner backend smoke `python -m sdrwatch.cli --list-profiles` and confirm CLI defaults remain native RTL-centered
- [X] T150 Run repository drift check for unsupported runtime claims with `rg -n "Soapy|HackRF|Airspy|rtlsdr|rtlsdr_native" README.md docs install-sdrwatch.sh sdrwatch-control.py sdrwatch`
- [ ] T151 Follow the no-hardware and controller/web smoke sections in `specs/008-multi-rtl-guard-rover/quickstart.md`
- [ ] T152 On Raspberry Pi 5, follow the one-RTL, two-RTL, and three-RTL acceptance sections in `specs/008-multi-rtl-guard-rover/quickstart.md` and save diagnostic bundle references in `docs/MULTI_RTL_GUARD_ROVER.md`
- [X] T153 Document any unavailable hardware telemetry, skipped Pi 5 hardware run, or remaining diagnostic-only provenance in `specs/008-multi-rtl-guard-rover/quickstart.md`

---

## Dependencies and Execution Order

### Phase Dependencies

- **Phase 1 Setup**: No dependencies.
- **Phase 2 Foundational**: Depends on Phase 1 and blocks all implementation slices.
- **Slice 1 Inventory/Capability**: Depends on Phase 2.
- **Slice 2 Identity/Role Assignment**: Depends on Slice 1 inventory and capability.
- **Slice 3 Lock/Lifecycle**: Depends on Slice 2 role assignment state.
- **Slice 4 One-Device GUARD**: Depends on Slice 3 atomic locking and role assignment.
- **Slice 5 Two-Device GUARD+ROVER**: Depends on Slice 4 child job metadata and Slice 3 atomic locking.
- **Slice 6 Telemetry/Provenance**: Depends on Slice 4 metadata plumbing; can run before Slice 5 UI work but should not precede child job metadata.
- **Slice 7 Three-Device Roles**: Depends on Slice 5 grouped role runs and Slice 6 reference telemetry.
- **Slice 8 UI/Docs**: Depends on controller/web APIs from Slices 1, 2, 5, 6, and 7.
- **Final Validation**: Depends on all selected slices.

### User Story Dependencies

- **US1 Inventory/capability**: MVP starting point after foundation.
- **US2 One-RTL GUARD**: Depends on US1 inventory and role assignment.
- **US3 Two-RTL GUARD+ROVER**: Depends on US2 child job and lock behavior.
- **US5 Telemetry**: Depends on role/job metadata from US2, then supports US3 and US4 benchmarking.
- **US4 Three-RTL roles**: Depends on grouped role runs and telemetry.

### Within Each Slice

- Write tests first and confirm they fail before implementation.
- Implement controller behavior before web proxy behavior.
- Implement web proxy behavior before control-page UI.
- Implement scanner metadata plumbing without changing detection thresholds, FM profile behavior, cross-sweep promotion logic, or source selection semantics.
- Run slice-specific acceptance tests before moving to the next slice.

---

## Parallel Opportunities

- T005 and T006 can run in parallel after T001-T004 because they touch different test files.
- T014-T019 can run in parallel because they define independent Slice 1 tests.
- T032-T037 can run in parallel because they define independent identity, role, and web proxy tests.
- T099-T104 can run in parallel because telemetry, bundle, and persistence tests touch distinct files or isolated sections.
- T141-T143 can run in parallel because they touch distinct documentation files.

## Parallel Example: Slice 1

```text
Task: "T014 [US1] Add zero/one/two/three RTL inventory tier tests in tests/test_multi_rtl_inventory.py"
Task: "T017 [US1] Add backend gating tests that reject unsupported starts in tests/test_multi_rtl_backend_gating.py"
Task: "T019 [US1] Add web proxy tests for GET /api/hardware/inventory in tests/test_multi_rtl_web_api.py"
```

## Parallel Example: Slice 6

```text
Task: "T099 [US5] Add diagnostic record provenance tests in tests/test_multi_rtl_telemetry.py"
Task: "T103 [US5] Add diagnostic bundle summary tests in tests/test_web_diagnostics_bundle.py"
Task: "T104 [US5] Add additive scan-update provenance migration tests in tests/test_multi_rtl_persistence.py"
```

---

## Implementation Strategy

### MVP First

1. Complete Phase 1 and Phase 2.
2. Complete Slice 1 inventory, capability tier, and backend gating.
3. Complete Slice 2 identity and manual role assignment state.
4. Complete Slice 3 lock/lifecycle hardening.
5. Complete Slice 4 one-device GUARD path.
6. Stop and validate one-RTL behavior through web/controller workflow before adding grouped multi-device starts.

### Incremental Delivery

1. Deliver inventory and honest capability reporting.
2. Add manual role assignments and stable identity warnings.
3. Harden locks and lifecycle.
4. Add one-RTL GUARD.
5. Add two-RTL GUARD+ROVER.
6. Add benchmark telemetry/provenance.
7. Add three-RTL role lanes.
8. Add compact UI/docs polish and run full regression.

### Scope Guardrails

- Do not enable Airspy, HackRF, Soapy, or other non-RTL scanner execution.
- Do not change detection thresholds, FM Broadcast profile behavior, cross-sweep promotion logic, or source selection semantics except for metadata plumbing.
- Do not add signal tracks, observations, fusion tables, destructive migrations, continuous raw IQ capture, automatic role assignment, scheduler optimization, high-band hazard monitoring, or a Rust DSP rewrite.
- Keep provenance diagnostic-first; add nullable scan update provenance only as the separate low-risk Slice 6 task.
