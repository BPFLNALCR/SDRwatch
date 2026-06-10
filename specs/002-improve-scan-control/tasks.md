# Tasks: Improve Scan Control

**Input**: Design documents from `specs/002-improve-scan-control/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/)

**Tests**: Required by the implementation plan. Use no-hardware web/template/API tests plus GUI/controller lifecycle manual verification. Scanner CLI checks are internal backend smoke tests only.

**Organization**: Tasks are grouped by user story so each story can be implemented and verified independently.

## Phase 1: Setup (Shared Test Scaffolding)

**Purpose**: Create the no-hardware validation surface for the control-page cleanup.

- [X] T001 Create `tests/test_control_page_scan_settings.py` with Flask app setup, temporary SQLite setup, and fake controller helpers modeled after `tests/test_web_diagnostics_bundle.py`
- [X] T002 Add control-page fixture data for one baseline, one enabled monitoring zone, one disabled monitoring zone, and one fake `rtl:0` device in `tests/test_control_page_scan_settings.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Build shared page helpers that all user stories depend on.

**CRITICAL**: No user story work should begin until this phase is complete.

- [X] T003 Refactor the existing start-button scan parameter assembly into a reusable `buildScanJobPayload()` function in `templates/control.html`
- [X] T004 Add a single safe-defaults map for all scan controls in `templates/control.html`
- [X] T005 Add shared browser helpers for reading controls, writing controls, updating slider value displays, and setting copy/reset status messages in `templates/control.html`
- [X] T006 Ensure `buildScanJobPayload()` preserves the existing `{device_key, label, baseline_id, params}` payload shape in `templates/control.html`

**Checkpoint**: Shared payload/default helpers exist and current start-scan behavior is still wired through them.

---

## Phase 3: User Story 1 - Run a Scan From Basic Controls (Priority: P1) MVP

**Goal**: The scan/control page presents the normal operator workflow as Basic controls and still starts, stops, and monitors scans through the existing controller lifecycle.

**Independent Test**: Render the page with fake controller data, select baseline/device/zones/run mode in the browser workflow, enable Diagnostics mode, start a scan, observe live-log wiring, and stop the scan without scanner CLI instructions.

### Tests for User Story 1

- [X] T007 [US1] Add a rendered-page test for Basic controls, Diagnostics mode, start/stop controls, run mode, zone controls, and live logs in `tests/test_control_page_scan_settings.py`
- [X] T008 [US1] Add a no-hardware `/api/jobs` compatibility test covering Basic-control payload fields, Diagnostics mode without manual path entry, and unchanged parameter names in `tests/test_control_page_scan_settings.py`

### Implementation for User Story 1

- [X] T009 [US1] Reorganize monitoring location, SDR device, monitoring zones, run mode, Diagnostics mode, start/stop, and live logs into a clearly labeled Basic controls section in `templates/control.html`
- [X] T010 [US1] Preserve baseline selection, device refresh, zone enable/disable, run mode duration, diagnostics export, active-state polling, and live-log polling wiring after the layout change in `templates/control.html`
- [X] T011 [US1] Keep Diagnostics mode pathless in Basic controls and submit only the existing diagnostics parameter through `buildScanJobPayload()` in `templates/control.html`
- [X] T012 [US1] Preserve start blocking and messages for missing baseline, missing enabled zones, and unavailable devices in `templates/control.html`
- [X] T013 [US1] Route the existing start-button handler through `buildScanJobPayload()` and keep the existing `POST /api/jobs` request behavior in `templates/control.html`

**Checkpoint**: User Story 1 is usable as the MVP scan workflow through the web GUI.

---

## Phase 4: User Story 2 - Tune Common RF Parameters Safely (Priority: P2)

**Goal**: Common RF tuning settings are grouped into Tuning controls with safer bounded UI, live values, and explanatory help.

**Independent Test**: Render the page, adjust every Tuning control, verify live values and help text, copy or submit generated settings, and confirm the same parameter names are used.

### Tests for User Story 2

- [X] T014 [US2] Add rendered-page tests for Tuning controls, required slider input types, visible value elements, bounded selects, gain mode controls, and help text in `tests/test_control_page_scan_settings.py`
- [X] T015 [US2] Add tests that scan controls and numeric controls include `autocomplete="off"` in `tests/test_control_page_scan_settings.py`
- [X] T016 [US2] Add a generated-settings compatibility test for `threshold_db`, `guard_bins`, `min_width_bins`, `cfar`, `cfar_alpha_db`, `cfar_quantile`, `fft`, `avg`, `samp_rate`, and `gain` in `tests/test_control_page_scan_settings.py`

### Implementation for User Story 2

- [X] T017 [US2] Add a clearly labeled Tuning controls section containing `threshold_db`, `guard_bins`, `min_width_bins`, `cfar`, `cfar_alpha_db`, `cfar_quantile`, `fft`, `avg`, `samp_rate`, and `gain` in `templates/control.html`
- [X] T018 [US2] Convert `threshold_db`, `guard_bins`, `min_width_bins`, `cfar_alpha_db`, and `cfar_quantile` to sliders with stable bounds and live numeric value displays in `templates/control.html`
- [X] T019 [US2] Convert CFAR mode, FFT, averaging, and sample rate to bounded select or segmented controls in `templates/control.html`
- [X] T020 [US2] Replace the raw gain text field with an explicit auto/manual gain control that submits manual gain only when manual mode is selected in `templates/control.html`
- [X] T021 [US2] Add concise inline descriptions or tooltips for every Tuning control in `templates/control.html`
- [X] T022 [US2] Add `autocomplete="off"` to the scan settings form and all numeric, path, and scan-parameter text controls in `templates/control.html`
- [X] T023 [US2] Update `buildScanJobPayload()` so all Tuning controls submit the existing `/api/jobs` parameter names and omit blank optional values consistently in `templates/control.html`

**Checkpoint**: User Story 2 tuning settings are safer to inspect and still submit compatible job parameters.

---

## Phase 5: User Story 3 - Review Expert Settings and Share Current Configuration (Priority: P3)

**Goal**: Advanced operators can find less common settings in Expert controls, reset all settings to safe defaults, copy GUI-generated scan settings as JSON, and copy generated scanner commands only as internal/debug context when available.

**Independent Test**: Open Expert controls, change representative expert settings, reset all settings, copy scan settings JSON, and verify internal/debug command copy appears only when job metadata contains a controller-generated command.

### Tests for User Story 3

- [X] T024 [US3] Add rendered-page tests for Expert controls containing width, clustering, occupancy, persistence, revisit, database path, JSONL path, and expert-only diagnostics override behavior in `tests/test_control_page_scan_settings.py`
- [X] T025 [US3] Add tests for Reset to safe defaults and Copy current scan settings controls and status elements in `tests/test_control_page_scan_settings.py`
- [X] T026 [US3] Add tests for internal/debug generated scanner command copy labeling and unavailable state when job detail has no `cmd` in `tests/test_control_page_scan_settings.py`

### Implementation for User Story 3

- [X] T027 [US3] Reorganize `cluster_merge_hz`, `max_detection_width_hz`, `max_detection_width_ratio`, `new_ema_occ`, persistence controls, revisit controls, `db`, `jsonl`, and remaining compatibility controls into Expert controls in `templates/control.html`
- [X] T028 [US3] Ensure any retained `diagnostic_jsonl` path override is Expert-only and never required for normal Diagnostics mode in `templates/control.html`
- [X] T029 [US3] Add a visible Reset to safe defaults button that restores every Basic, Tuning, and Expert setting and refreshes slider value displays in `templates/control.html`
- [X] T030 [US3] Add a visible Copy current scan settings button that copies `buildScanJobPayload()` output as formatted JSON and reports clipboard success or failure in `templates/control.html`
- [X] T031 [US3] Add internal/debug generated scanner command copy behavior using existing active or recent job `cmd` metadata from `/api/jobs/active` or `/api/jobs/<job_id>` in `templates/control.html`
- [X] T032 [US3] Disable or omit generated scanner command copy when no safe controller-generated `cmd` metadata is available in `templates/control.html`
- [X] T033 [US3] Verify Expert-control values still flow through `buildScanJobPayload()` with existing parameter names in `templates/control.html`

**Checkpoint**: User Story 3 allows expert review, safe reset, and shareable settings without making CLI usage the primary workflow.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Validate the full feature and keep the repository workflow clear.

- [X] T034 [P] Update `specs/002-improve-scan-control/quickstart.md` with any final implementation-specific GUI verification notes discovered during implementation
- [X] T035 Run `python -m pytest tests/test_control_page_scan_settings.py tests/test_web_diagnostics_bundle.py tests/test_control_diagnostics_mode.py -q` and record any environment limitation in `specs/002-improve-scan-control/quickstart.md`
- [X] T036 Complete the GUI manual verification checklist in `specs/002-improve-scan-control/quickstart.md`
- [X] T037 Review the final diff to confirm scanner, controller, database, detection, DSP, baseline, and dashboard behavior were not changed outside compatibility-preserving GUI parameter submission in `specs/002-improve-scan-control/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; start immediately.
- **Foundational (Phase 2)**: Depends on Setup; blocks all user stories.
- **User Story 1 (Phase 3)**: Depends on Foundational; delivers the MVP scan workflow.
- **User Story 2 (Phase 4)**: Depends on Foundational; can be implemented after or alongside US1 if file-edit conflicts are coordinated.
- **User Story 3 (Phase 5)**: Depends on Foundational; can be implemented after or alongside US1/US2 if file-edit conflicts are coordinated.
- **Polish (Phase 6)**: Depends on completed target stories.

### User Story Dependencies

- **US1**: No dependency on US2 or US3 after Foundational.
- **US2**: Depends on shared payload/default helpers from Foundational; does not require US3.
- **US3**: Depends on shared payload/default helpers from Foundational; benefits from US1 active/recent job state but remains independently testable with fake job metadata.

### Parallel Opportunities

- T034 can run in parallel with final code review once behavior is stable.
- Test-writing tasks within a story should be done before implementation but should be coordinated because they share `tests/test_control_page_scan_settings.py`.
- Implementation tasks touching `templates/control.html` should usually run sequentially to avoid conflicts.
- Existing diagnostics tests can be run in parallel with manual review after implementation.

## Parallel Example: Polish

```text
Task: "Update specs/002-improve-scan-control/quickstart.md with final GUI verification notes"
Task: "Run python -m pytest tests/test_control_page_scan_settings.py tests/test_web_diagnostics_bundle.py tests/test_control_diagnostics_mode.py -q"
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3.
3. Validate that the scan/control page works as the primary GUI workflow.
4. Stop and review before moving common tuning and expert controls.

### Incremental Delivery

1. Deliver US1 to keep the operator scan workflow clear.
2. Deliver US2 to make common RF tuning safer and autofill-resistant.
3. Deliver US3 to add reset, copy settings, expert grouping, and internal/debug command copy.
4. Run automated no-hardware tests and GUI manual verification after each story.

### Notes

- Every story must preserve existing `/api/jobs` payload shape and parameter names.
- Do not add a frontend framework.
- Do not change scanner, controller, database, detection, DSP, baseline, or classification behavior unless required to preserve existing GUI parameter submission.
- Operator acceptance is GUI/controller lifecycle based; scanner CLI checks are optional backend smoke tests only.
