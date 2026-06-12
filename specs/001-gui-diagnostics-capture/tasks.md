# Tasks: GUI Diagnostic Capture

**Input**: Design documents from `specs/001-gui-diagnostics-capture/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Required by FR-027 and FR-028. Tests must be no-hardware web/API, controller, and bundle-creation checks using temporary files and a temporary SQLite database.

**Organization**: Tasks are grouped by user story so each story can be implemented and validated independently through the web UI and controller lifecycle. Scanner CLI checks are internal backend smoke checks only.

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Establish shared constants and module boundaries for the diagnostic workflow.

- [X] T001 Add diagnostic export limit defaults and local bundle naming settings in `sdrwatch_web/config.py`
- [X] T002 Add controller diagnostic artifact directory constants and directory creation support in `sdrwatch-control.py`
- [X] T003 [P] Create the diagnostic bundle service module skeleton in `sdrwatch_web/diagnostics.py`
- [X] T004 [P] Add test fixture scaffolding comments/placeholders for diagnostics tests in `tests/test_web_diagnostics_bundle.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Implement shared helpers that all diagnostic capture and export stories depend on.

**Critical**: No user story work should begin until this phase is complete.

- [X] T005 Implement bounded line tail and JSONL tail helpers in `sdrwatch_web/diagnostics.py`
- [X] T006 Implement JSON serialization helpers for controller job data and SQLite rows in `sdrwatch_web/diagnostics.py`
- [X] T007 Implement safe bundle manifest structures for included, missing, and truncated evidence in `sdrwatch_web/diagnostics.py`
- [X] T008 Implement read-only SQLite evidence query helpers for `baselines`, `baseline_detections`, `scan_updates`, `monitoring_zones`, and `friendly_signals` in `sdrwatch_web/diagnostics.py`

**Checkpoint**: Shared diagnostic configuration and bundle helper primitives are ready.

---

## Phase 3: User Story 1 - Enable Diagnostics Before a Scan (Priority: P1) MVP

**Goal**: The operator can enable Diagnostics mode from the control page before starting a scan, and the controller automatically configures a safe diagnostic JSONL path.

**Independent Test**: Enable Diagnostics mode on the control page, start a scan through `/api/jobs`, and confirm the resulting job params contain a generated safe `diagnostic_jsonl` without any operator-entered path.

### Tests for User Story 1

- [X] T009 [P] [US1] Add controller tests for generated `diagnostic_jsonl`, existing explicit `diagnostic_jsonl`, and disabled diagnostics behavior in `tests/test_control_diagnostics_mode.py`
- [X] T010 [P] [US1] Add web API tests for Diagnostics mode scan-start compatibility with `/api/jobs` request and response shape in `tests/test_web_diagnostics_bundle.py`

### Implementation for User Story 1

- [X] T011 [US1] Implement safe per-job diagnostic JSONL path generation and params normalization in `sdrwatch-control.py`
- [X] T012 [US1] Ensure generated diagnostic params are persisted in `Job.params` and passed to the existing `--diagnostic-jsonl` scanner flag in `sdrwatch-control.py`
- [X] T013 [US1] Replace the operator-entered diagnostic JSONL path field with a visible Diagnostics mode toggle in `templates/control.html`
- [X] T014 [US1] Update scan-start JavaScript to send `params.diagnostics_mode` only when Diagnostics mode is enabled in `templates/control.html`
- [X] T015 [US1] Verify non-diagnostic scan start still omits diagnostic params and preserves existing `/api/jobs` behavior in `tests/test_web_diagnostics_bundle.py`

**Checkpoint**: User Story 1 is independently functional and is the suggested MVP.

---

## Phase 4: User Story 2 - Export a Diagnostic Bundle (Priority: P2)

**Goal**: The operator can export a bounded local diagnostic zip for a selected active or recent job from the web UI.

**Independent Test**: With representative local job metadata, log, diagnostic JSONL, and SQLite evidence present, export a bundle from the web/API path and inspect the downloaded zip contents and manifest.

### Tests for User Story 2

- [X] T016 [US2] Add bundle service test for complete zip contents using temporary logs, diagnostic JSONL, and SQLite rows in `tests/test_web_diagnostics_bundle.py`
- [X] T017 [US2] Add API download test for `GET /api/jobs/<job_id>/diagnostic-bundle` status, content type, filename, and auth behavior in `tests/test_web_diagnostics_bundle.py`
- [X] T018 [US2] Add truncation and missing-evidence manifest tests for large logs, large diagnostic JSONL, and absent optional files in `tests/test_web_diagnostics_bundle.py`

### Implementation for User Story 2

- [X] T019 [US2] Implement diagnostic bundle creation entry point returning zip bytes and metadata in `sdrwatch_web/diagnostics.py`
- [X] T020 [US2] Add job metadata, controller params, scanner command, scanner log tail, and diagnostic JSONL tail zip entries in `sdrwatch_web/diagnostics.py`
- [X] T021 [US2] Add baseline metadata, recent `baseline_detections`, recent `scan_updates`, monitoring zones, and friendly signals zip entries in `sdrwatch_web/diagnostics.py`
- [X] T022 [US2] Add `GET /api/jobs/<job_id>/diagnostic-bundle` route with existing auth and bounded query parameters in `sdrwatch_web/blueprints/api_jobs.py`
- [X] T023 [US2] Add an Export diagnostic bundle button and status/error handling to the control page in `templates/control.html`
- [X] T024 [US2] Update control page JavaScript to choose the active or most recent job and trigger the zip download in `templates/control.html`

**Checkpoint**: User Story 2 is independently functional after US1 or with prepared existing job artifacts.

---

## Phase 5: User Story 3 - Provide Problem Notes with the Evidence (Priority: P3)

**Goal**: Every exported bundle includes a Markdown notes template that captures expected behavior, actual behavior, affected frequency or band, and problem type.

**Independent Test**: Export a bundle and confirm the included notes file contains all required prompts and problem type choices.

### Tests for User Story 3

- [X] T025 [US3] Add notes template content test for required prompts and problem types in `tests/test_web_diagnostics_bundle.py`

### Implementation for User Story 3

- [X] T026 [US3] Implement the operator notes template renderer in `sdrwatch_web/diagnostics.py`
- [X] T027 [US3] Include `README.md` or `NOTES.md` in every diagnostic bundle and list it in the manifest in `sdrwatch_web/diagnostics.py`

**Checkpoint**: User Story 3 is independently verifiable by inspecting the bundle archive.

---

## Phase 6: User Story 4 - Create Bundles Without SDR Hardware (Priority: P4)

**Goal**: Maintainers can create and test bundles from existing local artifacts and a temporary SQLite database without requiring SDR hardware or manual scanner CLI commands.

**Independent Test**: Run no-hardware tests with fake controller data, temporary files, and temporary SQLite rows; bundle creation succeeds and records missing evidence when artifacts are partial.

### Tests for User Story 4

- [X] T028 [US4] Add temporary SQLite schema/data fixture covering `baselines`, `baseline_detections`, `scan_updates`, `monitoring_zones`, and `friendly_signals` in `tests/test_web_diagnostics_bundle.py`
- [X] T029 [US4] Add fake controller client tests for exporting active, recent, and finished jobs without SDR hardware in `tests/test_web_diagnostics_bundle.py`

### Implementation for User Story 4

- [X] T030 [US4] Ensure bundle creation tolerates missing controller logs, missing diagnostic files, missing tables, and empty optional evidence in `sdrwatch_web/diagnostics.py`
- [X] T031 [US4] Ensure web/API bundle export can use prepared local job artifacts without starting an SDR scan in `sdrwatch_web/blueprints/api_jobs.py`

**Checkpoint**: User Story 4 proves the export workflow is testable and useful without SDR hardware.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Documentation, compatibility checks, and final validation across the whole feature.

- [X] T032 [P] Document GUI-based diagnostic capture and export steps in `docs/DIAGNOSTIC_CAPTURE.md`
- [X] T033 [P] Update the main operator workflow documentation to point to GUI diagnostics instead of scanner CLI diagnostics in `README.md`
- [X] T034 [P] Update runtime artifact notes for controller-generated diagnostic JSONL and bundle exports in `docs/PROJECT_INVENTORY.md`
- [X] T035 Run the focused no-hardware tests documented in `specs/001-gui-diagnostics-capture/quickstart.md`
- [X] T036 Run the existing detection regression tests documented in `specs/001-gui-diagnostics-capture/quickstart.md`
- [X] T037 Validate the control page workflow manually through the web UI and record any command or environment updates in `specs/001-gui-diagnostics-capture/quickstart.md`
- [X] T038 Review changed files to confirm detection, CFAR, DSP, baseline, and scanner detection logic were not modified in `sdrwatch/dsp/detection.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational (Phase 2)**: Depends on Phase 1 and blocks user stories.
- **US1 (Phase 3)**: Depends on Phase 2. This is the MVP.
- **US2 (Phase 4)**: Depends on Phase 2 and can use prepared artifacts, but the GUI path is strongest after US1.
- **US3 (Phase 5)**: Depends on US2 bundle assembly.
- **US4 (Phase 6)**: Depends on US2 bundle assembly and can be completed after or alongside US3.
- **Polish (Phase 7)**: Depends on the implemented stories being validated.

### User Story Dependencies

- **US1**: No dependency on other stories after foundation.
- **US2**: Depends on bundle helper foundation; integrates with US1 for the active GUI workflow.
- **US3**: Depends on US2 archive creation.
- **US4**: Depends on US2 archive creation and validates no-hardware operation.

### Parallel Opportunities

- T003 and T004 can run in parallel with T001 and T002 after the task list is accepted.
- T009 and T010 can run in parallel because they target different test concerns.
- T032, T033, and T034 can run in parallel because they update separate documentation files.

---

## Parallel Example: User Story 1

```text
Task: "Add controller tests for generated diagnostic_jsonl, existing explicit diagnostic_jsonl, and disabled diagnostics behavior in tests/test_control_diagnostics_mode.py"
Task: "Add web API tests for Diagnostics mode scan-start compatibility with /api/jobs request and response shape in tests/test_web_diagnostics_bundle.py"
```

---

## Parallel Example: User Story 2

```text
Task: "Implement diagnostic bundle creation entry point returning zip bytes and metadata in sdrwatch_web/diagnostics.py"
Task: "Add GET /api/jobs/<job_id>/diagnostic-bundle route with existing auth and bounded query parameters in sdrwatch_web/blueprints/api_jobs.py"
Task: "Add an Export diagnostic bundle button and status/error handling to the control page in templates/control.html"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 setup.
2. Complete Phase 2 foundational helpers.
3. Complete Phase 3 to replace manual diagnostic paths with the Diagnostics mode toggle and controller-generated paths.
4. Stop and validate US1 through the web/API path before implementing bundle export.

### Incremental Delivery

1. Add US1 for GUI-first capture intent and controller-generated diagnostic paths.
2. Add US2 for web-downloadable diagnostic bundles.
3. Add US3 for operator notes inside bundles.
4. Add US4 to harden and prove no-hardware bundle creation.
5. Complete documentation and regression validation.

### Validation Focus

- Web UI and `/api/jobs` controller lifecycle are the operator acceptance path.
- Scanner CLI behavior remains internal and compatible.
- Tests must use temporary files and temporary SQLite databases for no-hardware coverage.
- Default bundle exports must be bounded and record truncation or missing evidence in `manifest.json`.
