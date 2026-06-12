# Tasks: Improve Scan Control

**Input**: Design documents from `specs/002-improve-scan-control/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/](./contracts/), [quickstart.md](./quickstart.md)

**Tests**: Required by the implementation plan. Use no-hardware web/template/API tests for controls, presets, and `/api/jobs` payloads. Operator acceptance and hardware validation must use the web GUI and controller job lifecycle; scanner CLI checks are internal backend smoke tests only.

**Organization**: Tasks are grouped by user story so each story can be implemented and verified independently. User Story 1 is the MVP guardrail; User Story 3 contains the new real-hardware preset work.

## Phase 1: Setup (Shared Context and Test Surface)

**Purpose**: Prepare the existing GUI test surface and capture the diagnostic context that drives preset/default work.

- [X] T001 Review the promotion/persistence evidence and recommended preset direction in `docs/DETECTION_TUNING_REPORT.md`
- [X] T002 Review the existing control-page implementation and current safe defaults in `templates/control.html`
- [X] T003 Review the existing no-hardware control-page tests and fixtures in `tests/test_control_page_scan_settings.py`
- [X] T004 [P] Confirm the active task scope and feature artifacts in `specs/002-improve-scan-control/plan.md`
- [X] T005 [P] Confirm `/api/jobs` payload expectations for presets in `specs/002-improve-scan-control/contracts/api-jobs-payload.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Add shared preset/default test scaffolding before changing page behavior.

**CRITICAL**: No user story implementation should begin until this phase is complete.

- [X] T006 Add reusable HTML extraction/assertion helpers for preset controls and preset copy text in `tests/test_control_page_scan_settings.py`
- [X] T007 Add reusable expected-preset dictionaries for RTL-SDR v4 Discovery, Stable Baseline, and Fast Wide Survey in `tests/test_control_page_scan_settings.py`
- [X] T008 Add a no-hardware rendered-page test that fails until GUI preset controls are present in `tests/test_control_page_scan_settings.py`
- [X] T009 Add a no-hardware settings-builder test scaffold that can assert preset-applied `/api/jobs` params in `tests/test_control_page_scan_settings.py`

**Checkpoint**: Tests can express the new preset/default contract before template changes begin.

---

## Phase 3: User Story 1 - Run a Scan From Basic Controls (Priority: P1) MVP

**Goal**: Preserve the existing browser-operated scan workflow while adding preset/default work elsewhere on the page.

**Independent Test**: Render the page with fake controller data, confirm Basic controls remain available, start a diagnostics-mode job through `/api/jobs`, poll logs/state, and stop the job without scanner CLI instructions.

### Tests for User Story 1

- [X] T010 [US1] Add a regression test that Basic controls remain present after preset additions in `tests/test_control_page_scan_settings.py`
- [X] T011 [US1] Add a regression test that Diagnostics mode still submits without a manual diagnostic path in `tests/test_control_page_scan_settings.py`
- [X] T012 [US1] Add a regression test that the start payload still uses `{device_key, label, baseline_id, params}` in `tests/test_control_page_scan_settings.py`

### Implementation for User Story 1

- [X] T013 [US1] Preserve monitoring location, device, monitoring zones, run mode, Diagnostics mode, start/stop, diagnostics export, and live-log controls while editing `templates/control.html`
- [X] T014 [US1] Preserve `buildScanJobPayload()` validation for missing baseline, missing enabled zones, and missing device in `templates/control.html`
- [X] T015 [US1] Preserve existing active-job polling, log polling, stop-job handling, and diagnostics bundle export wiring in `templates/control.html`

**Checkpoint**: User Story 1 remains a working MVP scan workflow through the web GUI.

---

## Phase 4: User Story 2 - Tune Common RF Parameters Safely (Priority: P2)

**Goal**: Keep common RF settings safe and bounded while clarifying FFT and gain tradeoffs.

**Independent Test**: Render the page, adjust Tuning controls, verify slider values/help text, and confirm generated settings still use existing `/api/jobs` parameter names.

### Tests for User Story 2

- [X] T016 [US2] Add tests that FFT help text explains speed versus characterization tradeoffs in `tests/test_control_page_scan_settings.py`
- [X] T017 [US2] Add tests that gain help text explains fixed-gain repeatability and overload risk in `tests/test_control_page_scan_settings.py`
- [X] T018 [US2] Add tests that Tuning controls still include bounded controls for `threshold_db`, `guard_bins`, `min_width_bins`, `cfar`, `cfar_alpha_db`, `cfar_quantile`, `fft`, `avg`, `samp_rate`, and `gain` in `tests/test_control_page_scan_settings.py`
- [X] T019 [US2] Add tests that scan controls still suppress autocomplete after preset additions in `tests/test_control_page_scan_settings.py`

### Implementation for User Story 2

- [X] T020 [US2] Update FFT help text to explain lower FFT speed versus higher FFT resolution in `templates/control.html`
- [X] T021 [US2] Update averaging help text to explain noise smoothing versus scan cadence in `templates/control.html`
- [X] T022 [US2] Update gain help text to prefer fixed gain for baseline/detection repeatability while warning about overload in `templates/control.html`
- [X] T023 [US2] Preserve bounded slider/select controls and live value displays for all Tuning controls in `templates/control.html`
- [X] T024 [US2] Preserve `autocomplete="off"` on scan form controls and path/text scan-parameter controls in `templates/control.html`

**Checkpoint**: User Story 2 remains independently testable and the UI does not imply FFT alone fixes zero-card behavior.

---

## Phase 5: User Story 3 - Choose a Real-Hardware Tuning Preset (Priority: P2)

**Goal**: Add GUI presets that apply existing scan parameters and make RTL-SDR v4 first-light cards possible without CLI workflows.

**Independent Test**: Select each preset in the browser, copy current scan settings, and confirm generated `/api/jobs` params match the documented preset values. On real RTL-SDR Blog v4 hardware, start the Discovery preset through the web GUI and confirm signal cards appear.

### Tests for User Story 3

- [X] T025 [US3] Add a rendered-page test for the preset selector and visible preset descriptions in `tests/test_control_page_scan_settings.py`
- [X] T026 [US3] Add a test that RTL-SDR v4 Discovery applies `gain=30`, `samp_rate=2400000`, `step=2400000`, `fft=8192`, `avg=8`, `persistence_min_hits=1`, and `persistence_min_windows=1` in `tests/test_control_page_scan_settings.py`
- [X] T027 [US3] Add a test that Stable Baseline applies `gain=30`, `samp_rate=2400000`, `step=1200000`, `fft=8192`, `avg=16`, `persistence_min_hits=2`, and `persistence_min_windows=2` in `tests/test_control_page_scan_settings.py`
- [X] T028 [US3] Add a test that Fast Wide Survey applies `gain=30`, `samp_rate=2400000`, `step=2400000`, `fft=4096`, `avg=8`, `persistence_min_hits=1`, and `persistence_min_windows=1` in `tests/test_control_page_scan_settings.py`
- [X] T029 [US3] Add a test that preset application does not submit a new backend-only preset identifier in `tests/test_control_page_scan_settings.py`
- [X] T030 [US3] Add a test that manual operator edits after applying a preset are reflected by `buildScanJobPayload()` in `tests/test_control_page_scan_settings.py`

### Implementation for User Story 3

- [X] T031 [US3] Add a GUI tuning preset control near the Tuning controls in `templates/control.html`
- [X] T032 [US3] Define a browser-side preset map for RTL-SDR v4 Discovery, Stable Baseline, and Fast Wide Survey in `templates/control.html`
- [X] T033 [US3] Implement preset application that updates sample rate, step, gain mode, gain, FFT, averaging, threshold, CFAR, and persistence controls in `templates/control.html`
- [X] T034 [US3] Set RTL-SDR v4 Discovery as the reset/default first-light preset in `templates/control.html`
- [X] T035 [US3] Ensure preset-applied manual gain submits through the existing `gain` parameter only when manual mode is selected in `templates/control.html`
- [X] T036 [US3] Ensure preset-applied `step`, `fft`, `avg`, `persistence_min_hits`, and `persistence_min_windows` flow through existing `params` names in `templates/control.html`
- [X] T037 [US3] Add preset descriptions that explain promotion/persistence, FFT speed/resolution, and fixed-gain overload tradeoffs in `templates/control.html`
- [X] T038 [US3] Ensure changing any preset-controlled field marks or displays the settings as custom without blocking scan start in `templates/control.html`

**Checkpoint**: User Story 3 can produce copied settings for each preset and is ready for real-hardware web GUI acceptance.

---

## Phase 6: User Story 4 - Review Expert Settings and Share Current Configuration (Priority: P3)

**Goal**: Preserve Expert controls, reset behavior, copy settings, and internal/debug command copy after preset defaults change.

**Independent Test**: Open Expert controls, change representative settings, apply a preset, reset defaults, copy current scan settings, and verify generated scanner command copy remains clearly internal/debug and metadata-driven.

### Tests for User Story 4

- [X] T039 [US4] Add a test that Expert controls remain present after preset additions in `tests/test_control_page_scan_settings.py`
- [X] T040 [US4] Add a test that Reset to safe defaults restores the documented RTL-SDR v4 Discovery defaults in `tests/test_control_page_scan_settings.py`
- [X] T041 [US4] Add a test that Copy current scan settings includes preset-applied values and excludes blank optional overrides in `tests/test_control_page_scan_settings.py`
- [X] T042 [US4] Add a regression test that generated scanner command copy remains labeled internal/debug and uses controller job metadata in `tests/test_control_page_scan_settings.py`

### Implementation for User Story 4

- [X] T043 [US4] Preserve Expert controls for width, clustering, occupancy, persistence, revisit, path, profile, bandplan, driver, sleep, latitude/longitude, and spur calibration settings in `templates/control.html`
- [X] T044 [US4] Update `SAFE_SCAN_DEFAULTS` so reset restores the chosen first-light preset values and documented blank expert semantics in `templates/control.html`
- [X] T045 [US4] Ensure Copy current scan settings still serializes `buildScanJobPayload()` output as valid JSON in `templates/control.html`
- [X] T046 [US4] Preserve generated scanner command copy behavior using existing active or recent job `cmd` metadata only in `templates/control.html`
- [X] T047 [US4] Ensure raw `diagnostic_jsonl` remains Expert-only and is never required for Basic Diagnostics mode in `templates/control.html`

**Checkpoint**: User Story 4 remains independently usable for maintainer review and remote troubleshooting without promoting CLI operation.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Validate the feature, record any environment limits, and protect scanner/controller/DSP boundaries.

- [X] T048 [P] Update final preset implementation notes and exact values in `specs/002-improve-scan-control/quickstart.md`
- [X] T049 [P] Update `docs/DETECTION_TUNING_REPORT.md` only if implementation changes require clarifying the recommended preset values
- [X] T050 Run `python -m pytest tests/test_control_page_scan_settings.py tests/test_web_diagnostics_bundle.py tests/test_control_diagnostics_mode.py -q` and record any environment limitation in `specs/002-improve-scan-control/quickstart.md`
- [X] T051 Perform a local browser smoke check of the control page and record the result in `specs/002-improve-scan-control/quickstart.md`
- [ ] T052 Perform Raspberry Pi 5 plus RTL-SDR Blog v4 Discovery preset validation through the web GUI and record whether signal cards appear in `specs/002-improve-scan-control/quickstart.md`
- [ ] T053 Export a diagnostic bundle from the hardware validation run and record whether promoted detections and nonempty `baseline_detections` are present in `specs/002-improve-scan-control/quickstart.md`
- [X] T054 Review the final diff to confirm no scanner/controller/database schema/CFAR/detection algorithm rewrites were introduced outside GUI-submitted parameter defaults in `specs/002-improve-scan-control/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; start immediately.
- **Foundational (Phase 2)**: Depends on Setup; blocks all user stories.
- **US1 (Phase 3)**: Depends on Foundational; protects the MVP scan workflow.
- **US2 (Phase 4)**: Depends on Foundational; can proceed after or alongside US1 with file-edit coordination.
- **US3 (Phase 5)**: Depends on Foundational and should follow US2 help-text decisions for FFT/gain copy.
- **US4 (Phase 6)**: Depends on US3 default/preset values and shared reset/copy helpers.
- **Polish (Phase 7)**: Depends on completed target stories.

### User Story Dependencies

- **US1**: No dependency on US2, US3, or US4 after Foundational.
- **US2**: No dependency on US3, but its FFT/gain wording should be reused by US3 preset descriptions.
- **US3**: Depends on shared control helpers and benefits from US2 wording; primary new functionality for real-hardware signal cards.
- **US4**: Depends on US3 because reset and copy behavior must reflect the selected first-light preset defaults.

### Parallel Opportunities

- T004 and T005 can run in parallel with T001-T003 because they read different planning files.
- T048 and T049 can run in parallel after preset values stabilize.
- T050 can run while T051 is prepared, but record results only after implementation is complete.
- Implementation tasks in `templates/control.html` should usually run sequentially to avoid conflicts.
- Test tasks in `tests/test_control_page_scan_settings.py` should usually run sequentially because they share one file.

## Parallel Example: Final Validation

```text
Task: "Update final preset implementation notes and exact values in specs/002-improve-scan-control/quickstart.md"
Task: "Update docs/DETECTION_TUNING_REPORT.md only if implementation changes require clarifying the recommended preset values"
Task: "Prepare local browser smoke check of the control page and record the result in specs/002-improve-scan-control/quickstart.md"
```

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3 to preserve the web GUI scan workflow.
3. Stop and validate that Basic controls still start, monitor, and stop jobs through the controller lifecycle.

### Incremental Delivery

1. Deliver US1 to protect the working GUI scan lifecycle.
2. Deliver US2 to clarify common tuning controls and FFT/gain tradeoffs.
3. Deliver US3 to add real-hardware presets that resolve the promotion/persistence mismatch through GUI-submitted parameters.
4. Deliver US4 to update reset/copy/expert behavior around the new preset defaults.
5. Run automated no-hardware tests, local browser smoke, and real RTL-SDR v4 web GUI validation.

### Boundary Rules

- Preserve existing `/api/jobs` payload shape and parameter names.
- Keep the scanner CLI internal/debug only; do not add CLI operator instructions.
- Do not rewrite CFAR, detector algorithms, persistence storage, database schema, or controller job lifecycle in this pass.
- Treat FFT as a speed/resolution and characterization knob, not as the primary fix for zero signal cards.
- Validate real-hardware success by seeing signal cards through the web UI and nonempty `baseline_detections` in the exported diagnostic bundle.
