# Tasks: FM Detection Card Stability

**Input**: Design documents from `specs/003-fm-detection-card-stability/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Required. The feature spec and plan require test-first implementation for FM-like fixtures, GUI payloads, controller/profile wiring, width behavior, two-pass behavior, and diagnostics evidence.

**Organization**: Tasks are grouped by user story to enable independent implementation and validation. Operator-facing acceptance runs through the web UI and controller job lifecycle; scanner CLI checks are internal backend smoke tests only.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it touches different files or isolated fixtures.
- **[Story]**: User story label for story phases only.
- Every task includes an exact file path.

## Phase 1: Setup (Shared Test Infrastructure)

**Purpose**: Create reusable no-hardware fixtures so FM stability tests stay deterministic and do not require SDR hardware.

- [X] T001 [P] Create shared synthetic segment and baseline fixture helpers in `tests/helpers_fm_detection.py`
- [X] T002 [P] Create reusable fake control-page profile/controller helpers in `tests/helpers_control.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Lock down current profile/controller contracts before changing GUI or scanner behavior.

- [X] T003 [P] Add `fm_broadcast` profile serialization and expected-value tests in `tests/test_fm_validation_profile.py`
- [X] T004 [P] Add controller `_build_cmd` tests for `profile`, `two_pass`, revisit, and width params in `tests/test_control_fm_validation.py`
- [X] T005 Add scanner CLI profile-application tests for FM-only hidden fields in `tests/test_fm_validation_profile.py`
- [X] T006 Implement any narrow controller/CLI passthrough required by T003-T005 in `sdrwatch-control.py` and `sdrwatch/cli.py`

**Checkpoint**: Profile and controller command behavior is test-covered before user-story work begins.

---

## Phase 3: User Story 1 - Validate FM Band Without Card Explosion (Priority: P1) MVP

**Goal**: FM Validation can be started from the GUI and produces a stable FM-band card set instead of hundreds of tiny sub-kHz cards.

**Independent Test**: Select FM Validation in the control page, copy settings, start a controller-backed diagnostics job, and confirm FM-specific settings plus bounded persisted cards in no-hardware fixtures and exported diagnostics.

### Tests for User Story 1

Write these tests first and confirm they fail before implementation.

- [X] T007 [P] [US1] Add FM Validation render/copy/payload tests in `tests/test_control_page_scan_settings.py`
- [X] T008 [P] [US1] Add wide/spiky FM-like bounded-card persistence test in `tests/test_fm_persistence_stability.py`
- [X] T009 [P] [US1] Add FM Validation effective-settings diagnostic bundle test in `tests/test_web_diagnostics_bundle.py`

### Implementation for User Story 1

- [X] T010 [US1] Add the FM Validation preset option, label, description, and values in `templates/control.html`
- [X] T011 [US1] Update `buildScanJobPayload()` so FM Validation submits existing `/api/jobs` params and omits backend-only `scan_preset` in `templates/control.html`
- [X] T012 [US1] Align `fm_broadcast` profile serialization and FM Validation values in `sdrwatch/io/profiles.py`
- [X] T013 [US1] Ensure FM Validation span shaping consumes match/display/min/max profile values in `sdrwatch/detection/engine.py`
- [X] T014 [US1] Ensure repeated nearby FM fragments update existing baseline detections under configured matching in `sdrwatch/baseline/persistence.py`
- [X] T015 [US1] Run US1 focused tests from `specs/003-fm-detection-card-stability/quickstart.md` and record the result in `specs/003-fm-detection-card-stability/quickstart.md`

**Checkpoint**: FM Validation MVP is independently functional and testable from the GUI/controller path.

---

## Phase 4: User Story 2 - Keep Discovery as First-Light (Priority: P1)

**Goal**: RTL-SDR v4 Discovery remains available, remains first-light/card-producing, and is not silently replaced by FM Validation behavior.

**Independent Test**: Render the control page, select Discovery, copy current scan settings, and verify the relaxed first-light values remain intact after FM Validation is added.

### Tests for User Story 2

- [X] T016 [P] [US2] Extend Discovery preset render and reset tests in `tests/test_control_page_scan_settings.py`
- [X] T017 [US2] Add Discovery payload regression coverage after FM Validation selection in `tests/test_control_page_scan_settings.py`

### Implementation for User Story 2

- [X] T018 [US2] Preserve RTL-SDR v4 Discovery option, descriptions, and first-light values in `templates/control.html`
- [X] T019 [US2] Ensure reset and copy behavior stays Discovery-first while FM Validation remains additive in `templates/control.html`
- [X] T020 [US2] Run Discovery regression tests from `specs/003-fm-detection-card-stability/quickstart.md` and record the result in `specs/003-fm-detection-card-stability/quickstart.md`

**Checkpoint**: Discovery still works as an independent first-light operator path.

---

## Phase 5: User Story 3 - Preserve Signal Separation and Narrow Non-FM Behavior (Priority: P2)

**Goal**: FM-specific stabilization groups fragments from one FM-like signal, preserves separated FM-like signals, and does not widen unrelated narrow non-FM detections.

**Independent Test**: Run deterministic no-hardware fixtures for one spiky FM-like signal, multiple separated FM-like signals, and narrow non-FM signals.

### Tests for User Story 3

- [X] T021 [P] [US3] Add separated FM-like signals persistence test in `tests/test_fm_persistence_stability.py`
- [X] T022 [P] [US3] Add narrow non-FM width-scope regression test in `tests/test_non_fm_width_scope.py`
- [X] T023 [P] [US3] Add FM width floor and cap regression tests in `tests/test_extent_hysteresis.py`

### Implementation for User Story 3

- [X] T024 [US3] Tune FM-specific center/match/cluster behavior without merging adjacent stations in `sdrwatch/detection/engine.py` and `sdrwatch/baseline/persistence.py`
- [X] T025 [US3] Scope FM min display/match widths to selected profile or explicit params in `sdrwatch/detection/engine.py`
- [X] T026 [US3] Run US3 fixture tests from `specs/003-fm-detection-card-stability/quickstart.md` and record the result in `specs/003-fm-detection-card-stability/quickstart.md`

**Checkpoint**: FM stability improvements do not damage signal separation or non-FM narrow detections.

---

## Phase 6: User Story 4 - Make Persistence Decisions Observable (Priority: P2)

**Goal**: Diagnostic bundles explain why detections were created, updated, missed, clamped, merged, or revisited.

**Independent Test**: Run persistence and diagnostics fixtures that emit bounded create/update/no-match/missing/width/revisit evidence and verify bundle export includes it.

### Tests for User Story 4

- [X] T027 [P] [US4] Add persistence decision logging tests in `tests/test_fm_persistence_diagnostics.py`
- [X] T028 [P] [US4] Add diagnostic bundle decision export and truncation tests in `tests/test_web_diagnostics_bundle.py`

### Implementation for User Story 4

- [X] T029 [US4] Add structured insert/update/no-match/missing decision logging in `sdrwatch/baseline/persistence.py`
- [X] T030 [US4] Add width floor/clamp decision logging in `sdrwatch/detection/engine.py` and `sdrwatch/baseline/persistence.py`
- [X] T031 [US4] Add effective settings and revisit summary diagnostics in `sdrwatch/sweep/sweeper.py`
- [X] T032 [US4] Extend diagnostic bundle export to include bounded decision evidence in `sdrwatch_web/diagnostics.py`
- [X] T033 [US4] Record missing or truncated decision evidence in `manifest.json` generation in `sdrwatch_web/diagnostics.py`
- [X] T034 [US4] Run US4 diagnostics tests from `specs/003-fm-detection-card-stability/quickstart.md` and record the result in `specs/003-fm-detection-card-stability/quickstart.md`

**Checkpoint**: Future FM diagnostic bundles can explain card stability decisions without inference from final rows alone.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Final verification, documentation, and hardware acceptance notes.

- [X] T035 [P] Update post-implementation FM evidence and comparison notes in `docs/FM_DETECTION_CARD_EXPLOSION_REPORT.md`
- [X] T036 [P] Update exact automated validation commands and results in `specs/003-fm-detection-card-stability/quickstart.md`
- [X] T037 [P] Update implementation-choice notes if profile-driven vs explicit-param wiring changed in `specs/003-fm-detection-card-stability/plan.md`
- [X] T038 Run the full focused no-hardware regression command and record the result in `specs/003-fm-detection-card-stability/quickstart.md`
- [ ] T039 Perform GUI/controller hardware validation and record exported bundle findings in `docs/FM_DETECTION_CARD_EXPLOSION_REPORT.md`
- [X] T040 Review final diff scope for unintended broad scanner/controller/database changes and record scope notes in `specs/003-fm-detection-card-stability/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup**: No dependencies.
- **Phase 2 Foundational**: Depends on Phase 1 helpers; blocks all user stories.
- **US1 FM Validation MVP**: Depends on Phase 2.
- **US2 Discovery Preservation**: Depends on Phase 2 and can proceed in parallel with US1 after shared preset wiring is understood.
- **US3 Separation and Non-FM Scope**: Depends on Phase 2; can proceed in parallel after US1 establishes FM Validation settings.
- **US4 Diagnostics Observability**: Depends on Phase 2; can proceed in parallel with US1/US3 if decision-event naming is coordinated.
- **Polish**: Depends on the completed stories selected for release.

### User Story Dependencies

- **US1 (P1)**: MVP. Delivers FM Validation and should be completed first.
- **US2 (P1)**: Protects Discovery. Can be worked alongside US1, but final validation depends on FM Validation existing.
- **US3 (P2)**: Protects RF behavior. Builds on the FM Validation settings decided by US1.
- **US4 (P2)**: Adds diagnostics observability. Can be incremental, but final acceptance requires US1 evidence.

### Implementation Order

1. Complete Phase 1 and Phase 2.
2. Complete US1 and validate the MVP independently.
3. Complete US2 before merging to prevent Discovery regression.
4. Complete US3 and US4 before hardware acceptance if implementation touches matching, width, or diagnostics.
5. Complete Polish validation and update evidence.

---

## Parallel Opportunities

- T001 and T002 can run in parallel.
- T003 and T004 can run in parallel.
- T007, T008, and T009 can run in parallel once Phase 2 is complete.
- T016 can run while T007-T009 are being written, because it focuses on Discovery preservation.
- T021, T022, and T023 can run in parallel because they use different fixture/test files.
- T027 and T028 can run in parallel if the diagnostic event names are agreed first.
- T035, T036, and T037 can run in parallel during documentation polish.

## Parallel Example: User Story 1

```text
Task: "Add FM Validation render/copy/payload tests in tests/test_control_page_scan_settings.py"
Task: "Add wide/spiky FM-like bounded-card persistence test in tests/test_fm_persistence_stability.py"
Task: "Add FM Validation effective-settings diagnostic bundle test in tests/test_web_diagnostics_bundle.py"
```

## Parallel Example: User Story 3

```text
Task: "Add separated FM-like signals persistence test in tests/test_fm_persistence_stability.py"
Task: "Add narrow non-FM width-scope regression test in tests/test_non_fm_width_scope.py"
Task: "Add FM width floor and cap regression tests in tests/test_extent_hysteresis.py"
```

---

## Implementation Strategy

### MVP First

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3, User Story 1.
3. Stop and validate FM Validation independently through no-hardware tests and the GUI/controller checklist in `specs/003-fm-detection-card-stability/quickstart.md`.

### Incremental Delivery

1. **MVP**: US1 adds FM Validation and proves the card explosion is bounded.
2. **Regression safety**: US2 proves Discovery remains first-light.
3. **RF correctness**: US3 proves separation and non-FM narrow behavior.
4. **Diagnosability**: US4 proves future bundles explain decisions.
5. **Polish**: Run focused tests, hardware acceptance, and update evidence.

### Notes

- Tests must be written before implementation tasks in each story.
- Keep changes small and tied to the files listed in `plan.md`.
- Preserve `/api/jobs` shape and controller ownership of scanner command generation.
- Treat scanner CLI invocations as backend smoke checks only; operator acceptance is web GUI plus controller lifecycle.
- Avoid threshold-only fixes and broad detector rewrites.
