# Tasks: FM Signal Characterization

**Input**: Design documents from `specs/004-fm-signal-characterization/`

**Prerequisites**: `plan.md`, `spec.md`, `research.md`, `data-model.md`, `contracts/`, `quickstart.md`

**Tests**: Required. The spec, plan, and user request require a narrow, test-first implementation with deterministic FM-like fixtures, diagnostics-first evidence checks, persistence invariant coverage, and GUI/controller validation guidance.

**Organization**: Tasks are grouped by user story to keep each increment independently testable while preserving the GUI-first workflow, the existing `/api/jobs` shape, current FM Validation stable-card behavior, and RTL-SDR v4 Discovery first-light behavior.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel because it touches different files or isolated fixtures.
- **[Story]**: User story label for story phases only.
- Every task includes an exact file path.

## Phase 1: Setup (Shared Characterization Test Infrastructure)

**Purpose**: Create reusable no-hardware fixtures and shared helpers so characterization work stays deterministic and diagnostics-first.

- [X] T001 [P] Create shared characterization fixture helpers for explicit `raw_*`, `measured_*`, `match_*`, and `display_*` evidence in `tests/helpers_fm_characterization.py`
- [X] T002 [P] Extend FM engine and store fixture defaults for characterization-friendly args and isolated temp databases in `tests/helpers_fm_detection.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Lock down contracts and introduce the diagnostics-first characterization scaffold before any user story work begins.

**⚠️ CRITICAL**: No user story implementation should begin until this phase is complete.

- [X] T003 [P] Add initial characterization record-shape tests for explicit `raw_*`, `measured_*`, `match_*`, and `display_*` fields in `tests/test_fm_characterization.py`
- [X] T004 [P] Add `/api/jobs` and controller command regression tests proving characterization remains derived and the existing payload shape is unchanged in `tests/test_control_page_scan_settings.py` and `tests/test_control_fm_validation.py`
- [X] T005 [P] Add bounded diagnostic summary contract tests for characterization exports in `tests/test_fm_characterization_diagnostics.py` and `tests/test_web_diagnostics_bundle.py`
- [X] T006 Implement shared diagnostics-first characterization dataclasses and serializers with explicit `raw_*`, `measured_*`, `match_*`, `display_*`, `classification_*`, and `bandplan_*` names in `sdrwatch/detection/types.py` and `sdrwatch/util/detection_diagnostics.py`
- [X] T007 Wire the shared characterization scaffold into coarse-pass detection records without changing persistence schema or Discovery behavior in `sdrwatch/detection/engine.py` and `sdrwatch/sweep/sweeper.py`

**Checkpoint**: The core characterization vocabulary, `/api/jobs` compatibility guardrails, and bounded diagnostics contract are in place.

---

## Phase 3: User Story 1 - Validate Stable FM Cards Without Confusing Widths (Priority: P1) MVP

**Goal**: FM Validation keeps stable station-scale cards while exposing measured bandwidth separately from raw width, match span, and display span.

**Independent Test**: Run deterministic FM-like fixtures plus bundle-export checks, then confirm the GUI/controller validation path in `quickstart.md` still describes stable FM Validation cards and separate raw/measured/match/display evidence.

### Tests for User Story 1

Write these tests first and confirm they fail before implementation.

- [X] T008 [P] [US1] Add wide and spiky FM-like bounded-card tests that also prove measured bandwidth is separate from display width in `tests/test_fm_characterization.py`
- [X] T009 [P] [US1] Add tiny-fragment tests proving narrow FFT fragments do not become fake measured FM bandwidth by themselves in `tests/test_fm_characterization_persistence.py`
- [X] T010 [P] [US1] Add diagnostic bundle tests for separate raw, measured, match, and display span export plus center-within-span regression scanning in `tests/test_web_diagnostics_bundle.py`

### Implementation for User Story 1

- [X] T011 [US1] Populate coarse-pass `raw_*`, `measured_*`, `match_*`, and `display_*` characterization evidence from existing segments and span shaping in `sdrwatch/detection/engine.py`
- [X] T012 [US1] Preserve current FM Validation stable-card behavior while keeping measured bandwidth separate from display span in `sdrwatch/baseline/persistence.py` and `sdrwatch/detection/engine.py`
- [X] T013 [US1] Emit coarse-pass characterization evidence and bounded per-window summaries in `sdrwatch/util/detection_diagnostics.py` and `sdrwatch/sweep/sweeper.py`
- [X] T014 [US1] Run focused US1 no-hardware characterization tests and record the exact command and results in `specs/004-fm-signal-characterization/quickstart.md`

**Checkpoint**: FM Validation remains stable and operator-friendly while the diagnostics surface distinguishes raw, measured, match, and display widths.

---

## Phase 4: User Story 2 - Inspect Characterization Evidence Without Overclaiming FM (Priority: P1)

**Goal**: Contextual bandplan and profile labels remain separate from measured characterization and cautious candidate evidence.

**Independent Test**: Run no-hardware fixtures and bundle-export checks that prove contextual labels, evidence sources, and unknown-candidate behavior remain separate while `/api/jobs` stays unchanged.

### Tests for User Story 2

- [ ] T015 [P] [US2] Add bandplan and profile-context separation plus unknown-candidate retention tests in `tests/test_fm_characterization.py`
- [ ] T016 [P] [US2] Add diagnostic bundle tests for contextual metadata, evidence sources, and bounded classification summaries in `tests/test_fm_characterization_diagnostics.py`
- [ ] T017 [P] [US2] Extend control-page regression coverage proving characterization remains derived rather than user-entered in `tests/test_control_page_scan_settings.py`

### Implementation for User Story 2

- [ ] T018 [US2] Carry `bandplan_*` and `profile_context` separately from measured characterization records in `sdrwatch/io/bandplan.py`, `sdrwatch/detection/engine.py`, and `sdrwatch/baseline/persistence.py`
- [ ] T019 [US2] Add diagnostics-first `classification_candidate`, `classification_evidence`, and evidence-source scaffolding that can remain unknown in `sdrwatch/detection/types.py` and `sdrwatch/detection/engine.py`
- [ ] T020 [US2] Export bounded context-versus-evidence summaries without changing the GUI job contract in `sdrwatch/util/detection_diagnostics.py` and `sdrwatch_web/diagnostics.py`
- [ ] T021 [US2] Run focused US2 diagnostics-first context-separation tests and record the exact command and results in `specs/004-fm-signal-characterization/quickstart.md`

**Checkpoint**: FM-related context remains useful but cannot be mistaken for measured proof or a forced classification.

---

## Phase 5: User Story 3 - Refine Measurements Through Revisit Without Multiplying Cards (Priority: P2)

**Goal**: Revisit-derived evidence improves measured center, measured bandwidth, and confidence without multiplying cards or breaking persistence invariants.

**Independent Test**: Run revisit-heavy fixtures and invariant checks proving measured fields update while card count stays bounded and `f_low_hz <= f_center_hz <= f_high_hz` remains true after coarse updates, revisit confirmation, and hysteresis.

### Tests for User Story 3

- [X] T022 [P] [US3] Add revisit refinement tests for measured center, bandwidth, and confidence updates without extra cards in `tests/test_fm_characterization_persistence.py`
- [X] T023 [P] [US3] Add persistence invariant coverage for coarse updates, revisit confirmation, and hysteresis behavior in `tests/test_extent_hysteresis.py`
- [X] T024 [P] [US3] Add revisit provenance and stability-summary tests in `tests/test_fm_characterization_diagnostics.py`

### Implementation for User Story 3

- [X] T025 [US3] Aggregate revisit-derived `measured_*`, `characterization_confidence`, `center_stability_hz`, and `bandwidth_stability_hz` values in `sdrwatch/baseline/persistence.py` and `sdrwatch/sweep/sweeper.py`
- [X] T026 [US3] Refine measured center, measured bandwidth, and confidence from revisit evidence without multiplying cards in `sdrwatch/detection/engine.py` and `sdrwatch/baseline/persistence.py`
- [X] T027 [US3] Fix any invariant or hysteresis edge cases surfaced by T023 while keeping schema unchanged in `sdrwatch/baseline/persistence.py`
- [X] T028 [US3] Run focused US3 revisit and invariant tests and record the exact command and results in `specs/004-fm-signal-characterization/quickstart.md`

**Checkpoint**: Revisit improves characterization quality without undoing card stability or persistence safety.

---

## Phase 6: User Story 4 - Keep Nearby Stations Separate And Protect Non-FM Signals (Priority: P2)

**Goal**: Nearby FM-like stations remain separate and narrow non-FM signals are not widened or labeled as FM candidates from contextual metadata alone.

**Independent Test**: Run nearby-station and narrow non-FM fixtures plus bundle-export checks proving measured evidence stays distinct and Discovery/FM Validation behavior remains separated.

### Tests for User Story 4

- [ ] T029 [P] [US4] Add nearby FM-like station separation tests for characterization and persistence in `tests/test_fm_characterization.py`
- [ ] T030 [P] [US4] Extend narrow non-FM and no-forced-FM-candidate coverage in `tests/test_non_fm_width_scope.py`
- [ ] T031 [P] [US4] Add diagnostic bundle tests ensuring nearby stations and non-FM evidence remain distinct in `tests/test_web_diagnostics_bundle.py`

### Implementation for User Story 4

- [ ] T032 [US4] Preserve nearby-station separation while emitting measured characterization in `sdrwatch/detection/engine.py` and `sdrwatch/baseline/persistence.py`
- [ ] T033 [US4] Scope FM candidate logic to measured evidence rather than contextual metadata alone in `sdrwatch/detection/engine.py` and `sdrwatch/baseline/persistence.py`
- [ ] T034 [US4] Verify Discovery and FM Validation behavior remain distinct while characterization stays FM-scoped in `tests/test_control_page_scan_settings.py` and `specs/004-fm-signal-characterization/quickstart.md`
- [ ] T035 [US4] Run focused US4 separation and non-FM regression tests and record the exact command and results in `specs/004-fm-signal-characterization/quickstart.md`

**Checkpoint**: Characterization stays honest and stable without merging adjacent stations or overlabeling non-FM signals.

---

## Phase 7: Polish & Cross-Cutting Concerns

**Purpose**: Finalize diagnostics-first documentation, preserve first-pass scope, and record hardware validation guidance.

- [ ] T036 [P] Update diagnostics-first implementation notes and explicitly defer schema migration unless later justified in `specs/004-fm-signal-characterization/plan.md` and `specs/004-fm-signal-characterization/data-model.md`
- [ ] T037 [P] Update the hardware validation checklist and bounded-export acceptance notes in `specs/004-fm-signal-characterization/quickstart.md` and `specs/004-fm-signal-characterization/contracts/diagnostic-characterization-evidence-contract.md`
- [ ] T038 [P] Record 19 kHz pilot and 57 kHz RDS/RBDS work as optional or deferred for the first pass in `specs/004-fm-signal-characterization/research.md` and `specs/004-fm-signal-characterization/plan.md`
- [ ] T039 Run the full focused no-hardware regression command and record the exact result in `specs/004-fm-signal-characterization/quickstart.md`
- [ ] T040 Perform GUI/controller hardware validation for FM Validation and Discovery and record characterization findings in `specs/004-fm-signal-characterization/quickstart.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Phase 1 Setup**: No dependencies.
- **Phase 2 Foundational**: Depends on Phase 1; blocks all user-story implementation.
- **US1 (Phase 3)**: Depends on Phase 2 and is the MVP.
- **US2 (Phase 4)**: Depends on Phase 2 and can proceed in parallel with US1 once characterization field names are stable.
- **US3 (Phase 5)**: Depends on Phase 2 and the coarse characterization scaffold from US1.
- **US4 (Phase 6)**: Depends on Phase 2 and should follow once US1 and US2 have stabilized measured-evidence separation.
- **Polish (Phase 7)**: Depends on the user stories selected for the first release candidate.

### User Story Dependencies

- **User Story 1 (P1)**: Establishes the MVP by proving stable FM cards and separate measured bandwidth evidence.
- **User Story 2 (P1)**: Protects evidence honesty and can be delivered alongside US1 once the foundational scaffold exists.
- **User Story 3 (P2)**: Builds on the coarse characterization scaffold to add revisit refinement and stability aggregation.
- **User Story 4 (P2)**: Builds on the earlier scaffold to protect nearby-station separation and non-FM behavior.

### Implementation Order

1. Complete Phase 1 and Phase 2.
2. Complete US1 and validate the MVP independently.
3. Complete US2 to lock in context-versus-evidence separation before expanding refinement logic.
4. Complete US3 for revisit-driven refinement and invariants.
5. Complete US4 to protect adjacent-station and non-FM regression boundaries.
6. Finish Phase 7 documentation, no-hardware validation, and hardware handoff notes.

---

## Parallel Opportunities

- T001 and T002 can run in parallel.
- T003, T004, and T005 can run in parallel once the setup helpers are available.
- T008, T009, and T010 can run in parallel because they target different test files.
- T015, T016, and T017 can run in parallel because they cover different surfaces.
- T022, T023, and T024 can run in parallel because they touch different revisit and diagnostics files.
- T029, T030, and T031 can run in parallel because they cover different regression files.
- T036, T037, and T038 can run in parallel during documentation polish.

## Parallel Example: User Story 1

```text
Task: "Add wide and spiky FM-like bounded-card tests that also prove measured bandwidth is separate from display width in tests/test_fm_characterization.py"
Task: "Add tiny-fragment tests proving narrow FFT fragments do not become fake measured FM bandwidth by themselves in tests/test_fm_characterization_persistence.py"
Task: "Add diagnostic bundle tests for separate raw, measured, match, and display span export plus center-within-span regression scanning in tests/test_web_diagnostics_bundle.py"
```

## Parallel Example: User Story 2

```text
Task: "Add bandplan and profile-context separation plus unknown-candidate retention tests in tests/test_fm_characterization.py"
Task: "Add diagnostic bundle tests for contextual metadata, evidence sources, and bounded classification summaries in tests/test_fm_characterization_diagnostics.py"
Task: "Extend control-page regression coverage proving characterization remains derived rather than user-entered in tests/test_control_page_scan_settings.py"
```

## Parallel Example: User Story 3

```text
Task: "Add revisit refinement tests for measured center, bandwidth, and confidence updates without extra cards in tests/test_fm_characterization_persistence.py"
Task: "Add persistence invariant coverage for coarse updates, revisit confirmation, and hysteresis behavior in tests/test_extent_hysteresis.py"
Task: "Add revisit provenance and stability-summary tests in tests/test_fm_characterization_diagnostics.py"
```

## Parallel Example: User Story 4

```text
Task: "Add nearby FM-like station separation tests for characterization and persistence in tests/test_fm_characterization.py"
Task: "Extend narrow non-FM and no-forced-FM-candidate coverage in tests/test_non_fm_width_scope.py"
Task: "Add diagnostic bundle tests ensuring nearby stations and non-FM evidence remain distinct in tests/test_web_diagnostics_bundle.py"
```

---

## Implementation Strategy

### MVP First

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3, User Story 1.
3. Stop and validate US1 independently through focused no-hardware tests plus the GUI/controller checklist in `specs/004-fm-signal-characterization/quickstart.md`.

### Incremental Delivery

1. **MVP**: US1 preserves stable FM Validation cards while exposing separate measured bandwidth evidence.
2. **Evidence safety**: US2 keeps contextual metadata separate from measured evidence and cautious candidate scaffolding.
3. **Refinement**: US3 adds revisit-driven measurement improvement and stability aggregation.
4. **Regression protection**: US4 proves nearby-station separation and non-FM protections still hold.
5. **Polish**: finalize no-hardware commands, hardware validation checklist, and first-pass deferrals.

### Notes

- Tests must be written before implementation tasks in each story.
- Prioritize diagnostics-first implementation and keep the first pass schema-conservative.
- No schema migration tasks are included in this first pass. If persistent characterization storage later becomes unavoidable, add a short written justification task in `specs/004-fm-signal-characterization/plan.md` before any migration work is attempted.
- Preserve `/api/jobs` shape and controller ownership of scanner command generation.
- Preserve RTL-SDR v4 Discovery first-light behavior and current FM Validation stable-card behavior.
- Do not implement full modulation classification in this pass.
- Do not implement 19 kHz pilot or 57 kHz RDS/RBDS detection in this pass unless it is explicitly reintroduced as optional follow-on work.
