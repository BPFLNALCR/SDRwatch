# Tasks: No-Hardware Simulation Mode

**Input**: Design documents from `/specs/003-add-sim-mode/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Verification is REQUIRED. This feature requires automated no-hardware tests for deterministic simulated output and SQLite writes in `tests/`, plus reproducible manual validation using `specs/003-add-sim-mode/quickstart.md`.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Scanner and driver code**: `sdrwatch/cli.py`, `sdrwatch/drivers/`, `sdrwatch/sweep/`
- **Persistence and pipeline reuse**: `sdrwatch/baseline/`, `sdrwatch/detection/`, `sdrwatch/dsp/`
- **Web/dashboard validation surfaces**: `sdrwatch_web/`, `templates/`, and `README.md`
- **Automated validation**: `tests/` and `.github/workflows/ci.yml`
- **Manual verification scenarios**: `specs/003-add-sim-mode/quickstart.md`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the feature-specific validation scaffolding and documentation anchors used by all stories.

- [X] T001 Create the simulation-focused test module `tests/test_sim_mode.py` with placeholders for deterministic source and DB-write coverage
- [X] T002 [P] Add the simulation-mode documentation anchor to `README.md` referencing `specs/003-add-sim-mode/quickstart.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish the shared simulation input boundary and scanner wiring that all user stories depend on.

**⚠️ CRITICAL**: No user story work should begin until this phase is complete.

- [X] T003 Add the synthetic driver module `sdrwatch/drivers/simulate.py` implementing the shared `tune`, `read`, and `close` source contract
- [X] T004 [P] Update `sdrwatch/sweep/runner.py` to dispatch explicit simulation mode while preserving existing physical-driver behavior and baseline-first execution
- [X] T005 [P] Update `sdrwatch/cli.py` help text and driver-facing argument messaging to document the explicit simulation selector without changing default real-hardware behavior
- [X] T006 Define shared simulation test helpers for temp databases and baseline creation in `tests/test_sim_mode.py`

**Checkpoint**: The scanner can select a synthetic source through the normal runner boundary, and all story work can build on that additive path.

---

## Phase 3: User Story 1 - Run a Simulated Scan (Priority: P1) 🎯 MVP

**Goal**: Let developers run a deterministic simulated scan through the normal PSD, detection, baseline, and persistence workflow without SDR hardware.

**Independent Test**: Create a baseline, run the documented simulated FM-band scan on a hardware-free machine, and confirm `scan_updates`, `baseline_detections`, and baseline stats rows are written to the normal database.

### Verification for User Story 1 (REQUIRED)

- [X] T007 [P] [US1] Add deterministic source-behavior tests in `tests/test_sim_mode.py` covering noise floor, stable carriers, intermittent signal cadence, and power-shift sequencing
- [X] T008 [P] [US1] Add end-to-end persistence tests in `tests/test_sim_mode.py` covering simulated scan writes to `baseline_noise`, `baseline_occupancy`, `baseline_detections`, and `scan_updates`
- [X] T009 [US1] Define the manual simulated FM-band CLI verification steps in `specs/003-add-sim-mode/quickstart.md` with commands and expected database outcomes

### Implementation for User Story 1

- [X] T010 [US1] Implement deterministic synthetic sample generation and sweep/window state handling in `sdrwatch/drivers/simulate.py`
- [X] T011 [US1] Wire the simulation driver through `sdrwatch/sweep/runner.py` and any necessary exports in `sdrwatch/drivers/__init__.py`
- [X] T012 [US1] Update `README.md` to document the explicit simulated scan CLI flow and clarify that physical drivers still require real hardware
- [X] T013 [US1] Fix any scan-path regressions uncovered by T007-T012 in `sdrwatch/cli.py`, `sdrwatch/sweep/runner.py`, and `sdrwatch/drivers/simulate.py`

**Checkpoint**: A developer can run a deterministic simulated scan without SDR hardware and persist normal baseline-oriented results.

---

## Phase 4: User Story 2 - Review Simulated Results in the Dashboard (Priority: P2)

**Goal**: Ensure the database produced by simulated scans is consumable by the existing dashboard without simulation-specific schema or rendering logic.

**Independent Test**: Populate a database with simulated scans, open the web UI against that database, and confirm the existing views show detections, baseline summaries, and recent activity.

### Verification for User Story 2 (REQUIRED)

- [X] T014 [P] [US2] Add dashboard-data compatibility assertions in `tests/test_sim_mode.py` for the persisted artifacts consumed by existing baseline helper queries
- [X] T015 [US2] Define the manual web-dashboard verification steps in `specs/003-add-sim-mode/quickstart.md` for opening `sim-sdrwatch.db` and confirming simulated detections appear

### Implementation for User Story 2

- [X] T016 [US2] Audit and adjust any persistence-side assumptions needed for simulated scans in `sdrwatch/baseline/store.py`, `sdrwatch/baseline/persistence.py`, and `sdrwatch/detection/engine.py` while keeping schema unchanged
- [X] T017 [US2] Update `README.md` to document that simulated scans populate the standard dashboard-visible database paths and do not require special web configuration
- [X] T018 [US2] Fix any dashboard-compatibility regressions uncovered by T014-T017 in `sdrwatch_web/baseline_helpers.py`, `sdrwatch_web/blueprints/views.py`, and the owning persistence layer files

**Checkpoint**: Databases produced by simulated scans are readable by the current dashboard and baseline helper queries without simulation-only paths.

---

## Phase 5: User Story 3 - Run Hardware-Free Automated Validation (Priority: P3)

**Goal**: Let maintainers run deterministic simulation validation in CI and local no-hardware environments without SDR dependencies.

**Independent Test**: Run the simulation-focused test suite in a hardware-free environment and confirm it succeeds without attached SDR devices or hardware-only dependency requirements.

### Verification for User Story 3 (REQUIRED)

- [X] T019 [P] [US3] Add repeat-run determinism checks in `tests/test_sim_mode.py` that compare fresh-baseline outcomes across identical simulated scan runs
- [X] T020 [US3] Define the no-hardware automated validation workflow in `specs/003-add-sim-mode/quickstart.md` and `README.md` for local and CI use

### Implementation for User Story 3

- [X] T021 [US3] Update `.github/workflows/ci.yml` to run the no-hardware simulation validation slice without requiring SDR hardware
- [X] T022 [US3] Update `README.md` to document the simulation-focused pytest command and CI-safe no-hardware expectations
- [X] T023 [US3] Fix any CI or test-isolation regressions uncovered by T019-T022 in `tests/test_sim_mode.py`, `.github/workflows/ci.yml`, and related test support code

**Checkpoint**: Maintainers can validate deterministic simulation behavior in CI and local no-hardware environments.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final consistency, regression protection, and operator-facing validation notes.

- [X] T024 Cross-check `sdrwatch/cli.py`, `sdrwatch/sweep/runner.py`, `sdrwatch/drivers/simulate.py`, `tests/test_sim_mode.py`, `README.md`, `.github/workflows/ci.yml`, and `specs/003-add-sim-mode/quickstart.md` against `specs/003-add-sim-mode/contracts/simulation-mode-contract.md`
- [X] T025 [P] Add compatibility coverage in `tests/test_sim_mode.py` or an adjacent test module to confirm explicit physical-driver selections still fail normally without silent simulation fallback
- [X] T026 Run the manual validation scenarios from `specs/003-add-sim-mode/quickstart.md` and apply final corrections in `README.md`, `specs/003-add-sim-mode/quickstart.md`, and the simulation implementation files

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; start immediately.
- **Foundational (Phase 2)**: Depends on Setup; blocks all user stories.
- **User Stories (Phases 3-5)**: Depend on Foundational completion.
- **Polish (Phase 6)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: Starts after Foundational; delivers the MVP no-hardware scan path.
- **User Story 2 (P2)**: Starts after Foundational and depends on US1-generated persisted artifacts conceptually, but remains independently testable against a simulated database.
- **User Story 3 (P3)**: Starts after Foundational and can proceed once the simulation path exists; it formalizes automated validation rather than adding new product behavior.

### Within Each User Story

- Automated tests or explicit manual verification steps should be defined before implementation when practical.
- Driver/source contract work before runner integration fixes.
- Scan-path implementation before docs polish and CI alignment.
- Story completion requires both verification tasks and implementation tasks.

### Parallel Opportunities

- `T002` can run in parallel with `T001` because documentation and test scaffolding are separate.
- `T004` and `T005` can run in parallel after `T003` because runner wiring and CLI messaging touch different files.
- `T007` and `T008` can run in parallel within US1 because deterministic source tests and persistence assertions can be authored independently in the same test module after helper scaffolding exists.
- `T014` and `T015` can run in parallel within US2 because automated compatibility checks and manual dashboard verification touch different artifacts.
- `T021` and `T022` can run in parallel within US3 because CI wiring and README updates are isolated to different files.
- `T025` can run in parallel with `T024` during polish because compatibility coverage is isolated from contract cross-check notes.

---

## Parallel Example: User Story 1

```text
Task: "Add deterministic source-behavior tests in tests/test_sim_mode.py covering noise floor, stable carriers, intermittent signal cadence, and power-shift sequencing"
Task: "Add end-to-end persistence tests in tests/test_sim_mode.py covering simulated scan writes to baseline_noise, baseline_occupancy, baseline_detections, and scan_updates"
```

## Parallel Example: User Story 2

```text
Task: "Add dashboard-data compatibility assertions in tests/test_sim_mode.py for the persisted artifacts consumed by existing baseline helper queries"
Task: "Define the manual web-dashboard verification steps in specs/003-add-sim-mode/quickstart.md for opening sim-sdrwatch.db and confirming simulated detections appear"
```

## Parallel Example: User Story 3

```text
Task: "Update .github/workflows/ci.yml to run the no-hardware simulation validation slice without requiring SDR hardware"
Task: "Update README.md to document the simulation-focused pytest command and CI-safe no-hardware expectations"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational.
3. Complete Phase 3: User Story 1.
4. Validate the simulated FM-band CLI workflow against a fresh baseline and database.

### Incremental Delivery

1. Deliver the deterministic simulation scan path first (US1).
2. Confirm dashboard compatibility using the generated database next (US2).
3. Add CI and automated no-hardware validation alignment last (US3).
4. Run the cross-cutting compatibility and manual validation pass in Phase 6.

### Team Strategy

1. One contributor can own the synthetic driver and runner wiring.
2. One contributor can own the simulation-focused tests and persistence assertions.
3. One contributor can own README, quickstart, and CI alignment once the simulation contract is stable.

---

## Notes

- Every task follows the required checklist format with task ID, optional parallel marker, story label where required, and explicit file paths.
- This feature keeps the existing runtime topology intact; tasks focus on the driver boundary, scan-path reuse, documentation, and no-hardware validation.
- Controller-discovered simulation devices are intentionally out of the MVP scope unless implementation reveals they are required to satisfy acceptance criteria.