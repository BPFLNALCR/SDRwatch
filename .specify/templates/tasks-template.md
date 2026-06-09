---

description: "Task list template for feature implementation"
---

# Tasks: [FEATURE NAME]

**Input**: Design documents from `/specs/[###-feature-name]/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Verification is REQUIRED. Include automated tests whenever practical, and
add a reproducible manual verification task whenever behavior changes or hardware,
deployment, or offline operation is involved.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Scanner and persistence code**: `sdrwatch/` and its subpackages
- **Web and API glue**: `sdrwatch_web/`, root controller or web entrypoints,
  `templates/`, and `static/`
- **Tests and validation**: `tests/` plus documented manual verification commands when
  hardware or deployment behavior is affected
- **Deployment and tooling**: root-level scripts such as `install-sdrwatch.sh` and
  related operational documentation

<!--
  ============================================================================
  IMPORTANT: The tasks below are SAMPLE TASKS for illustration purposes only.

  The /speckit.tasks command MUST replace these with actual tasks based on:
  - User stories from spec.md (with their priorities P1, P2, P3...)
  - Feature requirements from plan.md
  - Entities from data-model.md
  - Endpoints from contracts/

  Tasks MUST be organized by user story so each story can be:
  - Implemented independently
  - Tested independently
  - Delivered as an MVP increment

  DO NOT keep these sample tasks in the generated tasks.md file.
  ============================================================================
-->

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [ ] T001 Create project structure per implementation plan
- [ ] T002 Initialize [language] project with [framework] dependencies
- [ ] T003 [P] Configure linting and formatting tools

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

**⚠️ CRITICAL**: No user story work can begin until this phase is complete

Examples of foundational tasks (adjust based on your project):

- [ ] T004 Define migration-safe SQLite schema or upgrade steps for impacted tables
- [ ] T005 [P] Enforce `SDRWATCH_CONTROL_TOKEN` behavior for changed protected paths
- [ ] T006 [P] Establish controller, scanner, and web boundaries for the feature
- [ ] T007 Create or update shared models, schema helpers, or adapters required by all stories
- [ ] T008 Configure error handling, structured logging, and operator-visible diagnostics
- [ ] T009 Setup environment, deployment, or offline configuration changes

**Checkpoint**: Foundation ready - user story implementation can now begin in parallel

---

## Phase 3: User Story 1 - [Title] (Priority: P1) 🎯 MVP

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Verification for User Story 1 (REQUIRED) ⚠️

> **NOTE: Write automated tests FIRST when practical. If automation is not practical,
> define the exact manual verification workflow before implementation.**

- [ ] T010 [P] [US1] Contract test for [endpoint] in tests/contract/test_[name].py
- [ ] T011 [P] [US1] Integration test for [user journey] in tests/integration/test_[name].py
- [ ] T012 [US1] Manual verification for [CLI/API/hardware flow] with commands, expected output, and environment notes

### Implementation for User Story 1

- [ ] T013 [P] [US1] Create or update [Entity1] model in sdrwatch/[area]/[entity1].py
- [ ] T014 [P] [US1] Create or update [Entity2] model in sdrwatch/[area]/[entity2].py
- [ ] T015 [US1] Implement [Service] in sdrwatch/[area]/[service].py (depends on T013, T014)
- [ ] T016 [US1] Implement [endpoint/feature] in sdrwatch_web/[location]/[file].py or the owning layer entrypoint
- [ ] T017 [US1] Add validation, auth, and compatibility handling
- [ ] T018 [US1] Add logging and operator diagnostics for user story 1 operations

**Checkpoint**: At this point, User Story 1 should be fully functional and testable independently

---

## Phase 4: User Story 2 - [Title] (Priority: P2)

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Verification for User Story 2 (REQUIRED) ⚠️

- [ ] T019 [P] [US2] Contract test for [endpoint] in tests/contract/test_[name].py
- [ ] T020 [P] [US2] Integration test for [user journey] in tests/integration/test_[name].py
- [ ] T021 [US2] Manual verification for [CLI/API/hardware flow] with commands, expected output, and environment notes

### Implementation for User Story 2

- [ ] T022 [P] [US2] Create or update [Entity] model in sdrwatch/[area]/[entity].py
- [ ] T023 [US2] Implement [Service] in sdrwatch/[area]/[service].py
- [ ] T024 [US2] Implement [endpoint/feature] in sdrwatch_web/[location]/[file].py or the owning layer entrypoint
- [ ] T025 [US2] Integrate with User Story 1 components while preserving layer boundaries

**Checkpoint**: At this point, User Stories 1 AND 2 should both work independently

---

## Phase 5: User Story 3 - [Title] (Priority: P3)

**Goal**: [Brief description of what this story delivers]

**Independent Test**: [How to verify this story works on its own]

### Verification for User Story 3 (REQUIRED) ⚠️

- [ ] T026 [P] [US3] Contract test for [endpoint] in tests/contract/test_[name].py
- [ ] T027 [P] [US3] Integration test for [user journey] in tests/integration/test_[name].py
- [ ] T028 [US3] Manual verification for [CLI/API/hardware flow] with commands, expected output, and environment notes

### Implementation for User Story 3

- [ ] T029 [P] [US3] Create or update [Entity] model in sdrwatch/[area]/[entity].py
- [ ] T030 [US3] Implement [Service] in sdrwatch/[area]/[service].py
- [ ] T031 [US3] Implement [endpoint/feature] in sdrwatch_web/[location]/[file].py or the owning layer entrypoint

**Checkpoint**: All user stories should now be independently functional

---

[Add more user story phases as needed, following the same pattern]

---

## Phase N: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [ ] TXXX [P] Documentation updates in docs/
- [ ] TXXX Code cleanup and refactoring
- [ ] TXXX Performance optimization across all stories
- [ ] TXXX [P] Additional automated tests in tests/unit/ or other repo test paths
- [ ] TXXX Security hardening
- [ ] TXXX Validate Raspberry Pi 5 deployment, offline field workflow, and manual verification notes
- [ ] TXXX Run quickstart or operator workflow validation

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
  - User stories can then proceed in parallel (if staffed)
  - Or sequentially in priority order (P1 → P2 → P3)
- **Polish (Final Phase)**: Depends on all desired user stories being complete

### User Story Dependencies

- **User Story 1 (P1)**: Can start after Foundational (Phase 2) - No dependencies on other stories
- **User Story 2 (P2)**: Can start after Foundational (Phase 2) - May integrate with US1 but should be independently testable
- **User Story 3 (P3)**: Can start after Foundational (Phase 2) - May integrate with US1/US2 but should be independently testable

### Within Each User Story

- Automated tests MUST be written and FAIL before implementation when practical
- Manual verification steps MUST be defined before implementation when hardware or deployment blocks automation
- Models before services
- Services before endpoints
- Core implementation before integration
- Story complete before moving to next priority

### Parallel Opportunities

- All Setup tasks marked [P] can run in parallel
- All Foundational tasks marked [P] can run in parallel (within Phase 2)
- Once Foundational phase completes, all user stories can start in parallel (if team capacity allows)
- All tests for a user story marked [P] can run in parallel
- Models within a story marked [P] can run in parallel
- Different user stories can be worked on in parallel by different team members

---

## Parallel Example: User Story 1

```bash
# Launch all tests for User Story 1 together (if tests requested):
Task: "Contract test for [endpoint] in tests/contract/test_[name].py"
Task: "Integration test for [user journey] in tests/integration/test_[name].py"

# Launch all models for User Story 1 together:
Task: "Create [Entity1] model in src/models/[entity1].py"
Task: "Create [Entity2] model in src/models/[entity2].py"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational (CRITICAL - blocks all stories)
3. Complete Phase 3: User Story 1
4. **STOP and VALIDATE**: Test User Story 1 independently
5. Deploy/demo if ready

### Incremental Delivery

1. Complete Setup + Foundational → Foundation ready
2. Add User Story 1 → Test independently → Deploy/Demo (MVP!)
3. Add User Story 2 → Test independently → Deploy/Demo
4. Add User Story 3 → Test independently → Deploy/Demo
5. Each story adds value without breaking previous stories

### Parallel Team Strategy

With multiple developers:

1. Team completes Setup + Foundational together
2. Once Foundational is done:
   - Developer A: User Story 1
   - Developer B: User Story 2
   - Developer C: User Story 3
3. Stories complete and integrate independently

---

## Notes

- [P] tasks = different files, no dependencies
- [Story] label maps task to specific user story for traceability
- Each user story should be independently completable and testable
- Every meaningful change needs automated tests or a reproducible manual verification path
- Preserve CLI compatibility, migration safety, auth behavior, and offline field use in task design
- Commit after each task or logical group
- Stop at any checkpoint to validate story independently
- Avoid: vague tasks, same file conflicts, cross-story dependencies that break independence
