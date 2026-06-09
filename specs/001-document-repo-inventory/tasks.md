# Tasks: Repository Inventory and Stabilization

**Input**: Design documents from `/specs/001-document-repo-inventory/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Verification is REQUIRED. This feature is documentation-only, so tasks use reproducible manual validation instead of new automated tests.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Documentation output**: `docs/PROJECT_INVENTORY.md`
- **Feature artifacts**: `specs/001-document-repo-inventory/`
- **Runtime code evidence**: `sdrwatch/`, `sdrwatch_web/`, root entrypoint scripts, `templates/`, `static/`, and `.github/workflows/`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the inventory document target and align it to the planned contract before writing content.

- [X] T001 Create the `docs/` directory and seed `docs/PROJECT_INVENTORY.md` with the feature title and top-level section headings from `specs/001-document-repo-inventory/contracts/project-inventory-contract.md`
- [X] T002 Align the initial section order and scope notes in `docs/PROJECT_INVENTORY.md` with `specs/001-document-repo-inventory/plan.md` and `specs/001-document-repo-inventory/spec.md`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish shared documentation conventions that every user story depends on.

**⚠️ CRITICAL**: No user story work should begin until this phase is complete.

- [X] T003 Establish the purpose, scope, evidence rules, and committed-vs-generated artifact vocabulary in `docs/PROJECT_INVENTORY.md`
- [X] T004 Add the shared scaffold for runtime topology, schema ownership, deployment artifacts, workflow guidance, tests, CI, and repository gaps in `docs/PROJECT_INVENTORY.md`

**Checkpoint**: The target document is structured and ready for story-specific content.

---

## Phase 3: User Story 1 - Understand Current Runtime Topology (Priority: P1) 🎯 MVP

**Goal**: Give contributors a single document that identifies the main packages, scripts, authoritative entry points, and compatibility paths.

**Independent Test**: A contributor can read `docs/PROJECT_INVENTORY.md` and identify the canonical scanner, controller, web, and query entry points plus any compatibility shims without browsing source files.

### Verification for User Story 1 (REQUIRED)

- [X] T005 [US1] Verify the runtime topology claims in `docs/PROJECT_INVENTORY.md` against `sdrwatch/cli.py`, `sdrwatch.py`, `sdrwatch-control.py`, `sdrwatch-web.py`, and `query-sdrwatch.py`

### Implementation for User Story 1

- [X] T006 [US1] Document the top-level repository layout and main package or script responsibilities in `docs/PROJECT_INVENTORY.md`
- [X] T007 [US1] Document canonical entry points and compatibility shims in `docs/PROJECT_INVENTORY.md`
- [X] T008 [US1] Document scanner, controller, web, and direct-database runtime topology and layer ownership in `docs/PROJECT_INVENTORY.md`

**Checkpoint**: Contributors can understand where to start and which runtime entry points are authoritative.

---

## Phase 4: User Story 2 - Trace Deployment and Persistence Assets (Priority: P2)

**Goal**: Give maintainers a clear map of schema ownership, runtime artifacts, installer behavior, and service generation.

**Independent Test**: A maintainer can use `docs/PROJECT_INVENTORY.md` to locate schema initialization or migration logic, identify generated service files, and understand which files appear only at install time or runtime.

### Verification for User Story 2 (REQUIRED)

- [X] T009 [US2] Verify the persistence and deployment claims in `docs/PROJECT_INVENTORY.md` against `sdrwatch/baseline/store.py`, `sdrwatch_web/schema.py`, `sdrwatch_web/db.py`, `install-sdrwatch.sh`, and `uninstall-sdrwatch.sh`

### Implementation for User Story 2

- [X] T010 [US2] Document database schema initialization, startup migrations, and split schema ownership in `docs/PROJECT_INVENTORY.md`
- [X] T011 [US2] Document installer behavior, generated systemd units, generated environment files, deployment paths, and uninstall-preserved state in `docs/PROJECT_INVENTORY.md`
- [X] T012 [US2] Document runtime-generated artifacts, test layout, and CI surface in `docs/PROJECT_INVENTORY.md`

**Checkpoint**: Maintainers can trace deployment and persistence assets without rediscovering them from source.

---

## Phase 5: User Story 3 - Separate No-Hardware and Hardware Workflows (Priority: P3)

**Goal**: Make workstation-safe workflows, Raspberry Pi workflows, and current repository gaps explicit.

**Independent Test**: A developer can read `docs/PROJECT_INVENTORY.md` and distinguish no-hardware discovery workflows from Pi and RTL-SDR deployment workflows, while also spotting absent standard project files.

### Verification for User Story 3 (REQUIRED)

- [X] T013 [US3] Verify the workflow and repository-gap claims in `docs/PROJECT_INVENTORY.md` against `specs/001-document-repo-inventory/quickstart.md`, `README.md`, and `.github/workflows/ci.yml`

### Implementation for User Story 3

- [X] T014 [US3] Document missing standard project files, missing committed deployment assets, and generated-only artifacts in `docs/PROJECT_INVENTORY.md`
- [X] T015 [US3] Document local no-hardware workflows, supported read-only paths, and current limitations in `docs/PROJECT_INVENTORY.md`
- [X] T016 [US3] Document Raspberry Pi with RTL-SDR installation, service startup, and operational workflow in `docs/PROJECT_INVENTORY.md`

**Checkpoint**: Contributors can choose the right validation path for a workstation or a Raspberry Pi deployment.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final accuracy review and validation of the completed inventory document.

- [X] T017 Cross-check `docs/PROJECT_INVENTORY.md` against `specs/001-document-repo-inventory/contracts/project-inventory-contract.md` and `specs/001-document-repo-inventory/spec.md` for completeness, accuracy, and scope control
- [X] T018 Run the validation scenarios from `specs/001-document-repo-inventory/quickstart.md` and apply any final corrections in `docs/PROJECT_INVENTORY.md`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; start immediately.
- **Foundational (Phase 2)**: Depends on Setup; blocks all user stories.
- **User Stories (Phases 3-5)**: Depend on Foundational completion.
- **Polish (Phase 6)**: Depends on completion of all user story phases.

### User Story Dependencies

- **User Story 1 (P1)**: Starts after Foundational; defines the MVP documentation slice.
- **User Story 2 (P2)**: Starts after Foundational; independent of US1 in scope, but all edits still land in the same output document.
- **User Story 3 (P3)**: Starts after Foundational; independent of US1 and US2 in scope, but all edits still land in the same output document.

### Within Each User Story

- Verification first, then implementation.
- Shared-document edits should be merged sequentially to avoid conflicts in `docs/PROJECT_INVENTORY.md`.
- Story completion requires both content updates and the story's independent verification check.

### Parallel Opportunities

- No checklist tasks are marked `[P]` because the implementation converges on a single file: `docs/PROJECT_INVENTORY.md`.
- Evidence gathering from source files can happen informally in parallel, but document updates should be serialized.

---

## Parallel Example: User Story 1

```text
No safe [P] tasks identified for User Story 1 because repository-layout, entry-point, and runtime-topology work all update docs/PROJECT_INVENTORY.md.
```

## Parallel Example: User Story 2

```text
No safe [P] tasks identified for User Story 2 because schema, installer, and runtime-artifact content all update docs/PROJECT_INVENTORY.md.
```

## Parallel Example: User Story 3

```text
No safe [P] tasks identified for User Story 3 because workflow and repository-gap content all update docs/PROJECT_INVENTORY.md.
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational.
3. Complete Phase 3: User Story 1.
4. Validate that a contributor can identify authoritative entry points from `docs/PROJECT_INVENTORY.md` alone.

### Incremental Delivery

1. Deliver the runtime-topology slice first (US1).
2. Add deployment and persistence coverage next (US2).
3. Finish with workflow separation and missing-file guidance (US3).
4. Run the cross-cutting validation pass in Phase 6.

### Team Strategy

1. One contributor should own the final edits to `docs/PROJECT_INVENTORY.md`.
2. Other contributors can gather evidence from source files in parallel, then hand findings to the document owner for merge.

---

## Notes

- Every task follows the required checklist format with task ID, optional labels, and explicit file paths.
- This feature is documentation-only; tasks intentionally avoid runtime rewrites, packaging changes, or schema changes.
- Validation relies on the existing quickstart scenarios and repository evidence rather than new test code.