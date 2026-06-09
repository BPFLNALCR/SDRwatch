# Tasks: Reproducible Python Development Setup

**Input**: Design documents from `/specs/002-python-dev-setup/`

**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, data-model.md, contracts/

**Tests**: Verification is REQUIRED. This feature requires automated no-hardware smoke tests and packaging-metadata checks in `tests/`, plus reproducible Linux/Pi manual validation using `specs/002-python-dev-setup/quickstart.md`.

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Path Conventions

- **Canonical Python metadata**: `pyproject.toml`
- **Contributor and operator docs**: `README.md`
- **Automated validation**: `tests/` and `.github/workflows/ci.yml`
- **Deployment and Pi compatibility**: `install-sdrwatch.sh`
- **Manual validation scenarios**: `specs/002-python-dev-setup/quickstart.md`

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Create the canonical packaging and validation targets that every story will build on.

- [X] T001 Create the root `pyproject.toml` file with the build-system section and placeholder project metadata for SDRwatch
- [X] T002 [P] Create the shared no-hardware smoke-test module in `tests/test_import_smoke.py`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Establish the shared install target and validation scaffolding that MUST exist before story-specific work.

**⚠️ CRITICAL**: No user story work should begin until this phase is complete.

- [X] T003 Populate `pyproject.toml` with the shared project name, Python version floor, and package discovery for `sdrwatch/` and `sdrwatch_web/`
- [X] T004 [P] Audit and update `sdrwatch.py`, `sdrwatch-control.py`, `sdrwatch-web.py`, and `sdrwatch_web/app.py` to keep existing launch surfaces import-safe under editable install
- [X] T005 [P] Seed the shared local-development and deployment guidance sections in `README.md` from `specs/002-python-dev-setup/quickstart.md`

**Checkpoint**: Canonical metadata, launch surfaces, and documentation scaffolding are ready for story work.

---

## Phase 3: User Story 1 - Set Up a Local Environment (Priority: P1) 🎯 MVP

**Goal**: Give contributors a clean-checkout editable-install workflow with no-hardware CLI and web smoke validation.

**Independent Test**: From a fresh Linux/Pi or Windows venv, install SDRwatch from repository metadata, import `sdrwatch.cli`, import `sdrwatch_web.create_app`, and run the documented no-hardware smoke commands without invoking `install-sdrwatch.sh`.

### Verification for User Story 1 (REQUIRED)

- [X] T006 [P] [US1] Complete automated CLI and web smoke coverage in `tests/test_import_smoke.py` for `sdrwatch.cli` import, `sdrwatch_web.create_app` import, and no-hardware app-factory creation
- [X] T007 [P] [US1] Define the Linux/Pi and Windows no-hardware verification commands in `README.md` using the scenarios from `specs/002-python-dev-setup/quickstart.md`

### Implementation for User Story 1

- [X] T008 [US1] Implement the base editable-install contract and no-hardware runtime dependency set in `pyproject.toml`
- [X] T009 [US1] Document fresh venv creation, editable install, supported-platform matrix, CLI smoke test, and web smoke test in `README.md`
- [X] T010 [US1] Fix any no-hardware import or path regressions uncovered by T006-T009 in `sdrwatch.py`, `sdrwatch-control.py`, `sdrwatch-web.py`, and `sdrwatch_web/app.py`

**Checkpoint**: Contributors can install SDRwatch from a clean venv and run no-hardware smoke checks without the installer.

---

## Phase 4: User Story 2 - Use the Right Dependency Scope (Priority: P2)

**Goal**: Separate runtime, development, and OS-managed prerequisites so contributor installs stay complete and Pi deployments stay lean.

**Independent Test**: A maintainer can perform a runtime-only install from `pyproject.toml`, confirm the documented CLI and web launch/import commands work, then install the development extras and run the documented no-hardware pytest suite.

### Verification for User Story 2 (REQUIRED)

- [X] T011 [P] [US2] Add packaging-metadata contract coverage in `tests/test_pyproject_contract.py` for base dependencies, optional development dependencies, and exclusion of OS-managed SDR prerequisites from the base install set
- [X] T012 [P] [US2] Run the runtime-only versus development-install scenarios from `specs/002-python-dev-setup/quickstart.md` and capture required dependency-boundary clarifications in `README.md`

### Implementation for User Story 2

- [X] T013 [US2] Finalize optional dependency groups in `pyproject.toml` for contributor tooling and any optional Linux/Pi hardware helpers
- [X] T014 [US2] Document runtime-only install, development install, and OS-managed prerequisites such as SciPy, librtlsdr, SoapySDR, libusb, and udev rules in `README.md`

**Checkpoint**: Maintainers can distinguish Python-managed runtime and dev dependencies from Linux/Pi OS-managed hardware prerequisites.

---

## Phase 5: User Story 3 - Preserve Installer and CI Alignment (Priority: P3)

**Goal**: Keep the Raspberry Pi installer workflow intact while making CI and documentation consume the same canonical dependency contract.

**Independent Test**: The installer still provisions a Pi-oriented environment without a second dependency truth source, and the existing CI workflow installs from `pyproject.toml` and runs the documented no-hardware pytest and smoke commands.

### Verification for User Story 3 (REQUIRED)

- [X] T015 [US3] Run the installer-compatibility and Linux/Pi hardware-boundary scenarios from `specs/002-python-dev-setup/quickstart.md` and capture required adjustments in `README.md`, `install-sdrwatch.sh`, and `.github/workflows/ci.yml`

### Implementation for User Story 3

- [X] T016 [P] [US3] Update `install-sdrwatch.sh` to install Python dependencies from `pyproject.toml` while preserving APT provisioning, `--system-site-packages`, sanity checks, and systemd setup
- [X] T017 [P] [US3] Update `.github/workflows/ci.yml` to install from the canonical metadata and run the documented no-hardware pytest and CLI/web smoke commands on supported runners
- [X] T018 [US3] Update `README.md` to preserve `install-sdrwatch.sh` as the Raspberry Pi deployment path and align installer, CI, and manual hardware verification guidance with the canonical metadata contract

**Checkpoint**: Installer, CI, and documentation all point at the same dependency source and validation contract.

---

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Final consistency, contract review, and end-to-end validation.

- [X] T019 Cross-check `pyproject.toml`, `README.md`, `install-sdrwatch.sh`, `.github/workflows/ci.yml`, `tests/test_import_smoke.py`, and `tests/test_pyproject_contract.py` against `specs/002-python-dev-setup/contracts/development-setup-contract.md` and `specs/002-python-dev-setup/spec.md`
- [X] T020 Run the validation scenarios from `specs/002-python-dev-setup/quickstart.md` and apply final corrections in `pyproject.toml`, `README.md`, `install-sdrwatch.sh`, `.github/workflows/ci.yml`, `tests/test_import_smoke.py`, and `tests/test_pyproject_contract.py`

---

## Dependencies & Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies; start immediately.
- **Foundational (Phase 2)**: Depends on Setup; blocks all user stories.
- **User Stories (Phases 3-5)**: Depend on Foundational completion.
- **Polish (Phase 6)**: Depends on all desired user stories being complete.

### User Story Dependencies

- **User Story 1 (P1)**: Starts after Foundational; delivers the MVP contributor workflow.
- **User Story 2 (P2)**: Starts after Foundational; remains independently testable, but it refines `pyproject.toml` and `README.md` established in US1.
- **User Story 3 (P3)**: Starts after Foundational; remains independently testable, but it aligns the installer and CI to the canonical metadata created in US1 and refined in US2.

### Within Each User Story

- Automated tests or contract checks before implementation when practical.
- Manual verification steps defined from `specs/002-python-dev-setup/quickstart.md` before changing installer or hardware-boundary behavior.
- Canonical metadata before dependent docs, CI, or installer alignment.
- Story completion requires both the story-specific verification tasks and the implementation tasks.

### Parallel Opportunities

- `T002` can run in parallel with `T001` because it creates a separate test file.
- `T004` and `T005` can run in parallel after `T003` because they touch separate launch and documentation surfaces.
- `T006` and `T007` can run in parallel within US1 because they touch different files.
- `T011` and `T012` can run in parallel within US2 because they touch different files.
- `T016` and `T017` can run in parallel within US3 because installer and CI updates are isolated to different files.

---

## Parallel Example: User Story 1

```text
Task: "Complete automated CLI and web smoke coverage in tests/test_import_smoke.py"
Task: "Define the Linux/Pi and Windows no-hardware verification commands in README.md"
```

## Parallel Example: User Story 2

```text
Task: "Add packaging-metadata contract coverage in tests/test_pyproject_contract.py"
Task: "Run the runtime-only versus development-install scenarios from specs/002-python-dev-setup/quickstart.md and capture clarifications in README.md"
```

## Parallel Example: User Story 3

```text
Task: "Update install-sdrwatch.sh to install Python dependencies from pyproject.toml"
Task: "Update .github/workflows/ci.yml to install from the canonical metadata and run the documented no-hardware checks"
```

---

## Implementation Strategy

### MVP First (User Story 1 Only)

1. Complete Phase 1: Setup.
2. Complete Phase 2: Foundational.
3. Complete Phase 3: User Story 1.
4. Validate the fresh-vnev contributor workflow on Linux/Pi and Windows no-hardware setups.

### Incremental Delivery

1. Deliver the editable-install and smoke-test workflow first (US1).
2. Add dependency-scope separation and OS-managed prerequisite guidance next (US2).
3. Align installer and CI last without breaking the Pi deployment path (US3).
4. Run the cross-cutting validation pass in Phase 6.

### Team Strategy

1. One contributor can own `pyproject.toml` and packaging tests.
2. One contributor can own `README.md` and quickstart-aligned verification text.
3. One contributor can own installer and CI alignment once the canonical metadata is stable.

---

## Notes

- Every task follows the required checklist format with task ID, optional parallel marker, story label where required, and explicit file paths.
- This feature keeps the existing runtime topology intact; tasks focus on packaging, documentation, validation, installer compatibility, and CI alignment.
- Linux/Pi hardware verification remains manual and separate from no-hardware contributor and CI workflows.