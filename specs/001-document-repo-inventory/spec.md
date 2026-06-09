# Feature Specification: Repository Inventory and Stabilization

**Feature Branch**: `[001-document-repo-inventory]`

**Created**: 2026-06-08

**Status**: Draft

**Input**: User description: "Create a repository inventory and stabilization feature for SDRwatch.

Goal:
Document the current project structure, runtime entry points, dependencies, database files, service files, scripts, tests, and web/API components.

The output should be a new developer-facing document, preferably docs/PROJECT_INVENTORY.md, that explains:
- Main Python packages and scripts
- CLI entry points
- Web dashboard entry points
- Controller/API entry points
- Database schema initialization or migration locations
- Installer behavior
- systemd service files
- test layout, if any
- expected runtime files such as sdrwatch.db and events.jsonl
- known missing standard project files, such as pyproject.toml or requirements.txt if absent
- how to run the project locally without SDR hardware when possible
- how to run it on Raspberry Pi with RTL-SDR hardware

Do not rewrite the application yet. This is discovery and documentation only."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Understand Current Runtime Topology (Priority: P1)

As a contributor, I want a single inventory document that identifies the main packages, scripts, and runtime entry points so I can understand which scanner, controller, and web paths are authoritative before I change anything.

**Why this priority**: Contributors can make incorrect edits quickly if they cannot distinguish the preferred entry points from compatibility shims. This is the highest-value stabilization outcome because it reduces avoidable regressions before any code rewrite is attempted.

**Independent Test**: Read the inventory document and confirm that a new contributor can identify the canonical scanner CLI, controller, web dashboard, and any legacy wrappers without searching through source files.

**Acceptance Scenarios**:

1. **Given** a contributor who has just cloned the repository, **When** they read the inventory document, **Then** they can identify the primary scanner, controller, and web entry points and the role of each component.
2. **Given** the repository contains both canonical entry points and compatibility shims, **When** the contributor reads the inventory document, **Then** the document clearly distinguishes preferred paths from deprecated or compatibility-only paths.

---

### User Story 2 - Trace Deployment and Persistence Assets (Priority: P2)

As a maintainer, I want the inventory to explain where schema setup happens, which runtime files are created, and how install and service behavior works so I can troubleshoot deployments without rediscovering those details from scratch.

**Why this priority**: Operational troubleshooting depends on knowing what files are committed, what files are generated, and where runtime state lives. This is slightly lower priority than entry-point clarity, but it is still central to safe maintenance.

**Independent Test**: Read the inventory document and verify that a maintainer can locate schema ownership, runtime file expectations, installer behavior, and service generation details without browsing unrelated modules.

**Acceptance Scenarios**:

1. **Given** a maintainer investigating a deployment, **When** they read the inventory document, **Then** they can identify where the database schema is initialized or migrated and which runtime files are expected to appear during operation.
2. **Given** service unit files are generated during installation rather than committed to the repository, **When** the maintainer reads the inventory document, **Then** that distinction is stated explicitly together with the generated locations.

---

### User Story 3 - Separate No-Hardware and Hardware Workflows (Priority: P3)

As a developer planning follow-on stabilization work, I want the inventory to call out missing standard project files and document what can be run locally without SDR hardware versus what requires a Raspberry Pi and RTL-SDR so I can choose a safe validation path.

**Why this priority**: This work is still discovery-only, but it should reduce future cleanup effort by making current gaps and safe local workflows explicit.

**Independent Test**: Read the inventory document and confirm that it separates read-only or source-inspection workflows from hardware-required workflows, while also naming the most important missing repository-standard files if they are absent.

**Acceptance Scenarios**:

1. **Given** a developer working on a workstation without SDR hardware, **When** they follow the inventory document, **Then** they can identify supported local discovery or read-only workflows and clearly see which scan paths cannot be validated without hardware.
2. **Given** an operator deploying on Raspberry Pi with RTL-SDR hardware, **When** they follow the inventory document, **Then** they can identify the expected installation, runtime components, and service startup path for that environment.

### Edge Cases

- The inventory must distinguish repository files from files created later by the installer or at runtime.
- The inventory must explain canonical versus compatibility entry points when both are present.
- The inventory must describe local workflows that remain useful when no SDR hardware, no initialized database, or no reachable controller is available.
- The inventory must state when standard project files or committed service definitions are absent instead of implying that they exist elsewhere.
- The inventory must acknowledge legacy or transitional schema behavior if the repository currently supports more than one database shape.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The repository MUST gain a single developer-facing inventory document at `docs/PROJECT_INVENTORY.md` that captures the current codebase layout and runtime topology without changing application behavior.
- **FR-002**: The inventory document MUST identify the main packages, top-level scripts, and helper utilities, and it MUST summarize each artifact's responsibility in contributor-facing language.
- **FR-003**: The inventory document MUST identify scanner CLI, controller/API, web dashboard, and query or inspection entry points, and it MUST distinguish authoritative entry points from deprecated or compatibility wrappers.
- **FR-004**: The inventory document MUST describe how scanner, controller, web, and direct-database responsibilities are separated at runtime so contributors understand the current operational topology.
- **FR-005**: The inventory document MUST identify where database schema creation, schema evolution, or startup migrations currently occur, including cases where ownership is split across more than one module.
- **FR-006**: The inventory document MUST list expected runtime artifacts and operational files, including database files, JSONL outputs, state files, lock files, log locations, generated environment files, and generated service unit files when applicable.
- **FR-007**: The inventory document MUST describe installer and uninstall behavior relevant to deployment, including how service units are produced, where code and state are placed, and what state is preserved on uninstall.
- **FR-008**: The inventory document MUST summarize the current automated test layout and MUST explicitly note important standard project files, packaging metadata, dependency manifests, or committed deployment assets that are absent at the time of writing.
- **FR-009**: The inventory document MUST provide separate guidance for local discovery workflows without SDR hardware and for Raspberry Pi workflows with RTL-SDR hardware, making clear which paths are read-only, optional, or hardware-required.
- **FR-010**: This feature MUST remain discovery-only; it MUST not rename entry points, rewrite runtime behavior, add packaging files, or otherwise alter the application beyond the new inventory documentation.
- **FR-011**: The inventory document MUST base its statements on the observed repository state and MUST call out uncertainty, generated-only artifacts, or missing committed files instead of inferring unsupported structure.

### Key Entities *(include if feature involves data)*

- **Inventory Document**: The new contributor-facing source of truth that explains current repository structure, runtime topology, deployment assets, and known documentation gaps.
- **Runtime Component**: A major executable or package role such as the scanner CLI, controller, web application, installer, uninstaller, or query helper that contributors need to place correctly in the system.
- **Operational Artifact**: A file or directory created, read, or depended on during deployment or runtime, such as database files, JSONL event logs, lock directories, state files, generated environment files, or generated service units.
- **Repository Gap**: An expected but currently absent or generated-only project artifact that contributors should know about, such as missing packaging manifests, missing committed service files, or ad hoc migration ownership.

## Constitution Alignment *(mandatory)*

- **Pi 5 / Field Reliability Impact**: This feature is documentation-only and preserves existing Raspberry Pi deployment behavior while making current runtime and service expectations easier to inspect before any operational change is attempted.
- **Stack / Dependency Impact**: No new dependencies or stack changes are introduced. The work documents the current Python, Flask, SQLite, and local-service layout rather than expanding it.
- **Layer Ownership**: The inventory spans scanner, persistence, controller API, web UI, and install tooling, but only to document current ownership boundaries. No layer responsibilities are moved or merged by this feature.
- **Compatibility / Migration Impact**: No CLI, API, schema, or deployment behavior changes are introduced. The document will describe current compatibility shims, schema bootstrap locations, and migration-like startup logic as they exist today.
- **Security / Offline Impact**: No auth or offline behavior changes are introduced. The document will explain current token-protected controller and web interactions together with offline-friendly workflows that remain possible without hardware.
- **Verification Plan**: Validate the finished document against the current repository by tracing entry points, installer and uninstall scripts, schema ownership modules, runtime artifact paths, and the existing test layout. No hardware execution is required for this documentation feature.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: A new contributor can identify the canonical scanner, controller, and web entry points within 5 minutes using only the inventory document.
- **SC-002**: The finished inventory covers all requested categories: project structure, main packages and scripts, CLI entry points, web entry points, controller or API entry points, schema initialization or migration locations, installer behavior, service behavior, test layout, runtime files, missing standard files, local no-hardware workflows, and Raspberry Pi hardware workflows.
- **SC-003**: A maintainer can determine whether a major operational artifact is committed, installer-generated, or runtime-generated without reading more than two additional source files beyond the inventory document.
- **SC-004**: Maintainer review of the inventory document reports zero factual inaccuracies about current entry points, schema ownership, runtime artifacts, deployment files, or missing standard project files.

## Assumptions

- The primary audience is contributors and maintainers who need a developer-facing repository map rather than an end-user product guide.
- The inventory document will complement the existing README rather than replace it.
- A new `docs/` directory may need to be created because no documentation directory is currently committed.
- Local workflows without SDR hardware will focus on source inspection, controller startup, query tooling, and web or database read-only paths rather than live RF scanning.
- Missing standard project files, committed service units, or formal migration directories should be documented as current repository gaps rather than created as part of this feature.