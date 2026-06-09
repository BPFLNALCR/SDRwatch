# Feature Specification: Reproducible Python Development Setup

**Feature Branch**: `[002-add-python-dev-setup]`

**Created**: 2026-06-09

**Status**: Draft

**Input**: User description: "Add a reproducible Python dependency and development setup for SDRwatch."

## Clarifications

### Session 2026-06-09

- Q: Which dependency declaration approach should be canonical for local development? → A: Use a minimal `pyproject.toml` as the canonical project metadata and dependency source, with optional development dependencies for contributor tools.
- Q: Which developer platforms should be officially supported? → A: Support Raspberry Pi OS/Linux and Windows for no-hardware development; keep hardware validation Linux/Pi only.
- Q: Should the existing CI workflow be updated as part of this feature? → A: Yes. Update the existing CI workflow so it installs from the canonical dependency setup and runs the documented no-hardware smoke checks.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Set Up a Local Environment (Priority: P1)

As a contributor, I can create an isolated local SDRwatch environment from a clean checkout on a supported development platform, install the project without invoking the Raspberry Pi installer, and verify that the core CLI and web import surfaces work without SDR hardware attached.

**Why this priority**: This is the minimum slice that enables repeatable onboarding, day-to-day development, and no-hardware validation for contributors and CI.

**Independent Test**: On a clean checkout on supported Linux/Pi and Windows no-hardware development environments, follow the documented setup steps to create the environment, install SDRwatch, run the CLI import smoke test, run the web app import smoke test, and run the documented no-hardware test command.

**Acceptance Scenarios**:

1. **Given** a clean checkout on a supported local development platform and the documented prerequisites for that platform, **When** a developer follows the local setup instructions, **Then** SDRwatch installs into an isolated environment without requiring `install-sdrwatch.sh`.
2. **Given** a completed local install and no SDR hardware attached, **When** the developer runs the documented smoke checks, **Then** the scanner CLI module and web app factory import successfully without attempting to access a device.
3. **Given** a developer machine missing required system SDR prerequisites or running on Windows without supported SDR hardware tooling, **When** the developer follows the setup documentation, **Then** the unsupported hardware path is clearly separated from the allowed no-hardware workflow before any hardware-specific validation is attempted.

---

### User Story 2 - Use the Right Dependency Scope (Priority: P2)

As a maintainer, I can install only the runtime dependencies needed to launch SDRwatch or optionally add development tools for tests and contributor work, so Raspberry Pi deployments stay lean and contributor workflows remain complete.

**Why this priority**: SDRwatch must stay Pi-friendly and avoid forcing testing or linting tools into runtime environments that only need the scanner, controller, or web UI.

**Independent Test**: Install the runtime dependency set in a fresh environment and confirm the documented launch/import commands work; then install the development dependency set and confirm the documented no-hardware test command becomes available.

**Acceptance Scenarios**:

1. **Given** a runtime-only local install, **When** a maintainer runs the documented launch or smoke-test commands, **Then** the scanner CLI, controller, and web entrypoints are available without requiring development-only tools.
2. **Given** a contributor install that includes the development dependency set, **When** the contributor runs the documented no-hardware validation commands, **Then** the tests and smoke checks complete without replacing OS-managed SDR dependencies with Python packages.

---

### User Story 3 - Preserve Installer and CI Alignment (Priority: P3)

As a maintainer, I can keep the existing installer-based deployment flow intact while aligning local development documentation and the existing CI workflow with the same dependency split and smoke-check expectations.

**Why this priority**: Local setup improvements should reduce drift, not create a second unsupported installation path or force a deployment change on field systems.

**Independent Test**: Verify that the installer remains a supported path, that the new setup artifacts do not change installer responsibilities, and that the existing CI workflow reuses the documented dependency separation and no-hardware checks.

**Acceptance Scenarios**:

1. **Given** an existing installer-based deployment workflow, **When** the new local setup artifacts are added, **Then** `install-sdrwatch.sh` remains a supported and unchanged primary deployment path for Raspberry Pi service installs.
2. **Given** the repository's existing CI workflow, **When** this feature is implemented, **Then** CI uses the same declared dependency split and no-hardware checks as the documented local development path.

---

### Edge Cases

- A developer uses Raspberry Pi OS with APT-provided NumPy, SciPy, and SDR libraries; the local setup must not force incompatible source builds or wheel-only assumptions.
- A developer uses Windows for contributor work; the documentation must allow no-hardware install and smoke tests without implying that hardware scanning is supported there.
- A developer runs no-hardware tests on a machine with no SDR attached; smoke checks and documented non-hardware tests must still succeed.
- A developer lacks system SDR libraries, udev rules, or device permissions; the setup must distinguish blocked hardware validation from allowed install and no-hardware verification steps.
- Existing installer users continue to rely on `install-sdrwatch.sh`; new dependency declarations and docs must not introduce conflicting paths, environment variables, or service assumptions.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST provide a minimal, repository-owned `pyproject.toml` that serves as the canonical Python project metadata and dependency source for creating a local isolated SDRwatch environment without invoking the installer script.
- **FR-002**: System MUST separate the Python dependencies required to launch the scanner CLI, controller, and web app from the optional development-only tools used for testing, linting, or type checking, using optional dependency groupings rooted in the canonical project metadata.
- **FR-003**: System MUST document which prerequisites remain OS-managed rather than Python-managed, including SDR drivers and libraries, device-access requirements, Raspberry Pi-oriented numeric stack expectations, and the platform-specific limits of Windows no-hardware development.
- **FR-004**: System MUST provide a documented fresh-environment setup path that covers environment creation, project installation, and verification from a clean checkout for supported Linux/Pi development and Windows no-hardware development.
- **FR-005**: System MUST document at least one no-hardware validation path that includes a CLI import smoke test, a web app import smoke test, and the standard no-hardware test command.
- **FR-006**: System MUST document hardware-specific validation separately from no-hardware validation, including any prerequisite OS packages, permissions, or device checks that remain outside the Python environment, and it MUST define hardware validation as Linux/Pi-only.
- **FR-007**: System MUST preserve `install-sdrwatch.sh` as a supported deployment path and MUST NOT require existing installer users to adopt the new local-development workflow.
- **FR-008**: System MUST preserve Raspberry Pi OS compatibility by avoiding Python-managed replacements for system-level SDR components that are currently expected to come from the operating system.
- **FR-009**: System MUST treat `pyproject.toml` as the single source of truth for local Python dependency metadata; any auxiliary requirements files added for compatibility or workflow convenience MUST be documented as derived or secondary artifacts rather than peer sources of truth.
- **FR-010**: System MUST update the existing CI workflow to reuse or clearly map to the same canonical dependency split and no-hardware smoke checks documented for local development.
- **FR-011**: System MUST document a supported-platform matrix that distinguishes full Linux/Pi development and hardware validation from Windows no-hardware development only.

### Key Entities *(include if feature involves data)*

- **Runtime Dependency Set**: The minimal Python dependency group required to import and launch SDRwatch CLI, controller, and web entrypoints once OS-level SDR prerequisites are installed.
- **Development Dependency Set**: Additional tools and libraries used for tests, linting, type checking, and contributor workflows that are not required for normal runtime use.
- **Local Setup Guide**: Contributor-facing instructions that define prerequisites, environment creation, install commands, smoke tests, and no-hardware test commands.
- **No-Hardware Verification Path**: A documented sequence of checks that validates installation and core imports without attached SDR hardware.
- **Hardware Verification Path**: A documented sequence of checks that confirms SDR libraries, permissions, and physical devices are ready for RF scanning.

## Constitution Alignment *(mandatory)*

- **Pi 5 / Field Reliability Impact**: The feature keeps the Pi-first deployment model intact by leaving hardware drivers, heavy numeric libraries, and service provisioning in the existing OS-level and installer path while making contributor setup repeatable.
- **Stack / Dependency Impact**: The change introduces packaging or dependency declaration artifacts and documentation only. No new service layer, frontend framework, or cloud dependency is added.
- **Layer Ownership**: Affects contributor-facing entrypoints for the scanner CLI, controller, and web startup plus documentation and optional CI wiring. It does not move DSP, database, controller, or web responsibilities across layers.
- **Compatibility / Migration Impact**: No SQLite schema or REST contract changes are expected. Existing installer-based deployments remain supported, and the CI workflow must align with the declared manifests rather than create a separate dependency truth source.
- **Security / Offline Impact**: `SDRWATCH_CONTROL_TOKEN` behavior and offline field operation remain unchanged. Documentation may cover local token configuration for development, but auth flows are not altered, and Windows support is limited to no-hardware workflows.
- **Verification Plan**: Verify the feature through a documented clean-environment setup on supported Linux/Pi and Windows no-hardware environments, runtime-only install, contributor install, CLI import smoke test, web app import smoke test, existing no-hardware test command, CI execution of the same no-hardware checks, and a separately documented optional Linux/Pi hardware verification path.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: From a clean checkout with documented OS prerequisites installed, a developer can complete local setup and pass the documented CLI and web smoke checks in 15 minutes or less without using the installer script.
- **SC-002**: 100% of documented no-hardware validation steps run successfully on supported Linux/Pi and Windows development machines with no attached SDR device.
- **SC-003**: A runtime-only installation path excludes development and test tools while still allowing the documented CLI, controller, and web launch or smoke-test commands to work.
- **SC-004**: Maintainers can continue using the existing installer-based deployment flow with no additional mandatory steps introduced by the new local-development setup.
- **SC-005**: The existing CI workflow completes successfully using the canonical dependency declaration and the documented no-hardware smoke checks.

## Assumptions

- Local contributor installs standardize on a minimal `pyproject.toml`; any additional requirements files are optional compatibility outputs and not an equal source of dependency truth.
- Full contributor validation, including hardware checks, happens on Raspberry Pi OS or another supported Linux environment; Windows is an officially supported no-hardware development environment only.
- Heavy numeric libraries and SDR drivers that are currently provisioned by the OS or installer remain outside the Python dependency manifest unless proven safe and compatible on Raspberry Pi OS.
- The repository's existing CI workflow is in scope for alignment with the new dependency declarations, but adding a brand-new CI system is out of scope.
- This feature does not change database schema, REST contracts, or scan behavior; it standardizes dependency management, local setup, and verification guidance only.