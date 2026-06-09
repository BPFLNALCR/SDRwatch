# Data Model: Reproducible Python Development Setup

## Overview

This feature does not change application persistence. Its data model is a conceptual model for the developer-facing setup artifacts, dependency boundaries, and validation paths that implementation must introduce or update.

## Entities

### 1. Project Metadata Manifest

- **Purpose**: The canonical repository-owned Python metadata document expected at `pyproject.toml`.
- **Fields**:
  - `path`: manifest path.
  - `build_backend`: Python build backend definition.
  - `project_name`: installable project name.
  - `python_version_range`: supported Python version floor.
  - `runtime_dependencies`: base Python dependencies required for no-hardware runtime and import workflows.
  - `optional_dependency_groups`: named extras such as development or hardware-oriented groups.
  - `package_discovery_scope`: packages/modules exposed by editable install.
- **Validation rules**:
  - Must be the single maintained source of Python dependency truth.
  - Must not claim to replace OS-managed SDR drivers, system libraries, or udev configuration.
  - Must support editable installation from a clean checkout.

### 2. Dependency Group

- **Purpose**: A named install surface derived from the project metadata.
- **Fields**:
  - `name`: group identifier such as runtime, dev, or rtl.
  - `purpose`: why the group exists.
  - `included_python_packages`: Python packages intentionally installed by pip.
  - `excluded_os_prerequisites`: non-pip prerequisites that remain documented outside the group.
  - `consumers`: local developers, CI, installer, or Linux/Pi hardware users.
- **Validation rules**:
  - Each group must have a clear consumer and purpose.
  - Development-only tools must not be required for runtime smoke tests.
  - Hardware-specific groups must not be required for Windows no-hardware workflows.

### 3. Supported Platform Matrix

- **Purpose**: The documented compatibility boundary for development and validation.
- **Fields**:
  - `platform_name`: Linux/Pi or Windows.
  - `supported_modes`: no-hardware development, full development, hardware validation.
  - `python_install_method`: expected interpreter source.
  - `os_prerequisites`: platform-specific packages or tools.
  - `unsupported_capabilities`: explicitly excluded workflows.
- **Validation rules**:
  - Must distinguish Windows no-hardware development from Linux/Pi hardware validation.
  - Must state prerequisites separately from Python package installation.

### 4. Validation Scenario

- **Purpose**: A reproducible command sequence that proves the setup works.
- **Fields**:
  - `name`: scenario title.
  - `platform_scope`: where the scenario is expected to run.
  - `hardware_required`: whether SDR hardware is required.
  - `install_command`: how the environment is created and populated.
  - `verification_commands`: smoke-test, pytest, or help commands.
  - `expected_outcome`: observable success condition.
  - `ci_eligible`: whether the scenario should run in GitHub Actions.
- **Validation rules**:
  - At least one scenario must be runnable on both Linux/Pi and Windows without hardware.
  - Hardware-required scenarios must be documented separately and excluded from CI.

### 5. Installer Compatibility Guard

- **Purpose**: The set of behaviors from `install-sdrwatch.sh` that must remain intact.
- **Fields**:
  - `venv_mode`: whether the installer uses `--system-site-packages`.
  - `os_package_step`: APT provisioning behavior.
  - `pip_install_source`: where the installer gets Python dependency information.
  - `sanity_checks`: import checks, `rtl_test`, or related verification.
  - `service_flow`: generated env file and systemd behavior.
- **Validation rules**:
  - Existing service and deployment flow must remain recognizable.
  - Installer dependency installation must align with the canonical metadata rather than a peer manifest.

### 6. CI Validation Job

- **Purpose**: A no-hardware automation surface that verifies the canonical development setup.
- **Fields**:
  - `workflow_path`: CI workflow file path.
  - `runner_platform`: operating system used for the job.
  - `install_command`: install command sourced from project metadata.
  - `test_commands`: pytest and smoke-test commands.
  - `hardware_boundary`: explicit statement that SDR/device access is excluded.
- **Validation rules**:
  - Must install from the canonical project metadata.
  - Must only run no-hardware validation commands.
  - Must stay aligned with the documented local quickstart.

## Relationships

- One **Project Metadata Manifest** defines many **Dependency Groups**.
- The **Supported Platform Matrix** constrains which **Dependency Groups** and **Validation Scenarios** apply on each platform.
- The **Installer Compatibility Guard** consumes the **Project Metadata Manifest** while preserving the Pi deployment workflow.
- One or more **CI Validation Jobs** execute a subset of **Validation Scenarios** that are marked `ci_eligible`.

## State Transitions

### Dependency Group Lifecycle

- `proposed` -> `documented`: when the group is recorded in `pyproject.toml` and referenced in docs.
- `documented` -> `verified`: when the group's install and validation commands pass in local or CI checks.

### Validation Scenario Lifecycle

- `draft` -> `documented`: when the scenario is added to quickstart and README guidance.
- `documented` -> `verified`: when commands succeed in the intended environment.
- `verified` -> `stale`: when dependency metadata, platform support, or smoke-test commands change without updating the scenario.