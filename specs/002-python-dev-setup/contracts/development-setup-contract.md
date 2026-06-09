# Contract: Local Python Development Setup

## Purpose

Define the minimum artifact and validation contract for SDRwatch's reproducible Python development setup.

## Output

- **Primary metadata**: root `pyproject.toml`
- **Documentation**: updated local development guidance in repository docs
- **Automation**: updated `.github/workflows/ci.yml`
- **Scope**: editable install, dependency grouping, no-hardware validation, installer compatibility, and supported-platform guidance

## Required Capabilities

1. **Canonical Metadata**
   - `pyproject.toml` is the single maintained source of Python project metadata.
   - Any auxiliary requirements file must be clearly documented as derived or secondary rather than a peer source of truth.

2. **Runtime Install Interface**
   - A clean environment can install SDRwatch's no-hardware runtime surface from repository metadata.
   - The install path supports importing the scanner CLI module and web app package without SDR hardware attached.

3. **Development Install Interface**
   - A contributor can install development-only tools from a documented optional dependency group.
   - The documented install path enables the repository's existing pytest suite and smoke checks.

4. **Supported Platform Matrix**
   - Linux/Pi is documented as supporting both no-hardware development and hardware validation.
   - Windows is documented as supporting no-hardware development only.
   - Unsupported hardware workflows on Windows are stated explicitly.

5. **Smoke and Test Contract**
   - The documentation exposes at least one CLI import smoke test.
   - The documentation exposes at least one web app import smoke test.
   - The documentation exposes the standard no-hardware pytest command.
   - Hardware-specific checks are documented separately from no-hardware checks.

6. **Installer Compatibility**
   - `install-sdrwatch.sh` remains a supported Raspberry Pi deployment path.
   - Installer dependency installation aligns to the canonical Python metadata instead of maintaining a parallel dependency truth source.
   - Installer responsibilities beyond Python packaging, including APT packages, device checks, and service setup, remain intact.

7. **CI Alignment**
   - The existing CI workflow installs from canonical project metadata.
   - CI runs only no-hardware validation commands.
   - CI and local quickstart documentation use the same smoke-test and pytest contract.

## Evidence Rules

- The dependency split must be traceable to committed project metadata.
- Supported-platform claims must be traceable to documentation and validation commands.
- CI behavior must be traceable to the committed workflow file.
- Hardware requirements must be labeled as OS-managed prerequisites, not as pip-replaced dependencies.

## Acceptance Conditions

- A maintainer can create a fresh virtual environment and install SDRwatch from repository metadata without using `install-sdrwatch.sh`.
- A maintainer can run the documented no-hardware smoke tests and pytest command successfully.
- A maintainer can identify the platform boundary between Linux/Pi hardware validation and Windows no-hardware development.
- The installer remains usable for Raspberry Pi deployment without requiring a second maintained dependency manifest.