# Research: Reproducible Python Development Setup

## Decision 1: Use a minimal setuptools-based `pyproject.toml` as the only maintained Python dependency source

- **Decision**: Add a root `pyproject.toml` with setuptools build metadata and package discovery, and use it as the single maintained source of Python project metadata for local installs, CI installs, and any installer-side pip install step.
- **Rationale**: The repository currently has no committed packaging metadata or dependency manifests. A minimal `pyproject.toml` is the smallest standards-based change that enables `pip install -e .` and dependency grouping without introducing a second source of truth.
- **Alternatives considered**:
  - Keep using generated or hand-maintained `requirements*.txt` files as the primary dependency source. Rejected because it would preserve the current drift between installer, CI, and contributor workflows.
  - Add `pyproject.toml` plus peer `requirements*.txt` files as equally authoritative sources. Rejected because the specification explicitly requires one canonical dependency declaration.

## Decision 2: Keep the existing module and root-script launch surfaces instead of introducing new console entry points in this feature

- **Decision**: Preserve `python -m sdrwatch.cli`, `python sdrwatch-control.py`, and `python sdrwatch-web.py` as the documented launch and smoke-test commands for this feature.
- **Rationale**: Those entrypoints already exist, are referenced throughout the repository, and are enough to satisfy editable-install and verification goals. Avoiding new console scripts keeps the change smaller and lowers the risk of breaking existing operator habits.
- **Alternatives considered**:
  - Add new console-script entry points immediately. Rejected because the feature goal is reproducible setup rather than CLI surface redesign, and the current entrypoints already cover the required workflows.
  - Keep local development tied to raw source checkout without packaging metadata. Rejected because it does not provide a standard install path for clean virtual environments.

## Decision 3: Split dependencies by capability, with OS-managed hardware prerequisites remaining outside canonical Python runtime metadata

- **Decision**: Model the Python-managed dependency surface around cross-platform needs first: NumPy and Flask in the base runtime set, `pytest`/`ruff`/`mypy` in a development extra, and native RTL helper packages as optional hardware-oriented install surface. Keep SciPy, librtlsdr, rtl-sdr, SoapySDR, libusb, udev rules, and related device configuration documented as OS-managed prerequisites for Linux/Pi hardware workflows.
- **Rationale**: NumPy is imported broadly across the DSP and test surfaces and is required for no-hardware development on both Linux and Windows. Flask is required for controller serve mode and the web UI. By contrast, SciPy is already optional in code, and SDR device stacks depend on OS libraries and permissions that pip cannot replace safely.
- **Alternatives considered**:
  - Put all observed dependencies, including SciPy and hardware stacks, into the base Python dependency list. Rejected because it would blur the platform boundary and risk unnecessary or fragile installs on Raspberry Pi and Windows.
  - Keep NumPy out of canonical metadata and require manual installation. Rejected because it would undermine the goal of a reproducible no-hardware development environment.

## Decision 4: Preserve the installer flow by making it consume canonical project metadata inside its existing system-site-packages model

- **Decision**: Keep `install-sdrwatch.sh` as the Pi deployment path, but update its pip-install step to consume the project's canonical metadata instead of generating a repository-local requirements file as a parallel source of truth.
- **Rationale**: The current installer already encodes the correct Raspberry Pi assumptions: APT-provided numeric libraries and SDR tooling, `--system-site-packages`, runtime sanity checks, and optional systemd deployment. Reusing those behaviors while removing the ad hoc generated dependency list keeps deployment stable and aligns maintenance around one manifest.
- **Alternatives considered**:
  - Leave the installer's generated `requirements.sdrwatch.txt` behavior unchanged. Rejected because it would leave a second dependency truth source in place.
  - Replace the installer with a pure pip-driven workflow. Rejected because the installer also manages APT packages, udev/kernel configuration, and service deployment that remain necessary on Raspberry Pi.

## Decision 5: Align documentation and CI around one no-hardware validation contract, while keeping hardware validation Linux/Pi-only

- **Decision**: Update README and local development guidance to use the canonical editable-install commands, and update the existing GitHub Actions workflow to install from `pyproject.toml`, run the repository's no-hardware pytest suite, and run CLI/web smoke checks without touching SDR hardware. Use Linux/Pi-only manual validation for hardware checks.
- **Rationale**: The specification requires fresh-venv install instructions, test commands, CLI and web smoke tests, and a clear hardware boundary. Reusing the same commands in docs and CI is the best way to prevent drift.
- **Alternatives considered**:
  - Keep CI on its current ad hoc dependency install path. Rejected because it would allow documentation and automation to diverge immediately after the feature lands.
  - Require hardware or system SDR packages in CI. Rejected because the supported CI contract for this feature is explicitly no-hardware.