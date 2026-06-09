# Implementation Plan: Reproducible Python Development Setup

**Branch**: `[002-add-python-dev-setup]` | **Date**: 2026-06-09 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/002-python-dev-setup/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Add a minimal root `pyproject.toml` as SDRwatch's canonical Python metadata and dependency source, keep the existing module and root-script launch surfaces for scanner/controller/web workflows, document supported Linux/Pi and Windows no-hardware development paths, preserve `install-sdrwatch.sh` by aligning it to the same metadata rather than a generated dependency list, and update the existing GitHub Actions workflow to run the documented no-hardware smoke tests and pytest suite.

## Technical Context

<!--
  ACTION REQUIRED: Replace the content in this section with the technical details
  for the project. The structure here is presented in advisory capacity to guide
  the iteration process.
-->

**Language/Version**: Python 3.11+ on Raspberry Pi OS Trixie/Bookworm and Windows for no-hardware development.

**Primary Dependencies**: Setuptools build metadata in `pyproject.toml`; Python runtime dependencies centered on NumPy and Flask; development extras for `pytest`, `ruff`, and `mypy`; optional native RTL helper dependency for Linux/Pi hardware workflows; OS-managed SciPy, librtlsdr/rtl-sdr, SoapySDR, libusb, and udev configuration remain external prerequisites for hardware validation.

**Storage**: No application data-model changes. New committed metadata and documentation files only; existing SQLite, controller state, and JSONL behavior remain unchanged.

**Testing**: Editable-install smoke tests for CLI and web imports, the existing `pytest` unit tests under `tests/`, GitHub Actions no-hardware validation on supported developer platforms, and optional Linux/Pi hardware verification.

**Target Platform**: Raspberry Pi OS/Linux for full development and hardware validation, plus Windows for no-hardware development only.

**Project Type**: Python monorepo with a package-based scanner CLI, root controller and web entrypoints, a server-rendered Flask web package, and GitHub Actions CI.

**Performance Goals**: Preserve current runtime behavior and scan-path performance, enable a clean-checkout no-hardware developer setup in 15 minutes or less, and avoid adding steady-state runtime overhead.

**Constraints**: Preserve `install-sdrwatch.sh` as a supported deployment path, keep `pyproject.toml` as the single source of truth for Python dependency metadata, avoid replacing OS-managed SDR prerequisites with pip packages, require no SDR hardware in CI, and leave schema/API/runtime topology unchanged.

**Scale/Scope**: Root packaging metadata, dependency grouping, README or local-development documentation, supported-platform guidance, CI workflow alignment, optional installer dependency wiring, and no-hardware validation commands.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- PASS: Raspberry Pi 5 compatibility, field reliability, and offline operation are preserved because the feature changes packaging metadata, contributor docs, installer dependency wiring, and CI only; the existing hardware deployment path remains intact.
- PASS: The only new stack element is standard Python project metadata via `pyproject.toml`, which is the smallest maintainable way to support editable installs and dependency grouping without introducing a new framework.
- PASS: Layer ownership stays clean because scanner, persistence, controller, and web code paths are not being redistributed; the work focuses on packaging, docs, and automation surfaces around those layers.
- PASS: CLI behavior remains backward-compatible by keeping `python -m sdrwatch.cli`, `sdrwatch-control.py`, and `sdrwatch-web.py` as valid launch surfaces while documenting them more clearly.
- PASS: No database or auth changes are planned. Hardware-specific changes are limited to dependency declaration and validation boundaries, with RTL-SDR and Soapy prerequisites remaining explicit and Linux/Pi-only.
- PASS: Verification includes automated no-hardware tests and smoke checks plus a reproducible optional Linux/Pi hardware path, satisfying the verifiable-change requirement.

**Post-Design Re-check**: PASS. The design artifacts keep `pyproject.toml` as the only maintained dependency source, separate OS-managed hardware prerequisites from Python-managed development dependencies, preserve installer semantics, and define a reproducible no-hardware validation contract for both docs and CI.

## Project Structure

### Documentation (this feature)

```text
specs/002-python-dev-setup/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── development-setup-contract.md
└── tasks.md
```

### Source Code (repository root)

```text
sdrwatch/
├── baseline/
├── detection/
├── drivers/
├── dsp/
├── io/
├── sweep/
└── util/

sdrwatch_web/
├── blueprints/
└── [web support modules]

templates/
├── partials/
└── [server-rendered HTML templates]

static/
└── js/

tests/
└── [repo test modules]

sdrwatch-control.py
sdrwatch-web.py
sdrwatch.py
install-sdrwatch.sh
.github/workflows/ci.yml
README.md
pyproject.toml
```

**Structure Decision**: Extend the existing monorepo layout without moving runtime code. Add a root `pyproject.toml` for canonical metadata, keep the current module and root-script launch paths, update documentation in place, align `.github/workflows/ci.yml` to the canonical install path, and preserve the existing installer/service topology.

## Complexity Tracking

No constitution violations identified.
