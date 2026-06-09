# Implementation Plan: No-Hardware Simulation Mode

**Branch**: `[003-add-sim-mode]` | **Date**: 2026-06-09 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/003-add-sim-mode/spec.md`

**Note**: This template is filled in by the `/speckit.plan` command. See `.specify/templates/plan-template.md` for the execution workflow.

## Summary

Add an additive simulation scan mode that plugs a deterministic synthetic SDR source into the existing scanner runner, keeps the baseline-first sweep, detection, persistence, and dashboard database contracts unchanged, documents a simulated FM-band workflow, and adds no-hardware automated coverage for deterministic detections and required SQLite writes.

## Technical Context

**Language/Version**: Python 3.11+ on Raspberry Pi OS Trixie/Bookworm and Linux/Windows no-hardware development environments.

**Primary Dependencies**: Python standard library, NumPy for deterministic sample generation and existing DSP paths, Flask for controller and web surfaces, SQLite via the existing `Store`, optional hardware backends such as `pyrtlsdr` or SoapySDR for real scans, and existing SDRwatch DSP/baseline modules.

**Storage**: Existing SQLite baseline tables (`baselines`, `baseline_noise`, `baseline_occupancy`, `baseline_detections`, `scan_updates`, `baseline_snapshot`, related summaries), local controller logs, and optional JSONL output. No new persistence store is planned.

**Testing**: `pytest` unit and integration coverage using temp SQLite databases and no-hardware execution, plus a reproducible manual simulated-scan validation path. Real hardware validation remains separate for RTL-SDR regression confidence.

**Target Platform**: Raspberry Pi 5 on Raspberry Pi OS for production behavior, Linux and Windows for no-hardware development and CI, and controller/web deployments that already consume SQLite scan outputs.

**Project Type**: Python monorepo with CLI scanner, controller daemon/API, server-rendered Flask web UI, and pytest-based test suite.

**Performance Goals**: Keep simulation bounded and deterministic, preserve current sweep cadence and resource expectations for real scans, and avoid adding steady-state overhead to non-simulated runs.

**Constraints**: Preserve baseline-first semantics, keep schema/API changes additive or avoid them entirely, do not interfere with existing RTL-SDR or future Soapy paths, keep offline workflows intact, and ensure hardware-free validation runs without SDR attachments or hardware-only dependencies.

**Scale/Scope**: Scanner driver selection, synthetic source generation, CLI/help documentation, automated tests for deterministic outputs and DB writes, and documentation for dashboard-visible simulated data. Controller/web support for selecting a simulated device is optional and secondary to CLI-first validation.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- PASS: Raspberry Pi 5 compatibility and offline operation are preserved because the feature is additive, leaves real scan defaults unchanged, and introduces no cloud or service dependency.
- PASS: No new dependency beyond the existing Python and NumPy stack is required; deterministic sample generation can be implemented inside the current runtime footprint.
- PASS: Layer ownership stays clean. Synthetic sample generation belongs in the scanner/driver boundary, while persistence and the dashboard continue consuming the same DB contracts.
- PASS: CLI behavior stays backward-compatible by default. Simulation must be explicitly selected, and existing hardware error behavior remains intact when a physical driver is chosen on a machine without hardware.
- PASS: Database and auth contracts remain unchanged. The plan intentionally reuses existing baseline tables and web/controller reads; no token behavior change is in scope.
- PASS: Verification includes deterministic automated tests, DB-write validation, a documented manual simulated FM-band run, and a compatibility check that real-hardware paths still require explicit hardware drivers.

**Post-Design Re-check**: PASS. The design keeps simulation entirely within the scanner input boundary, reuses current SQLite and dashboard contracts, treats controller/UI simulation discovery as optional rather than a required contract change, and provides reproducible no-hardware validation without weakening real-device workflows.

## Project Structure

### Documentation (this feature)

```text
specs/003-add-sim-mode/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── simulation-mode-contract.md
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
README.md
pyproject.toml
```

**Structure Decision**: Extend the existing monorepo layout without moving responsibilities. Add the simulation source under `sdrwatch/drivers/`, update scanner source selection in `sdrwatch/sweep/runner.py`, keep persistence and web reads unchanged, document the simulated workflow in existing docs, and add no-hardware tests in `tests/`.

## Complexity Tracking

No constitution violations identified.
