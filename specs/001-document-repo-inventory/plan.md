# Implementation Plan: Repository Inventory and Stabilization

**Branch**: `[001-document-repo-inventory]` | **Date**: 2026-06-08 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `/specs/001-document-repo-inventory/spec.md`

## Summary

Create a single developer-facing repository inventory at `docs/PROJECT_INVENTORY.md` that maps SDRwatch's authoritative entry points, package responsibilities, schema ownership, runtime and generated artifacts, installer and uninstall behavior, tests and CI coverage, and workstation-versus-Raspberry Pi workflows. The implementation remains documentation-only and will ground every statement in the current repository state so contributors can orient themselves without reverse-engineering the codebase.

## Technical Context

**Language/Version**: Markdown documentation describing a Python 3.11-oriented codebase plus Bash deployment scripts.

**Primary Dependencies**: Existing repository sources, Python standard library, Flask, NumPy/SciPy, SQLite, RTL-SDR and SoapySDR integrations, and GitHub Actions metadata; no new dependencies are planned.

**Storage**: No new storage. The feature documents existing SQLite storage (`sdrwatch.db`), optional JSONL outputs, controller state/log/lock files, and installer-generated environment and service files.

**Testing**: Source-backed manual verification against repository files plus no-hardware validation paths such as CLI help output, profile listing, existing DSP pytest coverage, and optional web startup against an existing database.

**Target Platform**: Contributors on Windows and Linux workstations, with deployed runtime behavior documented for Raspberry Pi 5 on Raspberry Pi OS Trixie/Bookworm.

**Project Type**: Documentation feature inside a Python monorepo with a CLI scanner, controller daemon, and server-rendered Flask web UI.

**Performance Goals**: Zero runtime behavior or performance impact. The main quality target is documentation accuracy and contributor comprehension rather than code-path speed.

**Constraints**: Discovery-only scope, no runtime rewrites, no schema or API changes, explicit separation of committed files versus installer-generated assets versus runtime-generated artifacts, and consistent use of current SDRwatch terminology.

**Scale/Scope**: Repository-wide coverage across root scripts, `sdrwatch/`, `sdrwatch_web/`, templates, static assets, tests, CI, schema helpers, installer and uninstall scripts, and missing contributor-facing project files.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- PASS: Raspberry Pi 5 compatibility, field reliability, and offline operation are preserved because the change is documentation-only and does not alter scanner, controller, web, or deployment behavior.
- PASS: No dependency expansion is planned; the design stays within the existing Python, Flask, SQLite, and server-rendered HTML stack.
- PASS: Layer ownership remains clean because the feature documents current scanner, persistence, controller, and web boundaries instead of moving responsibilities between them.
- PASS: CLI behavior remains unchanged. The document will explicitly distinguish canonical entry points from compatibility shims rather than introducing any transition.
- PASS: Database, auth, and hardware impacts are descriptive only. The plan records current split schema ownership, token usage, and hardware-dependent workflows without modifying them.
- PASS: Verification is reproducible without hardware for the documentation itself and includes optional Raspberry Pi checks for confirming deployment claims where needed.

**Post-Design Re-check**: PASS. The resulting design artifacts keep the work documentation-only, avoid new operational surface area, and provide a reproducible manual validation path that matches the constitution's minimal-stack and verifiable-change requirements.

## Project Structure

### Documentation (this feature)

```text
specs/001-document-repo-inventory/
├── plan.md
├── research.md
├── data-model.md
├── quickstart.md
├── contracts/
│   └── project-inventory-contract.md
└── tasks.md

docs/
└── PROJECT_INVENTORY.md
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
└── [DSP-focused test modules]

.github/
└── workflows/ci.yml

sdrwatch-control.py
sdrwatch-web.py
sdrwatch.py
query-sdrwatch.py
install-sdrwatch.sh
uninstall-sdrwatch.sh
bandplan_eu.csv
bandplan_us_na.csv
```

**Structure Decision**: Add a `docs/` directory containing one new `PROJECT_INVENTORY.md` artifact. Keep all runtime code in place, cite existing source files as evidence, and avoid any restructuring beyond the documentation addition itself.

## Complexity Tracking

No constitution violations identified.
