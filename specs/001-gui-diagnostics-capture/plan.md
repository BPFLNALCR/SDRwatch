# Implementation Plan: GUI Diagnostic Capture

**Branch**: `004-gui-diagnostics-capture` | **Date**: 2026-06-09 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/001-gui-diagnostics-capture/spec.md`

**Note**: This plan is the `/speckit-plan` output for the GUI-first diagnostic capture workflow.

## Summary

Add an operator-facing Diagnostics mode to the web control workflow. When enabled, the web UI starts the scan through the existing controller job lifecycle with an opt-in diagnostic mode flag; the controller generates a safe per-job diagnostic JSONL path and still invokes the scanner internally with the existing `--diagnostic-jsonl` flag. Add a web/API export action that assembles a bounded local diagnostic zip for a selected or recent job from controller metadata, generated scanner command, scanner logs, diagnostic JSONL contents, recent SQLite evidence, baseline context, selected monitoring zones, friendly signals, and an operator notes template. Detection, DSP, CFAR, baseline, and scanner output semantics remain unchanged.

## Technical Context

**Language/Version**: Python 3 project; no committed project version pin on the active baseline.

**Primary Dependencies**: Flask web app, SQLite, server-rendered HTML templates, browser JavaScript in `templates/control.html`, Python standard-library file/JSON/zip helpers, existing controller HTTP client, existing scanner CLI diagnostics support.

**Storage**: Local SQLite database for baseline and monitoring context; controller state JSON plus controller-managed logs and diagnostic JSONL files under the controller base directory.

**Testing**: Existing tests use pytest-style Python tests; new verification should be no-hardware web/API and bundle-service tests using temporary files and temporary SQLite databases.

**Target Platform**: Raspberry Pi OS field deployment and local development on the current repository baseline.

**Project Type**: Local web dashboard plus controller service plus internal scanner backend.

**Operator Workflow Surface**: SDRwatch operator-facing features MUST use the web UI and controller job lifecycle. Treat scanner CLI work as internal backend tooling unless the feature is explicitly scanner-only.

**Performance Goals**: Default bundle export completes within 10 seconds for representative large logs or diagnostic JSONL files by using bounded tails or explicit export limits.

**Constraints**: Offline/local-only operation, no cloud services, bounded export size by default, no detection behavior changes, existing `/api/jobs` clients remain compatible, controller remains the process that spawns scanner jobs and owns lock files.

**Scale/Scope**: One control-page workflow, one diagnostic bundle export workflow, additive controller job metadata/path handling, additive web/API endpoint, focused no-hardware tests, and GUI-first documentation.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Raspberry Pi First Reliability**: PASS. The approach uses local files, SQLite, and bounded exports; no heavyweight runtime or cloud dependency is introduced.
- **II. Minimal Local Stack**: PASS. The feature stays within Flask, server-rendered HTML, SQLite, controller state, and Python standard library packaging.
- **III. Stable Interfaces and Clean Layering**: PASS. Existing scanner CLI diagnostic support remains an internal controller-invoked backend. The web UI adds an operator toggle, the controller generates safe paths and spawns the scanner, and the bundle exporter reads controller/database evidence without touching SDR hardware.
- **IV. Adapter-Based Hardware and Honest RF Claims**: PASS. No new SDR support or RF classification claims are added; notes template explicitly captures operator observations.
- **V. Migration-Safe, Verifiable Change**: PASS. No destructive database migration is required. Tests are no-hardware and exercise web/API and bundle creation with temporary SQLite and files. Auth behavior follows existing web/controller protections.
- **Operator Acceptance Gate**: PASS. Acceptance is through the web UI and controller job lifecycle; scanner CLI checks are limited to internal backend compatibility if needed.

## Project Structure

### Documentation (this feature)

```text
specs/001-gui-diagnostics-capture/
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   |-- diagnostic-bundle-api.md
|   `-- control-page-ui.md
`-- tasks.md
```

### Source Code (repository root)

```text
sdrwatch-control.py
  Controller job model, generated diagnostic path handling, job params/command metadata.

sdrwatch_web/
  blueprints/api_jobs.py
    Existing /api/jobs compatibility plus diagnostic bundle export endpoint.
  controller.py
    Controller client methods for job metadata and logs already available.
  diagnostics.py
    New local bundle assembly service for controller, log, diagnostic file, and SQLite evidence.
  config.py
    Local export bounds and optional diagnostic export settings.

templates/
  control.html
    Diagnostics mode toggle and export diagnostic bundle action.

tests/
  test_web_diagnostics_bundle.py
    No-hardware bundle creation and API tests with temporary files and SQLite.
  test_control_diagnostics_mode.py
    Controller/job command tests for generated diagnostic_jsonl without SDR hardware.

README.md
docs/
  GUI-first diagnostic capture documentation updates.
```

**Structure Decision**: Use the existing single repository layout. Keep job lifecycle changes in `sdrwatch-control.py`, web/API behavior in `sdrwatch_web/`, templates in `templates/`, and verification in `tests/`. Do not add a new frontend framework, service, package manager, or database migration framework.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0 Research

Research decisions are captured in [research.md](./research.md). Key outcomes:

- Use controller-generated per-job diagnostic filenames instead of operator-entered paths.
- Keep the scanner CLI flag unchanged and internal.
- Export a local zip containing bounded evidence plus a manifest that records truncation and missing artifacts.
- Use the current SQLite schema for baseline, detection, scan update, monitoring-zone, and friendly-signal evidence.

## Phase 1 Design

Design artifacts:

- [data-model.md](./data-model.md)
- [contracts/diagnostic-bundle-api.md](./contracts/diagnostic-bundle-api.md)
- [contracts/control-page-ui.md](./contracts/control-page-ui.md)
- [quickstart.md](./quickstart.md)

## Post-Design Constitution Check

- **Layering remains clean**: PASS. The controller continues to invoke scanner jobs internally; the web layer does not touch SDR hardware; bundle creation only reads already-produced evidence.
- **Existing interfaces remain compatible**: PASS. `/api/jobs` keeps current request/response behavior. New diagnostic mode fields are optional, and existing `diagnostic_jsonl` pass-through remains supported for non-GUI internal/backend callers.
- **Offline/local operation remains intact**: PASS. Bundle creation uses controller state/logs, local diagnostic JSONL, and local SQLite only.
- **Bounded resource use**: PASS. Default export limits are part of the contract and visible in the manifest.
- **Verification path is GUI/controller focused**: PASS. Quickstart and tests are web/API and no-hardware oriented, with no operator CLI requirement.
