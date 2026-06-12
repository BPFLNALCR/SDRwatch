# Implementation Plan: Improve Scan Control

**Branch**: `005-improve-scan-control` | **Date**: 2026-06-10 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/002-improve-scan-control/spec.md`

**Note**: This plan is the `/speckit-plan` output for the GUI-first scan/control page cleanup and real-hardware preset/default planning.

## Summary

Reorganize the existing server-rendered scan/control page so it is clearly the primary operator surface for SDRwatch scans. Keep the current controller job lifecycle and `/api/jobs` request shape intact while regrouping controls into Basic, Tuning, and Expert sections, replacing common RF raw numeric inputs with safer bounded controls, adding autofill defenses, adding safe-default reset, adding copyable GUI-generated job parameters, and exposing generated scanner command copying only as internal/debug information when the existing job metadata already provides it.

Additional diagnostic planning context from `docs/DETECTION_TUNING_REPORT.md` changes the default/preset target: SDRwatch is already generating raw candidates, emitted segments, and accepted hits on RTL-SDR Blog v4 hardware, but zero accepted hits are promoted into `baseline_detections`. The primary implementation direction is therefore to fix the GUI default/preset promotion mismatch first, not to rewrite CFAR/detection or treat FFT as the sole cause. Presets should encode the FFT speed/resolution tradeoff, fixed-gain baseline preference, and persistence settings needed to produce initial signal cards from the web workflow.

## Technical Context

**Language/Version**: Python 3 project; no committed project version pin on the active baseline.

**Primary Dependencies**: Flask web app, server-rendered Jinja templates, browser JavaScript embedded in `templates/control.html`, existing controller HTTP client, Python standard library, SQLite-backed web/scanner state already present in the repository.

**Storage**: Existing local SQLite database and controller state files only. This feature should not add database schema changes. It may change GUI-submitted scan parameters/defaults that affect existing persistence behavior, but should not rewrite the persistence storage layer.

**Testing**: Existing pytest-style Python tests plus focused no-hardware web/template/API tests for preset defaults and `/api/jobs` payloads. Browser-based manual verification is required for operator acceptance, including real RTL-SDR Blog v4 validation that signal cards appear through the web UI.

**Target Platform**: Raspberry Pi OS field deployment and local development on the current repository baseline.

**Project Type**: Local web dashboard plus controller service plus internal scanner backend.

**Operator Workflow Surface**: SDRwatch operator-facing features MUST use the web UI and controller job lifecycle. Treat scanner CLI work as internal backend tooling unless the feature is explicitly scanner-only.

**Performance Goals**: Control page remains responsive for representative configured zones and known signals; slider value updates, preset application, reset, and copy actions complete immediately from the operator's perspective; scan start payload generation remains a single local browser action. Presets must make scan-speed tradeoffs explicit, especially the Raspberry Pi 5 cost of larger FFTs and overlapping windows across wide sweeps.

**Constraints**: No frontend framework, no cloud service, no scanner/controller/database/DSP algorithm rewrites, no blind CFAR/detection rewrite, no simulation mode, no broad dashboard redesign, offline/local operation, preserve `/api/jobs` payload shape and parameter names. GUI defaults and presets may intentionally change submitted parameter values to resolve the promotion/persistence mismatch documented in the diagnostic report.

**Scale/Scope**: One server-rendered control page, its existing client-side scan-parameter builder, GUI tuning presets/defaults, existing controller job endpoints, optional internal/debug command copy from existing job metadata, focused tests and GUI-based verification.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Raspberry Pi First Reliability**: PASS. The planned change keeps the local control workflow lightweight and avoids new runtime services or heavy browser packages. Presets document Raspberry Pi 5 scan-speed tradeoffs for FFT size and overlapping windows.
- **II. Minimal Local Stack**: PASS. The implementation stays in Flask/Jinja and small browser JavaScript in the existing template; no frontend framework or cloud dependency is introduced.
- **III. Stable Interfaces and Clean Layering**: PASS. The web page continues to submit the same `/api/jobs` payload shape to the controller. The scanner CLI remains an internal backend invoked by the controller.
- **IV. Adapter-Based Hardware and Honest RF Claims**: PASS. No new SDR support, RF classification, detection algorithm rewrite, or emitter claims are introduced. Preset copy must describe measured tuning tradeoffs and avoid claiming FFT alone fixes the zero-card issue.
- **V. Migration-Safe, Verifiable Change**: PASS. No database migration is planned. Verification includes no-hardware web/API tests and a GUI/controller lifecycle manual path, with real RTL-SDR Blog v4 acceptance requiring signal cards to appear through the web UI.
- **Operator Acceptance Gate**: PASS. Acceptance is through the web GUI and controller job lifecycle; scanner CLI checks are optional backend smoke checks only.

## Project Structure

### Documentation (this feature)

```text
specs/002-improve-scan-control/
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- contracts/
|   |-- api-jobs-payload.md
|   `-- control-page-ui.md
`-- checklists/
    `-- requirements.md
```

### Source Code (repository root)

```text
templates/
  control.html
    Main server-rendered scan/control page, client-side settings builder,
    GUI tuning presets/defaults, safe-default reset, copy actions, live value
    display, and section layout.

sdrwatch_web/
  blueprints/api_jobs.py
    Existing /api/jobs, /api/jobs/active, /api/jobs/<id>, /api/jobs/<id>/logs
    behavior. No payload-shape change expected.
  controller.py
    Existing controller client methods for start, active/detail, and logs.

sdrwatch-control.py
  Existing controller job model and generated scanner command metadata.
  No behavioral change expected unless implementation discovers a compatibility
  gap in already-exposed job detail data.

tests/
  test_control_page_scan_settings.py
    New no-hardware tests for rendered controls, autocomplete suppression,
    preset/default metadata, and copy/settings payload behavior where testable.
  test_web_diagnostics_bundle.py
    Existing diagnostics-mode/API tests should continue passing.
  test_control_diagnostics_mode.py
    Existing controller diagnostics-mode tests should continue passing.

docs/
  DETECTION_TUNING_REPORT.md
    Diagnostic evidence that detection reaches accepted hits but fails to
    promote into baseline_detections under current defaults.
  Existing GUI/operator documentation may be updated only if needed to record
  browser-based manual verification for the cleaned-up control page and
  real-hardware preset behavior.
```

**Structure Decision**: Use the existing single repository layout. Keep the change centered on `templates/control.html`; preserve web/controller/scanner boundaries; add focused tests without adding a frontend build system, package manager, schema migration, or detector rewrite.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Phase 0 Research

Research decisions are captured in [research.md](./research.md). Key outcomes:

- Keep the scan workflow in the web GUI and keep `/api/jobs` request shape unchanged.
- Use the existing browser-side parameter-building flow as the source of truth for both scan submission and copied settings JSON.
- Use diagnostic evidence to update GUI defaults/presets so real RTL-SDR v4 scans can promote initial signal cards.
- Treat FFT as a preset speed/resolution and characterization knob, not as the primary fix for zero signal cards.
- Prefer fixed manual RTL-SDR gain for baseline/detection presets while keeping overload risk visible and configurable.
- Use existing controller job metadata for internal/debug generated command copying only after a job exists.
- Suppress browser autofill at the form and field level for scan controls.

## Phase 1 Design

Design artifacts:

- [data-model.md](./data-model.md)
- [contracts/control-page-ui.md](./contracts/control-page-ui.md)
- [contracts/api-jobs-payload.md](./contracts/api-jobs-payload.md)
- [quickstart.md](./quickstart.md)

## Post-Design Constitution Check

- **Layering remains clean**: PASS. The web page prepares operator-selected job parameters; the controller remains responsible for job lifecycle, device locks, and scanner invocation.
- **Existing interfaces remain compatible**: PASS. The design preserves `{device_key, label, baseline_id, params}` for `POST /api/jobs` and uses existing job detail data for generated command display when available.
- **Offline/local operation remains intact**: PASS. Reset, sliders, copied settings, scan start/stop, and logs are local browser/controller interactions.
- **No DSP or schema rewrites**: PASS. Presets change GUI-submitted values but do not rewrite CFAR, detection, persistence storage, or schema behavior.
- **Verification path is GUI/controller focused**: PASS. Quickstart scenarios exercise the browser control page, `/api/jobs` lifecycle, live logs, diagnostics export, and real-hardware signal-card appearance. CLI smoke checks are marked internal only.
