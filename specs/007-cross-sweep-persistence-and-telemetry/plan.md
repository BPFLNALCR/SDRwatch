# Implementation Plan: Cross-Sweep Persistence and Telemetry

**Branch**: `007-cross-sweep-persistence-and-telemetry` | **Date**: 2026-06-14 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/007-cross-sweep-persistence-and-telemetry/spec.md`

## Summary

Improve SDRwatch's passive RF monitoring trust surface by allowing a stable signal seen once per complete sweep loop to satisfy strict persistence thresholds across repeated loops, while making job telemetry and effective scan settings auditable in diagnostic bundles. The implementation should build on the `006-fm-signal-characterization` baseline, preserve the existing raw/measured/match/display/context separation, keep FM Broadcast as the regression-control band, and avoid demodulation, content decoding, offensive SIGINT workflows, broad classification, or a destructive schema redesign.

The preferred approach is a narrow, test-first change: add sweep-loop-aware candidate state to the scanner persistence path, expand controller/scanner parameter parity for existing characterization knobs, emit a structured effective-parameter manifest and aggregate diagnostics, and record device/gain telemetry when the driver exposes it. Operator validation remains web UI -> controller job -> diagnostic bundle export. Scanner CLI checks are backend smoke and parity checks only.

## Technical Context

**Language/Version**: Python 3 project; no committed project version pin on this branch.

**Primary Dependencies**: Flask web app, server-rendered Jinja templates, existing controller HTTP API, scanner modules under `sdrwatch/`, NumPy-based DSP, RTL-SDR native driver integration, SQLite baseline store, diagnostic JSONL and bundle export.

**Storage**: Existing SQLite baseline tables plus diagnostic JSONL and bounded diagnostic bundle exports. Plan assumes additive in-memory or sidecar cross-sweep candidate state first; persistent schema changes require explicit justification and compatibility validation.

**Testing**: Pytest no-hardware tests first, using existing FM detection/characterization fixtures and controller/web diagnostic bundle tests. Hardware acceptance remains Raspberry Pi 5 plus RTL-SDR Blog v4 through the web UI and controller lifecycle.

**Target Platform**: Raspberry Pi OS field deployment with RTL-SDR Blog v4, plus local Windows development through PowerShell and bundled Codex Python.

**Project Type**: Local web dashboard plus controller service plus internal scanner backend.

**Operator Workflow Surface**: SDRwatch operator-facing features MUST use the web UI and controller job lifecycle. Treat scanner CLI work as internal backend tooling unless the feature is explicitly scanner-only.

**Performance Goals**: Cross-sweep tracking must keep scan cadence usable on Raspberry Pi 5. The feature should avoid unbounded per-window or per-candidate growth, keep diagnostic bundle exports bounded, and retain FM Validation revisit limits.

**Constraints**: Preserve `/api/jobs` request shape; preserve Discovery first-light behavior; preserve FM Validation stable-card behavior; keep raw detector span, measured characterization, match span, display span, persisted span, and contextual metadata separate; do not solve failures by threshold-only suppression; avoid demodulation or content decoding; avoid a broad classifier; prefer additive data and sidecar state over destructive migrations.

**Scale/Scope**: One scanner/controller/diagnostics feature focused on repeated complete sweep loops, FM Broadcast regression coverage, controller passthrough parity, structured diagnostic summaries, and SDR device/gain telemetry.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Raspberry Pi First Reliability**: PASS. The plan keeps state bounded, avoids new services, and requires Pi/RTL-SDR hardware validation for real receiver telemetry.
- **II. Minimal Local Stack**: PASS. The feature stays in Python, Flask/Jinja, SQLite, and existing diagnostic export paths.
- **III. Stable Interfaces and Clean Layering**: PASS. The operator flow remains web UI -> controller -> scanner. The scanner owns DSP and persistence; the controller owns job command generation and device locks; the web layer exports diagnostics.
- **IV. Adapter-Based Hardware and Honest RF Claims**: PASS. The feature records receiver telemetry and measured signal evidence without claiming emitter identity or modulation certainty from bandplan context alone.
- **V. Migration-Safe, Verifiable Change**: PASS. The default design is additive and test-first. Any durable schema expansion is deferred until planning evidence shows diagnostics/sidecar state is insufficient.
- **Operator Acceptance Gate**: PASS. User-facing validation is explicitly through the web UI and controller job lifecycle; CLI checks are backend smoke and parity checks.

## Project Structure

### Documentation (this feature)

```text
specs/007-cross-sweep-persistence-and-telemetry/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- checklists/
|   `-- requirements.md
`-- contracts/
    |-- cross-sweep-persistence-contract.md
    |-- effective-parameter-manifest-contract.md
    |-- controller-scanner-parameter-contract.md
    `-- diagnostic-telemetry-contract.md
```

### Source Code (repository root)

```text
templates/
  control.html
    Preserve GUI-first scan setup, FM Validation, Discovery, diagnostics mode,
    Copy current scan settings, and existing `/api/jobs` payload shape.

sdrwatch-control.py
  Preserve controller job lifecycle and device locking. Extend command builder
  passthrough for scanner-supported characterization, revisit, persistence, and
  width parameters. Carry requested params into job records for manifest export.

sdrwatch/cli.py
  Preserve scanner CLI compatibility while adding any missing scanner-supported
  flags needed by controller parity and effective-parameter reporting.

sdrwatch/io/profiles.py
  Keep `fm_broadcast` as the control profile and expose profile defaults needed
  by effective-parameter manifests, including skipped-profile behavior.

sdrwatch/sweep/runner.py
  Capture job/session-level device telemetry around source selection and pass
  complete sweep loop identity into scanner diagnostics.

sdrwatch/sweep/sweeper.py
  Own full sweep loop orchestration, `sweep_seq`, per-window diagnostics,
  effective tuning params, revisit summaries, and aggregate scan records.

sdrwatch/detection/
  engine.py, types.py
    Add sweep-loop-aware candidate evidence and keep characterization spans
    explicit. Preserve bounded FM matching, stable display center, and unknown
    classification defaults.

sdrwatch/baseline/
  persistence.py, store.py
    Add or host bounded cross-sweep candidate state and persistence decisions.
    Preserve stored span invariants and avoid hiding invalid geometry in
    serializers or display helpers.

sdrwatch/util/
  detection_diagnostics.py, scan_logger.py
    Emit structured event summaries and effective-parameter/device telemetry
    records that bundles can summarize without log scraping.

sdrwatch_web/
  diagnostics.py
    Export bounded diagnostic summaries, effective-parameter manifest data,
    device/gain telemetry, and missing/truncation notes.

tests/
  helpers_fm_detection.py
  helpers_fm_characterization.py
  test_fm_persistence_stability.py
  test_fm_persistence_diagnostics.py
  test_fm_characterization.py
  test_fm_characterization_persistence.py
  test_fm_characterization_diagnostics.py
  test_control_fm_validation.py
  test_control_page_scan_settings.py
  test_web_diagnostics_bundle.py
  Additional focused tests should cover cross-sweep promotion, parameter
  passthrough parity, profile manifest, diagnostic aggregates, and device
  telemetry fixtures.
```

**Structure Decision**: Keep the feature inside the existing single-repository architecture. Center implementation on scanner persistence state, controller/scanner command parity, and bounded diagnostic export. Do not add a new service, frontend framework, broad detector rewrite, or wholesale database schema replacement.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Baseline Starting Point

The `006-fm-signal-characterization` branch already provides important behavior this feature must preserve:

- FM Validation is exposed through the web GUI and submits `profile=fm_broadcast`.
- Discovery remains a separate first-light preset.
- `sdrwatch/detection/engine.py` distinguishes raw cluster extent, match span, display span, stable center, and characterization records.
- `sdrwatch/baseline/persistence.py` owns persistence matching, width EMA/hysteresis, stable-center smoothing, revisit confirmation, and stored span invariant enforcement.
- `sdrwatch/sweep/sweeper.py` already logs `sweep_id`, window diagnostics, tuning params, segment inventory, revisit events, and sweep summaries.
- `sdrwatch_web/diagnostics.py` already exports bounded diagnostic JSONL tails, decision summaries, characterization summaries, database samples, scanner command text, and manifest missing/truncation notes.
- The controller command builder currently passes many scanner flags but does not yet cover all hidden characterization fields that the scanner profile can apply internally.

The main weakness is that promotion is still window-local: `DetectionEngine` clusters by `window_idx` within a sweep and flushes at sweep end, so a signal seen once per non-overlapping full sweep loop can fail strict multi-window thresholds even when it is stable over time.

## Test-First Implementation Strategy

1. Add deterministic cross-sweep promotion tests before changing runtime behavior. The fixture should simulate a stable signal appearing once per full sweep loop in a non-overlapping window and assert promotion only after the configured number of sweep-loop observations.
2. Add FM control-band non-regression tests for close but separable FM-like signals, bounded display widths, stable display center, and raw/measured/match/display field preservation.
3. Add controller command passthrough tests for the full supported characterization/revisit/persistence parameter set.
4. Add diagnostic bundle tests for effective-parameter manifest success, skipped-profile manifest behavior, aggregate event counts, and truncation/missing evidence reporting.
5. Add no-hardware driver/source fixtures for device telemetry availability and unavailability, then leave real hardware telemetry acceptance open until Raspberry Pi plus RTL-SDR validation.
6. Implement the smallest cross-sweep state that satisfies tests while preserving FM matching bounds and existing persistence invariants.

## Phase 0 Research

Research decisions are captured in [research.md](./research.md). Key outcomes:

- Cross-sweep persistence should count compatible observations across complete sweep loops, not adjacent same-sweep windows.
- Cross-sweep matching should reuse existing profile-compatible center/span/width rules rather than broadening FM defaults.
- Effective parameters should be emitted as a structured manifest, with requested/applied/skipped/profile/override/final values separate.
- Diagnostics should aggregate counts from structured events and avoid requiring log text parsing.
- Device/gain telemetry should be best-effort and nullable when unavailable.

## Phase 1 Design

Design artifacts:

- [data-model.md](./data-model.md)
- [contracts/cross-sweep-persistence-contract.md](./contracts/cross-sweep-persistence-contract.md)
- [contracts/effective-parameter-manifest-contract.md](./contracts/effective-parameter-manifest-contract.md)
- [contracts/controller-scanner-parameter-contract.md](./contracts/controller-scanner-parameter-contract.md)
- [contracts/diagnostic-telemetry-contract.md](./contracts/diagnostic-telemetry-contract.md)
- [quickstart.md](./quickstart.md)

## Post-Design Constitution Check

- **Layering remains clean**: PASS. Scanner layers own persistence and RF evidence; controller owns job invocation; web exports bundles and does not touch hardware.
- **Existing interfaces remain compatible**: PASS. `/api/jobs` shape stays `{device_key, label, baseline_id, params}` and scanner CLI compatibility is preserved or explicitly mapped.
- **Offline/local operation remains intact**: PASS. The feature uses local scanner state, local SQLite, local diagnostics, and web/controller workflows.
- **Honest RF claims remain explicit**: PASS. FM context remains contextual and classification candidates remain unknown unless evidence supports otherwise.
- **Migration risk remains bounded**: PASS. Planning prefers in-memory or sidecar cross-sweep state plus diagnostics-first exports before any schema expansion.
- **Verification path remains GUI/controller focused**: PASS. Quickstart requires browser/controller validation and treats CLI commands as backend smoke only.
