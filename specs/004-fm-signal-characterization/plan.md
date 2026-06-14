# Implementation Plan: FM Signal Characterization

**Branch**: `006-fm-signal-characterization` | **Date**: 2026-06-12 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/004-fm-signal-characterization/spec.md`

## Summary

Add an evidence-based FM signal characterization layer on top of the existing FM Validation stability work. The current code already separates raw detector segments from match span and display span, and it already stores contextual bandplan labels separately from web-side operator classifications. This feature builds on that foundation by introducing a distinct measured characterization layer for center estimate, occupied bandwidth estimate, stability, confidence, revisit-derived refinement, and optional FM-specific indicators while preserving stable station-scale cards, Discovery behavior, and the existing `/api/jobs` contract.

The plan should stay narrow. The preferred first implementation path is to expose characterization evidence through diagnostics and internal summaries before adding new database columns. A schema change remains explicitly optional and must be justified only if cross-sweep stability tracking or later classification work cannot be supported cleanly through the existing persistence contracts plus additive export evidence.

## Technical Context

**Language/Version**: Python 3 project on the current `devControl` baseline.

**Primary Dependencies**: Flask web app, server-rendered Jinja templates, controller HTTP API, scanner modules under `sdrwatch/`, NumPy-based DSP, SQLite persistence, diagnostic JSONL and bundle export.

**Storage**: Existing SQLite baseline tables plus diagnostic JSONL and bounded diagnostic bundle exports. No schema migration is assumed by default for this feature.

**Testing**: Pytest no-hardware tests first, focused on FM-like fixtures, persistence invariants, revisit refinement, diagnostics export, and GUI payload preservation. Hardware acceptance remains web UI and controller driven.

**Target Platform**: Raspberry Pi field deployment with RTL-SDR Blog v4 and the current local Windows development environment.

**Project Type**: Local web dashboard plus controller service plus internal scanner backend.

**Operator Workflow Surface**: SDRwatch operator-facing features MUST use the web UI and controller job lifecycle. Scanner CLI work remains internal backend smoke coverage only.

**Performance Goals**: Preserve the current FM Validation scan cadence and card stability on Raspberry Pi 5 while allowing bounded extra measurement work for revisit refinement and optional FM-specific evidence.

**Constraints**: Preserve `/api/jobs` request shape; preserve Discovery first-light behavior; preserve FM Validation stable cards; keep raw, measured, match, display, and contextual concepts separate; avoid a broad detector rewrite; avoid a schema migration unless justified; keep diagnostics bounded; avoid CLI-first workflows.

**Scale/Scope**: One characterization foundation focused on FM broadcast as the control target, with narrow changes in detection, persistence, diagnostics, and optionally UI evidence surfacing.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Raspberry Pi First Reliability**: PASS. The feature builds on the existing FM Validation path and keeps new measurement work bounded.
- **II. Minimal Local Stack**: PASS. The feature stays inside Python, Flask/Jinja, SQLite, and existing diagnostic/export paths.
- **III. Stable Interfaces and Clean Layering**: PASS. Operator flow remains browser -> web UI -> controller -> scanner CLI. The `/api/jobs` contract stays unchanged.
- **IV. Adapter-Based Hardware and Honest RF Claims**: PASS. The feature explicitly distinguishes contextual metadata from measured RF evidence and avoids overclaiming modulation or service identity.
- **V. Migration-Safe, Verifiable Change**: PASS. The default plan is additive, test-first, and schema-conservative. Any persistent-field expansion must be justified before implementation.
- **Operator Acceptance Gate**: PASS. Manual acceptance remains through the web UI and controller lifecycle, with CLI checks limited to backend smoke coverage.

## Baseline Starting Point

The current `003-fm-detection-card-stability` work already established several durable behaviors that this feature must preserve:

- FM Validation keeps stable station-scale FM cards rather than hundreds of tiny fragments.
- Detection engine shaping already distinguishes raw cluster extent, match span, and display span.
- Persistence already logs width and revisit decisions and preserves the existing `baseline_detections` contract.
- `bandplan_service`, `region`, and notes are already contextual metadata stored alongside persistent detections.
- Web-layer signal classification is already a separate concern from scanner-side bandplan labels.

This means the next feature should add a new measured characterization concept instead of overloading the existing match or display span concepts.

## Project Structure

### Documentation (this feature)

```text
specs/004-fm-signal-characterization/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
|-- checklists/
|   `-- requirements.md
`-- contracts/
    |-- characterization-persistence-contract.md
    |-- diagnostic-characterization-evidence-contract.md
    `-- fm-characterization-api-jobs-contract.md
```

### Source Code (repository root)

```text
sdrwatch/io/
  profiles.py
  bandplan.py
    Existing FM profile defaults and contextual bandplan metadata. Preserve the
    distinction between contextual labels and measured evidence.

sdrwatch/dsp/
  detection.py
    Raw detector segment formation and optional band-shape helpers. Avoid
    broad rewrites; use this layer only for narrow measurement improvements.

sdrwatch/detection/
  engine.py
    Current raw-cluster to match-span and display-span shaping. Candidate place
    for measured characterization records and confidence provenance.

sdrwatch/baseline/
  persistence.py
  store.py
    Current persistent signal matching, revisit updates, invariants, and
    contextual metadata storage. Preserve compatibility unless design proves a
    new additive persistence surface is necessary.

sdrwatch/sweep/
  sweeper.py
    Current coarse and revisit orchestration plus diagnostic window records.
    Likely place to surface characterization summaries and revisit provenance.

sdrwatch_web/
  diagnostics.py
  baseline_helpers.py
    Existing bounded export path and display helpers. Candidate place for
    characterization bundle summaries and later UI evidence surfacing.

templates/
  control.html
    Preserve FM Validation and Discovery separation. No operator-facing feature
    should require CLI-first usage.

tests/
  test_fm_persistence_stability.py
  test_fm_persistence_diagnostics.py
  test_non_fm_width_scope.py
  test_fm_validation_profile.py
  test_web_diagnostics_bundle.py
  Additional focused characterization tests should be added before any runtime
  change lands.
```

**Structure Decision**: Keep the feature inside the existing single-repository architecture and center implementation on narrow additions to detection, persistence, and diagnostics. Do not add a new service, a new frontend framework, or a large schema redesign.

## Test-First Implementation Strategy

1. Extend or add deterministic no-hardware tests for the eight requested cases before changing runtime behavior.
2. Define a characterization model that explicitly separates:
   - raw detector segment
   - measured characterization
   - persistence match span
   - display card span
   - contextual metadata
3. Add characterization evidence plumbing in diagnostics first so behavior can be inspected without committing to a schema migration.
4. Only after diagnostics-first evidence is clear, decide whether any additive persistent fields are required for center and bandwidth stability across sweeps.
5. Treat optional FM-specific indicators as non-blocking enhancements behind clear confidence and provenance reporting.

## Open Questions

- Should the first implementation add measured characterization fields to `baseline_detections` now, or should it emit them only through diagnostics and internal summaries until the persistence need is proven?
- If durable center and bandwidth stability metrics are needed across long runs, is the narrowest path additive columns on `baseline_detections`, a sidecar characterization structure, or derived summaries at export time?
- Are 19 kHz pilot and 57 kHz RDS or RBDS indicators reliable enough at current coarse and revisit resolutions to be part of the first implementation, or should they remain explicitly optional?
- How much of characterization evidence needs to be visible in the UI immediately versus only in the diagnostic bundle for the first pass?

## Risks And Mitigations

- **Risk**: Measured occupied bandwidth could be accidentally collapsed into match or display width.
  **Mitigation**: Keep separate field names, separate diagnostics, and dedicated tests for raw vs measured vs match vs display spans.
- **Risk**: Nearby FM stations could be merged while pursuing more stable measurements.
  **Mitigation**: Preserve existing FM match-span behavior as the compatibility baseline and keep dedicated nearby-station separation tests.
- **Risk**: Schema expansion could create migration pressure before the model is proven.
  **Mitigation**: Default to diagnostics-first evidence and require an explicit justification before adding persistent fields.
- **Risk**: Optional FM-specific indicators could be read as proof rather than supporting evidence.
  **Mitigation**: Keep those indicators inside evidence and confidence reporting, never as the sole classification trigger.
- **Risk**: Additional revisit or characterization work could hurt Raspberry Pi scan cadence.
  **Mitigation**: Keep revisit bounded, make added work profile-scoped to FM Validation, and preserve the current Discovery first-light path.

## Phase 0 Research

Research decisions are captured in [research.md](./research.md). The key directions are:

- Preserve stable FM cards while adding a distinct measured characterization layer.
- Treat bandplan and profile labels as contextual inputs, not measured classification proof.
- Prefer diagnostics-first emission for new characterization evidence and defer schema migration unless a persistent need is proven.
- Use revisit refinement and stability-over-sweeps as evidence sources, not as reasons to create more cards.

## Phase 1 Design

Design artifacts:

- [data-model.md](./data-model.md)
- [contracts/fm-characterization-api-jobs-contract.md](./contracts/fm-characterization-api-jobs-contract.md)
- [contracts/diagnostic-characterization-evidence-contract.md](./contracts/diagnostic-characterization-evidence-contract.md)
- [contracts/characterization-persistence-contract.md](./contracts/characterization-persistence-contract.md)
- [quickstart.md](./quickstart.md)

## Post-Design Constitution Check

- **Layering remains clean**: PASS. Characterization is planned as scanner-derived evidence with web and diagnostics acting as presentation layers.
- **Existing interfaces remain compatible**: PASS. `/api/jobs` shape, Discovery behavior, and current stable FM cards remain intact.
- **Offline and local operation remain intact**: PASS. The feature relies on local scanning, local SQLite, and bounded local exports.
- **Honest RF claims remain explicit**: PASS. Contextual labels stay separate from measured evidence and confidence.
- **No broad rewrite is planned**: PASS. The plan stays narrow, test-first, and schema-conservative.
- **Verification path remains GUI and controller focused**: PASS. Quickstart preserves browser and controller acceptance as the operator workflow.
