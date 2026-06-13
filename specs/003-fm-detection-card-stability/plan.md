# Implementation Plan: FM Detection Card Stability

**Branch**: `devControl` | **Date**: 2026-06-12 | **Spec**: [spec.md](./spec.md)

**Input**: Feature specification from `specs/003-fm-detection-card-stability/spec.md`

**Evidence**: [docs/FM_DETECTION_CARD_EXPLOSION_REPORT.md](../../docs/FM_DETECTION_CARD_EXPLOSION_REPORT.md)

## Summary

Add a GUI-first FM Validation path that stabilizes 88-108 MHz signal cards without regressing the existing RTL-SDR v4 Discovery first-light preset. The diagnostic bundle shows card overproduction, not starvation: a short FM-band run persisted 234 baseline detections, 209 were already missing at export, and 193 were under 1 kHz wide. The likely fix is not a threshold-only change or detector rewrite; it is to route FM Validation through FM-specific profile/span-shaping behavior, tighten baseline matching/upsert behavior for repeated nearby FM fragments, bound and observe width clamping, enable or explicitly cover two-pass behavior, and improve diagnostics so create/update/merge/missing/revisit decisions are visible.

## Technical Context

**Language/Version**: Python 3 project; no committed project version pin on the active baseline.

**Primary Dependencies**: Flask web app, server-rendered Jinja templates, embedded browser JavaScript in `templates/control.html`, existing controller HTTP API, scanner modules under `sdrwatch/`, NumPy-based DSP, SQLite baseline store.

**Storage**: Existing SQLite baseline tables only. No schema migration is planned. The change may alter how detections are matched, updated, or logged, but it should preserve current table contracts.

**Testing**: Pytest no-hardware tests first, covering detector/persistence fixtures, profile/controller command behavior, diagnostics, and control-page `/api/jobs` payloads. GUI/controller hardware validation remains required for acceptance.

**Target Platform**: Raspberry Pi OS field deployment and local development on the current `devControl` baseline.

**Project Type**: Local web dashboard plus controller service plus internal scanner backend.

**Operator Workflow Surface**: SDRwatch operator-facing features use the web UI and controller job lifecycle. Scanner CLI checks are internal backend smoke tests only.

**Performance Goals**: FM Validation should keep scan cadence usable on Raspberry Pi 5 with RTL-SDR Blog v4. Enabling overlap, two-pass, or larger FFT must be bounded by revisit limits and documented in the GUI preset.

**Constraints**: Preserve `/api/jobs` request shape; preserve Discovery as first-light; no frontend framework; no database schema rewrite; no broad detector rewrite; no threshold-only solution; no CLI-first operator workflow; diagnostics must remain bounded for bundle export.

**Scale/Scope**: One FM-band GUI preset/path, focused scanner/profile/controller passthrough if needed, targeted detection/persistence diagnostics, and regression tests. Code areas of interest are `sdrwatch/detection/engine.py`, `sdrwatch/baseline/persistence.py`, `sdrwatch/sweep/sweeper.py`, `sdrwatch/sweep/runner.py`, `sdrwatch/sweep/scheduler.py`, `sdrwatch/dsp/clustering.py`, `sdrwatch/detection/types.py`, `sdrwatch/io/profiles.py`, `sdrwatch/cli.py`, `sdrwatch-control.py`, `templates/control.html`, and related tests.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

- **I. Raspberry Pi First Reliability**: PASS. The planned changes stay local and bounded, and any two-pass/FM overlap behavior must be limited for Pi scan cadence.
- **II. Minimal Local Stack**: PASS. The plan stays in Python, Flask/Jinja, SQLite, and existing browser JavaScript.
- **III. Stable Interfaces and Clean Layering**: PASS. Operator workflow remains web/controller; scanner CLI remains backend. `/api/jobs` shape is preserved.
- **IV. Adapter-Based Hardware and Honest RF Claims**: PASS. No new SDR driver or emitter identity claim is introduced. The feature stabilizes measured FM-band signal cards only.
- **V. Migration-Safe, Verifiable Change**: PASS. No schema migration is planned. Automated no-hardware tests precede implementation, and hardware acceptance runs through the web UI/controller lifecycle.
- **Operator Acceptance Gate**: PASS. Acceptance is explicitly through the web GUI and controller jobs, with CLI checks limited to backend smoke.

## Project Structure

### Documentation (this feature)

```text
specs/003-fm-detection-card-stability/
|-- spec.md
|-- plan.md
|-- research.md
|-- data-model.md
|-- quickstart.md
`-- contracts/
    |-- diagnostic-evidence-contract.md
    `-- fm-validation-api-jobs-contract.md
```

### Source Code (repository root)

```text
templates/
  control.html
    Add or adjust the GUI FM Validation preset while preserving Discovery,
    Copy current scan settings, diagnostics mode, autocomplete behavior, and
    existing `/api/jobs` parameter names.

sdrwatch-control.py
  Preserve controller job lifecycle and command construction. Add narrow
  passthrough only if FM Validation cannot rely on existing `--profile`.

sdrwatch/io/profiles.py
  Existing `fm_broadcast` profile is the preferred source for FM-specific
  settings such as overlap, two-pass, width shaping, matching, and revisit
  limits.

sdrwatch/cli.py
  Preserve scanner CLI compatibility. Add only narrow existing-field flags if
  profile-only behavior is too implicit for GUI/controller validation.

sdrwatch/detection/
  engine.py, types.py
    Candidate clustering, span shaping, and diagnostics for cluster emit/reject.
    Prefer small FM-profile-aware fixes and observability over rewrites.

sdrwatch/baseline/
  persistence.py
    Matching/upsert, width EMA/hysteresis, missing marking, revisit scheduling,
    and diagnostics for create/update/no-match/width decisions.

sdrwatch/sweep/
  sweeper.py, runner.py, scheduler.py
    Keep orchestration thin. Ensure two-pass/revisit counts and tuning params
    remain visible in diagnostics.

sdrwatch/dsp/
  clustering.py
    Use existing bandwidth helpers where needed; avoid wholesale detector
    rewrites.

tests/
  test_control_page_scan_settings.py
  test_control_diagnostics_mode.py
  test_detection_diagnostics.py
  test_extent_hysteresis.py
  test_segment_splitting.py
  New or extended tests for FM Validation, FM-like fixtures, persistence
  upserts, width clamps, two-pass behavior, and diagnostics decisions.

docs/
  FM_DETECTION_CARD_EXPLOSION_REPORT.md
```

**Structure Decision**: Keep the change in the existing single repository layout. Center implementation on GUI preset/profile wiring, targeted persistence/span-shaping behavior, and diagnostics observability. Do not add a new service, frontend build system, schema migration, or detector rewrite.

## Complexity Tracking

No constitution violations or complexity exceptions are required.

## Evidence Summary

The required report is captured in `docs/FM_DETECTION_CARD_EXPLOSION_REPORT.md`. Key evidence:

- FM run range was 88-108 MHz with RTL-SDR Blog v4, `rtlsdr_native`, `samp_rate=2400000`, `step=2400000`, `fft=8192`, `avg=8`, `threshold_db=8`, `gain=30`, and Discovery-style `persistence_min_hits=1`, `persistence_min_windows=1`.
- `baseline-detections.json` contains 234 persisted rows.
- 25 rows were active and 209 rows were missing at export.
- 193 of 234 persisted rows were under 1 kHz wide; median persisted width was 586 Hz.
- All active rows were under 5 kHz wide, and 22 of 25 active rows were under 1 kHz wide.
- Missing rows were marked missing quickly: median 3.12 seconds after last seen, all within 10 seconds.
- Diagnostics show 1408 emitted segments, 1331 accepted hits, 828 promoted detections, and 249 new signals across 558 windows.
- Two-pass was disabled: no `--two-pass`, no `--profile`, `tuning_params.two_pass=false`, and all exported scan updates have `num_revisits=0`.

## Hypothesis Of The Failing Stage

The likely failing stage is a combination of FM-band segmentation plus promotion/upsert matching, with missing FM profile wiring and diagnostics opacity:

- **Segmentation**: FM-band windows emit many narrow spike fragments rather than station-scale spans.
- **Promotion**: Discovery `1/1` gates promote fragments immediately. This is desirable for first-light, but too permissive for FM Validation.
- **Baseline upsert/matching**: With tight sub-kHz spans and no FM-specific match/display shaping, repeated nearby fragments often become separate rows instead of updating stable FM-scale rows.
- **Width clamp behavior**: `max_detection_width_hz` was effectively disabled, and the GUI did not apply the existing FM profile's minimum display/match widths or hard cap.
- **Two-pass wiring**: The run did not include `--two-pass` or `--profile`, so revisit confirmation/pruning never ran.
- **Diagnostics opacity**: The bundle shows window-level emitted/promoted counts and final DB rows, but not enough insert/update/no-match/width/revisit decisions.

This is not candidate starvation. The system produces many candidates, accepted hits, promoted detections, and persisted rows.

## Minimal Implementation Strategy

1. **Add tests before implementation** for FM-like spiky signals, separated FM-like signals, non-FM narrow signals, repeated nearby upserts, width clamps, two-pass command/settings behavior, GUI payloads, and diagnostics decisions.
2. **Add a GUI FM Validation preset** separate from RTL-SDR v4 Discovery. The preset should be visible and copyable from the control page.
3. **Prefer existing `fm_broadcast` profile behavior** for FM Validation. The profile already contains key FM-specific settings: `step_hz=1.2e6`, `fft=8192`, `avg=10`, `guard_bins=3`, `min_width_bins=5`, `cfar_train=32`, `cfar_guard=6`, `cfar_quantile=0.6`, `persistence_min_hits=1`, `persistence_min_windows=1`, `two_pass=True`, `min_match_bandwidth_hz=80000`, `min_display_bandwidth_hz=200000`, `center_match_hz=60000`, `cluster_merge_hz=12000`, `max_detection_width_hz=270000`, centroid mode, and revisit limits.
4. **Make profile application explicit enough to test**. Either have FM Validation submit `profile=fm_broadcast` plus compatible visible values, or add narrow controller/CLI passthrough for existing profile fields if relying on hidden profile-only values is too opaque.
5. **Do not remove Discovery's relaxed `1/1` behavior**. Keep it as first-light. FM Validation can use profile shaping, overlap, width bounds, and two-pass behavior to stabilize cards.
6. **Improve persistence observability** with structured diagnostic events or compact summaries for `persist_no_match`, `persist_match`, insert/update, width clamp/floor, missing marking, revisit queue/filter/apply, and final counts.
7. **Keep changes narrow**. Do not rewrite CFAR, PSD, schema, dashboard rendering, or controller lifecycle unless a test exposes a small compatibility bug.

## Implementation Choice

Implemented on 2026-06-12 as a profile-driven GUI path. FM Validation is an additive control-page preset that submits existing `/api/jobs` params, including `profile=fm_broadcast`, explicit two-pass/revisit values, and visible width cap/matching controls. No new detector algorithm or database schema was added.

The scanner already applied the FM profile's hidden stability fields (`center_match_hz`, match/display bandwidth floors, centroid fields, and confidence normalizers). The implementation made those settings observable in sweeper diagnostic `tuning_params`, mirrored structured scan logs into the diagnostic JSONL when diagnostics are enabled, and added normalized `persistence_decision` and `width_decision` events for bundle summaries.

## Test Strategy Before Implementation

Automated tests should be written or extended before code changes:

- **FM-like wide/spiky signal**: Build synthetic segments or PSD fixtures where several narrow peaks occupy one FM station-scale region. Under FM Validation settings, assert a small bounded number of persisted rows, not dozens.
- **Separated FM-like signals**: Two or more FM-like regions separated by realistic spacing remain separate persistent detections.
- **Narrow non-FM signal**: Default/non-FM settings preserve a narrow carrier and do not force FM-scale display or match width.
- **Repeated nearby upsert**: Repeated centers within FM Validation tolerance update existing `baseline_detections`, clear missing state, and increase hit/window counts.
- **Width clamp behavior**: Min match/display width and max width cap are tested and diagnostics show before/after widths.
- **Two-pass behavior**: If FM Validation enables two-pass, controller command or scanner-applied settings and diagnostic revisit counts are covered.
- **GUI payload**: FM Validation submits correct existing `/api/jobs` params, including `profile=fm_broadcast` and/or explicit FM settings; Discovery remains available and card-producing.
- **Diagnostics export**: A no-hardware diagnostic fixture exports create/update/no-match/missing/width/revisit decisions within bundle bounds.

Suggested focused files:

- `tests/test_control_page_scan_settings.py`
- `tests/test_detection_diagnostics.py`
- `tests/test_extent_hysteresis.py`
- `tests/test_segment_splitting.py`
- New tests for `BaselinePersistence` upsert/matching and FM Validation profile behavior if no existing file fits.

## GUI/Controller Validation Path

1. Start from the web GUI, not scanner CLI.
2. Select an FM monitoring zone covering 88-108 MHz and an RTL-SDR Blog v4 device.
3. Select FM Validation.
4. Enable Diagnostics mode.
5. Use Copy current scan settings and confirm the JSON uses existing `/api/jobs` names and includes the planned FM Validation settings.
6. Start the scan from the web page and watch live logs.
7. Confirm the controller-generated command or diagnostic tuning params show FM profile/two-pass behavior as intended.
8. Export a diagnostic bundle from the GUI.
9. Confirm accepted hits exist, persisted card count is stable and not in the hundreds, active cards are not dominated by sub-5 kHz widths, and diagnostics explain create/update/merge/missing/revisit decisions.
10. Repeat Discovery from the GUI and confirm it remains available and still produces first-light cards.

## Risks And Non-Goals

**Risks**:

- FM profile values may be partly hidden behind `--profile`; tests must prove whether GUI/controller payloads actually apply them.
- Overlapping step and two-pass revisit can slow Raspberry Pi scans; revisit limits and GUI copy must make this visible.
- Too much FM merging can combine adjacent stations; fixtures must protect separation.
- Too much widening can break non-FM narrow-signal behavior; tests must scope FM-specific behavior.
- Existing tiny/missing baseline rows from prior runs may make validation noisy unless test baselines are fresh or cleanup is explicit.
- Diagnostic bundles are bounded; new evidence should be summarized so it remains useful under export limits.

**Non-goals**:

- No detector rewrite.
- No database schema migration.
- No dashboard redesign.
- No simulation mode.
- No new hardware driver.
- No CLI-first operator workflow.
- No threshold-only fix that hides cards instead of stabilizing them.

## Phase 0 Research

Research decisions are captured in [research.md](./research.md). Key outcomes:

- The failure is card overproduction and instability, not starvation.
- FM Validation should reuse or expose existing `fm_broadcast` profile behavior before adding new detection algorithms.
- Discovery and FM Validation should remain separate presets with separate acceptance expectations.
- Diagnostics need decision summaries for persistence and revisit behavior.

## Phase 1 Design

Design artifacts:

- [data-model.md](./data-model.md)
- [contracts/fm-validation-api-jobs-contract.md](./contracts/fm-validation-api-jobs-contract.md)
- [contracts/diagnostic-evidence-contract.md](./contracts/diagnostic-evidence-contract.md)
- [quickstart.md](./quickstart.md)

## Post-Design Constitution Check

- **Layering remains clean**: PASS. UI submits params; controller owns jobs and command generation; scanner owns DSP/detection/persistence.
- **Existing interfaces remain compatible**: PASS. `/api/jobs` shape and baseline tables remain unchanged.
- **Offline/local operation remains intact**: PASS. No cloud or new service is planned.
- **Discovery remains available**: PASS. FM Validation is additive and separately tested.
- **No broad rewrite**: PASS. Planned implementation is profile wiring, targeted matching/span behavior, and diagnostics.
- **Verification path is GUI/controller focused**: PASS. Quickstart uses browser/controller lifecycle; CLI commands are internal smoke only.
