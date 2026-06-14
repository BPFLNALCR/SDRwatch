# Tasks: Cross-Sweep Persistence and Telemetry

**Input**: Design artifacts from `specs/007-cross-sweep-persistence-and-telemetry/`
**Prerequisites**: `spec.md`, `plan.md`, `research.md`, `data-model.md`, `quickstart.md`, `contracts/`, `checklists/requirements.md`
**Branch**: `007-cross-sweep-persistence-and-telemetry`

**Tests**: Required. This feature is test-first because persistence, scanner/controller parity, diagnostics, and telemetry are existing operator contracts.
**Operator boundary**: SDRwatch remains GUI/controller-first. The scanner CLI is an internal backend entrypoint used for smoke tests and parity checks.
**Safety boundary**: No demodulation, content interception, private communication decoding, offensive SIGINT workflows, broad classifier work, multi-SDR coordination, alert/rules engine work, ML training, major UI redesign, or destructive schema replacement.

## Task Format

Tasks use `- [ ] T### [P] [US#] Description with exact file path`.

- `[P]` means the task touches independent files and can run in parallel with other `[P]` tasks in the same phase.
- `[US#]` maps to the user story in `spec.md`.
- Setup, foundational, and final validation tasks do not require a user-story label.

## Phase 1: Setup And Baseline Verification

**Purpose**: Confirm the implementation starts from the intended branch and the 006 FM characterization baseline before code changes.
**Dependencies**: None.

- [X] T001 Verify the current branch is `007-cross-sweep-persistence-and-telemetry` with `git branch --show-current` and record the command/result in `specs/007-cross-sweep-persistence-and-telemetry/quickstart.md`.
- [X] T002 Confirm the behavioral baseline is `006-fm-signal-characterization` by reviewing `specs/007-cross-sweep-persistence-and-telemetry/plan.md` and `specs/007-cross-sweep-persistence-and-telemetry/research.md`.
- [X] T003 Run the existing no-hardware FM characterization subset from `specs/007-cross-sweep-persistence-and-telemetry/quickstart.md` covering `tests/test_fm_characterization.py`, `tests/test_fm_characterization_persistence.py`, `tests/test_fm_characterization_diagnostics.py`, `tests/test_fm_persistence_stability.py`, `tests/test_fm_persistence_diagnostics.py`, and `tests/test_non_fm_width_scope.py`.
- [X] T004 Run the existing web/controller diagnostics subset from `specs/007-cross-sweep-persistence-and-telemetry/quickstart.md` covering `tests/test_control_fm_validation.py`, `tests/test_control_page_scan_settings.py`, and `tests/test_web_diagnostics_bundle.py`.

**Checkpoint**: Existing FM characterization and controller diagnostic behavior is known before implementation starts.

## Phase 2: Contract Tests Before Implementation

**Purpose**: Add failing contract tests for every required behavior before runtime changes.
**Dependencies**: Phase 1 complete.

- [X] T005 [US1] Add a cross-sweep promotion fixture where one stable signal appears once per full sweep loop in a non-overlapping window and promotes only after the configured loop count in `tests/test_cross_sweep_persistence.py`.
- [X] T006 [US1] Add a same-loop non-promotion fixture proving repeated observations inside one sweep loop cannot satisfy multi-loop persistence thresholds in `tests/test_cross_sweep_persistence.py`.
- [X] T007 [US1] Add a nearby signal separation fixture proving bounded center/span/width matching does not merge incompatible adjacent emitters in `tests/test_cross_sweep_persistence.py`.
- [X] T008 [P] [US2] Add an FM Broadcast non-regression fixture for two close but separable FM-like signals with bounded display widths and separate cards in `tests/test_fm_persistence_stability.py`.
- [X] T009 [P] [US2] Extend FM characterization diagnostics assertions so raw detector span, measured bandwidth, match span, display span, persisted span, and contextual metadata remain distinct in `tests/test_fm_characterization_diagnostics.py`.
- [X] T010 [P] [US4] Add a controller/scanner passthrough contract test for every supported characterization, revisit, and persistence parameter in `tests/test_control_fm_validation.py`.
- [X] T011 [US3] Add an in-band FM Broadcast effective-parameter manifest test asserting requested/applied profile metadata and final effective scanner parameters in `tests/test_effective_parameter_manifest.py`.
- [X] T012 [US3] Add an out-of-band FM Broadcast manifest test asserting `profile_applied=false`, `profile_skip_reason`, fallback values, and final effective scanner parameters in `tests/test_effective_parameter_manifest.py`.
- [X] T013 [P] [US5] Add a diagnostic aggregate summary test for `segment_inventory`, emitted/rejected clusters, persistence match/no-match/promote, width, revisit, and characterization counts in `tests/test_web_diagnostics_bundle.py`.
- [X] T014 [US6] Add a device telemetry availability fixture for requested gain, actual gain, gain mode, supported gains, device identity, sample rate, FFT, bin width, and selected profile in `tests/test_device_telemetry.py`.
- [X] T015 [US6] Add a device telemetry unavailability fixture proving missing driver telemetry is represented as `null` or `unavailable_fields` without failing the scan in `tests/test_device_telemetry.py`.
- [X] T016 [P] [US2] Add GUI acceptance preservation assertions for Discovery, FM Validation, Copy current scan settings, and diagnostic bundle export controls in `tests/test_control_page_scan_settings.py`.

**Checkpoint**: New tests fail for missing feature behavior and existing tests still define the control-band baseline.

## Phase 3: Cross-Sweep Persistence Implementation

**Purpose**: Make persistence loop-aware without weakening FM/control-band defaults or collapsing characterization spans.
**Dependencies**: Phase 2 US1 and US2 tests written first.

- [X] T017 [US1] Add sweep-loop observation and candidate-state data structures for `sweep_loop_id`, `observed_sweep_ids`, compatible center/span/width evidence, and promotion status in `sdrwatch/detection/types.py`.
- [X] T018 [US1] Thread the full sweep loop identifier from `sdrwatch/sweep/runner.py` through `sdrwatch/sweep/sweeper.py` into detection ingestion records.
- [X] T019 [US1] Add loop-aware candidate state in `sdrwatch/detection/engine.py` so a candidate can count at most one observation per complete sweep loop toward multi-loop thresholds.
- [X] T020 [US1] Prevent same-loop repeated windows from satisfying multi-loop persistence thresholds in `sdrwatch/detection/engine.py`.
- [X] T021 [US1] Reuse bounded center/span/width compatibility logic for cross-sweep candidates instead of widening FM match rules in `sdrwatch/detection/engine.py`.
- [X] T022 [US1] Emit structured no-match and match persistence decisions for cross-sweep candidates while keeping unpromoted candidates out of `baseline_detections` in `sdrwatch/baseline/persistence.py`.
- [X] T023 [US1] Add bounded pruning for stale cross-sweep candidate state by sweep age, frequency incompatibility, and memory limits in `sdrwatch/detection/engine.py`.
- [X] T024 [US2] Preserve raw detector span, measured center/bandwidth, match span, display span, persisted span, and contextual metadata when cross-sweep promotion creates or updates cards in `sdrwatch/baseline/persistence.py`.
- [X] T025 [US2] Verify revisit/refinement cannot permanently ratchet cards wider from a single wide observation by updating assertions in `tests/test_fm_characterization_persistence.py`.

**Checkpoint**: US1 cross-sweep promotion works, same-loop promotion is blocked, and US2 FM span separation is preserved.

## Phase 4: Scanner/Controller Parameter Parity

**Purpose**: Make web/API scans and direct scanner invocations use equivalent characterization behavior while preserving `/api/jobs`.
**Dependencies**: Phase 2 US4 test written first; Phase 3 can proceed independently except where new persistence flags are introduced.

- [X] T026 [US4] Add scanner CLI flags for `segment_center_mode`, `segment_centroid_span_hz`, `segment_centroid_drop_db`, and `segment_centroid_floor_margin_db` in `sdrwatch/cli.py`.
- [X] T027 [US4] Add scanner CLI flags for `match_bandwidth_pad_hz`, `min_match_bandwidth_hz`, `display_bandwidth_pad_hz`, and `min_display_bandwidth_hz` in `sdrwatch/cli.py`.
- [X] T028 [US4] Add scanner CLI flags or documented mapped equivalents for `max_persist_width_hz`, `max_card_width_hz`, `center_match_hz`, and `persistence_min_sweep_loops` in `sdrwatch/cli.py`.
- [X] T029 [US4] Extend profile-to-argument application so scanner CLI flags, profile defaults, and operator overrides resolve consistently in `sdrwatch/io/profiles.py`.
- [X] T030 [US4] Extend web/controller command construction so all scanner-supported characterization, revisit, and persistence params pass through from `params` in `sdrwatch-control.py`.
- [X] T031 [US4] Preserve the `POST /api/jobs` payload shape `{device_key, label, baseline_id, params}` while adding passthrough coverage in `sdrwatch-control.py`.
- [X] T032 [US4] Document any intentionally unsupported or mapped parameter names in `specs/007-cross-sweep-persistence-and-telemetry/contracts/controller-scanner-parameter-contract.md`.
- [X] T033 [US4] Update profile serialization visibility for new or mapped scanner parameters in `sdrwatch/io/profiles.py`.

**Checkpoint**: Controller/API parameter behavior matches direct scanner invocation or is explicitly documented as unsupported/mapped.

## Phase 5: Effective-Parameter Manifest

**Purpose**: Make every scan/job diagnostic bundle auditable and reproducible from structured effective settings.
**Dependencies**: Phase 2 US3 tests written first; Phase 4 should define final parameter names before finalizing this phase.

- [X] T034 [US3] Add an effective-parameter manifest builder for requested profile, applied profile, `profile_applied`, `profile_skip_reason`, profile defaults, overrides, fallback values, and final scanner params in `sdrwatch/util/detection_diagnostics.py`.
- [X] T035 [US3] Record in-band FM profile application status and final effective values from the scanner run path in `sdrwatch/sweep/sweeper.py`.
- [X] T036 [US3] Record out-of-band FM profile skipped status, skip reason, fallback values, and final effective values from the scanner run path in `sdrwatch/sweep/sweeper.py`.
- [X] T037 [US3] Include frequency range, step size, sample rate, FFT size, bin width, persistence thresholds, revisit settings, segment center mode, match/display span controls, and width caps in the manifest in `sdrwatch/util/detection_diagnostics.py`.
- [X] T038 [US3] Include requested gain, gain mode, driver/device metadata placeholders, and selected profile in the manifest in `sdrwatch/util/detection_diagnostics.py`.
- [X] T039 [US3] Mirror the effective-parameter manifest into scan log JSONL output in `sdrwatch/util/scan_logger.py`.
- [X] T040 [US3] Add the effective-parameter manifest to diagnostic bundle export in `sdrwatch_web/diagnostics.py`.
- [X] T041 [US3] Update quickstart diagnostics inspection commands for the new manifest file and JSONL event in `specs/007-cross-sweep-persistence-and-telemetry/quickstart.md`.

**Checkpoint**: Diagnostic bundles show requested/applied/skipped profile state and final effective scanner parameters for both in-band and out-of-band FM requests.

## Phase 6: Device And Gain Telemetry

**Purpose**: Record receiver state as best-effort telemetry so scan differences are not mistaken for RF-environment changes.
**Dependencies**: Phase 2 US6 tests written first; Phase 5 manifest structure in place.

- [X] T042 [US6] Add a best-effort device telemetry snapshot type for requested gain, actual gain, gain mode, supported gains, device index, serial, tuner, sample rate, actual sample rate, FFT, bin width, selected profile, and `unavailable_fields` in `sdrwatch/detection/types.py`.
- [X] T043 [US6] Capture requested gain, gain mode, sample rate, FFT, and bin width from configured scanner parameters in `sdrwatch/sweep/sweeper.py`.
- [X] T044 [US6] Capture actual gain, supported gains, device index, serial, tuner, and actual sample rate when the RTL-SDR driver exposes them in `sdrwatch/sweep/runner.py`.
- [X] T045 [US6] Represent missing driver telemetry as `null` values and explicit `unavailable_fields` without failing scans in `sdrwatch/sweep/runner.py`.
- [X] T046 [US6] Add device telemetry to the effective-parameter manifest and diagnostic JSONL events in `sdrwatch/util/detection_diagnostics.py`.
- [X] T047 [US6] Include device telemetry in diagnostic bundle export with bounded missing-field reporting in `sdrwatch_web/diagnostics.py`.

**Checkpoint**: Completed scans expose requested and available actual receiver state, and missing hardware details do not break no-hardware tests.

## Phase 7: Structured Diagnostics

**Purpose**: Make scanner/controller decisions machine-readable without scraping human log lines.
**Dependencies**: Phase 2 US5 tests written first; Phase 3, Phase 5, and Phase 6 provide event sources.

- [X] T048 [US5] Emit or normalize structured `segment_inventory`, `cluster_emit`, and `cluster_reject` events in `sdrwatch/detection/engine.py`.
- [X] T049 [US5] Emit structured `persistence_decision` events for `match`, `no_match`, and `cross_sweep_promote` actions in `sdrwatch/baseline/persistence.py`.
- [X] T050 [US5] Emit structured `width_decision`, `revisit_queue`, `revisit_result`, and `characterization_record` events with compact evidence fields in `sdrwatch/baseline/persistence.py`.
- [X] T051 [US5] Aggregate diagnostic event counts for segment inventory, emitted clusters, rejected clusters, persistence decisions, width decisions, revisit results, and characterization records in `sdrwatch_web/diagnostics.py`.
- [X] T052 [US5] Preserve bounded diagnostic tails and add explicit truncation or missing-evidence reporting in `sdrwatch_web/diagnostics.py`.
- [X] T053 [US5] Update diagnostic contract examples for event names, aggregate counts, and truncation fields in `specs/007-cross-sweep-persistence-and-telemetry/contracts/diagnostic-telemetry-contract.md`.

**Checkpoint**: Diagnostic bundles contain detailed JSONL-style events plus compact aggregate counts for debugging persistence and characterization failures.

## Phase 8: GUI/Controller Acceptance Preservation

**Purpose**: Preserve operator workflows while adding only minimal effective-settings/status exposure.
**Dependencies**: Phase 4, Phase 5, Phase 6, and Phase 7 complete enough for controller/bundle output.

- [X] T054 [US2] Preserve the FM Validation preset behavior and profile selection path in `templates/control.html`.
- [X] T055 [US2] Preserve Discovery as the first-light/default behavior in `templates/control.html`.
- [X] T056 [US4] Preserve Copy current scan settings behavior while adding passthrough-aware hidden or serialized settings in `templates/control.html`.
- [X] T057 [US5] Preserve diagnostic bundle export from the web UI while surfacing minimal profile applied/skipped/effective-settings status in `templates/control.html`.
- [X] T058 [US3] Add controller response or bundle metadata coverage for profile applied/skipped/effective-settings status without changing `/api/jobs` shape in `sdrwatch-control.py`.

**Checkpoint**: Operators can still run Discovery, FM Validation, copy current scan settings, and export diagnostics from the GUI/controller workflow.

## Phase 9: Final Validation

**Purpose**: Prove the no-hardware contract and document remaining hardware-only telemetry acceptance.
**Dependencies**: Phases 1 through 8 complete.

- [X] T059 Run the new cross-sweep and FM non-regression tests from `specs/007-cross-sweep-persistence-and-telemetry/quickstart.md`, including `tests/test_cross_sweep_persistence.py`, `tests/test_fm_persistence_stability.py`, and `tests/test_fm_characterization_diagnostics.py`.
- [X] T060 Run controller/web diagnostics tests from `specs/007-cross-sweep-persistence-and-telemetry/quickstart.md`, including `tests/test_control_fm_validation.py`, `tests/test_control_page_scan_settings.py`, `tests/test_web_diagnostics_bundle.py`, and `tests/test_effective_parameter_manifest.py`.
- [X] T061 Run device telemetry no-hardware tests from `specs/007-cross-sweep-persistence-and-telemetry/quickstart.md`, including `tests/test_device_telemetry.py`.
- [X] T062 Run existing FM characterization regression tests from `specs/007-cross-sweep-persistence-and-telemetry/quickstart.md`, including `tests/test_fm_characterization.py`, `tests/test_fm_characterization_persistence.py`, `tests/test_fm_persistence_diagnostics.py`, and `tests/test_non_fm_width_scope.py`.
- [X] T063 Run backend scanner smoke check `python -m sdrwatch.cli --list-profiles` and record the result in `specs/007-cross-sweep-persistence-and-telemetry/quickstart.md`.
- [X] T064 Leave Raspberry Pi 5 plus RTL-SDR Blog v4 hardware telemetry acceptance as a documented manual validation step in `specs/007-cross-sweep-persistence-and-telemetry/quickstart.md`.

**Checkpoint**: No-hardware tests cover the feature, scanner profile listing still works, and hardware telemetry validation remains explicit.

## Dependencies

- Phase 1 must complete before adding tests or implementation changes.
- Phase 2 tests must be added before the implementation phases they validate.
- Phase 3 implements US1 and protects US2 invariants; it depends on the US1 and US2 tests in Phase 2.
- Phase 4 implements US4 and should settle final parameter names before Phase 5 finalizes manifest fields.
- Phase 5 implements US3 and depends on Phase 4 for scanner/controller parameter names.
- Phase 6 implements US6 and depends on Phase 5 manifest structure.
- Phase 7 implements US5 and depends on event sources from Phases 3, 5, and 6.
- Phase 8 preserves GUI/controller workflows after the backend/controller diagnostics behavior exists.
- Phase 9 runs only after implementation and GUI/controller preservation work is complete.

## User Story Mapping

- **US1 Promote Signals Across Sweep Loops**: T005, T006, T007, T017, T018, T019, T020, T021, T022, T023
- **US2 Keep FM Broadcast Stable As Control Band**: T008, T009, T016, T024, T025, T054, T055
- **US3 Audit Effective Scan Parameters**: T011, T012, T034, T035, T036, T037, T038, T039, T040, T041, T058
- **US4 Match Web/API Parameters To Scanner Behavior**: T010, T026, T027, T028, T029, T030, T031, T032, T033, T056
- **US5 Debug With Structured Diagnostic Summaries**: T013, T048, T049, T050, T051, T052, T053, T057
- **US6 Record SDR Device And Gain Telemetry**: T014, T015, T042, T043, T044, T045, T046, T047

## Parallel Work Examples

After Phase 1, these tests can be drafted in parallel because they touch independent files:

```text
T005 tests/test_cross_sweep_persistence.py
T008 tests/test_fm_persistence_stability.py
T010 tests/test_control_fm_validation.py
T011 tests/test_effective_parameter_manifest.py
T013 tests/test_web_diagnostics_bundle.py
T014 tests/test_device_telemetry.py
T016 tests/test_control_page_scan_settings.py
```

After Phase 2, these implementation tracks can proceed mostly in parallel once shared parameter names are agreed:

```text
T017-T023 cross-sweep persistence in sdrwatch/detection/ and sdrwatch/baseline/
T026-T033 scanner/controller parameter parity in sdrwatch/cli.py, sdrwatch/io/profiles.py, and sdrwatch-control.py
T034-T041 effective-parameter manifest in sdrwatch/util/ and sdrwatch_web/
T042-T047 device telemetry in sdrwatch/sweep/, sdrwatch/detection/types.py, and sdrwatch_web/
```

## Independent Test Criteria

- **US1**: `tests/test_cross_sweep_persistence.py` can prove a stable once-per-loop signal promotes after the configured loop threshold, same-loop repeats do not promote, and nearby incompatible signals remain separate.
- **US2**: `tests/test_fm_persistence_stability.py` and `tests/test_fm_characterization_diagnostics.py` can prove FM cards remain stable, bounded, separated, and preserve raw/measured/match/display/persisted/context fields.
- **US3**: `tests/test_effective_parameter_manifest.py` can prove in-band FM requests apply profiles, out-of-band requests record skipped profiles, and final effective scanner parameters are reproducible.
- **US4**: `tests/test_control_fm_validation.py` can prove all supported characterization, revisit, and persistence parameters pass from web/API params to scanner CLI flags without changing `/api/jobs`.
- **US5**: `tests/test_web_diagnostics_bundle.py` can prove diagnostic bundles include detailed structured events and aggregate counts without parsing human scanner logs.
- **US6**: `tests/test_device_telemetry.py` can prove hardware telemetry is captured when available and recorded as unavailable/null when absent.

## Suggested MVP Scope

MVP is Phase 1, all Phase 2 contract tests required for US1/US2, and Phase 3 cross-sweep persistence implementation. Because the user requested all contract tests before implementation, the full Phase 2 test shell should be created before starting runtime code.

## Final Acceptance Coverage Checklist

- [X] Stable signal observed once per full sweep loop promotes only after the configured number of sweep observations.
- [X] Same-loop repeated observations cannot satisfy multi-loop persistence thresholds.
- [X] Nearby FM-like or compatible-band signals remain separate under bounded center/span/width matching.
- [X] FM Broadcast profile remains a stable control band with bounded separated cards.
- [X] Raw detector span, measured bandwidth, match span, display span, persisted span, and contextual metadata remain distinct.
- [X] Web/API launched scans pass every supported characterization, revisit, and persistence parameter to the scanner or document unsupported/mapped names.
- [X] In-band FM profile diagnostics record requested profile, applied profile, successful application, overrides, and final effective scanner parameters.
- [X] Out-of-band FM profile diagnostics record requested profile, skipped application, skip reason, fallback values, and final effective scanner parameters.
- [X] Diagnostic bundles include aggregate counts for emitted clusters, rejected clusters, persistence matches/no-matches/promotions, width decisions, revisit results, and characterization records.
- [X] Device/gain telemetry records requested gain, actual gain when available, gain mode, device identity/index when available, sample rate, FFT size, bin width, selected profile, and unavailable fields.
- [X] Discovery, FM Validation, Copy current scan settings, `/api/jobs`, and web diagnostic bundle export remain preserved.
- [X] Hardware telemetry validation on Raspberry Pi 5 plus RTL-SDR Blog v4 remains explicitly documented as manual acceptance when not run.
