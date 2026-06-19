# Tasks: Profile-Governed Signal Identity Span and Revisit Authority

**Input**: Design documents from `specs/008-multi-rtl-guard-rover/`

**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md), [data-model.md](./data-model.md), [contracts/signal-span-policy-contract.md](./contracts/signal-span-policy-contract.md), [quickstart.md](./quickstart.md)

**Tests**: Required. The specification and plan require no-hardware tests before or alongside implementation, plus regression coverage for cross-sweep persistence, effective-parameter/profile export, Slice 1 multi-RTL inventory/backend gating, and legacy `/api/jobs` compatibility. Operator acceptance remains web GUI -> controller job lifecycle -> scanner backend; scanner CLI checks are backend smoke only.

**Organization**: Tasks are grouped by the current span-policy user stories. This task list replaces the stale hardware-aware multi-RTL task list; do not continue from the old multi-RTL role-run slices.

**Feature Boundary**: This is a generic profile-policy update. Do not add FM-specific tuning, 88-108 MHz special cases, new multi-RTL role assignment or grouped role-run work, Airspy/HackRF/Soapy runtime support, UI redesign, signal fusion schema, database migration, continuous IQ capture, Rust DSP rewrite, or broad detector threshold retuning from the live FM canary.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Parallelizable only when the task touches different files and does not depend on an incomplete behavior.
- **[Story]**: Required for user-story phases.
- All descriptions include concrete file paths.

---

## Phase 1: Setup and Artifact Alignment

**Purpose**: Confirm the current Spec Kit artifacts and prepare focused no-hardware fixtures before runtime edits.

- [X] T001 Review the aligned span-policy artifacts in `specs/008-multi-rtl-guard-rover/spec.md`, `specs/008-multi-rtl-guard-rover/plan.md`, `specs/008-multi-rtl-guard-rover/data-model.md`, `specs/008-multi-rtl-guard-rover/contracts/signal-span-policy-contract.md`, and `specs/008-multi-rtl-guard-rover/quickstart.md` before editing runtime code
- [X] T002 Verify `.specify/feature.json` points to `specs/008-multi-rtl-guard-rover` and do not restore stale multi-RTL task content in `specs/008-multi-rtl-guard-rover/tasks.md`
- [X] T003 [P] Create focused shared fixtures for broad continuous, narrowband, discovery, and guard/event policy scenarios in `tests/test_signal_span_policy.py`
- [X] T004 [P] Inventory existing span and revisit touchpoints in `sdrwatch/detection/engine.py`, `sdrwatch/baseline/persistence.py`, `sdrwatch/detection/types.py`, `sdrwatch/sweep/sweeper.py`, and `sdrwatch/util/detection_diagnostics.py`

**Checkpoint**: Current artifacts and test fixture locations are confirmed before implementation.

---

## Phase 2: Foundational Policy Plumbing

**Purpose**: Add the shared policy representation and effective-parameter plumbing that all user stories depend on.

**Critical**: No story implementation should rely on profile-name branches such as `profile == "fm_broadcast"` in generic detection or persistence code.

- [X] T005 Add a `SignalSpanPolicy` dataclass and `resolve_signal_span_policy(args)` helper in `sdrwatch/detection/span_policy.py`
- [X] T006 Extend `ScanProfile` with optional span-policy fields and profile dictionary serialization in `sdrwatch/io/profiles.py`
- [X] T007 Add scanner CLI arguments, default normalization, and profile application for span-policy fields in `sdrwatch/cli.py`
- [X] T008 Add controller pass-through mappings for new span-policy params without changing the `/api/jobs` payload shape in `sdrwatch-control.py`
- [X] T009 Wire the resolved `SignalSpanPolicy` into detection and persistence construction in `sdrwatch/detection/engine.py` and `sdrwatch/baseline/persistence.py`
- [X] T010 Expose the derived policy under `signal_span_policy` or additive `span_controls` fields in `sdrwatch/util/detection_diagnostics.py` and `sdrwatch/sweep/sweeper.py`

**Checkpoint**: Policy fields can flow from profile/operator params to scanner internals and effective parameters without changing database schema or API shape.

---

## Phase 3: User Story 1 - Preserve Honest Raw Evidence While Showing Policy-Shaped Identity (Priority: P1) MVP

**Goal**: Raw detector/revisit fragments remain honest and tiny when appropriate, while identity/match, persisted/card, and display spans are separately policy-shaped.

**Independent Test**: A no-hardware broad-profile fixture emits a tiny raw segment and proves raw width remains tiny while identity, persisted/card, and display spans obey their own policy floors.

### Tests First

- [X] T011 [P] [US1] Add failing policy-default tests for `min_identity_bandwidth_hz` and `min_persist_bandwidth_hz` deriving from `min_match_bandwidth_hz` in `tests/test_signal_span_policy.py`
- [X] T012 [P] [US1] Add a tiny raw detector fragment test proving `raw_fragment_bandwidth_hz` remains tiny while `identity_match_bandwidth_hz` is floored in `tests/test_fm_characterization_persistence.py`
- [X] T013 [P] [US1] Add a persistence EMA test proving `persisted_card_bandwidth_hz` cannot shrink below `min_persist_bandwidth_hz` in `tests/test_extent_hysteresis.py`
- [X] T014 [P] [US1] Add a display-span independence test proving `display_bandwidth_hz` follows `min_display_bandwidth_hz` and is not reported as measured occupied bandwidth in `tests/test_fm_characterization_diagnostics.py`
- [X] T015 [P] [US1] Add a scan-edge clipping test with `baseline_clipped` and `clip_reason` diagnostics in `tests/test_signal_span_policy.py`

### Implementation

- [X] T016 [US1] Add additive raw-fragment aliases and bandwidth interpretation fields to `CharacterizationEvidence` serialization in `sdrwatch/detection/types.py`
- [X] T017 [US1] Apply `min_identity_bandwidth_hz` and `min_match_bandwidth_hz` when deriving identity/match spans in `sdrwatch/detection/engine.py`
- [X] T018 [US1] Emit `identity_match_bandwidth_hz`, `width_floor_applied_hz`, and `bandwidth_interpretation` in characterization records from `sdrwatch/detection/engine.py`
- [X] T019 [US1] Apply `min_persist_bandwidth_hz` before and after persistence width EMA blending in `sdrwatch/baseline/persistence.py`
- [X] T020 [US1] Clamp final persisted/card spans with `min_persist_bandwidth_hz` and `max_persist_bandwidth_hz` before store insert/update in `sdrwatch/baseline/persistence.py`
- [X] T021 [US1] Emit `persisted_card_bandwidth_hz`, `persist_width_floor_applied_hz`, `baseline_clipped`, and `clip_reason` diagnostics from `sdrwatch/baseline/persistence.py`
- [X] T022 [US1] Preserve display span shaping as operator-facing presentation and prevent display width from replacing measured occupied bandwidth in `sdrwatch/detection/engine.py`

### Validation

- [X] T023 [US1] Run `python -m pytest -q tests/test_signal_span_policy.py tests/test_extent_hysteresis.py tests/test_fm_characterization_persistence.py tests/test_fm_characterization_diagnostics.py --basetemp .test-tmp\\span-policy-us1` from repository root `C:\Users\User\SDRwatch`

**Checkpoint**: US1 is independently testable and demonstrates the core MVP behavior.

---

## Phase 4: User Story 2 - Gate Revisit Authority Separately from Confirmation (Priority: P1)

**Goal**: Revisit evidence can confirm presence without automatically moving center or updating identity/persisted width.

**Independent Test**: Tiny, far-offset, and fragmented revisit fixtures record confirmation evidence but are blocked from identity updates when policy gates fail.

### Tests First

- [ ] T024 [P] [US2] Add a tiny revisit test that expects `revisit_authority=confirmation_only` when bandwidth is below `min_revisit_bandwidth_for_identity_update_hz` in `tests/test_fm_characterization_persistence.py`
- [ ] T025 [P] [US2] Add a large center-delta revisit test that prevents stable center movement when `max_revisit_center_delta_for_identity_update_hz` is exceeded in `tests/test_fm_characterization_persistence.py`
- [ ] T026 [P] [US2] Add a fragmented revisit policy test that records confirmation-only evidence for ambiguous revisit fragments in `tests/test_signal_span_policy.py`
- [ ] T027 [P] [US2] Add revisit authority diagnostic assertions for `identity_update_allowed`, `confirmation_recorded`, `revisit_bandwidth_policy_result`, and `revisit_center_policy_result` in `tests/test_fm_characterization_diagnostics.py`

### Implementation

- [ ] T028 [US2] Add a `RevisitAuthorityDecision` representation or helper functions in `sdrwatch/detection/span_policy.py`
- [ ] T029 [US2] Gate revisit center movement, width update, and shrink authority in `BaselinePersistence.apply_revisit_confirmation` in `sdrwatch/baseline/persistence.py`
- [ ] T030 [US2] Allow confirmation-only revisits to clear missing state and record evidence without changing identity center or persisted/card width in `sdrwatch/baseline/persistence.py`
- [ ] T031 [US2] Update `DetectionEngine.apply_revisit_confirmation` to attach revisit authority decisions to emitted characterization evidence in `sdrwatch/detection/engine.py`
- [ ] T032 [US2] Add revisit authority fields to revisit result logging in `sdrwatch/sweep/sweeper.py`
- [ ] T033 [US2] Replace the `profile == "fm_broadcast"` center smoothing branch with `center_smoothing_enabled` or `center_stability_mode` policy in `sdrwatch/baseline/persistence.py`

### Validation

- [ ] T034 [US2] Run `python -m pytest -q tests/test_signal_span_policy.py tests/test_fm_characterization_persistence.py tests/test_fm_characterization_diagnostics.py --basetemp .test-tmp\\span-policy-us2` from repository root `C:\Users\User\SDRwatch`

**Checkpoint**: US2 is independently testable and revisit confirmation no longer implies identity update authority.

---

## Phase 5: User Story 3 - Support Profile-Neutral Width Policies (Priority: P2)

**Goal**: Broad continuous, narrowband, unknown discovery, and guard/event profiles can express different span and revisit policies without FM-specific generic logic.

**Independent Test**: Multiple profile fixtures prove broad profiles may have larger floors while narrowband and discovery profiles remain narrow or unset.

### Tests First

- [ ] T035 [P] [US3] Add profile serialization and CLI profile application tests for new policy fields in `tests/test_fm_validation_profile.py`
- [ ] T036 [P] [US3] Add narrowband profile tests proving small identity, persist, display, and revisit gates remain small in `tests/test_non_fm_width_scope.py`
- [ ] T037 [P] [US3] Add unknown discovery tests proving unset floors do not inherit broad display or persist widths in `tests/test_non_fm_width_scope.py`
- [ ] T038 [P] [US3] Add guard/event policy tests proving fast candidate evidence does not automatically imply stable baseline-card identity in `tests/test_signal_span_policy.py`
- [ ] T039 [P] [US3] Add controller pass-through tests for policy params while preserving `/api/jobs` shape in `tests/test_control_fm_validation.py`

### Implementation

- [ ] T040 [US3] Configure broad continuous policy values as profile data, not generic branches, in `sdrwatch/io/profiles.py`
- [ ] T041 [US3] Ensure narrowband and discovery profiles keep small or unset span-policy values in `sdrwatch/io/profiles.py`
- [ ] T042 [US3] Ensure `python -m sdrwatch.cli --list-profiles` and profile dictionaries expose new policy fields without removing existing fields in `sdrwatch/io/profiles.py`
- [ ] T043 [US3] Pass new policy params through existing controller command construction while preserving legacy `/api/jobs` compatibility in `sdrwatch-control.py`
- [ ] T044 [US3] Preserve web proxy compatibility for `/api/jobs` and existing job status responses in `sdrwatch_web/blueprints/api_jobs.py`

### Validation

- [ ] T045 [US3] Run `python -m pytest -q tests/test_fm_validation_profile.py tests/test_non_fm_width_scope.py tests/test_control_fm_validation.py tests/test_legacy_job_compatibility.py --basetemp .test-tmp\\span-policy-us3` from repository root `C:\Users\User\SDRwatch`

**Checkpoint**: US3 is independently testable and proves this feature is profile-neutral.

---

## Phase 6: User Story 4 - Preserve Close-Signal Separation (Priority: P2)

**Goal**: Width floors stabilize identity and persisted cards without over-merging close but distinct signals.

**Independent Test**: Close-signal fixtures remain separate under active center/cluster policy even when a broad identity or persist floor exists.

### Tests First

- [ ] T046 [P] [US4] Add a close-signal regression proving broad policy floors do not merge separable cards in `tests/test_fm_persistence_stability.py`
- [ ] T047 [P] [US4] Add a raw-cluster test proving raw candidate extents are not widened before close-signal matching in `tests/test_signal_span_policy.py`
- [ ] T048 [P] [US4] Add a width-ratio and max-width cap regression for nearby candidates in `tests/test_cross_sweep_persistence.py`
- [ ] T049 [P] [US4] Add a detector diagnostics regression for two nearby signals split by valley while policy floors are configured in `tests/test_detection_diagnostics.py`

### Implementation

- [ ] T050 [US4] Keep `cluster_merge_hz`, `center_match_hz`, and raw segment overlap decisions policy-free before identity span shaping in `sdrwatch/detection/engine.py`
- [ ] T051 [US4] Apply identity floors only after raw candidate and cluster formation in `sdrwatch/detection/engine.py`
- [ ] T052 [US4] Preserve width-ratio rejection and max width cap behavior while applying persist floors in `sdrwatch/baseline/persistence.py`
- [ ] T053 [US4] Ensure `max_persist_bandwidth_hz` caps persisted/card width without widening live raw cluster extents in `sdrwatch/baseline/persistence.py`

### Validation

- [ ] T054 [US4] Run `python -m pytest -q tests/test_fm_persistence_stability.py tests/test_cross_sweep_persistence.py tests/test_detection_diagnostics.py tests/test_signal_span_policy.py --basetemp .test-tmp\\span-policy-us4` from repository root `C:\Users\User\SDRwatch`

**Checkpoint**: US4 is independently testable and proves width floors are not over-merge rules.

---

## Phase 7: User Story 5 - Audit Effective Policy and Diagnostics End-to-End (Priority: P3)

**Goal**: Effective parameters, diagnostic JSONL, and diagnostic bundles expose the active signal span policy and explain bandwidth/revisit decisions while preserving old fields.

**Independent Test**: A normal single-device RTL diagnostic fixture shows requested/applied profile agreement, null role-run fields, active signal span policy, bandwidth interpretation, and revisit authority decisions.

### Tests First

- [ ] T055 [P] [US5] Add effective-parameter manifest tests for `signal_span_policy` or additive `span_controls` fields in `tests/test_effective_parameter_manifest.py`
- [ ] T056 [P] [US5] Add characterization diagnostic tests for raw, measured, identity, persisted/card, display, and bandwidth interpretation fields in `tests/test_fm_characterization_diagnostics.py`
- [ ] T057 [P] [US5] Add diagnostic bundle summary tests for signal span policy and revisit authority fields in `tests/test_web_diagnostics_bundle.py`
- [ ] T058 [P] [US5] Add legacy single-device diagnostic tests proving role-run fields remain null while policy fields are present in `tests/test_legacy_job_compatibility.py`
- [ ] T059 [P] [US5] Add effective-parameter fallback precedence tests proving scanner-owned profile audit data remains authoritative in `tests/test_effective_parameter_manifest.py`

### Implementation

- [ ] T060 [US5] Extend `build_effective_parameter_manifest` with the derived signal span policy and compatibility aliases in `sdrwatch/util/detection_diagnostics.py`
- [ ] T061 [US5] Preserve old characterization field names while adding raw/identity/persist/display aliases in `sdrwatch/detection/types.py`
- [ ] T062 [US5] Add revisit authority decision fields to characterization and revisit records in `sdrwatch/detection/types.py`
- [ ] T063 [US5] Update diagnostic bundle summarization for signal span policy, bandwidth interpretation, and revisit authority in `sdrwatch_web/diagnostics.py`
- [ ] T064 [US5] Ensure normal single-device RTL scans continue exporting `driver/backend`, `device_key`, requested/applied profile, and null role-run metadata in `sdrwatch/util/detection_diagnostics.py`

### Validation

- [ ] T065 [US5] Run `python -m pytest -q tests/test_effective_parameter_manifest.py tests/test_fm_characterization_diagnostics.py tests/test_web_diagnostics_bundle.py tests/test_legacy_job_compatibility.py --basetemp .test-tmp\\span-policy-us5` from repository root `C:\Users\User\SDRwatch`

**Checkpoint**: US5 is independently testable and diagnostics clearly explain span policy decisions.

---

## Final Phase: Regression, Documentation, and Acceptance Readiness

**Purpose**: Prove the focused update stays inside scope and preserve existing behavior.

- [ ] T066 [P] Update signal span policy diagnostic documentation in `docs/DIAGNOSTIC_CAPTURE.md`
- [ ] T067 [P] Update implementation notes or validation outcomes for the focused span-policy update in `specs/008-multi-rtl-guard-rover/quickstart.md`
- [ ] T068 Run focused no-hardware validation from `specs/008-multi-rtl-guard-rover/quickstart.md` with `python -m pytest -q tests/test_signal_span_policy.py tests/test_extent_hysteresis.py tests/test_fm_characterization_persistence.py tests/test_non_fm_width_scope.py --basetemp .test-tmp\\span-policy-focused`
- [ ] T069 Run profile, controller, and diagnostics contract validation from `specs/008-multi-rtl-guard-rover/quickstart.md` with `python -m pytest -q tests/test_fm_validation_profile.py tests/test_effective_parameter_manifest.py tests/test_control_fm_validation.py tests/test_fm_characterization_diagnostics.py tests/test_web_diagnostics_bundle.py --basetemp .test-tmp\\span-policy-contracts`
- [ ] T070 Run existing regression suites from `specs/008-multi-rtl-guard-rover/quickstart.md` with `python -m pytest -q tests/test_cross_sweep_persistence.py tests/test_device_telemetry.py tests/test_multi_rtl_inventory.py tests/test_multi_rtl_backend_gating.py tests/test_multi_rtl_guard.py tests/test_multi_rtl_telemetry.py tests/test_legacy_job_compatibility.py --basetemp .test-tmp\\span-policy-regression`
- [ ] T071 Run scanner backend smoke `python -m sdrwatch.cli --list-profiles` from repository root `C:\Users\User\SDRwatch` and confirm this remains a backend smoke check, not operator acceptance
- [ ] T072 Follow the no-hardware web/controller validation path in `specs/008-multi-rtl-guard-rover/quickstart.md` and keep `/api/jobs` compatibility visible through `tests/test_control_fm_validation.py` and `tests/test_legacy_job_compatibility.py`
- [ ] T073 On Raspberry Pi 5, run the optional single-device RTL canary through the web GUI/controller path described in `specs/008-multi-rtl-guard-rover/quickstart.md`; if hardware is unavailable, record the unrun hardware acceptance gap in `specs/008-multi-rtl-guard-rover/quickstart.md`
- [ ] T074 Verify no database migration, UI redesign, FM-specific branch, new multi-RTL role-run work, or non-RTL runnable backend was added by reviewing `sdrwatch/baseline/store.py`, `templates/control.html`, `sdrwatch/baseline/persistence.py`, `sdrwatch/detection/engine.py`, `sdrwatch-control.py`, and `sdrwatch/sweep/runner.py`

---

## Dependencies and Execution Order

### Phase Dependencies

- **Setup (Phase 1)**: No dependencies.
- **Foundational Policy Plumbing (Phase 2)**: Depends on Phase 1 and blocks all user stories.
- **US1 Raw Evidence and Policy-Shaped Spans (Phase 3)**: Depends on Phase 2 and is the MVP.
- **US2 Revisit Authority (Phase 4)**: Depends on Phase 2 and can proceed after US1 policy semantics exist.
- **US3 Profile-Neutral Width Policies (Phase 5)**: Depends on Phase 2 and can proceed in parallel with US4 after US1's core policy behavior is stable.
- **US4 Close-Signal Separation (Phase 6)**: Depends on US1 span enforcement and should run before final regression.
- **US5 Effective Policy and Diagnostics (Phase 7)**: Depends on policy fields and story-level decision fields from US1/US2.
- **Final Validation**: Depends on all selected user stories.

### User Story Dependencies

- **US1 (P1)**: First implementation slice and suggested MVP.
- **US2 (P1)**: Can start after foundational policy plumbing; uses the same policy object and persistence paths as US1.
- **US3 (P2)**: Can start after foundational profile/CLI/controller plumbing; should not depend on FM-specific values.
- **US4 (P2)**: Depends on US1 span enforcement because it verifies floors are applied at the right semantic boundary.
- **US5 (P3)**: Depends on US1/US2 diagnostic data but can add manifest tests early.

### Within Each User Story

- Write or update tests first and confirm they fail before implementation.
- Add model/policy helpers before services that consume them.
- Apply detection policy before persistence card enforcement.
- Apply revisit gates before diagnostic summaries that explain them.
- Run each story's validation command before moving to the next story.

---

## Parallel Opportunities

- T003 and T004 can run in parallel because they touch test scaffolding and inspection notes separately.
- T011-T015 can run in parallel because they add US1 tests in separate files.
- T024-T027 can run in parallel because they add independent revisit tests.
- T035-T039 can run in parallel because profile, non-FM, guard/event, and controller pass-through tests are separate.
- T046-T049 can run in parallel because close-signal tests touch separate test modules.
- T055-T059 can run in parallel because diagnostics and effective-parameter tests touch separate files or independent sections.
- T066 and T067 can run in parallel because they update distinct documentation files.

## Parallel Example: User Story 1

```text
Task: "T011 [US1] Add policy-default tests in tests/test_signal_span_policy.py"
Task: "T012 [US1] Add tiny raw detector fragment test in tests/test_fm_characterization_persistence.py"
Task: "T013 [US1] Add persistence EMA floor test in tests/test_extent_hysteresis.py"
Task: "T014 [US1] Add display-span independence test in tests/test_fm_characterization_diagnostics.py"
```

## Parallel Example: User Story 3

```text
Task: "T035 [US3] Add profile serialization and CLI profile application tests in tests/test_fm_validation_profile.py"
Task: "T036 [US3] Add narrowband profile tests in tests/test_non_fm_width_scope.py"
Task: "T039 [US3] Add controller pass-through tests in tests/test_control_fm_validation.py"
```

---

## Implementation Strategy

### MVP First

1. Complete Phase 1 and Phase 2.
2. Complete Phase 3 (US1) to preserve raw evidence and enforce identity/persist/display semantics.
3. Stop and validate US1 independently with the focused no-hardware tests.

### Incremental Delivery

1. Deliver policy plumbing and effective-parameter visibility.
2. Deliver US1 raw/identity/persist/display span separation.
3. Deliver US2 revisit authority gating.
4. Deliver US3 profile-neutral examples and compatibility plumbing.
5. Deliver US4 close-signal separation protections.
6. Deliver US5 diagnostic bundle and audit clarity.
7. Run final no-hardware regression, backend smoke, and optional Pi 5 web/controller canary.

### Scope Guardrails

- Do not hard-code FM Broadcast, 88-108 MHz, or universal 200 kHz behavior in generic code.
- Do not widen raw detector segments or live cluster extents before candidate formation.
- Do not change detector thresholds solely from the FM canary.
- Do not change `/api/jobs` top-level shape.
- Do not add database migrations.
- Do not undo Slice 1 multi-RTL inventory, capability reporting, backend gating, or legacy single-device compatibility.
- Leave unrun browser or Pi 5 hardware acceptance explicitly open in `specs/008-multi-rtl-guard-rover/quickstart.md`.
