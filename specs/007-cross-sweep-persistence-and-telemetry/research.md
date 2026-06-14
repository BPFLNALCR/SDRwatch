# Phase 0 Research: Cross-Sweep Persistence and Telemetry

## Decision: Count persistence across complete sweep loops, not only adjacent windows

**Rationale**: A signal can be real and stable even when it appears in only one window per full sweep loop. With non-overlapping windows, requiring multiple same-sweep windows can miss such signals. The scanner already has a complete-loop counter (`sweep_seq`) in the runner/sweeper path, so cross-sweep observations can be tracked without redefining one sweep window as repeated evidence.

**Alternatives considered**:

- Lower FM defaults to one hit in one window: rejected because it weakens the control band and recreates fragment promotion risk.
- Force overlapping windows everywhere: rejected because it increases scan cost and does not solve the semantic problem of repeated full-loop observations.
- Treat several observations in one sweep loop as equivalent to several sweep loops: rejected because it can promote transient same-loop artifacts too early.

## Decision: Reuse bounded profile matching rules for cross-sweep compatibility

**Rationale**: The current FM work already tuned center match, match bandwidth floors, display bandwidth floors, width caps, and stable center behavior to avoid merging nearby stations. Cross-sweep state should use the same bounded center/span/width concepts rather than inventing broad frequency buckets.

**Alternatives considered**:

- Frequency-bin-only matching: rejected because it ignores width compatibility and can merge nearby signals.
- Broad display-span matching: rejected because display spans are intentionally wider and human-facing.
- Raw segment overlap only: rejected because FM raw fragments can be narrow and drift between sweeps.

## Decision: Keep cross-sweep candidate state additive and bounded

**Rationale**: The feature needs memory across sweep loops, but a schema migration is not required to prove the behavior. A bounded in-memory candidate table keyed by compatible center/span evidence can promote within a running job and can later become an additive sidecar if long-running restart persistence is required.

**Alternatives considered**:

- Add new database tables immediately: rejected because current acceptance can be proven within a running scan and diagnostics-first evidence is lower risk.
- Store cross-sweep state in existing `baseline_detections` rows before promotion: rejected because it would blur unpromoted candidates with persistent cards.
- Keep state only in per-window clusters: rejected because it is the source of the current weakness.

## Decision: Emit an effective-parameter manifest as structured diagnostic evidence

**Rationale**: Operators and developers need to reconstruct what a job actually used: requested profile, profile application status, skipped reason, overrides, final scanner args, FFT/bin width, persistence thresholds, revisit settings, width settings, gain settings, and device metadata. These facts should be explicit JSON, not inferred from scanner logs or command strings.

**Alternatives considered**:

- Use only `job/scanner-command.txt`: rejected because command strings do not explain skipped profiles, profile defaults, or unavailable telemetry.
- Use only per-window `tuning_params`: rejected because the job-level request/application/final distinction belongs in a manifest.
- Add a UI-only summary: rejected because developers need machine-readable bundles for replay and tuning.

## Decision: Make profile application status explicit

**Rationale**: The scanner can skip profile defaults when the requested range is outside the profile band. The diagnostics bundle must show the difference between requested profile, applied profile, skipped profile, skip reason, and fallback values.

**Alternatives considered**:

- Treat `profile=fm_broadcast` in the request as proof of application: rejected because the scanner can skip it.
- Hide skipped profiles as warnings in logs: rejected because tests and diagnostics should not scrape log text.
- Fail out-of-band profile requests: rejected because existing behavior can continue safely if the fallback is explicit.

## Decision: Expand controller passthrough to match scanner-supported characterization flags

**Rationale**: `sdrwatch/cli.py` already supports profile-hidden characterization fields such as centroid mode and match/display bandwidth shaping. The web/controller path should be equivalent to scanner smoke checks for supported tuning parameters, with unsupported names documented instead of silently dropped.

**Alternatives considered**:

- Rely only on `profile=fm_broadcast`: rejected because explicit overrides and parity tests require direct passthrough.
- Use `extra_args` for operator-facing controls: rejected because it bypasses the stable `/api/jobs` params contract and is not ergonomic for the web UI.
- Add a separate controller endpoint: rejected because the existing job lifecycle already owns scan starts.

## Decision: Aggregate diagnostics from structured events

**Rationale**: Bundles already include structured events like `segment_inventory`, `cluster_emit`, `cluster_reject`, `persistence_decision`, `width_decision`, `revisit_*`, and `characterization_record`. The next step is to make aggregate counts durable in summaries so failure analysis does not depend on human-readable log text or unbounded JSONL tails.

**Alternatives considered**:

- Parse scanner log lines: rejected because log text is unstable and hard to test.
- Export full unbounded JSONL: rejected because bundles must stay bounded.
- Keep only final database rows: rejected because it hides why candidates were rejected, unmatched, or revisited.

## Decision: Record device and gain telemetry best-effort

**Rationale**: Requested gain, actual applied gain, gain mode, supported gains, device identity, sample rate, FFT size, and bin width materially affect RF observations. Driver support may be partial, so unavailable values should be represented as null or unavailable rather than failing scans.

**Alternatives considered**:

- Require actual gain and device serial from every driver: rejected because current driver support may vary.
- Record only requested gain: rejected because requested and actual receiver state can differ.
- Fail scans when telemetry is unavailable: rejected because telemetry should improve auditability without reducing field reliability.

## Decision: Keep FM Broadcast as the control profile and preserve characterization boundaries

**Rationale**: FM Broadcast remains the best stable control band for regression testing. The 006 baseline already separates raw detector span, measured characterization, match span, display span, persisted span, and contextual metadata. Cross-sweep persistence must not collapse those meanings or use bandplan/profile context as proof of FM.

**Alternatives considered**:

- Use FM context to assign confident classification candidates: rejected because context is not measured proof.
- Use displayed card width as measured occupied bandwidth: rejected because display width is deliberately wider for operator stability.
- Merge adjacent FM-like signals to simplify persistence: rejected because nearby-station separation is a core invariant.

## Resolved Unknowns

- **Cross-sweep state durability**: First implementation can be running-job scoped and diagnostics-visible. Add persistent sidecar storage only if implementation evidence shows restart durability is required for acceptance.
- **Profile skipped semantics**: A skipped profile is not failure by default; it is a manifest state with `requested_profile`, `applied_profile=null`, `profile_applied=false`, `profile_skip_reason`, fallback defaults, and final effective values.
- **Telemetry availability**: All driver-specific telemetry fields are nullable or carry an unavailable reason.
