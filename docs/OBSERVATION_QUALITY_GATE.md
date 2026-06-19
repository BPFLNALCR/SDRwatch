# Observation Quality Gate

This gate defines profile-neutral scan quality checks for SDRwatch regression
validation. It is not an FM broadcast tuning target; FM broadcast is useful as a
live canary because it is repeatable in many environments, but generic scanner
behavior must be judged with synthetic fixtures, replay captures, and
profile-specific anchors.

## Metrics

- Known-anchor detection rate: percentage of configured anchors, watchlist
  entries, or friendly channels detected inside the profile-specific tolerance.
- Center error: median and p95 absolute center offset from anchor or replay
  truth.
- Duplicate/split rate: number of extra cards within the configured merge radius
  around one anchor, plus orphan near-neighbor cards per scanned span.
- Width stability: measured, match, display, and persisted widths tracked
  separately, with p95 change and invalid-span counts.
- Persistence sanity: insert, match, missing, clear, and promotion ratios,
  including false one-off cards that do not recur.
- False one-off card rate: cards that appear once with weak support and then
  immediately disappear or age out.
- Timing sanity: p50 and p95 tune, flush, read, FFT, detect, database update,
  JSONL, and total window timing.
- Read sanity: samples requested versus samples read, short-read count, and
  dropped-read unavailable-or-below-threshold status.
- Legacy `/api/jobs` compatibility: existing payload shape, native RTL command
  construction, no role metadata for normal jobs, and pre-spawn rejection of
  unsupported devices.
- Effective-parameter audit consistency: scanner log, decision summary,
  `job/effective-parameters.json`, and bundle manifest agree on requested and
  applied profile state or explicitly mark the audit partial.

## Test Inputs

- Deterministic synthetic detector and persistence fixtures with known centers,
  widths, recurrence patterns, and noise floors.
- Diagnostic or PSD replay captures from real runs, including at least one broad
  profile and at least one narrow/watchlist/friendly-channel profile.
- Live FM canary scans as controlled RF smoke, without using exact station count
  or FM bandwidth aesthetics as the architecture target.
- Fake-device controller tests covering inventory, backend gating, command
  construction, diagnostics mode, and legacy `/api/jobs` compatibility.
- Future narrowband, watchlist, and friendly-channel fixtures that exercise the
  same metrics outside the FM broadcast band.

## Pass/Fail Expectations

No-hardware regression checks should be deterministic:

- Synthetic fixtures detect expected anchors and reject known false one-offs.
- `persistence_min_sweep_loops=1` preserves immediate legacy promotion.
- Cross-sweep candidate behavior appears only when
  `persistence_min_sweep_loops > 1`.
- Diagnostic bundle effective parameters use scanner-owned events first,
  decision-summary effective settings second, and partial controller fallback
  only when scanner-owned evidence is unavailable.
- Legacy `/api/jobs` command construction remains native RTL, includes
  `--device-key rtl:0`, and omits role-run flags for normal single-device jobs.

Hardware acceptance should be trend and sanity based:

- Known anchors in the local RF environment are detected within configured
  tolerances.
- Duplicate/split and one-off rates stay within profile-specific bounds.
- Timing and read metrics remain stable enough for the target platform.
- Missing platform telemetry is exported as `null` plus `unavailable_fields`
  rather than treated as a scan failure.

## Reporting

Each validation report should identify:

- Profile or range under test.
- Test source: synthetic, replay, live FM canary, live narrowband/watchlist, or
  fake-device controller.
- Metrics that passed and failed.
- Whether failures are scanner behavior, persistence behavior, controller/API
  compatibility, or diagnostic audit/export integrity.
