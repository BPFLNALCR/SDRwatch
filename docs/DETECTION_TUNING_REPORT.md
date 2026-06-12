# SDRwatch Detection Tuning Report

Created: 2026-06-12

Diagnostic bundle analyzed: `C:/Users/User/Downloads/sdrwatch-diagnostics-05d06f44c8b6/`

## Summary

SDRwatch is detecting RF energy on the RTL-SDR Blog v4. The diagnostic JSONL shows raw candidate segments, final detector-emitted segments, and accepted hits across the scan. The failure is later in the chain: none of the accepted hits are promoted into `baseline_detections`, so the web UI has no signal records to render as cards.

The strongest evidence points to the current persistence defaults being incompatible with the current wide-sweep window lifecycle. The detection engine keeps candidate clusters only within a single sweep, while the GUI/job defaults used here require at least two distinct windows before a cluster can be promoted. With `step=2.4e6` and `samp_rate=2.4e6`, the sweep windows are effectively non-overlapping. Narrow real signals recur across repeated sweeps in the same window, not across two windows in the same sweep, so they never satisfy `persistence_min_windows=2`.

This report does not recommend a detector rewrite in this pass. It does recommend changing GUI defaults/presets so real hardware can produce initial cards through the web workflow.

## Hardware/setup

- Platform under test: Raspberry Pi 5.
- SDR hardware: RTL-SDR Blog v4.
- Operator surface: SDRwatch web GUI.
- Controller job id: `05d06f44c8b6`.
- Baseline id: `3`, name `NEW TEST JUN12`.
- Enabled monitoring zone: `VHF/UHF General`, `118.000-470.000 MHz`.
- Scanner log confirms hardware discovery:
  - `Found Rafael Micro R828D tuner`
  - `RTL-SDR Blog V4 Detected`

## GUI settings used

The exported job params show the exact GUI/controller payload used for this run:

| Setting | Value |
| --- | --- |
| Range | `118000000` to `470000000` Hz |
| Sample rate | `2400000` Hz |
| Step | `2400000` Hz |
| Driver | `rtlsdr_native` |
| FFT | `8192` |
| Average | `8` |
| Threshold | `8` dB |
| Guard bins | `1` |
| Min width bins | `2` |
| CFAR | CLI default `os` |
| CFAR train/guard/quantile | `24` / `4` / `0.75` |
| Gain | CLI default `auto` |
| Persistence mode | `hits` |
| Persistence hit ratio | `0.6` |
| Persistence min hits | `2` |
| Persistence min windows | `2` |
| Persistence min seconds | `10` |
| Max detection width ratio | `3` |
| Diagnostics mode | `true` |
| Two-pass/revisit | `false` |
| Profile | none |

Important note: the current GUI safe defaults in `templates/control.html` still show `fft: '4096'`, but this diagnostic bundle used `fft=8192`. The FFT concern should therefore be separated into two cases: current source default is 4096, but this specific failed real-hardware run failed with 8192.

The generated scanner command did not include `--gain`, `--cfar`, or `--profile`, so the scanner filled `gain="auto"` and `cfar="os"` from CLI defaults.

## Expected behavior

When operated from the web GUI against an enabled monitoring zone, SDRwatch should detect obvious local RF activity and promote stable detections into signal cards. SDR++ on an equivalent RTL-SDR Blog v4 sees visible RF peaks/signals in the same environment, so the RF environment is not empty.

## Actual behavior

The GUI workflow and diagnostic export path are working, but no signal cards appear. The exported database rows confirm that this is not merely a card rendering issue: `baseline_detections` is empty.

The scan did generate backend activity:

- Five completed `scan_updates` rows were exported.
- Each completed sweep recorded hundreds of segments and hits.
- `num_new_signals` was `0` for every exported sweep.
- `baseline_detections.json` is `[]`.
- `friendly-signals.json` is `[]`.

## Evidence from diagnostic JSONL

The exported JSONL tail contains 809 detection window records: five complete sweeps of 147 windows each, plus part of a sixth sweep.

Aggregate JSONL evidence:

| Metric | Value |
| --- | ---: |
| Window records | 809 |
| Raw candidate segments | 1377 |
| Final detector-emitted segments | 1377 |
| Accepted hits | 1368 |
| Promoted detections | 0 |
| New signals | 0 |
| Spur ignored | 0 |
| Windows with raw segments | 366 |
| Windows with accepted hits | 357 |
| Anomalous-power windows | 16 |

The detector is not dropping candidate segments before emission: `raw_candidate_segment_count` equals `final_emitted_segment_count` for all records in this bundle. The only accepted-hit mismatches are anomalous-power windows around 211.6, 214.0, and 216.4 MHz, where the power monitor intentionally withholds ingestion.

Strong repeated examples exist in the JSONL:

| Approx. frequency | Evidence |
| --- | --- |
| 125.000 MHz | Repeated strong segments, SNR around 42-43 dB |
| 349.998 MHz | Repeated strong segments, SNR around 54-58 dB |
| 449.997 MHz | Repeated strong segments, SNR around 44-46 dB |

Across the exported tail, many candidate centers repeat across multiple sweeps. Grouping emitted segment centers to the nearest 1 kHz found 289 repeated groups across at least two sweeps. None of those repeated groups appeared in more than one distinct sweep window index. This is the key shape of the failure: repeated real-looking hits recur across loop sweeps, but the current promotion gate only sees per-sweep clusters.

Segment widths are mostly above the configured width floor:

- Bin width with this run: `2400000 / 8192 = 292.96875 Hz`.
- Promotion width floor from `min_width_bins=2`: `585.9375 Hz`.
- 1362 of 1377 emitted segments were at or above that width floor.
- Median emitted bandwidth was about `878.9 Hz`.

So width filtering may reject a small number of tiny/zero-width estimates, but it is not the primary reason that all cards are missing.

Auto gain evidence is weaker. The JSONL reports `gain: "auto"` in every tuning record, but it does not include actual tuner gain telemetry. Same-center local noise medians were fairly stable across repeated sweeps: median same-center range was about `0.52 dB`, only one center exceeded `2 dB`, and the maximum same-center range was about `3.33 dB`. This bundle does not prove auto gain is destabilizing the noise floor, but auto gain remains a poor default for baseline comparison because the actual gain state is not fixed or recorded.

## Evidence from scanner logs

The scanner log tail shows normal scan execution with no fatal errors:

- Baseline bin repair: `repaired baseline bin_hz to 293.0 Hz based on current sweep`.
- Baseline use: `using baseline id=3 name='NEW TEST JUN12' span=118.000-470.000 MHz bin=293.0 Hz`.
- Begin sweep line: `step=2.400 samp_rate=2.400 fft=8192 avg=8`.
- Completed sweep summaries:

| Sweep | Hits | Promoted | New |
| --- | ---: | ---: | ---: |
| 1 | 278 | 0 | 0 |
| 2 | 197 | 0 | 0 |
| 3 | 255 | 0 | 0 |
| 4 | 258 | 0 | 0 |
| 5 | 217 | 0 | 0 |

The per-window log lines repeatedly show nonzero `det_count` and `accepted`, with `promoted=0` and `new_sig=0`.

The exported bundle does not include the structured `sdrwatch-scan.log` that would contain `cluster_reject` debug records from the detection engine. That limits direct proof of each individual rejection reason, but the code path and JSONL shape are enough to locate the failing stage.

## Evidence from database rows

Exported database snapshot:

- `baseline.json`: baseline id `3`, span `118000000-470000000`, `bin_hz=292.96875`, `total_windows=793`, `total_observed_ms=21411`.
- `scan-updates.json`: five recent sweep rows with `num_segments` from 198 to 280 and `num_hits` from 197 to 278.
- `scan-updates.json`: all exported rows have `num_new_signals=0`, `num_revisits=0`, `num_confirmed=0`, and `num_false_positive=0`.
- `baseline-detections.json`: empty array.
- `friendly-signals.json`: empty array.
- `monitoring-zones.json`: enabled VHF/UHF General zone covering the whole scan range.

The web views and signal APIs read signal cards from `baseline_detections`. Since that table is empty for this baseline, the lack of signal cards is consistent with the database state and is not primarily a UI surfacing bug.

## Where the detection chain appears to fail

The chain status for this bundle:

| Stage | Status |
| --- | --- |
| GUI job creation | Works |
| Controller scanner command generation | Works |
| Hardware discovery | Works |
| Scanner sweep execution | Works |
| PSD/CFAR raw candidate generation | Works |
| Detector segment emission | Works |
| Hit acceptance by detection engine | Works, except anomalous-power windows |
| Promotion into persistent detections | Fails for all candidates |
| Database insertion into `baseline_detections` | Does not occur |
| Web signal cards | Empty because there are no persisted detections |

Most likely root cause: promotion defaults require multi-window persistence inside a single sweep, but the scan schedule and cluster lifetime only present most real narrowband signals as one window per sweep.

Relevant current behavior:

- `WindowScheduler` emits windows spaced by `step_hz`.
- The GUI job used `step_hz=2.4 MHz`, equal to the sample rate, so windows are effectively adjacent/non-overlapping.
- `DetectionEngine` is created inside each `Sweeper.run`, so candidate clusters do not persist across loop sweeps.
- `DetectionEngine._cluster_gate_status()` requires `hits >= persistence_min_hits` and `distinct windows >= persistence_min_windows`.
- The GUI/job used `persistence_min_hits=2` and `persistence_min_windows=2`.

A narrow real signal detected once in one window during each sweep can be seen repeatedly forever but still never become a card, because each sweep starts a fresh detection engine and each per-sweep cluster only has one window.

## Recommended default setting changes

These are GUI/controller default recommendations, not CLI-only operator instructions.

1. Change the default RTL-SDR Blog v4/operator preset away from auto gain. Use fixed manual gain as the normal starting point for baseline/detection work. A practical first preset is around `30 dB`, with the GUI making overload symptoms and supported gain rounding explicit.

2. Fix the promotion/window mismatch in GUI defaults. Under the current engine, either:
   - use overlapping windows by setting `step` below `samp_rate`, for example `1.2e6` with `samp_rate=2.4e6`, so a stable signal can appear in more than one adjacent window, or
   - lower discovery-mode promotion gates to `persistence_min_hits=1` and `persistence_min_windows=1`.

3. For a first-light/discovery default, prefer a configuration that intentionally produces cards, then let classification/friendly filtering clean up the UI. A no-card default is worse for real-hardware onboarding than a somewhat noisy first pass.

4. Keep OS-CFAR as the default for now. In this bundle, CFAR was not too aggressive: it emitted many segments and accepted many hits. The issue is promotion, not candidate starvation.

5. Treat FFT/averaging as quality tuning rather than the primary root cause. This failed run used `fft=8192`, `avg=8`, and still produced strong candidates. For real RTL-SDR Blog v4 defaults, consider `avg=16` if sweep speed remains acceptable. Use `fft=4096` or `8192` based on desired scan speed and frequency detail, but do not expect FFT alone to fix no-card behavior.

6. Consider raising `threshold_db` only after promotion is fixed. A threshold around `10-12 dB` may reduce noisy cards, but increasing it now would hide the real promotion problem.

## Recommended GUI tuning presets

Add these as operator-facing web GUI presets, not as CLI recipes.

### RTL-SDR v4 Discovery

Purpose: make visible real-hardware cards appear quickly enough to prove end-to-end operation.

- Gain mode: manual.
- Gain: `30 dB` starting point.
- Sample rate: `2.4e6`.
- Step: `2.4e6` for speed, or `1.2e6` if keeping multi-window persistence.
- FFT: `4096` or `8192`.
- Averaging: `8-16`.
- Threshold: `8-10 dB`.
- CFAR: OS, train `24`, guard `4`, quantile `0.75`.
- Persistence: `min_hits=1`, `min_windows=1` if step remains `2.4e6`; otherwise `min_hits=2`, `min_windows=2` can be tried with `step=1.2e6`.
- Two-pass: optional/off until promotion is confirmed.

### RTL-SDR v4 Stable Baseline

Purpose: slower, cleaner scans after first-light confidence is established.

- Gain mode: manual.
- Gain: start `25-30 dB`, adjust down if overloaded.
- Sample rate: `2.4e6`.
- Step: `1.2e6` for overlap.
- FFT: `8192`.
- Averaging: `16`.
- Threshold: `10-12 dB`.
- CFAR: OS, train `24-32`, guard `4-6`, quantile `0.75-0.8`.
- Persistence: `min_hits=2`, `min_windows=2`, hit ratio around `0.5-0.6`.
- Two-pass: enable only after confirming that promotion produces reasonable coarse detections.

### Fast Wide Survey

Purpose: broad situational scan where missing fewer obvious peaks matters more than precise card quality.

- Gain mode: manual.
- Gain: `30 dB` starting point.
- Sample rate: `2.4e6`.
- Step: `2.4e6`.
- FFT: `4096`.
- Averaging: `8`.
- Threshold: `10-12 dB`.
- Persistence: `min_hits=1`, `min_windows=1`.

## Implementation note

The GUI preset implementation selected exact values from the ranges above:

| Preset | Gain | Sample rate | Step | FFT | Avg | Persistence |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| RTL-SDR v4 Discovery | manual `30 dB` | `2400000` | `2400000` | `8192` | `8` | `1/1` |
| Stable Baseline | manual `30 dB` | `2400000` | `1200000` | `8192` | `16` | `2/2` |
| Fast Wide Survey | manual `30 dB` | `2400000` | `2400000` | `4096` | `8` | `1/1` |

This preserves the diagnosis that FFT is not the primary fix for zero cards. Discovery uses relaxed promotion gates with the same fast non-overlapping step so the GUI can produce first-light cards; Stable Baseline uses overlapping windows and stricter persistence for slower baseline work.

## Risks/tradeoffs

- Lowering persistence gates to `1/1` will create more cards, including spurs and short-lived peaks. That is acceptable for a discovery preset but may be too noisy as the long-term baseline default.
- Overlapping sweep windows (`step=1.2e6`) can preserve the existing `2/2` promotion idea, but it roughly doubles the number of windows across a wide band and slows scan cadence.
- Fixed gain improves baseline comparability, but a hard-coded `30 dB` default can overload in strong-signal environments. The GUI should make it easy to step down to `20-25 dB`.
- Raising thresholds too early could mask weak but real signals and make the current no-card problem look better than it is.
- Current diagnostics do not record actual tuner gain, gain changes, or structured cluster reject reasons in the exported bundle. That limits certainty around auto-gain effects and individual promotion failures.

## Follow-up implementation plan

1. Add GUI presets for RTL-SDR v4 Discovery, RTL-SDR v4 Stable Baseline, and Fast Wide Survey. Keep the operator workflow in the web GUI and controller job lifecycle.

2. Change the default/safe GUI copy to stop describing auto gain as safest for normal detection scans. Prefer fixed gain for baseline work, with inline overload guidance.

3. For immediate real-hardware usability, choose one default path:
   - `step=2.4e6`, `persistence_min_hits=1`, `persistence_min_windows=1`, or
   - `step=1.2e6`, `persistence_min_hits=2`, `persistence_min_windows=2`.

4. Extend diagnostic bundle contents to include structured scan logger JSONL or a compact promotion summary with `cluster_emit` and `cluster_reject` counts/reasons. This would make future tuning reports less inferential.

5. Add scanner/controller telemetry for actual RTL-SDR gain mode/value when available, so auto-gain behavior can be measured rather than inferred.

6. After default/preset changes, validate through the web GUI on Raspberry Pi 5 with RTL-SDR Blog v4:
   - start a diagnostics-mode scan from the control page,
   - export a diagnostic bundle,
   - confirm nonzero `promoted`,
   - confirm nonempty `baseline_detections`,
   - confirm signal cards appear in the web UI.

7. Only after the GUI-default fix is verified, consider deeper detection algorithm changes such as cross-sweep pending-cluster persistence or a promotion gate that distinguishes sweep windows from repeated loop sweeps.
