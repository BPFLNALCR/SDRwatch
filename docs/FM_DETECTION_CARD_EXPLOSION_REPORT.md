# FM Detection Card Explosion Report

Created: 2026-06-12

Diagnostic bundle analyzed: `C:/Users/User/Downloads/sdrwatch-diagnostics-532516344b4b/`

## Summary

The FM-band diagnostic bundle shows that the GUI-started scan now produces persisted signal cards, but the FM-band behavior is unstable and over-productive. In a short 88-108 MHz run, SDRwatch persisted 234 `baseline_detections`, of which 209 were already marked missing at export time. Most persisted detections are only one to three FFT bins wide, so the UI would show many tiny cards instead of a smaller set of stable FM-band signals.

This is not candidate starvation. The scan accepted hits, promoted detections, and inserted rows. The likely failure is later and more specific: FM-band segmentation emits narrow spike fragments; the Discovery `1/1` promotion gate promotes those fragments immediately; persistence matching/upsert does not group enough nearby FM fragments into stable station-scale records; `max_detection_width_hz` is effectively disabled; and two-pass/revisit behavior was not enabled for this GUI run.

## Scan Settings

| Setting | Value |
| --- | ---: |
| Scan range | `88000000` to `108000000` Hz |
| Device | RTL-SDR Blog v4 |
| Driver | `rtlsdr_native` |
| Sample rate | `2400000` Hz |
| Step | `2400000` Hz |
| FFT | `8192` |
| Bin width | `292.96875` Hz |
| Average | `8` |
| Threshold | `8` dB |
| Gain | `30` |
| Guard bins | `1` |
| Min width bins | `2` |
| CFAR train/guard/quantile | `24` / `4` / `0.75` |
| Persistence hit ratio | `0.6` |
| Persistence min hits | `1` |
| Persistence min windows | `1` |
| Persistence min seconds | `10` |
| Max detection width ratio | `3` |
| Max detection width Hz | Scanner default `0.0` because no `--max-detection-width-hz` flag was present |
| Two-pass | Disabled |
| Profile | None |
| Diagnostics mode | Enabled |

The generated scanner command was:

```text
/opt/sdrwatch/.venv/bin/python3 -m sdrwatch.cli --baseline-id 1 --driver rtlsdr_native --start 88000000 --stop 108000000 --step 2400000 --samp-rate 2400000 --fft 8192 --avg 8 --threshold-db 8 --guard-bins 1 --min-width-bins 2 --sleep-between-sweeps 0 --persistence-hit-ratio 0.6 --persistence-min-seconds 10 --persistence-min-hits 1 --persistence-min-windows 1 --max-detection-width-ratio 3 --cfar-train 24 --cfar-guard 4 --cfar-quantile 0.75 --db /var/lib/sdrwatch/sdrwatch.db --gain 30 --diagnostic-jsonl /tmp/sdrwatch-control/diagnostics/532516344b4b.diagnostic.jsonl --loop
```

## Card Counts

| Evidence | Count |
| --- | ---: |
| Persisted `baseline_detections` rows | 234 |
| Active detections at export | 25 |
| Missing detections at export | 209 |
| Recent `scan_updates` rows exported | 62 |
| Diagnostic window records exported | 558 |
| Diagnostic emitted segments | 1408 |
| Diagnostic accepted hits | 1331 |
| Diagnostic promoted detections | 828 |
| Diagnostic new signals | 249 |
| Revisit attempts | 0 |
| Revisit confirmations | 0 |
| Revisit false positives | 0 |

Service labeling confirms this is overwhelmingly FM-band activity: 230 of 234 persisted rows are labeled `FM Broadcast`; 4 have no service label.

## Bandwidth Distribution

Persisted `baseline_detections` widths are dominated by sub-kHz spans:

| Metric | Width |
| --- | ---: |
| Min | 293 Hz |
| P10 | 586 Hz |
| P25 | 586 Hz |
| Median | 586 Hz |
| P75 | 880 Hz |
| P90 | 11424 Hz |
| P95 | 11720 Hz |
| Max | 22852 Hz |

Width buckets:

| Bucket | Count |
| --- | ---: |
| `<1 kHz` | 193 |
| `1-5 kHz` | 17 |
| `5-20 kHz` | 18 |
| `20-80 kHz` | 6 |
| `80-200 kHz` | 0 |
| `>=200 kHz` | 0 |

The most common persisted widths were 586 Hz (129 detections), 878 Hz (36 detections), and 880 Hz (26 detections). The diagnostic emitted segments were also narrow: 550 of 1408 emitted segments were under 1 kHz, 1406 of 1408 were under 5 kHz, and none reached 80 kHz.

## Active vs Missing Detections

| State | Count | Median width | Under 1 kHz | Under 5 kHz |
| --- | ---: | ---: | ---: | ---: |
| Active | 25 | 586 Hz | 22 | 25 |
| Missing | 209 | 586 Hz | 171 | 185 |

Missing detections were marked missing quickly after their last seen time:

| Metric | Seconds from `last_seen_utc` to `missing_since_utc` |
| --- | ---: |
| Min | 2.20 |
| Median | 3.12 |
| P90 | 3.80 |
| Max | 4.11 |

All 209 missing detections were marked missing within 10 seconds of last being seen. This supports the instability diagnosis: rows are being created from narrow fragments and then immediately marked absent on following sweeps.

## Two-Pass Evidence

Two-pass was disabled for this bundle:

- `job/scanner-command.txt` does not contain `--two-pass`.
- `job/scanner-command.txt` does not contain `--profile`.
- `job/scanner-command.txt` does not contain revisit flags such as `--revisit-span-limit-hz`.
- `job/params.json` has no `two_pass` or `profile` field.
- All 558 diagnostic window records reported `tuning_params.two_pass=false`.
- All 62 exported `scan_updates` rows have `num_revisits=0`, `num_confirmed=0`, and `num_false_positive=0`.

## Likely Failure Stage

The failure is after raw candidate generation and after promotion becomes possible:

| Stage | Status |
| --- | --- |
| GUI job creation | Works |
| Controller command generation | Works |
| Hardware scan execution | Works |
| Candidate segmentation | Works, but emits many narrow FM spike fragments |
| Promotion | Works, but Discovery `1/1` promotes fragments immediately |
| Baseline upsert/matching | Likely too fine-grained for FM fragments without FM match/display shaping |
| Width clamp behavior | Not active because `max_detection_width_hz` was absent/effective `0.0` |
| Missing marking | Works, but marks many freshly created tiny rows missing within seconds |
| Two-pass/revisit | Disabled, so no confirmation or pruning path ran |
| Diagnostics | Good for window evidence, but too opaque for create/update/merge/missing decisions |

The highest-probability failure is a combination of segmentation and baseline upsert/matching, amplified by the Discovery preset. FM broadcast stations can present narrow peaks, pilots, edges, or spiky lobes in a coarse sweep. The current GUI Discovery settings allow each tiny fragment to become its own persistent row, while the run does not use the existing `fm_broadcast` profile's FM-scale match/display span shaping, width cap, centroid mode, or two-pass defaults.

## Recommended Implementation Direction

1. Preserve RTL-SDR v4 Discovery as a first-light preset that can produce cards, but do not use it as the FM-band validation path.
2. Add or improve a GUI-visible FM Validation preset for 88-108 MHz that submits existing `/api/jobs` params through the controller lifecycle.
3. Prefer reusing the existing `fm_broadcast` profile behavior for FM Validation: overlapping step, FM-specific CFAR/min-width values, two-pass enabled, segment centroiding, tight match spans, wider display spans, center matching, and width caps.
4. If profile-only wiring is too implicit, add narrow controller/CLI passthrough for the existing FM shaping fields rather than rewriting detection.
5. Add tests before implementation for FM-like spiky signals, separated FM-like signals, non-FM narrow signals, repeated nearby upserts, bounded width clamps, two-pass command/diagnostic behavior, and GUI `/api/jobs` payloads.
6. Extend diagnostics so bundles expose create/update/no-match/match/merge/width-clamp/revisit decisions directly instead of requiring inference from final rows.

Do not solve this by simply raising thresholds until cards disappear. The evidence shows actual overproduction and unstable persistence, not a lack of accepted RF candidates.

## Post-Implementation Update

Updated: 2026-06-12

The implementation follows the recommended profile-driven direction:

- Added a GUI-visible `FM Validation` preset while preserving `RTL-SDR v4 Discovery` as the selected first-light default.
- FM Validation submits existing `/api/jobs` params, including `profile=fm_broadcast`, `two_pass=true`, bounded revisit settings, `cluster_merge_hz=12000`, `max_detection_width_ratio=2.5`, and `max_detection_width_hz=270000`.
- Preset application clears preset-controlled fields before applying the selected preset so FM profile/revisit settings do not leak back into Discovery.
- Sweeper diagnostic `tuning_params` now includes FM profile-applied effective settings such as center matching, match/display bandwidth floors, and revisit values.
- Structured `persistence_decision` and `width_decision` events now record insert/update/no-match/missing/missing-clear and width floor/cap behavior.
- Diagnostic bundles now include `diagnostics/decision-summary.json` and record missing or tail-limited decision evidence in `manifest.json`.

Automated validation completed with the bundled Python runtime:

- Focused FM/card-stability suite: `44 passed in 1.53s`.
- Focused no-hardware regression suite: `55 passed in 1.53s`.
- Backend profile smoke: `python -m sdrwatch.cli --list-profiles` succeeded and reported the expected `fm_broadcast` stability fields.

Hardware GUI/controller validation on Raspberry Pi 5 with RTL-SDR Blog v4 was not run in this environment. Field acceptance still needs a fresh GUI FM Validation diagnostic bundle confirming that persisted FM cards are no longer dominated by 293-880 Hz widths and that the exported decision summary explains create/update/missing/width/revisit behavior.
