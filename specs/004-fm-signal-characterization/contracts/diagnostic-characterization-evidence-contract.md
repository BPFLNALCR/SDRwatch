# Contract: Diagnostic Characterization Evidence

This contract defines the bounded evidence needed to validate FM signal characterization without collapsing measurement concepts or requiring unbounded exports.

## Existing Diagnostic Bundle Evidence To Preserve

Bundles must continue to include bounded versions of:

- `job/job.json`
- `job/params.json`
- `job/scanner-command.txt`
- `logs/scanner-log-tail.txt`
- `diagnostics/diagnostic-jsonl-tail.jsonl`
- `database/baseline.json`
- `database/baseline-detections.json`
- `database/scan-updates.json`
- `database/monitoring-zones.json`
- `database/friendly-signals.json`
- `manifest.json`
- `NOTES.md`

Missing or truncated evidence must continue to be reported in `manifest.json`.

## Required Characterization Separation

For every characterized signal included in an export sample, the bundle must make these concepts separately identifiable:

- raw detector segment span
- measured center estimate
- measured occupied bandwidth estimate
- persistence or match span
- display or card span
- contextual bandplan or profile metadata
- characterization confidence and evidence sources
- revisit provenance when revisit contributed to the result

## Suggested Summary Shape

The exact file names may follow existing bundle conventions, but the exported evidence should support a record shaped like this:

```json
{
  "signal_id": 123,
  "raw_segment": {
    "low_hz": 100099121,
    "high_hz": 100100879,
    "bandwidth_hz": 1758
  },
  "measured_characterization": {
    "center_hz": 100100000,
    "occupied_bandwidth_hz": 182000,
    "bandwidth_confidence": 0.72,
    "center_stability_hz": 1800,
    "bandwidth_stability_hz": 6400,
    "characterization_method": "coarse+revisit"
  },
  "match_span": {
    "low_hz": 100060000,
    "high_hz": 100140000,
    "bandwidth_hz": 80000
  },
  "display_span": {
    "low_hz": 100000000,
    "high_hz": 100200000,
    "bandwidth_hz": 200000
  },
  "context": {
    "bandplan_service": "FM Broadcast",
    "profile_context": "fm_broadcast"
  },
  "classification": {
    "candidate": "fm_broadcast_candidate",
    "evidence": ["stable_center", "revisit_refinement"],
    "context_only": false
  }
}
```

## Confidence And Evidence Rules

- Confidence must be accompanied by evidence sources or method labels.
- Contextual labels alone must not be sufficient to justify a candidate classification.
- Optional FM-specific evidence such as pilot or RDS indicators should appear as supporting evidence when present.

## Bounded Export Rules

- High-volume evidence may be summarized rather than fully expanded.
- Summary counts are preferred over unbounded raw event dumps.
- Representative decision records are acceptable when full logs would exceed bundle limits.
- `manifest.json` must record any missing or truncated characterization evidence.

## Acceptance Checks

- A maintainer can tell raw width from measured occupied bandwidth.
- A maintainer can tell measured occupied bandwidth from display width.
- A maintainer can tell match span from display span.
- Contextual bandplan or profile labels remain visually and structurally separate from measured classification evidence.
- Revisit contribution is visible when revisit actually influenced the result.
- Export bounds and missing-evidence reporting remain intact.
