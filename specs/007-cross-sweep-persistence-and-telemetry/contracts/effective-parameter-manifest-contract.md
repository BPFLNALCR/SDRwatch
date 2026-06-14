# Contract: Effective-Parameter Manifest

Diagnostic bundles must include a structured manifest that explains what a job requested, what profile behavior was applied or skipped, what the operator overrode, and what scanner parameters were ultimately effective.

## Bundle Location

The exact archive path may follow existing bundle conventions, but the bundle must include either:

- an expanded `manifest.json` section named `effective_parameters`, or
- a dedicated JSON file such as `job/effective-parameters.json` referenced from `manifest.json`

## Required Shape

```json
{
  "job_id": "abc123def456",
  "requested_profile": "fm_broadcast",
  "applied_profile": "fm_broadcast",
  "profile_applied": true,
  "profile_skip_reason": null,
  "operator_overrides": {
    "gain": "30"
  },
  "profile_defaults": {
    "step_hz": 1200000,
    "samp_rate": 2400000,
    "fft": 8192
  },
  "fallback_defaults": {},
  "final_effective_params": {
    "start_hz": 88000000,
    "stop_hz": 108000000,
    "step_hz": 1200000,
    "sample_rate_hz": 2400000,
    "fft": 8192,
    "bin_width_hz": 292.96875
  },
  "persistence": {
    "mode": "hits",
    "min_hits": 2,
    "min_windows": 2,
    "min_sweep_loops": 2,
    "hit_ratio": 0.25
  },
  "revisit": {
    "two_pass": true,
    "fft": 32768,
    "avg": 4,
    "margin_hz": 200000,
    "span_limit_hz": 420000,
    "max_bands": 40
  },
  "span_controls": {
    "segment_center_mode": "centroid",
    "segment_centroid_span_hz": 240000,
    "match_bandwidth_pad_hz": 10000,
    "min_match_bandwidth_hz": 80000,
    "display_bandwidth_pad_hz": 30000,
    "min_display_bandwidth_hz": 200000,
    "max_persist_width_hz": 270000,
    "max_card_width_hz": 270000
  },
  "gain": {
    "requested_gain": "30",
    "gain_mode": "manual",
    "actual_gain": 30.0,
    "supported_gains": [0.0, 9.9, 19.7, 29.7]
  },
  "device": {
    "device_key": "rtl:0",
    "device_index": 0,
    "device_serial": "00000001",
    "driver": "rtlsdr_native",
    "tuner": null,
    "unavailable_fields": ["tuner"]
  }
}
```

## Profile Application Rules

In-band FM profile request:

- `requested_profile` is `fm_broadcast`
- `applied_profile` is `fm_broadcast`
- `profile_applied` is `true`
- `profile_skip_reason` is `null`
- final effective values include profile defaults except operator overrides

Out-of-band FM profile request:

- `requested_profile` is `fm_broadcast`
- `applied_profile` is `null`
- `profile_applied` is `false`
- `profile_skip_reason` explains the range mismatch
- fallback/default/final effective values are present

## Required Field Groups

- requested profile
- applied profile
- profile application status
- skipped reason when skipped
- operator/API overrides
- final effective scanner parameters
- frequency range
- step size
- sample rate
- FFT size
- bin width
- persistence thresholds
- revisit settings
- segment center mode
- match and display bandwidth settings
- width caps
- gain settings
- driver/device metadata when available

## Acceptance Checks

- A developer can reconstruct the important scanner/profile/controller decisions without reading scanner log text.
- A skipped profile is visible as skipped, not merely absent.
- Operator overrides are distinguishable from profile defaults.
- Missing hardware telemetry is represented as null or unavailable rather than omitted silently.
