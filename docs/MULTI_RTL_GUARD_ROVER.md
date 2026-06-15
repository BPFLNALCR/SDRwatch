# Hardware-Aware Multi-RTL Guard/Rover Mode

This feature is RTL-only. Scanner execution currently supports native RTL-SDR through `rtlsdr_native`. Airspy, HackRF, and SoapySDR devices may appear as planned or unsupported hardware classes, but they are not runnable scanner backends in this build.

## Capability Tiers

| Tier | Operator meaning |
| --- | --- |
| Tier 0 | No runnable RTL-SDR receiver is detected. Scanner starts that require RTL hardware should not be offered. |
| Tier 1 | One RTL-SDR is detected. Existing single-device scanning and one GUARD window are available. |
| Tier 2 | Two RTL-SDR receivers are detected. GUARD + ROVER role operation is available when locks are free. |
| Tier 2+ | Three or more RTL-SDR receivers are detected. Two GUARD lanes plus REFERENCE or ROVER are available. |

## Manual Roles

- GUARD: parked on a priority window for low-latency observation.
- ROVER: sweeps configured lower-priority ranges with the existing sequential scanner behavior.
- REFERENCE: parked on a stable/noise/reference window for contextual telemetry only.

Automatic role assignment and scheduler optimization are deferred. Operators assign roles manually from the Control page.

## Device Identity

SDRwatch prefers stable identities based on unique RTL serials, reported as `rtl:serial:<serial>`. If serials are missing or duplicated, SDRwatch falls back to runtime index identity such as `rtl:index:0` and marks the assignment as unstable or session-scoped.

Index-only identity can change after replug, reboot, or USB enumeration changes. Persistent role assignment is intended only for unique serial identities.

## Resource Expectations

Target hardware is Raspberry Pi 5 with 4 GB RAM, NVMe storage, and active cooling. The feature is designed for conservative CPU, RAM, USB, and I/O use:

- No continuous raw IQ archive by default.
- Favor metadata, PSD summaries, events, diagnostic JSONL, and bounded future triggered captures.
- Avoid running every receiver at maximum sample rate by default.
- Use diagnostic timing and resource telemetry before deeper DSP optimization.

## Diagnostics

Role-aware scanner jobs include receiver role, role lane, device identity, runtime index, serial when available, backend, job ID, role-run ID, source task/profile, sample accounting, timing fields, process ID, and available CPU/RSS telemetry in diagnostics.

SQLite persistence remains additive and low risk. Role/device/job/source provenance is diagnostic-first, with nullable `scan_updates` provenance columns available for summary storage. Full `signal_tracks` and `observations` fusion is deferred.

## Hardware Acceptance Status

No Raspberry Pi 5 one/two/three-RTL hardware acceptance bundle has been captured in this implementation environment yet. Save diagnostic bundle references here after field validation.
