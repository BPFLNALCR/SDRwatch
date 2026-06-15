# Quickstart: Hardware-Aware Multi-RTL Guard/Rover Mode

This guide describes how to validate the feature after implementation. It is not an implementation script.

## Prerequisites

- Current feature artifacts under `specs/008-multi-rtl-guard-rover/`.
- Controller and web app configured as in the existing SDRwatch workflow.
- For no-hardware checks: fake device discovery and fake process fixtures in tests.
- For hardware acceptance: Raspberry Pi 5, 4 GB RAM, NVMe storage, active cooling, and one to three native RTL-SDR receivers.

## Branch Hygiene

The current planning pass was created while checked out on `007-cross-sweep-persistence-and-telemetry`. Before implementation, use a stacked feature branch such as:

```powershell
git switch -c 008-multi-rtl-guard-rover
```

If the branch already exists, switch to it instead. Keep the feature directory as `specs/008-multi-rtl-guard-rover`.

## No-Hardware Validation

Run the focused automated tests after implementation. Use the workspace-local Python that has project dependencies installed.

```powershell
python -m pytest -q tests/test_multi_rtl_inventory.py tests/test_multi_rtl_roles.py tests/test_multi_rtl_role_runs.py tests/test_multi_rtl_telemetry.py
```

Expected outcomes:

- Zero/one/two/three fake RTL inventory fixtures produce Tier 0, Tier 1, Tier 2, and Tier 2+.
- Missing serials, duplicate serials, and index-only devices produce warnings.
- Only `rtlsdr_native` is reported runnable.
- Airspy, HackRF, and Soapy start attempts are rejected before process spawn.
- Role assignment set/list/clear works.
- Duplicate active receiver assignment is rejected.
- Atomic lock tests prevent concurrent same-device starts.
- Stale locks and dead processes are reconciled.
- Role-run status degrades when a child job fails.
- Diagnostic records include role/device/job/task provenance and timing fields.

## Regression Validation

Run existing no-hardware regression tests that protect the current behavior.

```powershell
python -m pytest -q tests/test_cross_sweep_persistence.py tests/test_device_telemetry.py tests/test_effective_parameter_manifest.py tests/test_control_fm_validation.py tests/test_control_page_scan_settings.py tests/test_web_diagnostics_bundle.py
```

Expected outcomes:

- Existing FM Broadcast profile behavior remains stable.
- Existing cross-sweep persistence promotion remains stable.
- Diagnostic JSONL and bundle behavior remain useful.
- Existing `/api/jobs` payload compatibility is preserved.

## Controller/Web Smoke Validation

Start the controller and web app using the existing project workflow, then validate through the browser.

1. Open the SDRwatch control page.
2. Confirm the hardware inventory panel shows capability tier and receiver warnings.
3. Confirm existing single-device scan controls still show the device selector and can build the same `/api/jobs` payload as before.
4. Assign one receiver to GUARD.
5. Start one GUARD role run.
6. Confirm the role-run status shows the role, receiver identity, child job ID, and running state.
7. Stop the run and confirm the receiver becomes available.

## Two-RTL Acceptance

With two RTL receivers attached:

1. Refresh hardware inventory.
2. Confirm Tier 2.
3. Assign one receiver to GUARD and the other to ROVER.
4. Start a grouped role run.
5. Confirm two child jobs run with different receiver identities.
6. Attempt to assign or start the same receiver twice and confirm the action is rejected.
7. Stop the grouped run and confirm both locks are released.

Expected diagnostic evidence:

- Each child job has role, role lane, device identity, serial/index, job ID, and role-run ID.
- Per-window records include sample rate, FFT size, averaging, segment count, samples read, timing fields, and unavailable fields where needed.

## Three-RTL Acceptance

With three RTL receivers attached:

1. Refresh hardware inventory.
2. Confirm Tier 2+.
3. Assign friendly GUARD, watchlist GUARD, and either REFERENCE or ROVER.
4. Start the grouped run.
5. Confirm three child jobs run and no physical receiver is assigned twice.
6. Stop one child job directly and confirm grouped status becomes degraded or terminal as appropriate.
7. Stop the group and confirm all remaining child jobs stop cleanly.

## Pi 5 Benchmark Capture

Run one, two, and three receiver scenarios with diagnostics enabled.

Collect:

- Diagnostic bundle for each run.
- Controller job logs.
- Hardware inventory response.
- Role-run status response.
- Timing summaries for tune, flush, read, transform, detect, database update, JSONL logging, and total window duration.
- Resource telemetry for CPU load and RSS memory when available.

Expected outcome:

- Bottlenecks can be identified from metadata and diagnostics without continuous raw IQ capture.
- Missing platform metrics are marked as unavailable rather than causing scan failure.

## Provenance Boundary

The first implementation keeps full per-window timing, sample accounting, resource
telemetry, and receiver role/device/job provenance diagnostic-first in JSONL and
diagnostic bundles. Controller role assignments, role runs, and child job
metadata are durable in controller state. SQLite `scan_updates` has nullable
role/device/job/source provenance columns for low-risk summary storage, while
full detection-level observation/fusion provenance remains deferred.

## Documentation Check

Review README and operator docs after implementation:

- Native RTL-SDR scanner execution is the only current runnable backend.
- Airspy, HackRF, and Soapy are clearly marked planned/future or unsupported.
- Capability tiers, role semantics, identity warnings, and Pi 5 resource expectations are documented.
- No continuous raw IQ capture is described as the default.

## Implementation Environment Notes

- No-hardware automated validation and fake-controller HTTP `/control` smoke were run in the development environment.
- In-app browser validation was attempted but could not complete because browser automation failed with a Windows sandbox permission error.
- Raspberry Pi 5 one/two/three-RTL hardware acceptance was not run in this environment.
- CPU load, RSS memory, and dropped-read telemetry should be treated as platform-dependent fields and verified during Pi 5 bundle capture.
