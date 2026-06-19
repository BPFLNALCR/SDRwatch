# Quickstart: Profile-Governed Signal Identity Span and Revisit Authority

## Purpose

Validate that SDRwatch keeps raw spectral fragments separate from signal identity span, persisted/card span, operator display span, and revisit authority.

Normal operator acceptance remains web GUI -> controller job lifecycle -> scanner backend. Direct scanner CLI checks are backend smoke only.

## Prerequisites

- Workspace: `C:\Users\User\SDRwatch`
- Branch: `008-multi-rtl-guard-rover`
- Use bundled Python in this environment when system Python is unavailable:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest -q --basetemp .test-tmp\span-policy
```

Use a fresh workspace-local `--basetemp` for each broad run.

## No-Hardware Validation

Run focused tests first:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest -q tests/test_signal_span_policy.py tests/test_extent_hysteresis.py tests/test_fm_characterization_persistence.py tests/test_non_fm_width_scope.py --basetemp .test-tmp\span-policy-focused
```

Expected outcomes:

- Tiny raw segments remain available as raw fragments.
- Identity/match span does not shrink below active profile identity floor.
- Persisted/card span does not shrink below active profile persist floor except scan-edge clipping.
- Display span remains governed by display policy.
- Tiny or far-offset revisit evidence is confirmation-only when policy forbids identity update.
- Narrowband and discovery profiles are not forced into broad FM-like widths.

Run profile, controller, and diagnostics contract tests:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest -q tests/test_fm_validation_profile.py tests/test_effective_parameter_manifest.py tests/test_control_fm_validation.py tests/test_fm_characterization_diagnostics.py tests/test_web_diagnostics_bundle.py --basetemp .test-tmp\span-policy-contracts
```

Expected outcomes:

- New policy fields serialize from profiles and apply through CLI args.
- Controller passes policy params through existing `/api/jobs` `params`.
- Effective parameters expose the derived signal span policy.
- Diagnostics preserve old fields and add raw/identity/persist/display/revisit authority fields.

Run existing regression suites:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m pytest -q tests/test_cross_sweep_persistence.py tests/test_device_telemetry.py tests/test_multi_rtl_inventory.py tests/test_multi_rtl_backend_gating.py tests/test_multi_rtl_guard.py tests/test_multi_rtl_telemetry.py tests/test_legacy_job_compatibility.py --basetemp .test-tmp\span-policy-regression
```

Expected outcomes:

- Cross-sweep persistence still passes.
- Effective-parameter/profile export remains coherent.
- Slice 1 hardware inventory/backend gating tests still pass.
- Legacy single-device job compatibility remains intact.

## Backend Smoke

Use CLI only as scanner backend smoke:

```powershell
& 'C:\Users\User\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -m sdrwatch.cli --list-profiles
```

Expected outcomes:

- Profiles list successfully.
- Broad profile policy fields appear for the FM Broadcast canary.
- Narrow/default profiles do not inherit FM-like display floors unless configured.

## Optional Pi 5 Hardware Acceptance

Run the normal single-device RTL scan from the web UI/controller path using FM Broadcast as the live canary:

- driver/backend: `rtlsdr_native`
- device: `rtl:0` or the controller-discovered RTL key
- requested profile: `fm_broadcast`
- diagnostics enabled

Export the diagnostic bundle from the web UI.

Expected outcomes:

- `requested_profile`, `applied_profile`, and `profile_applied` agree in effective parameters.
- `receiver_role`, `role_lane`, and `role_run_id` remain `null` for a legacy single-device job.
- Raw/revisit fragment widths can remain tiny and are labeled as raw fragments.
- Ordinary persisted/card spans do not fall below the active persist floor except documented scan-edge clipping.
- Revisit authority diagnostics explain confirmation-only cases.
- The card count need not match an exact FM station count; the acceptance goal is fewer misleading narrow persisted cards and clearer semantics.

## Out of Scope for Validation

- No FM-specific hard-coding.
- No 88-108 MHz special-case logic in generic detection/persistence code.
- No Airspy/HackRF/Soapy runtime support.
- No new multi-RTL role assignment or grouped role-run work.
- No UI redesign.
- No signal fusion schema.
- No Rust DSP rewrite.
- No continuous IQ capture.
- No broad detector threshold retuning based only on the FM canary.
