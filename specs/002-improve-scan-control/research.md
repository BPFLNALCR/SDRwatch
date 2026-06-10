# Phase 0 Research: Improve Scan Control

## Decision: Keep the web GUI as the scan workflow source of truth

**Rationale**: The constitution and feature spec both require operators to configure and run scans through the web UI and controller job lifecycle. The current page already selects baselines, devices, zones, run mode, diagnostics mode, start/stop, and live logs.

**Alternatives considered**:

- Add CLI instructions for scan setup: rejected because the scanner CLI is an internal backend surface, not the operator workflow.
- Add a separate frontend application: rejected because the project constitution requires the minimal local stack and the feature explicitly forbids a frontend framework.

## Decision: Preserve `/api/jobs` payload shape and parameter names

**Rationale**: `templates/control.html` currently submits `{device_key, label, baseline_id, params}` to `POST /api/jobs`, and `sdrwatch-control.py` maps existing `params` keys to scanner flags. Compatibility requires preserving these names so the UI cleanup does not alter controller or scanner behavior.

**Alternatives considered**:

- Introduce a new typed scan-configuration endpoint: rejected because it would add interface churn without being required for the page cleanup.
- Rename GUI fields to friendlier parameter names: rejected because copied settings and job submission must use the same existing parameter names.

## Decision: Use one GUI-generated settings builder for submit, reset comparison, and copy JSON

**Rationale**: A single local settings builder avoids divergence between what the page submits and what "Copy current scan settings" shares with Codex. It also supports focused tests that verify the generated JSON and posted `params` match.

**Alternatives considered**:

- Build copy JSON separately from the submit payload: rejected because duplicate mapping is likely to drift.
- Copy raw form fields without normalizing to job params: rejected because reviewers need the job-parameter names actually submitted to `/api/jobs`.

## Decision: Restore safe defaults from existing SDRwatch defaults

**Rationale**: The feature must not tune DSP behavior. Safe defaults should therefore reflect the defaults already present in the page and scanner/controller path: common examples include `samp_rate` 2.4e6, `gain` auto, `fft` 4096, `avg` 8, `threshold_db` 8, `guard_bins` 1, `min_width_bins` 2, CFAR quantile 0.75, persistence hit ratio 0.6, persistence minimum seconds 10, minimum hits 2, minimum windows 2, `max_detection_width_ratio` 3.0, and `new_ema_occ` 0.02.

**Alternatives considered**:

- Invent new RF defaults: rejected because that would be DSP/product tuning outside the feature scope.
- Reset only visible Basic controls: rejected because acceptance requires every Basic, Tuning, and Expert setting to reset.

## Decision: Move common RF parameters to bounded Tuning controls

**Rationale**: The current advanced form exposes many raw numeric fields together. Sliders with visible values for the required common RF controls reduce accidental edits and make current values easier to scan. Selects or segmented controls are appropriate for bounded choices such as CFAR mode, FFT, averaging, sample rate, and gain mode.

**Alternatives considered**:

- Keep all fields as text or number inputs: rejected because it does not address autofill and usability problems.
- Hide tuning parameters entirely: rejected because operators still need common RF adjustments from the GUI.

## Decision: Put less common and risky controls in Expert controls

**Rationale**: Parameters such as clustering width, max detection width, new occupancy threshold, persistence gates, revisit behavior, paths, and remaining compatibility controls need to remain available but should not crowd the normal scan workflow.

**Alternatives considered**:

- Remove expert controls from the page: rejected because existing GUI parameter submission must be preserved.
- Leave all expert fields in a generic Advanced block: rejected because the spec requires Basic, Tuning, and Expert sections.

## Decision: Suppress browser autofill at the scan form and field level

**Rationale**: Browser autofill can silently insert unrelated numeric values into fields like latitude, longitude, thresholds, or paths. Adding `autocomplete="off"` to the scan form and numeric controls, and using bounded controls where practical, directly reduces this risk.

**Alternatives considered**:

- Rely on operator review before submit: rejected because the feature goal is to prevent silent autofill damage.
- Disable all saved state including selected baseline: rejected because baseline persistence is part of the existing page workflow and is not the same as browser autofill.

## Decision: Use existing job detail `cmd` for generated scanner command copy

**Rationale**: The controller job model already includes `cmd`, and job detail returns the job dictionary. This makes command copy safe to plan as an internal/debug action for an active or recent job without adding a command-preview endpoint or teaching operators to run the CLI.

**Alternatives considered**:

- Add a command-preview endpoint: rejected as unnecessary interface expansion for this feature.
- Generate a scanner command in browser JavaScript: rejected because the controller owns scanner invocation and command construction.
- Make scanner command copy a primary action: rejected because the operator workflow must remain GUI-first.

## Decision: Keep diagnostics mode pathless in Basic controls

**Rationale**: Existing diagnostics mode support can generate a safe per-job diagnostic path through the controller when `diagnostics_mode` is set. Normal operators should not type diagnostic paths. Any raw path override, if retained for compatibility, belongs in Expert controls.

**Alternatives considered**:

- Ask operators to enter diagnostic paths: rejected by the spec and prior diagnostics plan.
- Remove raw diagnostic path compatibility entirely: deferred to implementation review because compatibility may require preserving an expert override.

## Decision: Verify with no-hardware web tests and GUI manual acceptance

**Rationale**: This is a control-page usability and payload-shape change. Tests can render the page, verify required controls and autofill attributes, and exercise `/api/jobs` with fake controller clients. Final acceptance must use the browser and controller lifecycle.

**Alternatives considered**:

- Require SDR hardware for all verification: rejected because most behavior is page/payload behavior and should be testable without hardware.
- Treat scanner CLI checks as acceptance: rejected because CLI checks only validate internal backend behavior.
