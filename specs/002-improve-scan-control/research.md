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

## Decision: Update GUI defaults and presets using diagnostic evidence

**Rationale**: `docs/DETECTION_TUNING_REPORT.md` shows that real RTL-SDR Blog v4 scans already produce raw candidates, emitted detector segments, and accepted hits. The failure is promotion into `baseline_detections`, not an absence of RF energy and not a GUI card-rendering problem. Existing GUI defaults therefore are not safe enough for first-light hardware use because they can run indefinitely with accepted hits and zero cards. Safe defaults/presets should still preserve `/api/jobs` names and scanner layering, but they may intentionally change submitted parameter values to produce signal cards.

**Alternatives considered**:

- Keep existing defaults unchanged: rejected because the diagnostic bundle shows current defaults can produce no signal cards despite accepted hits.
- Rewrite CFAR/detection algorithms first: rejected because candidates and accepted hits already exist; the immediate failure is promotion/persistence.
- Treat FFT as the root cause: rejected for this failure because the analyzed run used `fft=8192`, `avg=8`, and still promoted zero detections.
- Reset only visible Basic controls: rejected because acceptance requires every Basic, Tuning, and Expert setting to reset.

## Decision: Resolve the promotion/persistence mismatch through GUI presets first

**Rationale**: The current sweep uses `step=2.4e6` with `samp_rate=2.4e6`, creating effectively non-overlapping windows. The detection engine clusters candidates within a single sweep, while the default promotion gate requires `persistence_min_hits=2` and `persistence_min_windows=2`. Narrow stable signals recur across repeated sweeps in the same window, not across two windows in one sweep, so they can be accepted repeatedly without promotion.

**Alternatives considered**:

- Persist candidate clusters across sweeps immediately: deferred as a deeper detection-engine change after GUI defaults prove the desired operator behavior.
- Require operators to run custom scanner CLI flags: rejected because SDRwatch is GUI-operated.
- Change only FFT/averaging: rejected because the diagnostic data shows detector emission and hit acceptance already work.

## Decision: Provide three GUI tuning presets

**Rationale**: Operators need clear first-light and baseline choices without learning raw scanner flags. Presets can remain web-only configuration helpers that populate existing `/api/jobs` parameters.

**Preset directions**:

- **RTL-SDR v4 Discovery**: fast non-overlapping `step=2.4e6`, fixed manual gain around `30 dB`, `fft=4096` or `8192`, `avg=8`, and relaxed promotion gates such as `persistence_min_hits=1` and `persistence_min_windows=1` so initial cards appear.
- **Stable Baseline**: overlapping `step=1.2e6`, fixed manual gain around `25-30 dB`, `fft=8192`, `avg=16`, and stricter promotion gates such as `persistence_min_hits=2` and `persistence_min_windows=2`, accepting the scan-speed penalty.
- **Fast Wide Survey**: `fft=4096`, `avg=8`, `step=2.4e6`, fixed manual gain around `30 dB`, and relaxed promotion gates for broad survey speed.

**Alternatives considered**:

- One universal default: rejected because wide fast survey, first-light discovery, and stable baseline scans have different speed/noise/persistence tradeoffs.
- Hide presets behind CLI examples: rejected because operator workflows must stay in the web GUI.
- Make the slow stable baseline preset the only default: rejected because first-light hardware onboarding should produce visible cards quickly.

## Decision: Prefer fixed manual RTL-SDR gain for baseline/detection presets

**Rationale**: Auto gain can move the apparent noise floor and makes baseline comparison harder to reason about. The diagnostic bundle did not prove severe auto-gain instability because actual tuner gain telemetry was not recorded, but fixed gain is still the better baseline/detection default. A manual RTL-SDR gain around `30 dB` is a reasonable starting candidate for RTL-SDR Blog v4, subject to supported gain values and overload behavior.

**Alternatives considered**:

- Keep auto gain as the primary safe default: rejected for baseline/detection presets because repeatability matters more than convenience.
- Force a single non-configurable gain: rejected because strong-signal environments can overload and operators need an easy way to reduce gain.

## Decision: Treat FFT as a preset speed/resolution knob

**Rationale**: Lower FFT sizes scan faster but provide wider bins, less precise peak centering, and less reliable bandwidth characterization. Higher FFT sizes improve frequency resolution and signal characterization, but slow wide sweeps on Raspberry Pi 5. The diagnostic run used `fft=8192`, so FFT is not the primary fix for zero cards; it should be presented as a scan-quality/speed tradeoff once promotion works.

**Alternatives considered**:

- Set maximum FFT everywhere: rejected because wide sweeps on Raspberry Pi 5 need usable scan cadence.
- Set minimum FFT everywhere: rejected because characterization quality suffers.
- Claim FFT fixes no-card behavior: rejected because the evidence points to promotion/persistence.

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
