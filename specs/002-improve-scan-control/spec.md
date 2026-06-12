# Feature Specification: Improve Scan Control

**Feature Branch**: `005-improve-scan-control`

**Created**: 2026-06-10

**Status**: Draft

**Input**: User description: "Clean up and improve the SDRwatch scan/control page so the web GUI is the clear primary operator interface for configuring and running scans, with safer controls, organized settings, copyable generated settings, GUI-based verification, and real-hardware defaults/presets that produce initial signal cards on RTL-SDR Blog v4 hardware."

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Run a Scan From Basic Controls (Priority: P1)

As an SDRwatch operator, I can use the scan/control page to choose the monitoring location or baseline, select an SDR device, select or enable monitoring zones, choose the run mode, enable Diagnostics mode when needed, start or stop the scan, and watch live logs without using scanner commands.

**Why this priority**: The scan/control page must be the normal operator surface for running SDRwatch. If this flow is unclear, operators may misconfigure scans or bypass the controller workflow.

**Independent Test**: Can be fully tested through the browser by selecting a baseline, selecting a device, choosing monitoring zones and run mode, optionally enabling Diagnostics mode, starting a scan through the page, observing live logs, and stopping the scan through the page.

**Acceptance Scenarios**:

1. **Given** the scan/control page has available baselines and devices, **When** the operator selects a baseline, device, monitoring zones, run mode, and starts a scan, **Then** the scan starts through the existing controller job lifecycle and live logs remain visible on the page.
2. **Given** a scan is running, **When** the operator uses the page stop control, **Then** the scan stops through the existing controller job lifecycle and the page reflects the stopped state.
3. **Given** the operator enables Diagnostics mode before starting a scan, **When** the scan starts, **Then** diagnostics are enabled without requiring the operator to type or choose a diagnostic path.
4. **Given** the operator does not need advanced settings, **When** they use only Basic controls, **Then** they can complete the normal scan workflow without reviewing raw tuning or expert parameters.

---

### User Story 2 - Tune Common RF Parameters Safely (Priority: P2)

As an SDRwatch operator adjusting common RF behavior, I can use a Tuning controls section with sliders, selects, segmented choices, visible values, and inline help so I can understand and adjust common scan parameters without browser autofill or raw numeric fields silently changing important settings.

**Why this priority**: Operators need access to common RF settings, but the current raw numeric form is hard to scan and vulnerable to accidental autofill changes.

**Independent Test**: Can be tested through the browser by changing each Tuning control, verifying live values and descriptions are visible, starting a scan, and confirming the generated job settings carry the same parameter names as before.

**Acceptance Scenarios**:

1. **Given** the operator opens the Tuning controls section, **When** they adjust `threshold_db`, `guard_bins`, `min_width_bins`, `cfar_alpha_db`, or `cfar_quantile`, **Then** each control uses a slider and shows the current numeric value immediately.
2. **Given** the operator adjusts CFAR mode, FFT, averaging, sample rate, or gain, **When** they change those controls, **Then** the choices are presented as safer bounded controls rather than free-form raw numeric entry where practical.
3. **Given** the operator reviews a tuning control, **When** they look at the control label or help affordance, **Then** they can see a short explanation of what the setting affects.
4. **Given** the browser has stored autofill values from other forms, **When** the scan form is displayed or submitted, **Then** browser autofill does not silently replace scan parameters.

---

### User Story 3 - Choose a Real-Hardware Tuning Preset (Priority: P2)

As an SDRwatch operator using an RTL-SDR Blog v4, I can choose a GUI tuning preset that explains the scan-speed, FFT-resolution, gain, and persistence tradeoffs so my first scan can produce signal cards without using scanner CLI commands.

**Why this priority**: The diagnostic report shows SDRwatch detects and accepts RF candidates but can promote zero detections under current defaults. Operators need web presets that produce first-light cards while still allowing slower, cleaner baseline scans.

**Independent Test**: Can be tested through the browser by selecting each preset, copying the current scan settings, and confirming the generated `/api/jobs` parameters match the documented preset values. Real hardware acceptance requires starting from the web GUI and confirming signal cards appear.

**Acceptance Scenarios**:

1. **Given** the operator selects an RTL-SDR v4 Discovery preset, **When** they copy or start the scan, **Then** the generated job parameters use fixed manual gain, fast wide-sweep settings, and promotion gates that can produce initial signal cards.
2. **Given** the operator selects a Stable Baseline preset, **When** they copy or start the scan, **Then** the generated job parameters use overlapping sweep windows and stricter persistence gates with an explicit slower-scan tradeoff.
3. **Given** the operator reviews FFT choices, **When** they compare presets, **Then** the UI explains that lower FFT scans faster but characterizes peaks less accurately, while higher FFT improves frequency resolution and slows wide sweeps.
4. **Given** the operator is using fixed manual gain, **When** they review the gain control or preset text, **Then** overload risk remains visible and the manual gain remains configurable.

---

### User Story 4 - Review Expert Settings and Share Current Configuration (Priority: P3)

As an advanced operator or maintainer, I can find less common or dangerous settings in an Expert controls section, reset the page to known safe defaults, and copy the current GUI-generated scan settings as JSON for sharing with Codex or another reviewer.

**Why this priority**: Expert parameters must remain available for compatibility and troubleshooting, but they should not dominate the normal operator workflow. Copyable settings make remote review safer than screenshots or manually transcribed values.

**Independent Test**: Can be tested through the browser by opening Expert controls, changing representative expert settings, using Reset to safe defaults, using Copy current scan settings, and verifying the copied JSON matches the job parameters the page would submit.

**Acceptance Scenarios**:

1. **Given** the operator needs an uncommon setting, **When** they open Expert controls, **Then** settings such as cluster width, persistence, revisit, database path, JSONL path, and any still-needed raw diagnostic path override are grouped away from Basic controls.
2. **Given** the operator has changed multiple settings, **When** they choose Reset to safe defaults, **Then** all scan settings return to the documented safe defaults for the page.
3. **Given** the operator chooses Copy current scan settings, **When** the action completes, **Then** the clipboard contains valid JSON using the same GUI-generated job parameter names that would be submitted for a scan.
4. **Given** the controller already exposes a generated scanner command safely, **When** the page provides a copy action for that command, **Then** it is clearly labeled as internal/debug information and not presented as the primary operator workflow.
5. **Given** the generated scanner command is not safely available, **When** the page is displayed, **Then** the page does not add a misleading or broken scanner-command copy action.

### Edge Cases

- No baseline or monitoring location is available yet.
- No SDR device is currently available or device discovery fails.
- Monitoring zones are disabled, empty, or only partially configured.
- Diagnostics mode is enabled but a raw diagnostic path override is not shown or is blank.
- The browser attempts to autofill number-like fields with unrelated saved values.
- A slider reaches its minimum or maximum allowed value.
- The operator changes gain between auto and manual modes.
- The operator changes between presets after modifying individual tuning values.
- A preset uses a fixed manual gain that may overload in a strong-signal environment.
- A low FFT preset scans faster but gives coarser frequency and bandwidth characterization.
- A high FFT preset gives better frequency resolution but slows a wide Raspberry Pi scan.
- A fast non-overlapping preset uses relaxed persistence gates to produce first-light cards.
- A stable baseline preset uses overlapping windows and therefore scans more slowly.
- The operator resets defaults after changing both Basic and Expert controls.
- The operator copies settings before starting a scan.
- Clipboard access is blocked by the browser.
- The generated internal scanner command is unavailable, omitted, or not safe to expose.
- Existing job-start consumers rely on the current job parameter names.
- Live logs are empty, delayed, or temporarily unavailable while a job is starting or stopping.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The scan/control page MUST remain the primary operator interface for configuring, starting, stopping, and monitoring SDRwatch scans.
- **FR-002**: The page MUST preserve the existing web workflow for selecting monitoring location or baseline, selecting SDR device, selecting or enabling monitoring zones, choosing run mode, starting and stopping scans, and viewing live logs.
- **FR-003**: The page MUST organize scan settings into three clearly labeled sections: Basic controls, Tuning controls, and Expert controls.
- **FR-004**: Basic controls MUST include only the normal operator workflow controls: baseline or location, device, monitoring zones, run mode, Diagnostics mode, and start/stop controls.
- **FR-005**: Diagnostics mode MUST be available from Basic controls and MUST NOT require the operator to manually type a diagnostic path for normal use.
- **FR-006**: Tuning controls MUST include `threshold_db`, `guard_bins`, `min_width_bins`, CFAR mode, `cfar_alpha_db`, `cfar_quantile`, FFT, averaging, sample rate, and gain.
- **FR-007**: `threshold_db` MUST be represented as a slider with a visible live numeric value.
- **FR-008**: `guard_bins` MUST be represented as a slider with a visible live numeric value.
- **FR-009**: `min_width_bins` MUST be represented as a slider with a visible live numeric value.
- **FR-010**: CFAR mode MUST be represented as a bounded selection control.
- **FR-011**: `cfar_alpha_db` MUST be represented as a slider with a visible live numeric value.
- **FR-012**: `cfar_quantile` MUST be represented as a slider with a visible live numeric value.
- **FR-013**: FFT MUST be represented as a bounded selection control.
- **FR-014**: Averaging MUST be represented as a bounded selection control or segmented button group.
- **FR-015**: Sample rate MUST be represented as a bounded selection control.
- **FR-016**: Gain MUST support an auto/manual choice, and manual gain MUST require an explicit operator choice before submitting a manual value.
- **FR-017**: Each Tuning control MUST provide an inline description or tooltip explaining what the setting affects.
- **FR-018**: Expert controls MUST contain less common or higher-risk parameters, including `cluster_merge_hz`, `max_detection_width_hz`, `max_detection_width_ratio`, `new_ema_occ`, persistence mode, persistence hit ratio, persistence minimum seconds, persistence minimum hits, persistence minimum windows, revisit parameters, database path, JSONL path, and any still-needed raw diagnostic path override.
- **FR-019**: Expert controls MUST remain available without making them part of the normal Basic controls workflow.
- **FR-020**: The scan form and numeric controls MUST suppress browser autocomplete or autofill so saved browser values do not silently alter scan settings.
- **FR-021**: The page MUST provide a visible Reset to safe defaults action.
- **FR-022**: Reset to safe defaults MUST restore every Basic, Tuning, and Expert scan setting to the page's documented safe default value.
- **FR-023**: The page MUST provide a Copy current scan settings action.
- **FR-024**: Copy current scan settings MUST place valid JSON on the clipboard containing the GUI-generated job parameters for the current page state.
- **FR-025**: Copied scan settings MUST use the same parameter names the GUI would submit when starting a scan.
- **FR-026**: The scan start workflow MUST preserve the existing `/api/jobs` payload shape and parameter names.
- **FR-027**: Existing scan parameter submission behavior MUST remain compatible unless a value was intentionally changed by the operator in the GUI.
- **FR-028**: The page MAY include a Copy generated scanner command action only when the controller already exposes that command safely.
- **FR-029**: If Copy generated scanner command is present, it MUST be labeled as internal/debug information and MUST NOT be presented as the primary workflow for operators.
- **FR-030**: If the generated scanner command is unavailable or unsafe to expose, the page MUST omit the copy action or show it as unavailable without blocking the scan workflow.
- **FR-031**: The feature MUST NOT rewrite scanner, controller, database, detection, DSP, baseline, or classification internals; it MAY intentionally change GUI-submitted default/preset parameter values while preserving existing `/api/jobs` names and controller lifecycle.
- **FR-032**: The feature MUST NOT add simulation mode or new SDR capabilities.
- **FR-033**: The feature MUST NOT redesign dashboard areas outside the scan/control page.
- **FR-034**: The page MUST continue to operate as a local, server-rendered web GUI and MUST NOT introduce a separate frontend application framework.
- **FR-035**: Manual verification instructions for this feature MUST be GUI-based and MUST treat scanner CLI checks only as optional internal backend smoke tests.
- **FR-036**: The page MUST provide GUI-accessible tuning presets for at least RTL-SDR v4 Discovery, Stable Baseline, and Fast Wide Survey workflows.
- **FR-037**: Presets MUST be applied by the web GUI as `/api/jobs` parameters and MUST NOT instruct operators to run scanner CLI commands.
- **FR-038**: The RTL-SDR v4 Discovery preset MUST favor producing initial signal cards on real hardware by resolving the promotion/persistence mismatch identified in `docs/DETECTION_TUNING_REPORT.md`.
- **FR-039**: The Stable Baseline preset MUST document and encode the slower overlapping-window tradeoff needed for stricter multi-window persistence.
- **FR-040**: Preset help text MUST explain FFT as a speed/resolution and characterization tradeoff, not as the primary fix for zero signal cards.
- **FR-041**: Preset help text MUST explain that fixed manual RTL-SDR gain is preferred for baseline/detection consistency while preserving operator control for overload conditions.
- **FR-042**: The feature MUST NOT rewrite CFAR or detection algorithms as part of preset/default work unless a later implementation finds an obvious one-line bug fix.

### Key Entities *(include if feature involves data)*

- **Scan Configuration**: The complete set of operator-selected values that the GUI converts into job parameters for starting a scan.
- **Basic Control Section**: The normal operator workflow controls required to choose scan context, choose hardware, choose monitoring scope, choose run mode, enable diagnostics, and start or stop scans.
- **Tuning Control Section**: Common RF-related controls that operators may adjust using bounded UI controls with visible values and explanations.
- **Expert Control Section**: Less common or higher-risk controls that remain available for compatibility and troubleshooting while staying outside the normal workflow.
- **Safe Defaults**: The documented default values restored by the reset action for every scan setting on the page.
- **GUI Tuning Preset**: A named set of GUI-applied scan parameters such as RTL-SDR v4 Discovery, Stable Baseline, or Fast Wide Survey.
- **Generated Job Parameters**: The JSON-compatible parameters produced by the GUI for the current scan configuration and submitted to the existing job-start workflow.
- **Diagnostics Mode Setting**: The operator-visible toggle that enables diagnostic capture for a scan without requiring manual path entry.
- **Generated Scanner Command**: Optional internal/debug information produced by the controller when safely available; not an operator workflow requirement.
- **Live Logs View**: The operator-visible job log stream or refresh area used to observe scan startup, runtime, and stop behavior.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: An operator can select scan context, choose a device, choose monitoring zones, choose run mode, enable Diagnostics mode, start a scan, view logs, and stop the scan from the web GUI in under 3 minutes without running scanner CLI commands.
- **SC-002**: During review, 100% of the required Basic, Tuning, and Expert controls are present in the correct section.
- **SC-003**: 100% of required slider controls display their current numeric value immediately after the operator changes them.
- **SC-004**: Reset to safe defaults restores all scan settings to their documented defaults after representative changes across Basic, Tuning, and Expert sections.
- **SC-005**: Copy current scan settings produces parseable JSON that includes the same parameter names the GUI submits for starting a scan.
- **SC-006**: Starting a scan from the GUI continues to submit the existing `/api/jobs` parameter names for unchanged settings.
- **SC-007**: Diagnostics can be enabled from Basic controls without the operator manually typing a diagnostic path.
- **SC-008**: Browser autofill does not silently replace any scan parameter during initial page display, operator editing, or scan submission.
- **SC-009**: Existing detection, baseline, database schema, controller job lifecycle, and scanner backend interfaces remain unchanged, while GUI-submitted default/preset parameter values are intentionally updated through the existing job payload.
- **SC-010**: Manual verification steps guide the tester through the browser and controller job lifecycle, with any scanner CLI check clearly marked as internal backend smoke testing only.
- **SC-011**: Copy current scan settings after selecting each tuning preset shows the documented preset values using existing `/api/jobs` parameter names.
- **SC-012**: On Raspberry Pi 5 with RTL-SDR Blog v4 and an active RF environment, the RTL-SDR v4 Discovery preset produces at least one promoted signal card through the web GUI during manual hardware validation.
- **SC-013**: Preset documentation clearly states that FFT affects scan speed and characterization quality, while the zero-card diagnostic failure is primarily a promotion/persistence mismatch.

## Assumptions

- "Safe defaults" means the current safe scan defaults already used by the GUI or controller unless planning identifies a more appropriate existing SDRwatch default.
- The diagnostic report in `docs/DETECTION_TUNING_REPORT.md` supersedes earlier assumptions that existing safe defaults are sufficient for real RTL-SDR v4 first-light scans.
- A fixed RTL-SDR gain around 30 dB is a reasonable initial GUI preset candidate, but the operator must be able to adjust it if overload appears.
- Expert controls may be visually collapsed or otherwise de-emphasized, as long as they remain discoverable and usable from the scan/control page.
- Existing authentication, permissions, controller token behavior, and device-discovery behavior remain unchanged.
- Clipboard actions can provide a visible failure message when browser permissions block copying.
- Raw diagnostic path override remains an Expert control only if existing behavior still requires preserving that override for compatibility.
- Any generated scanner command copied from the page is for debugging and sharing context, not for instructing operators to run scans through the CLI.
