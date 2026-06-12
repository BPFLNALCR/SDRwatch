# Contract: Control Page UI

## Diagnostics Mode Toggle

**Location**: Scan/control page before or near the Start Scan action.

**Behavior**:

- The operator can enable or disable Diagnostics mode before starting a scan.
- The control page must not ask the operator to type a diagnostic JSONL path.
- When enabled, the scan-start request includes diagnostic intent through job params.
- When disabled, the scan-start request omits diagnostic intent and preserves current behavior.

**Visible states**:

- Off: normal scan behavior.
- On: next scan captures diagnostic evidence automatically.
- Running job with diagnostics: export action is available for the current job.
- Recent job available: export action is available for the last known job.

## Export Diagnostic Bundle Action

**Location**: Scan/control page job status area or recent-job controls.

**Behavior**:

- The action downloads a zip for the selected active or recent job.
- The action remains available during a running scan when a job ID is known.
- The UI shows a helpful error if no job can be selected.
- The UI does not require internet access or cloud sign-in.

**Download result**:

- Browser receives a zip attachment.
- Filename identifies SDRwatch diagnostics and the job ID.

## Operator Notes Template

Every downloaded bundle includes a Markdown notes template with these prompts:

- Expected behavior
- Actual behavior
- Frequency or band affected
- Problem type
- Additional notes

Problem type choices:

- false positive
- false negative
- wrong bandwidth
- wrong center
- merged signals
- split signals
- unstable baseline
- other
