# Contract: PROJECT_INVENTORY.md

## Purpose

Define the minimum structure and evidence requirements for the future `docs/PROJECT_INVENTORY.md` document.

## Output

- **Path**: `docs/PROJECT_INVENTORY.md`
- **Audience**: Contributors and maintainers
- **Format**: Markdown
- **Scope**: Current repository structure and runtime topology only

## Required Sections

The document must include the following sections or clearly equivalent headings:

1. **Purpose and Scope**
   - Explain that the document is a current-state repository inventory.
   - State that it is discovery-focused and not a rewrite proposal.

2. **Top-Level Repository Layout**
   - Summarize major directories and root scripts.
   - Distinguish runtime code, web code, templates, static assets, tests, install tooling, and governance/docs files.

3. **Canonical Entry Points and Compatibility Paths**
   - Identify the authoritative scanner CLI, controller, web, and query helpers.
   - Explicitly label deprecated or compatibility-only shims.

4. **Runtime Topology**
   - Describe how scanner, controller, web, and direct-database responsibilities interact.
   - Make clear that the controller owns device locking and scan process spawning.

5. **Database and Schema Ownership**
   - Identify where tables are created.
   - Identify where schema-altering startup migrations occur.
   - Note any split ownership or legacy compatibility behavior that affects contributor understanding.

6. **Installer, Service, and Deployment Behavior**
   - Describe what the install and uninstall scripts do.
   - Identify generated env files, generated systemd units, deployment directories, and preserved state.
   - State whether service unit files are committed or generated.

7. **Runtime and Generated Artifacts**
   - List expected database files, JSONL outputs, controller state, locks, logs, and other generated paths.
   - Identify who creates each artifact and when.

8. **Tests and CI Surface**
   - Summarize the current test layout.
   - Mention CI workflow coverage and its limits where visible.

9. **Known Missing Standard Project Files**
   - Explicitly note absent packaging metadata, dependency manifests, committed service files, docs directory, migrations directory, or similar contributor-facing gaps if absent.

10. **Local No-Hardware Workflow**
    - Explain what can be run or validated without SDR hardware.
    - Clarify the limits of those workflows.

11. **Raspberry Pi / RTL-SDR Workflow**
    - Describe the intended install and service path on Raspberry Pi.
    - Include the main operational commands or control points a maintainer would expect.

## Evidence Rules

- Every major claim must be traceable to current repository files.
- Generated assets must be labeled as installer-generated or runtime-generated, not as committed repository files.
- Missing assets must be identified as absent or generated-only, not implied to exist elsewhere.
- The document must avoid speculative future-state architecture or rewrite recommendations.
- The document must preserve current SDRwatch terminology for scanner, controller, web, baseline, and runtime artifacts.

## Acceptance Conditions

- The completed document satisfies all functional requirements in `spec.md`.
- A maintainer can use it to identify authoritative entry points and schema ownership without source-code spelunking.
- A workstation contributor without SDR hardware can understand which workflows remain available.
- A Raspberry Pi maintainer can identify install-time and runtime-generated operational files.