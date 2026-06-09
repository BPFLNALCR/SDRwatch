<!--
Sync Impact Report
Version change: 1.0.0 -> 1.0.1
Modified principles:
- Clarified III. Stable Interfaces and Clean Layering with GUI-first operator workflow
- Clarified V. Migration-Safe, Verifiable Change with web UI/controller acceptance testing
Added sections:
- None
Removed sections:
- None
Templates requiring updates:
- .specify/templates/plan-template.md updated for GUI-first operator workflow
- .specify/templates/spec-template.md updated for web UI acceptance tests
- .specify/templates/tasks-template.md updated for controller/web validation tasks
Follow-up TODOs:
- None
-->
# SDRwatch Constitution

## Core Principles

### I. Raspberry Pi First Reliability
SDRwatch MUST preserve Raspberry Pi 5 compatibility, predictable long-running
operation, and clear field workflow before adding new capability. Changes MUST keep
the scanner, controller, and web experience usable on Raspberry Pi OS with local
storage, bounded resource use, and diagnosable failures. Rationale: the primary
deployment target is a Pi-based monitoring station, so workstation-only assumptions
are defects.

### II. Minimal Local Stack
New features MUST prefer maintainable Python, Flask, SQLite, and server-rendered
HTML. Heavy frontend frameworks, mandatory cloud services, internet-dependent
control paths, and speculative infrastructure MUST NOT be introduced unless the plan
documents a measured operational benefit and rejects a simpler local alternative.
Rationale: a small local stack is easier to operate offline, audit, and repair in the
field.

### III. Stable Interfaces and Clean Layering
Existing CLI scan behavior MUST remain backward-compatible unless a change includes
an explicit compatibility transition, upgrade notes, and verification coverage.
Scanner logic, persistence and schema ownership, controller API behavior, and web UI
concerns MUST remain separated so each layer can evolve without hidden coupling. The
web UI and controller job lifecycle are the operator workflow; the scanner CLI is
an internal backend interface, not the primary user interface.
Rationale: stable contracts and clean boundaries keep a hardware-facing system
diagnosable and reduce regression scope.

### IV. Adapter-Based Hardware and Honest RF Claims
Any new SDR support MUST be introduced through adapter-style driver interfaces so
RTL-SDR support remains stable as the default path. Features that classify, label, or
describe RF activity MUST distinguish measured facts from operator annotation or
model inference and MUST NOT claim emitter identity, intent, or certainty without
actual supporting data and provenance. Rationale: hardware expansion must not
destabilize the core platform, and RF interpretation must remain technically honest.

### V. Migration-Safe, Verifiable Change
Database changes MUST be migration-safe and backward-compatible when practical, with
upgrade behavior documented before merge. Every meaningful change MUST include
automated tests or a reproducible manual verification path, and security-sensitive
endpoints MUST verify `SDRWATCH_CONTROL_TOKEN` behavior whenever auth is in scope.
Operator-facing changes MUST be verified through the web UI and controller job
lifecycle; direct CLI checks only cover internal scanner backend behavior.
Rationale: stateful monitoring software stays trustworthy only when upgrades and
security behavior are explicit and repeatable.

## Operational Constraints

- Offline field use is a first-class requirement. Core scanning, SQLite persistence,
  controller operations, and the primary web workflow MUST continue without cloud
  services or internet connectivity.
- SDRwatch is GUI-operated for normal use and user acceptance testing. Human
  operators create/select baselines, start/stop scans, inspect logs/status, and
  review results through the web UI backed by controller jobs.
- The controller remains the only process that acquires device locks and spawns scan
  jobs. Web code MUST NOT touch SDR hardware directly, and controller/web layers MUST
  NOT reimplement scanner DSP logic.
- Database evolution MUST prefer additive schema changes, explicit defaults, and
  compatibility with existing SQLite files when practical. Destructive changes
  require a migration path, rollback guidance, and documented operator impact.
- Security-sensitive controller and web endpoints MUST honor
  `SDRWATCH_CONTROL_TOKEN` consistently for direct and proxied access paths when
  token auth is enabled.
- User-facing RF labels, alerts, and dashboards MUST present confidence or
  supporting evidence when inference is involved and MUST avoid language that implies
  certainty the data does not provide.

## Quality Gates

- CLI behavior: existing scan flags, defaults, exit behavior, JSONL output, and
  baseline selection semantics MUST remain stable for the internal scanner backend
  or ship with a documented transition and a reproducible compatibility check.
- Database migrations: schema changes MUST include migration or compatibility steps,
  validation against pre-existing SQLite data, and confirmation that critical reads
  and writes still succeed.
- Web/API behavior: changed routes and templates MUST preserve documented contracts
  or version them explicitly, validate auth behavior, and keep the operator workflow
  clear and complete in server-rendered views.
- Operator acceptance: user-facing features MUST be validated through the web UI
  and controller job lifecycle. CLI-only validation is sufficient only for
  explicitly internal scanner tooling.
- Service deployment: installer, environment, and service changes MUST document
  start, stop, lock cleanup, and logging behavior on the Raspberry Pi deployment
  path.
- Hardware-dependent features: changes touching drivers, sampling, tuning, or RF
  interpretation MUST include a real-device verification path or a reproducible
  capture workflow, and MUST confirm RTL-SDR remains stable.
- Offline field use: any affected workflow MUST be verified without assuming network
  access, remote APIs, or cloud-hosted state.

## Governance

This constitution overrides conflicting local guidance. Plans, specs, tasks,
reviews, and releases MUST check compliance with these principles and applicable
quality gates. When a change requires an exception, the implementation plan MUST
record the violation, why it is necessary, the rejected simpler alternative, and the
intended exit path.

Amendments MUST update this document and any affected templates or guidance files in
the same change. Versioning follows semantic rules: MAJOR for incompatible
governance changes or removed principles, MINOR for new principles or materially
expanded policy, and PATCH for clarifications that do not alter the governing
meaning.

Compliance review happens at planning time and again before merge or deployment.
Feature work is not complete until the documented tests or manual verification path
has been executed or explicitly handed off with reproducible steps. Operator-facing
features MUST include web UI/controller lifecycle verification, not only scanner CLI
commands.

The operational guidance in `.github/copilot-instructions.md`, `AGENTS.md`, and the
Spec Kit templates MUST remain subordinate to and consistent with this constitution.

**Version**: 1.0.1 | **Ratified**: 2026-06-09 | **Last Amended**: 2026-06-09
