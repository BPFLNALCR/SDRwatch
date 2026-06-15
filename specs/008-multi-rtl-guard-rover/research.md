# Phase 0 Research: Hardware-Aware Multi-RTL Guard/Rover Mode

## Decision: Keep the Feature RTL-Only and Native-Runner-Only

**Rationale**: The current scanner runner and CLI accept `rtlsdr_native` as the only runnable backend. The first multi-receiver feature should make that truth explicit instead of exposing planned hardware as runnable. This preserves working FM behavior and avoids introducing untested hardware paths.

**Alternatives considered**:

- Enable HackRF, Airspy, or Soapy because docs mention them: rejected because actual runner support is not present for this feature.
- Introduce a hardware abstraction first: rejected as too broad and likely to delay useful multi-RTL operation.

## Decision: Use Serial-First Receiver Identity with Index Fallback Warnings

**Rationale**: RTL device indexes can change across unplug, reboot, or enumeration order. Unique serial values are the best available stable identity. Missing or duplicate serials must be visible to the operator and must not be treated as durable identity.

**Alternatives considered**:

- Continue using `rtl:<index>` as the primary identity: rejected because it cannot reliably distinguish multiple RTLs over time.
- Require every RTL to have a unique serial before use: rejected for first implementation because index-only operation is still useful when clearly session-scoped and warned.

## Decision: Store Role Assignments in Controller State First

**Rationale**: The controller already owns device discovery, lock files, job state, and scanner process lifecycle. Role assignment is control-plane state, not RF detection evidence. Controller state can support stable serial assignments, session-scoped index assignments, and startup reconciliation without a broad database migration.

**Alternatives considered**:

- Store role assignments only in browser state: rejected because the controller must enforce duplicate assignment and job locks.
- Add a new database table first: deferred because existing controller JSON state is a smaller additive path.

## Decision: Introduce Role-Run Grouping as an Additive Orchestration Layer

**Rationale**: GUARD+ROVER and two-GUARD+REFERENCE runs are one operator action with multiple child jobs. A group ID gives the UI and diagnostics a stable way to show overall health, stop all children, and report degraded status. Individual child jobs remain normal jobs for compatibility.

**Alternatives considered**:

- Represent grouping only through job metadata: rejected because stop/status workflows become ambiguous when multiple child jobs must be handled as one operator run.
- Replace the job lifecycle with a scheduler: rejected as out of scope.

## Decision: Harden Device Locks Before Multi-Job Starts

**Rationale**: The existing file lock behavior is adequate for single-device workflows but can race under concurrent same-device starts. Multi-RTL role operation depends on preventing duplicate assignment of the same physical receiver.

**Alternatives considered**:

- Rely on UI disable states only: rejected because concurrent HTTP requests or multiple browser sessions can bypass UI timing.
- Defer locking hardening: rejected because it is prerequisite safety for grouped starts.

## Decision: Map GUARD and REFERENCE to Narrow Existing Scanner Runs

**Rationale**: The existing scanner loop already supports configured frequency ranges, repeated loops, detection, baselines, cross-sweep persistence, and diagnostics. A GUARD or REFERENCE role can be represented as a narrow parked task without rewriting DSP or scheduler logic.

**Alternatives considered**:

- Build a new fixed-frequency scanner path: deferred because it risks duplication and regression.
- Implement automatic scheduler optimization now: rejected as a non-goal.

## Decision: Preserve ROVER as Existing Sequential Sweep

**Rationale**: The rover role is naturally the current sequential tune/read/FFT/detect loop over lower-priority ranges. Reusing it preserves current detection behavior and lets this feature focus on hardware awareness and orchestration.

**Alternatives considered**:

- Implement a new opportunistic scheduler: rejected as a later optimization after telemetry exists.

## Decision: Make Provenance Diagnostic-First, with Minimal Durable Columns

**Rationale**: Diagnostic JSONL is already the richest evidence path and can record per-window fields without reshaping baseline persistence. Durable provenance is still useful for scan updates and job records, but full detection observation/fusion tables are deferred.

**Alternatives considered**:

- Add full signal track and observation tables now: rejected as a major schema rewrite and outside the first implementation.
- Store no durable provenance: rejected because future debugging and benchmark comparisons need at least job/run/device context.

## Decision: Add Per-Window Timing and Resource Telemetry Before Optimization

**Rationale**: Pi 5 multi-RTL performance risk spans USB, CPU, memory, and file I/O. Timing/resource fields let maintainers identify bottlenecks before adding scheduler or DSP optimization.

**Alternatives considered**:

- Optimize sample rates or DSP first: rejected because bottlenecks are not yet measured.
- Record raw IQ from every receiver: rejected due CPU/USB/I/O risk and explicit feature boundary.

## Decision: Add Minimal UI Surfaces Instead of a Broad Redesign

**Rationale**: The current control page is already the operator entrypoint. A compact inventory/tier/role-run section can support the feature while preserving Discovery, FM Validation, diagnostics mode, and current job start workflows.

**Alternatives considered**:

- Build a separate multi-receiver dashboard: deferred as too broad for the first implementation.
- Keep multi-RTL control API-only: rejected because operator-facing features must be usable through the web UI.

## Decision: Correct Documentation Drift as Part of the Feature

**Rationale**: The current README can be read as claiming Soapy scanner execution is available. This feature depends on honest backend capability reporting, so docs must match runtime support.

**Alternatives considered**:

- Leave docs until non-RTL support is added: rejected because it would keep misleading operator expectations in place.
