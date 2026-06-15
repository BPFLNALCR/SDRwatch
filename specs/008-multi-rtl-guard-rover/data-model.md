# Data Model: Hardware-Aware Multi-RTL Guard/Rover Mode

## Hardware Inventory Entry

Represents one detected receiver or planned hardware class in the operator inventory.

**Fields**

- `device_identity`: stable identity key when available, such as `rtl:serial:<serial>`.
- `legacy_device_key`: current compatibility key, such as `rtl:0`.
- `device_kind`: `rtlsdr`, `airspy`, `hackrf`, `soapy`, or planned class name.
- `label`: operator-facing label.
- `runtime_index`: current detected index when applicable.
- `serial`: serial string when available.
- `identity_confidence`: `stable`, `ambiguous`, `index_only`, or `unknown`.
- `identity_scope`: `persistent`, `session`, or `none`.
- `warnings`: list of warning codes and messages.
- `detected`: boolean.
- `runnable`: boolean.
- `support_state`: `runnable`, `unsupported`, or `planned`.
- `runnable_backend`: `rtlsdr_native` when runnable, otherwise `null`.
- `backend_status`: machine-readable backend status.
- `busy`: boolean.
- `locked`: boolean.
- `lock_owner`: job or role-run identity when known.
- `active_job_id`: active child job ID when any.
- `assigned_role`: role name when assigned.
- `role_lane`: role lane when assigned.
- `assignment_id`: assignment identity when assigned.
- `last_seen_ts`: last controller discovery time.

**Validation Rules**

- `runnable=true` is allowed only for native RTL receiver inventory entries in this feature.
- Unique serial values produce `identity_confidence=stable` and `identity_scope=persistent`.
- Missing serials produce `identity_confidence=index_only` and session scope.
- Duplicate serials produce `identity_confidence=ambiguous` and must include a warning.
- Unsupported/planned hardware classes must not include a runnable backend.

## Capability Tier

Represents the station capability inferred from runnable RTL receiver count.

**Fields**

- `tier`: `0`, `1`, `2`, or `2_plus`.
- `label`: operator-facing tier label.
- `runnable_rtl_count`: count of runnable RTL receivers.
- `max_role_count`: number of role jobs recommended by the tier.
- `supported_roles`: roles available at this tier.
- `warnings`: tier-level warnings.
- `generated_ts`: inventory generation time.

**Rules**

- Tier 0: zero runnable RTLs.
- Tier 1: one runnable RTL, single-device scan and one GUARD.
- Tier 2: two runnable RTLs, GUARD plus ROVER.
- Tier 2+: three or more runnable RTLs, two GUARD roles plus REFERENCE or ROVER.

## Receiver Role Assignment

Represents manual binding of one receiver identity to one role lane.

**Fields**

- `assignment_id`: stable controller identity for the assignment record.
- `role`: `GUARD`, `ROVER`, or `REFERENCE`.
- `role_lane`: `guard_primary`, `guard_secondary`, `rover`, or `reference`.
- `display_name`: operator label such as "Friendly Guard" or "Watchlist Guard".
- `device_identity`: best available identity.
- `legacy_device_key`: runtime index key for compatibility.
- `serial`: serial when available.
- `runtime_index`: index at assignment time.
- `identity_confidence`: copied from inventory.
- `assignment_scope`: `persistent` or `session`.
- `task`: associated scan task summary when configured.
- `assigned_ts`: timestamp.
- `updated_ts`: timestamp.
- `assigned_by`: operator/session marker when available.
- `warnings`: warnings accepted by the operator.

**Validation Rules**

- A role lane can have at most one assignment.
- A physical receiver identity can be assigned to at most one active role lane when running.
- Persistent assignment requires `identity_confidence=stable`.
- Session assignment is allowed for index-only identity only with a warning and must not silently survive restart as durable truth.

## Role-Aware Run

Represents a grouped operator run that owns one or more child scan jobs.

**Fields**

- `role_run_id`: group identity.
- `status`: `pending`, `starting`, `running`, `degraded`, `stopping`, `finished`, `failed`, or `cancelled`.
- `capability_tier_at_start`: tier snapshot.
- `started_ts`: timestamp.
- `updated_ts`: timestamp.
- `finished_ts`: timestamp when terminal.
- `child_jobs`: list of child scan job references.
- `requested_roles`: roles requested for the group.
- `active_device_count`: number of distinct receivers assigned.
- `active_role_count`: number of active roles.
- `error`: group error message when any.
- `warnings`: group-level warnings.

**State Transitions**

- `pending` -> `starting` when child job starts begin.
- `starting` -> `running` when all required children are running.
- `starting` -> `degraded` when at least one child starts and another fails.
- `running` -> `degraded` when one child stops or fails unexpectedly.
- `running` or `degraded` -> `stopping` when the operator stops the group.
- `stopping` -> `finished` when all children are terminal cleanly.
- Any non-terminal state -> `failed` when no useful child job remains or startup cannot proceed.

## Child Scan Job

Extends the existing controller job with role-aware metadata.

**Fields**

- Existing job fields: `id`, `created_ts`, `label`, `device_key`, `baseline_id`, `status`, `pid`, `cmd`, `log_path`, `params`, `exit_code`, `finished_ts`.
- `role_run_id`: parent role-run ID when any.
- `receiver_role`: `GUARD`, `ROVER`, or `REFERENCE`.
- `role_lane`: role lane.
- `source_task`: task label such as `guard_window`, `rover_sweep`, or `reference_window`.
- `device_identity`: best available identity.
- `device_serial`: serial when available.
- `device_index`: runtime index.
- `identity_confidence`: identity confidence at start.
- `active_device_count`: role-run active device count at start.
- `active_role_count`: role-run active role count at start.
- `last_update_ts`: most recent controller status update.
- `error_message`: error string when any.

**Rules**

- Role-aware metadata must be additive and optional for existing jobs.
- Existing single-job workflows remain valid without role metadata.

## Scan Task

Represents the configured work for a receiver role.

**Fields**

- `task_id`: identity within a role run.
- `source_task`: `guard_window`, `rover_sweep`, or `reference_window`.
- `profile`: scanner profile name when used.
- `start_hz`: start frequency.
- `stop_hz`: stop frequency.
- `center_hz`: center frequency when parked.
- `step_hz`: step size.
- `sample_rate`: requested sample rate.
- `fft`: FFT size.
- `avg`: averaging count.
- `persistence`: persistence settings summary.
- `diagnostics_enabled`: boolean.

**Rules**

- GUARD and REFERENCE tasks should be narrow enough to behave as parked windows.
- ROVER tasks may span broader configured ranges.
- Task parameters must not silently change FM Validation defaults.

## Telemetry Record

Represents diagnostic evidence emitted by scanner or controller.

**Fields**

- `event`: diagnostic event name.
- `timestamp`: event timestamp.
- `role_run_id`: role-run ID when any.
- `job_id`: child job ID when any.
- `receiver_role`: role name.
- `role_lane`: role lane.
- `device_identity`: best available identity.
- `device_key`: compatibility key.
- `device_serial`: serial when available.
- `device_index`: runtime index.
- `device_kind`: hardware kind.
- `backend`: runnable backend.
- `source_profile`: profile when any.
- `source_task`: task label.
- `center_hz`: center frequency or window center.
- `sample_rate`: sample rate.
- `fft`: FFT size.
- `avg`: averaging count.
- `num_segments`: detected segment count.
- `samples_requested`: requested samples.
- `samples_read`: actual samples read.
- `short_read`: boolean or null.
- `dropped_reads`: count or null.
- `tune_ms`, `flush_ms`, `read_ms`, `fft_ms`, `detect_ms`, `db_update_ms`, `jsonl_ms`, `total_window_ms`: timing values or null.
- `pid`: process ID.
- `active_device_count`: active receiver count.
- `active_role_count`: active role count.
- `cpu_load`: CPU load when available.
- `rss_memory_bytes`: resident memory when available.
- `unavailable_fields`: list of fields unavailable on the platform.

## Detection Provenance

Represents device/role/job context attached to detection or scan-update evidence.

**Fields**

- `receiver_role`
- `role_lane`
- `device_identity`
- `device_key`
- `device_serial`
- `device_index`
- `job_id`
- `role_run_id`
- `source_profile`
- `source_task`
- `observed_ts`

**Durability**

- Required in diagnostic records for the first implementation.
- Preferred in scan updates through nullable additive fields.
- Optional/deferred for baseline detections unless implementation shows a low-risk last-seen provenance path.

## Unsupported Hardware Class

Represents hardware that may be detected or documented but is not runnable in this feature.

**Fields**

- `hardware_kind`: `airspy`, `hackrf`, `soapy`, or other planned class.
- `detected`: boolean.
- `support_state`: `unsupported` or `planned`.
- `runnable`: false.
- `runnable_backend`: null.
- `message`: operator-facing explanation.
- `docs_url`: local documentation reference when available.

**Rules**

- Unsupported hardware classes must not appear in runnable start choices.
- Any start attempt must fail before process spawn.
