# SDRwatch Project Inventory

Last verified: 2026-06-09

## Purpose

This document records the current repository shape so contributors can understand
the codebase without continuing feature development. It is descriptive, not a
rewrite proposal.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `sdrwatch/` | Core scanner package: CLI, sweep orchestration, drivers, DSP, detection, baseline persistence, and utilities. |
| `sdrwatch_web/` | Flask web package: app factory, controller client, SQLite helpers, startup migrations, blueprints, filters, and formatting. |
| `templates/` | Server-rendered HTML templates for dashboard, control, changes, signals, debug, and partial views. |
| `static/js/` | Browser-side behavior for dashboard and changes pages plus bundled helper scripts. |
| `tests/` | Automated tests currently focused on DSP and detection diagnostics behavior. |
| `.specify/` | Spec Kit configuration, templates, scripts, integrations, and workflow metadata. |
| `.agents/skills/` | Codex Spec Kit skills. |
| `.github/agents/`, `.github/prompts/` | Copilot Spec Kit agent and prompt files. |
| `.github/copilot-instructions.md` | Operational guidance for Copilot and other AI coding assistants. |
| `AGENTS.md` | Codex-facing repository guidance. |
| `install-sdrwatch.sh` | Raspberry Pi installation and service-generation script. |
| `sdrwatch.py` | Legacy scanner wrapper that forwards to the package CLI. |
| `sdrwatch-control.py` | Controller/API process that discovers devices, owns locks, and spawns scanner jobs. |
| `sdrwatch-web.py` | Web entry point for the Flask dashboard. |

## Canonical Runtime Entry Points

| Component | Preferred invocation | Notes |
| --- | --- | --- |
| Scanner CLI | `python -m sdrwatch.cli` | Authoritative scanner command surface. |
| Controller | `python sdrwatch-control.py serve --host 127.0.0.1 --port 8765 --token <token>` | Owns device discovery, lock files, job lifecycle, and controller REST endpoints. |
| Web dashboard | `python sdrwatch-web.py --db sdrwatch.db --host 0.0.0.0 --port 8080` | Server-rendered Flask app that reads SQLite and proxies control-plane actions. |
| Legacy scanner wrapper | `python sdrwatch.py` | Compatibility shim only. |
| Query helper | `python query-sdrwatch.py ...` | Inspection helper; some queries target older table names. |

## Runtime Topology

```text
operator
  |
  +--> sdrwatch-web.py / sdrwatch_web/
  |       +--> reads SQLite for dashboard data
  |       +--> proxies control actions to controller HTTP endpoints
  |
  +--> sdrwatch-control.py
          +--> discovers SDR devices
          +--> owns state.json, lock files, and job logs
          +--> spawns scanner jobs
                  +--> python -m sdrwatch.cli
                          +--> sdrwatch.sweep.runner
                                  +--> drivers + DSP + detection
                                  +--> baseline SQLite writes
                                  +--> optional JSONL diagnostics
```

Layer boundaries to preserve:

- Scanner code owns SDR capture, DSP, detection, baseline updates, and scanner-side
  persistence.
- Controller code is the only layer that should acquire device locks and spawn scan
  processes.
- Web code should not touch SDR hardware or reimplement DSP logic.
- SQLite schema ownership is split between scanner baseline helpers and web startup
  migration helpers.

## Database Ownership

`sdrwatch/baseline/store.py` owns the baseline-first scanner schema, including:

- `baselines`
- `baseline_noise`
- `baseline_occupancy`
- `baseline_detections`
- `scan_updates`
- `spur_map`
- `baseline_band_summary`
- `baseline_summary_meta`
- `baseline_snapshot`

`sdrwatch_web/db.py` and `sdrwatch_web/schema.py` apply web startup migration work,
including detection classification columns, monitoring zones, and friendly signals.

Important caveat: `query-sdrwatch.py` still references older `scans`, `detections`,
and `baseline` table names in places. Treat the baseline-first schema as
authoritative for current scanner and web work.

## Spec Kit and AI Tooling State

The repository has both Codex and Copilot Spec Kit assets:

- Codex skills are present under `.agents/skills/`.
- Copilot agents and prompts are present under `.github/agents/` and
  `.github/prompts/`.
- `.specify/integration.json` currently identifies Codex as the default integration.
- `.github/copilot-instructions.md` remains the main operational guide for Copilot
  and general AI coding assistants.
- `AGENTS.md` remains the Codex entry point and should point to the active feature
  plan or consolidation docs.

Current consolidation recommendation: keep Codex as the default Spec Kit integration,
preserve Copilot usability, and review numbered Spec Kit branches against
`devControl` before merging.

## Tests and Checks Surface

Committed tests on `devControl` include:

- `tests/test_detection_diagnostics.py`
- `tests/test_extent_hysteresis.py`
- `tests/test_segment_splitting.py`

Useful no-hardware checks:

```bash
python -m pytest -q
python -m sdrwatch.cli --list-profiles
python -c "from sdrwatch_web import create_app; app = create_app(); print(app.name)"
```

Hardware-dependent confidence still requires Raspberry Pi and RTL-SDR validation for
real scan execution, device locking, controller job lifecycle, and installer/service
behavior.

## Generated or Runtime Artifacts

| Artifact | Typical location | Created by |
| --- | --- | --- |
| SQLite database | `sdrwatch.db` or installer-configured state path | Scanner/web/installer |
| Detection JSONL | Operator-selected `--jsonl` path | Scanner CLI |
| Diagnostic JSONL | Operator-selected `--diagnostic-jsonl` path | Scanner CLI |
| Controller state | `${SDRWATCH_CONTROL_BASE}/state.json` | Controller |
| Controller lock files | `${SDRWATCH_CONTROL_BASE}/locks/*.lock` | Controller |
| Controller job logs | `${SDRWATCH_CONTROL_BASE}/logs/*.log` | Controller |
| `/etc/sdrwatch.env` | System path | Installer |
| systemd service files | `/etc/systemd/system/` | Installer |

## Known Gaps

- No committed `pyproject.toml` on `devControl`.
- No committed requirements lock file.
- No formal migration framework such as Alembic.
- CI does not yet provide full hardware or controller/web integration coverage.
- Some helper documentation and scripts still mention compatibility names or older
  schema concepts.
