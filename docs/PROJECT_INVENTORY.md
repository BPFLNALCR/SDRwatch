# SDRwatch Project Inventory

Last verified: 2026-06-08

## Purpose and Scope

This document is a current-state inventory of the SDRwatch repository for contributors and maintainers. It is intentionally descriptive rather than aspirational: it explains what is in the repository today, which entry points are authoritative, which assets are generated later by the installer or at runtime, and where the main operational responsibilities currently live.

This document does not propose an application rewrite. It is meant to reduce repository spelunking before making changes.

## Top-Level Repository Layout

| Path | Purpose | Notes |
| --- | --- | --- |
| `sdrwatch/` | Core scanner package | Scanner CLI, sweep orchestration, drivers, DSP, baseline persistence, and utility helpers. |
| `sdrwatch_web/` | Flask web package | App factory, controller client, SQLite read helpers, startup migrations, blueprints, and web-specific helpers. |
| `templates/` | Server-rendered HTML templates | Dashboard, control, changes, signals, debug, and partial templates. |
| `static/js/` | Frontend JavaScript assets | Dashboard and changes page behavior plus bundled helper scripts. |
| `tests/` | Automated test suite | Currently focused on DSP and detection helpers rather than full controller or web integration. |
| `bandplan_eu.csv`, `bandplan_us_na.csv` | Bandplan data | Frequency allocation reference data loaded by the scanner or helper code. |
| `sdrwatch-control.py` | Controller entry point | Long-running job manager and REST API server. |
| `sdrwatch-web.py` | Web entry point | Thin CLI wrapper around `sdrwatch_web.create_app()`. |
| `sdrwatch.py` | Legacy scanner wrapper | Deprecated compatibility shim that forwards to `python -m sdrwatch.cli`. |
| `query-sdrwatch.py` | Database inspection helper | Useful for inspection, but it targets older `scans` / `detections` / `baseline` tables rather than the current baseline-first schema. |
| `install-sdrwatch.sh` | Raspberry Pi installer | Installs packages, creates a venv, writes env and systemd files, and optionally deploys code. |
| `uninstall-sdrwatch.sh` | Uninstall helper | Removes generated deployment artifacts while intentionally preserving state. |
| `.github/workflows/ci.yml` | CI workflow | Lint, type-check, and smoke-test pipeline. |
| `specs/`, `.specify/`, `AGENTS.md`, `.github/copilot-instructions.md` | Planning and agent guidance | Contributor and Spec Kit workflow assets; not part of the runtime stack. |

Other root scripts such as `remove_sections.py` are ad hoc repository helpers rather than part of the deployed control, web, or scanner runtime.

## Canonical Entry Points and Compatibility Paths

| Component | Preferred invocation | Status | Notes |
| --- | --- | --- | --- |
| Scanner CLI | `python -m sdrwatch.cli` | Canonical | Parses flags, enforces `--baseline-id`, and dispatches to `sdrwatch.sweep.runner.run_scan()`. |
| Scanner wrapper | `python sdrwatch.py` | Compatibility shim | Emits a deprecation warning and forwards to the package CLI. |
| Controller / API | `python sdrwatch-control.py serve --host 127.0.0.1 --port 8765 --token <token>` | Canonical | Discovers devices, builds scan commands, owns lock files, manages jobs, and serves control-plane endpoints. |
| Web dashboard | `python sdrwatch-web.py --db sdrwatch.db --host 0.0.0.0 --port 8080` | Canonical | Thin Flask launcher that expects a SQLite path and optional controller environment variables. |
| Legacy web names | `sdrwatch-web-simple.py`, `sdrwatch_web_simple.py` | Compatibility-only | Referenced by the installer and CI as optional backward-compatible names; not committed as primary entry points in the repository. |
| Query helper | `python query-sdrwatch.py ...` | Compatibility / inspection helper | Useful for local database inspection, but its SQL still targets the older `scans`, `detections`, and `baseline` tables. |

## Runtime Topology

```text
operator
  |
  +--> sdrwatch-web.py / sdrwatch_web/
  |       |
  |       +--> reads SQLite directly (mostly read-only after startup migration)
  |       +--> proxies control-plane actions to the controller over HTTP
  |
  +--> sdrwatch-control.py
          |
          +--> discovers SDR devices
          +--> owns state.json, lock files, and per-job logs
          +--> spawns scanner jobs
                  |
                  +--> python -m sdrwatch.cli
                          |
                          +--> sdrwatch.sweep.runner
                                  |
                                  +--> drivers + DSP + baseline persistence
                                  +--> SQLite + optional JSONL output
```

Current layer boundaries are:

- The scanner package in `sdrwatch/` owns signal capture, DSP, baseline updates, and write-side persistence.
- The controller in `sdrwatch-control.py` is the only process that should acquire device locks and spawn scanner jobs.
- The web app in `sdrwatch_web/` should not talk to SDR hardware directly. It renders pages, proxies control actions to the controller, and reads SQLite data directly.
- `query-sdrwatch.py` bypasses the controller and web app entirely and talks to SQLite directly, but it should be treated as a helper rather than part of the baseline-first control path.

## Main Python Packages and Scripts

### Core scanner package: `sdrwatch/`

| Path | Responsibility |
| --- | --- |
| `sdrwatch/cli.py` | Canonical scanner CLI argument parsing, profile listing, and dispatch to the sweep runner. |
| `sdrwatch/sweep/` | Scheduling and orchestration: converts scan parameters to windows and runs the sweep loop. |
| `sdrwatch/drivers/` | SDR hardware adapters, including native RTL-SDR and SoapySDR support. |
| `sdrwatch/dsp/` | Pure DSP helpers such as FFT, CFAR, clustering, noise estimation, and power monitoring. |
| `sdrwatch/detection/` | Detection engine and types that translate DSP output into persisted detections and revisit behavior. |
| `sdrwatch/baseline/` | Baseline context, persistence, band summaries, spur handling, and the core SQLite schema store. |
| `sdrwatch/io/` | Bandplan loading and built-in scan profiles. |
| `sdrwatch/util/` | Logging, exit codes, JSONL scan logging, duration parsing, timing, and diagnostics. |

### Web package: `sdrwatch_web/`

| Path | Responsibility |
| --- | --- |
| `sdrwatch_web/app.py` | Flask application factory and blueprint registration. |
| `sdrwatch_web/controller.py` | HTTP client used by the web layer to talk to `sdrwatch-control.py`. |
| `sdrwatch_web/db.py` | Startup migration runner plus ongoing read-only SQLite access helpers. |
| `sdrwatch_web/schema.py` | Web-owned schema helpers for classification columns, monitoring zones, and friendly signals. |
| `sdrwatch_web/blueprints/views.py` | Server-rendered HTML pages such as dashboard, control, live view, changes, signals, debug, and spur map. |
| `sdrwatch_web/blueprints/api_jobs.py` | `/api/jobs`, `/api/scans`, `/api/logs`, and active-job helpers. |
| `sdrwatch_web/blueprints/api_baselines.py` | `/api/baselines` endpoints and baseline proxy helpers. |
| `sdrwatch_web/blueprints/api_signals.py` | Signal CRUD and selection APIs. |
| `sdrwatch_web/blueprints/api_zones.py` | Monitoring-zone and friendly-signal APIs. |
| `sdrwatch_web/blueprints/api_debug.py` | Debug and health endpoints such as `/api/debug/health` and `/api/debug/db-stats`. |
| `sdrwatch_web/blueprints/ctl.py` | `/ctl/*` controller proxy routes for device and discovery debugging. |

### Templates and static assets

- `templates/base.html` is the shared shell for server-rendered pages.
- `templates/dashboard.html`, `control.html`, `live.html`, `changes.html`, `signals.html`, `signal.html`, `spur_map.html`, and `debug.html` are the main web views.
- `static/js/dashboard.js` and `static/js/changes.js` hold most page-specific frontend behavior.

## Web Dashboard and API Components

The repository exposes two API layers:

1. Controller REST endpoints from `sdrwatch-control.py`
   - `GET /devices`
   - `GET /jobs`
   - `POST /jobs`
   - `GET /jobs/<id>`
   - `GET /jobs/<id>/logs`
   - `DELETE /jobs/<id>`
   - `GET /profiles`
   - `GET /baselines`, `POST /baselines`, `PATCH /baselines/<id>`

2. Web-layer JSON endpoints from `sdrwatch_web/blueprints/`
   - `/api/jobs`, `/api/scans`, `/api/logs`
   - `/api/baselines`
   - `/api/signals`
   - `/api/baseline/<id>/zones`, `/api/zones/*`
   - `/api/baseline/<id>/friendly`, `/api/friendly/*`
   - `/api/debug/*`
   - `/ctl/*` controller proxy helpers

Authentication is environment-driven. `sdrwatch-web.py` documents `SDRWATCH_TOKEN` for protecting web API routes and `SDRWATCH_CONTROL_URL` / `SDRWATCH_CONTROL_TOKEN` for controller proxying.

## Database and Schema Ownership

The repository does not have a standalone migrations directory or schema tool such as Alembic. Schema responsibility is currently split between the scanner package and the web package.

### Core baseline-first schema owner

`sdrwatch/baseline/store.py` is the main owner of the baseline-first SQLite schema. It creates or maintains the core tables used by the scanner and baseline workflow, including:

- `baselines`
- `baseline_noise`
- `baseline_occupancy`
- `baseline_detections`
- `scan_updates`
- `spur_map`
- `baseline_band_summary`
- `baseline_summary_meta`
- `baseline_snapshot`

It also applies additive column updates with `_ensure_column()` when newer columns are missing.

### Web startup migration path

`sdrwatch_web/db.py` performs startup migration work if the database file already exists:

- It opens a writable SQLite connection during startup.
- It calls `sdrwatch_web.schema.migrate_detection_classification()` to add classification and annotation columns to `baseline_detections` if needed.
- It calls `sdrwatch_web.schema.ensure_monitoring_zones_schema()` to create `monitoring_zones` and `friendly_signals`.
- After startup work, ongoing web reads use a read-only SQLite connection.

### Important caveat

`query-sdrwatch.py` still queries `scans`, `detections`, and `baseline`, which means not every helper in the repo targets the same schema shape. Treat the baseline-first schema above as the authoritative model for current scanner and web work.

## Installer, Service, and Deployment Behavior

### `install-sdrwatch.sh`

The installer is the authoritative source for the Raspberry Pi deployment path. Its main responsibilities are:

- Install APT packages for Python, RTL-SDR, and build dependencies.
- Create a Python virtual environment with `--system-site-packages`.
- Install lightweight Python packages with pip and generate `requirements.sdrwatch.txt` in the project or deployed directory.
- Optionally build `rtl-sdr` from source into `.build-rtl-sdr/` if the packaged `rtl_test` path is not working.
- Apply RTL-SDR udev rules and kernel-module blacklist entries.
- Create or populate state, cache, and runtime directories.
- Optionally deploy the repository to `/opt/sdrwatch/`.
- Generate `/etc/sdrwatch.env`.
- Generate systemd units at `/etc/systemd/system/sdrwatch-control.service` and `/etc/systemd/system/sdrwatch-web.service`.
- Optionally create compatibility symlinks for the old web entrypoint names.

### `uninstall-sdrwatch.sh`

The uninstall script stops and removes generated service artifacts, env files, deployed code, and runtime directories, but it intentionally preserves state and database files under the configured state directory unless the operator removes them manually.

### Committed vs generated service files

The repository does not commit service unit files. The installer writes them during deployment, which means the installer script itself is the source of truth for service definitions.

## Runtime and Generated Artifacts

| Artifact | Default location(s) | Created by | Notes |
| --- | --- | --- | --- |
| SQLite database | `sdrwatch.db` by CLI default; installer default `/var/lib/sdrwatch/sdrwatch.db` | Scanner, web startup helpers, or installer-configured services | Actual path depends on CLI flags or `SDRWATCH_DB`. |
| Detection JSONL output | Operator-chosen `--jsonl` path; README examples use `events.jsonl` | Scanner CLI | No fixed filename is enforced; `events.jsonl` is a common example rather than a hardcoded default. |
| Diagnostic JSONL output | Operator-chosen `--diagnostic-jsonl` path | Scanner CLI | Optional per-window diagnostics output. |
| Controller state file | `${SDRWATCH_CONTROL_BASE}/state.json` | Controller | Manual default base is `/tmp/sdrwatch-control`; installer/systemd commonly uses `/run/sdrwatch-control`. |
| Controller lock files | `${SDRWATCH_CONTROL_BASE}/locks/*.lock` | Controller | Prevents simultaneous use of the same SDR device. |
| Controller job logs | `${SDRWATCH_CONTROL_BASE}/logs/*.log` | Controller | Per-job scanner stdout/stderr capture. |
| Generated env file | `/etc/sdrwatch.env` | Installer | Stores service paths, ports, tokens, and runtime directory settings. |
| Generated systemd units | `/etc/systemd/system/sdrwatch-control.service`, `/etc/systemd/system/sdrwatch-web.service` | Installer | Not committed in the repository. |
| Deployed code copy | `/opt/sdrwatch/` | Installer | Optional deployed copy used by generated services. |
| Service runtime directories | `/run/sdrwatch/`, `/run/sdrwatch-control/` | Installer + systemd | Used by generated services for runtime state. |
| Service state and cache | `/var/lib/sdrwatch/`, `/var/cache/sdrwatch/` | Installer + systemd | Persistent DB/state and cache locations for installed services. |
| Generated helper requirements file | `requirements.sdrwatch.txt` | Installer | Generated on install; not committed source-of-truth dependency metadata. |

## Tests and CI Surface

### Current test layout

| Path | Coverage |
| --- | --- |
| `tests/test_detection_diagnostics.py` | Detection diagnostics shape and expected fields. |
| `tests/test_extent_hysteresis.py` | Extent hysteresis behavior. |
| `tests/test_segment_splitting.py` | Segment splitting and nearby-peak behavior. |

These tests are focused on DSP and detection logic. The repository does not currently include committed controller integration tests, web endpoint integration tests, or hardware-simulation tests.

### CI workflow

`.github/workflows/ci.yml` currently:

- Installs Python 3.11 and SDR-related system packages.
- Installs runtime dependencies directly with pip rather than from a committed requirements or packaging file.
- Runs `flake8` and `mypy`, but both steps are `continue-on-error`.
- Runs smoke tests that import key libraries and exercise `--help` paths for root scripts.
- Does not run the `tests/` suite with `pytest`.

## Known Missing Standard Project Files and Contributor Gaps

As of this document's verification date, the repository still lacks several standard contributor-facing files or committed deployment assets:

- No `pyproject.toml`
- No committed `requirements.txt`, `requirements-dev.txt`, or equivalent lock file
- No `setup.py` or `setup.cfg`
- No versioned `migrations/` directory or formal migration framework
- No committed systemd unit files; units are generated by the installer
- No Dockerfile or container deployment manifest

There is also a tooling mismatch worth knowing about:

- `query-sdrwatch.py` still expects older `scans` / `detections` / `baseline` tables, while the current scanner and web stack are baseline-first.

## Local No-Hardware Workflow

The following activities are practical on a workstation without SDR hardware:

1. Inspect canonical entry points and CLI surfaces:
   - `python -m sdrwatch.cli --help`
   - `python -m sdrwatch.cli --list-profiles`
   - `python sdrwatch-control.py --help`
   - `python sdrwatch-web.py --help`
   - `python query-sdrwatch.py --help`

2. Run the committed DSP-focused tests in `tests/`.

3. Inspect an existing SQLite file with the web app or helper scripts.
   - The web app can be launched against an existing DB path.
   - If the DB is missing or incomplete, the web package is designed to surface a waiting or unavailable state rather than pretending the data exists.

4. Review the controller and web APIs from source and README documentation without touching hardware.

Current workstation limitations without hardware are:

- No live SDR device discovery
- No real scan execution
- No meaningful controller job start without an attached device and valid baseline path
- No end-to-end validation of the Raspberry Pi install path

## Raspberry Pi with RTL-SDR Workflow

The intended hardware deployment path is Raspberry Pi OS on a Pi 5 with RTL-SDR as the default device.

### Typical install path

```bash
git clone https://github.com/SDRwatch/sdr-watch.git
cd sdr-watch
chmod +x install-sdrwatch.sh
SDRWATCH_AUTO_YES=1 ./install-sdrwatch.sh
```

What that gives you:

- A prepared Python environment
- Generated `/etc/sdrwatch.env`
- Generated systemd services for controller and web
- State and runtime directories under `/var/lib/sdrwatch`, `/var/cache/sdrwatch`, `/run/sdrwatch`, and `/run/sdrwatch-control`

### Service-oriented operation

```bash
sudo systemctl status sdrwatch-control
sudo systemctl status sdrwatch-web
sudo journalctl -u sdrwatch-control -f
sudo journalctl -u sdrwatch-web -f
```

Default ports used by the generated services are:

- Controller: `127.0.0.1:8765`
- Web UI: `0.0.0.0:8080`

### Manual scanner invocation

For one-off scans, the canonical scanner CLI remains:

```bash
python -m sdrwatch.cli --baseline-id 3 --start 88e6 --stop 108e6 --step 1.8e6 \
  --samp-rate 2.4e6 --fft 4096 --avg 8 --driver rtlsdr --gain auto
```

The controller is expected to be the normal long-running owner of scan process spawning and device locks. Manual scanner runs are best treated as operator or debugging workflows.

## Practical Contributor Notes

- If you need to know which file owns a runtime responsibility, start with `sdrwatch/cli.py`, `sdrwatch-control.py`, `sdrwatch-web.py`, `sdrwatch/baseline/store.py`, `sdrwatch_web/db.py`, and `sdrwatch_web/schema.py`.
- If you need to know whether a file should already exist in a fresh clone, check whether it is committed here or generated later by `install-sdrwatch.sh` or by the running controller.
- If you are documenting or fixing schema behavior, treat the baseline-first tables as authoritative and call out any helpers that still assume the older schema.