# SDRwatch — AI coding assistant guide (for Copilot/ChatGPT)

Keep guidance short and operational. Prefer **small diffs** over whole-file rewrites. Prioritize **runtime topology**, **CLI↔service wiring**, **DB schema & migrations**, **device locking**, and **tests**.

---

## Layered runtime model (scanner ↔ controller ↔ web)

* **Scanner (`sdrwatch.py`)** owns SDR/DSP loops, detection heuristics, spur calibration, and all SQLite writes (`scans`, `detections`, `baseline`, `spur_map`). It now exports built-in profiles via `--list-profiles`.
* **Controller (`sdrwatch-control.py`)** is the only process that spawns scanners. It accepts structured scan requests (`device_key`, `label`, `params`), builds CLI args, enforces device locks, monitors child processes, and exposes metadata like `/profiles` by shelling out `sdrwatch.py --list-profiles`.
* **Web (`sdrwatch-web-simple.py`)** never touches SDR hardware or IQ streams. It only talks to the controller’s HTTP API (start/stop/list/logs/profiles) and reads SQLite directly for dashboards. Do not duplicate DSP logic outside the scanner.

Treat the scanner’s DB schema + CLI behavior as authoritative. Controller/web layers should remain thin adapters around those contracts.

---

## 1) Big picture (runtime topology)

**Authoritative script purposes (use these names in prompts/commits):**

* **`sdrwatch.py` ("scanner")** — CLI tool that performs wideband sweeps on an SDR device, computes PSD/CFAR, updates the **SQLite DB** (`scans`, `detections`, `baseline`, `spur_map`), and can emit JSONL and desktop/CLI notifications. Single-process, device-bound. Includes scan profiles, anomalous-window gating, spur calibration/suppression, and a DetectionEngine that promotes persistent hits with confidence scores.

* **`sdrwatch-control.py` ("controller" / "manager")** — Long‑lived job manager. Discovers devices (serial/index/driver), enforces **file‑lock ownership** per device under `${SDRWATCH_CONTROL_BASE}/locks`, constructs the scanner command (`_build_cmd`), spawns and monitors scanner subprocesses, exposes an **HTTP API** for start/stop/list/health, and reaps stale locks.

* **`sdrwatch-web-simple.py` ("web" / "UI")** — Flask front‑end. **Read‑only** queries to the SQLite DB (tables above). Proxies **control actions** to the controller via RESTful endpoints (`/api/jobs`, `/api/jobs/<id>`, `/api/jobs/<id>/logs`, etc.) pointing at `SDRWATCH_CONTROL_URL` with bearer auth. No direct device access. Treat the web layer as a thin REST client; avoid reintroducing stateful coupling.

* **Installer script (repo root, e.g., `install-sdrwatch.sh`)** — Provisioning helper for Raspberry Pi OS **Trixie**: installs apt deps (SoapySDR, plugins, Python build reqs), creates Python venv, installs Python deps, sets up directories (`/opt/sdrwatch`, logs, state), optionally installs/updates the systemd unit and env file.

* **Bandplan CSVs (`bandplan_us_na.csv`, `bandplaneu.csv`)** — Data inputs loaded by the scanner to label detections. Header contract: `low_hz,high_hz,service,region,notes` (Hz units). Prefer adding rows over renaming headers.

* **Systemd unit (e.g., `sdrwatch.service`)** — Runs the controller or a configured scanner as a service on the Pi. Holds environment in `/etc/sdrwatch/env`. Logging via `journalctl -u sdrwatch`.

* **Migrations folder (`migrations/`, optional)** — Idempotent SQL scripts plus a `schema_version` table to evolve DB schema when needed.

**Colloquial → file mapping:**

* *scanner* → `sdrwatch.py`
* *controller/manager/daemon* → `sdrwatch-control.py`
* *web/UI* → `sdrwatch-web-simple.py`
* *installer* → `install-sdrwatch.sh` (exact name may differ; see repo root)

**State layout** (overridable via env):

* `SDRWATCH_CONTROL_BASE` (default `/tmp/sdrwatch-control/`) → `locks/`, `logs/`, and `state.json` at the base.
* Default DB file: `sdrwatch.db` (overridable with `--db`).

**Goal for contributors**: make changes safe for long‑running service on Raspberry Pi OS **Trixie** + Pi 5 with RTL‑SDR by default, optional HackRF/SoapySDR.

### ScanRequest JSON (web → controller)

* Web posts to controller `/jobs` with payload `{ "device_key": "rtl:0", "label": "web", "params": { ... } }`.
* `params` mirrors CLI flags: `start`, `stop`, `step`, `samp_rate`, `fft`, `avg`, `gain`, `threshold_db`, `cfar_*`, `loop`, `repeat`, `duration`, `sleep_between_sweeps`, `jsonl`, etc.
* High-level toggles map directly to scanner flags: `params["profile"]` → `--profile`, `params["spur_calibration"]` → `--spur-calibration`. Controller must keep this translation 1:1; do not reimplement DSP logic elsewhere.
* Any optional hints (`soapy_args`, `extra_args`, `notify`) should pass straight through to `_build_cmd` without filtering.

---

## 2) Key entrypoints & responsibilities

* `sdrwatch.py`

  * CLI parsing + scan profile application (`--profile`) with fixed gain, absolute power floors, and optional spur calibration mode.
  * Sweep loop: tune → capture → PSD → CFAR/abs-floor filtering → anomalous-window gating → baseline EMA updates for normal windows.
  * DetectionEngine clusters hits across windows, consults `spur_map`, computes `confidence`, and only promotes persistent detections (JSONL emission mirrors DB rows).
* `sdrwatch-control.py`

  * Device discovery (serial/index/driver).
  * **Lock protocol**: file locks under `locks/` named by `device_key`.
  * Job lifecycle: build command (`_build_cmd`), spawn, supervise, reap stale locks.
  * Resolves `sdrwatch.py` via `resolve_scanner_paths()` every time a job starts so deployments under `/opt/sdrwatch`, symlinked releases, or manual runs keep working without restarting the daemon.
  * HTTP API: RESTful start/stop/list jobs + log streaming, plus `/profiles` which caches `sdrwatch.py --list-profiles` output for the web.
* `sdrwatch-web-simple.py`

  * Flask endpoints to read tables (scans/detections/baseline).
  * REST-first control layer: `/api/jobs`, `/api/jobs/active`, `/api/jobs/<id>`, `/api/jobs/<id>/logs`, `/ctl/devices`, plus `/profiles` consumption.
  * Templates in `templates/`. Prefer server endpoints that act as stateless REST clients for the controller; UI scripts should call those REST endpoints, not flask internals.
  * Control page renders a **profile dropdown** populated from controller `/profiles`, exposes a **mode selector** (“Normal scan” vs “Spur calibration”), and posts `params` so `profile`/`spur_calibration` flow through unchanged.
  * Detections view surfaces `confidence`, a `spur?` badge for frequencies near spur_map bins, and “New/Known” status derived from baseline EMA occupancy (currently `<0.2` = new). Keep these annotations data-driven; no DSP logic in the web layer.

---

## 3) SQLite schema (do not silently change)

Tables referenced by code/UI (column names are contract):

* **`scans`**: `id INTEGER PK AUTOINCREMENT`, `t_start_utc TEXT`, `t_end_utc TEXT`, `f_start_hz INTEGER`, `f_stop_hz INTEGER`, `step_hz INTEGER`, `samp_rate INTEGER`, `fft INTEGER`, `avg INTEGER`, `device TEXT`, `driver TEXT`.
* **`detections`**: `scan_id INTEGER`, `time_utc TEXT`, `f_center_hz INTEGER`, `f_low_hz INTEGER`, `f_high_hz INTEGER`, `peak_db REAL`, `noise_db REAL`, `snr_db REAL`, `service TEXT`, `region TEXT`, `notes TEXT`, `confidence REAL`.
* **`baseline`**: `bin_hz INTEGER PRIMARY KEY`, `ema_occ REAL`, `ema_power_db REAL`, `last_seen_utc TEXT`, `total_obs INTEGER`, `hits INTEGER`.
* **`spur_map`**: `bin_hz INTEGER PRIMARY KEY`, `mean_power_db REAL`, `hits INTEGER`, `last_seen_utc TEXT` (populated via `--spur-calibration`).

**Migrations**: if you must change types/names, add a migration script in `migrations/` + version table `schema_version(version INTEGER, applied_ts TEXT)` and bump version in code. Update Web UI queries accordingly.

---

## 4) Bandplan & outputs

* **Bandplan CSV** (header contract): `low_hz,high_hz,service,region,notes`.

  * Add rows; avoid renaming headers. Unit is Hz.
* **JSONL emit** (optional): one object per detection (current keys):

  ```json
  {
    "time_utc": "2025-11-10T21:30:05Z",
    "f_center_hz": 100500000,
    "f_low_hz": 100437500,
    "f_high_hz": 100562500,
    "peak_db": -37.1,
    "noise_db": -51.3,
    "snr_db": 14.2,
    "service": "FM Broadcast",
    "region": "Global",
    "notes": "",
    "is_new": true,
    "confidence": 0.82
  }
  ```

---

## 5) Device drivers & discovery

* Default path: **SoapySDR** (`--driver rtlsdr|hackrf|…`); fallback/native: `--driver rtlsdr_native`.
* Discovery metadata is passed via `--soapy-args` (serial, index). The Controller converts device selection → CLI flags in `_build_cmd`.
* Keep **optional imports** and clear error messages in CLI; do not hard-require SciPy/Soapy if pyrtlsdr path is chosen.

---

## 6) Lock protocol (must remain compatible)

* Lock files: `${SDRWATCH_CONTROL_BASE}/locks/{device_key}.lock`.
* Acquire: controller writes an owner token (job id). It may briefly write `pending` then update to the job id.
* Release: controller deletes the file when the child exits; a background reaper and startup reconciliation clear stale locks.
* Preserve: file naming, simple text owner content, stale-lock reaper behavior, and atomic create/delete semantics.

---

## 7) Performance & resource constraints (Pi 5, Trixie)

* **FFT sizes** should fit memory; avoid ballooning arrays—prefer streaming windows.
* Prefer **vectorized** numpy ops over Python loops.
* Place temporary arrays in function scope (let GC free between sweeps).
* CLI defaults should be conservative for RTL-SDR (2.4 MS/s, FFT≤4096, `avg` small).
* When adding features, guard with `--flag` and keep defaults fast.

---

## 8) Testing guidance (hardware-first)

* Primary testing is done directly with external SDR hardware on hand.
* No simulated SDR layer or fake capture system is required; the development cycle includes physically operating and observing the hardware.
* Use repeatable test bands (e.g., FM, ADS-B, known carriers) for verifying changes.
* Capture small reference sweeps for later regression comparisons (optional but useful).
* Ensure each code change preserves expected hardware behavior (lock creation, DB writes, detection counts, runtime stability).
* Optional lightweight tests can verify math correctness (CFAR thresholds, PSD outputs) if desired but are not mandatory for normal iteration.
* Maintainer runs validation on real hardware; Copilot should skip automated tests locally and instead call out any specific on-device checks the maintainer should perform.

---

## 9) Developer tasks & editor integration

* **VS Code tasks** (local):

  * `lint`: `ruff check . && ruff format --check .`
  * `type`: `mypy src`
  * `test`: `pytest -q`
  * `ship`: `ruff + mypy + pytest` → `rsync` to Pi → `systemctl restart sdrwatch` → show status.
* **Systemd** (on Pi):

  * `ExecStart=/usr/bin/python3 -m sdrwatch ...`
  * `Restart=on-failure`; `User=sdrwatch`; `EnvironmentFile=/etc/sdrwatch/env`
  * Logs via `journalctl -u sdrwatch -f`.

---

## 10) CLI patterns (examples)

* One-shot FM scan (optionally with a profile):

  ```bash
  python3 sdrwatch.py \
    --start 88e6 --stop 108e6 --step 1.8e6 \
    --samp-rate 2.4e6 --fft 4096 --avg 8 \
    --driver rtlsdr --gain auto --profile fm_broadcast
  ```
* Controller API serve:

  ```bash
  python3 sdrwatch-control.py serve --host 127.0.0.1 --port 8765 --token secret123
  ```
* Web UI against controller:

  ```bash
  SDRWATCH_CONTROL_URL=http://127.0.0.1:8765 \
  SDRWATCH_CONTROL_TOKEN=secret123 \
  python3 sdrwatch-web-simple.py --db sdrwatch.db --host 0.0.0.0 --port 8080
  ```

* Spur calibration sweep:

  ```bash
  python3 sdrwatch.py --start 400e6 --stop 470e6 --step 2.4e6 \
    --samp-rate 2.4e6 --fft 4096 --avg 8 --driver rtlsdr --spur-calibration
  ```

---

## 11) Trixie / Raspberry Pi OS considerations

* Use **`uv` or `pip-tools`** to lock dependencies (SciPy wheels for aarch64 on Debian 13 are available; avoid building from source on Pi if possible).
* If SoapySDR from apt is missing RTL plugin, install `soapysdr-module-rtlsdr`.
* Keep `numpy` and `numba` versions compatible with Python on Pi; avoid numba unless clearly beneficial.

---

## 12) Safe-change guidelines (Copilot guardrails)

* **Do not** rename SQLite columns or change types without adding a migration and updating Web UI queries.
* Preserve optional dependency behavior (SciPy/Soapy presence toggles paths).
* When touching lock logic, keep file names, stale reaper, and atomicity.
* For detection math, keep SciPy fallback functionally equivalent; add/adjust tests first.
* When touching controller/web layers, do **not** parse IQ samples or rebuild DSP logic—use scanner outputs/DB (`confidence`, baseline status, spur_map) as the source of truth.

---

## 13) Controller + Web REST contract

**Controller server (authoritative):**

* `GET /devices` → discovered devices + keys
* `GET /jobs` → list jobs
* `POST /jobs` → start scan `{device_key, label, params}`
* `GET /jobs/<id>` → job detail
* `GET /jobs/<id>/logs` → log tail (optional `?tail=N`)
* `DELETE /jobs/<id>` → stop job
* `GET /profiles` → cached profiles from `sdrwatch.py --list-profiles`
* Auth: bearer token via `SDRWATCH_CONTROL_TOKEN` when the server is started with a token.

**Web proxy (RESTful façade used by templates/JS):**

* `GET /api/jobs` → passthrough list
* `GET /api/jobs/active` → derived active-state payload
* `POST /api/jobs` (alias `/api/scans`) → start job
* `GET /api/jobs/<id>` → passthrough detail
* `DELETE /api/jobs/<id>` + `/api/scans/active` shim → stop job
* `GET /api/jobs/<id>/logs` + `/api/logs` shim → streaming logs (requires auth)
* Always keep these endpoints stateless and aligned with the controller. When adding UI features, extend the REST layer first, then build UI on top.

---

## 14) Observability

* Optional: add timing around capture→PSD→DB if needed (no `duration_ms` field in `scans` currently).
* Optional: Prometheus **textfile** writer with a few gauges/counters (scans/sec, detections/sec, db_commit_ms).
* Debug logging toggled by `--verbose` and/or `SDRWATCH_DEBUG=1`.

---

## 15) Contribution flow for AI tools (patch-first)

1. Open a **small scope**: file + function(s) + failing test/log.
2. Request a **unified diff** only; forbid wide rewrites.
3. Ensure new/changed behavior is covered by manual or hardware test validation.
4. Run `ruff + mypy + pytest` locally; then `ship`.
5. If DB/HTTP/API changed: add/adjust docs + migration + Web UI glue.

---

## 16) Open questions for the maintainer

* Adopt ad-hoc migrations (SQL scripts) vs Alembic?
* Standardize `SDRWATCH_CONTROL_BASE` for tests/CI (tmp dir per run)?
* Where to pin JSONL output format (doc + conformance test)?
* Minimal **FakeSDR** interface not planned (hardware-based workflow confirmed).
* Should `/api` remain Flask-only or split into FastAPI later? (Current plan: keep Flask but REST-first.)
