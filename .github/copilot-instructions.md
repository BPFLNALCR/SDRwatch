# SDRwatch — AI coding assistant guide (for Copilot/ChatGPT)

Keep guidance short and operational. Prefer **small diffs** over whole-file rewrites. Prioritize **runtime topology**, **CLI↔service wiring**, **DB schema & migrations**, **device locking**, and **tests**.

---

## 1) Big picture (runtime topology)

**Authoritative script purposes (use these names in prompts/commits):**

* **`sdrwatch.py` ("scanner")** — CLI tool that performs wideband sweeps on an SDR device, computes PSD/CFAR, updates the **SQLite DB** (`scans`, `detections`, `baseline`), and can emit JSONL and desktop/CLI notifications. Single-process, device-bound. Does **not** manage multiple devices.

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

---

## 2) Key entrypoints & responsibilities

* `sdrwatch.py`

  * CLI parsing, driver selection (SoapySDR default; `--driver rtlsdr_native` uses pyrtlsdr).
  * Sweep loop: tune → capture → PSD (SciPy if present, fallback else) → CFAR/thresholding → batch DB writes → (optional) JSONL emit.
  * Baseline maintenance (EMA/occurrence) and new-signal detection thresholding.
* `sdrwatch-control.py`

  * Device discovery (serial/index/driver).
  * **Lock protocol**: file locks under `locks/` named by `device_key`.
  * Job lifecycle: build command (`_build_cmd`), spawn, supervise, reap stale locks.
  * Resolves `sdrwatch.py` via `resolve_scanner_paths()` every time a job starts so deployments under `/opt/sdrwatch`, symlinked releases, or manual runs keep working without restarting the daemon.
  * HTTP API: RESTful start/stop/list jobs + log streaming.
* `sdrwatch-web-simple.py`

  * Flask endpoints to read tables (scans/detections/baseline).
  * REST-first control layer: `/api/jobs`, `/api/jobs/active`, `/api/jobs/<id>`, `/api/jobs/<id>/logs`, `/ctl/devices`.
  * Templates in `templates/`. Prefer server endpoints that act as stateless REST clients for the controller; UI scripts should call those REST endpoints, not flask internals.

---

## 3) SQLite schema (do not silently change)

Tables referenced by code/UI (column names are contract):

* **`scans`**: `id INTEGER PK AUTOINCREMENT`, `t_start_utc TEXT`, `t_end_utc TEXT`, `f_start_hz INTEGER`, `f_stop_hz INTEGER`, `step_hz INTEGER`, `samp_rate INTEGER`, `fft INTEGER`, `avg INTEGER`, `device TEXT`, `driver TEXT`.
* **`detections`**: `scan_id INTEGER`, `time_utc TEXT`, `f_center_hz INTEGER`, `f_low_hz INTEGER`, `f_high_hz INTEGER`, `peak_db REAL`, `noise_db REAL`, `snr_db REAL`, `service TEXT`, `region TEXT`, `notes TEXT`.
* **`baseline`**: `bin_hz INTEGER PRIMARY KEY`, `ema_occ REAL`, `ema_power_db REAL`, `last_seen_utc TEXT`, `total_obs INTEGER`, `hits INTEGER`.

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
    "is_new": true
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

* One-shot FM scan:

  ```bash
  python3 sdrwatch.py \
    --start 88e6 --stop 108e6 --step 1.8e6 \
    --samp-rate 2.4e6 --fft 4096 --avg 8 \
    --driver rtlsdr --gain auto
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

---

## 13) Controller + Web REST contract

**Controller server (authoritative):**

* `GET /devices` → discovered devices + keys
* `GET /jobs` → list jobs
* `POST /jobs` → start scan `{device_key, label, params}`
* `GET /jobs/<id>` → job detail
* `GET /jobs/<id>/logs` → log tail (optional `?tail=N`)
* `DELETE /jobs/<id>` → stop job
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
