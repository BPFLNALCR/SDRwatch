# SDRwatch — AI coding assistant guide (for Copilot/ChatGPT)

Keep guidance short and operational. Prefer **small diffs** over whole-file rewrites. Prioritize **runtime topology**, **baseline-first DB contracts**, **CLI↔service wiring**, **device locking**, and **tests**.

---

## Layered runtime model (baseline-oriented scanner ↔ controller ↔ web)

* **Scanner (`sdrwatch.py`)** owns SDR/DSP loops and now operates in a **baseline-first** flow: every scan is tied to an active baseline (`--baseline-id`). Each sweep loads the baseline, runs PSD/CFAR, updates **baseline EMAs/occupancy**, persists detections into `baseline_detections`, writes lightweight `scan_updates`, and handles spur calibration into `spur_map`. It still exports built-in profiles via `--list-profiles` and can emit JSONL per detection tagged with `baseline_id`.
* **Controller (`sdrwatch-control.py`)** is the only process that spawns scanners. All jobs must include a `baseline_id`; the controller builds the CLI, enforces device locks, monitors child processes, shells out for `/profiles`, and brokers baseline metadata via new `/baselines` endpoints.
* **Web (`sdrwatch-web-simple.py`)** never touches SDR hardware or IQ streams. It manages baseline selection/creation, proxies control actions (jobs/logs/devices/profiles/baselines) to the controller, and reads SQLite directly for dashboards that highlight baseline stats and persistent detections. No DSP logic outside the scanner.

Treat the scanner’s baseline-oriented DB schema + CLI behavior as authoritative. Controller/web layers stay thin adapters around those contracts.

---

## 1) Big picture (runtime topology)

**Authoritative script purposes (use these names in prompts/commits):**

* **`sdrwatch.py` ("scanner")** — CLI tool that performs wideband sweeps tied to an active baseline, computes PSD/CFAR, updates **baseline tables** (`baselines`, `baseline_stats`, `baseline_detections`, `scan_updates`, `spur_map`), and can emit JSONL events that reference the baseline. Single-process, device-bound. Includes scan profiles, anomalous-window gating, spur calibration/suppression, and a DetectionEngine that now promotes persistence via baseline tracking.

* **`sdrwatch-control.py` ("controller" / "manager")** — Long‑lived job manager. Discovers devices (serial/index/driver), enforces **file‑lock ownership** per device under `${SDRWATCH_CONTROL_BASE}/locks`, constructs the scanner command (`_build_cmd`) with the required `--baseline-id`, spawns and monitors scanner subprocesses, exposes an **HTTP API** for start/stop/list/health/baseline CRUD, and reaps stale locks.

* **`sdrwatch-web-simple.py` ("web" / "UI")** — Baseline-centric Flask front‑end. **Read‑only** queries to the SQLite DB (baseline tables + spur map). Proxies **control actions** (jobs/logs/devices/profiles/baselines) to the controller via RESTful endpoints pointing at `SDRWATCH_CONTROL_URL` with bearer auth. No direct device access. Treat the web layer as a thin REST client; avoid reintroducing stateful coupling.

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

**Goal for contributors**: make changes safe for long‑running service on Raspberry Pi OS **Trixie** + Pi 5 with RTL‑SDR by default, optional HackRF/SoapySDR, while keeping baseline persistence the primary unit of state.

### ScanRequest JSON (web → controller)

* Web posts to controller `/jobs` with payload `{ "device_key": "rtl:0", "label": "web", "baseline_id": 3, "params": { ... } }`.
* `params` mirrors CLI flags: `start`, `stop`, `step`, `samp_rate`, `fft`, `avg`, `gain`, `threshold_db`, `cfar_*`, `loop`, `repeat`, `duration`, `sleep_between_sweeps`, `jsonl`, etc.
* High-level toggles map directly to scanner flags: `params["profile"]` → `--profile`, `params["spur_calibration"]` → `--spur-calibration`. Controller must keep this translation 1:1; do not reimplement DSP logic elsewhere, and always pass the selected `baseline_id` through as `--baseline-id`.
* Any optional hints (`soapy_args`, `extra_args`, `notify`) should pass straight through to `_build_cmd` without filtering.

---

## 2) Key entrypoints & responsibilities

* `sdrwatch.py`

  * CLI parsing + scan profile application (`--profile`) with fixed gain, absolute power floors, baseline loader (`--baseline-id` required), and optional spur calibration mode.
  * Sweep loop: tune → capture → PSD → CFAR/abs-floor filtering → anomalous-window gating → baseline EMA updates + occupancy tracking → persistence updates in `baseline_detections` and lightweight `scan_updates` rows.
  * DetectionEngine clusters hits across windows, consults `spur_map`, computes `confidence`, and only promotes persistent, baseline-linked detections (JSONL emission mirrors `baseline_detections` and always includes `baseline_id`).
* `sdrwatch-control.py`

  * Device discovery (serial/index/driver).
  * **Lock protocol**: file locks under `locks/` named by `device_key`.
  * Job lifecycle: build command (`_build_cmd`) including `--baseline-id`, spawn, supervise, reap stale locks.
  * Resolves `sdrwatch.py` via `resolve_scanner_paths()` every time a job starts so deployments under `/opt/sdrwatch`, symlinked releases, or manual runs keep working without restarting the daemon.
  * HTTP API: RESTful start/stop/list jobs + log streaming, `/profiles` cache from `sdrwatch.py --list-profiles`, and `/baselines` CRUD that always enforce `baseline_id` presence when starting scans.
* `sdrwatch-web-simple.py`

  * Flask endpoints to read baseline tables (`baselines`, `baseline_stats`, `baseline_detections`, `scan_updates`, `spur_map`). Legacy scan pages should be deprecated rather than expanded.
  * REST-first control layer: `/api/jobs`, `/api/jobs/active`, `/api/jobs/<id>`, `/api/jobs/<id>/logs`, `/ctl/devices`, `/profiles`, plus new `/baselines` passthroughs.
  * Templates in `templates/`. Prefer server endpoints that act as stateless REST clients for the controller; UI scripts should call those REST endpoints, not flask internals.
  * Control page renders a **baseline selector/creator** plus **profile dropdown** populated from controller `/profiles`, exposes a **mode selector** (“Normal scan” vs “Spur calibration”), and posts `baseline_id` + `params` so `profile`/`spur_calibration` flow through unchanged.
  * Detections view surfaces `confidence`, a `spur?` badge for frequencies near spur_map bins, and “New/Known” status derived from baseline EMA occupancy (<0.2 = new). Keep these annotations data-driven; no DSP logic in the web layer.

---

## 3) SQLite schema (do not silently change)

Tables referenced by code/UI (column names are contract):

* **`baselines`**: `id`, `name`, `created_at`, `location_lat`, `location_lon`, `sdr_serial`, `antenna`, `notes`, `freq_start_hz`, `freq_stop_hz`, `bin_hz`, `baseline_version`.
* **`baseline_stats`**: `baseline_id`, `freq_hz` (or `bin_index` depending on migration), `noise_floor_ema`, `power_ema`, `occ_count`, `last_seen_utc`.
* **`baseline_detections`**: `id`, `baseline_id`, `f_low_hz`, `f_high_hz`, `f_center_hz`, `first_seen_utc`, `last_seen_utc`, `total_hits`, `total_windows`, `confidence`.
* **`scan_updates`**: `id`, `baseline_id`, `timestamp_utc`, `num_hits`, `num_segments`, `num_new_signals`.
* **`spur_map`**: `bin_hz INTEGER PRIMARY KEY`, `mean_power_db REAL`, `hits INTEGER`, `last_seen_utc TEXT` (populated via `--spur-calibration`).

---

## 4) Bandplan & outputs

* **Bandplan CSV** (header contract): `low_hz,high_hz,service,region,notes`.

  * Add rows; avoid renaming headers. Unit is Hz.
* **JSONL emit** (optional): one object per detection (must include the active `baseline_id` alongside the existing fields):

  ```json
  {
    "baseline_id": 3,
    "time_utc": "2025-11-10T21:30:05Z",
    "f_center_hz": 100500000,
    "f_low_hz": 100437500,
    "f_high_hz": 100562500,
    "bandwidth_hz": 125000,
    "peak_db": -37.1,
    "noise_db": -51.3,
    "snr_db": 14.2,
    "service": "FM Broadcast",
    "region": "Global",
    "notes": "",
    "is_new": true,
    "confidence": 0.82,
    "window_ratio": 0.73,
    "duration_s": 42.0,
    "persistence_mode": "hits"
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
* Baseline EMA and occupancy updates must remain lightweight (array ops, avoid per-row commits).
* Place temporary arrays in function scope (let GC free between sweeps).
* CLI defaults should be conservative for RTL-SDR (2.4 MS/s, FFT≤4096, `avg` small).
* When adding features, guard with `--flag` and keep defaults fast.

---

## 8) Testing guidance (hardware-first)

* Primary testing is done directly with external SDR hardware on hand.
* No simulated SDR layer or fake capture system is required; the development cycle includes physically operating and observing the hardware.
* Use repeatable test bands (e.g., FM, ADS-B, known carriers) for verifying changes.
* Capture small reference sweeps for later regression comparisons (optional but useful).
* Validate baseline workflows: create/select baselines when swapping antennas/locations and ensure scanner refuses jobs without `baseline_id`.
* Observe that persistent carriers accumulate `total_windows`/`total_hits` while NEW events transition as expected once occupancy rises.
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

* One-shot FM scan (optionally with a profile) tied to an active baseline (create/select the baseline via controller or web first):

  ```bash
  python3 sdrwatch.py \
    --baseline-id 3 \
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
* Keep the baseline-first workflow intact: scanner jobs must require `baseline_id`, controller/web endpoints must pass it through, and baseline persistence tables remain the source of truth.
* Preserve optional dependency behavior (SciPy/Soapy presence toggles paths).
* When touching lock logic, keep file names, stale reaper, and atomicity.
* For detection math, keep SciPy fallback functionally equivalent; add/adjust tests first.
* When touching controller/web layers, do **not** parse IQ samples or rebuild DSP logic—use scanner outputs/DB (`confidence`, baseline status, spur_map) as the source of truth.

---

## 13) Controller + Web REST contract

**Controller server (authoritative):**

* `GET /devices` → discovered devices + keys
* `GET /jobs` → list jobs
* `POST /jobs` → start scan `{device_key, label, baseline_id, params}` (controller rejects requests missing `baseline_id`)
* `GET /jobs/<id>` → job detail
* `GET /jobs/<id>/logs` → log tail (optional `?tail=N`)
* `DELETE /jobs/<id>` → stop job
* `GET /profiles` → cached profiles from `sdrwatch.py --list-profiles`
* `GET /baselines` / `POST /baselines` / `PATCH /baselines/<id>` → baseline list/create/update helpers shared with the web UI
* Auth: bearer token via `SDRWATCH_CONTROL_TOKEN` when the server is started with a token.

**Web proxy (RESTful façade used by templates/JS):**

* `GET /api/jobs` → passthrough list
* `GET /api/jobs/active` → derived active-state payload
* `POST /api/jobs` (alias `/api/scans`) → start job (requires `baseline_id`)
* `GET /api/jobs/<id>` → passthrough detail
* `DELETE /api/jobs/<id>` + `/api/scans/active` shim → stop job
* `GET /api/jobs/<id>/logs` + `/api/logs` shim → streaming logs (requires auth)
* `GET/POST/PATCH /api/baselines` → passthrough baseline CRUD
* Always keep these endpoints stateless and aligned with the controller. When adding UI features, extend the REST layer first, then build UI on top.

---

## 14) Observability

* Optional: add timing around capture→PSD→DB if needed (no `duration_ms` field in `scan_updates` currently).
* Optional: Prometheus **textfile** writer with a few gauges/counters (scans/sec, detections/sec, db_commit_ms).
* Consider exporting baseline health metrics (age, number of persistent detections, recent NEW events) for dashboards.
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
* How should baseline definitions be versioned for long-term stability?
* Should `baseline_detections` support geometric clustering updates?
* Should the web tier expose comparison dashboards across multiple baselines?
