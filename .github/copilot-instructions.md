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

**Updated runtime stack (match the refactored package layout in `sdrwatch/`):**

* **`python -m sdrwatch.cli` ("scanner CLI")** — Authoritative entrypoint that parses flags, applies scan profiles from `sdrwatch.io.profiles`, and hands execution to `sdrwatch.sweep.runner`. The legacy root-level `sdrwatch.py` now shims into this module and should be treated as deprecated.

* **`sdrwatch.sweep.runner` ("sweeper/orchestrator")** — Binds drivers (`sdrwatch.drivers.*`), DSP (`sdrwatch.dsp.*`), detection (`sdrwatch.detection.*`), and baseline stores (`sdrwatch.baseline.*`). Every sweep is baseline-first: load the selected baseline, run PSD/CFAR, update `baseline_stats`, persist detections/scan updates, and optionally emit JSONL.

* **`sdrwatch-control.py` ("controller" / "manager")** — Long-lived job manager. Discovers SDR hardware, enforces `${SDRWATCH_CONTROL_BASE}/locks`, builds scanner commands (always including `--baseline-id`), and exposes the HTTP API for jobs/profiles/baselines/logs. It shells out to `python -m sdrwatch.cli` (not the shim) for each scan and re-resolves paths per job.

* **`sdrwatch-web-simple.py` ("web" / "UI")** — Baseline-centric Flask front end. Queries SQLite tables directly for dashboards while proxying all control-plane actions to the controller via REST with bearer auth. Never touch SDR hardware or DSP internals here.

* **Support scripts & assets** — Installer (`install-sdrwatch.sh`), bandplan CSVs, optional migrations, and systemd unit definitions still behave as before but now reference the package entrypoint.

**Colloquial → module mapping (use these names in prompts/commits):**

* *scanner CLI* → `python -m sdrwatch.cli`
* *sweeper* → `sdrwatch.sweep.runner` (and helpers under `sdrwatch/sweep/`)
* *scanner shim* → `sdrwatch.py` (invoke only for legacy compatibility)
* *controller/manager/daemon* → `sdrwatch-control.py`
* *web/UI* → `sdrwatch-web-simple.py`
* *installer* → `install-sdrwatch.sh`

**State layout** (overridable via env):

* `SDRWATCH_CONTROL_BASE` (default `/tmp/sdrwatch-control/`) → `locks/`, `logs/`, and `state.json`.
* SQLite default: `sdrwatch.db` (scan CLI flag `--db` or controller/job params can override).

**Goal for contributors**: respect the modular split in [sdrwatch/architecture.md](../sdrwatch/architecture.md). Keep DSP modules pure, baseline modules stateful, the sweeper thin, and ensure controller/web remain adapters around the scanner CLI and SQLite contracts. make changes safe for long‑running service on Raspberry Pi OS **Trixie** + Pi 5 with RTL‑SDR by default, optional HackRF/SoapySDR, while keeping baseline persistence the primary unit of state.

### ScanRequest JSON (web → controller)

* Web posts to controller `/jobs` with payload `{ "device_key": "rtl:0", "label": "web", "baseline_id": 3, "params": { ... } }`.
* `params` mirrors CLI flags surfaced by `sdrwatch.cli`: `start`, `stop`, `step`, `samp_rate`, `fft`, `avg`, `gain`, `threshold_db`, `cfar_*`, `loop`, `repeat`, `duration`, `sleep_between_sweeps`, `jsonl`, etc.
* Toggles map 1:1 to CLI switches: `params["profile"]` → `--profile`, `params["spur_calibration"]` → `--spur-calibration`, `params["two_pass"]` → `--two-pass`, etc. Controller must only translate, never reinterpret DSP logic, and must always include the chosen `baseline_id` in `_build_cmd`.
* Optional hints (`soapy_args`, `extra_args`, `notify`) flow straight to `_build_cmd` so the CLI sees the same argument surface area as local runs.

---

## 2) Key entrypoints & responsibilities

* `sdrwatch.cli`

  * Parses CLI flags, applies `ScanProfile` presets from `sdrwatch.io.profiles`, enforces that a `baseline_id` or `latest` alias is supplied, and then calls `sdrwatch.sweep.runner.run_scan()`.
  * Emits `--list-profiles` JSON directly from the profile module so the controller/web UI can stay in sync without bespoke serialization.

* `sdrwatch.sweep` package

  * `scheduler.py` converts CLI/profile settings into concrete sweep windows (including revisit queues and span clamps from `revisit_*` flags).
  * `sweeper.py` / `runner.py` bind the driver, DSP, detection, and baseline persistence layers. Keep them orchestration-only: tune → capture → PSD → CFAR → clustering → baseline updates → DB writes → optional JSONL/logging.

* `sdrwatch.drivers`

  * `rtlsdr.py`, `soapy.py`, and any future drivers expose a shared interface (`open`, `tune`, `read_samples`, `set_gain`, etc.). The sweeper chooses the module at runtime per CLI options; do not reach into driver internals from other layers.

* `sdrwatch.dsp`

  * Houses FFT/windowing, CFAR, clustering, and noise estimation logic. These modules must remain pure/functional so they can be tested independently and reused by both coarse and revisit passes.

* `sdrwatch.detection`

  * `engine.py` translates DSP segments into detection records, handles spur suppression via `sdrwatch.baseline.spur`, computes `confidence`, and classifies NEW vs known vs quieted hits based on baseline stats.

* `sdrwatch.baseline`

  * `model.py`/`context.py` load baseline metadata; `stats.py` updates per-bin EMAs and occupancy; `persistence.py` maintains `baseline_detections`; `events.py` surfaces NEW/QUIETED/POWER_SHIFT notifications; `spur.py` records calibration sweeps into `spur_map`; `store.py` centralizes SQLite accessors.

* `sdrwatch.io`

  * `bandplan.py` maps detections to services/regions. `profiles.py` stores built-in scan presets. Future DB helpers or WAL tuning should live here rather than the CLI.

* `sdrwatch-control.py`

  * Handles device discovery, lock enforcement, scanner process lifecycle, `/profiles` caching (by shelling out to `python -m sdrwatch.cli --list-profiles`), and `/baselines` CRUD that ensures every job includes a valid baseline id.

* `sdrwatch-web-simple.py`

  * Flask views are thin REST clients for the controller and direct readers of SQLite baseline tables. UI templates highlight baseline health, persistent detections, and spur-map annotations without introducing DSP logic.

---

## 3) SQLite schema (do not silently change)

Tables referenced by code/UI (column names are contract). Ownership lives in `sdrwatch.baseline.*` and `sdrwatch.io.store`, so adjust those modules if schema changes:

* **`baselines`**: `id`, `name`, `created_at`, `location_lat`, `location_lon`, `sdr_serial`, `antenna`, `notes`, `freq_start_hz`, `freq_stop_hz`, `bin_hz`, `baseline_version`.
* **`baseline_stats`**: `baseline_id`, `freq_hz` (or migration-specific `bin_index`), `noise_floor_ema`, `power_ema`, `occ_count`, `last_seen_utc`.
* **`baseline_detections`**: `id`, `baseline_id`, `f_low_hz`, `f_high_hz`, `f_center_hz`, `first_seen_utc`, `last_seen_utc`, `total_hits`, `total_windows`, `confidence`.
* **`scan_updates`**: `id`, `baseline_id`, `timestamp_utc`, `num_hits`, `num_segments`, `num_new_signals`.
* **`spur_map`**: `bin_hz INTEGER PRIMARY KEY`, `mean_power_db REAL`, `hits INTEGER`, `last_seen_utc TEXT` (populated via spur calibration mode).

Baseline math should continue to flow through the context/persistence helpers rather than issuing ad-hoc SQL in new code.

---

## 4) Bandplan & outputs

* **Bandplan CSV** (header contract): `low_hz,high_hz,service,region,notes`.

  * `sdrwatch.io.bandplan.load_bandplan()` consumes these files and the detection engine expects Hz units. Add rows instead of renaming headers.
* **JSONL emit** (optional): `sdrwatch.util.scan_logger` and sweeper helpers write one object per detection (must include the active `baseline_id` alongside the existing fields):

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

* `sdrwatch.drivers.soapy` is the preferred path; `sdrwatch.drivers.rtlsdr` handles the native librtlsdr fallback when Soapy is unavailable. Future drivers should expose the same interface so the sweeper stays generic.
* Discovery metadata comes from the controller (serial/index/driver) and is passed via `--driver` + `--soapy-args`. `_build_cmd` must not alter the meaning of these options—just forward them to the CLI.
* Keep optional imports/lightweight dependency checks (`HAVE_SOAPY`, `HAVE_RTLSDR`) inside the driver modules. The CLI should emit actionable error messages instead of crashing when a backend is missing.

---

## 6) Lock protocol (must remain compatible)

* Lock files: `${SDRWATCH_CONTROL_BASE}/locks/{device_key}.lock`.
* Acquire: controller writes an owner token (job id). It may briefly write `pending` then update to the job id.
* Release: controller deletes the file when the child exits; a background reaper and startup reconciliation clear stale locks.
* Preserve: file naming, simple text owner content, stale-lock reaper behavior, and atomic create/delete semantics. Do **not** move locking into the sweeper—this all belongs in the controller.

---

## 7) Performance & resource constraints (Pi 5, Trixie)

* Respect the separation of concerns: keep heavy math inside `sdrwatch.dsp` (vectorized NumPy) and let `sdrwatch.sweep` orchestrate without copying large buffers.
* FFT sizes must fit in Pi 5 memory; favor streaming Welch segments and reusing scratch arrays (`tmpdir` usage if needed) instead of allocating per window.
* Baseline EMA/occupancy updates in `sdrwatch.baseline.stats` should stay batched (array ops, single transaction per sweep) to avoid DB thrash.
* Guard feature experiments with CLI flags/profile settings so default scans remain RTL-SDR friendly (≈2.4 MS/s, FFT ≤4096, modest averages).
* Revisit windows double FFT/avg by default—ensure revisit-specific overrides do not explode runtime or RAM.

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

* Use `--revisit-span-limit-hz` whenever a job needs to widen/narrow the confirmation span beyond the profile default (FM profile ships with 420 kHz).
* One-shot FM scan (optionally with a profile) tied to an active baseline (create/select the baseline via controller or web first):

  ```bash
  python3 -m sdrwatch.cli \
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
  python3 -m sdrwatch.cli --start 400e6 --stop 470e6 --step 2.4e6 \
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
