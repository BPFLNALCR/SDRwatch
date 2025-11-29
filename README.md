# SDR-Watch 📡🔎

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![CI](https://github.com/BPFLNALCR/sdr-watch/actions/workflows/ci.yml/badge.svg)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205-red)
![SDR](https://img.shields.io/badge/SDR-RTL--SDR-blue)
![Planned SDRs](https://img.shields.io/badge/Planned-HackRF%2C%20Airspy%2C%20LimeSDR%2C%20USRP-yellow)
![WebUI](https://img.shields.io/badge/WebUI-Flask-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**Wideband spectrum scanner, baseline builder, and bandplan mapper for SDR devices with a lightweight web dashboard.**

SDR-Watch transforms a Raspberry Pi 5 and SDR dongle into a **persistent spectrum monitoring station**. It sweeps wide frequency ranges, detects and logs signals, builds long-term baselines of spectrum activity, and maps detections to official frequency allocations. It now includes a **simple web dashboard** for real-time monitoring and control. 🌐

👉 Example applications:

- **Electronic Protection**: Detect interference, jamming attempts, or unusual transmissions in critical bands.
- **Spectrum Security**: Identify unauthorized users, validate coordination, and monitor long-term occupancy.
- **Research & Development**: Study waveform usage, analyze antenna performance, and collect environmental RF data.
- **Ops & Training**: Enable live visualization of RF activity during field exercises or experimental events.

This is a **lightweight but powerful tool** that makes you the one who knows what’s really happening in the air first.

At its current stage of development, SDR-Watch is:

* ✅ Optimized for **RTL-SDR** devices (support for HackRF, Airspy, LimeSDR, USRP, etc. planned).
* ✅ Intended to run on a **Raspberry Pi 5** with a **32 GB SD card** using **Raspbian Lite OS**.
* ✅ Usable as both a CLI tool and a web dashboard.

---

## ✨ Features

- **📶 Wideband Sweeps**: Scan across frequency ranges using RTL-SDR, HackRF, Airspy, LimeSDR, USRP (planned).
- **🔎 Signal Detection**: Robust noise floor estimation (median + MAD) with thresholding.
- **📊 Baseline Tracking**: Long-term exponential moving average to separate normal vs. anomalous signals.
- **🗺️ Bandplan Mapping**: Map detections to FCC, CEPT, ITU-R, and other official allocations.
- **💾 Data Logging**: Store all scans, detections, and baselines in SQLite.
- **🌍 Web Dashboard**:
  - 📈 Real-time graphs and histograms.
  - 🎛️ Control buttons for common scan presets (FM band, full sweep, etc.).
  - 👀 At-a-glance monitoring of activity and occupancy.
  - 🕒 Trend timeline with hourly/daily buckets and strongest SNR snapshots.
  - 🗺️ Frequency-coverage heatmap with optional LAT/LONG tagging per scan.
  - 🎚️ Interactive filters for service, SNR, frequency span, and lookback windows.
- **🧠 Self-healing controller**:
  - Auto-discovers `sdrwatch.py` every launch via `resolve_scanner_paths()` so `/opt/sdrwatch` deployments, symlinked releases, and fresh installs work without restarting the daemon.
  - Exposes a RESTful job API (`/jobs`, `/jobs/<id>`, `/jobs/<id>/logs`) that the web layer simply proxies.
- **🔔 Alerts & Outputs**:
  - Desktop notifications (`notify-send`) for new detections.
  - JSONL stream for integration with Grafana, Loki, ELK (each record now includes the window coverage ratio, observed duration, bandwidth estimate, and persistence mode used for gating).
- **⚙️ Services Integration**: Systemd units for `sdrwatch-control` (API manager) and `sdrwatch-web` (dashboard).
  - Web UI now uses REST endpoints (`/api/jobs`, `/api/jobs/active`, `/api/jobs/<id>`, `/api/jobs/<id>/logs`) so multiple browser sessions stay in sync and the architecture can scale horizontally.

---

## 🛠️ Installation (Raspberry Pi 5 – Raspbian Lite 64-bit)

Quick install with the included one-shot installer:

```bash
git clone https://github.com/SDRwatch/sdr-watch.git
cd sdr-watch
chmod +x install-sdrwatch.sh
./install-sdrwatch.sh
```

The installer will:

- Install dependencies (RTL-SDR, HackRF, SoapySDR, NumPy/SciPy, Flask, etc.).
- Set up a Python venv with system packages.
- Verify hardware (`rtl_test`, `hackrf_info`).
- Apply kernel blacklist + udev rules for RTL2832U dongles.
- Optionally configure + enable **systemd services** for automatic startup.

🔧 Non-interactive mode:

```bash
SDRWATCH_AUTO_YES=1 ./install-sdrwatch.sh
```

---

## 🚀 Usage

### Command Line

Sweep the FM band once (default behavior):

```bash
python3 sdrwatch.py --start 88e6 --stop 108e6 --step 1.8e6 \
  --samp-rate 2.4e6 --fft 4096 --avg 8 --driver rtlsdr --gain auto
```

Notes:
- By default, the scanner runs a single full sweep and exits.
- Use `--loop` for continuous sweeps, `--repeat N` for a fixed number of sweeps, or `--duration 10m` to run until the time elapses.
- Tag scan metadata with `--latitude`/`--longitude` (decimal degrees) when location context matters; the values are stored in the `scans` table and surfaced in the dashboard + heatmap.

Continuous monitoring across 30 MHz – 1.7 GHz:

```bash
python3 sdrwatch.py --start 30e6 --stop 1700e6 --step 2.4e6 \
  --samp-rate 2.4e6 --fft 4096 --avg 8 --driver rtlsdr \
  --gain auto --loop --notify --db sdrwatch.db --jsonl events.jsonl
```

### Configurable persistence gates

Use `--persistence-mode` to choose how detections are promoted to persistent records:

| Mode | Description |
| --- | --- |
| `hits` (default) | Requires a minimum number of hits/windows **and** the specified `--persistence-hit-ratio` coverage within the cluster span. |
| `duration` | Requires the cluster to remain active for at least `--persistence-min-seconds` while still meeting the minimum hit/window counts. |
| `both` | Requires both coverage and duration thresholds to pass.

Additional knobs:

- `--persistence-hit-ratio` (default `0.6`)
- `--persistence-min-seconds` (default `10`)
- `--persistence-min-hits` / `--persistence-min-windows` (defaults `2` / `2`)

The **Control** page exposes the same options so you can switch between hit-ratio, duration, or combined persistence gating directly from the dashboard.

### Web Dashboard 🌐

If installed with services enabled, the dashboard is always on at boot:\
`http://<raspberrypi-ip>:8080`

Manual launch:

```bash
python3 sdrwatch-web-simple.py --db sdrwatch.db --host 0.0.0.0 --port 8080
```

The dashboard can render immediately even if the database is empty; it will auto-switch to the "waiting" state until the first scan populates tables.

---

## 🔗 Controller REST API & Tokens

The controller exposes a stable REST surface consumed by the Flask frontend or any automation:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/devices` | Enumerate SDRs (key, kind, label, metadata). |
| `GET` | `/jobs` | List jobs (status, params, timestamps). |
| `POST` | `/jobs` | Start a job `{device_key, label, params}`. |
| `GET` | `/jobs/<id>` | Inspect a specific job. |
| `GET` | `/jobs/<id>/logs?tail=N` | Stream scanner logs from disk. |
| `DELETE` | `/jobs/<id>` | Stop the job (SIGTERM/SIGKILL fallback). |

Set `SDRWATCH_CONTROL_TOKEN` when running `sdrwatch-control.py serve` to require bearer auth. The web app reads `SDRWATCH_CONTROL_URL` / `SDRWATCH_CONTROL_TOKEN` and simply forwards REST calls, exposing its own `Bearer SDRWATCH_TOKEN` guard for the `/api/*` routes.

**Scanner path discovery:** the controller looks for `sdrwatch.py` in the following order each time a job starts: `SDRWATCH_SCRIPT`, `SDRWATCH_PROJECT_DIR/sdrwatch.py`, alongside the controller script, project parent, current working directory, and `/opt/sdrwatch[/current]`. Override via environment variables if you relocate scripts.

---

## 🗄️ Database Schema

- **scans**: sweep metadata
- **detections**: detected signals
- **baseline**: persistent occupancy statistics

---

## 📑 Bandplan CSV Format

```csv
low_hz,high_hz,service,region,notes
433050000,434790000,ISM,ITU-R1 (EU),Short-range devices
902000000,928000000,ISM,US (FCC),902-928 MHz ISM
2400000000,2483500000,ISM,Global,2.4 GHz ISM
```

---

## 🔍 Inspecting Collected Data

Query detections:

```bash
sqlite3 -header -column sdrwatch.db "SELECT time_utc, f_center_hz/1e6 AS MHz, snr_db, service FROM detections ORDER BY id DESC LIMIT 20;"
```

Query baseline:

```bash
sqlite3 -header -column sdrwatch.db "SELECT bin_hz/1e6 AS MHz, round(ema_occ,3) AS occ FROM baseline ORDER BY occ DESC LIMIT 20;"
```

Export:

```bash
sqlite3 -header -csv sdrwatch.db "SELECT * FROM detections;" > detections.csv
```

---

## 🛣️ Roadmap

- Expand SDR support (HackRF, Airspy, LimeSDR, USRP).
- Add CFAR-style detection to reduce false positives.
- Implement duty-cycle analysis for bursty signals.
- Enhance web dashboard with interactive filters & charts.
- Multi-SDR coordination for distributed scanning.
- Expand region-specific bandplans (FCC, CEPT, BNetzA).

---

## 📜 License

MIT License. See [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

Inspired by `rtl_power`, `SoapyPower`, and GNU Radio’s `gr-inspector`, but extended for **persistent monitoring, automated mapping, and a real-time dashboard**.

