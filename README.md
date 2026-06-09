# SDR-Watch 📡🔎

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![CI](https://github.com/BPFLNALCR/sdr-watch/actions/workflows/ci.yml/badge.svg)
![Platform](https://img.shields.io/badge/platform-Raspberry%20Pi%205-red)
![SDR](https://img.shields.io/badge/SDR-RTL--SDR%20%7C%20SoapySDR-blue)
![Planned SDRs](https://img.shields.io/badge/Planned-HackRF%2C%20Airspy%2C%20LimeSDR%2C%20USRP-yellow)
![WebUI](https://img.shields.io/badge/WebUI-Flask-orange)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

**Tactical spectrum situational awareness for SDR devices—wideband scanning, baseline tracking, evidence-backed signal classification support, and a real-time web dashboard.**

SDR-Watch transforms a Raspberry Pi 5 and SDR dongle into a **persistent spectrum monitoring station**. It sweeps wide frequency ranges, detects and logs signals, builds long-term baselines of spectrum activity, and maps detections to official frequency allocations. The **tactical web dashboard** provides real-time monitoring, signal classification, and actionable situational awareness.

**Example applications:**

- **Electronic Protection**: Detect interference, jamming attempts, or unusual transmissions in critical bands.
- **Spectrum Security**: Identify unauthorized users, validate coordination, and monitor long-term occupancy.
- **Research & Development**: Study waveform usage, analyze antenna performance, and collect environmental RF data.
- **Field Operations**: Enable live visualization of RF activity during exercises, events, or security operations.
- **Signal Classification**: Support operator-applied Friendly/Ambient/Hostile labels for rapid triage without overstating certainty.

**Current capabilities:**

* ✅ Optimized for **RTL-SDR** devices via native driver or **SoapySDR** abstraction.
* ✅ Runs on **Raspberry Pi 5** with **Raspberry Pi OS** (Trixie/Bookworm).
* ✅ Full CLI tool and tactical web dashboard with signal management.

---

## ✨ Features

### Core Scanning
- **Wideband Sweeps**: Scan across frequency ranges using RTL-SDR (native or SoapySDR).
- **Signal Detection**: CFAR-based detection with robust noise floor estimation (median + MAD).
- **Baseline Tracking**: Long-term exponential moving average to separate normal vs. anomalous signals.
- **Bandplan Mapping**: Map detections to FCC, CEPT, ITU-R, and other official allocations.
- **Data Logging**: Store all scans, detections, and baselines in SQLite.
- **Two-Pass Verification**: Refine bandwidths and suppress false positives with revisit sweeps.

### Tactical Web Dashboard
- **Signal Cards**: Visual grid of detected signals with frequency, bandwidth, SNR, and confidence.
- **Signal Classification**: Support operator-applied **Friendly** (green), **Ambient** (gray), or **Hostile** (red) labels.
- **Signal Selection**: Star/highlight signals of interest for tracking across sessions.
- **Human-Friendly IDs**: Each signal gets a unique identifier (e.g., `SIG-0042`) for easy reference.
- **Signal Labels & Notes**: Add custom labels and freeform notes to any detection.
- **User Bandwidth Corrections**: Override detected bandwidth for display purposes.
- **Signal Detail Pages**: Deep-dive view for individual signals with full metadata and edit forms.
- **Signals List**: Browse all signals across baselines with filtering and bulk operations.
- **Changes Panel**: Real-time delta feed showing NEW, QUIETED, and POWER_SHIFT events.
- **Band Summary**: At-a-glance view of spectrum occupancy by frequency band.
- **Baseline Management**: Create, select, and switch between multiple baselines.
- **Control Panel**: Start/stop scans, select profiles, configure parameters.
- **Spur Map**: View and manage known spurious signals for calibration.
- **Debug Dashboard**: Developer tools showing DB stats, errors, and system health.

### Controller & API
- **RESTful Job API**: `/jobs`, `/devices`, `/profiles`, `/baselines` endpoints.
- **Signal Management API**: `/api/signals` for CRUD operations on detections.
- **Bearer Token Auth**: Secure controller and web API with `SDRWATCH_CONTROL_TOKEN`.
- **Auto-Discovery**: Controller resolves scanner paths dynamically for flexible deployments.
- **Self-Healing**: Stale lock cleanup, process monitoring, and graceful restarts.

### Outputs & Integration
- **JSONL Streaming**: Per-detection events with baseline ID, confidence, bandwidth, and classification.
- **Desktop Notifications**: `notify-send` alerts for new detections.
- **Systemd Integration**: Service units for `sdrwatch-control` and `sdrwatch-web`.

---

## 🛠️ Installation (Raspberry Pi 5)

Quick install with the included one-shot installer:

```bash
git clone https://github.com/SDRwatch/sdr-watch.git
cd sdr-watch
chmod +x install-sdrwatch.sh
./install-sdrwatch.sh
```

The installer will:

- Install dependencies (RTL-SDR, SoapySDR, NumPy/SciPy, Flask, etc.).
- Set up a Python venv with system packages.
- Verify hardware (`rtl_test`).
- Apply kernel blacklist + udev rules for RTL2832U dongles.
- Optionally configure + enable **systemd services** for automatic startup.

Non-interactive mode:

```bash
SDRWATCH_AUTO_YES=1 ./install-sdrwatch.sh
```

## Local Development Setup

The installer remains the preferred Raspberry Pi deployment path. For contributor work,
SDRwatch now supports editable installs directly from a repository checkout.

### Supported Platforms

| Platform | No-Hardware Development | Hardware Validation |
| --- | --- | --- |
| Raspberry Pi OS / Linux | Supported | Supported |
| Windows | Supported | Not supported |

### Fresh Venv Install

Linux or Raspberry Pi OS, runtime-only install:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e .
```

Linux or Raspberry Pi OS, contributor install with tests and tooling:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

Windows, contributor install with tests and tooling:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

### Linux and Raspberry Pi Hardware Prerequisites

Hardware validation stays outside pip. Install SDR and numeric prerequisites with the OS package manager first, then add the optional RTL Python extra.

```bash
sudo apt update
sudo apt install -y python3-venv python3-numpy python3-scipy librtlsdr0 librtlsdr-dev rtl-sdr libusb-1.0-0 libusb-1.0-0-dev
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev,rtl]"
```

Optional SoapySDR support remains OS-managed as well, for example via `soapysdr-module-rtlsdr` on Debian or Raspberry Pi OS.

Windows support is limited to no-hardware development. Do not expect RTL-SDR, SoapySDR, `rtl_test`, or live scan validation to work on Windows as part of this workflow.

### No-Hardware Smoke Tests

CLI import smoke test:

```bash
python -c "import sdrwatch.cli; print('cli import ok')"
```

Web app import smoke test:

```bash
python -c "from sdrwatch_web import create_app; create_app('does-not-exist.db'); print('web import ok')"
```

Entry-point help smoke tests:

```bash
python -m sdrwatch.cli --help
python sdrwatch-control.py --help
python sdrwatch-web.py --help
```

Standard no-hardware test command:

```bash
python -m pytest -q tests
```

Simulation-mode smoke test:

```bash
python -m sdrwatch.cli --driver sim --profile fm_broadcast --baseline-id latest --start 88e6 --stop 108e6 --db sim-sdrwatch.db --repeat 2
python -m pytest -q tests/test_sim_mode.py
```

The `latest` baseline alias will reuse the newest baseline in the target DB or create one on first run, so this smoke path works on a fresh checkout with no SDR attached. To inspect the generated data in the dashboard, point the web app at `sim-sdrwatch.db`.
For the full baseline creation, dashboard inspection, and CI-equivalent workflow, see [specs/003-add-sim-mode/quickstart.md](specs/003-add-sim-mode/quickstart.md).

### Hardware-Specific Checks (Linux / Raspberry Pi Only)

Run these separately from the no-hardware smoke tests after OS packages, permissions,
and device access are in place:

```bash
rtl_test -t
python -m sdrwatch.cli --list-profiles
python -m sdrwatch.cli --baseline-id 1 --start 88e6 --stop 108e6 --step 1.8e6 --duration 10
```

---

## 🚀 Usage

### Command Line

All sweeps must be associated with a baseline (`--baseline-id <id>` or `--baseline-id latest`).

Use `--driver rtlsdr_native` for real RTL-SDR scans and `--driver sim` for deterministic no-hardware development and CI validation.

Sweep the FM band once:

```bash
python3 -m sdrwatch.cli --baseline-id 3 --start 88e6 --stop 108e6 --step 1.8e6 \
  --samp-rate 2.4e6 --fft 4096 --avg 8 --driver rtlsdr_native --gain auto
```

Continuous monitoring across 30 MHz – 1.7 GHz:

```bash
python3 -m sdrwatch.cli --baseline-id 3 --start 30e6 --stop 1700e6 --step 2.4e6 \
  --samp-rate 2.4e6 --fft 4096 --avg 8 --driver rtlsdr_native \
  --gain auto --loop --notify --db sdrwatch.db --jsonl events.jsonl
```

Deterministic simulated FM-band scan without hardware:

```bash
python3 -m sdrwatch.cli --driver sim --profile fm_broadcast --baseline-id latest \
  --start 88e6 --stop 108e6 --db sim-sdrwatch.db --repeat 3
```

Inspect the resulting database in the web UI:

```bash
python3 sdrwatch-web.py --db sim-sdrwatch.db --host 0.0.0.0 --port 8080
```

Two-pass verification for refined bandwidth detection:

```bash
python3 -m sdrwatch.cli --baseline-id 3 --start 88e6 --stop 108e6 --step 2.4e6 \
  --samp-rate 2.4e6 --fft 4096 --avg 8 --two-pass \
  --revisit-margin-hz 150e3 --revisit-max-bands 40
```

#### Scan Profiles

Use `--profile <name>` for preset configurations:

| Profile | Description |
| --- | --- |
| `fm_broadcast` | FM band (88-108 MHz), optimized for broadcast detection |
| `vhf_uhf_general` | 400-470 MHz wideband sweep with balanced VHF/UHF defaults |
| `ism_902` | 902-928 MHz ISM band scan tuned for narrowband emitters |

#### Persistence Modes

Use `--persistence-mode` to control how detections are promoted:

| Mode | Description |
| --- | --- |
| `hits` (default) | Requires minimum hits/windows and hit-ratio coverage |
| `duration` | Requires minimum active time |
| `both` | Requires both coverage and duration thresholds |

### Web Dashboard

If installed with services enabled, the dashboard is available at boot:
`http://<raspberrypi-ip>:8080`

Manual launch:

```bash
python3 sdrwatch-web.py --db sdrwatch.db --host 0.0.0.0 --port 8080
```

#### Dashboard Features

- **Signal Cards**: Click any signal card to view details or use inline controls.
- **Classification**: Use dropdown or quick-mark buttons (Friendly/Ambient/Hostile).
- **Selection**: Click the star icon to highlight signals of interest.
- **Labels**: Add short labels like "Base Station" or "Jammer" for quick identification.
- **Notes**: Record observations, timestamps, or investigation notes.
- **Signal Detail**: Click "View details" for full signal page with edit forms.

---

## 🔗 Controller REST API

The controller exposes a REST API consumed by the web frontend or automation:

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/devices` | Enumerate SDRs (key, kind, label, metadata) |
| `GET` | `/jobs` | List jobs (status, params, timestamps) |
| `POST` | `/jobs` | Start a job `{device_key, label, baseline_id, params}` |
| `GET` | `/jobs/<id>` | Inspect a specific job |
| `GET` | `/jobs/<id>/logs?tail=N` | Stream scanner logs |
| `DELETE` | `/jobs/<id>` | Stop the job |
| `GET` | `/profiles` | List available scan profiles |
| `GET` | `/baselines` | List baselines |
| `POST` | `/baselines` | Create a new baseline |

### Signal API

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/api/signals` | List all signals with classification data |
| `GET` | `/api/signals/<id>` | Get signal details |
| `PATCH` | `/api/signals/<id>` | Update classification, label, notes, user_bw_hz |
| `POST` | `/api/signals/<id>/toggle-selected` | Toggle selection status |
| `GET` | `/api/signals/selected` | List selected signals only |

Set `SDRWATCH_CONTROL_TOKEN` for bearer auth. The web app reads `SDRWATCH_CONTROL_URL` and `SDRWATCH_CONTROL_TOKEN` to proxy requests.

---

## 🗄️ Database Schema

Key tables:

| Table | Purpose |
| --- | --- |
| `baselines` | Baseline metadata (name, frequency range, location) |
| `baseline_detections` | Persistent signal records with classification |
| `baseline_noise` | Per-bin noise floor EMA |
| `baseline_occupancy` | Per-bin occupancy counts |
| `scan_updates` | Per-sweep summary statistics |
| `spur_map` | Known spurious signals for calibration |

### Signal Classification Columns

The `baseline_detections` table includes tactical awareness fields:

| Column | Type | Description |
| --- | --- | --- |
| `label` | TEXT | User-defined short label |
| `classification` | TEXT | `friendly`, `ambient`, `hostile`, or `unknown` |
| `user_bw_hz` | INTEGER | User-corrected bandwidth (display only) |
| `notes` | TEXT | Freeform user notes |
| `selected` | INTEGER | Boolean flag for highlighting |

---

## 📑 Bandplan CSV Format

```csv
low_hz,high_hz,service,region,notes
433050000,434790000,ISM,ITU-R1 (EU),Short-range devices
902000000,928000000,ISM,US (FCC),902-928 MHz ISM
2400000000,2483500000,ISM,Global,2.4 GHz ISM
```

---

## 🔍 Inspecting Data

Query recent detections:

```bash
sqlite3 -header -column sdrwatch.db \
  "SELECT id, f_center_hz/1e6 AS MHz, classification, label FROM baseline_detections ORDER BY id DESC LIMIT 20;"
```

Query selected signals:

```bash
sqlite3 -header -column sdrwatch.db \
  "SELECT id, f_center_hz/1e6 AS MHz, classification FROM baseline_detections WHERE selected=1;"
```

Export to CSV:

```bash
sqlite3 -header -csv sdrwatch.db "SELECT * FROM baseline_detections;" > signals.csv
```

---

## 🛣️ Roadmap

- [ ] Additional SDR support (HackRF, Airspy, LimeSDR, USRP via SoapySDR)
- [ ] Duty-cycle analysis for bursty signals
- [ ] Multi-SDR coordination for distributed scanning
- [ ] Enhanced charting and spectrum waterfall
- [ ] Export/import of signal classifications
- [ ] Alert rules based on classification and frequency

---

## 📜 License

MIT License. See [LICENSE](LICENSE).

---

## 🙏 Acknowledgements

Inspired by `rtl_power`, `SoapyPower`, and GNU Radio's `gr-inspector`, extended for **persistent monitoring, baseline tracking, signal classification, and tactical situational awareness**.
