# **architecture.md**

## **Purpose**

SDRwatch is a modular spectrum-monitoring system that performs:

1. **RF acquisition** from an SDR device
2. **Spectrum analysis** (FFT, noise estimation, CFAR, clustering)
3. **Signal detection** and bandwidth characterization
4. **Baseline modeling** (long-term statistics of noise/power/occupancy)
5. **Persistent detection tracking** (new, known, quieted)
6. **Event production** (new signal, power shift, quieted)
7. **Logging + output for external systems**
8. **Control daemon** for job lifecycle
9. **Web UI** for visualization and tactical summaries

Currently this logic is tightly coupled in a single large script.
The refactor aims to separate SDRwatch into well-defined subsystems with clean interfaces.

---

# **High-Level Architecture**

SDRwatch will be organized into these major components:

```
sdrwatch/
    cli.py               # command-line interface wrapping the Sweeper
    sweep/               # sweep orchestration layer
    drivers/             # SDR device backends
    dsp/                 # FFT, CFAR, clustering, noise estimation
    detection/           # segmentation, bandwidth estimation, confidence
    baseline/            # baseline model, stats DB, persistence logic
    io/                  # DB access, bandplan loader, profiles
    util/                # logging, helpers, time, math
```

The system is structured around **pure DSP modules**, **stateful baseline modules**, and **small orchestrators** that tie them together.

---

# **Subsystem Overview**

## **1. Drivers Layer (`drivers/`)**

**Responsibility:**
Abstract physical SDR devices so the rest of the system doesn't care whether the hardware is RTL-SDR, Soapy, HackRF, etc.

**Key interfaces:**

```
class SDRDriver:
    def open(self): ...
    def close(self): ...
    def tune(self, hz: int): ...
    def read_samples(self, count: int) -> np.ndarray: ...
    def set_gain(self, gain: float | str): ...
    @property
    def sample_rate(self) -> float: ...
```

**Modules:**

* `rtlsdr.py`
* `soapy.py`
* `hackrf.py`

---

## **2. DSP Layer (`dsp/`)**

**Responsibility:**
Pure, stateless signal-processing routines.
No DB, no IO, no global state.
Must be fully testable in isolation.

### Modules:

### `windowing.py`

* Hann windows
* Overlap logic
* PSD normalization

### `fft.py`

* FFT computation
* Frequency axis generation
* PSD (power spectrum) computation

### `noise_estimation.py`

* Noise floor estimation per window
* EMA for noise
* Percentile-based estimators

### `detection.py`

* CFAR / thresholding
* Initial segmentation (bins above threshold)
* Gap-merge heuristics
* Segment splitting (future: peak-valley segmentation)
* Local SNR calculation

### `clustering.py`

* Merge segment lists across windows
* Split multi-peak segments
* Compute bandwidth from PSD shape

**Outputs:**
A list of raw, per-window **Segment** objects:

```
Segment {
    f_low_hz
    f_high_hz
    f_center_hz
    peak_db
    noise_db
    snr_db
}
```

---

## **3. Detection Layer (`detection/`)**

Higher-order logic wrapping DSP output.

### Responsibilities:

* Filter segments near spurs
* Bandwidth stabilization
* Confidence scoring
* Mapping window-local segments to persistent/db entries
* Classify segments: new / known / quieted

This layer transforms `Segment` → `DetectionRecord` suitable for database insertion.

---

## **4. Baseline Layer (`baseline/`)**

**Responsibilities:**

* Maintain long-term statistical model per frequency bin
* Maintain persistent detections (evolving “signals of interest”)
* Produce event summaries (new, power-shift, quieted)
* Track total windows, occupancy, power, noise, spur bins

### Modules:

### `model.py`

Defines baseline structure:

```
BaselineModel:
    freq_start_hz
    freq_stop_hz
    bin_hz
    total_windows
```

### `stats.py`

* Baseline stats table (per-bin noise floor EMA, power EMA, occupancy counters)
* Update logic from each sweep window

### `persistence.py`

* Track persistent signals across sweeps
* Update f_low/f_high evolution
* Handle first_seen / last_seen
* Compute confidence
* Handle quieted signals

### `spur.py`

* Track known spur bins
* Spur detection and suppression logic

### `event.py`

Generate database-ready events:

* NEW_SIGNAL
* QUIETED
* POWER_SHIFT

---

## **5. Sweep Orchestration Layer (`sweep/`)**

This replaces the monolithic scanning loop.

### `scheduler.py`

* Expands sweep profile into a list of center frequencies
* Handles revisit queues
* Defines window size, step size, etc.

### `sweeper.py`

Main execution engine:

```
class Sweeper:
    def run(self):
        for window in self.scheduler:
            samples = driver.read(...)
            psd = fft(...)
            segments = detector.detect(...)
            baseline.update(...)
            persist detections
            write logs
```

Responsibilities:

* Bind the driver, DSP, detection, baseline, and output pipelines
* No heavy logic here—just calls into subsystems

---

## **6. DB + IO Layer (`io/`)**

### `db.py`

All SQL consolidated here:

* Insert scans
* Insert detections
* Update baseline_stats
* Update baseline_detections
* Query helpers
* WAL settings

### `bandplan.py`

Load bandplan CSV, map frequency ranges to services.

### `profiles.py`

Hold scan profiles as structured dataclasses:

```
@dataclass
class ScanProfile:
    samp_rate: float
    fft_bins: int
    dwell_ms: int
    threshold_db: float
    merge_gap_hz: float
    ...
```

The CLI and control daemon will use `ScanProfile` objects instead of raw dicts.

---

## **7. Utilities Layer (`util/`)**

### Modules:

* `logging.py` → structured logging formatting
* `math.py` → SNR, dB conversions
* `time.py` → UTC, ISO formatting
* `geo.py` → lat/lon helpers if needed

Everything that does not belong to a subsystem stays here.

---

# **CLI and Frontend Layers (Outside the Core Package)**

## **8. CLI (`cli.py`)**

Thin wrapper around Sweeper:

* Parses CLI arguments
* Loads a profile
* Starts sweep
* Writes logs to stdout or JSONL
* No algorithmic logic here

---

## **9. Control Daemon (`sdrwatch-control.py`)**

**(Already works well; minimal change needed)**

Responsibilities:

* Job lifecycle
* Device enumeration
* IPC to scanner
* HTTP API used by web UI
* Exclusive locking

Only required change: point the scanner command to `python -m sdrwatch.cli ...`

---

## **10. Web Layer (`sdrwatch-web-simple.py`)**

**(Already decoupled; no refactor required)**

Responsibilities:

* Render dashboards
* Query database
* Hit control daemon API
* No coupling to scanner internals

After refactor, the DB schema remains consistent, so the web layer should need **zero** changes beyond file paths.

---

# **Data Flow Diagram (Conceptual)**

```
 SDRDriver          DSP (FFT/CFAR)       Detection Layer         Baseline Layer          DB
    |                    |                     |                       |                 |
    | samples →          | PSD →               | segments →            | updates →       |
    |------------------→ |------------------→  |----------------------→ |----------------→|
                         |                     |                        |
                         |                     | persistent dets →      |
                         |                     | events →               |
```

---

# **Design Principles**

1. **Separation of concerns**:
   DSP is pure; baseline is stateful; sweeper is orchestration.

2. **Testability**:
   DSP modules can be tested with synthetic I/Q; baseline modules can be tested with synthetic segments.

3. **No side effects in DSP**:
   No DB writes, no logs, no mutations.

4. **Profiles as configuration objects**:
   No large dict objects passed around.

5. **Stable interfaces**:
   Each subsystem exposes a clear, minimal API.

6. **Compatibility with current DB schema**:
   No database changes required for now.

---

# **Refactor Plan Summary**

### **Phase 1 — Structural split**

* Break up the monolithic script into modules without any logic change.

### **Phase 2 — Isolate DSP**

* Extract FFT, windowing, CFAR, clustering, noise modeling.

### **Phase 3 — Consolidate baseline**

* Move all baseline logic (stats, persistent signals, spur, events) into `baseline/`.

### **Phase 4 — Rewrite the sweeper**

* Build a small orchestration engine that calls into subsystems.

### **Phase 5 — Update CLI**

* Replace the current direct invocation with `sdrwatch.cli`.

### **Phase 6 — Validate control + web**

* Minimal path updates, no functional refactors needed.

---
