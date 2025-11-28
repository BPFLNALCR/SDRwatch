#!/usr/bin/env python3
"""
SDRWatch — wideband scanner, baseline builder, and bandplan mapper

Goals
-----
- Sweep a frequency range with an SDR (SoapySDR backend by default; optional native RTL-SDR backend).
- Estimate noise floor robustly and detect signals via energy thresholding (CFAR‑like).
- Build a baseline (per‑bin occupancy over time) and flag "new" signals relative to that baseline.
- Map detections to a bandplan (FCC/CEPT/etc.) from a CSV file or built‑in minimal defaults.
- Log everything to SQLite and optionally emit desktop notifications or webhook JSON lines.

Hardware
--------
Any SoapySDR‑supported device (RTL‑SDR, HackRF, Airspy, SDRplay, LimeSDR, USRP...).
Alternatively, a native RTL-SDR path via pyrtlsdr (librtlsdr) is available with --driver rtlsdr_native.

Trixie notes
------------
- Python 3.12/3.13 compatible (no distutils; avoids deprecated numpy aliases; has
  a robust sliding_window_view fallback).
- Allows overriding TMPDIR via --tmpdir or environment to steer scratch I/O off
  /tmp (which is tmpfs on Raspberry Pi OS Trixie).
- Minor resilience around SoapySDR readStream return types.

DB schema (SQLite)
------------------
- scans(id INTEGER PK, t_start_utc TEXT, t_end_utc TEXT, f_start_hz INT, f_stop_hz INT, step_hz INT, samp_rate INT, fft INT, avg INT,
        device TEXT, driver TEXT)
- detections(scan_id INT, time_utc TEXT, f_center_hz INT, f_low_hz INT, f_high_hz INT, peak_db REAL, noise_db REAL, snr_db REAL,
             service TEXT, region TEXT, notes TEXT)
- baseline(bin_hz INT PK, ema_occ REAL, ema_power_db REAL, last_seen_utc TEXT, total_obs INT, hits INT)

License: MIT
"""

import argparse
import csv
import json
import math
import os
import sqlite3
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Set, Tuple

import numpy as np  # type: ignore

# SciPy is optional. If missing, we fall back to a simple periodogram average.
try:
    from scipy.signal import welch  # type: ignore

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# SoapySDR is optional (used for multi-device support).
try:
    import SoapySDR  # type: ignore
    from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX  # type: ignore

    HAVE_SOAPY = True
except Exception:
    HAVE_SOAPY = False

# pyrtlsdr (native RTL-SDR path) is optional
try:
    from rtlsdr import RtlSdr  # type: ignore

    HAVE_RTLSDR = True
except Exception:
    HAVE_RTLSDR = False

# ------------------------------
# Utility
# ------------------------------

def utc_now_str() -> str:
    return datetime.now(timezone.utc).isoformat()


def db10(x: np.ndarray) -> np.ndarray:
    # avoid log(0)
    return 10.0 * np.log10(np.maximum(x, 1e-20))


def robust_noise_floor_db(psd_db: np.ndarray) -> float:
    """Robust noise floor estimate using median + 1.4826*MAD (approx std for Gaussian)."""
    med = np.median(psd_db)
    mad = np.median(np.abs(psd_db - med))
    return float(med + 1.4826 * mad)


@dataclass
class Segment:
    f_low_hz: int
    f_high_hz: int
    f_center_hz: int
    peak_db: float
    noise_db: float
    snr_db: float


@dataclass
class ScanProfile:
    name: str
    f_low_hz: int
    f_high_hz: int
    samp_rate: float
    fft: int
    avg: int
    gain_db: float
    threshold_db: float
    min_width_bins: int
    guard_bins: int
    abs_power_floor_db: Optional[float] = None


def default_scan_profiles() -> Dict[str, ScanProfile]:
    profiles = [
        ScanProfile(
            name="vhf_uhf_general",
            f_low_hz=400_000_000,
            f_high_hz=470_000_000,
            samp_rate=2.4e6,
            fft=4096,
            avg=8,
            gain_db=20.0,
            threshold_db=10.0,
            min_width_bins=3,
            guard_bins=1,
            abs_power_floor_db=-95.0,
        ),
        ScanProfile(
            name="fm_broadcast",
            f_low_hz=88_000_000,
            f_high_hz=108_000_000,
            samp_rate=2.0e6,
            fft=4096,
            avg=6,
            gain_db=12.0,
            threshold_db=8.0,
            min_width_bins=4,
            guard_bins=1,
            abs_power_floor_db=-90.0,
        ),
        ScanProfile(
            name="ism_902",
            f_low_hz=902_000_000,
            f_high_hz=928_000_000,
            samp_rate=2.4e6,
            fft=4096,
            avg=10,
            gain_db=25.0,
            threshold_db=12.0,
            min_width_bins=3,
            guard_bins=1,
            abs_power_floor_db=-92.0,
        ),
    ]
    return {p.name.lower(): p for p in profiles}


def _emit_profiles_json() -> None:
    profiles = default_scan_profiles()
    ordered = sorted(profiles.values(), key=lambda p: p.name.lower())
    payload = {
        "profiles": [
            {
                "name": prof.name,
                "f_low_hz": prof.f_low_hz,
                "f_high_hz": prof.f_high_hz,
                "samp_rate": prof.samp_rate,
                "fft": prof.fft,
                "avg": prof.avg,
                "gain_db": prof.gain_db,
                "threshold_db": prof.threshold_db,
                "min_width_bins": prof.min_width_bins,
                "guard_bins": prof.guard_bins,
                "abs_power_floor_db": prof.abs_power_floor_db,
            }
            for prof in ordered
        ]
    }
    print(json.dumps(payload))


# ------------------------------
# Bandplan CSV lookup (minimal)
# ------------------------------

@dataclass
class Band:
    low_hz: int
    high_hz: int
    service: str
    region: str
    notes: str


class Bandplan:
    def __init__(self, csv_path: Optional[str] = None):
        self.bands: List[Band] = []
        if csv_path and os.path.exists(csv_path):
            self._load_csv(csv_path)
        else:
            # Minimal defaults. Extend with official CSVs.
            self.bands = [
                Band(433_050_000, 434_790_000, "ISM/SRD", "ITU-R1 (EU)", "Short-range devices"),
                Band(902_000_000, 928_000_000, "ISM", "US (FCC)", "902-928 MHz ISM"),
                Band(2_400_000_000, 2_483_500_000, "ISM", "Global", "2.4 GHz ISM"),
                Band(1_420_000_000, 1_427_000_000, "Radio Astronomy", "Global", "Hydrogen line"),
                Band(88_000_000, 108_000_000, "FM Broadcast", "Global", "88-108 MHz Radio"),
            ]

    def _load_csv(self, path: str):
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # Prefer old headers, but accept fallback names if present
                    low = row.get("low_hz") or row.get("f_low_hz")
                    high = row.get("high_hz") or row.get("f_high_hz")
                    if low is None or high is None:
                        continue  # skip rows with missing frequency bounds
                    self.bands.append(
                        Band(
                            int(float(low)),
                            int(float(high)),
                            (row.get("service") or "").strip(),
                            (row.get("region") or "").strip(),
                            (row.get("notes") or "").strip(),
                        )
                    )
                except Exception:
                    continue

    def lookup(self, f_hz: int) -> Tuple[str, str, str]:
        for b in self.bands:
            if b.low_hz <= f_hz <= b.high_hz:
                return b.service, b.region, b.notes
        return "", "", ""

# ------------------------------
# SQLite store
# ------------------------------

class Store:
    def __init__(self, path: str):
        # SQLite 3.44+ on Trixie — keep default isolation; WAL can be enabled via pragma if needed.
        self.con = sqlite3.connect(path, timeout=30.0)
        try:
            self.con.execute("PRAGMA journal_mode=WAL")
        except sqlite3.OperationalError:
            pass
        self.con.execute("PRAGMA busy_timeout=5000")
        self._init()

    def _init(self):
                cur = self.con.cursor()
                cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS scans (
                                id INTEGER PRIMARY KEY AUTOINCREMENT,
                                t_start_utc TEXT,
                                t_end_utc   TEXT,
                                f_start_hz  INTEGER,
                                f_stop_hz   INTEGER,
                                step_hz     INTEGER,
                                samp_rate   INTEGER,
                                fft         INTEGER,
                                avg         INTEGER,
                                device      TEXT,
                                driver      TEXT
                        )
                        """
                )
                cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS detections (
                            scan_id     INTEGER,
                            time_utc    TEXT,
                            f_center_hz INTEGER,
                            f_low_hz    INTEGER,
                            f_high_hz   INTEGER,
                            peak_db     REAL,
                            noise_db    REAL,
                            snr_db      REAL,
                            service     TEXT,
                            region      TEXT,
                            notes       TEXT,
                            confidence  REAL
                        )
                        """
                )
                cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS baseline (
                            bin_hz      INTEGER PRIMARY KEY,
                            ema_occ     REAL,
                            ema_power_db REAL,
                            last_seen_utc TEXT,
                            total_obs   INTEGER,
                            hits        INTEGER
                        )
                        """
                )
                cur.execute(
                        """
                        CREATE TABLE IF NOT EXISTS spur_map (
                            bin_hz        INTEGER PRIMARY KEY,
                            mean_power_db REAL,
                            hits          INTEGER,
                            last_seen_utc TEXT
                        )
                        """
                )
                self.con.commit()
                # Backfill new nullable columns when upgrading existing deployments
                self._ensure_column("scans", "latitude", "latitude REAL")
                self._ensure_column("scans", "longitude", "longitude REAL")
                self._ensure_column("detections", "confidence", "confidence REAL")

    # Transaction helpers
    def begin(self):
        self.con.execute("BEGIN")

    def commit(self):
        self.con.commit()

    def _ensure_column(self, table: str, column: str, ddl: str) -> None:
        cur = self.con.execute(f"PRAGMA table_info({table})")
        existing = {row[1] for row in cur.fetchall()}
        if column not in existing:
            self.con.execute(f"ALTER TABLE {table} ADD COLUMN {ddl}")
            self.con.commit()

    def start_scan(self, meta: dict) -> int:
        cur = self.con.cursor()
        cur.execute(
            """
            INSERT INTO scans(t_start_utc, t_end_utc, f_start_hz, f_stop_hz, step_hz, samp_rate, fft, avg, device, driver, latitude, longitude)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                meta.get("t_start_utc"),
                meta.get("t_end_utc"),
                meta.get("f_start_hz"),
                meta.get("f_stop_hz"),
                meta.get("step_hz"),
                meta.get("samp_rate"),
                meta.get("fft"),
                meta.get("avg"),
                meta.get("device"),
                meta.get("driver"),
                meta.get("latitude"),
                meta.get("longitude"),
            ),
        )
        self.con.commit()
        if cur.lastrowid is None:
            raise RuntimeError("Failed to retrieve lastrowid from scan insert")
        return int(cur.lastrowid)

    def end_scan(self, scan_id: int, t_end_utc: str):
        self.con.execute("UPDATE scans SET t_end_utc = ? WHERE id = ?", (t_end_utc, scan_id))
        self.con.commit()

    def add_detection(self, scan_id: int, seg: Segment, service: str, region: str, notes: str, confidence: Optional[float] = None):
        self.con.execute(
            """
            INSERT INTO detections(scan_id, time_utc, f_center_hz, f_low_hz, f_high_hz, peak_db, noise_db, snr_db, service, region, notes, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                scan_id,
                utc_now_str(),
                seg.f_center_hz,
                seg.f_low_hz,
                seg.f_high_hz,
                seg.peak_db,
                seg.noise_db,
                seg.snr_db,
                service,
                region,
                notes,
                float(confidence) if confidence is not None else None,
            ),
        )

    def update_baseline(self, freqs_hz: np.ndarray, psd_db: np.ndarray, occupied_mask: np.ndarray, ema_alpha: float = 0.05):
        cur = self.con.cursor()
        tnow = utc_now_str()
        for f, p, occ in zip(freqs_hz.astype(int), psd_db.astype(float), occupied_mask.astype(int)):
            cur.execute("SELECT ema_occ, ema_power_db, last_seen_utc, total_obs, hits FROM baseline WHERE bin_hz = ?", (int(f),))
            row = cur.fetchone()
            if row is None:
                ema_occ = occ
                ema_pow = p
                tot = 1
                hits = occ
            else:
                ema_occ_prev, ema_pow_prev, _, total_obs, hits_prev = row
                ema_occ = (1 - ema_alpha) * (ema_occ_prev if ema_occ_prev is not None else 0.0) + ema_alpha * occ
                ema_pow = (1 - ema_alpha) * (ema_pow_prev if ema_pow_prev is not None else p) + ema_alpha * p
                tot = (total_obs or 0) + 1
                hits = (hits_prev or 0) + occ
            cur.execute(
                """
                INSERT INTO baseline(bin_hz, ema_occ, ema_power_db, last_seen_utc, total_obs, hits)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(bin_hz) DO UPDATE SET
                    ema_occ=excluded.ema_occ,
                    ema_power_db=excluded.ema_power_db,
                    last_seen_utc=excluded.last_seen_utc,
                    total_obs=excluded.total_obs,
                    hits=excluded.hits
                """,
                (int(f), float(ema_occ), float(ema_pow), tnow, int(tot), int(hits)),
            )

    def baseline_occ(self, f_center_hz: int) -> Optional[float]:
        cur = self.con.cursor()
        cur.execute("SELECT ema_occ FROM baseline WHERE bin_hz = ?", (int(f_center_hz),))
        row = cur.fetchone()
        return float(row[0]) if row and row[0] is not None else None

    def update_spur_bin(self, bin_hz: int, power_db: float, hits_increment: int = 1, ema_alpha: float = 0.2):
        cur = self.con.cursor()
        cur.execute("SELECT mean_power_db, hits FROM spur_map WHERE bin_hz = ?", (int(bin_hz),))
        row = cur.fetchone()
        tnow = utc_now_str()
        if row is None:
            cur.execute(
                """
                INSERT INTO spur_map(bin_hz, mean_power_db, hits, last_seen_utc)
                VALUES (?, ?, ?, ?)
                """,
                (int(bin_hz), float(power_db), int(max(1, hits_increment)), tnow),
            )
        else:
            prev_mean, prev_hits = row
            prev_mean = float(prev_mean) if prev_mean is not None else float(power_db)
            prev_hits = int(prev_hits or 0)
            new_mean = (1.0 - ema_alpha) * prev_mean + ema_alpha * float(power_db)
            cur.execute(
                """
                UPDATE spur_map
                SET mean_power_db = ?, hits = ?, last_seen_utc = ?
                WHERE bin_hz = ?
                """,
                (float(new_mean), int(prev_hits + max(1, hits_increment)), tnow, int(bin_hz)),
            )
        self.con.commit()

    def lookup_spur(self, f_center_hz: int, tolerance_hz: int = 5_000) -> Optional[Tuple[int, float, int]]:
        cur = self.con.cursor()
        low = int(f_center_hz - tolerance_hz)
        high = int(f_center_hz + tolerance_hz)
        cur.execute(
            """
            SELECT bin_hz, mean_power_db, hits
            FROM spur_map
            WHERE bin_hz BETWEEN ? AND ?
            ORDER BY ABS(bin_hz - ?)
            LIMIT 1
            """,
            (low, high, int(f_center_hz)),
        )
        row = cur.fetchone()
        if not row:
            return None
        bin_hz_val, mean_power_db, hits = row
        return int(bin_hz_val), float(mean_power_db), int(hits)


# ------------------------------
# SDR sources
# ------------------------------

class SDRSource:
    """SoapySDR source (generic)."""

    def __init__(self, driver: str, samp_rate: float, gain: str | float, soapy_args: Optional[Dict[str, str]] = None):
        if not HAVE_SOAPY:
            raise RuntimeError("SoapySDR not available")
        dev_args: Dict[str, str] = {"driver": driver}
        if soapy_args:
            dev_args.update({str(k): str(v) for k, v in soapy_args.items()})
        self.dev = SoapySDR.Device(dev_args)
        self.dev.setSampleRate(SOAPY_SDR_RX, 0, samp_rate)
        if isinstance(gain, str) and gain == "auto":
            try:
                self.dev.setGainMode(SOAPY_SDR_RX, 0, True)
            except Exception:
                pass
        else:
            self.dev.setGain(SOAPY_SDR_RX, 0, float(gain))
        self.stream = self.dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.dev.activateStream(self.stream)

    def tune(self, center_hz: float):
        self.dev.setFrequency(SOAPY_SDR_RX, 0, center_hz)

    def read(self, count: int) -> np.ndarray:
        buffs: List[np.ndarray] = []
        got = 0
        while got < count:
            sr = int(min(8192, count - got))
            buff = np.empty(sr, dtype=np.complex64)
            st = self.dev.readStream(self.stream, [buff], sr)
            # Soapy returns either an int or a structure with .ret
            n = getattr(st, "ret", st)
            if isinstance(n, tuple):
                n = n[0]  # extreme edge cases
            if isinstance(n, (list, np.ndarray)):
                n = int(n[0])
            if int(n) > 0:
                buffs.append(buff[: int(n)])
                got += int(n)
            else:
                time.sleep(0.001)
        if not buffs:
            return np.zeros(count, dtype=np.complex64)
        return np.concatenate(buffs)

    def close(self):
        try:
            self.dev.deactivateStream(self.stream)
            self.dev.closeStream(self.stream)
        except Exception:
            pass


class RTLSDRSource:
    """Native librtlsdr via pyrtlsdr."""

    def __init__(self, samp_rate: float, gain: str | float, *, device_index: Optional[int] = None, serial_number: Optional[str] = None):
        if not HAVE_RTLSDR:
            raise RuntimeError("pyrtlsdr not available")
        # Prefer targeting by serial, else index, else default
        if serial_number:
            self.dev = RtlSdr(serial_number=str(serial_number))
        elif device_index is not None:
            self.dev = RtlSdr(device_index=int(device_index))
        else:
            self.dev = RtlSdr()
        self.dev.sample_rate = samp_rate
        if isinstance(gain, str) and gain == "auto":
            self.dev.gain = "auto"
        else:
            self.dev.gain = float(gain)

    def tune(self, center_hz: float):
        self.dev.center_freq = center_hz

    def read(self, count: int) -> np.ndarray:
        # pyrtlsdr returns np.complex64
        return self.dev.read_samples(count)

    def close(self):
        try:
            self.dev.close()
        except Exception:
            pass


# ------------------------------
# CFAR helpers
# ------------------------------

def _sliding_window_view(x: np.ndarray, window: int) -> np.ndarray:
    """Return a sliding window view over the last axis. Fallback if not available."""
    try:
        return np.lib.stride_tricks.sliding_window_view(x, window)
    except Exception:
        # Minimal fallback for 1D arrays
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("sliding window fallback only supports 1D arrays")
        shape = (x.size - window + 1, window)
        if shape[0] <= 0:
            return np.empty((0, window), dtype=x.dtype)
        strides = (x.strides[0], x.strides[0])
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


def cfar_os_mask(psd_db: np.ndarray, train: int, guard: int, quantile: float, alpha_db: float) -> Tuple[np.ndarray, np.ndarray]:
    """Order-Statistic CFAR (OS-CFAR) on a 1D PSD in dB.
    Returns (mask_above, noise_est_db_per_bin). The threshold is noise_est + alpha_db, applied in **linear** power domain.
    """
    psd_db = np.asarray(psd_db).astype(np.float64)
    N = psd_db.size
    if N == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    # Convert to linear power
    psd_lin = np.power(10.0, psd_db / 10.0)
    # Build sliding windows with padding so we have a window for each bin
    win = 2 * train + 2 * guard + 1
    if win <= 1:
        # Degenerate: no training cells; fall back to global median as noise
        noise_db = np.full(N, float(np.median(psd_db)))
        above = psd_db > (noise_db + alpha_db)
        return above, noise_db
    pad = train + guard
    padded = np.pad(psd_lin, (pad, pad), mode="edge")
    windows = _sliding_window_view(padded, win)  # shape (N, win)
    # Exclude guard + CUT region by masking them out
    mask = np.ones(win, dtype=bool)
    mask[train : train + 2 * guard + 1] = False  # False over guard + CUT
    train_windows = windows[:, mask]  # shape (N, 2*train)
    # Order statistic via quantile over training cells
    q = float(np.clip(quantile, 1e-6, 1.0 - 1e-6))
    noise_lin = np.quantile(train_windows, q, axis=1)
    alpha = np.power(10.0, alpha_db / 10.0)
    threshold_lin = noise_lin * alpha
    above = psd_lin > threshold_lin
    noise_db = 10.0 * np.log10(np.maximum(noise_lin, 1e-20))
    return above, noise_db


# ------------------------------
# Detection & PSD
# ------------------------------

def detect_segments(
    freqs_hz: np.ndarray,
    psd_db: np.ndarray,
    thresh_db: float,
    guard_bins: int = 1,
    min_width_bins: int = 2,
    cfar_mode: str = "off",
    cfar_train: int = 24,
    cfar_guard: int = 4,
    cfar_quantile: float = 0.75,
    cfar_alpha_db: Optional[float] = None,
    abs_power_floor_db: Optional[float] = None,
) -> Tuple[List[Segment], np.ndarray, np.ndarray]:
    """Detect contiguous energy segments.
    If cfar_mode != 'off', use OS-CFAR to produce the detection mask. Otherwise use a global robust noise floor.
    Returns (segments, above_mask, noise_est_db_per_bin).
    """
    psd_db = np.asarray(psd_db).astype(np.float64)
    freqs_hz = np.asarray(freqs_hz).astype(np.float64)
    N = psd_db.size
    if N == 0:
        return [], np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)

    if cfar_mode and cfar_mode.lower() != "off":
        alpha_db = float(cfar_alpha_db if cfar_alpha_db is not None else thresh_db)
        above, noise_local_db = cfar_os_mask(psd_db, cfar_train, cfar_guard, cfar_quantile, alpha_db)
        noise_for_snr_db = noise_local_db
    else:
        # Global robust threshold
        nf = robust_noise_floor_db(psd_db)
        dynamic = nf + float(thresh_db)
        above = psd_db > dynamic
        noise_for_snr_db = np.full(N, nf, dtype=np.float64)

    # Merge small gaps (guard_bins) and form contiguous segments
    segs: List[Segment] = []
    i = 0
    while i < N:
        if bool(above[i]):
            start_i = i
            j = i + 1
            gap = 0
            while j < N and (bool(above[j]) or gap < guard_bins):
                if bool(above[j]):
                    gap = 0
                else:
                    gap += 1
                j += 1
            end_i = j  # exclusive
            # Ensure minimum width
            if (end_i - start_i) >= min_width_bins:
                sl = slice(start_i, end_i)
                peak_idx_local = int(np.argmax(psd_db[sl]))
                peak_idx = start_i + peak_idx_local
                peak_db = float(psd_db[peak_idx])
                # Representative noise for SNR = local noise at the peak bin
                noise_db = float(noise_for_snr_db[peak_idx])
                snr_db = float(peak_db - noise_db)
                # freq bounds (use bin edges assuming uniform spacing)
                f_low = float(freqs_hz[start_i])
                f_high = float(freqs_hz[end_i - 1])
                f_center = float(freqs_hz[(start_i + end_i) // 2])
                segs.append(
                    Segment(
                        f_low_hz=int(round(f_low)),
                        f_high_hz=int(round(f_high)),
                        f_center_hz=int(round(f_center)),
                        peak_db=peak_db,
                        noise_db=noise_db,
                        snr_db=snr_db,
                    )
                )
            i = j
        else:
            i += 1

    if abs_power_floor_db is not None:
        floor = float(abs_power_floor_db)
        segs = [seg for seg in segs if seg.peak_db >= floor]

    return segs, above, noise_for_snr_db


@dataclass
class DetectionCluster:
    f_low_hz: int
    f_high_hz: int
    first_seen_ts: str
    last_seen_ts: str
    first_window: int
    last_window: int
    hits: int = 0
    windows: Set[int] = field(default_factory=set)
    best_seg: Segment = field(default_factory=lambda: Segment(0, 0, 0, -999.0, -999.0, -999.0))
    emitted: bool = False


class DetectionEngine:
    def __init__(
        self,
        store: "Store",
        bandplan: Bandplan,
        args,
        *,
        bin_hz: float,
        min_hits: int = 2,
        min_windows: int = 2,
        max_gap_windows: int = 3,
        freq_merge_hz: Optional[float] = None,
    ):
        self.store = store
        self.bandplan = bandplan
        self.args = args
        self.bin_hz = float(bin_hz)
        self.min_hits = max(1, int(min_hits))
        self.min_windows = max(1, int(min_windows))
        self.max_gap_windows = max(1, int(max_gap_windows))
        self.freq_merge_hz = float(freq_merge_hz if freq_merge_hz is not None else max(self.bin_hz * 2, 25_000.0))
        self.min_width_hz = max(self.bin_hz * float(args.min_width_bins), self.bin_hz)
        self.clusters: List[DetectionCluster] = []
        self._last_window_idx = -1
        self.spur_tolerance_hz = 5_000.0
        self.spur_margin_db = 4.0
        self.spur_min_hits = 5
        self.spur_override_snr = 10.0
        self.spur_penalty_max = 0.35
        self._pending_emits = 0

    def ingest(self, scan_id: int, window_idx: int, segments: List[Segment]) -> Tuple[int, int, int]:
        self._last_window_idx = max(self._last_window_idx, window_idx)
        accepted = 0
        spur_ignored = 0
        if not segments:
            self._prune_clusters(scan_id, window_idx)
            return accepted, spur_ignored, self._drain_pending_emits()
        timestamp = utc_now_str()
        for seg in segments:
            if self._spur_should_ignore(seg):
                spur_ignored += 1
                continue
            self._record_hit(scan_id, window_idx, seg, timestamp)
            accepted += 1
        self._prune_clusters(scan_id, window_idx)
        return accepted, spur_ignored, self._drain_pending_emits()

    def flush(self, scan_id: int) -> int:
        self._prune_clusters(scan_id, self._last_window_idx if self._last_window_idx >= 0 else 0, force=True)
        return self._drain_pending_emits()

    def _record_hit(self, scan_id: int, window_idx: int, seg: Segment, timestamp: str):
        cluster = self._find_cluster(seg)
        if cluster is None:
            cluster = DetectionCluster(
                f_low_hz=seg.f_low_hz,
                f_high_hz=seg.f_high_hz,
                first_seen_ts=timestamp,
                last_seen_ts=timestamp,
                first_window=window_idx,
                last_window=window_idx,
                hits=1,
                windows={window_idx},
                best_seg=seg,
            )
            self.clusters.append(cluster)
        else:
            cluster.f_low_hz = min(cluster.f_low_hz, seg.f_low_hz)
            cluster.f_high_hz = max(cluster.f_high_hz, seg.f_high_hz)
            cluster.last_seen_ts = timestamp
            cluster.last_window = window_idx
            cluster.hits += 1
            cluster.windows.add(window_idx)
            if seg.snr_db >= cluster.best_seg.snr_db:
                cluster.best_seg = seg

        self._maybe_emit_cluster(scan_id, cluster)

    def _find_cluster(self, seg: Segment) -> Optional[DetectionCluster]:
        for cluster in self.clusters:
            if self._segments_overlap(cluster, seg):
                return cluster
        return None

    def _segments_overlap(self, cluster: DetectionCluster, seg: Segment) -> bool:
        return not (
            seg.f_high_hz < (cluster.f_low_hz - self.freq_merge_hz)
            or seg.f_low_hz > (cluster.f_high_hz + self.freq_merge_hz)
        )

    def _maybe_emit_cluster(self, scan_id: int, cluster: DetectionCluster):
        if cluster.emitted:
            return
        if not self._cluster_qualifies(cluster):
            return
        self._emit_detection(scan_id, cluster)

    def _cluster_qualifies(self, cluster: DetectionCluster) -> bool:
        width_hz = float(cluster.f_high_hz - cluster.f_low_hz)
        return (
            cluster.hits >= self.min_hits
            and len(cluster.windows) >= self.min_windows
            and width_hz >= self.min_width_hz
        )

    def _emit_detection(self, scan_id: int, cluster: DetectionCluster):
        cluster.emitted = True
        best_seg = cluster.best_seg
        confidence = self._compute_confidence(cluster)
        combined_seg = Segment(
            f_low_hz=cluster.f_low_hz,
            f_high_hz=cluster.f_high_hz,
            f_center_hz=int((cluster.f_low_hz + cluster.f_high_hz) / 2),
            peak_db=best_seg.peak_db,
            noise_db=best_seg.noise_db,
            snr_db=best_seg.snr_db,
        )
        svc, reg, note = self.bandplan.lookup(combined_seg.f_center_hz)

        self.store.begin()
        self.store.add_detection(scan_id, combined_seg, svc, reg, note, confidence)
        self.store.commit()
        self._pending_emits += 1

        occ = self.store.baseline_occ(combined_seg.f_center_hz)
        is_new = occ is not None and occ < self.args.new_ema_occ

        record = {
            "time_utc": utc_now_str(),
            "f_center_hz": combined_seg.f_center_hz,
            "f_low_hz": combined_seg.f_low_hz,
            "f_high_hz": combined_seg.f_high_hz,
            "peak_db": combined_seg.peak_db,
            "noise_db": combined_seg.noise_db,
            "snr_db": combined_seg.snr_db,
            "service": svc,
            "region": reg,
            "notes": note,
            "is_new": bool(is_new),
            "confidence": confidence,
        }
        maybe_emit_jsonl(self.args.jsonl, record)
        if is_new:
            body = f"{combined_seg.f_center_hz/1e6:.6f} MHz; SNR {combined_seg.snr_db:.1f} dB; {svc or 'Unknown'} {reg or ''}"
            maybe_notify("SDRWatch: New signal", body, self.args.notify)

    def _prune_clusters(self, scan_id: int, window_idx: int, force: bool = False):
        to_remove: List[DetectionCluster] = []
        for cluster in self.clusters:
            gap = window_idx - cluster.last_window
            if force or gap > self.max_gap_windows:
                if not cluster.emitted and self._cluster_qualifies(cluster):
                    self._emit_detection(scan_id, cluster)
                to_remove.append(cluster)
        for cluster in to_remove:
            self.clusters.remove(cluster)

    def _spur_should_ignore(self, seg: Segment) -> bool:
        if getattr(self.args, "spur_calibration", False):
            return False
        spur = self.store.lookup_spur(seg.f_center_hz, int(self.spur_tolerance_hz))
        if not spur:
            return False
        _, mean_power_db, hits = spur
        if hits < self.spur_min_hits:
            return False
        if seg.peak_db >= mean_power_db + self.spur_margin_db:
            return False
        if seg.snr_db >= self.spur_override_snr:
            return False
        return True

    def _compute_confidence(self, cluster: DetectionCluster) -> float:
        best_seg = cluster.best_seg
        snr_component = float(np.clip(best_seg.snr_db / 30.0, 0.0, 1.0))
        hit_component = float(np.clip(cluster.hits / 6.0, 0.0, 1.0))
        span_windows = max(cluster.last_window - cluster.first_window + 1, 1)
        persistence_component = float(np.clip(len(cluster.windows) / span_windows, 0.0, 1.0))
        duration_component = float(np.clip(span_windows / 8.0, 0.0, 1.0))
        raw_score = (
            0.45 * snr_component
            + 0.25 * hit_component
            + 0.2 * persistence_component
            + 0.1 * duration_component
        )
        penalty = self._spur_confidence_penalty(cluster)
        return float(np.clip(raw_score - penalty, 0.0, 1.0))

    def _spur_confidence_penalty(self, cluster: DetectionCluster) -> float:
        if getattr(self.args, "spur_calibration", False):
            return 0.0
        seg = cluster.best_seg
        spur = self.store.lookup_spur(seg.f_center_hz, int(self.spur_tolerance_hz))
        if not spur:
            return 0.0
        _, mean_power_db, hits = spur
        if hits < self.spur_min_hits:
            return 0.0
        diff = float(seg.peak_db - mean_power_db)
        if diff >= self.spur_margin_db + 5.0:
            return 0.05
        if diff >= self.spur_margin_db:
            return min(0.15, self.spur_penalty_max)
        return self.spur_penalty_max

    def _drain_pending_emits(self) -> int:
        emitted = self._pending_emits
        self._pending_emits = 0
        return emitted


class WindowPowerMonitor:
    def __init__(self, spike_db: float = 8.0, ema_alpha: float = 0.2, warmup_windows: int = 3):
        self.spike_db = float(spike_db)
        self.ema_alpha = float(np.clip(ema_alpha, 1e-3, 1.0))
        self.warmup_windows = max(0, int(warmup_windows))
        self.ema: Optional[float] = None
        self.count = 0

    def update(self, mean_db: float) -> Tuple[bool, float, float]:
        self.count += 1
        if self.ema is None:
            self.ema = mean_db
            return False, mean_db, 0.0
        delta = float(mean_db - self.ema)
        is_anom = self.count > self.warmup_windows and delta > self.spike_db
        self.ema = (1.0 - self.ema_alpha) * self.ema + self.ema_alpha * mean_db
        return is_anom, float(self.ema), delta


def _track_spur_hits(tracker: Dict[int, Dict[str, float]], segments: List[Segment]):
    for seg in segments:
        entry = tracker.setdefault(seg.f_center_hz, {"hits": 0.0, "power_sum": 0.0})
        entry["hits"] += 1.0
        entry["power_sum"] += float(seg.peak_db)


def _persist_spur_calibration(store: Store, tracker: Dict[int, Dict[str, float]], total_windows: int, min_ratio: float = 0.6):
    if not total_windows:
        return
    min_hits = max(1, int(total_windows * min_ratio))
    for bin_hz, stats in tracker.items():
        hits = int(stats.get("hits", 0))
        if hits < min_hits:
            continue
        avg_power = float(stats.get("power_sum", 0.0)) / float(max(hits, 1))
        store.update_spur_bin(bin_hz, avg_power, hits_increment=hits)


def compute_psd_db(samples: np.ndarray, samp_rate: float, fft_size: int, avg: int) -> Tuple[np.ndarray, np.ndarray]:
    if HAVE_SCIPY:
        # Welch PSD over 'avg' segments
        nperseg = fft_size
        noverlap = 0
        freqs, psd = welch(samples, fs=samp_rate, nperseg=nperseg, noverlap=noverlap, return_onesided=False, scaling="density")
        # Welch returns frequencies in ascending order; we want baseband centered
        # Convert to centered, consistent with fftshift convention
        order = np.argsort(freqs)
        freqs = freqs[order]
        psd = psd[order]
        # shift to baseband (-Fs/2..+Fs/2)
        mid = len(freqs) // 2
        freqs = np.concatenate((freqs[mid:], freqs[:mid]))
        psd = np.concatenate((psd[mid:], psd[:mid]))
    else:
        # Simple averaged periodogram
        seg = fft_size
        windows: List[np.ndarray] = []
        for i in range(avg):
            start = i * seg
            x = samples[start : start + seg]
            if len(x) < seg:
                break
            X = np.fft.fftshift(np.fft.fft(x * np.hanning(seg), n=seg))
            Pxx = (np.abs(X) ** 2) / (seg * samp_rate)
            windows.append(Pxx)
        if not windows:
            X = np.fft.fftshift(np.fft.fft(samples[:fft_size] * np.hanning(fft_size), n=fft_size))
            Pxx = (np.abs(X) ** 2) / (fft_size * samp_rate)
            windows = [Pxx]
        psd = np.mean(np.vstack(windows), axis=0)
        freqs = np.linspace(-samp_rate / 2, samp_rate / 2, len(psd), endpoint=False)

    psd_db = db10(psd)
    return freqs, psd_db


# ------------------------------
# Output helpers
# ------------------------------

def maybe_notify(title: str, body: str, enabled: bool):
    if not enabled:
        return
    try:
        import subprocess

        subprocess.Popen(["notify-send", title, body])
    except Exception:
        pass


def maybe_emit_jsonl(path: Optional[str], record: dict):
    if not path:
        return
    try:
        import json
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


# ------------------------------
# Main sweep logic
# ------------------------------

def _parse_duration_to_seconds(text: Optional[str]) -> Optional[float]:
    """Parse a human-friendly duration string to seconds.
    Supports integers/floats (seconds) or suffixes: s, m, h, d.
    Returns None if text is falsy.
    """
    if not text:
        return None
    try:
        t = str(text).strip().lower()
        mult = 1.0
        if t.endswith("s"):
            t = t[:-1]
        elif t.endswith("m"):
            mult = 60.0
            t = t[:-1]
        elif t.endswith("h"):
            mult = 3600.0
            t = t[:-1]
        elif t.endswith("d"):
            mult = 86400.0
            t = t[:-1]
        return float(t) * mult
    except Exception:
        raise ValueError(f"Invalid duration string: {text}")


def _set_default(args, overrides: set, name: str, value):
    if hasattr(args, name):
        overrides.add(name)
    else:
        setattr(args, name, value)


def _apply_scan_profile(args, parser: argparse.ArgumentParser):
    profile_name = getattr(args, "profile", None)
    if not profile_name:
        return
    profiles = default_scan_profiles()
    prof = profiles.get(profile_name.lower())
    if prof is None:
        parser.error(f"Unknown scan profile '{profile_name}'. Available: {', '.join(sorted(profiles.keys()))}")

    requested_low = min(args.start, args.stop)
    requested_high = max(args.start, args.stop)
    if requested_low < prof.f_low_hz or requested_high > prof.f_high_hz:
        print(
            f"[profile] Requested span {requested_low/1e6:.3f}-{requested_high/1e6:.3f} MHz outside profile '{prof.name}' band, skipping profile defaults.",
            file=sys.stderr,
        )
        return

    def maybe_set(attr: str, value):
        if attr not in args._cli_overrides:
            setattr(args, attr, value)

    maybe_set("samp_rate", prof.samp_rate)
    maybe_set("fft", prof.fft)
    maybe_set("avg", prof.avg)
    maybe_set("threshold_db", prof.threshold_db)
    maybe_set("guard_bins", prof.guard_bins)
    maybe_set("min_width_bins", prof.min_width_bins)

    if prof.abs_power_floor_db is not None:
        setattr(args, "abs_power_floor_db", prof.abs_power_floor_db)

    gain_override = "gain" in args._cli_overrides and not (
        isinstance(getattr(args, "gain"), str) and getattr(args, "gain").lower() == "auto"
    )
    if not gain_override:
        if isinstance(getattr(args, "gain"), str) and getattr(args, "gain").lower() == "auto":
            print(
                f"[profile] Overriding auto gain with fixed {prof.gain_db:.1f} dB from profile '{prof.name}'.",
                file=sys.stderr,
            )
        setattr(args, "gain", float(prof.gain_db))

    print(f"[profile] Applied profile '{prof.name}'", flush=True)


def _do_one_sweep(args, store: Store, bandplan: Bandplan, src) -> int:
    """Perform a single full sweep across [start, stop] inclusive, returning the scan_id."""
    meta = dict(
        t_start_utc=utc_now_str(),
        f_start_hz=int(args.start),
        f_stop_hz=int(args.stop),
        step_hz=int(args.step),
        samp_rate=int(args.samp_rate),
        fft=int(args.fft),
        avg=int(args.avg),
        device=str(getattr(src, 'dev', getattr(src, 'device', '')) if HAVE_SOAPY else getattr(src, 'device', '')),
        driver=args.driver,
        latitude=args.latitude,
        longitude=args.longitude,
    )
    scan_id = store.start_scan(meta)

    bin_hz = float(args.samp_rate) / float(args.fft) if args.fft else float(args.samp_rate)
    detection_engine: Optional[DetectionEngine]
    if args.spur_calibration:
        detection_engine = None
    else:
        detection_engine = DetectionEngine(
            store,
            bandplan,
            args,
            bin_hz=bin_hz,
        )
    power_monitor = WindowPowerMonitor()
    spur_tracker: Dict[int, Dict[str, float]] = {}

    try:
        center = args.start
        window_idx = 0
        print(f"[scan] begin sweep id={scan_id} range={args.start/1e6:.3f}-{args.stop/1e6:.3f} MHz step={args.step/1e6:.3f} samp_rate={args.samp_rate/1e6:.3f} fft={args.fft} avg={args.avg}", flush=True)
        while center <= args.stop:
            src.tune(center)
            nsamps = int(args.fft * args.avg)
            # Discard a small warmup buffer to allow tuner/AGC to settle
            _ = src.read(int(args.fft))
            samples = src.read(nsamps)
            baseband_f, psd_db = compute_psd_db(samples, args.samp_rate, args.fft, args.avg)
            # Translate baseband freqs to RF
            rf_freqs = baseband_f + center

            # Detect segments
            segs, occ_mask_cfar, _noise_local_db = detect_segments(
                rf_freqs,
                psd_db,
                thresh_db=args.threshold_db,
                guard_bins=args.guard_bins,
                min_width_bins=args.min_width_bins,
                cfar_mode=args.cfar,
                cfar_train=args.cfar_train,
                cfar_guard=args.cfar_guard,
                cfar_quantile=args.cfar_quantile,
                cfar_alpha_db=args.cfar_alpha_db,
                abs_power_floor_db=getattr(args, "abs_power_floor_db", None),
            )

            mean_psd_db = float(np.mean(psd_db))
            p90_psd_db = float(np.percentile(psd_db, 90.0))
            is_anom, _ema_power_db, _delta_db = power_monitor.update(mean_psd_db)
            accepted_hits = 0
            spur_ignored = 0
            promoted = 0

            if is_anom:
                if detection_engine:
                    accepted_hits, spur_ignored, promoted = detection_engine.ingest(scan_id, window_idx, [])
            else:
                # Occupancy mask per bin for baseline update
                noise_db = robust_noise_floor_db(psd_db)
                dynamic = noise_db + args.threshold_db
                occupied_mask = occ_mask_cfar if (args.cfar and args.cfar != 'off') else (psd_db > dynamic)

                # --- begin per-window batched DB writes ---
                store.begin()

                store.update_baseline(rf_freqs, psd_db, occupied_mask)

                # commit batched writes for this window
                store.commit()

                if args.spur_calibration:
                    _track_spur_hits(spur_tracker, segs)
                if detection_engine:
                    accepted_hits, spur_ignored, promoted = detection_engine.ingest(scan_id, window_idx, segs)
            window_idx += 1

            # Progress log every window
            fields = [
                f"center_hz={center:.1f}",
                f"det_count={len(segs)}",
                f"mean_db={mean_psd_db:.1f}",
                f"p90_db={p90_psd_db:.1f}",
                f"anomalous={1 if is_anom else 0}",
            ]
            if detection_engine:
                fields.extend(
                    [
                        f"accepted={accepted_hits}",
                        f"promoted={promoted}",
                        f"spur_masked={spur_ignored}",
                    ]
                )
            print(f"[scan] window {' '.join(fields)}", flush=True)
            # Advance center frequency
            center += args.step

    finally:
        if detection_engine:
            flushed = detection_engine.flush(scan_id)
            if flushed:
                print(f"[scan] sweep id={scan_id} flushed pending detections={flushed}", flush=True)
        if args.spur_calibration:
            _persist_spur_calibration(store, spur_tracker, window_idx)
        # End scan (always set end time)
        store.end_scan(scan_id, utc_now_str())
        print(f"[scan] end sweep id={scan_id}", flush=True)

    return scan_id


def run(args):
    """Scanning pipeline overview.

    run() decides how many sweeps execute by inspecting --loop/--repeat/--duration,
    then repeatedly calls _do_one_sweep() until that termination policy is satisfied.
    _do_one_sweep() walks center frequency from --start to --stop in --step increments,
    tuning the SDR, collecting samples, and invoking compute_psd_db() to turn each
    capture into a baseband PSD. detect_segments() (with CFAR via cfar_os_mask) turns
    each PSD into contiguous hits, while Store.update_baseline() ingests every bin's
    PSD/noise occupancy EMA. DetectionEngine accumulates those hits over multiple
    windows, consults spur_map to suppress known hardware artifacts, and only invokes
    Store.add_detection() (plus JSONL/notifies) once a frequency region shows
    repeat/persistent energy. In --spur-calibration mode the same sweep populates
    spur_map instead of emitting detections.
    """
    if getattr(args, "list_profiles", False):
        _emit_profiles_json()
        return

    # Optional TMPDIR override (steer off tmpfs /tmp on Trixie)
    if args.tmpdir:
        os.environ["TMPDIR"] = args.tmpdir

    bandplan = Bandplan(args.bandplan)
    store = Store(args.db)

    # Parse --soapy-args into a dict if present
    soapy_args_dict: Optional[Dict[str, str]] = None
    if getattr(args, "soapy_args", None):
        soapy_args_dict = {}
        for kv in str(args.soapy_args).split(","):
            if "=" in kv:
                k, v = kv.split("=", 1)
                soapy_args_dict[k.strip()] = v.strip()

    # Select source backend
    if args.driver == "rtlsdr_native":
        src = RTLSDRSource(samp_rate=args.samp_rate, gain=args.gain)
        hwkey = "RTL-SDR (native)"
        setattr(src, 'device', hwkey)
    else:
        # Prefer Soapy; if device creation fails with 'no match', fallback to native RTL for rtlsdr
        try:
            src = SDRSource(driver=args.driver, samp_rate=args.samp_rate, gain=args.gain, soapy_args=soapy_args_dict)
        except Exception as e:
            msg = str(e)
            if args.driver == "rtlsdr" and ("no match" in msg.lower() or "Device::make" in msg or "rtlsdr" in msg.lower()):
                # Fallback to native librtlsdr path; target the same device by serial/index if known
                idx_hint = None
                serial_hint = None
                if soapy_args_dict:
                    if 'serial' in soapy_args_dict:
                        serial_hint = soapy_args_dict.get('serial')
                    if 'index' in soapy_args_dict:
                        try:
                            val = soapy_args_dict.get('index')
                            if val is not None:
                                idx_hint = int(val)
                        except Exception:
                            idx_hint = None
                # Retry loop in case the interface is momentarily busy
                last_err = None
                for attempt in range(3):
                    try:
                        src = RTLSDRSource(samp_rate=args.samp_rate, gain=args.gain, device_index=idx_hint, serial_number=serial_hint)
                        break
                    except Exception as e2:
                        last_err = e2
                        time.sleep(0.2)
                if 'src' not in locals():
                    raise last_err if last_err else e
                hwkey = "RTL-SDR (native fallback)"
                setattr(src, 'device', hwkey)
                args.driver = "rtlsdr_native"
            else:
                raise

    # Determine termination policy
    duration_s = _parse_duration_to_seconds(args.duration)
    start_time = time.time()

    # Compute how many sweeps to run: None for infinite
    # Rules:
    # - If --loop, infinite.
    # - If --repeat N, exactly N sweeps.
    # - If --duration is provided WITHOUT --loop/--repeat, run until time expires (infinite sweeps governed by time).
    # - Otherwise (no flags), run exactly one sweep.
    if args.loop:
        sweeps_remaining: Optional[int] = None
    elif args.repeat is not None:
        sweeps_remaining = int(args.repeat)
    elif duration_s is not None:
        sweeps_remaining = None  # duration governs
    else:
        sweeps_remaining = 1  # default single sweep

    try:
        while True:
            # Duration check (before starting next sweep)
            if duration_s is not None and (time.time() - start_time) >= duration_s:
                break

            _do_one_sweep(args, store, bandplan, src)

            # After each sweep, respect duration again
            if duration_s is not None and (time.time() - start_time) >= duration_s:
                break

            if sweeps_remaining is not None:
                sweeps_remaining -= 1
                if sweeps_remaining <= 0:
                    break

            # Sleep between sweeps if requested
            if args.sleep_between_sweeps > 0:
                time.sleep(args.sleep_between_sweeps)

    except KeyboardInterrupt:
        # Graceful exit on Ctrl-C
        pass
    finally:
        try:
            src.close()
        except Exception:
            pass


# ------------------------------
# CLI
# ------------------------------

def parse_args(argv: Optional[List[str]] = None):
    if argv is None:
        argv = sys.argv[1:]

    p = argparse.ArgumentParser(
        description="Wideband scanner & baseline builder using SoapySDR or native RTL-SDR",
        argument_default=argparse.SUPPRESS,
    )
    p.add_argument("--start", type=float, help="Start frequency in Hz (e.g., 88e6)")
    p.add_argument("--stop", type=float, help="Stop frequency in Hz (e.g., 108e6)")
    p.add_argument("--step", type=float, help="Center frequency step per window [Hz] (default 2.4e6)")

    p.add_argument("--samp-rate", dest="samp_rate", type=float, help="Sample rate [Hz] (default 2.4e6)")
    p.add_argument("--fft", type=int, help="FFT size (per Welch segment) (default 4096)")
    p.add_argument("--avg", type=int, help="Averaging factor (segments per PSD) (default 8)")

    p.add_argument("--driver", type=str, help="Soapy driver key (e.g., rtlsdr, hackrf, airspy, etc.) or 'rtlsdr_native' for direct librtlsdr (default rtlsdr)")
    p.add_argument("--soapy-args", type=str, help="Comma-separated Soapy device args (e.g., 'serial=00000001,index=0')")
    p.add_argument("--gain", type=str, help='Gain in dB or "auto" (default auto)')

    p.add_argument("--threshold-db", dest="threshold_db", type=float, help="Detection threshold above noise floor [dB] (default 8.0)")
    p.add_argument("--guard-bins", dest="guard_bins", type=int, help="Allow this many below-threshold bins inside a detection (default 1)")
    p.add_argument("--min-width-bins", dest="min_width_bins", type=int, help="Minimum contiguous bins for a detection (default 2)")
    # CFAR options
    p.add_argument("--cfar", choices=["off", "os", "ca"], help="CFAR mode (default: os)")
    p.add_argument("--cfar-train", dest="cfar_train", type=int, help="Training cells per side for CFAR (default 24)")
    p.add_argument("--cfar-guard", dest="cfar_guard", type=int, help="Guard cells per side (excluded around CUT) for CFAR (default 4)")
    p.add_argument("--cfar-quantile", dest="cfar_quantile", type=float, help="Quantile (0..1) for OS-CFAR order statistic (default 0.75)")
    p.add_argument("--cfar-alpha-db", dest="cfar_alpha_db", type=float, help="Override threshold scaling for CFAR in dB; defaults to --threshold-db")

    p.add_argument("--bandplan", type=str, help="Optional bandplan CSV to map detections")
    p.add_argument("--db", type=str, help="SQLite DB path (default sdrwatch.db)")
    p.add_argument("--jsonl", type=str, help="Emit detections as line-delimited JSON to this path")
    p.add_argument("--notify", action="store_true", help="Desktop notifications for new signals")
    p.add_argument("--new-ema-occ", dest="new_ema_occ", type=float, help="EMA occupancy threshold to flag a bin as NEW (default 0.02)")
    p.add_argument("--latitude", type=float, help="Optional latitude in decimal degrees for this scan")
    p.add_argument("--longitude", type=float, help="Optional longitude in decimal degrees for this scan")
    p.add_argument("--profile", type=str, help="Scan profile name to pre-load sane defaults (see documentation)")
    p.add_argument("--spur-calibration", dest="spur_calibration", action="store_true", help="Learn persistent internal spurs instead of emitting detections")
    p.add_argument("--list-profiles", dest="list_profiles", action="store_true", help="Print built-in scan profiles as JSON and exit")

    # Sweep control modes (mutually exclusive)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--loop", action="store_true", help="Run continuous sweep cycles until cancelled")
    group.add_argument("--repeat", type=int, help="Run exactly N full sweep cycles, then exit")
    group.add_argument("--duration", type=str, help="Run sweeps for a duration (e.g., '300', '10m', '2h'). Overrides --repeat count while time remains")

    p.add_argument("--sleep-between-sweeps", dest="sleep_between_sweeps", type=float, help="Seconds to sleep between sweep cycles (default 0)")

    # Trixie scratch steering
    p.add_argument("--tmpdir", type=str, help="Scratch directory for temp files (defaults to $TMPDIR)")

    args = p.parse_args(argv)
    args._cli_overrides = set()

    _set_default(args, args._cli_overrides, "step", 2.4e6)
    _set_default(args, args._cli_overrides, "samp_rate", 2.4e6)
    _set_default(args, args._cli_overrides, "fft", 4096)
    _set_default(args, args._cli_overrides, "avg", 8)
    _set_default(args, args._cli_overrides, "driver", "rtlsdr")
    _set_default(args, args._cli_overrides, "soapy_args", None)
    _set_default(args, args._cli_overrides, "gain", "auto")
    _set_default(args, args._cli_overrides, "threshold_db", 8.0)
    _set_default(args, args._cli_overrides, "guard_bins", 1)
    _set_default(args, args._cli_overrides, "min_width_bins", 2)
    _set_default(args, args._cli_overrides, "cfar", "os")
    _set_default(args, args._cli_overrides, "cfar_train", 24)
    _set_default(args, args._cli_overrides, "cfar_guard", 4)
    _set_default(args, args._cli_overrides, "cfar_quantile", 0.75)
    _set_default(args, args._cli_overrides, "cfar_alpha_db", None)
    _set_default(args, args._cli_overrides, "bandplan", None)
    _set_default(args, args._cli_overrides, "db", "sdrwatch.db")
    _set_default(args, args._cli_overrides, "jsonl", None)
    _set_default(args, args._cli_overrides, "notify", False)
    _set_default(args, args._cli_overrides, "new_ema_occ", 0.02)
    _set_default(args, args._cli_overrides, "latitude", None)
    _set_default(args, args._cli_overrides, "longitude", None)
    _set_default(args, args._cli_overrides, "profile", None)
    _set_default(args, args._cli_overrides, "spur_calibration", False)
    _set_default(args, args._cli_overrides, "list_profiles", False)
    _set_default(args, args._cli_overrides, "loop", False)
    _set_default(args, args._cli_overrides, "repeat", None)
    _set_default(args, args._cli_overrides, "duration", None)
    _set_default(args, args._cli_overrides, "sleep_between_sweeps", 0.0)
    _set_default(args, args._cli_overrides, "tmpdir", os.environ.get("TMPDIR"))
    setattr(args, "abs_power_floor_db", None)

    has_span = hasattr(args, "start") and hasattr(args, "stop")
    if not args.list_profiles and not has_span:
        p.error("--start and --stop are required unless --list-profiles is used")

    if has_span:
        _apply_scan_profile(args, p)

    if hasattr(args, "_cli_overrides"):
        delattr(args, "_cli_overrides")

    # Backend availability check: only require hardware dependencies when scanning
    if not args.list_profiles:
        if args.driver != "rtlsdr_native" and not HAVE_SOAPY:
            p.error("python3-soapysdr not installed. Install it (or use --driver rtlsdr_native).")
        if args.driver == "rtlsdr_native" and not HAVE_RTLSDR:
            p.error("pyrtlsdr not installed. Install with: pip3 install pyrtlsdr")
        if args.stop < args.start:
            p.error("--stop must be >= --start")

    # Validate duration string early
    if args.duration:
        _ = _parse_duration_to_seconds(args.duration)

    return args


if __name__ == "__main__":
    run(parse_args())
