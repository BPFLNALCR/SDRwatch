#!/usr/bin/env python3
"""
SDRWatch ΓÇö wideband scanner, baseline builder, and bandplan mapper

Goals
-----
- Sweep a frequency range with an SDR (SoapySDR backend by default; optional native RTL-SDR backend).
- Estimate noise floor robustly and detect signals via energy thresholding (CFARΓÇælike).
- Build a baseline (perΓÇæbin occupancy over time) and flag "new" signals relative to that baseline.
- Map detections to a bandplan (FCC/CEPT/etc.) from a CSV file or builtΓÇæin minimal defaults.
- Log everything to SQLite and optionally emit desktop notifications or webhook JSON lines.

Hardware
--------
Any SoapySDRΓÇæsupported device (RTLΓÇæSDR, HackRF, Airspy, SDRplay, LimeSDR, USRP...).
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
- baselines(id INTEGER PK AUTOINCREMENT, name TEXT, created_at TEXT, location_lat REAL, location_lon REAL,
    sdr_serial TEXT, antenna TEXT, notes TEXT, freq_start_hz INTEGER, freq_stop_hz INTEGER,
    bin_hz REAL, baseline_version INTEGER, total_windows INTEGER DEFAULT 0)
- baseline_stats(baseline_id INTEGER, bin_index INTEGER, noise_floor_ema REAL, power_ema REAL,
     occ_count INTEGER, last_seen_utc TEXT, PRIMARY KEY (baseline_id, bin_index))
- baseline_detections(id INTEGER PK AUTOINCREMENT, baseline_id INTEGER, f_low_hz INTEGER, f_high_hz INTEGER,
          f_center_hz INTEGER, first_seen_utc TEXT, last_seen_utc TEXT,
          total_hits INTEGER, total_windows INTEGER, confidence REAL)
- scan_updates(id INTEGER PK AUTOINCREMENT, baseline_id INTEGER, timestamp_utc TEXT,
       num_hits INTEGER, num_segments INTEGER, num_new_signals INTEGER)
- spur_map(bin_hz INTEGER PRIMARY KEY, mean_power_db REAL, hits INTEGER, last_seen_utc TEXT)

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
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

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
    bandwidth_hz: float = 0.0


@dataclass
class BaselineContext:
    id: int
    name: str
    freq_start_hz: int
    freq_stop_hz: int
    bin_hz: float
    baseline_version: int
    total_windows: int


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
    step_hz: Optional[float] = None
    cfar_train: Optional[int] = None
    cfar_guard: Optional[int] = None
    cfar_quantile: Optional[float] = None
    persistence_hit_ratio: Optional[float] = None
    persistence_min_seconds: Optional[float] = None
    persistence_min_hits: Optional[int] = None
    persistence_min_windows: Optional[int] = None
    revisit_fft: Optional[int] = None
    revisit_avg: Optional[int] = None
    revisit_margin_hz: Optional[float] = None
    revisit_max_bands: Optional[int] = None
    revisit_floor_threshold_db: Optional[float] = None
    two_pass: Optional[bool] = None
    bandwidth_pad_hz: Optional[float] = None
    min_emit_bandwidth_hz: Optional[float] = None
    confidence_hit_normalizer: Optional[float] = None
    confidence_duration_norm: Optional[float] = None
    confidence_bias: Optional[float] = None


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
            samp_rate=2.4e6,
            fft=8192,
            avg=10,
            gain_db=20.0,
            threshold_db=6.0,
            min_width_bins=12,
            guard_bins=3,
            abs_power_floor_db=-92.0,
            step_hz=1.2e6,
            cfar_train=32,
            cfar_guard=6,
            cfar_quantile=0.6,
            persistence_hit_ratio=0.25,
            persistence_min_seconds=2.0,
            persistence_min_hits=1,
            persistence_min_windows=1,
            revisit_fft=32768,
            revisit_avg=4,
            revisit_margin_hz=200_000.0,
            revisit_max_bands=40,
            revisit_floor_threshold_db=6.0,
            two_pass=True,
            bandwidth_pad_hz=60_000.0,
            min_emit_bandwidth_hz=180_000.0,
            confidence_hit_normalizer=2.0,
            confidence_duration_norm=2.0,
            confidence_bias=0.05,
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
                "step_hz": prof.step_hz,
                "cfar_train": prof.cfar_train,
                "cfar_guard": prof.cfar_guard,
                "cfar_quantile": prof.cfar_quantile,
                "persistence_hit_ratio": prof.persistence_hit_ratio,
                "persistence_min_seconds": prof.persistence_min_seconds,
                "persistence_min_hits": prof.persistence_min_hits,
                "persistence_min_windows": prof.persistence_min_windows,
                "revisit_fft": prof.revisit_fft,
                "revisit_avg": prof.revisit_avg,
                "revisit_margin_hz": prof.revisit_margin_hz,
                "revisit_max_bands": prof.revisit_max_bands,
                "revisit_floor_threshold_db": prof.revisit_floor_threshold_db,
                "two_pass": prof.two_pass,
                "bandwidth_pad_hz": prof.bandwidth_pad_hz,
                "min_emit_bandwidth_hz": prof.min_emit_bandwidth_hz,
                "confidence_hit_normalizer": prof.confidence_hit_normalizer,
                "confidence_duration_norm": prof.confidence_duration_norm,
                "confidence_bias": prof.confidence_bias,
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

    def _init(self) -> None:
        cur = self.con.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS baselines (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                location_lat REAL,
                location_lon REAL,
                sdr_serial TEXT,
                antenna TEXT,
                notes TEXT,
                freq_start_hz INTEGER NOT NULL,
                freq_stop_hz INTEGER NOT NULL,
                bin_hz REAL NOT NULL,
                baseline_version INTEGER NOT NULL DEFAULT 1,
                total_windows INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS baseline_stats (
                baseline_id INTEGER NOT NULL,
                bin_index INTEGER NOT NULL,
                noise_floor_ema REAL NOT NULL,
                power_ema REAL NOT NULL,
                occ_count INTEGER NOT NULL,
                last_seen_utc TEXT NOT NULL,
                PRIMARY KEY (baseline_id, bin_index)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS baseline_detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                baseline_id INTEGER NOT NULL,
                f_low_hz INTEGER NOT NULL,
                f_high_hz INTEGER NOT NULL,
                f_center_hz INTEGER NOT NULL,
                first_seen_utc TEXT NOT NULL,
                last_seen_utc TEXT NOT NULL,
                total_hits INTEGER NOT NULL,
                total_windows INTEGER NOT NULL,
                confidence REAL NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS scan_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                baseline_id INTEGER NOT NULL,
                timestamp_utc TEXT NOT NULL,
                num_hits INTEGER NOT NULL,
                num_segments INTEGER NOT NULL,
                num_new_signals INTEGER NOT NULL,
                num_revisits INTEGER NOT NULL DEFAULT 0,
                num_confirmed INTEGER NOT NULL DEFAULT 0,
                num_false_positive INTEGER NOT NULL DEFAULT 0
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

        # Schema migrations (idempotent)
        self._ensure_column(
            "baseline_detections",
            "missing_since_utc",
            "TEXT",
        )
        self._ensure_column("scan_updates", "num_revisits", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("scan_updates", "num_confirmed", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("scan_updates", "num_false_positive", "INTEGER NOT NULL DEFAULT 0")

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        cur = self.con.execute(f"PRAGMA table_info({table})")
        cols = {row[1] for row in cur.fetchall()}
        if column in cols:
            return
        try:
            self.con.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            self.con.commit()
        except sqlite3.OperationalError:
            # Column may already exist under races; ignore.
            pass

    def begin(self) -> None:
        if not self.con.in_transaction:
            self.con.execute("BEGIN")

    def commit(self) -> None:
        if self.con.in_transaction:
            self.con.commit()

    def get_latest_baseline_id(self) -> Optional[int]:
        cur = self.con.execute("SELECT id FROM baselines ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        return int(row[0]) if row else None

    def get_baseline(self, baseline_id: int) -> Optional[BaselineContext]:
        cur = self.con.execute(
            """
            SELECT id, name, freq_start_hz, freq_stop_hz, bin_hz, baseline_version, total_windows
            FROM baselines
            WHERE id = ?
            """,
            (int(baseline_id),),
        )
        row = cur.fetchone()
        if not row:
            return None
        return BaselineContext(
            id=int(row[0]),
            name=str(row[1]),
            freq_start_hz=int(row[2] or 0),
            freq_stop_hz=int(row[3] or 0),
            bin_hz=float(row[4]),
            baseline_version=int(row[5]),
            total_windows=int(row[6] or 0),
        )

    def increment_baseline_windows(self, baseline_id: int, delta: int = 1) -> int:
        cur = self.con.cursor()
        cur.execute(
            "UPDATE baselines SET total_windows = total_windows + ? WHERE id = ?",
            (int(delta), int(baseline_id)),
        )
        cur.execute("SELECT total_windows FROM baselines WHERE id = ?", (int(baseline_id),))
        row = cur.fetchone()
        return int(row[0] or 0) if row else 0

    def update_baseline_span(self, baseline_id: int, sweep_start_hz: float, sweep_stop_hz: float) -> Tuple[int, int]:
        start = int(min(sweep_start_hz, sweep_stop_hz))
        stop = int(max(sweep_start_hz, sweep_stop_hz))
        cur = self.con.cursor()
        cur.execute(
            "SELECT freq_start_hz, freq_stop_hz, total_windows FROM baselines WHERE id = ?",
            (int(baseline_id),),
        )
        row = cur.fetchone()
        if not row:
            raise RuntimeError(f"Baseline id {baseline_id} not found for span update")
        current_start = int(row[0] or 0)
        current_stop = int(row[1] or 0)
        current_windows = int(row[2] or 0)
        if stop <= 0:
            return current_start, current_stop
        new_start = current_start
        if current_start == 0 or current_windows == 0:
            new_start = start
        new_stop = stop if current_stop == 0 else max(current_stop, stop)
        if new_start <= 0 and new_stop <= 0:
            return current_start, current_stop
        if new_start != current_start or new_stop != current_stop:
            cur.execute(
                "UPDATE baselines SET freq_start_hz = ?, freq_stop_hz = ? WHERE id = ?",
                (int(new_start), int(new_stop), int(baseline_id)),
            )
            self.con.commit()
            return new_start, new_stop
        return current_start, current_stop

    def update_baseline_stats(
        self,
        baseline_id: int,
        bin_indices: Iterable[int],
        noise_floor_db: Iterable[float],
        power_db: Iterable[float],
        occupied_mask: Iterable[bool],
        timestamp_utc: str,
        ema_alpha: float = 0.05,
    ) -> None:
        cur = self.con.cursor()
        for idx, noise, power, occupied in zip(bin_indices, noise_floor_db, power_db, occupied_mask):
            if idx is None:
                continue
            noise_val = float(noise)
            power_val = float(power)
            if not math.isfinite(noise_val):
                noise_val = power_val
            if not math.isfinite(power_val):
                continue
            cur.execute(
                """
                SELECT noise_floor_ema, power_ema, occ_count
                FROM baseline_stats
                WHERE baseline_id = ? AND bin_index = ?
                """,
                (int(baseline_id), int(idx)),
            )
            row = cur.fetchone()
            if row is None:
                occ_count = 1 if occupied else 0
                cur.execute(
                    """
                    INSERT INTO baseline_stats(baseline_id, bin_index, noise_floor_ema, power_ema, occ_count, last_seen_utc)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (int(baseline_id), int(idx), float(noise_val), float(power_val), int(occ_count), timestamp_utc),
                )
            else:
                prev_noise, prev_power, prev_occ = row
                prev_noise = float(prev_noise) if prev_noise is not None else noise_val
                prev_power = float(prev_power) if prev_power is not None else power_val
                prev_occ = int(prev_occ or 0)
                noise_ema = (1.0 - ema_alpha) * prev_noise + ema_alpha * noise_val
                power_ema = (1.0 - ema_alpha) * prev_power + ema_alpha * power_val
                occ_count = prev_occ + (1 if occupied else 0)
                cur.execute(
                    """
                    UPDATE baseline_stats
                    SET noise_floor_ema = ?, power_ema = ?, occ_count = ?, last_seen_utc = ?
                    WHERE baseline_id = ? AND bin_index = ?
                    """,
                    (float(noise_ema), float(power_ema), int(occ_count), timestamp_utc, int(baseline_id), int(idx)),
                )

    def load_baseline_detections(self, baseline_id: int) -> List["PersistentDetection"]:
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT id, baseline_id, f_low_hz, f_high_hz, f_center_hz,
                 first_seen_utc, last_seen_utc, total_hits, total_windows, confidence,
                 missing_since_utc
            FROM baseline_detections
            WHERE baseline_id = ?
            ORDER BY f_center_hz
            """,
            (int(baseline_id),),
        )
        rows = cur.fetchall()
        detections: List[PersistentDetection] = []
        for row in rows:
            detections.append(
                PersistentDetection(
                    id=int(row[0]),
                    baseline_id=int(row[1]),
                    f_low_hz=int(row[2]),
                    f_high_hz=int(row[3]),
                    f_center_hz=int(row[4]),
                    first_seen_utc=str(row[5]),
                    last_seen_utc=str(row[6]),
                    total_hits=int(row[7]),
                    total_windows=int(row[8]),
                    confidence=float(row[9]),
                    missing_since_utc=str(row[10]) if row[10] else None,
                )
            )
        return detections

    def insert_baseline_detection(
        self,
        baseline_id: int,
        f_low_hz: int,
        f_high_hz: int,
        f_center_hz: int,
        first_seen_utc: str,
        last_seen_utc: str,
        total_hits: int,
        total_windows: int,
        confidence: float,
        missing_since_utc: Optional[str] = None,
    ) -> int:
        cur = self.con.cursor()
        cur.execute(
            """
            INSERT INTO baseline_detections(
                baseline_id, f_low_hz, f_high_hz, f_center_hz,
                first_seen_utc, last_seen_utc, total_hits, total_windows, confidence, missing_since_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(baseline_id),
                int(f_low_hz),
                int(f_high_hz),
                int(f_center_hz),
                first_seen_utc,
                last_seen_utc,
                int(total_hits),
                int(total_windows),
                float(confidence),
                missing_since_utc,
            ),
        )
        if cur.lastrowid is None:
            raise RuntimeError("Failed to insert baseline detection")
        return int(cur.lastrowid)

    def update_baseline_detection(self, detection: "PersistentDetection") -> None:
        self.con.execute(
            """
            UPDATE baseline_detections
            SET f_low_hz = ?, f_high_hz = ?, f_center_hz = ?,
                first_seen_utc = ?, last_seen_utc = ?,
                total_hits = ?, total_windows = ?, confidence = ?,
                missing_since_utc = ?
            WHERE id = ? AND baseline_id = ?
            """,
            (
                int(detection.f_low_hz),
                int(detection.f_high_hz),
                int(detection.f_center_hz),
                detection.first_seen_utc,
                detection.last_seen_utc,
                int(detection.total_hits),
                int(detection.total_windows),
                float(detection.confidence),
                detection.missing_since_utc,
                int(detection.id),
                int(detection.baseline_id),
            ),
        )

    def insert_scan_update(
        self,
        baseline_id: int,
        timestamp_utc: str,
        num_hits: int,
        num_segments: int,
        num_new_signals: int,
        *,
        num_revisits: int = 0,
        num_confirmed: int = 0,
        num_false_positive: int = 0,
    ) -> None:
        self.con.execute(
            """
            INSERT INTO scan_updates(
                baseline_id, timestamp_utc,
                num_hits, num_segments, num_new_signals,
                num_revisits, num_confirmed, num_false_positive
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                int(baseline_id),
                timestamp_utc,
                int(num_hits),
                int(num_segments),
                int(num_new_signals),
                int(num_revisits),
                int(num_confirmed),
                int(num_false_positive),
            ),
        )

    def mark_detection_missing(self, detection_id: int, baseline_id: int, missing_ts: str) -> None:
        self.con.execute(
            """
            UPDATE baseline_detections
            SET missing_since_utc = COALESCE(missing_since_utc, ?)
            WHERE id = ? AND baseline_id = ?
            """,
            (missing_ts, int(detection_id), int(baseline_id)),
        )

    def clear_detection_missing(self, detection_id: int, baseline_id: int) -> None:
        self.con.execute(
            """
            UPDATE baseline_detections
            SET missing_since_utc = NULL
            WHERE id = ? AND baseline_id = ?
            """,
            (int(detection_id), int(baseline_id)),
        )

    def delete_baseline_detection(self, detection_id: int, baseline_id: int) -> None:
        self.con.execute(
            "DELETE FROM baseline_detections WHERE id = ? AND baseline_id = ?",
            (int(detection_id), int(baseline_id)),
        )

    def baseline_occ_ratio(self, baseline_id: int, bin_index: int) -> Optional[float]:
        cur = self.con.cursor()
        cur.execute(
            "SELECT occ_count FROM baseline_stats WHERE baseline_id = ? AND bin_index = ?",
            (int(baseline_id), int(bin_index)),
        )
        row = cur.fetchone()
        if not row:
            return None
        occ_count = int(row[0] or 0)
        cur.execute("SELECT total_windows FROM baselines WHERE id = ?", (int(baseline_id),))
        total_row = cur.fetchone()
        total_windows = int(total_row[0] or 0) if total_row else 0
        if total_windows <= 0:
            return None
        return float(occ_count) / float(total_windows)

    def update_spur_bin(self, bin_hz: int, power_db: float, hits_increment: int = 1, ema_alpha: float = 0.2) -> None:
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

def _expand_bandwidth_indices(
    psd_db: np.ndarray,
    noise_db: np.ndarray,
    peak_idx: int,
    *,
    floor_margin_db: float,
    peak_drop_db: float,
    max_gap_bins: int,
) -> Tuple[int, int]:
    """Walk away from the peak until power falls near noise or gaps out."""
    N = psd_db.size
    if N == 0:
        return 0, 0
    threshold_peak = psd_db[peak_idx] - float(max(0.0, peak_drop_db))

    def walk(direction: int) -> int:
        idx = peak_idx
        best = peak_idx
        gap = 0
        while True:
            nxt = idx + direction
            if nxt < 0 or nxt >= N:
                break
            local_noise = float(noise_db[nxt]) if noise_db.shape == psd_db.shape else float(noise_db[peak_idx])
            floor_threshold = local_noise + float(floor_margin_db)
            threshold = max(floor_threshold, threshold_peak)
            if psd_db[nxt] >= threshold:
                best = nxt
                gap = 0
            else:
                gap += 1
                if gap > max_gap_bins:
                    break
            idx = nxt
        return best

    low_idx = walk(-1)
    high_idx = walk(1)
    return low_idx, high_idx


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
    *,
    bandwidth_floor_db: float = 2.0,
    bandwidth_peak_drop_db: float = 18.0,
    bandwidth_gap_hz: float = 15_000.0,
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

    if N > 1:
        diffs = np.diff(freqs_hz)
        bin_hz = float(np.median(diffs)) if diffs.size else 0.0
    else:
        bin_hz = 0.0
    gap_bins = max(1, int(round(bandwidth_gap_hz / max(bin_hz, 1.0)))) if bin_hz > 0 else max(1, min_width_bins)

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
                low_idx, high_idx = _expand_bandwidth_indices(
                    psd_db,
                    noise_for_snr_db,
                    peak_idx,
                    floor_margin_db=float(bandwidth_floor_db),
                    peak_drop_db=float(bandwidth_peak_drop_db),
                    max_gap_bins=gap_bins,
                )
                idx_low = min(low_idx, high_idx)
                idx_high = max(low_idx, high_idx)
                f_low = float(freqs_hz[idx_low])
                f_high = float(freqs_hz[idx_high])
                center_idx = (idx_low + idx_high) // 2
                f_center = float(freqs_hz[center_idx])
                bandwidth_hz = max(f_high - f_low + (bin_hz if bin_hz > 0 else 0.0), 0.0)
                segs.append(
                    Segment(
                        f_low_hz=int(round(f_low)),
                        f_high_hz=int(round(f_high)),
                        f_center_hz=int(round(f_center)),
                        peak_db=peak_db,
                        noise_db=noise_db,
                        snr_db=snr_db,
                        bandwidth_hz=float(bandwidth_hz),
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
    best_seg: Segment = field(default_factory=lambda: Segment(0, 0, 0, -999.0, -999.0, -999.0, 0.0))
    emitted: bool = False
    center_weight_sum: float = 0.0
    center_weight_total: float = 0.0


@dataclass
class PersistentDetection:
    id: int
    baseline_id: int
    f_low_hz: int
    f_high_hz: int
    f_center_hz: int
    first_seen_utc: str
    last_seen_utc: str
    total_hits: int
    total_windows: int
    confidence: float
    missing_since_utc: Optional[str] = None


@dataclass
class RevisitTag:
    tag_id: str
    detection_id: Optional[int]
    f_center_hz: int
    f_low_hz: int
    f_high_hz: int
    reason: str  # "new", "missing"
    created_utc: str


class DetectionEngine:
    def __init__(
        self,
        store: "Store",
        bandplan: Bandplan,
        args,
        *,
        bin_hz: float,
        baseline_ctx: BaselineContext,
        min_hits: int = 2,
        min_windows: int = 2,
        max_gap_windows: int = 3,
        freq_merge_hz: Optional[float] = None,
        logger: Optional["ScanLogger"] = None,
    ):
        self.store = store
        self.bandplan = bandplan
        self.args = args
        self.bin_hz = float(bin_hz)
        self.baseline_ctx = baseline_ctx
        self.min_hits = max(1, int(min_hits))
        self.min_windows = max(1, int(min_windows))
        self.max_gap_windows = max(1, int(max_gap_windows))
        self.freq_merge_hz = float(freq_merge_hz if freq_merge_hz is not None else max(self.bin_hz * 2, 25_000.0))
        self.center_match_hz = max(self.bin_hz * 2.0, self.freq_merge_hz / 2.0)
        raw_mode = str(getattr(args, "persistence_mode", "hits") or "hits").lower()
        self.persistence_mode = raw_mode if raw_mode in {"hits", "duration", "both"} else "hits"
        raw_ratio = getattr(args, "persistence_hit_ratio", 0.0)
        ratio_val = 0.0 if raw_ratio is None else float(raw_ratio)
        self.persistence_hit_ratio = float(np.clip(ratio_val, 0.0, 1.0))
        raw_duration = getattr(args, "persistence_min_seconds", 0.0)
        self.persistence_min_seconds = float(max(0.0, float(raw_duration if raw_duration is not None else 0.0)))
        self.min_width_hz = max(self.bin_hz * float(args.min_width_bins), self.bin_hz)
        self.clusters: List[DetectionCluster] = []
        self._last_window_idx = -1
        self.spur_tolerance_hz = 5_000.0
        self.spur_margin_db = 4.0
        self.spur_min_hits = 5
        self.spur_override_snr = 10.0
        self.spur_penalty_max = 0.35
        self._pending_emits = 0
        self._pending_new_signals = 0
        self._persisted: List[PersistentDetection] = self.store.load_baseline_detections(self.baseline_ctx.id)
        self._seen_persistent: Set[int] = set()
        self.two_pass_enabled = bool(getattr(args, "two_pass", False))
        self.revisit_margin_hz = float(
            getattr(args, "revisit_margin_hz", max(self.freq_merge_hz, 25_000.0)) or max(self.freq_merge_hz, 25_000.0)
        )
        self._revisit_tags: List[RevisitTag] = []
        self._tag_counter = 0
        self.logger = logger
        self.profile_name = getattr(args, "profile", None)
        self.bandwidth_pad_hz = max(0.0, float(getattr(args, "bandwidth_pad_hz", 0.0) or 0.0))
        self.min_emit_bandwidth_hz = max(0.0, float(getattr(args, "min_emit_bandwidth_hz", 0.0) or 0.0))
        raw_hit_norm = getattr(args, "confidence_hit_normalizer", None)
        raw_duration_norm = getattr(args, "confidence_duration_norm", None)
        raw_bias = getattr(args, "confidence_bias", None)
        self.conf_hit_normalizer = max(1.0, float(raw_hit_norm if raw_hit_norm not in (None, 0) else 6.0))
        self.conf_duration_norm = max(1.0, float(raw_duration_norm if raw_duration_norm not in (None, 0) else 8.0))
        self.confidence_bias = float(raw_bias if raw_bias is not None else 0.0)
        
    def _log(self, event: str, **fields: Any) -> None:
        if not self.logger:
            return
        payload = dict(fields)
        if self.profile_name:
            payload.setdefault("profile", self.profile_name)
        self.logger.log(event, **payload)

    def ingest(self, window_idx: int, segments: List[Segment]) -> Tuple[int, int, int, int]:
        self._last_window_idx = max(self._last_window_idx, window_idx)
        accepted = 0
        spur_ignored = 0
        if not segments:
            self._prune_clusters(window_idx)
            emitted, new_emitted = self._drain_pending_emits()
            return accepted, spur_ignored, emitted, new_emitted
        timestamp = utc_now_str()
        for seg in segments:
            if self._spur_should_ignore(seg):
                spur_ignored += 1
                continue
            self._record_hit(window_idx, seg, timestamp)
            accepted += 1
        self._prune_clusters(window_idx)
        emitted, new_emitted = self._drain_pending_emits()
        return accepted, spur_ignored, emitted, new_emitted

    def flush(self) -> Tuple[int, int]:
        self._prune_clusters(self._last_window_idx if self._last_window_idx >= 0 else 0, force=True)
        return self._drain_pending_emits()

    def _record_hit(self, window_idx: int, seg: Segment, timestamp: str):
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

        self._update_cluster_center(cluster, seg)
        self._maybe_emit_cluster(cluster)

    def _segment_weight(self, seg: Segment) -> float:
        try:
            return float(max(1e-3, 10.0 ** (seg.snr_db / 10.0)))
        except Exception:
            return 1.0

    def _update_cluster_center(self, cluster: DetectionCluster, seg: Segment) -> None:
        weight = self._segment_weight(seg)
        cluster.center_weight_sum += weight * float(seg.f_center_hz)
        cluster.center_weight_total += weight

    def _cluster_center_hz(self, cluster: DetectionCluster) -> int:
        if cluster.center_weight_total <= 0.0:
            return int((cluster.f_low_hz + cluster.f_high_hz) / 2)
        return int(round(cluster.center_weight_sum / cluster.center_weight_total))

    def _shape_emit_span(self, center_hz: int, raw_low: int, raw_high: int) -> Tuple[int, int]:
        width = max(float(raw_high - raw_low), self.bin_hz)
        if self.bandwidth_pad_hz > 0.0:
            width += self.bandwidth_pad_hz * 2.0
        min_emit = self.min_emit_bandwidth_hz
        if min_emit > 0.0 and width < min_emit:
            width = min_emit
        half = width / 2.0
        low = int(round(center_hz - half))
        high = int(round(center_hz + half))
        low = max(low, self.baseline_ctx.freq_start_hz)
        high = min(high, self.baseline_ctx.freq_stop_hz)
        if high <= low:
            high = low + int(max(1.0, self.bin_hz))
        return low, high

    def _blend_centers(self, center_a: int, weight_a: int, center_b: int, weight_b: int) -> int:
        wa = max(1, int(weight_a))
        wb = max(1, int(weight_b))
        return int(round((center_a * wa + center_b * wb) / float(wa + wb)))

    def _cluster_window_ratio(self, cluster: DetectionCluster) -> float:
        span_windows = max(cluster.last_window - cluster.first_window + 1, 1)
        return float(len(cluster.windows)) / float(span_windows)

    def _cluster_duration_seconds(self, cluster: DetectionCluster) -> float:
        try:
            t0 = self._parse_timestamp(cluster.first_seen_ts)
            t1 = self._parse_timestamp(cluster.last_seen_ts)
        except Exception:
            return 0.0
        return max(0.0, (t1 - t0).total_seconds())

    def _parse_timestamp(self, text: str) -> datetime:
        if not text:
            raise ValueError("empty timestamp")
        cleaned = text.strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        return datetime.fromisoformat(cleaned)

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

    def _maybe_emit_cluster(self, cluster: DetectionCluster):
        if cluster.emitted:
            return
        qualifies, reasons = self._cluster_gate_status(cluster)
        if not qualifies:
            best_seg = cluster.best_seg
            self._log(
                "cluster_reject",
                baseline_id=self.baseline_ctx.id,
                center_hz=self._cluster_center_hz(cluster),
                width_hz=max(float(cluster.f_high_hz - cluster.f_low_hz), 0.0),
                hits=cluster.hits,
                windows=len(cluster.windows),
                window_ratio=self._cluster_window_ratio(cluster),
                duration_s=self._cluster_duration_seconds(cluster),
                snr_db=best_seg.snr_db,
                peak_db=best_seg.peak_db,
                noise_db=best_seg.noise_db,
                min_width_hz=self.min_width_hz,
                reasons=reasons,
            )
            return
        self._emit_detection(cluster)

    def _cluster_gate_status(self, cluster: DetectionCluster) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        width_hz = float(cluster.f_high_hz - cluster.f_low_hz)
        if width_hz < self.min_width_hz:
            reasons.append(f"width={width_hz:.1f} < min_width={self.min_width_hz:.1f}")
        if cluster.hits < self.min_hits:
            reasons.append(f"hits={cluster.hits} < min_hits={self.min_hits}")
        win_count = len(cluster.windows)
        if win_count < self.min_windows:
            reasons.append(f"windows={win_count} < min_windows={self.min_windows}")
        ratio_threshold = float(self.persistence_hit_ratio)
        ratio_value = self._cluster_window_ratio(cluster)
        ratio_ok = True if ratio_threshold <= 0.0 else (ratio_value >= ratio_threshold)
        duration_threshold = float(self.persistence_min_seconds)
        duration_value = self._cluster_duration_seconds(cluster)
        duration_ok = True if duration_threshold <= 0.0 else (duration_value >= duration_threshold)
        mode = self.persistence_mode
        if mode == "duration":
            if not duration_ok:
                reasons.append(
                    f"duration={duration_value:.2f}s < min_duration={duration_threshold:.2f}s"
                )
        elif mode == "both":
            if not ratio_ok:
                reasons.append(
                    f"ratio={ratio_value:.2f} < threshold={ratio_threshold:.2f}"
                )
            if not duration_ok:
                reasons.append(
                    f"duration={duration_value:.2f}s < min_duration={duration_threshold:.2f}s"
                )
        else:  # hits ratio mode
            if not ratio_ok:
                reasons.append(
                    f"ratio={ratio_value:.2f} < threshold={ratio_threshold:.2f}"
                )
        return (len(reasons) == 0, reasons)

    def _cluster_qualifies(self, cluster: DetectionCluster) -> bool:
        qualifies, _ = self._cluster_gate_status(cluster)
        return qualifies

    def _emit_detection(self, cluster: DetectionCluster):
        cluster.emitted = True
        best_seg = cluster.best_seg
        confidence = self._compute_confidence(cluster)
        cluster_center_hz = self._cluster_center_hz(cluster)
        window_ratio = self._cluster_window_ratio(cluster)
        duration_seconds = self._cluster_duration_seconds(cluster)
        emit_low, emit_high = self._shape_emit_span(cluster_center_hz, cluster.f_low_hz, cluster.f_high_hz)
        cluster.f_low_hz = emit_low
        cluster.f_high_hz = emit_high
        span_width = max(float(cluster.f_high_hz - cluster.f_low_hz), float(best_seg.bandwidth_hz), self.bin_hz)
        combined_seg = Segment(
            f_low_hz=cluster.f_low_hz,
            f_high_hz=cluster.f_high_hz,
            f_center_hz=cluster_center_hz,
            peak_db=best_seg.peak_db,
            noise_db=best_seg.noise_db,
            snr_db=best_seg.snr_db,
            bandwidth_hz=span_width,
        )
        svc, reg, note = self.bandplan.lookup(combined_seg.f_center_hz)

        is_new_detection = self._persist_detection(cluster, combined_seg, confidence)
        self._pending_emits += 1
        if is_new_detection:
            self._pending_new_signals += 1

        occ_ratio = self._lookup_occ_ratio(combined_seg.f_center_hz)
        is_new_flag = bool(is_new_detection or (occ_ratio is not None and occ_ratio < self.args.new_ema_occ))

        self._log(
            "cluster_emit",
            baseline_id=self.baseline_ctx.id,
            center_hz=combined_seg.f_center_hz,
            width_hz=combined_seg.bandwidth_hz,
            snr_db=combined_seg.snr_db,
            peak_db=combined_seg.peak_db,
            noise_db=combined_seg.noise_db,
            confidence=confidence,
            hits=cluster.hits,
            windows=len(cluster.windows),
            window_ratio=window_ratio,
            duration_s=duration_seconds,
            is_new=is_new_flag,
            occ_ratio=occ_ratio,
            service=svc,
            region=reg,
        )

        record = {
            "baseline_id": self.baseline_ctx.id,
            "time_utc": utc_now_str(),
            "f_center_hz": combined_seg.f_center_hz,
            "f_low_hz": combined_seg.f_low_hz,
            "f_high_hz": combined_seg.f_high_hz,
            "bandwidth_hz": combined_seg.bandwidth_hz,
            "peak_db": combined_seg.peak_db,
            "noise_db": combined_seg.noise_db,
            "snr_db": combined_seg.snr_db,
            "service": svc,
            "region": reg,
            "notes": note,
            "is_new": is_new_flag,
            "confidence": confidence,
            "window_ratio": window_ratio,
            "duration_s": duration_seconds,
            "persistence_mode": self.persistence_mode,
        }
        if self.profile_name:
            record["profile"] = self.profile_name
        maybe_emit_jsonl(self.args.jsonl, record)
        if is_new_flag:
            body = f"{combined_seg.f_center_hz/1e6:.6f} MHz; SNR {combined_seg.snr_db:.1f} dB; {svc or 'Unknown'} {reg or ''}"
            maybe_notify("SDRWatch: New signal", body, self.args.notify)

    def _persist_detection(self, cluster: DetectionCluster, seg: Segment, confidence: float) -> bool:
        timestamp = utc_now_str()
        match = self._match_persistent(seg)
        cluster_center_hz = seg.f_center_hz
        self.store.begin()
        try:
            if match:
                match.f_low_hz = min(match.f_low_hz, cluster.f_low_hz)
                match.f_high_hz = max(match.f_high_hz, cluster.f_high_hz)
                match.f_center_hz = self._blend_centers(match.f_center_hz, match.total_hits, cluster_center_hz, cluster.hits)
                match.last_seen_utc = timestamp
                match.total_hits += cluster.hits
                match.total_windows += len(cluster.windows)
                match.confidence = confidence
                self.store.update_baseline_detection(match)
                is_new = False
            else:
                detection_id = self.store.insert_baseline_detection(
                    self.baseline_ctx.id,
                    cluster.f_low_hz,
                    cluster.f_high_hz,
                    cluster_center_hz,
                    cluster.first_seen_ts,
                    cluster.last_seen_ts,
                    cluster.hits,
                    len(cluster.windows),
                    confidence,
                )
                new_det = PersistentDetection(
                    id=detection_id,
                    baseline_id=self.baseline_ctx.id,
                    f_low_hz=cluster.f_low_hz,
                    f_high_hz=cluster.f_high_hz,
                    f_center_hz=cluster_center_hz,
                    first_seen_utc=cluster.first_seen_ts,
                    last_seen_utc=cluster.last_seen_ts,
                    total_hits=cluster.hits,
                    total_windows=len(cluster.windows),
                    confidence=confidence,
                )
                self._persisted.append(new_det)
                self._seen_persistent.add(detection_id)
                is_new = True
                if self.two_pass_enabled:
                    self._schedule_revisit(detection_id=detection_id, seg=seg, reason="new")
        finally:
            self.store.commit()
        return is_new

    def _match_persistent(self, seg: Segment) -> Optional[PersistentDetection]:
        for det in self._persisted:
            spans_overlap = not (
                seg.f_high_hz < (det.f_low_hz - self.freq_merge_hz)
                or seg.f_low_hz > (det.f_high_hz + self.freq_merge_hz)
            )
            center_close = abs(seg.f_center_hz - det.f_center_hz) <= self.center_match_hz
            if spans_overlap or center_close:
                self._seen_persistent.add(det.id)
                if det.missing_since_utc:
                    det.missing_since_utc = None
                    self.store.clear_detection_missing(det.id, det.baseline_id)
                self._log(
                    "persist_match",
                    detection_id=det.id,
                    baseline_id=self.baseline_ctx.id,
                    center_delta_hz=int(seg.f_center_hz - det.f_center_hz),
                    spans_overlap=spans_overlap,
                    center_close=center_close,
                    seg_width_hz=max(seg.bandwidth_hz, 0.0),
                    persisted_width_hz=max(det.f_high_hz - det.f_low_hz, 0),
                )
                return det
        self._log(
            "persist_no_match",
            baseline_id=self.baseline_ctx.id,
            center_hz=seg.f_center_hz,
            width_hz=max(seg.bandwidth_hz, 0.0),
        )
        return None

    def _find_persistent_by_id(self, detection_id: int) -> Optional[PersistentDetection]:
        for det in self._persisted:
            if det.id == detection_id:
                return det
        return None

    def _lookup_occ_ratio(self, freq_hz: int) -> Optional[float]:
        bin_index = self._bin_index_for_freq(freq_hz)
        if bin_index is None:
            return None
        return self.store.baseline_occ_ratio(self.baseline_ctx.id, bin_index)

    def _bin_index_for_freq(self, freq_hz: int) -> Optional[int]:
        if freq_hz < self.baseline_ctx.freq_start_hz or freq_hz > self.baseline_ctx.freq_stop_hz:
            return None
        offset = (freq_hz - self.baseline_ctx.freq_start_hz) / max(self.baseline_ctx.bin_hz, 1.0)
        return int(round(offset))

    def _schedule_revisit(self, *, detection_id: Optional[int], seg: Segment, reason: str) -> None:
        if not self.two_pass_enabled:
            return
        margin = max(self.revisit_margin_hz, float(seg.bandwidth_hz or self.bin_hz))
        low = int(max(seg.f_low_hz - margin, 0))
        high = int(seg.f_high_hz + margin)
        tag_id = f"rv{self.baseline_ctx.id}_{self._tag_counter}"
        self._tag_counter += 1
        tag = RevisitTag(
            tag_id=tag_id,
            detection_id=detection_id,
            f_center_hz=int(seg.f_center_hz),
            f_low_hz=low,
            f_high_hz=high,
            reason=reason,
            created_utc=utc_now_str(),
        )
        blocked = reason != "missing" and self._tag_overlaps_known(tag)
        if blocked:
            self._log(
                "revisit_queue",
                action="skipped_overlap",
                tag_id=tag.tag_id,
                detection_id=detection_id,
                reason=reason,
                center_hz=tag.f_center_hz,
                width_hz=max(tag.f_high_hz - tag.f_low_hz, 0),
            )
            return
        self._revisit_tags.append(tag)
        self._log(
            "revisit_queue",
            action="queued",
            tag_id=tag.tag_id,
            detection_id=detection_id,
            reason=reason,
            center_hz=tag.f_center_hz,
            width_hz=max(tag.f_high_hz - tag.f_low_hz, 0),
        )

    def _tag_overlaps_known(self, tag: RevisitTag) -> bool:
        for det in self._persisted:
            if det.id == tag.detection_id:
                continue
            if det.missing_since_utc:
                continue
            if not (tag.f_high_hz < det.f_low_hz or tag.f_low_hz > det.f_high_hz):
                return True
        return False

    def _filter_tags(self, tags: List[RevisitTag]) -> List[RevisitTag]:
        seen: Set[str] = set()
        filtered: List[RevisitTag] = []
        dup_dropped = 0
        overlap_dropped = 0
        for tag in tags:
            key = f"{tag.detection_id}:{tag.f_center_hz}:{tag.reason}"
            if key in seen:
                dup_dropped += 1
                continue
            if tag.reason != "missing" and self._tag_overlaps_known(tag):
                overlap_dropped += 1
                continue
            seen.add(key)
            filtered.append(tag)
        self._log(
            "revisit_filter_summary",
            input=len(tags),
            output=len(filtered),
            duplicates=dup_dropped,
            overlap_blocked=overlap_dropped,
        )
        return filtered

    def finalize_coarse_pass(self) -> List[RevisitTag]:
        missing_ts = utc_now_str()
        to_mark: List[PersistentDetection] = []
        for det in self._persisted:
            if det.id in self._seen_persistent:
                continue
            to_mark.append(det)
            det.missing_since_utc = det.missing_since_utc or missing_ts
            self._log(
                "persist_missing",
                detection_id=det.id,
                baseline_id=det.baseline_id,
                center_hz=det.f_center_hz,
                width_hz=max(det.f_high_hz - det.f_low_hz, 0),
            )
            if self.two_pass_enabled:
                self._schedule_revisit(
                    detection_id=det.id,
                    seg=Segment(
                        f_low_hz=det.f_low_hz,
                        f_high_hz=det.f_high_hz,
                        f_center_hz=det.f_center_hz,
                        peak_db=0.0,
                        noise_db=0.0,
                        snr_db=0.0,
                        bandwidth_hz=float(det.f_high_hz - det.f_low_hz),
                    ),
                    reason="missing",
                )
        if to_mark:
            self.store.begin()
            for det in to_mark:
                self.store.mark_detection_missing(det.id, det.baseline_id, missing_ts)
            self.store.commit()
        tags = self._filter_tags(self._revisit_tags) if self.two_pass_enabled else []
        self._revisit_tags = []
        self._log(
            "sweep_finalize",
            seen_persistent=len(self._seen_persistent),
            missing_marked=len(to_mark),
            tags_emitted=len(tags),
        )
        self._seen_persistent.clear()
        return tags

    def apply_revisit_confirmation(self, tag: RevisitTag, seg: Segment) -> None:
        if not tag.detection_id:
            return
        det = self._find_persistent_by_id(tag.detection_id)
        if det is None:
            return
        det.f_low_hz = min(det.f_low_hz, seg.f_low_hz)
        det.f_high_hz = max(det.f_high_hz, seg.f_high_hz)
        det.f_center_hz = int(seg.f_center_hz)
        det.last_seen_utc = utc_now_str()
        det.missing_since_utc = None
        self.store.begin()
        self.store.update_baseline_detection(det)
        self.store.commit()
        self._log(
            "revisit_apply",
            action="confirmed",
            tag_id=tag.tag_id,
            detection_id=det.id,
            center_hz=det.f_center_hz,
            width_hz=max(det.f_high_hz - det.f_low_hz, 0),
        )

    def apply_revisit_miss(self, tag: RevisitTag) -> None:
        if tag.detection_id is None:
            return
        det = self._find_persistent_by_id(tag.detection_id)
        if det is None:
            return
        timestamp = utc_now_str()
        if tag.reason == "new":
            self.store.begin()
            self.store.delete_baseline_detection(det.id, det.baseline_id)
            self.store.commit()
            self._persisted = [d for d in self._persisted if d.id != det.id]
            self._log(
                "revisit_apply",
                action="pruned",
                tag_id=tag.tag_id,
                detection_id=det.id,
                reason=tag.reason,
                center_hz=det.f_center_hz,
            )
            return
        det.missing_since_utc = det.missing_since_utc or timestamp
        self.store.begin()
        self.store.mark_detection_missing(det.id, det.baseline_id, timestamp)
        self.store.commit()
        self._log(
            "revisit_apply",
            action="marked_missing",
            tag_id=tag.tag_id,
            detection_id=det.id,
            reason=tag.reason,
            center_hz=det.f_center_hz,
        )

    def _prune_clusters(self, window_idx: int, force: bool = False):
        to_remove: List[DetectionCluster] = []
        for cluster in self.clusters:
            gap = window_idx - cluster.last_window
            if force or gap > self.max_gap_windows:
                if not cluster.emitted and self._cluster_qualifies(cluster):
                    self._emit_detection(cluster)
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
        hit_component = float(np.clip(cluster.hits / self.conf_hit_normalizer, 0.0, 1.0))
        span_windows = max(cluster.last_window - cluster.first_window + 1, 1)
        persistence_component = float(np.clip(len(cluster.windows) / span_windows, 0.0, 1.0))
        duration_component = float(np.clip(span_windows / self.conf_duration_norm, 0.0, 1.0))
        raw_score = (
            0.45 * snr_component
            + 0.25 * hit_component
            + 0.2 * persistence_component
            + 0.1 * duration_component
        )
        raw_score += self.confidence_bias
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

    def _drain_pending_emits(self) -> Tuple[int, int]:
        emitted = self._pending_emits
        new_emitted = self._pending_new_signals
        self._pending_emits = 0
        self._pending_new_signals = 0
        return emitted, new_emitted


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


class ScanLogger:
    def __init__(self, log_path: Path, mirror_paths: Optional[List[Path]] = None):
        self.log_path = log_path
        self.mirror_paths: List[Path] = []
        self._ensure_parent(self.log_path)
        seen: Set[str] = {str(self.log_path)}
        for mirror in mirror_paths or []:
            try:
                resolved = mirror
                if not resolved.is_absolute():
                    resolved = (Path.cwd() / resolved).absolute()
                if str(resolved) in seen:
                    continue
                self._ensure_parent(resolved)
                self.mirror_paths.append(resolved)
                seen.add(str(resolved))
            except Exception:
                continue
        self.run_id = f"run-{int(time.time() * 1000)}-pid{os.getpid()}"
        self.current_sweep: Optional[int] = None

    @staticmethod
    def _ensure_parent(path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    @classmethod
    def from_db_path(cls, db_path: str, extra_targets: Optional[List[str]] = None) -> "ScanLogger":
        if not db_path or db_path == ":memory:":
            base_dir = Path.cwd()
        else:
            expanded = Path(db_path).expanduser()
            if not expanded.is_absolute():
                expanded = (Path.cwd() / expanded).absolute()
            base_dir = expanded.parent if expanded.parent != Path("") else Path.cwd()
        log_path = base_dir / "sdrwatch-scan.log"
        extra_paths: List[Path] = []
        for target in extra_targets or []:
            if not target:
                continue
            extra_paths.append(Path(target).expanduser())
        return cls(log_path, extra_paths)

    def start_sweep(self, sweep_id: int, **metadata: Any) -> None:
        self.current_sweep = sweep_id
        self.log("sweep_start", **metadata)

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "ts": utc_now_str(),
            "run_id": self.run_id,
            "sweep_id": self.current_sweep,
            "event": event,
            **fields,
        }
        targets = [self.log_path] + self.mirror_paths
        for target in targets:
            try:
                with target.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
            except Exception:
                continue


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

    if prof.step_hz is not None:
        maybe_set("step", prof.step_hz)
    maybe_set("samp_rate", prof.samp_rate)
    maybe_set("fft", prof.fft)
    maybe_set("avg", prof.avg)
    maybe_set("threshold_db", prof.threshold_db)
    maybe_set("guard_bins", prof.guard_bins)
    maybe_set("min_width_bins", prof.min_width_bins)
    if prof.cfar_train is not None:
        maybe_set("cfar_train", prof.cfar_train)
    if prof.cfar_guard is not None:
        maybe_set("cfar_guard", prof.cfar_guard)
    if prof.cfar_quantile is not None:
        maybe_set("cfar_quantile", prof.cfar_quantile)
    if prof.persistence_hit_ratio is not None:
        maybe_set("persistence_hit_ratio", prof.persistence_hit_ratio)
    if prof.persistence_min_seconds is not None:
        maybe_set("persistence_min_seconds", prof.persistence_min_seconds)
    if prof.persistence_min_hits is not None:
        maybe_set("persistence_min_hits", prof.persistence_min_hits)
    if prof.persistence_min_windows is not None:
        maybe_set("persistence_min_windows", prof.persistence_min_windows)
    if prof.revisit_fft is not None:
        maybe_set("revisit_fft", prof.revisit_fft)
    if prof.revisit_avg is not None:
        maybe_set("revisit_avg", prof.revisit_avg)
    if prof.revisit_margin_hz is not None:
        maybe_set("revisit_margin_hz", prof.revisit_margin_hz)
    if prof.revisit_max_bands is not None:
        maybe_set("revisit_max_bands", prof.revisit_max_bands)
    if prof.revisit_floor_threshold_db is not None:
        maybe_set("revisit_floor_threshold_db", prof.revisit_floor_threshold_db)
    if prof.two_pass is not None:
        maybe_set("two_pass", bool(prof.two_pass))
    if prof.bandwidth_pad_hz is not None:
        setattr(args, "bandwidth_pad_hz", float(prof.bandwidth_pad_hz))
    if prof.min_emit_bandwidth_hz is not None:
        setattr(args, "min_emit_bandwidth_hz", float(prof.min_emit_bandwidth_hz))
    if prof.confidence_hit_normalizer is not None:
        setattr(args, "confidence_hit_normalizer", float(prof.confidence_hit_normalizer))
    if prof.confidence_duration_norm is not None:
        setattr(args, "confidence_duration_norm", float(prof.confidence_duration_norm))
    if prof.confidence_bias is not None:
        setattr(args, "confidence_bias", float(prof.confidence_bias))

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


def _resolve_baseline_context(store: "Store", baseline_arg) -> BaselineContext:
    if baseline_arg is None:
        raise SystemExit("--baseline-id must be provided")
    if isinstance(baseline_arg, str) and baseline_arg.lower() == "latest":
        baseline_id = store.get_latest_baseline_id()
        if baseline_id is None:
            raise SystemExit("No baselines exist yet; create one before running scans")
    else:
        try:
            baseline_id = int(baseline_arg)
        except Exception as exc:  # pragma: no cover - defensive
            raise SystemExit(f"Invalid --baseline-id '{baseline_arg}': {exc}")
    ctx = store.get_baseline(baseline_id)
    if ctx is None:
        raise SystemExit(f"Baseline id {baseline_id} not found in database")
    return ctx


def _do_one_sweep(
    args,
    store: Store,
    bandplan: Bandplan,
    src,
    baseline_ctx: BaselineContext,
    sweep_seq: int,
    logger: Optional[ScanLogger] = None,
) -> None:
    """Perform a single full sweep across [start, stop] inclusive, updating baseline stats/detections."""
    bin_hz = float(args.samp_rate) / float(args.fft) if args.fft else float(args.samp_rate)
    if baseline_ctx.bin_hz > 0:
        diff = abs(bin_hz - baseline_ctx.bin_hz)
        if diff > max(1.0, baseline_ctx.bin_hz * 0.05):
            print(
                f"[baseline] WARNING: sweep bin {bin_hz:.2f} Hz differs from baseline bin {baseline_ctx.bin_hz:.2f} Hz",
                file=sys.stderr,
            )
    detection_engine: Optional[DetectionEngine]
    if args.spur_calibration:
        detection_engine = None
    else:
        detection_engine = DetectionEngine(
            store,
            bandplan,
            args,
            bin_hz=bin_hz,
            baseline_ctx=baseline_ctx,
            min_hits=int(getattr(args, "persistence_min_hits", 2)),
            min_windows=int(getattr(args, "persistence_min_windows", 2)),
            logger=logger,
        )
    power_monitor = WindowPowerMonitor()
    spur_tracker: Dict[int, Dict[str, float]] = {}

    if logger:
        sweep_params = {
            "start_hz": args.start,
            "stop_hz": args.stop,
            "step_hz": args.step,
            "samp_rate_hz": args.samp_rate,
            "fft": args.fft,
            "avg": args.avg,
            "threshold_db": args.threshold_db,
            "guard_bins": args.guard_bins,
            "min_width_bins": args.min_width_bins,
            "cfar_mode": args.cfar,
            "cfar_train": args.cfar_train,
            "cfar_guard": args.cfar_guard,
            "cfar_quantile": args.cfar_quantile,
            "cfar_alpha_db": args.cfar_alpha_db,
            "gain": args.gain,
            "driver": args.driver,
            "profile": getattr(args, "profile", None),
            "spur_calibration": bool(args.spur_calibration),
            "two_pass": bool(getattr(args, "two_pass", False)),
            "persistence_mode": getattr(args, "persistence_mode", None),
            "persistence_hit_ratio": getattr(args, "persistence_hit_ratio", None),
            "persistence_min_seconds": getattr(args, "persistence_min_seconds", None),
            "persistence_min_hits": getattr(args, "persistence_min_hits", None),
            "persistence_min_windows": getattr(args, "persistence_min_windows", None),
            "bandwidth_pad_hz": getattr(args, "bandwidth_pad_hz", None),
            "min_emit_bandwidth_hz": getattr(args, "min_emit_bandwidth_hz", None),
            "confidence_hit_normalizer": getattr(args, "confidence_hit_normalizer", None),
            "confidence_duration_norm": getattr(args, "confidence_duration_norm", None),
            "confidence_bias": getattr(args, "confidence_bias", None),
        }
        logger.start_sweep(
            sweep_seq,
            baseline_id=baseline_ctx.id,
            baseline_span_hz=[baseline_ctx.freq_start_hz, baseline_ctx.freq_stop_hz],
            bin_hz=bin_hz,
            params=sweep_params,
        )

    total_segments = 0
    total_hits = 0
    total_new_signals = 0
    total_promoted = 0
    total_revisits = 0
    total_revisit_confirmed = 0
    total_revisit_false = 0

    try:
        center = args.start
        window_idx = 0
        print(
            f"[scan] begin sweep baseline={baseline_ctx.id} range={args.start/1e6:.3f}-{args.stop/1e6:.3f} MHz step={args.step/1e6:.3f} samp_rate={args.samp_rate/1e6:.3f} fft={args.fft} avg={args.avg}",
            flush=True,
        )
        warned_bin_mismatch = False
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
            segs, occ_mask_cfar, noise_per_bin_db = detect_segments(
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

            if logger:
                widths = np.array([max(float(seg.bandwidth_hz), 0.0) for seg in segs], dtype=float)
                avg_bw = float(np.mean(widths)) if widths.size else None
                min_bw = float(np.min(widths)) if widths.size else None
                median_bw = float(np.median(widths)) if widths.size else None
                max_bw = float(np.max(widths)) if widths.size else None
                strongest_snr = max((seg.snr_db for seg in segs), default=None)
                logger.log(
                    "segment_inventory",
                    window_idx=window_idx,
                    center_hz=float(center),
                    num_segments=len(segs),
                    avg_bandwidth_hz=avg_bw,
                    min_bandwidth_hz=min_bw,
                    median_bandwidth_hz=median_bw,
                    max_bandwidth_hz=max_bw,
                    strongest_snr_db=strongest_snr,
                )

            mean_psd_db = float(np.mean(psd_db))
            p90_psd_db = float(np.percentile(psd_db, 90.0))
            is_anom, _ema_power_db, _delta_db = power_monitor.update(mean_psd_db)
            accepted_hits = 0
            spur_ignored = 0
            promoted = 0
            new_signals = 0

            if is_anom:
                if detection_engine:
                    accepted_hits, spur_ignored, promoted, new_signals = detection_engine.ingest(window_idx, [])
            else:
                # Occupancy mask per bin for baseline update
                noise_db = robust_noise_floor_db(psd_db)
                dynamic = noise_db + args.threshold_db
                occupied_mask = occ_mask_cfar if (args.cfar and args.cfar != 'off') else (psd_db > dynamic)

                # Baseline stats update
                if baseline_ctx.bin_hz > 0:
                    bin_pos = (rf_freqs - baseline_ctx.freq_start_hz) / baseline_ctx.bin_hz
                    valid_mask = (rf_freqs >= baseline_ctx.freq_start_hz) & (rf_freqs <= baseline_ctx.freq_stop_hz)
                else:
                    if not warned_bin_mismatch:
                        print("[baseline] bin_hz invalid; skipping stats update", file=sys.stderr)
                        warned_bin_mismatch = True
                    valid_mask = np.zeros_like(rf_freqs, dtype=bool)

                if valid_mask.any():
                    bin_indices = np.rint(bin_pos[valid_mask]).astype(int)
                    noise_vec = noise_per_bin_db[valid_mask].astype(float)
                    window_ts = utc_now_str()
                    store.begin()
                    store.update_baseline_stats(
                        baseline_ctx.id,
                        bin_indices,
                        noise_floor_db=noise_vec,
                        power_db=psd_db[valid_mask].astype(float),
                        occupied_mask=occupied_mask[valid_mask],
                        timestamp_utc=window_ts,
                    )
                    total_windows = store.increment_baseline_windows(baseline_ctx.id, 1)
                    baseline_ctx.total_windows = total_windows
                    store.commit()
                elif not warned_bin_mismatch:
                    print(
                        f"[baseline] sweep window {center/1e6:.3f} MHz outside baseline span {baseline_ctx.freq_start_hz/1e6:.3f}-{baseline_ctx.freq_stop_hz/1e6:.3f} MHz",
                        file=sys.stderr,
                    )
                    warned_bin_mismatch = True

                if args.spur_calibration:
                    _track_spur_hits(spur_tracker, segs)
                if detection_engine:
                    accepted_hits, spur_ignored, promoted, new_signals = detection_engine.ingest(window_idx, segs)
            total_segments += len(segs)
            total_hits += accepted_hits
            total_promoted += promoted
            total_new_signals += new_signals
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
                        f"new_sig={new_signals}",
                        f"spur_masked={spur_ignored}",
                    ]
                )
            print(f"[scan] window {' '.join(fields)}", flush=True)
            # Advance center frequency
            center += args.step

        revisit_tags: List[RevisitTag] = []
        if detection_engine:
            revisit_tags = detection_engine.finalize_coarse_pass()
        if detection_engine and getattr(args, "two_pass", False) and revisit_tags:
            max_bands = int(getattr(args, "revisit_max_bands", 0) or 0)
            if max_bands > 0:
                revisit_tags = revisit_tags[:max_bands]
            revisit_stats = _run_revisit_pass(
                args,
                src,
                detection_engine,
                revisit_tags,
                logger,
            )
            total_revisits += revisit_stats.get("total", 0)
            total_revisit_confirmed += revisit_stats.get("confirmed", 0)
            total_revisit_false += revisit_stats.get("false_positive", 0)

    finally:
        if detection_engine:
            flushed, new_flush = detection_engine.flush()
            if flushed:
                total_promoted += flushed
                total_new_signals += new_flush
                print(f"[scan] sweep baseline={baseline_ctx.id} flushed pending detections={flushed}", flush=True)
        if args.spur_calibration:
            _persist_spur_calibration(store, spur_tracker, window_idx)
        store.begin()
        store.insert_scan_update(
            baseline_ctx.id,
            utc_now_str(),
            num_hits=total_hits,
            num_segments=total_segments,
            num_new_signals=total_new_signals,
            num_revisits=total_revisits,
            num_confirmed=total_revisit_confirmed,
            num_false_positive=total_revisit_false,
        )
        store.commit()
        updated_start, updated_stop = store.update_baseline_span(
            baseline_ctx.id,
            min(args.start, args.stop),
            max(args.start, args.stop),
        )
        if updated_start:
            baseline_ctx.freq_start_hz = updated_start
        if updated_stop:
            baseline_ctx.freq_stop_hz = updated_stop
        print(
            f"[scan] end sweep baseline={baseline_ctx.id} hits={total_hits} promoted={total_promoted} new={total_new_signals}",
            flush=True,
        )
        if logger:
            logger.log(
                "sweep_summary",
                baseline_id=baseline_ctx.id,
                hits=total_hits,
                segments=total_segments,
                promoted=total_promoted,
                new_signals=total_new_signals,
                revisits_total=total_revisits,
                revisits_confirmed=total_revisit_confirmed,
                revisits_false_positive=total_revisit_false,
            )


def _select_revisit_segment(tag: RevisitTag, segments: List[Segment]) -> Optional[Segment]:
    for seg in segments:
        if seg.f_low_hz <= tag.f_center_hz <= seg.f_high_hz:
            return seg
        if tag.f_low_hz <= seg.f_center_hz <= tag.f_high_hz:
            return seg
    return None


def _run_revisit_pass(
    args,
    src,
    detection_engine: DetectionEngine,
    tags: List[RevisitTag],
    logger: Optional[ScanLogger] = None,
) -> Dict[str, int]:
    stats = {"total": len(tags), "confirmed": 0, "false_positive": 0}
    if not tags:
        return stats

    revisit_fft = int(getattr(args, "revisit_fft", 0) or max(int(args.fft), int(args.fft * 2)))
    revisit_avg = int(getattr(args, "revisit_avg", 0) or max(int(args.avg), 4))
    revisit_threshold = float(getattr(args, "revisit_floor_threshold_db", args.threshold_db))
    revisit_guard = int(getattr(args, "guard_bins", 1))
    revisit_min_width_bins = int(max(1, getattr(args, "min_width_bins", 2)))
    raw_margin = getattr(args, "revisit_margin_hz", None)
    if raw_margin is None or float(raw_margin) <= 0.0:
        revisit_margin_hz = float(getattr(detection_engine, "revisit_margin_hz", 0.0))
    else:
        revisit_margin_hz = float(raw_margin)
    revisit_params = {
        "fft": revisit_fft,
        "avg": revisit_avg,
        "threshold_db": revisit_threshold,
        "guard_bins": revisit_guard,
        "min_width_bins": revisit_min_width_bins,
        "margin_hz": revisit_margin_hz,
        "samp_rate_hz": args.samp_rate,
        "two_pass": bool(getattr(args, "two_pass", False)),
        "max_bands": int(getattr(args, "revisit_max_bands", 0) or 0),
    }
    if logger:
        logger.log(
            "revisit_start",
            baseline_id=detection_engine.baseline_ctx.id,
            tag_count=len(tags),
            revisit_params=revisit_params,
        )

    for idx, tag in enumerate(tags, start=1):
        print(f"[revisit] tag={tag.tag_id} reason={tag.reason} center={tag.f_center_hz/1e6:.6f}MHz", flush=True)
        try:
            src.tune(tag.f_center_hz)
            _ = src.read(int(revisit_fft))
            samples = src.read(int(revisit_fft * revisit_avg))
        except Exception as exc:
            print(f"[revisit] tag={tag.tag_id} tune_error={exc}", file=sys.stderr)
            detection_engine.apply_revisit_miss(tag)
            if tag.reason == "new":
                stats["false_positive"] += 1
            if logger:
                logger.log(
                    "revisit_result",
                    tag_id=tag.tag_id,
                    reason=tag.reason,
                    center_hz=tag.f_center_hz,
                    coarse_width_hz=max(tag.f_high_hz - tag.f_low_hz, 0),
                    error=str(exc),
                    matched=False,
                )
            continue

        baseband_f, psd_db = compute_psd_db(samples, args.samp_rate, revisit_fft, revisit_avg)
        rf_freqs = baseband_f + tag.f_center_hz
        segs, _, _ = detect_segments(
            rf_freqs,
            psd_db,
            thresh_db=revisit_threshold,
            guard_bins=revisit_guard,
            min_width_bins=revisit_min_width_bins,
            cfar_mode=args.cfar,
            cfar_train=args.cfar_train,
            cfar_guard=args.cfar_guard,
            cfar_quantile=args.cfar_quantile,
            cfar_alpha_db=args.cfar_alpha_db,
            abs_power_floor_db=getattr(args, "abs_power_floor_db", None),
        )
        match = _select_revisit_segment(tag, segs)
        if match:
            detection_engine.apply_revisit_confirmation(tag, match)
            stats["confirmed"] += 1
        else:
            detection_engine.apply_revisit_miss(tag)
            if tag.reason == "new":
                stats["false_positive"] += 1
        if logger:
            measured_width = float(match.bandwidth_hz) if match else None
            logger.log(
                "revisit_result",
                tag_id=tag.tag_id,
                reason=tag.reason,
                center_hz=tag.f_center_hz,
                coarse_width_hz=max(tag.f_high_hz - tag.f_low_hz, 0),
                matched=bool(match),
                measured_width_hz=measured_width,
                segment_count=len(segs),
                strongest_snr_db=(max(seg.snr_db for seg in segs) if segs else None),
            )
    if logger:
        logger.log(
            "revisit_summary",
            total=stats.get("total", 0),
            confirmed=stats.get("confirmed", 0),
            false_positive=stats.get("false_positive", 0),
        )
    return stats


def run(args):
    """Scanning pipeline overview.

    run() decides how many sweeps execute by inspecting --loop/--repeat/--duration,
    then repeatedly calls _do_one_sweep() until that termination policy is satisfied.
    _do_one_sweep() walks center frequency from --start to --stop in --step increments,
    tuning the SDR, collecting samples, and invoking compute_psd_db() to turn each
    capture into a baseband PSD. detect_segments() (with CFAR via cfar_os_mask) turns
    each PSD into contiguous hits, while Store.update_baseline_stats() ingests every
    bin's PSD/noise occupancy EMA tied to the active baseline. DetectionEngine
    accumulates those hits over multiple windows, consults spur_map to suppress known
    hardware artifacts, and only promotes baseline_detections (plus JSONL/notifies)
    once a frequency region shows persistent energy. In --spur-calibration mode the
    same sweep populates spur_map instead of emitting detections. After each sweep,
    scan_updates rows summarize hits/segments/new-signal counts for the baseline.
    """
    if getattr(args, "list_profiles", False):
        _emit_profiles_json()
        return

    # Optional TMPDIR override (steer off tmpfs /tmp on Trixie)
    if args.tmpdir:
        os.environ["TMPDIR"] = args.tmpdir

    extra_targets = [args.jsonl] if getattr(args, "jsonl", None) else None
    logger = ScanLogger.from_db_path(args.db, extra_targets=extra_targets)
    bandplan = Bandplan(args.bandplan)
    store = Store(args.db)

    baseline_ctx = _resolve_baseline_context(store, getattr(args, "baseline_id", None))
    logger.log(
        "baseline_context",
        baseline_id=baseline_ctx.id,
        baseline_span_hz=[baseline_ctx.freq_start_hz, baseline_ctx.freq_stop_hz],
        bin_hz=baseline_ctx.bin_hz,
        db_path=os.path.abspath(args.db),
    )
    args.baseline_id = baseline_ctx.id
    planned_start = min(args.start, args.stop)
    planned_stop = max(args.start, args.stop)
    if baseline_ctx.freq_start_hz <= 0:
        baseline_ctx.freq_start_hz = planned_start
    if planned_stop > baseline_ctx.freq_stop_hz:
        baseline_ctx.freq_stop_hz = planned_stop
    if baseline_ctx.freq_start_hz > 0 and baseline_ctx.freq_stop_hz > 0:
        span_text = f"{baseline_ctx.freq_start_hz/1e6:.3f}-{baseline_ctx.freq_stop_hz/1e6:.3f} MHz"
    else:
        span_text = "auto (pending scans)"
    print(
        f"[baseline] using id={baseline_ctx.id} name='{baseline_ctx.name}' span={span_text} bin={baseline_ctx.bin_hz:.1f} Hz",
        flush=True,
    )

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

    sweep_seq = 1
    try:
        while True:
            # Duration check (before starting next sweep)
            if duration_s is not None and (time.time() - start_time) >= duration_s:
                break

            _do_one_sweep(args, store, bandplan, src, baseline_ctx, sweep_seq, logger)
            sweep_seq += 1

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
    p.add_argument(
        "--persistence-mode",
        choices=["hits", "duration", "both"],
        help="Persistence gate: hit/window ratio, wall-clock duration, or both (default hits)",
    )
    p.add_argument(
        "--persistence-hit-ratio",
        dest="persistence_hit_ratio",
        type=float,
        help="Minimum occupied-window ratio (0-1) within a cluster span to mark persistent (default 0.6)",
    )
    p.add_argument(
        "--persistence-min-seconds",
        dest="persistence_min_seconds",
        type=float,
        help="Minimum wall-clock duration in seconds for duration-based persistence (default 10)",
    )
    p.add_argument(
        "--persistence-min-hits",
        dest="persistence_min_hits",
        type=int,
        help="Minimum hits required before persistence evaluation (default 2)",
    )
    p.add_argument(
        "--persistence-min-windows",
        dest="persistence_min_windows",
        type=int,
        help="Minimum distinct windows required before persistence evaluation (default 2)",
    )
    # CFAR options
    p.add_argument("--cfar", choices=["off", "os", "ca"], help="CFAR mode (default: os)")
    p.add_argument("--cfar-train", dest="cfar_train", type=int, help="Training cells per side for CFAR (default 24)")
    p.add_argument("--cfar-guard", dest="cfar_guard", type=int, help="Guard cells per side (excluded around CUT) for CFAR (default 4)")
    p.add_argument("--cfar-quantile", dest="cfar_quantile", type=float, help="Quantile (0..1) for OS-CFAR order statistic (default 0.75)")
    p.add_argument("--cfar-alpha-db", dest="cfar_alpha_db", type=float, help="Override threshold scaling for CFAR in dB; defaults to --threshold-db")

    p.add_argument("--bandplan", type=str, help="Optional bandplan CSV to map detections")
    p.add_argument("--db", type=str, help="SQLite DB path (default sdrwatch.db)")
    p.add_argument("--baseline-id", dest="baseline_id", type=str, help="Baseline id to attach scans to (or 'latest')")
    p.add_argument("--jsonl", type=str, help="Emit detections as line-delimited JSON to this path")
    p.add_argument("--notify", action="store_true", help="Desktop notifications for new signals")
    p.add_argument("--new-ema-occ", dest="new_ema_occ", type=float, help="EMA occupancy threshold to flag a bin as NEW (default 0.02)")
    p.add_argument("--latitude", type=float, help="Optional latitude in decimal degrees for this scan")
    p.add_argument("--longitude", type=float, help="Optional longitude in decimal degrees for this scan")
    p.add_argument("--profile", type=str, help="Scan profile name to pre-load sane defaults (see documentation)")
    p.add_argument("--spur-calibration", dest="spur_calibration", action="store_true", help="Learn persistent internal spurs instead of emitting detections")
    p.add_argument("--list-profiles", dest="list_profiles", action="store_true", help="Print built-in scan profiles as JSON and exit")
    p.add_argument("--two-pass", dest="two_pass", action="store_true", help="Enable coarse + targeted revisit confirmation sweep")
    p.add_argument("--revisit-fft", dest="revisit_fft", type=int, help="FFT size for revisit windows (defaults to 2x --fft)")
    p.add_argument("--revisit-avg", dest="revisit_avg", type=int, help="Averaging factor for revisit windows (defaults to max(--avg,4))")
    p.add_argument(
        "--revisit-margin-hz",
        dest="revisit_margin_hz",
        type=float,
        help="Additional Hz margin added to revisit windows around each tagged center",
    )
    p.add_argument(
        "--revisit-max-bands",
        dest="revisit_max_bands",
        type=int,
        help="Maximum revisit targets per sweep (0 = unlimited)",
    )
    p.add_argument(
        "--revisit-floor-threshold-db",
        dest="revisit_floor_threshold_db",
        type=float,
        help="Detection threshold (dB) used during revisit windows (defaults to --threshold-db)",
    )

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
    _set_default(args, args._cli_overrides, "persistence_mode", "hits")
    _set_default(args, args._cli_overrides, "persistence_hit_ratio", 0.6)
    _set_default(args, args._cli_overrides, "persistence_min_seconds", 10.0)
    _set_default(args, args._cli_overrides, "persistence_min_hits", 2)
    _set_default(args, args._cli_overrides, "persistence_min_windows", 2)
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
    _set_default(args, args._cli_overrides, "two_pass", False)
    _set_default(args, args._cli_overrides, "revisit_fft", None)
    _set_default(args, args._cli_overrides, "revisit_avg", None)
    _set_default(args, args._cli_overrides, "revisit_margin_hz", None)
    _set_default(args, args._cli_overrides, "revisit_max_bands", 0)
    _set_default(args, args._cli_overrides, "revisit_floor_threshold_db", None)
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

    if not args.list_profiles:
        baseline_raw = getattr(args, "baseline_id", None)
        if baseline_raw is None:
            p.error("--baseline-id is required for scanning runs")
        baseline_text = str(baseline_raw).strip()
        if not baseline_text:
            p.error("--baseline-id is required for scanning runs")
        if baseline_text.lower() == "latest":
            setattr(args, "baseline_id", "latest")
        else:
            try:
                baseline_val = int(baseline_text)
            except ValueError:
                p.error("--baseline-id must be an integer or 'latest'")
            setattr(args, "baseline_id", baseline_val)

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
