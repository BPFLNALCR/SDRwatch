"""Baseline database helpers (SQLite-backed Store)."""

from __future__ import annotations

import sqlite3
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np  # type: ignore

from sdrwatch.baseline.model import BaselineContext
from sdrwatch.detection.types import PersistentDetection
from sdrwatch.util.time import utc_now_str


class Store:
    def __init__(self, path: str):
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

        self._ensure_column("baseline_detections", "missing_since_utc", "TEXT")
        self._ensure_column("scan_updates", "num_revisits", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("scan_updates", "num_confirmed", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("scan_updates", "num_false_positive", "INTEGER NOT NULL DEFAULT 0")

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        cur = self.con.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = {row[1] for row in cur.fetchall()}
        if column not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            self.con.commit()

    def begin(self) -> None:
        self.con.execute("BEGIN")

    def commit(self) -> None:
        self.con.commit()

    def rollback(self) -> None:
        self.con.rollback()

    def get_latest_baseline_id(self) -> Optional[int]:
        cur = self.con.cursor()
        cur.execute("SELECT id FROM baselines ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        return int(row[0]) if row else None

    def get_baseline(self, baseline_id: int) -> Optional[BaselineContext]:
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT id, name, freq_start_hz, freq_stop_hz, bin_hz, baseline_version, total_windows
            FROM baselines WHERE id = ?
            """,
            (int(baseline_id),),
        )
        row = cur.fetchone()
        if not row:
            return None
        return BaselineContext(
            id=int(row[0]),
            name=str(row[1]),
            freq_start_hz=int(row[2]),
            freq_stop_hz=int(row[3]),
            bin_hz=float(row[4]),
            baseline_version=int(row[5]),
            total_windows=int(row[6] or 0),
        )

    def create_baseline(
        self,
        *,
        name: Optional[str] = None,
        freq_start_hz: int = 0,
        freq_stop_hz: int = 0,
        bin_hz: float = 0.0,
    ) -> BaselineContext:
        base_name = name or f"baseline-{utc_now_str().replace(':', '').replace('-', '')}"
        created_at = utc_now_str()
        cur = self.con.cursor()
        cur.execute(
            """
            INSERT INTO baselines(name, created_at, freq_start_hz, freq_stop_hz, bin_hz, baseline_version, total_windows)
            VALUES (?, ?, ?, ?, ?, 1, 0)
            """,
            (base_name, created_at, int(freq_start_hz), int(freq_stop_hz), float(bin_hz)),
        )
        self.con.commit()
        lastrow = cur.lastrowid
        if lastrow is None:
            raise RuntimeError("Failed to retrieve lastrowid after inserting baseline")
        baseline_id = int(lastrow)
        ctx = self.get_baseline(baseline_id)
        if ctx is None:
            raise RuntimeError("Failed to create baseline context")
        return ctx

    def update_baseline_span(self, baseline_id: int, low_hz: float, high_hz: float) -> Tuple[Optional[int], Optional[int]]:
        cur = self.con.cursor()
        cur.execute(
            "SELECT freq_start_hz, freq_stop_hz FROM baselines WHERE id = ?",
            (int(baseline_id),),
        )
        row = cur.fetchone()
        if not row:
            return None, None
        current_low, current_high = int(row[0]), int(row[1])
        new_low = min(current_low, int(low_hz)) if current_low > 0 else int(low_hz)
        new_high = max(current_high, int(high_hz)) if current_high > 0 else int(high_hz)
        if new_low != current_low or new_high != current_high:
            cur.execute(
                "UPDATE baselines SET freq_start_hz = ?, freq_stop_hz = ? WHERE id = ?",
                (new_low, new_high, int(baseline_id)),
            )
            self.con.commit()
            return new_low, new_high
        return None, None

    def increment_baseline_windows(self, baseline_id: int, delta: int) -> int:
        cur = self.con.cursor()
        cur.execute(
            """
            UPDATE baselines
            SET total_windows = COALESCE(total_windows, 0) + ?
            WHERE id = ?
            RETURNING total_windows
            """,
            (int(delta), int(baseline_id)),
        )
        row = cur.fetchone()
        self.con.commit()
        return int(row[0]) if row else 0

    def update_baseline_stats(
        self,
        baseline_id: int,
        bin_indices: np.ndarray,
        *,
        noise_floor_db: np.ndarray,
        power_db: np.ndarray,
        occupied_mask: np.ndarray,
        timestamp_utc: str,
        ema_alpha: float = 0.2,
    ) -> None:
        cur = self.con.cursor()
        for idx, noise_db, power_db_val, occupied in zip(bin_indices, noise_floor_db, power_db, occupied_mask):
            idx_int = int(idx)
            cur.execute(
                """
                SELECT noise_floor_ema, power_ema, occ_count FROM baseline_stats
                WHERE baseline_id = ? AND bin_index = ?
                """,
                (int(baseline_id), idx_int),
            )
            row = cur.fetchone()
            if row:
                noise_ema = float(row[0]) if row[0] is not None else float(noise_db)
                power_ema = float(row[1]) if row[1] is not None else float(power_db_val)
                occ_count = int(row[2] or 0)
                noise_ema = (1.0 - ema_alpha) * noise_ema + ema_alpha * float(noise_db)
                power_ema = (1.0 - ema_alpha) * power_ema + ema_alpha * float(power_db_val)
                occ_count = occ_count + (1 if bool(occupied) else 0)
                cur.execute(
                    """
                    UPDATE baseline_stats
                    SET noise_floor_ema = ?, power_ema = ?, occ_count = ?, last_seen_utc = ?
                    WHERE baseline_id = ? AND bin_index = ?
                    """,
                    (noise_ema, power_ema, occ_count, timestamp_utc, int(baseline_id), idx_int),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO baseline_stats(
                        baseline_id, bin_index, noise_floor_ema, power_ema, occ_count, last_seen_utc
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        int(baseline_id),
                        idx_int,
                        float(noise_db),
                        float(power_db_val),
                        1 if bool(occupied) else 0,
                        timestamp_utc,
                    ),
                )
        self.con.commit()

    def load_baseline_detections(self, baseline_id: int) -> List["PersistentDetection"]:
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT id, baseline_id, f_low_hz, f_high_hz, f_center_hz,
                   first_seen_utc, last_seen_utc, total_hits, total_windows,
                   confidence, missing_since_utc
            FROM baseline_detections
            WHERE baseline_id = ?
            ORDER BY f_center_hz
            """,
            (int(baseline_id),),
        )
        detections = []
        for row in cur.fetchall():
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
                    missing_since_utc=row[10],
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
        *,
        missing_since_utc: Optional[str] = None,
    ) -> int:
        cur = self.con.cursor()
        cur.execute(
            """
            INSERT INTO baseline_detections(
                baseline_id, f_low_hz, f_high_hz, f_center_hz,
                first_seen_utc, last_seen_utc, total_hits, total_windows, confidence, missing_since_utc
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
