"""Baseline database helpers (SQLite-backed Store)."""

from __future__ import annotations

import sqlite3
import math
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np  # type: ignore

from sdrwatch.baseline.model import BaselineContext
from sdrwatch.baseline.summary import BandSummaryConfig, band_partitions, tactical_recent_minutes
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
            CREATE TABLE IF NOT EXISTS baseline_noise (
                baseline_id INTEGER NOT NULL,
                bin_index INTEGER NOT NULL,
                noise_floor_ema REAL NOT NULL,
                power_ema REAL NOT NULL,
                last_seen_utc TEXT NOT NULL,
                PRIMARY KEY (baseline_id, bin_index)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS baseline_occupancy (
                baseline_id INTEGER NOT NULL,
                bin_index INTEGER NOT NULL,
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
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS baseline_band_summary (
                baseline_id INTEGER NOT NULL,
                band_index INTEGER NOT NULL,
                f_low_hz INTEGER NOT NULL,
                f_high_hz INTEGER NOT NULL,
                persistent_signals INTEGER NOT NULL,
                recent_new_signals INTEGER NOT NULL,
                occupied_fraction REAL NOT NULL,
                avg_noise_db REAL,
                avg_power_db REAL,
                last_updated_utc TEXT NOT NULL,
                PRIMARY KEY (baseline_id, band_index)
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS baseline_summary_meta (
                baseline_id INTEGER PRIMARY KEY,
                band_count INTEGER NOT NULL,
                band_width_hz REAL NOT NULL,
                recent_minutes INTEGER NOT NULL,
                occ_threshold REAL NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS baseline_snapshot (
                baseline_id INTEGER PRIMARY KEY,
                total_windows INTEGER NOT NULL,
                persistent_detections INTEGER NOT NULL,
                last_detection_utc TEXT,
                last_update_utc TEXT,
                recent_new_signals INTEGER NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        self.con.commit()

        self._ensure_column("baseline_detections", "missing_since_utc", "TEXT")
        self._ensure_column("baseline_detections", "peak_db", "REAL")
        self._ensure_column("baseline_detections", "noise_db", "REAL")
        self._ensure_column("baseline_detections", "snr_db", "REAL")
        self._ensure_column("baseline_detections", "service", "TEXT")
        self._ensure_column("baseline_detections", "region", "TEXT")
        self._ensure_column("baseline_detections", "bandplan_notes", "TEXT")
        self._ensure_column("scan_updates", "num_revisits", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("scan_updates", "num_confirmed", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("scan_updates", "num_false_positive", "INTEGER NOT NULL DEFAULT 0")
        self._ensure_column("scan_updates", "duration_ms", "REAL")
        self._ensure_column("baselines", "bandplan_path", "TEXT")
        self._migrate_baseline_stats()

    def _ensure_column(self, table: str, column: str, definition: str) -> None:
        cur = self.con.cursor()
        cur.execute(f"PRAGMA table_info({table})")
        cols = {row[1] for row in cur.fetchall()}
        if column not in cols:
            cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
            self.con.commit()

    def _table_exists(self, table: str) -> bool:
        cur = self.con.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name = ?",
            (table,),
        )
        return cur.fetchone() is not None

    def _migrate_baseline_stats(self) -> None:
        if not self._table_exists("baseline_stats"):
            return
        cur = self.con.cursor()
        cur.execute("PRAGMA table_info(baseline_stats)")
        columns = {row[1] for row in cur.fetchall()}
        required = {"baseline_id", "bin_index", "noise_floor_ema", "power_ema", "occ_count", "last_seen_utc"}
        if not required.issubset(columns):
            # Unknown legacy layout; keep table for manual handling.
            return
        count_row = cur.execute("SELECT COUNT(*) FROM baseline_stats").fetchone()
        total = int(count_row[0]) if count_row and count_row[0] is not None else 0
        if total == 0:
            cur.execute("DROP TABLE baseline_stats")
            self.con.commit()
            return
        self.begin()
        try:
            select_cur = self.con.cursor()
            select_cur.execute(
                """
                SELECT baseline_id, bin_index, noise_floor_ema, power_ema, occ_count, last_seen_utc
                FROM baseline_stats
                """
            )
            insert_cur = self.con.cursor()
            for row in select_cur:
                baseline_id, bin_index, noise_floor, power, occ_count, last_seen = row
                ts_val = str(last_seen) if last_seen is not None else utc_now_str()
                noise_val = float(noise_floor) if noise_floor is not None else 0.0
                power_val = float(power) if power is not None else 0.0
                occ_val = int(occ_count or 0)
                insert_cur.execute(
                    """
                    INSERT INTO baseline_noise(baseline_id, bin_index, noise_floor_ema, power_ema, last_seen_utc)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(baseline_id, bin_index)
                    DO UPDATE SET
                        noise_floor_ema = excluded.noise_floor_ema,
                        power_ema = excluded.power_ema,
                        last_seen_utc = excluded.last_seen_utc
                    """,
                    (int(baseline_id), int(bin_index), noise_val, power_val, ts_val),
                )
                insert_cur.execute(
                    """
                    INSERT INTO baseline_occupancy(baseline_id, bin_index, occ_count, last_seen_utc)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(baseline_id, bin_index)
                    DO UPDATE SET
                        occ_count = excluded.occ_count,
                        last_seen_utc = excluded.last_seen_utc
                    """,
                    (int(baseline_id), int(bin_index), occ_val, ts_val),
                )
            insert_cur.execute("DROP TABLE baseline_stats")
            self.commit()
        except Exception:
            self.rollback()
            raise

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

    def set_baseline_bin(self, baseline_id: int, bin_hz: float) -> None:
        """Persist a corrected bin_hz for legacy baselines."""

        if bin_hz <= 0.0:
            return
        self.con.execute(
            "UPDATE baselines SET bin_hz = ? WHERE id = ?",
            (float(bin_hz), int(baseline_id)),
        )
        self.con.commit()

    def set_baseline_bandplan(self, baseline_id: int, bandplan_path: str) -> None:
        """Store the bandplan file path used for this baseline."""

        if not bandplan_path:
            return
        self.con.execute(
            "UPDATE baselines SET bandplan_path = ? WHERE id = ?",
            (str(bandplan_path), int(baseline_id)),
        )
        self.con.commit()

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
            noise_row = cur.execute(
                """
                SELECT noise_floor_ema, power_ema FROM baseline_noise
                WHERE baseline_id = ? AND bin_index = ?
                """,
                (int(baseline_id), idx_int),
            ).fetchone()
            noise_val = float(noise_db)
            power_val = float(power_db_val)
            if noise_row:
                prev_noise = float(noise_row[0]) if noise_row[0] is not None else noise_val
                prev_power = float(noise_row[1]) if noise_row[1] is not None else power_val
                noise_val = (1.0 - ema_alpha) * prev_noise + ema_alpha * noise_val
                power_val = (1.0 - ema_alpha) * prev_power + ema_alpha * power_val
                cur.execute(
                    """
                    UPDATE baseline_noise
                    SET noise_floor_ema = ?, power_ema = ?, last_seen_utc = ?
                    WHERE baseline_id = ? AND bin_index = ?
                    """,
                    (noise_val, power_val, timestamp_utc, int(baseline_id), idx_int),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO baseline_noise(
                        baseline_id, bin_index, noise_floor_ema, power_ema, last_seen_utc
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (int(baseline_id), idx_int, noise_val, power_val, timestamp_utc),
                )

            occ_increment = 1 if bool(occupied) else 0
            occ_row = cur.execute(
                """
                SELECT occ_count FROM baseline_occupancy
                WHERE baseline_id = ? AND bin_index = ?
                """,
                (int(baseline_id), idx_int),
            ).fetchone()
            if occ_row:
                occ_count = int(occ_row[0] or 0) + occ_increment
                cur.execute(
                    """
                    UPDATE baseline_occupancy
                    SET occ_count = ?, last_seen_utc = ?
                    WHERE baseline_id = ? AND bin_index = ?
                    """,
                    (occ_count, timestamp_utc, int(baseline_id), idx_int),
                )
            else:
                cur.execute(
                    """
                    INSERT INTO baseline_occupancy(
                        baseline_id, bin_index, occ_count, last_seen_utc
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (int(baseline_id), idx_int, occ_increment, timestamp_utc),
                )
        self.con.commit()

    def load_baseline_detections(self, baseline_id: int) -> List["PersistentDetection"]:
        cur = self.con.cursor()
        cur.execute(
            """
            SELECT id, baseline_id, f_low_hz, f_high_hz, f_center_hz,
                   first_seen_utc, last_seen_utc, total_hits, total_windows,
                   confidence, missing_since_utc, peak_db, noise_db, snr_db,
                   service, region, bandplan_notes
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
                    peak_db=float(row[11]) if row[11] is not None else None,
                    noise_db=float(row[12]) if row[12] is not None else None,
                    snr_db=float(row[13]) if row[13] is not None else None,
                    service=row[14],
                    region=row[15],
                    bandplan_notes=row[16],
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
        peak_db: Optional[float] = None,
        noise_db: Optional[float] = None,
        snr_db: Optional[float] = None,
        service: Optional[str] = None,
        region: Optional[str] = None,
        bandplan_notes: Optional[str] = None,
    ) -> int:
        cur = self.con.cursor()
        cur.execute(
            """
            INSERT INTO baseline_detections(
                baseline_id, f_low_hz, f_high_hz, f_center_hz,
                first_seen_utc, last_seen_utc, total_hits, total_windows, confidence,
                missing_since_utc, peak_db, noise_db, snr_db, service, region, bandplan_notes
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                peak_db,
                noise_db,
                snr_db,
                service,
                region,
                bandplan_notes,
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
                missing_since_utc = ?, peak_db = ?, noise_db = ?, snr_db = ?,
                service = ?, region = ?, bandplan_notes = ?
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
                detection.peak_db,
                detection.noise_db,
                detection.snr_db,
                detection.service,
                detection.region,
                detection.bandplan_notes,
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
        duration_ms: Optional[float] = None,
    ) -> None:
        self.con.execute(
            """
            INSERT INTO scan_updates(
                baseline_id, timestamp_utc,
                num_hits, num_segments, num_new_signals,
                num_revisits, num_confirmed, num_false_positive,
                duration_ms
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                float(duration_ms) if duration_ms is not None else None,
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
            "SELECT occ_count FROM baseline_occupancy WHERE baseline_id = ? AND bin_index = ?",
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

    def refresh_baseline_snapshot(
        self,
        baseline_ctx: BaselineContext,
        *,
        last_update_utc: str,
        recent_minutes: Optional[int] = None,
    ) -> None:
        baseline_id = int(baseline_ctx.id)
        recent_window = max(1, int(recent_minutes or tactical_recent_minutes()))
        cutoff_dt = datetime.now(timezone.utc) - timedelta(minutes=recent_window)
        cutoff_ts = cutoff_dt.isoformat()
        cur = self.con.cursor()
        det_row = cur.execute(
            """
            SELECT COUNT(*) AS cnt, MAX(last_seen_utc) AS last_seen
            FROM baseline_detections
            WHERE baseline_id = ?
            """,
            (baseline_id,),
        ).fetchone()
        persistent = int(det_row[0] or 0) if det_row else 0
        last_detection = det_row[1] if det_row else None
        recent_row = cur.execute(
            """
            SELECT COALESCE(SUM(num_new_signals), 0) AS new_sum
            FROM scan_updates
            WHERE baseline_id = ? AND timestamp_utc >= ?
            """,
            (baseline_id, cutoff_ts),
        ).fetchone()
        recent_new = int(recent_row[0] or 0) if recent_row else 0
        snapshot_ts = utc_now_str()
        cur.execute(
            """
            INSERT INTO baseline_snapshot(
                baseline_id,
                total_windows,
                persistent_detections,
                last_detection_utc,
                last_update_utc,
                recent_new_signals,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(baseline_id)
            DO UPDATE SET
                total_windows = excluded.total_windows,
                persistent_detections = excluded.persistent_detections,
                last_detection_utc = excluded.last_detection_utc,
                last_update_utc = excluded.last_update_utc,
                recent_new_signals = excluded.recent_new_signals,
                updated_at = excluded.updated_at
            """,
            (
                baseline_id,
                int(baseline_ctx.total_windows or 0),
                persistent,
                last_detection,
                last_update_utc,
                recent_new,
                snapshot_ts,
            ),
        )
        self.con.commit()

    def refresh_band_summary(
        self,
        baseline_ctx: BaselineContext,
        *,
        config: Optional[BandSummaryConfig] = None,
    ) -> None:
        cfg = config or BandSummaryConfig.from_env()
        freq_start = float(baseline_ctx.freq_start_hz)
        freq_stop = float(baseline_ctx.freq_stop_hz)
        bin_hz = float(baseline_ctx.bin_hz)
        baseline_id = int(baseline_ctx.id)
        if freq_stop <= freq_start or bin_hz <= 0.0:
            self.begin()
            try:
                cur = self.con.cursor()
                cur.execute("DELETE FROM baseline_band_summary WHERE baseline_id = ?", (baseline_id,))
                cur.execute("DELETE FROM baseline_summary_meta WHERE baseline_id = ?", (baseline_id,))
                self.commit()
            except Exception:
                self.rollback()
                raise
            return

        partitions, band_width = band_partitions(freq_start, freq_stop, cfg)
        if not partitions or band_width <= 0:
            self.begin()
            try:
                cur = self.con.cursor()
                cur.execute("DELETE FROM baseline_band_summary WHERE baseline_id = ?", (baseline_id,))
                cur.execute("DELETE FROM baseline_summary_meta WHERE baseline_id = ?", (baseline_id,))
                self.commit()
            except Exception:
                self.rollback()
                raise
            return

        band_count = len(partitions)
        persistent = [0 for _ in range(band_count)]
        recent_new = [0 for _ in range(band_count)]
        noise_sums = [0.0 for _ in range(band_count)]
        noise_counts = [0 for _ in range(band_count)]
        power_sums = [0.0 for _ in range(band_count)]
        power_counts = [0 for _ in range(band_count)]
        occupied_bins = [0 for _ in range(band_count)]
        bin_counts = [0 for _ in range(band_count)]
        registered_bins: set[Tuple[int, int]] = set()

        cutoff_dt = datetime.now(timezone.utc) - timedelta(minutes=max(1, int(cfg.recent_minutes or 1)))
        total_windows = max(0, int(baseline_ctx.total_windows or 0))
        occ_ratio = max(0.0, min(1.0, cfg.occ_threshold_ratio))
        occ_threshold = max(1, int(math.ceil(total_windows * occ_ratio))) if total_windows > 0 else 0

        def band_index_for_freq(freq_hz: float) -> Optional[int]:
            if freq_hz < freq_start or freq_hz > freq_stop:
                return None
            rel = (freq_hz - freq_start) / band_width if band_width > 0 else 0.0
            idx_val = int(math.floor(rel))
            if idx_val < 0:
                return None
            if idx_val >= band_count:
                idx_val = band_count - 1
            return idx_val

        def parse_timestamp(text: Optional[str]) -> Optional[datetime]:
            if text in (None, ""):
                return None
            cleaned = str(text).strip()
            if not cleaned:
                return None
            if cleaned.endswith("Z"):
                cleaned = cleaned[:-1] + "+00:00"
            try:
                parsed = datetime.fromisoformat(cleaned)
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)

        cur = self.con.cursor()
        det_cur = self.con.cursor()
        det_cur.execute(
            """
            SELECT f_center_hz, first_seen_utc
            FROM baseline_detections
            WHERE baseline_id = ? AND f_center_hz BETWEEN ? AND ?
            """,
            (baseline_id, int(freq_start), int(freq_stop)),
        )
        for center_hz, first_seen in det_cur:
            idx = band_index_for_freq(float(center_hz))
            if idx is None:
                continue
            persistent[idx] += 1
            ts_val = parse_timestamp(first_seen)
            if ts_val is not None and ts_val >= cutoff_dt:
                recent_new[idx] += 1

        noise_cur = self.con.cursor()
        noise_cur.execute(
            """
            SELECT bin_index, noise_floor_ema, power_ema
            FROM baseline_noise
            WHERE baseline_id = ?
            """,
            (baseline_id,),
        )
        origin = float(baseline_ctx.freq_start_hz)
        for bin_index, noise_val, power_val in noise_cur:
            freq = origin + float(bin_index) * bin_hz
            idx = band_index_for_freq(freq)
            if idx is None:
                continue
            bin_key = (idx, int(bin_index))
            if bin_key not in registered_bins:
                registered_bins.add(bin_key)
                bin_counts[idx] += 1
            if noise_val is not None:
                noise_sums[idx] += float(noise_val)
                noise_counts[idx] += 1
            if power_val is not None:
                power_sums[idx] += float(power_val)
                power_counts[idx] += 1

        occ_cur = self.con.cursor()
        occ_cur.execute(
            """
            SELECT bin_index, occ_count
            FROM baseline_occupancy
            WHERE baseline_id = ?
            """,
            (baseline_id,),
        )
        for bin_index, occ_count in occ_cur:
            freq = origin + float(bin_index) * bin_hz
            idx = band_index_for_freq(freq)
            if idx is None:
                continue
            bin_key = (idx, int(bin_index))
            if bin_key not in registered_bins:
                registered_bins.add(bin_key)
                bin_counts[idx] += 1
            if occ_threshold > 0 and int(occ_count or 0) >= occ_threshold:
                occupied_bins[idx] += 1

        summary_ts = utc_now_str()
        rows: List[Tuple[int, int, int, int, int, int, float, Optional[float], Optional[float], str]] = []
        for idx, (low, high) in enumerate(partitions):
            bin_count = max(1, bin_counts[idx])
            occ_fraction = 0.0
            if occupied_bins[idx] > 0:
                occ_fraction = min(1.0, max(0.0, occupied_bins[idx] / float(bin_count)))
            avg_noise = (noise_sums[idx] / noise_counts[idx]) if noise_counts[idx] else None
            avg_power = (power_sums[idx] / power_counts[idx]) if power_counts[idx] else None
            rows.append(
                (
                    baseline_id,
                    idx,
                    int(low),
                    int(high),
                    int(persistent[idx]),
                    int(recent_new[idx]),
                    occ_fraction,
                    avg_noise,
                    avg_power,
                    summary_ts,
                )
            )

        self.begin()
        try:
            cur.execute("DELETE FROM baseline_band_summary WHERE baseline_id = ?", (baseline_id,))
            cur.executemany(
                """
                INSERT INTO baseline_band_summary(
                    baseline_id,
                    band_index,
                    f_low_hz,
                    f_high_hz,
                    persistent_signals,
                    recent_new_signals,
                    occupied_fraction,
                    avg_noise_db,
                    avg_power_db,
                    last_updated_utc
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            cur.execute(
                """
                INSERT INTO baseline_summary_meta(
                    baseline_id,
                    band_count,
                    band_width_hz,
                    recent_minutes,
                    occ_threshold,
                    updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(baseline_id)
                DO UPDATE SET
                    band_count = excluded.band_count,
                    band_width_hz = excluded.band_width_hz,
                    recent_minutes = excluded.recent_minutes,
                    occ_threshold = excluded.occ_threshold,
                    updated_at = excluded.updated_at
                """,
                (
                    baseline_id,
                    band_count,
                    float(band_width),
                    int(cfg.recent_minutes),
                    occ_ratio,
                    summary_ts,
                ),
            )
            self.commit()
        except Exception:
            self.rollback()
            raise

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
