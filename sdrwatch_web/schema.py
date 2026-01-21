"""
Schema DDL and baseline creation for SDRwatch Web.

Provides functions for ensuring database schema exists and creating
new baseline entries.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from flask import current_app

from sdrwatch_web.db import reset_ro_connection


# ---------------------------------------------------------------------------
# Classification columns migration
# ---------------------------------------------------------------------------


def migrate_detection_classification(conn: sqlite3.Connection) -> None:
    """
    Add classification columns to baseline_detections if missing.

    Columns added:
      - label TEXT: user-defined short label
      - classification TEXT: friendly/ambient/hostile/unknown
      - user_bw_hz INTEGER: user-corrected bandwidth (display only)
      - notes TEXT: freeform user notes
      - selected INTEGER: boolean flag for highlighting
      - missing_since_utc TEXT: timestamp when signal went quiet
      - peak_db REAL: peak power at last detection
      - noise_db REAL: noise floor at last detection
      - snr_db REAL: signal-to-noise ratio at last detection
      - service TEXT: bandplan service allocation
      - region TEXT: bandplan region
      - bandplan_notes TEXT: bandplan notes

    Args:
        conn: Writable SQLite connection.
    """
    # Check which columns already exist
    cursor = conn.execute("PRAGMA table_info(baseline_detections)")
    existing = {row[1] for row in cursor.fetchall()}

    migrations = [
        ("label", "ALTER TABLE baseline_detections ADD COLUMN label TEXT"),
        ("classification", "ALTER TABLE baseline_detections ADD COLUMN classification TEXT DEFAULT 'unknown'"),
        ("user_bw_hz", "ALTER TABLE baseline_detections ADD COLUMN user_bw_hz INTEGER"),
        ("notes", "ALTER TABLE baseline_detections ADD COLUMN notes TEXT"),
        ("selected", "ALTER TABLE baseline_detections ADD COLUMN selected INTEGER DEFAULT 0"),
        ("missing_since_utc", "ALTER TABLE baseline_detections ADD COLUMN missing_since_utc TEXT"),
        ("peak_db", "ALTER TABLE baseline_detections ADD COLUMN peak_db REAL"),
        ("noise_db", "ALTER TABLE baseline_detections ADD COLUMN noise_db REAL"),
        ("snr_db", "ALTER TABLE baseline_detections ADD COLUMN snr_db REAL"),
        ("service", "ALTER TABLE baseline_detections ADD COLUMN service TEXT"),
        ("region", "ALTER TABLE baseline_detections ADD COLUMN region TEXT"),
        ("bandplan_notes", "ALTER TABLE baseline_detections ADD COLUMN bandplan_notes TEXT"),
    ]

    for col_name, alter_stmt in migrations:
        if col_name not in existing:
            try:
                conn.execute(alter_stmt)
            except sqlite3.OperationalError:
                # Column may already exist from a partial migration
                pass


def migrate_baselines_bandplan(conn: sqlite3.Connection) -> None:
    """
    Add bandplan_path column to baselines if missing.

    Args:
        conn: Writable SQLite connection.
    """
    cursor = conn.execute("PRAGMA table_info(baselines)")
    existing = {row[1] for row in cursor.fetchall()}

    if "bandplan_path" not in existing:
        try:
            conn.execute("ALTER TABLE baselines ADD COLUMN bandplan_path TEXT")
        except sqlite3.OperationalError:
            pass


def ensure_baseline_schema(conn: sqlite3.Connection) -> None:
    """
    Ensure all baseline-related tables exist.

    Args:
        conn: Writable SQLite connection.
    """
    stmts = [
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
            total_windows INTEGER NOT NULL DEFAULT 0,
            bandplan_path TEXT
        )
        """,
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
            confidence REAL NOT NULL,
            label TEXT,
            classification TEXT DEFAULT 'unknown',
            user_bw_hz INTEGER,
            notes TEXT,
            selected INTEGER DEFAULT 0
        )
        """,
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
            num_false_positive INTEGER NOT NULL DEFAULT 0,
            duration_ms INTEGER
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS spur_map (
            bin_hz        INTEGER PRIMARY KEY,
            mean_power_db REAL,
            hits          INTEGER,
            last_seen_utc TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS baseline_noise (
            baseline_id INTEGER NOT NULL,
            bin_index INTEGER NOT NULL,
            noise_floor_ema REAL NOT NULL,
            power_ema REAL NOT NULL,
            last_seen_utc TEXT NOT NULL,
            PRIMARY KEY (baseline_id, bin_index)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS baseline_occupancy (
            baseline_id INTEGER NOT NULL,
            bin_index INTEGER NOT NULL,
            occ_count INTEGER NOT NULL,
            last_seen_utc TEXT NOT NULL,
            PRIMARY KEY (baseline_id, bin_index)
        )
        """,
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
        """,
        """
        CREATE TABLE IF NOT EXISTS baseline_summary_meta (
            baseline_id INTEGER PRIMARY KEY,
            band_count INTEGER NOT NULL,
            band_width_hz REAL NOT NULL,
            recent_minutes INTEGER NOT NULL,
            occ_threshold REAL NOT NULL,
            updated_at TEXT NOT NULL
        )
        """,
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
        """,
    ]
    for stmt in stmts:
        conn.execute(stmt)

    # Run migrations for existing tables
    migrate_detection_classification(conn)
    migrate_baselines_bandplan(conn)


def create_baseline_entry(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a new baseline entry in the database.

    Args:
        data: Dict with baseline fields (name required, others optional).

    Returns:
        The created baseline record as a dict.

    Raises:
        ValueError: If required fields are missing or invalid.
        RuntimeError: If database operations fail.
    """
    name = str(data.get("name", "")).strip()
    if not name:
        raise ValueError("name is required")

    def _coerce_float(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except Exception as exc:
            raise ValueError("Invalid coordinate value") from exc

    bin_val_raw = data.get("bin_hz")
    try:
        bin_hz = float(bin_val_raw) if bin_val_raw not in (None, "") else 0.0
    except Exception as exc:
        raise ValueError("bin_hz must be numeric if provided") from exc

    payload = {
        "name": name,
        "created_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat() + "Z",
        "location_lat": _coerce_float(data.get("location_lat")),
        "location_lon": _coerce_float(data.get("location_lon")),
        "sdr_serial": (str(data.get("sdr_serial") or "").strip() or None),
        "antenna": (str(data.get("antenna") or "").strip() or None),
        "notes": (str(data.get("notes") or "").strip() or None),
        "freq_start_hz": 0,
        "freq_stop_hz": 0,
        "bin_hz": bin_hz,
        "baseline_version": int(data.get("baseline_version") or 1),
    }

    app = current_app._get_current_object()
    db_path = app.config.get('SDRWATCH_DB_PATH') or app.config.get('DB_PATH')
    if not db_path:
        raise RuntimeError("No database path configured")

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
    except sqlite3.Error as exc:
        raise RuntimeError(f"failed to open database for baseline creation: {exc}")

    try:
        ensure_baseline_schema(conn)
        cur = conn.execute(
            """
            INSERT INTO baselines(
                name, created_at, location_lat, location_lon,
                sdr_serial, antenna, notes, freq_start_hz, freq_stop_hz,
                bin_hz, baseline_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                payload["name"],
                payload["created_at"],
                payload["location_lat"],
                payload["location_lon"],
                payload["sdr_serial"],
                payload["antenna"],
                payload["notes"],
                payload["freq_start_hz"],
                payload["freq_stop_hz"],
                payload["bin_hz"],
                payload["baseline_version"],
            ),
        )
        lastrow = cur.lastrowid
        if lastrow is None:
            raise RuntimeError("baseline insert did not return a row id")
        baseline_id = int(lastrow)
        conn.commit()
        row = conn.execute(
            """
            SELECT id, name, created_at, location_lat, location_lon,
                   sdr_serial, antenna, notes, freq_start_hz, freq_stop_hz,
                   bin_hz, total_windows
            FROM baselines WHERE id = ?
            """,
            (baseline_id,),
        ).fetchone()
    finally:
        conn.close()
        reset_ro_connection(app)
        # Re-open read-only connection
        from sdrwatch_web.db import _ensure_con
        _ensure_con(app)

    if row is None:
        raise RuntimeError("baseline created but could not be reloaded")

    result = dict(row)
    result["baseline_id"] = result.get("id")
    return result
