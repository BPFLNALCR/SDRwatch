"""
Baseline payload helpers for SDRwatch Web.

Provides functions for building baseline-related API payloads:
- Baseline list with summaries
- Tactical snapshots
- Hotspot heatmaps
- Change events (NEW, QUIETED, POWER_SHIFT)
- Band summaries
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from flask import current_app

from sdrwatch_web.config import (
    ACTIVE_SIGNAL_WINDOW_MINUTES,
    BAND_SUMMARY_MAX_BANDS,
    BAND_SUMMARY_OCC_THRESHOLD,
    BAND_SUMMARY_RECENT_MINUTES,
    BAND_SUMMARY_TARGET_WIDTH_HZ,
    CHANGE_EVENT_LIMIT,
    CHANGE_WINDOW_MINUTES,
    HOTSPOT_BUCKET_COUNT,
    NEW_SIGNAL_WINDOW_MINUTES,
    POWER_SHIFT_THRESHOLD_DB,
    QUIETED_MIN_WINDOWS,
    QUIETED_TIMEOUT_MINUTES,
    TACTICAL_RECENT_MINUTES,
)
from sdrwatch_web.controller import get_controller
from sdrwatch_web.db import get_con, get_con_optional, q1, qa, table_columns, table_exists
from sdrwatch_web.formatting import (
    compute_display_bandwidth_hz,
    format_bandwidth_khz,
    format_freq_label,
    isoformat_utc,
    now_utc,
    parse_ts_utc,
)


# ---------------------------------------------------------------------------
# Baseline stats span map (for controller baselines enrichment)
# ---------------------------------------------------------------------------


def baseline_stats_span_map() -> Dict[int, Dict[str, Optional[float]]]:
    """
    Build a map of baseline_id -> {min_hz, max_hz} from stats tables.

    Returns:
        Dict mapping baseline IDs to frequency span info.
    """
    spans: Dict[int, Dict[str, Optional[float]]] = {}
    connection = get_con_optional()
    if connection is None:
        return spans

    try:
        if table_exists("baseline_noise"):
            idx_rows = qa(
                connection,
                """
                SELECT baseline_id,
                       MIN(bin_index) AS min_idx,
                       MAX(bin_index) AS max_idx
                FROM baseline_noise
                WHERE bin_index IS NOT NULL
                GROUP BY baseline_id
                """,
            )
            if idx_rows:
                meta_rows = qa(connection, "SELECT id, freq_start_hz, bin_hz FROM baselines")
                meta: Dict[int, Tuple[float, float]] = {}
                for meta_row in meta_rows:
                    raw_id = meta_row.get("id") if isinstance(meta_row, dict) else None
                    if raw_id is None:
                        continue
                    try:
                        bid = int(raw_id)
                    except Exception:
                        continue
                    freq_start = float(meta_row.get("freq_start_hz") or 0.0)
                    bin_hz = float(meta_row.get("bin_hz") or 0.0)
                    meta[bid] = (freq_start, bin_hz)

                for row in idx_rows:
                    try:
                        bid = int(row.get("baseline_id"))
                    except Exception:
                        continue
                    info = meta.get(bid)
                    if not info:
                        continue
                    freq_start, bin_hz = info
                    if bin_hz <= 0 or freq_start <= 0:
                        continue
                    min_idx = row.get("min_idx")
                    max_idx = row.get("max_idx")
                    if min_idx is None or max_idx is None:
                        continue
                    try:
                        min_freq = freq_start + int(min_idx) * bin_hz
                        max_freq = freq_start + int(max_idx) * bin_hz
                    except Exception:
                        continue
                    spans[bid] = {"min_hz": float(min_freq), "max_hz": float(max_freq)}

        elif table_exists("baseline_stats"):
            stats_columns = table_columns("baseline_stats")
            if "freq_hz" in stats_columns:
                rows = qa(
                    connection,
                    """
                    SELECT baseline_id,
                           MIN(freq_hz) AS min_hz,
                           MAX(freq_hz) AS max_hz
                    FROM baseline_stats
                    WHERE freq_hz IS NOT NULL
                    GROUP BY baseline_id
                    """,
                )
                for row in rows:
                    try:
                        bid = int(row.get("baseline_id"))
                    except Exception:
                        continue
                    spans[bid] = {
                        "min_hz": float(row.get("min_hz")) if row.get("min_hz") is not None else None,
                        "max_hz": float(row.get("max_hz")) if row.get("max_hz") is not None else None,
                    }
            elif "bin_index" in stats_columns:
                idx_rows = qa(
                    connection,
                    """
                    SELECT baseline_id,
                           MIN(bin_index) AS min_idx,
                           MAX(bin_index) AS max_idx
                    FROM baseline_stats
                    WHERE bin_index IS NOT NULL
                    GROUP BY baseline_id
                    """,
                )
                if idx_rows:
                    meta_rows = qa(connection, "SELECT id, freq_start_hz, bin_hz FROM baselines")
                    meta = {}
                    for meta_row in meta_rows:
                        raw_id = meta_row.get("id")
                        if raw_id is None:
                            continue
                        try:
                            bid = int(raw_id)
                        except Exception:
                            continue
                        freq_start = float(meta_row.get("freq_start_hz") or 0.0)
                        bin_hz = float(meta_row.get("bin_hz") or 0.0)
                        meta[bid] = (freq_start, bin_hz)

                    for row in idx_rows:
                        try:
                            bid = int(row.get("baseline_id"))
                        except Exception:
                            continue
                        info = meta.get(bid)
                        if not info:
                            continue
                        freq_start, bin_hz = info
                        if bin_hz <= 0 or freq_start <= 0:
                            continue
                        min_idx = row.get("min_idx")
                        max_idx = row.get("max_idx")
                        if min_idx is None or max_idx is None:
                            continue
                        try:
                            min_freq = freq_start + int(min_idx) * bin_hz
                            max_freq = freq_start + int(max_idx) * bin_hz
                        except Exception:
                            continue
                        spans[bid] = {"min_hz": float(min_freq), "max_hz": float(max_freq)}

    except sqlite3.OperationalError:
        return spans

    return spans


def apply_span_metadata(
    records: Iterable[Dict[str, Any]],
    span_map: Dict[int, Dict[str, Optional[float]]],
) -> None:
    """
    Add stats_min_hz and stats_max_hz to records from span_map.

    Modifies records in place.
    """
    if not records or not span_map:
        return
    for record in records:
        raw_bid = record.get("id")
        if raw_bid is None:
            raw_bid = record.get("baseline_id")
        if raw_bid is None:
            continue
        try:
            bid_int = int(raw_bid)
        except Exception:
            try:
                bid_int = int(float(raw_bid))
            except Exception:
                continue
        span = span_map.get(bid_int)
        if not span:
            continue
        record["stats_min_hz"] = span.get("min_hz")
        record["stats_max_hz"] = span.get("max_hz")


def controller_baselines() -> List[Dict[str, Any]]:
    """
    Fetch baselines from controller and enrich with span metadata.

    Returns:
        List of baseline dicts with stats_min_hz/stats_max_hz added.
    """
    ctl = get_controller()
    try:
        data = ctl.baselines()
    except Exception as exc:
        current_app.logger.warning("controller /baselines fetch failed: %s", exc)
        return []

    if isinstance(data, list):
        span_map = baseline_stats_span_map()
        apply_span_metadata(data, span_map)
        return data

    current_app.logger.warning("controller /baselines unexpected payload: %r", data)
    return []


# ---------------------------------------------------------------------------
# Baseline summary map
# ---------------------------------------------------------------------------


def baseline_summary_map() -> Dict[int, Dict[str, Any]]:
    """
    Build a map of baseline_id -> summary stats.

    Returns:
        Dict mapping baseline IDs to summary dicts.
    """
    connection = get_con_optional()
    if connection is None:
        return {}

    summaries: Dict[int, Dict[str, Any]] = {}

    def ensure_entry(baseline_id: int) -> Dict[str, Any]:
        return summaries.setdefault(
            baseline_id,
            {
                "baseline_id": baseline_id,
                "persistent_detections": 0,
                "last_detection_utc": None,
                "last_update_utc": None,
                "total_windows": 0,
                "recent_new_signals": 0,
            },
        )

    try:
        rows = qa(connection, "SELECT id AS baseline_id, total_windows FROM baselines")
        for row in rows:
            raw_id = row.get("baseline_id")
            if raw_id is None:
                continue
            try:
                bid = int(raw_id)
            except Exception:
                continue
            entry = ensure_entry(bid)
            entry["total_windows"] = int(row.get("total_windows") or 0)
    except sqlite3.OperationalError:
        return summaries

    missing_ids: Set[int] = set(summaries.keys())

    if table_exists("baseline_snapshot"):
        try:
            snapshot_rows = qa(
                connection,
                """
                SELECT baseline_id, total_windows, persistent_detections,
                       last_detection_utc, last_update_utc, recent_new_signals
                FROM baseline_snapshot
                """,
            )
        except sqlite3.OperationalError:
            snapshot_rows = []

        for row in snapshot_rows:
            raw_id = row.get("baseline_id")
            if raw_id is None:
                continue
            try:
                bid = int(raw_id)
            except Exception:
                continue
            entry = ensure_entry(bid)
            entry["total_windows"] = int(row.get("total_windows") or entry.get("total_windows") or 0)
            entry["persistent_detections"] = int(row.get("persistent_detections") or 0)
            entry["last_detection_utc"] = row.get("last_detection_utc")
            entry["last_update_utc"] = row.get("last_update_utc")
            entry["recent_new_signals"] = row.get("recent_new_signals")
            if bid in missing_ids:
                missing_ids.remove(bid)

    def _run_det_query(target_ids: Set[int]) -> None:
        if not target_ids or not table_exists("baseline_detections"):
            return
        placeholders = ",".join("?" for _ in target_ids)
        try:
            det_rows = qa(
                connection,
                f"""
                SELECT baseline_id, COUNT(*) AS detection_count, MAX(last_seen_utc) AS last_detection_utc
                FROM baseline_detections
                WHERE baseline_id IN ({placeholders})
                GROUP BY baseline_id
                """,
                tuple(target_ids),
            )
        except sqlite3.OperationalError:
            det_rows = []
        for row in det_rows:
            raw_id = row.get("baseline_id")
            if raw_id is None:
                continue
            try:
                bid = int(raw_id)
            except Exception:
                continue
            entry = ensure_entry(bid)
            entry["persistent_detections"] = int(row.get("detection_count") or 0)
            entry["last_detection_utc"] = row.get("last_detection_utc")

    def _run_update_query(target_ids: Set[int]) -> None:
        if not target_ids or not table_exists("scan_updates"):
            return
        placeholders = ",".join("?" for _ in target_ids)
        try:
            update_rows = qa(
                connection,
                f"""
                SELECT baseline_id, MAX(timestamp_utc) AS last_update_utc
                FROM scan_updates
                WHERE baseline_id IN ({placeholders})
                GROUP BY baseline_id
                """,
                tuple(target_ids),
            )
        except sqlite3.OperationalError:
            update_rows = []
        for row in update_rows:
            raw_id = row.get("baseline_id")
            if raw_id is None:
                continue
            try:
                bid = int(raw_id)
            except Exception:
                continue
            entry = ensure_entry(bid)
            entry["last_update_utc"] = row.get("last_update_utc")

    if missing_ids:
        ids_copy = set(missing_ids)
        _run_det_query(ids_copy)
        _run_update_query(ids_copy)

    return summaries


# ---------------------------------------------------------------------------
# Baseline record fetch
# ---------------------------------------------------------------------------


def fetch_baseline_record(baseline_id: int) -> Optional[Dict[str, Any]]:
    """
    Fetch a single baseline record by ID.

    Args:
        baseline_id: Baseline ID.

    Returns:
        Baseline dict or None if not found.
    """
    connection = get_con_optional()
    if connection is None:
        return None

    try:
        return q1(
            connection,
            """
            SELECT id, name, created_at, location_lat, location_lon,
                   sdr_serial, antenna, notes, freq_start_hz, freq_stop_hz,
                   bin_hz, total_windows
            FROM baselines
            WHERE id = ?
            """,
            (int(baseline_id),),
        )
    except sqlite3.OperationalError:
        return None


# ---------------------------------------------------------------------------
# Band summary helpers
# ---------------------------------------------------------------------------


def _band_partitions(freq_start_hz: float, freq_stop_hz: float) -> Tuple[List[Tuple[float, float]], float]:
    """Calculate band partition boundaries."""
    span = freq_stop_hz - freq_start_hz
    if span <= 0:
        return [], 0.0
    target_width = BAND_SUMMARY_TARGET_WIDTH_HZ if BAND_SUMMARY_TARGET_WIDTH_HZ > 0 else span
    target_width = max(1_000_000.0, target_width)
    approx_count = max(1, int(math.ceil(span / target_width)))
    band_count = max(1, min(BAND_SUMMARY_MAX_BANDS, approx_count))
    band_width = span / band_count if band_count else span
    partitions: List[Tuple[float, float]] = []
    for idx in range(band_count):
        low = freq_start_hz + idx * band_width
        high = freq_start_hz + (idx + 1) * band_width if idx < band_count - 1 else freq_stop_hz
        partitions.append((low, high))
    return partitions, band_width


def _band_label(f_low_hz: float, f_high_hz: float) -> str:
    """Generate a human-readable band label."""
    span_mhz = max(0.0, (f_high_hz - f_low_hz) / 1e6)
    decimals = 0 if span_mhz >= 10 else 1
    return f"{f_low_hz / 1e6:.{decimals}f}–{f_high_hz / 1e6:.{decimals}f} MHz"


def _band_occupancy_level(fraction: float) -> str:
    """Classify occupancy fraction as High/Medium/Low."""
    if fraction >= 0.6:
        return "High"
    if fraction >= 0.3:
        return "Medium"
    return "Low"


def _band_summary_note(persistent: int, recent: int, fraction: float) -> str:
    """Generate a summary note for a band."""
    if recent >= 2 or (recent >= 1 and fraction >= 0.2):
        return "Active / changing"
    if persistent >= 2 and recent == 0:
        return "Stable"
    if persistent == 0 and recent == 0 and fraction < 0.1:
        return "Quiet"
    if recent == 0 and persistent > 0:
        return "Stable"
    return "Active / changing"


def band_summary_for_baseline(baseline_row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build band summary for a baseline.

    Args:
        baseline_row: Baseline record dict.

    Returns:
        Dict with "bands" list and "meta" dict.
    """
    freq_start = float(baseline_row.get("freq_start_hz") or 0.0)
    freq_stop = float(baseline_row.get("freq_stop_hz") or 0.0)
    raw_id = baseline_row.get("id") or baseline_row.get("baseline_id")
    baseline_id = None
    try:
        if raw_id is not None:
            baseline_id = int(float(str(raw_id)))
    except Exception:
        baseline_id = None

    meta: Dict[str, Any] = {
        "band_count": 0,
        "band_width_mhz": None,
        "recent_minutes": BAND_SUMMARY_RECENT_MINUTES,
    }

    if baseline_id is None or freq_stop <= freq_start:
        return {"bands": [], "meta": meta}

    connection = get_con_optional()
    if connection is None:
        return {"bands": [], "meta": meta}

    # Try to load from stored band summary
    if table_exists("baseline_band_summary"):
        try:
            stored_rows = qa(
                connection,
                """
                SELECT band_index, f_low_hz, f_high_hz,
                       persistent_signals, recent_new_signals,
                       occupied_fraction, avg_noise_db, avg_power_db
                FROM baseline_band_summary
                WHERE baseline_id = ?
                ORDER BY band_index
                """,
                (baseline_id,),
            )
        except sqlite3.OperationalError:
            stored_rows = []

        if stored_rows:
            summary_meta = None
            if table_exists("baseline_summary_meta"):
                try:
                    summary_meta = q1(
                        connection,
                        "SELECT band_count, band_width_hz, recent_minutes FROM baseline_summary_meta WHERE baseline_id = ?",
                        (baseline_id,),
                    )
                except sqlite3.OperationalError:
                    summary_meta = None

            bands: List[Dict[str, Any]] = []
            for row in stored_rows:
                low = float(row.get("f_low_hz") or 0.0)
                high = float(row.get("f_high_hz") or 0.0)
                occ_fraction = float(row.get("occupied_fraction") or 0.0)
                idx_val = row.get("band_index")
                try:
                    idx = int(idx_val)
                except Exception:
                    idx = len(bands)
                band = {
                    "band_index": idx,
                    "label": _band_label(low, high),
                    "f_low_hz": low,
                    "f_high_hz": high,
                    "persistent_signals": int(row.get("persistent_signals") or 0),
                    "recent_new": int(row.get("recent_new_signals") or 0),
                    "avg_noise_db": row.get("avg_noise_db"),
                    "avg_power_db": row.get("avg_power_db"),
                    "occupied_fraction": max(0.0, min(1.0, occ_fraction)),
                }
                band["occupancy_level"] = _band_occupancy_level(band["occupied_fraction"])
                band["note"] = _band_summary_note(
                    band["persistent_signals"], band["recent_new"], band["occupied_fraction"]
                )
                bands.append(band)

            if summary_meta:
                meta["band_count"] = int(summary_meta.get("band_count") or len(bands))
                bw = summary_meta.get("band_width_hz")
                meta["band_width_mhz"] = (float(bw) / 1e6) if bw else None
                meta["recent_minutes"] = int(summary_meta.get("recent_minutes") or BAND_SUMMARY_RECENT_MINUTES)
            else:
                meta["band_count"] = len(bands)
                if bands:
                    meta["band_width_mhz"] = max(0.0, (bands[0]["f_high_hz"] - bands[0]["f_low_hz"]) / 1e6)

            return {"bands": bands, "meta": meta}

    # Fall back to computing on-the-fly
    partitions, band_width = _band_partitions(freq_start, freq_stop)
    if not partitions or band_width <= 0:
        return {"bands": [], "meta": meta}

    meta["band_count"] = len(partitions)
    meta["band_width_mhz"] = band_width / 1e6

    bands = []
    for idx, (low, high) in enumerate(partitions):
        bands.append({
            "band_index": idx,
            "label": _band_label(low, high),
            "f_low_hz": low,
            "f_high_hz": high,
            "persistent_signals": 0,
            "recent_new": 0,
            "avg_noise_db": None,
            "avg_power_db": None,
            "occupied_fraction": 0.0,
            "occupancy_level": "Low",
            "note": "Quiet",
        })

    # Query detections
    band_width_param = band_width if band_width > 0 else 1.0
    recent_clause = f"-{max(1, BAND_SUMMARY_RECENT_MINUTES)} minutes"

    if table_exists("baseline_detections"):
        try:
            detection_rows = qa(
                connection,
                """
                SELECT
                    CAST((f_center_hz - :start_hz) / :band_width AS INTEGER) AS band_idx,
                    COUNT(*) AS persistent_count,
                    SUM(CASE WHEN first_seen_utc >= datetime('now', :recent_clause) THEN 1 ELSE 0 END) AS recent_new
                FROM baseline_detections
                WHERE baseline_id = :baseline_id
                  AND f_center_hz BETWEEN :start_hz AND :stop_hz
                GROUP BY band_idx
                """,
                {
                    "baseline_id": baseline_id,
                    "start_hz": freq_start,
                    "stop_hz": freq_stop,
                    "band_width": band_width_param,
                    "recent_clause": recent_clause,
                },
            )
        except sqlite3.OperationalError:
            detection_rows = []

        for row in detection_rows:
            idx_val = row.get("band_idx")
            if idx_val is None:
                continue
            try:
                idx = int(idx_val)
            except Exception:
                continue
            if 0 <= idx < len(bands):
                bands[idx]["persistent_signals"] = int(row.get("persistent_count") or 0)
                bands[idx]["recent_new"] = int(row.get("recent_new") or 0)

    # Recompute derived fields
    for band in bands:
        band["occupancy_level"] = _band_occupancy_level(band["occupied_fraction"])
        band["note"] = _band_summary_note(band["persistent_signals"], band["recent_new"], band["occupied_fraction"])

    return {"bands": bands, "meta": meta}


# ---------------------------------------------------------------------------
# Tactical snapshot payload
# ---------------------------------------------------------------------------


def tactical_snapshot_payload(baseline_id: int) -> Optional[Dict[str, Any]]:
    """
    Build tactical snapshot payload for a baseline.

    Args:
        baseline_id: Baseline ID.

    Returns:
        Dict with baseline, snapshot, active_signals, and band_summary.
    """
    baseline_row = fetch_baseline_record(baseline_id)
    if not baseline_row:
        return None

    con = get_con()

    def _safe_count(sql: str, params: Tuple[Any, ...], default: int = 0) -> int:
        try:
            row = q1(con, sql, params)
        except sqlite3.OperationalError:
            return default
        if not row:
            return default
        try:
            return int(row.get("c") or row.get("count") or row.get("sum") or default)
        except Exception:
            return default

    snapshot_row = None
    last_update: Optional[str] = None
    recent_new = 0

    if table_exists("baseline_snapshot"):
        try:
            snapshot_row = q1(
                con,
                """
                SELECT persistent_detections, last_detection_utc, last_update_utc, recent_new_signals
                FROM baseline_snapshot
                WHERE baseline_id = ?
                """,
                (baseline_id,),
            )
        except sqlite3.OperationalError:
            snapshot_row = None

    if snapshot_row:
        persistent_signals = int(snapshot_row.get("persistent_detections") or 0)
        last_update = snapshot_row.get("last_update_utc")
        recent_new = int(snapshot_row.get("recent_new_signals") or 0)
    else:
        persistent_signals = _safe_count(
            "SELECT COUNT(*) AS c FROM baseline_detections WHERE baseline_id = ?",
            (baseline_id,),
        )
        try:
            last_update_row = q1(
                con,
                "SELECT MAX(timestamp_utc) AS last_ts FROM scan_updates WHERE baseline_id = ?",
                (baseline_id,),
            )
            last_update = last_update_row.get("last_ts") if last_update_row else None
        except sqlite3.OperationalError:
            last_update = None

        recent_window_clause = f"-{TACTICAL_RECENT_MINUTES} minutes"
        try:
            recent_row = q1(
                con,
                "SELECT COALESCE(SUM(num_new_signals), 0) AS total FROM scan_updates WHERE baseline_id = ? AND timestamp_utc >= datetime('now', ?)",
                (baseline_id, recent_window_clause),
            )
            recent_new = int(recent_row.get("total") or 0) if recent_row else 0
        except sqlite3.OperationalError:
            recent_new = 0

    try:
        latest_update_row = q1(
            con,
            "SELECT id, timestamp_utc, num_hits, num_new_signals FROM scan_updates WHERE baseline_id = ? ORDER BY timestamp_utc DESC LIMIT 1",
            (baseline_id,),
        )
    except sqlite3.OperationalError:
        latest_update_row = None

    try:
        active_rows = qa(
            con,
            """
            SELECT id,
                f_center_hz,
                f_low_hz,
                f_high_hz,
                (f_high_hz - f_low_hz) AS bandwidth_hz,
                confidence,
                last_seen_utc,
                first_seen_utc,
                total_hits,
                total_windows
            FROM baseline_detections
            WHERE baseline_id = ?
              AND last_seen_utc >= datetime('now', ?)
            ORDER BY last_seen_utc DESC
            LIMIT 50
            """,
            (baseline_id, f"-{ACTIVE_SIGNAL_WINDOW_MINUTES} minutes"),
        )
    except sqlite3.OperationalError:
        active_rows = []

    active_payload: List[Dict[str, Any]] = []
    for row in active_rows:
        try:
            center = float(row.get("f_center_hz"))
        except Exception:
            center = None
        try:
            f_low = float(row.get("f_low_hz"))
        except Exception:
            f_low = None
        try:
            f_high = float(row.get("f_high_hz"))
        except Exception:
            f_high = None

        bandwidth = row.get("bandwidth_hz")
        if bandwidth is None and f_low is not None and f_high is not None:
            bandwidth = max(0.0, f_high - f_low)
        if bandwidth is not None:
            try:
                bandwidth = max(0.0, float(bandwidth))
            except Exception:
                bandwidth = None

        display_bandwidth = compute_display_bandwidth_hz(
            baseline_row=baseline_row,
            f_low_hz=f_low,
            f_high_hz=f_high,
            bandwidth_hz=bandwidth,
        )

        active_payload.append({
            "id": row.get("id"),
            "f_center_hz": center,
            "bandwidth_hz": bandwidth,
            "bandwidth_hz_display": display_bandwidth,
            "confidence": row.get("confidence"),
            "last_seen_utc": row.get("last_seen_utc"),
            "first_seen_utc": row.get("first_seen_utc"),
            "total_hits": row.get("total_hits"),
            "total_windows": row.get("total_windows"),
        })

    snapshot = {
        "persistent_signals": persistent_signals,
        "last_update": last_update,
        "recent_new": recent_new,
        "recent_window_minutes": TACTICAL_RECENT_MINUTES,
        "active_window_minutes": ACTIVE_SIGNAL_WINDOW_MINUTES,
        "latest_update": latest_update_row,
    }

    band_summary = band_summary_for_baseline(baseline_row)

    return {
        "baseline": baseline_row,
        "snapshot": snapshot,
        "active_signals": active_payload,
        "band_summary": band_summary.get("bands", []),
        "band_summary_meta": band_summary.get("meta", {}),
    }


# ---------------------------------------------------------------------------
# Hotspots payload
# ---------------------------------------------------------------------------


def hotspots_payload(baseline_id: int) -> Optional[Dict[str, Any]]:
    """
    Build hotspots heatmap payload for a baseline.

    Args:
        baseline_id: Baseline ID.

    Returns:
        Dict with frequency buckets and occupancy/power data.
    """
    baseline_row = fetch_baseline_record(baseline_id)
    if not baseline_row:
        return None

    try:
        freq_start = float(baseline_row.get("freq_start_hz") or 0.0)
        freq_stop = float(baseline_row.get("freq_stop_hz") or 0.0)
    except Exception:
        return None

    if not (freq_stop > freq_start):
        return {
            "baseline_id": baseline_id,
            "freq_start_hz": freq_start,
            "freq_stop_hz": freq_stop,
            "bucket_width_hz": 0,
            "buckets": [],
            "max_occ": 0,
            "max_power": 0,
            "has_samples": False,
        }

    con = get_con()
    bucket_count = max(10, HOTSPOT_BUCKET_COUNT)
    bucket_width = (freq_stop - freq_start) / float(bucket_count)
    total_windows = int(baseline_row.get("total_windows") or 0)

    bin_hz = None
    try:
        raw_bin = baseline_row.get("bin_hz")
        if raw_bin is not None:
            bin_hz = float(raw_bin)
    except Exception:
        bin_hz = None

    stats_rows: List[Dict[str, Any]] = []
    if table_exists("baseline_noise"):
        try:
            stats_rows = qa(
                con,
                """
                SELECT bn.bin_index AS bin_index,
                       NULL AS freq_hz,
                       bn.noise_floor_ema AS noise_floor_ema,
                       bn.power_ema AS power_ema,
                       COALESCE(bo.occ_count, 0) AS occ_count
                FROM baseline_noise AS bn
                LEFT JOIN baseline_occupancy AS bo
                  ON bn.baseline_id = bo.baseline_id AND bn.bin_index = bo.bin_index
                WHERE bn.baseline_id = ?
                """,
                (baseline_id,),
            )
        except sqlite3.OperationalError:
            stats_rows = []
    elif table_exists("baseline_stats"):
        try:
            stats_rows = qa(
                con,
                """
                SELECT bin_index, freq_hz, noise_floor_ema, power_ema, occ_count
                FROM baseline_stats
                WHERE baseline_id = ?
                """,
                (baseline_id,),
            )
        except sqlite3.OperationalError:
            stats_rows = []

    buckets: List[Dict[str, Any]] = []
    for idx in range(bucket_count):
        start_hz = freq_start + idx * bucket_width
        buckets.append({
            "f_low_hz": start_hz,
            "f_high_hz": start_hz + bucket_width,
            "occ_sum": 0.0,
            "occ_samples": 0,
            "power_sum": 0.0,
            "power_samples": 0,
        })

    def _row_freq(row: Dict[str, Any]) -> Optional[float]:
        val = row.get("freq_hz")
        if val is not None:
            try:
                return float(val)
            except Exception:
                return None
        idx_val = row.get("bin_index")
        if idx_val is None or bin_hz in (None, 0):
            return None
        try:
            idx_numeric = float(idx_val)
        except Exception:
            return None
        return freq_start + idx_numeric * float(bin_hz)

    for row in stats_rows:
        freq_val = _row_freq(row)
        if freq_val is None or freq_val < freq_start or freq_val >= freq_stop:
            continue
        bucket_idx = int(min(bucket_count - 1, max(0, math.floor((freq_val - freq_start) / bucket_width))))
        bucket = buckets[bucket_idx]

        occ_count = row.get("occ_count")
        if occ_count is not None:
            try:
                occ_val = float(occ_count)
            except Exception:
                occ_val = 0.0
            occupancy = occ_val / total_windows if total_windows > 0 else occ_val
            bucket["occ_sum"] += occupancy
            bucket["occ_samples"] += 1

        power = row.get("power_ema")
        if power is not None:
            try:
                bucket["power_sum"] += float(power)
                bucket["power_samples"] += 1
            except Exception:
                pass

    max_occ = 0.0
    max_power = None
    samples_present = False

    for bucket in buckets:
        occ_avg = bucket["occ_sum"] / bucket["occ_samples"] if bucket["occ_samples"] else 0.0
        power_avg = bucket["power_sum"] / bucket["power_samples"] if bucket["power_samples"] else None
        if bucket["occ_samples"] or bucket["power_samples"]:
            samples_present = True
        bucket["avg_occ"] = occ_avg
        bucket["avg_power_ema"] = power_avg
        bucket.pop("occ_sum", None)
        bucket.pop("occ_samples", None)
        bucket.pop("power_sum", None)
        bucket.pop("power_samples", None)
        max_occ = max(max_occ, occ_avg or 0.0)
        if power_avg is not None:
            max_power = power_avg if max_power is None else max(max_power, power_avg)

    return {
        "baseline_id": baseline_id,
        "freq_start_hz": freq_start,
        "freq_stop_hz": freq_stop,
        "bucket_width_hz": bucket_width,
        "buckets": buckets,
        "max_occ": max_occ,
        "max_power": max_power or 0.0,
        "has_samples": samples_present,
    }


# ---------------------------------------------------------------------------
# Change events payload
# ---------------------------------------------------------------------------


def change_events_payload(
    baseline_id: int,
    *,
    window_minutes: Optional[int] = None,
    event_types: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    """
    Build change events payload for a baseline.

    Args:
        baseline_id: Baseline ID.
        window_minutes: Lookback window for events.
        event_types: Filter to specific event types (NEW_SIGNAL, QUIETED, POWER_SHIFT).

    Returns:
        Dict with events list and counts.
    """
    baseline_row = fetch_baseline_record(baseline_id)
    if not baseline_row:
        return None

    connection = get_con_optional()
    if connection is None:
        return None

    window_value = window_minutes if window_minutes else CHANGE_WINDOW_MINUTES
    new_window_value = NEW_SIGNAL_WINDOW_MINUTES
    quiet_timeout_value = QUIETED_TIMEOUT_MINUTES

    requested_types: Set[str] = set()
    if event_types:
        for t in event_types:
            requested_types.add(str(t).upper().strip())

    base_cutoff_dt = now_utc() - timedelta(minutes=max(1, window_value))
    base_cutoff = isoformat_utc(base_cutoff_dt)

    events: List[Dict[str, Any]] = []

    def append_event(ev: Dict[str, Any]) -> None:
        ev_type = str(ev.get("type", "")).upper()
        if requested_types and ev_type not in requested_types:
            return
        ts_val = parse_ts_utc(ev.get("time_utc"))
        ev["_sort_ts"] = ts_val or base_cutoff_dt
        events.append(ev)

    freq_start = float(baseline_row.get("freq_start_hz") or 0.0)
    freq_stop = float(baseline_row.get("freq_stop_hz") or 0.0)
    bin_hz = float(baseline_row.get("bin_hz") or 0.0)
    total_windows = int(baseline_row.get("total_windows") or 0)

    # NEW_SIGNAL events
    if not requested_types or "NEW_SIGNAL" in requested_types:
        new_cutoff_dt = now_utc() - timedelta(minutes=max(1, new_window_value))
        new_cutoff = isoformat_utc(new_cutoff_dt)
        if table_exists("baseline_detections"):
            try:
                new_rows = qa(
                    connection,
                    """
                    SELECT id, f_center_hz, f_low_hz, f_high_hz, first_seen_utc,
                           last_seen_utc, confidence, total_hits, total_windows
                    FROM baseline_detections
                    WHERE baseline_id = ?
                      AND first_seen_utc >= ?
                    ORDER BY first_seen_utc DESC
                    LIMIT ?
                    """,
                    (baseline_id, new_cutoff, CHANGE_EVENT_LIMIT),
                )
            except sqlite3.OperationalError:
                new_rows = []

            for row in new_rows:
                fc = row.get("f_center_hz")
                f_low = row.get("f_low_hz")
                f_high = row.get("f_high_hz")
                bw = None
                if f_low is not None and f_high is not None:
                    try:
                        bw = max(0.0, float(f_high) - float(f_low))
                    except Exception:
                        bw = None

                display_bw = compute_display_bandwidth_hz(
                    baseline_row=baseline_row,
                    f_low_hz=float(f_low) if f_low else None,
                    f_high_hz=float(f_high) if f_high else None,
                    bandwidth_hz=bw,
                )

                event = {
                    "type": "NEW_SIGNAL",
                    "time_utc": row.get("first_seen_utc"),
                    "f_center_hz": fc,
                    "bandwidth_hz": bw,
                    "bandwidth_hz_display": display_bw,
                    "confidence": row.get("confidence"),
                    "details": f"First seen at {row.get('first_seen_utc')}",
                }
                append_event(event)

    # QUIETED events
    if not requested_types or "QUIETED" in requested_types:
        quiet_cutoff_dt = now_utc() - timedelta(minutes=max(1, quiet_timeout_value))
        quiet_cutoff = isoformat_utc(quiet_cutoff_dt)
        if table_exists("baseline_detections"):
            try:
                quiet_rows = qa(
                    connection,
                    """
                    SELECT id, f_center_hz, f_low_hz, f_high_hz, last_seen_utc,
                           first_seen_utc, confidence, total_windows
                    FROM baseline_detections
                    WHERE baseline_id = ?
                      AND last_seen_utc < ?
                      AND total_windows >= ?
                    ORDER BY last_seen_utc DESC
                    LIMIT ?
                    """,
                    (baseline_id, quiet_cutoff, QUIETED_MIN_WINDOWS, CHANGE_EVENT_LIMIT),
                )
            except sqlite3.OperationalError:
                quiet_rows = []

            for row in quiet_rows:
                fc = row.get("f_center_hz")
                f_low = row.get("f_low_hz")
                f_high = row.get("f_high_hz")
                bw = None
                if f_low is not None and f_high is not None:
                    try:
                        bw = max(0.0, float(f_high) - float(f_low))
                    except Exception:
                        bw = None

                display_bw = compute_display_bandwidth_hz(
                    baseline_row=baseline_row,
                    f_low_hz=float(f_low) if f_low else None,
                    f_high_hz=float(f_high) if f_high else None,
                    bandwidth_hz=bw,
                )

                event = {
                    "type": "QUIETED",
                    "time_utc": row.get("last_seen_utc"),
                    "f_center_hz": fc,
                    "bandwidth_hz": bw,
                    "bandwidth_hz_display": display_bw,
                    "confidence": row.get("confidence"),
                    "last_seen_utc": row.get("last_seen_utc"),
                    "total_windows": row.get("total_windows"),
                    "details": f"Last seen {row.get('last_seen_utc')}, was persistent ({row.get('total_windows')} windows)",
                }
                append_event(event)

    # POWER_SHIFT events
    if not requested_types or "POWER_SHIFT" in requested_types:
        if table_exists("baseline_noise") and bin_hz > 0:
            try:
                power_rows = qa(
                    connection,
                    """
                    SELECT bn.bin_index, bn.noise_floor_ema, bn.power_ema, bn.last_seen_utc
                    FROM baseline_noise AS bn
                    WHERE bn.baseline_id = ?
                    """,
                    (baseline_id,),
                )
            except sqlite3.OperationalError:
                power_rows = []

            for row in power_rows:
                idx_val = row.get("bin_index")
                if idx_val is None:
                    continue
                try:
                    freq_hz = freq_start + float(idx_val) * bin_hz
                except Exception:
                    continue

                power = row.get("power_ema")
                noise = row.get("noise_floor_ema")
                if power is None or noise is None:
                    continue
                try:
                    power_val = float(power)
                    noise_val = float(noise)
                except Exception:
                    continue

                delta_db = power_val - noise_val
                if delta_db < POWER_SHIFT_THRESHOLD_DB:
                    continue

                last_seen = row.get("last_seen_utc")
                ts_dt = parse_ts_utc(last_seen)
                if ts_dt is None or ts_dt < base_cutoff_dt:
                    continue

                display_bw = compute_display_bandwidth_hz(
                    baseline_row=baseline_row,
                    f_low_hz=freq_hz,
                    f_high_hz=freq_hz + bin_hz if bin_hz > 0 else freq_hz,
                    bandwidth_hz=bin_hz if bin_hz > 0 else None,
                )

                event = {
                    "type": "POWER_SHIFT",
                    "time_utc": last_seen,
                    "f_center_hz": freq_hz,
                    "bandwidth_hz": bin_hz if bin_hz > 0 else None,
                    "bandwidth_hz_display": display_bw,
                    "confidence": None,
                    "delta_db": delta_db,
                    "power_db": power_val,
                    "noise_floor_db": noise_val,
                    "details": f"Power EMA {power_val:.1f} dB vs noise {noise_val:.1f} dB.",
                }
                append_event(event)

    events.sort(key=lambda ev: ev.get("_sort_ts", base_cutoff_dt), reverse=True)
    events = events[:CHANGE_EVENT_LIMIT]

    counts: Dict[str, int] = {"NEW_SIGNAL": 0, "POWER_SHIFT": 0, "QUIETED": 0}
    for ev in events:
        ev_type = str(ev.get("type") or "").upper()
        counts[ev_type] = counts.get(ev_type, 0) + 1
        ev.pop("_sort_ts", None)

    active_filter = "ALL"
    if requested_types and len(requested_types) == 1:
        active_filter = next(iter(requested_types))

    return {
        "baseline": baseline_row,
        "baseline_id": baseline_id,
        "events": events,
        "counts": counts,
        "total_events": len(events),
        "window_minutes": window_value,
        "new_signal_window_minutes": new_window_value,
        "quiet_timeout_minutes": quiet_timeout_value,
        "power_shift_threshold_db": POWER_SHIFT_THRESHOLD_DB,
        "event_limit": CHANGE_EVENT_LIMIT,
        "generated_at": isoformat_utc(now_utc()),
        "active_filter": active_filter,
        "requested_types": sorted(requested_types) if requested_types else [],
    }
