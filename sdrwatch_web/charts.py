"""
Chart and visualization data builders for SDRwatch Web.

Provides functions that query the database and build data structures
suitable for rendering charts in templates (SNR histograms, timelines,
frequency bins, heatmaps, etc.).
"""
from __future__ import annotations

import math
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

from sdrwatch_web.config import CHART_HEIGHT_PX
from sdrwatch_web.db import qa
from sdrwatch_web.filters import detection_predicates, scan_predicates
from sdrwatch_web.formatting import format_ts_label


def _percentile(xs: List[float], p: float) -> Optional[float]:
    """Calculate percentile value from a sorted list."""
    if not xs:
        return None
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return float(xs[f])
    return float(xs[f] + (xs[c] - xs[f]) * (k - f))


def _scale_counts_to_px(
    series: List[Dict[str, Any]],
    count_key: str = "count",
    max_height: int = CHART_HEIGHT_PX,
) -> float:
    """
    Scale count values to pixel heights for chart rendering.

    Modifies each dict in series to add a "height_px" key.

    Args:
        series: List of chart data dicts.
        count_key: Key containing the count value.
        max_height: Maximum bar height in pixels.

    Returns:
        Maximum count value in the series.
    """
    values: List[float] = []
    for x in series:
        try:
            v = float(x.get(count_key, 0) or 0)
        except Exception:
            v = 0.0
        values.append(v)

    maxc = max(values) if values else 0.0

    for i, x in enumerate(series):
        c = values[i]
        if maxc <= 0 or c <= 0:
            x["height_px"] = 0
        else:
            h = int(round((c / maxc) * max_height))
            x["height_px"] = max(2, h)

    return maxc


def snr_histogram(
    con: sqlite3.Connection,
    filters: Dict[str, Any],
    bucket_db: int = 3,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Build SNR histogram data from detections.

    Args:
        con: Database connection.
        filters: Detection filter dict.
        bucket_db: Bucket width in dB.

    Returns:
        Tuple of (histogram data list, stats dict or None).
    """
    conds, params = detection_predicates(filters, alias="d")
    conds.append("d.snr_db IS NOT NULL")
    where_sql = " WHERE " + " AND ".join(conds) if conds else ""

    rows = qa(con, f"SELECT d.snr_db FROM detections d{where_sql}", tuple(params))

    vals: List[float] = []
    for r in rows:
        try:
            vals.append(float(r['snr_db']))
        except Exception:
            pass

    buckets: Dict[int, int] = {}
    for s in vals:
        b = int(math.floor(s / bucket_db)) * bucket_db
        buckets[b] = buckets.get(b, 0) + 1

    labels_sorted = sorted(buckets.keys())
    hist = [{"label": f"{b}–{b+bucket_db}", "count": buckets[b]} for b in labels_sorted]

    _scale_counts_to_px(hist, "count")

    for h in hist:
        h["style_attr"] = f'style="height:{int(h.get("height_px", 0))}px;"'

    stats = None
    if vals:
        stats = {
            "count": len(vals),
            "p50": _percentile(vals, 0.50) or 0.0,
            "p90": _percentile(vals, 0.90) or 0.0,
            "p100": max(vals),
        }

    return hist, stats


def timeline_metrics(
    con: sqlite3.Connection,
    filters: Dict[str, Any],
    *,
    max_buckets: int = 60,
) -> Dict[str, Any]:
    """
    Build timeline chart data showing detections and scans over time.

    Args:
        con: Database connection.
        filters: Detection filter dict.
        max_buckets: Maximum number of time buckets.

    Returns:
        Dict with bucket data, maximums, and style attributes.
    """
    lookback_hours = filters.get("since_hours") or 168
    lookback_hours = max(1, min(int(lookback_hours), 24 * 180))
    bucket_hours = 1 if lookback_hours <= 72 else 24
    bucket_count = max(1, min(max_buckets, math.ceil(lookback_hours / bucket_hours)))

    conds, params = detection_predicates(filters, alias="d")
    where_sql = " WHERE " + " AND ".join(conds) if conds else ""
    params_tuple = tuple(params)

    time_expr = (
        "strftime('%Y-%m-%d %H:00:00', d.time_utc)"
        if bucket_hours == 1
        else "strftime('%Y-%m-%d 00:00:00', d.time_utc)"
    )

    det_rows = qa(
        con,
        f"""
        SELECT {time_expr} AS bucket,
               COUNT(*) AS det_count,
               MAX(d.snr_db) AS max_snr
        FROM detections d
        {where_sql}
        GROUP BY bucket
        """,
        params_tuple,
    )

    det_map = {
        r["bucket"]: {
            "count": int(r.get("det_count") or 0),
            "max_snr": (float(r.get("max_snr")) if r.get("max_snr") is not None else None),
        }
        for r in det_rows
        if r.get("bucket") is not None
    }

    scan_conds, scan_params = scan_predicates(filters, alias="s")
    scan_where = " WHERE " + " AND ".join(scan_conds) if scan_conds else ""
    scan_time_expr = (
        "strftime('%Y-%m-%d %H:00:00', COALESCE(s.t_end_utc, s.t_start_utc))"
        if bucket_hours == 1
        else "strftime('%Y-%m-%d 00:00:00', COALESCE(s.t_end_utc, s.t_start_utc))"
    )

    scan_rows = qa(
        con,
        f"""
        SELECT {scan_time_expr} AS bucket,
               COUNT(*) AS scan_count
        FROM scans s
        {scan_where}
        GROUP BY bucket
        """,
        tuple(scan_params),
    )

    scan_map = {
        r["bucket"]: int(r.get("scan_count") or 0)
        for r in scan_rows
        if r.get("bucket") is not None
    }

    now = datetime.utcnow().replace(minute=0, second=0, microsecond=0)
    if bucket_hours >= 24:
        now = now.replace(hour=0)
    start = now - timedelta(hours=bucket_hours * (bucket_count - 1))

    buckets: List[Dict[str, Any]] = []
    det_max = 0
    scan_max = 0
    snr_max = 0.0

    for i in range(bucket_count):
        bucket_start = start + timedelta(hours=bucket_hours * i)
        if bucket_hours == 1:
            key = bucket_start.strftime("%Y-%m-%d %H:00:00")
            label = bucket_start.strftime("%d %Hh")
        else:
            key = bucket_start.strftime("%Y-%m-%d 00:00:00")
            label = bucket_start.strftime("%b %d")

        det_info = det_map.get(key, {"count": 0, "max_snr": None})
        det_count = det_info["count"]
        scan_count = scan_map.get(key, 0)
        max_snr_val = det_info.get("max_snr")

        det_max = max(det_max, det_count)
        scan_max = max(scan_max, scan_count)
        if max_snr_val is not None:
            snr_max = max(snr_max, max_snr_val)

        buckets.append(
            {
                "key": key,
                "label": label,
                "detections": det_count,
                "scans": scan_count,
                "max_snr": max_snr_val,
            }
        )

    for b in buckets:
        det_count = b["detections"]
        scan_count = b["scans"]
        max_snr_val = b["max_snr"] or 0.0
        b["det_height"] = 0 if det_max == 0 else max(2, int(round((det_count / det_max) * CHART_HEIGHT_PX)))
        b["scan_height"] = 0 if scan_max == 0 else max(2, int(round((scan_count / scan_max) * (CHART_HEIGHT_PX * 0.8))))
        b["snr_height"] = 0 if snr_max <= 0 else max(2, int(round((max_snr_val / snr_max) * (CHART_HEIGHT_PX * 0.6))))
        b["det_style_attr"] = f'style="height:{b["det_height"]}px;"'
        b["scan_style_attr"] = f'style="height:{b["scan_height"]}px;"'
        b["snr_style_attr"] = f'style="height:{b["snr_height"]}px;"'

    buckets.reverse()  # Show newest buckets first (left to right)

    return {
        "bucket_hours": bucket_hours,
        "buckets": buckets,
        "det_max": det_max,
        "scan_max": scan_max,
        "snr_max": snr_max,
        "style_attr": f'style="height:{CHART_HEIGHT_PX}px;"',
    }


def frequency_bins_latest_scan(
    con: sqlite3.Connection,
    filters: Dict[str, Any],
    num_bins: int = 40,
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]], int]:
    """
    Build frequency bin data for the latest scan.

    Args:
        con: Database connection.
        filters: Detection filter dict.
        num_bins: Number of frequency bins.

    Returns:
        Tuple of (bins list, latest scan dict, max count).
    """
    conds, params = detection_predicates(filters, alias="d")
    where_sql = " WHERE " + " AND ".join(conds) if conds else ""

    latest = None
    try:
        from sdrwatch_web.db import q1
        latest = q1(
            con,
            f"""
            SELECT s.id, s.t_start_utc, s.t_end_utc, s.f_start_hz, s.f_stop_hz, s.latitude, s.longitude
            FROM detections d
            JOIN scans s ON s.id = d.scan_id
            {where_sql}
            ORDER BY d.time_utc DESC
            LIMIT 1
            """,
            tuple(params),
        )
    except Exception:
        return [], None, 0

    if not latest:
        return [], None, 0

    f0 = float(latest['f_start_hz'])
    f1 = float(latest['f_stop_hz'])
    if not (f1 > f0):
        return [], latest, 0

    conds_scan, params_scan = detection_predicates(filters, alias="d")
    conds_scan.append("d.scan_id = ?")
    params_scan.append(latest['id'])
    where_scan = " WHERE " + " AND ".join(conds_scan)

    dets = qa(con, f"SELECT d.f_center_hz FROM detections d{where_scan}", tuple(params_scan))
    if not dets:
        return [], latest, 0

    width = (f1 - f0) / max(1, num_bins)
    bins = [
        {"count": 0, "mhz_start": (f0 + i * width) / 1e6, "mhz_end": (f0 + (i + 1) * width) / 1e6}
        for i in range(num_bins)
    ]

    for r in dets:
        try:
            fc = float(r['f_center_hz'])
        except Exception:
            continue
        if fc < f0 or fc >= f1:
            continue
        idx = int((fc - f0) // width)
        idx = max(0, min(num_bins - 1, idx))
        bins[idx]["count"] += 1

    maxc = _scale_counts_to_px(bins, "count")
    for b in bins:
        b["style_attr"] = f'style="height:{int(b.get("height_px", 0))}px;"'

    return bins, latest, int(maxc)


def frequency_bins_all_scans_avg(
    con: sqlite3.Connection,
    filters: Dict[str, Any],
    num_bins: int = 40,
) -> Tuple[List[Dict[str, Any]], float, float, float]:
    """
    Build averaged frequency bin data across all matching scans.

    Args:
        con: Database connection.
        filters: Detection filter dict.
        num_bins: Number of frequency bins.

    Returns:
        Tuple of (bins list, f_start_mhz, f_stop_mhz, max_avg).
    """
    conds, params = detection_predicates(filters, alias="d")
    where_sql = " WHERE " + " AND ".join(conds) if conds else ""
    params_tuple = tuple(params)

    from sdrwatch_web.db import q1

    bounds = q1(
        con,
        f"SELECT MIN(d.f_center_hz) AS fmin, MAX(d.f_center_hz) AS fmax FROM detections d{where_sql}",
        params_tuple,
    )
    if not bounds or bounds['fmin'] is None or bounds['fmax'] is None:
        return [], 0.0, 0.0, 0.0

    f0 = float(bounds['fmin'])
    f1 = float(bounds['fmax'])
    if not (f1 > f0):
        return [], 0.0, 0.0, 0.0

    dets = qa(con, f"SELECT d.f_center_hz FROM detections d{where_sql}", params_tuple)
    scan_conds, scan_params = scan_predicates(filters, alias="s")
    scan_where = " WHERE " + " AND ".join(scan_conds) if scan_conds else ""
    scans = qa(con, f"SELECT s.f_start_hz, s.f_stop_hz FROM scans s{scan_where}", tuple(scan_params))

    width = (f1 - f0) / max(1, num_bins)
    bins: List[Dict[str, Any]] = [
        {"count": 0.0, "coverage": 0, "mhz_start": (f0 + i * width) / 1e6, "mhz_end": (f0 + (i + 1) * width) / 1e6}
        for i in range(num_bins)
    ]

    for r in dets:
        try:
            fc = float(r['f_center_hz'])
        except Exception:
            continue
        if fc < f0 or fc >= f1:
            continue
        idx = int((fc - f0) // width)
        idx = max(0, min(num_bins - 1, idx))
        bins[idx]["count"] += 1.0

    for i in range(num_bins):
        b_start = f0 + i * width
        b_end = f0 + (i + 1) * width
        cov = 0
        for s in scans:
            try:
                s0 = float(s['f_start_hz'])
                s1 = float(s['f_stop_hz'])
            except Exception:
                continue
            if (s0 < b_end) and (s1 > b_start):
                cov += 1
        bins[i]["coverage"] = cov
        bins[i]["count"] = bins[i]["count"] / float(cov) if cov > 0 else 0.0

    maxc = _scale_counts_to_px(bins, "count")
    for b in bins:
        b["style_attr"] = f'style="height:{int(b.get("height_px", 0))}px;"'

    return bins, f0 / 1e6, f1 / 1e6, maxc


def strongest_signals(
    con: sqlite3.Connection,
    filters: Dict[str, Any],
    limit: int = 10,
    *,
    include_confidence: bool = False,
) -> List[Dict[str, Any]]:
    """
    Get the strongest signals by SNR.

    Args:
        con: Database connection.
        filters: Detection filter dict.
        limit: Maximum number of results.
        include_confidence: Whether to include confidence column.

    Returns:
        List of detection dicts sorted by SNR descending.
    """
    conds, params = detection_predicates(filters, alias="d")
    conds.append("d.snr_db IS NOT NULL")
    where_sql = " WHERE " + " AND ".join(conds)
    params_tuple = tuple(params)
    confidence_expr = "d.confidence" if include_confidence else "NULL"

    return qa(
        con,
        f"""
        SELECT d.f_center_hz, d.snr_db, d.service, {confidence_expr} AS confidence
        FROM detections d
        {where_sql}
        ORDER BY d.snr_db DESC
        LIMIT ?
        """,
        params_tuple + (limit,),
    )


def top_services(
    con: sqlite3.Connection,
    filters: Dict[str, Any],
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Get top services by detection count.

    Args:
        con: Database connection.
        filters: Detection filter dict.
        limit: Maximum number of results.

    Returns:
        List of dicts with service name and count.
    """
    conds, params = detection_predicates(filters, alias="d")
    where_sql = " WHERE " + " AND ".join(conds) if conds else ""

    return qa(
        con,
        f"""
        SELECT COALESCE(d.service,'Unknown') AS service, COUNT(*) AS count
        FROM detections d
        {where_sql}
        GROUP BY COALESCE(d.service,'Unknown')
        ORDER BY count DESC
        LIMIT ?
        """,
        tuple(params) + (limit,),
    )


def coverage_heatmap(
    con: sqlite3.Connection,
    filters: Dict[str, Any],
    *,
    max_scans: int = 20,
    num_bins: int = 36,
) -> Dict[str, Any]:
    """
    Build coverage heatmap data showing detection density by scan and frequency.

    Args:
        con: Database connection.
        filters: Detection filter dict.
        max_scans: Maximum number of recent scans to include.
        num_bins: Number of frequency bins.

    Returns:
        Dict with rows, bin labels, and metadata.
    """
    conds, params = detection_predicates(filters, alias="d")
    where_sql = " WHERE " + " AND ".join(conds) if conds else ""
    params_tuple = tuple(params)

    from sdrwatch_web.db import q1

    bounds = q1(
        con,
        f"SELECT MIN(d.f_center_hz) AS fmin, MAX(d.f_center_hz) AS fmax FROM detections d{where_sql}",
        params_tuple,
    )
    if not bounds or bounds['fmin'] is None or bounds['fmax'] is None:
        return {
            "rows": [],
            "bin_labels": [],
            "f_start_mhz": None,
            "f_stop_mhz": None,
            "bin_width_mhz": None,
            "max_count": 0,
        }

    f0 = float(bounds['fmin'])
    f1 = float(bounds['fmax'])
    if not (f1 > f0):
        return {
            "rows": [],
            "bin_labels": [],
            "f_start_mhz": None,
            "f_stop_mhz": None,
            "bin_width_mhz": None,
            "max_count": 0,
        }

    scan_rows = qa(
        con,
        f"""
        SELECT s.id,
               MAX(d.time_utc) AS last_detection,
               s.t_start_utc,
               s.t_end_utc,
               s.latitude,
               s.longitude
        FROM detections d
        JOIN scans s ON s.id = d.scan_id
        {where_sql}
        GROUP BY s.id
        ORDER BY last_detection DESC
        LIMIT ?
        """,
        params_tuple + (max_scans,),
    )

    if not scan_rows:
        return {
            "rows": [],
            "bin_labels": [],
            "f_start_mhz": f0 / 1e6,
            "f_stop_mhz": f1 / 1e6,
            "bin_width_mhz": (f1 - f0) / max(1, num_bins) / 1e6,
            "max_count": 0,
        }

    scan_ids = [row['id'] for row in scan_rows if row.get('id') is not None]
    if not scan_ids:
        return {
            "rows": [],
            "bin_labels": [],
            "f_start_mhz": f0 / 1e6,
            "f_stop_mhz": f1 / 1e6,
            "bin_width_mhz": (f1 - f0) / max(1, num_bins) / 1e6,
            "max_count": 0,
        }

    conds_counts, params_counts = detection_predicates(filters, alias="d")
    placeholders = ",".join(["?"] * len(scan_ids))
    conds_counts.append(f"d.scan_id IN ({placeholders})")
    params_counts.extend(scan_ids)
    conds_counts.append("d.f_center_hz BETWEEN ? AND ?")
    params_counts.extend([int(f0), int(f1)])
    where_counts = " WHERE " + " AND ".join(conds_counts)

    det_rows = qa(
        con,
        f"SELECT d.scan_id, d.f_center_hz FROM detections d{where_counts}",
        tuple(params_counts),
    )

    bin_width = (f1 - f0) / max(1, num_bins)
    bin_labels = [f"{(f0 + i * bin_width) / 1e6:.2f}" for i in range(num_bins)]
    grid: Dict[int, List[int]] = {sid: [0 for _ in range(num_bins)] for sid in scan_ids}
    max_count = 0

    for row in det_rows:
        sid = row.get('scan_id')
        if sid not in grid:
            continue
        try:
            fc = float(row['f_center_hz'])
        except Exception:
            continue
        if fc < f0 or fc >= f1:
            continue
        idx = int((fc - f0) // bin_width)
        idx = max(0, min(num_bins - 1, idx))
        grid[sid][idx] += 1
        if grid[sid][idx] > max_count:
            max_count = grid[sid][idx]

    rows: List[Dict[str, Any]] = []
    for row in scan_rows:
        sid = row['id']
        cells_raw = grid.get(sid, [0 for _ in range(num_bins)])
        cells = []
        for count in cells_raw:
            intensity = 0.0 if max_count == 0 else count / max_count
            cells.append({
                "count": count,
                "intensity": intensity,
                "style_attr": f'style="height:18px; background: rgba(14,165,233, {intensity:.2f});"',
            })

        coords = None
        lat = row.get('latitude')
        lon = row.get('longitude')
        if lat is not None and lon is not None:
            coords = f"{float(lat):.4f}, {float(lon):.4f}"

        label = format_ts_label(row.get('last_detection') or row.get('t_end_utc') or row.get('t_start_utc'))
        rows.append({
            "scan_id": sid,
            "label": label,
            "coords": coords,
            "cells": cells,
        })

    return {
        "rows": rows,
        "bin_labels": bin_labels,
        "f_start_mhz": f0 / 1e6,
        "f_stop_mhz": f1 / 1e6,
        "bin_width_mhz": bin_width / 1e6,
        "max_count": max_count,
    }
