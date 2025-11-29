#!/usr/bin/env python3
"""
SDRwatch Web (controller-integrated) — Refactored to Flask templates

This version proxies start/stop/logs to the local sdrwatch-control.py HTTP API
and renders HTML from /templates/*.html to keep this file lean.

Run (example):
  # terminal 1 (controller)
  python sdrwatch-control.py serve --host 127.0.0.1 --port 8765 --token secret123

  # terminal 2 (web)
  SDRWATCH_CONTROL_URL=http://127.0.0.1:8765 \
  SDRWATCH_CONTROL_TOKEN=secret123 \
  python3 sdrwatch-web-simple.py --db sdrwatch.db --host 0.0.0.0 --port 8080

Auth notes:
- SDRWATCH_TOKEN       (optional) protects the web app's /api/* endpoints.
- SDRWATCH_CONTROL_TOKEN is used by the server to talk to the controller.
  If not set, we fall back to SDRWATCH_TOKEN as a convenience.
"""
from __future__ import annotations
import argparse, os, io, sqlite3, math, json
from bisect import bisect_left
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple
from flask import Flask, request, Response, render_template, render_template_string, jsonify, abort, url_for  # type: ignore
from urllib import request as urlreq, parse as urlparse, error as urlerr

# ================================
# Config
# ================================
CHART_HEIGHT_PX = 160


def _int_env(name: str, default: int) -> int:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return max(1, int(float(val)))
    except Exception:
        return default

def _float_env(name: str, default: float) -> float:
    val = os.getenv(name)
    if not val:
        return default
    try:
        return float(val)
    except Exception:
        return default


API_TOKEN = os.getenv("SDRWATCH_TOKEN", "")  # page auth (optional)
CONTROL_URL = os.getenv("SDRWATCH_CONTROL_URL", "http://127.0.0.1:8765")
CONTROL_TOKEN = os.getenv("SDRWATCH_CONTROL_TOKEN", "") or os.getenv("SDRWATCH_TOKEN", "")
BASELINE_NEW_THRESHOLD = 0.2
TACTICAL_RECENT_MINUTES = _int_env("SDRWATCH_TACTICAL_RECENT_MINUTES", 30)
ACTIVE_SIGNAL_WINDOW_MINUTES = _int_env("SDRWATCH_ACTIVE_SIGNAL_MINUTES", 15)
HOTSPOT_BUCKET_COUNT = _int_env("SDRWATCH_HOTSPOT_BUCKETS", 60)
CHANGE_WINDOW_MINUTES = _int_env("SDRWATCH_CHANGE_WINDOW_MINUTES", 60)
NEW_SIGNAL_WINDOW_MINUTES = _int_env("SDRWATCH_NEW_SIGNAL_MINUTES", CHANGE_WINDOW_MINUTES)
POWER_SHIFT_THRESHOLD_DB = _float_env("SDRWATCH_POWER_SHIFT_THRESHOLD_DB", 6.0)
QUIETED_TIMEOUT_MINUTES = _int_env("SDRWATCH_QUIETED_TIMEOUT_MINUTES", 15)
QUIETED_MIN_WINDOWS = _int_env("SDRWATCH_QUIETED_MIN_WINDOWS", 20)
CHANGE_EVENT_LIMIT = _int_env("SDRWATCH_CHANGE_EVENT_LIMIT", 120)
BAND_SUMMARY_MAX_BANDS = _int_env("SDRWATCH_BAND_SUMMARY_MAX_BANDS", 6)
BAND_SUMMARY_TARGET_WIDTH_HZ = _float_env("SDRWATCH_BAND_SUMMARY_TARGET_WIDTH_HZ", 10_000_000.0)
BAND_SUMMARY_RECENT_MINUTES = _int_env("SDRWATCH_BAND_SUMMARY_RECENT_MINUTES", 30)
BAND_SUMMARY_OCC_THRESHOLD = _float_env("SDRWATCH_BAND_SUMMARY_OCC_THRESHOLD", BASELINE_NEW_THRESHOLD)

# ================================
# DB helpers
# ================================

def open_db_ro(path: str) -> sqlite3.Connection:
    abspath = os.path.abspath(path)
    con = sqlite3.connect(f"file:{abspath}?mode=ro", uri=True, check_same_thread=False)
    con.execute("PRAGMA busy_timeout=2000;")
    con.row_factory = lambda cur, row: {d[0]: row[i] for i, d in enumerate(cur.description)}
    return con

def q1(con: sqlite3.Connection, sql: str, params: Any = ()):  # one row
    cur = con.execute(sql, params)
    return cur.fetchone()

def qa(con: sqlite3.Connection, sql: str, params: Any = ()):  # all rows
    cur = con.execute(sql, params)
    return cur.fetchall()

# ================================
# Graph helpers
# ================================

def _percentile(xs: List[float], p: float) -> Optional[float]:
    if not xs: return None
    xs = sorted(xs); k = (len(xs) - 1) * p; f = int(math.floor(k)); c = int(math.ceil(k))
    if f == c: return float(xs[f])
    return float(xs[f] + (xs[c] - xs[f]) * (k - f))

def _scale_counts_to_px(series: List[Dict[str, Any]], count_key: str = "count") -> float:
    values: List[float] = []
    for x in series:
        try: v = float(x.get(count_key, 0) or 0)
        except Exception: v = 0.0
        values.append(v)
    maxc = max(values) if values else 0.0
    for i, x in enumerate(series):
        c = values[i]
        if maxc <= 0 or c <= 0: x["height_px"] = 0
        else:
            h = int(round((c / maxc) * CHART_HEIGHT_PX))
            x["height_px"] = max(2, h)
    return maxc


def parse_detection_filters(args, *, default_since_hours: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """Parse query params into normalized detection filters and form defaults."""
    def _clean_str(key: str) -> str:
        val = args.get(key) if hasattr(args, "get") else args.get(key)
        if val is None:
            return ""
        if isinstance(val, str):
            return val.strip()
        return str(val).strip()

    def _coerce_float(text: str) -> Optional[float]:
        if text == "":
            return None
        try:
            return float(text)
        except Exception:
            return None

    filters: Dict[str, Any] = {
        "service": None,
        "min_snr": None,
        "f_min_hz": None,
        "f_max_hz": None,
        "since_hours": default_since_hours,
        "min_conf": None,
    }
    form_defaults: Dict[str, str] = {
        "service": "",
        "min_snr": "",
        "f_min_mhz": "",
        "f_max_mhz": "",
        "since_hours": "" if default_since_hours is None else str(default_since_hours),
        "min_conf": "",
    }

    service = _clean_str("service")
    if service:
        filters["service"] = service
        form_defaults["service"] = service

    min_snr_raw = _clean_str("min_snr")
    min_snr_val = _coerce_float(min_snr_raw)
    if min_snr_val is not None:
        filters["min_snr"] = min_snr_val
    if min_snr_raw:
        form_defaults["min_snr"] = min_snr_raw

    f_min_raw = _clean_str("f_min_mhz")
    f_min_val = _coerce_float(f_min_raw)
    if f_min_val is not None:
        filters["f_min_hz"] = int(f_min_val * 1e6)
    if f_min_raw:
        form_defaults["f_min_mhz"] = f_min_raw

    f_max_raw = _clean_str("f_max_mhz")
    f_max_val = _coerce_float(f_max_raw)
    if f_max_val is not None:
        filters["f_max_hz"] = int(f_max_val * 1e6)
    if f_max_raw:
        form_defaults["f_max_mhz"] = f_max_raw

    since_raw = _clean_str("since_hours")
    if since_raw:
        try:
            since_val = max(1, int(float(since_raw)))
            filters["since_hours"] = since_val
            form_defaults["since_hours"] = str(since_val)
        except Exception:
            if default_since_hours is not None:
                form_defaults["since_hours"] = str(default_since_hours)
    elif default_since_hours is None:
        filters["since_hours"] = None

    min_conf_raw = _clean_str("min_conf")
    min_conf_val = _coerce_float(min_conf_raw)
    if min_conf_val is not None:
        filters["min_conf"] = float(min_conf_val)
    if min_conf_raw:
        form_defaults["min_conf"] = min_conf_raw

    return filters, form_defaults


def detection_predicates(filters: Dict[str, Any], *, alias: str = "d") -> Tuple[List[str], List[Any]]:
    conds: List[str] = []
    params: List[Any] = []
    service = filters.get("service")
    if service:
        conds.append(f"COALESCE({alias}.service,'Unknown') = ?")
        params.append(service)
    min_snr = filters.get("min_snr")
    if min_snr is not None:
        conds.append(f"{alias}.snr_db >= ?")
        params.append(float(min_snr))
    f_min = filters.get("f_min_hz")
    if f_min is not None:
        conds.append(f"{alias}.f_center_hz >= ?")
        params.append(int(f_min))
    f_max = filters.get("f_max_hz")
    if f_max is not None:
        conds.append(f"{alias}.f_center_hz <= ?")
        params.append(int(f_max))
    since_hours = filters.get("since_hours")
    if since_hours is not None and since_hours > 0:
        conds.append(f"{alias}.time_utc >= datetime('now', ?)")
        params.append(f"-{int(since_hours)} hours")
    min_conf = filters.get("min_conf")
    confidence_available = bool(filters.get("__confidence_available"))
    if confidence_available and min_conf is not None:
        conds.append(f"{alias}.confidence >= ?")
        params.append(float(min_conf))
    return conds, params

def scan_predicates(filters: Dict[str, Any], *, alias: str = "s") -> Tuple[List[str], List[Any]]:
    conds: List[str] = []
    params: List[Any] = []
    since_hours = filters.get("since_hours")
    if since_hours is not None and since_hours > 0:
        conds.append(f"COALESCE({alias}.t_end_utc, {alias}.t_start_utc) >= datetime('now', ?)")
        params.append(f"-{int(since_hours)} hours")
    f_min = filters.get("f_min_hz")
    if f_min is not None:
        conds.append(f"{alias}.f_stop_hz >= ?")
        params.append(int(f_min))
    f_max = filters.get("f_max_hz")
    if f_max is not None:
        conds.append(f"{alias}.f_start_hz <= ?")
        params.append(int(f_max))
    return conds, params


def format_ts_label(ts: Optional[str]) -> str:
    if not ts:
        return "—"
    text = ts.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        return dt.strftime("%b %d %H:%M")
    except Exception:
        return ts

def snr_histogram(con: sqlite3.Connection, filters: Dict[str, Any], bucket_db: int = 3):
    conds, params = detection_predicates(filters, alias="d")
    conds.append("d.snr_db IS NOT NULL")
    where_sql = " WHERE " + " AND ".join(conds) if conds else ""
    rows = qa(con, f"SELECT d.snr_db FROM detections d{where_sql}", tuple(params))
    vals: List[float] = []
    for r in rows:
        try: vals.append(float(r['snr_db']))
        except Exception: pass
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
        stats = {"count": len(vals), "p50": _percentile(vals, 0.50) or 0.0, "p90": _percentile(vals, 0.90) or 0.0, "p100": max(vals)}
    return hist, stats

def timeline_metrics(con: sqlite3.Connection, filters: Dict[str, Any], *, max_buckets: int = 60) -> Dict[str, Any]:
    lookback_hours = filters.get("since_hours") or 168
    lookback_hours = max(1, min(int(lookback_hours), 24 * 180))
    bucket_hours = 1 if lookback_hours <= 72 else 24
    bucket_count = max(1, min(max_buckets, math.ceil(lookback_hours / bucket_hours)))

    conds, params = detection_predicates(filters, alias="d")
    where_sql = " WHERE " + " AND ".join(conds) if conds else ""
    params_tuple = tuple(params)

    time_expr = "strftime('%Y-%m-%d %H:00:00', d.time_utc)" if bucket_hours == 1 else "strftime('%Y-%m-%d 00:00:00', d.time_utc)"
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
    scan_time_expr = "strftime('%Y-%m-%d %H:00:00', COALESCE(s.t_end_utc, s.t_start_utc))" if bucket_hours == 1 else "strftime('%Y-%m-%d 00:00:00', COALESCE(s.t_end_utc, s.t_start_utc))"
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
    scan_map = {r["bucket"]: int(r.get("scan_count") or 0) for r in scan_rows if r.get("bucket") is not None}

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

def frequency_bins_latest_scan(con: sqlite3.Connection, filters: Dict[str, Any], num_bins: int = 40):
    conds, params = detection_predicates(filters, alias="d")
    where_sql = " WHERE " + " AND ".join(conds) if conds else ""
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
    if not latest:
        return [], None, 0
    f0 = float(latest['f_start_hz']); f1 = float(latest['f_stop_hz'])
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
    bins = [{"count":0, "mhz_start": (f0 + i*width)/1e6, "mhz_end": (f0 + (i+1)*width)/1e6} for i in range(num_bins)]
    for r in dets:
        try: fc = float(r['f_center_hz'])
        except Exception: continue
        if fc < f0 or fc >= f1: continue
        idx = int((fc - f0) // width); idx = max(0, min(num_bins-1, idx)); bins[idx]["count"] += 1
    maxc = _scale_counts_to_px(bins, "count")
    for b in bins:
        b["style_attr"] = f'style="height:{int(b.get("height_px", 0))}px;"'
    return bins, latest, int(maxc)

def frequency_bins_all_scans_avg(con: sqlite3.Connection, filters: Dict[str, Any], num_bins: int = 40):
    conds, params = detection_predicates(filters, alias="d")
    where_sql = " WHERE " + " AND ".join(conds) if conds else ""
    params_tuple = tuple(params)
    bounds = q1(con, f"SELECT MIN(d.f_center_hz) AS fmin, MAX(d.f_center_hz) AS fmax FROM detections d{where_sql}", params_tuple)
    if not bounds or bounds['fmin'] is None or bounds['fmax'] is None:
        return [], 0.0, 0.0, 0.0
    f0 = float(bounds['fmin']); f1 = float(bounds['fmax'])
    if not (f1 > f0):
        return [], 0.0, 0.0, 0.0

    dets = qa(con, f"SELECT d.f_center_hz FROM detections d{where_sql}", params_tuple)
    scan_conds, scan_params = scan_predicates(filters, alias="s")
    scan_where = " WHERE " + " AND ".join(scan_conds) if scan_conds else ""
    scans = qa(con, f"SELECT s.f_start_hz, s.f_stop_hz FROM scans s{scan_where}", tuple(scan_params))

    width = (f1 - f0) / max(1, num_bins)
    bins: List[Dict[str, Any]] = [{"count":0.0, "coverage":0, "mhz_start": (f0 + i*width)/1e6, "mhz_end": (f0 + (i+1)*width)/1e6} for i in range(num_bins)]
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

def strongest_signals(con: sqlite3.Connection, filters: Dict[str, Any], limit: int = 10, *, include_confidence: bool = False):
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


def top_services(con: sqlite3.Connection, filters: Dict[str, Any], limit: int = 10):
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
    conds, params = detection_predicates(filters, alias="d")
    where_sql = " WHERE " + " AND ".join(conds) if conds else ""
    params_tuple = tuple(params)

    bounds = q1(con, f"SELECT MIN(d.f_center_hz) AS fmin, MAX(d.f_center_hz) AS fmax FROM detections d{where_sql}", params_tuple)
    if not bounds or bounds['fmin'] is None or bounds['fmax'] is None:
        return {"rows": [], "bin_labels": [], "f_start_mhz": None, "f_stop_mhz": None, "bin_width_mhz": None, "max_count": 0}

    f0 = float(bounds['fmin'])
    f1 = float(bounds['fmax'])
    if not (f1 > f0):
        return {"rows": [], "bin_labels": [], "f_start_mhz": None, "f_stop_mhz": None, "bin_width_mhz": None, "max_count": 0}

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
        return {"rows": [], "bin_labels": [], "f_start_mhz": f0 / 1e6, "f_stop_mhz": f1 / 1e6, "bin_width_mhz": (f1 - f0) / max(1, num_bins) / 1e6, "max_count": 0}

    scan_ids = [row['id'] for row in scan_rows if row.get('id') is not None]
    if not scan_ids:
        return {"rows": [], "bin_labels": [], "f_start_mhz": f0 / 1e6, "f_stop_mhz": f1 / 1e6, "bin_width_mhz": (f1 - f0) / max(1, num_bins) / 1e6, "max_count": 0}

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

# ================================
# Controller HTTP client
# ================================
class ControllerClient:
    def __init__(self, base_url: str, token: str = ""):
        self.base = base_url.rstrip('/')
        self.token = token

    def _req(self, method: str, path: str, params: Dict[str, Any] | None = None, body: Dict[str, Any] | None = None, want_text: bool = False):
        url = self.base + path
        if params:
            q = urlparse.urlencode(params)
            url += ("?" + q)
        data = None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if body is not None:
            data = json.dumps(body).encode('utf-8')
        req = urlreq.Request(url, data, headers=headers, method=method.upper())
        try:
            with urlreq.urlopen(req, timeout=10) as resp:
                ct = resp.headers.get('Content-Type','')
                raw = resp.read()
                if want_text or not ct.startswith('application/json'):
                    return raw.decode('utf-8', errors='replace')
                return json.loads(raw.decode('utf-8'))
        except urlerr.HTTPError as e:
            raise RuntimeError(f"controller HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}")
        except Exception as e:
            raise RuntimeError(str(e))

    # API wrappers
    def devices(self):
        return self._req('GET', '/devices')

    def list_jobs(self):
        return self._req('GET', '/jobs')

    def start_job(self, device_key: str, label: str, baseline_id: int, params: Dict[str, Any]):
        body = {
            "device_key": device_key,
            "label": label,
            "baseline_id": int(baseline_id),
            "params": params,
        }
        return self._req('POST', '/jobs', body=body)

    def job_detail(self, job_id: str):
        return self._req('GET', f'/jobs/{job_id}')

    def stop_job(self, job_id: str):
        return self._req('DELETE', f'/jobs/{job_id}')

    def job_logs(self, job_id: str, tail: Optional[int] = None) -> str:
        params = {"tail": int(tail)} if tail else None
        return self._req('GET', f'/jobs/{job_id}/logs', params=params, want_text=True)

    def profiles(self):
        return self._req('GET', '/profiles')

    def baselines(self):
        return self._req('GET', '/baselines')

# ================================
# Flask app
# ================================

def create_app(db_path: str) -> Flask:
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app._db_path = db_path
    app._db_error = None
    app._con = None
    app._has_confidence_column = None
    app._table_columns_cache = {}
    app._table_exists_cache = {}
    app._ctl = ControllerClient(CONTROL_URL, CONTROL_TOKEN)

    def _ensure_con() -> Optional[sqlite3.Connection]:
        if app._con is not None:
            return app._con
        try:
            app._con = open_db_ro(app._db_path)
            app._db_error = None
            app._has_confidence_column = None
        except Exception as exc:
            app._con = None
            app._db_error = str(exc)
            app._has_confidence_column = None
        return app._con

    def reset_ro_connection() -> None:
        if app._con is not None:
            try:
                app._con.close()
            except Exception:
                pass
        app._con = None
        app._table_columns_cache = {}
        app._table_exists_cache = {}

    # Attempt initial connection (tolerates failure if DB is missing)
    _ensure_con()

    def con() -> sqlite3.Connection:
        connection = _ensure_con()
        if connection is None:
            raise RuntimeError("database connection unavailable")
        return connection

    def table_columns(table_name: str) -> Set[str]:
        cache = app._table_columns_cache
        key = table_name.lower()
        if key in cache:
            return cache[key]
        connection = _ensure_con()
        columns: Set[str] = set()
        if connection is None:
            cache[key] = columns
            return columns
        try:
            rows = qa(connection, f"PRAGMA table_info({table_name})")
        except sqlite3.OperationalError:
            cache[key] = columns
            return columns
        for row in rows:
            name = row.get("name") if isinstance(row, dict) else row[1]
            if name:
                columns.add(str(name).lower())
        cache[key] = columns
        return columns

    def table_exists(table_name: str) -> bool:
        cache = app._table_exists_cache
        key = table_name.lower()
        if key in cache:
            return cache[key]
        connection = _ensure_con()
        if connection is None:
            cache[key] = False
            return False
        try:
            row = q1(
                connection,
                "SELECT name FROM sqlite_master WHERE type='table' AND lower(name)=?",
                (key,),
            )
            exists = bool(row and row.get('name'))
        except sqlite3.OperationalError:
            exists = False
        cache[key] = exists
        return exists

    def db_state() -> Tuple[str, str]:
        connection = _ensure_con()
        if connection is None:
            return (
                "unavailable",
                app._db_error or "Database file could not be opened in read-only mode.",
            )
        try:
            cur = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
            names: Set[str] = set()
            for row in cur.fetchall():
                if isinstance(row, dict):
                    value = row.get('name', '')
                else:
                    value = row[0]
                if value:
                    names.add(str(value).lower())
            modern_ready = "baselines" in names
            legacy_required = {"scans", "detections", "baseline"}
            if modern_ready or legacy_required.issubset(names):
                return ("ready", "")
            return ("waiting", "")
        except sqlite3.OperationalError as exc:
            app._con = None
            app._db_error = str(exc)
            return (
                "waiting",
                f"Database not initialized yet ({exc}). Start a scan to populate it.",
            )

    def db_waiting_context(state: str, message: str) -> Dict[str, Any]:
        return {
            "db_status": state,
            "db_status_message": message,
            "db_path": app._db_path,
        }

    def detections_have_confidence() -> bool:
        cached = app._has_confidence_column
        if cached is not None:
            return bool(cached)
        connection = _ensure_con()
        if connection is None:
            app._has_confidence_column = False
            return False
        try:
            cur = connection.execute("PRAGMA table_info(detections)")
            rows = cur.fetchall()
        except sqlite3.OperationalError:
            app._has_confidence_column = False
            return False
        has_col = False
        for row in rows:
            name = row.get("name") if isinstance(row, dict) else row[1]
            if name == "confidence":
                has_col = True
                break
        app._has_confidence_column = has_col
        return has_col

    def require_auth():
        if not API_TOKEN: return
        hdr = request.headers.get("Authorization", "")
        if hdr != f"Bearer {API_TOKEN}":
            abort(401)

    def controller_active_job() -> Optional[Dict[str, Any]]:
        try:
            jobs = app._ctl.list_jobs()
        except Exception:
            return None
        running = [j for j in jobs if str(j.get("status", "")).lower() == "running"]
        running.sort(key=lambda j: float(j.get("created_ts") or 0.0), reverse=True)
        return running[0] if running else None

    def parse_window_log_line(line: str) -> Optional[Dict[str, Any]]:
        prefix = "[scan] window"
        if not isinstance(line, str):
            return None
        text = line.strip()
        if not text.startswith(prefix):
            return None
        payload = text[len(prefix) :].strip()
        if not payload:
            return None
        result: Dict[str, Any] = {"raw": line.rstrip("\n")}
        required_keys = {"center_hz", "det_count", "mean_db", "p90_db", "anomalous"}
        seen: Dict[str, Any] = {}
        for chunk in payload.split():
            if "=" not in chunk:
                continue
            key, value = chunk.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key == "center_hz":
                try:
                    seen[key] = float(value)
                except Exception:
                    return None
            elif key == "det_count":
                try:
                    seen[key] = int(float(value))
                except Exception:
                    return None
            elif key in {"mean_db", "p90_db"}:
                try:
                    seen[key] = float(value)
                except Exception:
                    return None
            elif key == "anomalous":
                try:
                    seen[key] = bool(int(float(value)))
                except Exception:
                    if value.lower() in {"true", "false"}:
                        seen[key] = value.lower() == "true"
                    else:
                        return None
            else:
                continue
        if not required_keys.issubset(seen.keys()):
            return None
        result.update(seen)
        return result

    def controller_profiles() -> List[Dict[str, Any]]:
        try:
            data = app._ctl.profiles()
        except Exception as exc:
            app.logger.warning("controller /profiles fetch failed: %s", exc)
            return []
        if isinstance(data, dict):
            profiles = data.get("profiles")
            if isinstance(profiles, list):
                return profiles
            app.logger.warning("controller /profiles payload missing list: %s", data)
            return []
        app.logger.warning("controller /profiles unexpected payload type: %r", type(data))
        return []

    def baseline_stats_span_map() -> Dict[int, Dict[str, Optional[float]]]:
        spans: Dict[int, Dict[str, Optional[float]]] = {}
        if not table_exists("baseline_stats"):
            return spans
        connection = _ensure_con()
        if connection is None:
            return spans
        stats_columns = table_columns("baseline_stats")
        if not stats_columns:
            return spans
        try:
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
                    meta_rows = qa(
                        connection,
                        "SELECT id, freq_start_hz, bin_hz FROM baselines",
                    )
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
                        spans[bid] = {
                            "min_hz": float(min_freq),
                            "max_hz": float(max_freq),
                        }
        except sqlite3.OperationalError:
            return spans
        return spans

    def apply_span_metadata(records: Iterable[Dict[str, Any]], span_map: Dict[int, Dict[str, Optional[float]]]) -> None:
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
        try:
            data = app._ctl.baselines()
        except Exception as exc:
            app.logger.warning("controller /baselines fetch failed: %s", exc)
            return []
        if isinstance(data, list):
            span_map = baseline_stats_span_map()
            apply_span_metadata(data, span_map)
            return data
        app.logger.warning("controller /baselines unexpected payload: %r", data)
        return []

    def baseline_summary_map() -> Dict[int, Dict[str, Any]]:
        connection = _ensure_con()
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
                },
            )

        try:
            rows = qa(
                connection,
                "SELECT id AS baseline_id, total_windows FROM baselines",
            )
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

        try:
            det_rows = qa(
                connection,
                """
                SELECT baseline_id, COUNT(*) AS detection_count, MAX(last_seen_utc) AS last_detection_utc
                FROM baseline_detections
                GROUP BY baseline_id
                """,
            )
            for row in det_rows:
                raw_id = row.get("baseline_id")
                if raw_id is None:
                    continue
                bid = int(raw_id)
                entry = ensure_entry(bid)
                entry["persistent_detections"] = int(row.get("detection_count") or 0)
                entry["last_detection_utc"] = row.get("last_detection_utc")
        except sqlite3.OperationalError:
            pass

        try:
            update_rows = qa(
                connection,
                """
                SELECT baseline_id, MAX(timestamp_utc) AS last_update_utc
                FROM scan_updates
                GROUP BY baseline_id
                """,
            )
            for row in update_rows:
                raw_id = row.get("baseline_id")
                if raw_id is None:
                    continue
                bid = int(raw_id)
                entry = ensure_entry(bid)
                entry["last_update_utc"] = row.get("last_update_utc")
        except sqlite3.OperationalError:
            pass

        return summaries

    def fetch_baseline_record(baseline_id: int) -> Optional[Dict[str, Any]]:
        connection = _ensure_con()
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

    def tactical_snapshot_payload(baseline_id: int) -> Optional[Dict[str, Any]]:
        baseline_row = fetch_baseline_record(baseline_id)
        if not baseline_row:
            return None

        def _safe_count(sql: str, params: Tuple[Any, ...], default: int = 0) -> int:
            try:
                row = q1(con(), sql, params)
            except sqlite3.OperationalError:
                return default
            if not row:
                return default
            try:
                return int(row.get("c") or row.get("count") or row.get("sum") or default)
            except Exception:
                return default

        persistent_signals = _safe_count(
            "SELECT COUNT(*) AS c FROM baseline_detections WHERE baseline_id = ?",
            (baseline_id,),
        )

        try:
            last_update_row = q1(
                con(),
                "SELECT MAX(timestamp_utc) AS last_ts FROM scan_updates WHERE baseline_id = ?",
                (baseline_id,),
            )
            last_update = last_update_row.get("last_ts") if last_update_row else None
        except sqlite3.OperationalError:
            last_update = None

        recent_window_clause = f"-{TACTICAL_RECENT_MINUTES} minutes"
        try:
            recent_row = q1(
                con(),
                "SELECT COALESCE(SUM(num_new_signals), 0) AS total FROM scan_updates WHERE baseline_id = ? AND timestamp_utc >= datetime('now', ?)",
                (baseline_id, recent_window_clause),
            )
            recent_new = int(recent_row.get("total") or 0) if recent_row else 0
        except sqlite3.OperationalError:
            recent_new = 0

        try:
            latest_update_row = q1(
                con(),
                "SELECT id, timestamp_utc, num_hits, num_new_signals FROM scan_updates WHERE baseline_id = ? ORDER BY timestamp_utc DESC LIMIT 1",
                (baseline_id,),
            )
        except sqlite3.OperationalError:
            latest_update_row = None

        try:
            active_rows = qa(
                con(),
                """
                SELECT id, f_center_hz, f_low_hz, f_high_hz, confidence,
                       last_seen_utc, first_seen_utc, total_hits, total_windows
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
            bandwidth = None
            if f_low is not None and f_high is not None:
                bandwidth = max(0.0, f_high - f_low)
            active_payload.append(
                {
                    "id": row.get("id"),
                    "f_center_hz": center,
                    "bandwidth_hz": bandwidth,
                    "confidence": row.get("confidence"),
                    "last_seen_utc": row.get("last_seen_utc"),
                    "first_seen_utc": row.get("first_seen_utc"),
                    "total_hits": row.get("total_hits"),
                    "total_windows": row.get("total_windows"),
                }
            )

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

    def hotspots_payload(baseline_id: int) -> Optional[Dict[str, Any]]:
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

        try:
            stats_rows = qa(
                con(),
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
            buckets.append(
                {
                    "f_low_hz": start_hz,
                    "f_high_hz": start_hz + bucket_width,
                    "occ_sum": 0.0,
                    "occ_samples": 0,
                    "power_sum": 0.0,
                    "power_samples": 0,
                }
            )

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

    def _band_partitions(freq_start_hz: float, freq_stop_hz: float) -> Tuple[List[Tuple[float, float]], float]:
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
        span_mhz = max(0.0, (f_high_hz - f_low_hz) / 1e6)
        decimals = 0 if span_mhz >= 10 else 1
        return f"{f_low_hz / 1e6:.{decimals}f}–{f_high_hz / 1e6:.{decimals}f} MHz"

    def _band_occupancy_level(fraction: float) -> str:
        if fraction >= 0.6:
            return "High"
        if fraction >= 0.3:
            return "Medium"
        return "Low"

    def _band_summary_note(persistent: int, recent: int, fraction: float) -> str:
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
        partitions, band_width = _band_partitions(freq_start, freq_stop)
        if not partitions or band_width <= 0:
            return {"bands": [], "meta": meta}
        meta["band_count"] = len(partitions)
        meta["band_width_mhz"] = band_width / 1e6

        bands: List[Dict[str, Any]] = []
        for idx, (low, high) in enumerate(partitions):
            bands.append(
                {
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
                }
            )

        band_width_param = band_width if band_width > 0 else 1.0
        recent_clause = f"-{max(1, BAND_SUMMARY_RECENT_MINUTES)} minutes"

        detection_rows: List[Dict[str, Any]] = []
        if table_exists("baseline_detections"):
            try:
                detection_rows = qa(
                    con(),
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
        detection_map: Dict[int, Dict[str, Any]] = {}
        for row in detection_rows:
            idx_val = row.get("band_idx")
            if idx_val is None:
                continue
            try:
                idx = int(idx_val)
            except Exception:
                continue
            if idx < 0 or idx >= len(bands):
                continue
            detection_map[idx] = row

        stats_rows: List[Dict[str, Any]] = []
        stats_columns = table_columns("baseline_stats") if table_exists("baseline_stats") else set()
        bin_hz = float(baseline_row.get("bin_hz") or 0.0)
        total_windows = int(baseline_row.get("total_windows") or 0)
        occ_threshold = 1
        if total_windows > 0:
            occ_threshold = max(1, int(math.ceil(total_windows * max(0.0, BAND_SUMMARY_OCC_THRESHOLD))))
        freq_expr = None
        where_clause = ""
        stats_params: Dict[str, Any] = {
            "baseline_id": baseline_id,
            "start_hz": freq_start,
            "stop_hz": freq_stop,
            "band_width": band_width_param,
            "occ_threshold": occ_threshold,
            "freq_start": freq_start,
            "bin_hz": bin_hz,
        }
        if "freq_hz" in stats_columns:
            freq_expr = "freq_hz"
            where_clause = "AND freq_hz BETWEEN :start_hz AND :stop_hz"
        elif "bin_index" in stats_columns and bin_hz > 0:
            freq_expr = "(:freq_start + bin_index * :bin_hz)"
        if freq_expr:
            try:
                stats_rows = qa(
                    con(),
                    f"""
                    SELECT
                        CAST(({freq_expr} - :start_hz) / :band_width AS INTEGER) AS band_idx,
                        COUNT(*) AS bin_count,
                        AVG(noise_floor_ema) AS avg_noise_db,
                        AVG(power_ema) AS avg_power_db,
                        SUM(CASE WHEN occ_count >= :occ_threshold THEN 1 ELSE 0 END) AS occupied_bins
                    FROM baseline_stats
                    WHERE baseline_id = :baseline_id
                    {where_clause}
                    GROUP BY band_idx
                    """,
                    stats_params,
                )
            except sqlite3.OperationalError:
                stats_rows = []
        stats_map: Dict[int, Dict[str, Any]] = {}
        for row in stats_rows:
            idx_val = row.get("band_idx")
            if idx_val is None:
                continue
            try:
                idx = int(idx_val)
            except Exception:
                continue
            if idx < 0 or idx >= len(bands):
                continue
            stats_map[idx] = row

        for band in bands:
            idx = band["band_index"]
            det = detection_map.get(idx, {})
            stats = stats_map.get(idx, {})
            try:
                band["persistent_signals"] = int(det.get("persistent_count", 0) or 0)
            except Exception:
                band["persistent_signals"] = 0
            try:
                band["recent_new"] = int(det.get("recent_new", 0) or 0)
            except Exception:
                band["recent_new"] = 0
            try:
                avg_noise = stats.get("avg_noise_db")
                band["avg_noise_db"] = float(avg_noise) if avg_noise is not None else None
            except Exception:
                band["avg_noise_db"] = None
            try:
                avg_power = stats.get("avg_power_db")
                band["avg_power_db"] = float(avg_power) if avg_power is not None else None
            except Exception:
                band["avg_power_db"] = None
            try:
                bin_count = int(stats.get("bin_count", 0) or 0)
                occupied_bins = int(stats.get("occupied_bins", 0) or 0)
            except Exception:
                bin_count = 0
                occupied_bins = 0
            fraction = (occupied_bins / bin_count) if bin_count > 0 else 0.0
            band["occupied_fraction"] = max(0.0, min(1.0, fraction))
            band["occupancy_level"] = _band_occupancy_level(band["occupied_fraction"])
            band["note"] = _band_summary_note(band["persistent_signals"], band["recent_new"], band["occupied_fraction"])

        return {"bands": bands, "meta": meta}

    def _now_utc() -> datetime:
        return datetime.utcnow().replace(microsecond=0)

    def _isoformat_utc(dt_val: datetime) -> str:
        return dt_val.replace(microsecond=0).isoformat() + "Z"

    def parse_ts_utc(text: Optional[str]) -> Optional[datetime]:
        if text is None:
            return None
        candidate = text.strip()
        if not candidate:
            return None
        if candidate.endswith("Z"):
            candidate = candidate[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(candidate)
        except ValueError:
            return None
        if parsed.tzinfo:
            return parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed

    def format_change_time_label(ts: Optional[str]) -> str:
        dt_val = parse_ts_utc(ts)
        if not dt_val:
            return ts or "—"
        now = _now_utc()
        if dt_val.date() == now.date():
            return dt_val.strftime("%H:%M:%SZ")
        return dt_val.strftime("%b %d %H:%MZ")

    def format_freq_label(value: Any) -> str:
        if value in (None, ""):
            return "—"
        try:
            return f"{float(value) / 1e6:.3f}"
        except Exception:
            return "—"

    def format_bandwidth_khz(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            width = float(value)
        except Exception:
            return None
        if width <= 0:
            return None
        return width / 1e3

    def format_confidence_value(value: Any) -> Optional[float]:
        if value in (None, ""):
            return None
        try:
            val = float(value)
        except Exception:
            return None
        if val < 0:
            return None
        return val

    def format_change_summary(event: Dict[str, Any]) -> str:
        etype = str(event.get("type") or "").upper()
        freq_label = format_freq_label(event.get("f_center_hz"))
        freq_display = f"{freq_label} MHz" if freq_label != "—" else "—"
        time_label = format_change_time_label(event.get("time_utc"))
        if etype == "NEW_SIGNAL":
            bw = format_bandwidth_khz(event.get("bandwidth_hz"))
            bw_label = f" (+{bw:.0f} kHz)" if bw is not None else ""
            conf = format_confidence_value(event.get("confidence"))
            conf_label = f"{conf:.2f}" if conf is not None else "—"
            return f"NEW_SIGNAL @ {freq_display}{bw_label} – confidence {conf_label} – first seen {time_label}"
        if etype == "POWER_SHIFT":
            bw = format_bandwidth_khz(event.get("bandwidth_hz"))
            bin_label = f" ({bw:.0f} kHz bin)" if bw is not None else ""
            try:
                delta_label = f"{float(event.get('delta_db', 0.0)):+.1f} dB"
            except Exception:
                delta_label = "shift"
            return f"POWER_SHIFT @ {freq_display}{bin_label} – {delta_label} vs baseline – observed {time_label}"
        if etype == "QUIETED":
            service = event.get("service")
            service_label = f" ({service})" if service else ""
            return f"QUIETED @ {freq_display}{service_label} – last seen {time_label} – was persistent"
        return f"{etype or 'EVENT'} @ {freq_display} – observed {time_label}"

    def change_events_payload(
        baseline_id: int,
        *,
        window_minutes: Optional[int] = None,
        event_types: Optional[Iterable[str]] = None,
    ) -> Optional[Dict[str, Any]]:
        baseline_row = fetch_baseline_record(baseline_id)
        if not baseline_row:
            return None
        connection = _ensure_con()
        if connection is None:
            return None

        window_value = int(window_minutes or CHANGE_WINDOW_MINUTES)
        window_value = max(5, min(window_value, 24 * 60))
        now_ts = _now_utc()
        base_cutoff_dt = now_ts - timedelta(minutes=window_value)
        base_cutoff_iso = _isoformat_utc(base_cutoff_dt)

        new_window_value = max(1, min(window_value, NEW_SIGNAL_WINDOW_MINUTES))
        new_cutoff_iso = _isoformat_utc(now_ts - timedelta(minutes=new_window_value))
        quiet_timeout_value = max(1, QUIETED_TIMEOUT_MINUTES)
        quiet_cutoff_dt = now_ts - timedelta(minutes=quiet_timeout_value)
        quiet_cutoff_iso = _isoformat_utc(quiet_cutoff_dt)

        def normalize_types(tokens: Optional[Iterable[str]]) -> Optional[Set[str]]:
            if not tokens:
                return None
            mapped: Set[str] = set()
            for token in tokens:
                if not token:
                    continue
                text = str(token).strip().upper()
                if text in {"", "ALL"}:
                    continue
                if text in {"NEW", "NEW_SIGNAL"}:
                    mapped.add("NEW_SIGNAL")
                elif text in {"POWER", "POWER_SHIFT"}:
                    mapped.add("POWER_SHIFT")
                elif text in {"QUIET", "QUIETED"}:
                    mapped.add("QUIETED")
            return mapped or None

        requested_types = normalize_types(event_types)

        def include_type(name: str) -> bool:
            return requested_types is None or name in requested_types

        events: List[Dict[str, Any]] = []

        def append_event(event: Dict[str, Any]) -> None:
            stamp = parse_ts_utc(event.get("time_utc"))
            event["_sort_ts"] = stamp or base_cutoff_dt
            event["summary"] = format_change_summary(event)
            event["time_label"] = format_change_time_label(event.get("time_utc"))
            events.append(event)

        if include_type("NEW_SIGNAL") and table_exists("baseline_detections"):
            try:
                rows = qa(
                    con(),
                    """
                    SELECT id, f_center_hz, f_low_hz, f_high_hz,
                           first_seen_utc, last_seen_utc, confidence,
                           total_hits, total_windows
                    FROM baseline_detections
                    WHERE baseline_id = ?
                      AND first_seen_utc >= ?
                    ORDER BY first_seen_utc DESC
                    LIMIT ?
                    """,
                    (baseline_id, new_cutoff_iso, CHANGE_EVENT_LIMIT * 2),
                )
            except sqlite3.OperationalError:
                rows = []
            for row in rows:
                f_low = row.get("f_low_hz")
                f_high = row.get("f_high_hz")
                bandwidth = None
                if f_low is not None and f_high is not None:
                    try:
                        bandwidth = max(0.0, float(f_high) - float(f_low))
                    except Exception:
                        bandwidth = None
                event = {
                    "type": "NEW_SIGNAL",
                    "time_utc": row.get("first_seen_utc"),
                    "f_center_hz": row.get("f_center_hz"),
                    "bandwidth_hz": bandwidth,
                    "confidence": row.get("confidence"),
                    "details": "First seen relative to baseline.",
                    "detection_id": row.get("id"),
                    "total_hits": row.get("total_hits"),
                }
                append_event(event)

        quiet_window_active = quiet_cutoff_dt > base_cutoff_dt
        if include_type("QUIETED") and quiet_window_active and table_exists("baseline_detections"):
            try:
                rows = qa(
                    con(),
                    """
                    SELECT id, f_center_hz, f_low_hz, f_high_hz,
                           first_seen_utc, last_seen_utc, total_hits,
                           total_windows, confidence
                    FROM baseline_detections
                    WHERE baseline_id = ?
                      AND total_windows >= ?
                      AND last_seen_utc >= ?
                      AND last_seen_utc <= ?
                    ORDER BY last_seen_utc DESC
                    LIMIT ?
                    """,
                    (
                        baseline_id,
                        QUIETED_MIN_WINDOWS,
                        base_cutoff_iso,
                        quiet_cutoff_iso,
                        CHANGE_EVENT_LIMIT * 2,
                    ),
                )
            except sqlite3.OperationalError:
                rows = []
            for row in rows:
                f_low = row.get("f_low_hz")
                f_high = row.get("f_high_hz")
                bandwidth = None
                if f_low is not None and f_high is not None:
                    try:
                        bandwidth = max(0.0, float(f_high) - float(f_low))
                    except Exception:
                        bandwidth = None
                last_seen = row.get("last_seen_utc")
                downtime_minutes = None
                last_seen_dt = parse_ts_utc(last_seen)
                if last_seen_dt:
                    downtime_minutes = max(0.0, (now_ts - last_seen_dt).total_seconds() / 60.0)
                details = "Previously persistent, now not seen recently."
                if downtime_minutes is not None:
                    details = f"Previously persistent, quiet for ~{downtime_minutes:.0f} min."
                event = {
                    "type": "QUIETED",
                    "time_utc": last_seen,
                    "f_center_hz": row.get("f_center_hz"),
                    "bandwidth_hz": bandwidth,
                    "confidence": row.get("confidence"),
                    "details": details,
                    "detection_id": row.get("id"),
                    "total_windows": row.get("total_windows"),
                    "downtime_minutes": downtime_minutes,
                }
                append_event(event)

        if include_type("POWER_SHIFT") and table_exists("baseline_stats"):
            try:
                rows = qa(
                    con(),
                    """
                    SELECT bin_index, freq_hz, power_ema, noise_floor_ema,
                           last_seen_utc
                    FROM baseline_stats
                    WHERE baseline_id = ?
                      AND last_seen_utc >= ?
                    ORDER BY last_seen_utc DESC
                    LIMIT ?
                    """,
                    (baseline_id, base_cutoff_iso, CHANGE_EVENT_LIMIT * 2),
                )
            except sqlite3.OperationalError:
                rows = []
            freq_start = float(baseline_row.get("freq_start_hz") or 0.0)
            bin_hz = float(baseline_row.get("bin_hz") or 0.0)
            for row in rows:
                freq_val = row.get("freq_hz")
                freq_hz = None
                if freq_val is not None:
                    try:
                        freq_hz = float(freq_val)
                    except Exception:
                        freq_hz = None
                elif row.get("bin_index") is not None and bin_hz > 0:
                    try:
                        freq_hz = freq_start + float(row.get("bin_index")) * bin_hz
                    except Exception:
                        freq_hz = None
                if freq_hz is None:
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
                event = {
                    "type": "POWER_SHIFT",
                    "time_utc": last_seen,
                    "f_center_hz": freq_hz,
                    "bandwidth_hz": bin_hz if bin_hz > 0 else None,
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

        payload = {
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
            "generated_at": _isoformat_utc(_now_utc()),
            "active_filter": active_filter,
            "requested_types": sorted(requested_types) if requested_types else [],
        }
        return payload

    def ensure_baseline_schema(conn: sqlite3.Connection) -> None:
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
                total_windows INTEGER NOT NULL DEFAULT 0
            )
            """,
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
                confidence REAL NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS scan_updates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                baseline_id INTEGER NOT NULL,
                timestamp_utc TEXT NOT NULL,
                num_hits INTEGER NOT NULL,
                num_segments INTEGER NOT NULL,
                num_new_signals INTEGER NOT NULL
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
        ]
        for stmt in stmts:
            conn.execute(stmt)

    def create_baseline_entry(data: Dict[str, Any]) -> Dict[str, Any]:
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
            "created_at": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
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

        try:
            conn = sqlite3.connect(app._db_path)
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
            reset_ro_connection()
            _ensure_con()

        if row is None:
            raise RuntimeError("baseline created but could not be reloaded")
        result = dict(row)
        result["baseline_id"] = result.get("id")
        return result

    def load_spur_bins() -> List[int]:
        connection = _ensure_con()
        if connection is None:
            return []
        try:
            rows = qa(connection, "SELECT bin_hz FROM spur_map")
        except sqlite3.OperationalError:
            return []
        except sqlite3.Error:
            return []
        bins: List[int] = []
        for row in rows:
            try:
                val = row.get("bin_hz") if isinstance(row, dict) else row[0]
                if val is not None:
                    bins.append(int(val))
            except Exception:
                continue
        bins.sort()
        return bins

    def annotate_near_spur(records: List[Dict[str, Any]], bins: List[int], *, tolerance_hz: int = 5_000) -> None:
        if not bins:
            for rec in records:
                rec["near_spur"] = False
            return
        for rec in records:
            fc = rec.get("f_center_hz")
            near = False
            if fc is not None:
                try:
                    fc_int = int(fc)
                    idx = bisect_left(bins, fc_int)
                    for pos in (idx, idx - 1):
                        if 0 <= pos < len(bins) and abs(fc_int - bins[pos]) <= tolerance_hz:
                            near = True
                            break
                except Exception:
                    near = False
            rec["near_spur"] = near
    def load_baseline_bins(f_min: Optional[int], f_max: Optional[int]) -> List[Tuple[int, float]]:
        connection = _ensure_con()
        if connection is None:
            return []
        query = "SELECT bin_hz, ema_occ FROM baseline"
        params: List[Any] = []
        clauses: List[str] = []
        if f_min is not None:
            clauses.append("bin_hz >= ?")
            params.append(int(f_min))
        if f_max is not None:
            clauses.append("bin_hz <= ?")
            params.append(int(f_max))
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY bin_hz"
        try:
            rows = qa(connection, query, tuple(params))
        except sqlite3.OperationalError:
            return []
        except sqlite3.Error:
            return []
        bins: List[Tuple[int, float]] = []
        for row in rows:
            try:
                bin_hz = row.get("bin_hz") if isinstance(row, dict) else row[0]
                occ = row.get("ema_occ") if isinstance(row, dict) else row[1]
                if bin_hz is None or occ is None:
                    continue
                bins.append((int(bin_hz), float(occ)))
            except Exception:
                continue
        return bins

    def annotate_baseline_status(records: List[Dict[str, Any]], bins: List[Tuple[int, float]], threshold: float) -> None:
        if not bins:
            for rec in records:
                rec["baseline_status"] = "unknown"
                rec["is_new"] = None
            return
        freqs = [b[0] for b in bins]
        for rec in records:
            fc = rec.get("f_center_hz")
            status = "unknown"
            is_new = None
            if fc is not None:
                try:
                    fc_int = int(fc)
                    idx = bisect_left(freqs, fc_int)
                    closest_occ = None
                    best_diff = None
                    for pos in (idx, idx - 1):
                        if 0 <= pos < len(freqs):
                            diff = abs(fc_int - freqs[pos])
                            if best_diff is None or diff < best_diff:
                                closest_occ = bins[pos]
                                best_diff = diff
                    if closest_occ is not None:
                        occ_val = closest_occ[1]
                        is_new = bool(occ_val < threshold)
                        status = "new" if is_new else "known"
                except Exception:
                    status = "unknown"
            rec["baseline_status"] = status
            rec["is_new"] = is_new

    def start_job_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        device_key = payload.get('device_key')
        if not device_key:
            abort(400, description='device_key is required')
        label = payload.get('label') or 'web'
        params = payload.get('params') or {}
        baseline_id_raw = payload.get('baseline_id')
        if baseline_id_raw is None:
            abort(400, description='baseline_id is required; create or select a baseline first.')
        if isinstance(baseline_id_raw, str):
            candidate = baseline_id_raw.strip()
            if not candidate:
                abort(400, description='baseline_id is required; create or select a baseline first.')
            baseline_id_token: Any = candidate
        else:
            baseline_id_token = baseline_id_raw
        try:
            baseline_id_int = int(baseline_id_token)
        except Exception as exc:
            abort(400, description=f'invalid baseline_id: {exc}')
        try:
            return app._ctl.start_job(device_key, label, baseline_id_int, params)
        except Exception as exc:
            abort(400, description=str(exc))

    def stop_job_by_id(job_id: str) -> Dict[str, Any]:
        try:
            return app._ctl.stop_job(job_id)
        except Exception as exc:
            abort(400, description=str(exc))
            # abort raises an HTTPException; provide an explicit return to satisfy static type checkers
            return {"error": str(exc)}

    def job_logs_response(job_id: str, tail: Optional[int] = None) -> Response:
        try:
            data = app._ctl.job_logs(job_id, tail=tail)
            return Response(data or "", mimetype='text/plain')
        except Exception as exc:
            abort(404, description=str(exc))

    # ---------- Pages ----------
    @app.get('/control')
    def control():
        baseline_list = controller_baselines()
        summaries = baseline_summary_map()
        return render_template(
            "control.html",
            db_path=app._db_path,
            profiles=controller_profiles(),
            baselines=baseline_list,
            baseline_summaries=summaries,
        )

    @app.route('/live')
    def live():
        state, state_message = db_state()
        return render_template(
            "live.html",
            db_status=state,
            db_status_message=state_message,
            db_path=app._db_path,
        )

    @app.route('/')
    def dashboard():
        state, state_message = db_state()
        tactical_config = {
            "active_window_minutes": ACTIVE_SIGNAL_WINDOW_MINUTES,
            "recent_new_minutes": TACTICAL_RECENT_MINUTES,
            "hotspot_bucket_count": HOTSPOT_BUCKET_COUNT,
        }
        if state != "ready":
            context = db_waiting_context(state, state_message)
            context["tactical_config"] = tactical_config
            if request.headers.get("HX-Request"):
                return render_template("partials/dashboard_empty.html", **context)
            return render_template("dashboard.html", **context)

        def safe_table_count(table: str) -> int:
            if not table_exists(table):
                return 0
            try:
                row = q1(con(), f"SELECT COUNT(*) AS c FROM {table}")
            except sqlite3.OperationalError:
                return 0
            return int(row.get('c') or 0)

        detections_available = table_exists("detections")
        scans_available = table_exists("scans")

        scans_total = safe_table_count("scans")
        detections_total = safe_table_count("detections")
        baseline_total = safe_table_count("baselines")

        filters, form_defaults = parse_detection_filters(request.args, default_since_hours=168)
        confidence_available = detections_have_confidence()
        filters["__confidence_available"] = confidence_available

        snr_bucket_db = 3
        snr_bucket_raw = request.args.get('snr_bucket_db')
        if snr_bucket_raw:
            try:
                snr_bucket_db = max(1, int(float(snr_bucket_raw)))
            except Exception:
                pass

        heatmap_scans = 20
        heatmap_scans_raw = request.args.get('heatmap_scans')
        if heatmap_scans_raw:
            try:
                heatmap_scans = max(5, min(100, int(float(heatmap_scans_raw))))
            except Exception:
                pass

        heatmap_bins = 36
        heatmap_bins_raw = request.args.get('heatmap_bins')
        if heatmap_bins_raw:
            try:
                heatmap_bins = max(12, min(120, int(float(heatmap_bins_raw))))
            except Exception:
                pass

        def empty_timeline_payload() -> Dict[str, Any]:
            return {
                "bucket_hours": 1,
                "buckets": [],
                "det_max": 0,
                "scan_max": 0,
                "snr_max": 0,
                "style_attr": f'style="height:{CHART_HEIGHT_PX}px;"',
            }

        def empty_heatmap_payload() -> Dict[str, Any]:
            return {
                "rows": [],
                "bin_labels": [],
                "f_start_mhz": None,
                "f_stop_mhz": None,
                "bin_width_mhz": None,
                "max_count": 0,
            }

        filtered_detections = 0
        snr_hist: List[Dict[str, Any]] = []
        snr_stats: Optional[Dict[str, Any]] = None
        freq_bins: List[Dict[str, Any]] = []
        latest: Optional[Dict[str, Any]] = None
        freq_max = 0
        avg_bins: List[Dict[str, Any]] = []
        avg_start_mhz = 0.0
        avg_stop_mhz = 0.0
        avg_max = 0.0
        timeline = empty_timeline_payload()
        heatmap = empty_heatmap_payload()
        services: List[str] = []
        top_services_data: List[Dict[str, Any]] = []
        strongest: List[Dict[str, Any]] = []

        if detections_available:
            conds_count, params_count = detection_predicates(filters, alias="d")
            count_where = " WHERE " + " AND ".join(conds_count) if conds_count else ""
            if count_where:
                try:
                    filtered_row = q1(
                        con(),
                        f"SELECT COUNT(*) AS c FROM detections d{count_where}",
                        tuple(params_count),
                    )
                    filtered_detections = int((filtered_row or {}).get("c") or 0)
                except sqlite3.OperationalError:
                    filtered_detections = 0
            else:
                filtered_detections = detections_total

            try:
                snr_hist, snr_stats = snr_histogram(con(), filters, bucket_db=snr_bucket_db)
            except sqlite3.OperationalError:
                snr_hist, snr_stats = [], None

            if scans_available:
                try:
                    freq_bins, latest, freq_max = frequency_bins_latest_scan(con(), filters, num_bins=40)
                except sqlite3.OperationalError:
                    freq_bins, latest, freq_max = [], None, 0
            if scans_available:
                try:
                    avg_bins, avg_start_mhz, avg_stop_mhz, avg_max = frequency_bins_all_scans_avg(con(), filters, num_bins=40)
                except sqlite3.OperationalError:
                    avg_bins, avg_start_mhz, avg_stop_mhz, avg_max = [], 0.0, 0.0, 0.0

            if scans_available:
                try:
                    timeline = timeline_metrics(con(), filters)
                except sqlite3.OperationalError:
                    timeline = empty_timeline_payload()
                try:
                    heatmap = coverage_heatmap(con(), filters, max_scans=heatmap_scans, num_bins=heatmap_bins)
                except sqlite3.OperationalError:
                    heatmap = empty_heatmap_payload()
            try:
                services = [r['service'] for r in qa(con(), "SELECT DISTINCT COALESCE(service,'Unknown') AS service FROM detections ORDER BY service")]
            except sqlite3.OperationalError:
                services = []
            try:
                top_services_data = top_services(con(), filters, limit=10)
            except sqlite3.OperationalError:
                top_services_data = []
            try:
                strongest = strongest_signals(con(), filters, limit=10, include_confidence=confidence_available)
            except sqlite3.OperationalError:
                strongest = []
        else:
            services = []

        active_filters: List[Dict[str, str]] = []
        if filters.get('service'):
            active_filters.append({"label": "Service", "value": str(filters['service'])})
        if filters.get('min_snr') is not None:
            active_filters.append({"label": "Min SNR", "value": f"{filters['min_snr']:.1f} dB"})
        if confidence_available and filters.get('min_conf') is not None:
            active_filters.append({"label": "Confidence", "value": f"≥ {filters['min_conf']:.2f}"})
        if filters.get('f_min_hz') is not None or filters.get('f_max_hz') is not None:
            lo = filters.get('f_min_hz')
            hi = filters.get('f_max_hz')
            if lo is not None and hi is not None:
                active_filters.append({"label": "Freq", "value": f"{lo/1e6:.3f}–{hi/1e6:.3f} MHz"})
            elif lo is not None:
                active_filters.append({"label": "Freq ≥", "value": f"{lo/1e6:.3f} MHz"})
            elif hi is not None:
                active_filters.append({"label": "Freq ≤", "value": f"{hi/1e6:.3f} MHz"})
        if filters.get('since_hours'):
            active_filters.append({"label": "Lookback", "value": f"{int(filters['since_hours'])} h"})

        context = dict(
            scans_total=scans_total,
            detections_total=detections_total,
            baseline_total=baseline_total,
            latest=latest,
            snr_hist=snr_hist,
            snr_stats=snr_stats,
            snr_bucket_db=snr_bucket_db,
            freq_bins=freq_bins,
            freq_max=freq_max,
            avg_bins=avg_bins,
            avg_start_mhz=avg_start_mhz,
            avg_stop_mhz=avg_stop_mhz,
            avg_max=avg_max,
            timeline=timeline,
            heatmap=heatmap,
            top_services=top_services_data,
            strongest=strongest,
            chart_px=CHART_HEIGHT_PX,
            chart_style_attr=f'style="height:{CHART_HEIGHT_PX}px;"',
            services=services,
            form_defaults=form_defaults,
            active_filters=active_filters,
            filtered_detections=filtered_detections,
            heatmap_settings={"scans": heatmap_scans, "bins": heatmap_bins},
            format_ts_label=format_ts_label,
            confidence_available=confidence_available,
            db_status="ready",
            db_status_message="",
            db_path=app._db_path,
            tactical_config=tactical_config,
        )

        if request.headers.get("HX-Request"):
            return render_template("partials/dashboard_content.html", **context)

        return render_template("dashboard.html", **context)

    @app.route('/changes')
    def changes():
        state, state_message = db_state()
        change_config = {
            "window_minutes": CHANGE_WINDOW_MINUTES,
            "new_signal_window_minutes": NEW_SIGNAL_WINDOW_MINUTES,
            "quiet_timeout_minutes": QUIETED_TIMEOUT_MINUTES,
            "power_shift_threshold_db": POWER_SHIFT_THRESHOLD_DB,
            "event_limit": CHANGE_EVENT_LIMIT,
        }
        if state != "ready":
            context = db_waiting_context(state, state_message)
            context.update(
                {
                    "change_config": change_config,
                    "baselines": [],
                    "selected_baseline": None,
                    "selected_baseline_id": "",
                    "change_payload": None,
                    "active_filter": "ALL",
                }
            )
            return render_template("changes.html", **context)
        baselines = qa(
            con(),
            """
            SELECT id, name, freq_start_hz, freq_stop_hz, bin_hz, total_windows
            FROM baselines
            ORDER BY id DESC
            """,
        )
        if not baselines:
            return render_template(
                "changes.html",
                baselines=[],
                selected_baseline=None,
                selected_baseline_id="",
                change_payload=None,
                change_config=change_config,
                active_filter="ALL",
            )
        baseline_id_param = (request.args.get('baseline_id') or "").strip()
        selected_baseline = None
        if baseline_id_param:
            for row in baselines:
                if str(row.get('id')) == baseline_id_param:
                    selected_baseline = row
                    break
        if selected_baseline is None:
            selected_baseline = baselines[0]
        raw_id = selected_baseline.get('id')
        baseline_id = None
        try:
            if raw_id is not None:
                # Normalize to string/float first so mypy/typers accept the argument to int()
                baseline_id = int(float(str(raw_id)))
        except Exception:
            baseline_id = None
        filter_param_raw = (request.args.get('type') or "ALL").strip().upper()
        if filter_param_raw in {"NEW", "NEW_SIGNAL"}:
            active_filter = "NEW_SIGNAL"
            filter_values: Optional[List[str]] = ["NEW_SIGNAL"]
        elif filter_param_raw in {"POWER", "POWER_SHIFT"}:
            active_filter = "POWER_SHIFT"
            filter_values = ["POWER_SHIFT"]
        elif filter_param_raw in {"QUIET", "QUIETED"}:
            active_filter = "QUIETED"
            filter_values = ["QUIETED"]
        else:
            active_filter = "ALL"
            filter_values = None
        payload = None
        if baseline_id is not None:
            payload = change_events_payload(
                baseline_id,
                window_minutes=CHANGE_WINDOW_MINUTES,
                event_types=filter_values,
            )
        selected_baseline_id = str(raw_id or "")
        return render_template(
            "changes.html",
            baselines=baselines,
            selected_baseline=selected_baseline,
            selected_baseline_id=selected_baseline_id,
            change_payload=payload,
            change_config=change_config,
            active_filter=active_filter,
        )

    @app.route('/spur-map')
    def spur_map():
        state, state_message = db_state()
        if state != "ready":
            return render_template("db_waiting.html", **db_waiting_context(state, state_message))
        try:
            exists_row = q1(
                con(),
                "SELECT name FROM sqlite_master WHERE type='table' AND name='spur_map'",
            )
            has_table = bool(exists_row and exists_row.get('name'))
        except sqlite3.OperationalError:
            has_table = False
        rows: List[Dict[str, Any]] = []
        if has_table:
            rows = qa(
                con(),
                """
                SELECT bin_hz, mean_power_db, hits, last_seen_utc
                FROM spur_map
                ORDER BY bin_hz
                """,
            )
        return render_template("spur_map.html", rows=rows, has_table=has_table)

    @app.route('/export/detections.csv')
    def export_csv():
        state, state_message = db_state()
        if state != "ready":
            abort(409, description=state_message or "Database not initialized yet")
        args = request.args
        params: List[Any] = []
        where = []
        if args.get('service'): where.append("COALESCE(service,'Unknown') = ?"); params.append(args.get('service'))
        if args.get('min_snr'): where.append("snr_db >= ?"); params.append(float(args.get('min_snr')))
        min_conf_arg = args.get('min_conf')
        confidence_available = detections_have_confidence()
        if min_conf_arg and confidence_available:
            where.append("confidence >= ?")
            params.append(float(min_conf_arg))
        if args.get('f_min_mhz'): where.append("f_center_hz >= ?"); params.append(int(float(args.get('f_min_mhz'))*1e6))
        if args.get('f_max_mhz'): where.append("f_center_hz <= ?"); params.append(int(float(args.get('f_max_mhz'))*1e6))
        if args.get('since_hours'): where.append("time_utc >= datetime('now', ?)"); params.append(f"-{int(float(args.get('since_hours')))} hours")
        where_sql = (" WHERE "+" AND ".join(where)) if where else ""
        confidence_sql = "confidence" if confidence_available else "NULL"
        rows = qa(con(), f"""
            SELECT time_utc, scan_id, f_center_hz, f_low_hz, f_high_hz,
                   peak_db, noise_db, snr_db, service, region, notes,
                   {confidence_sql} AS confidence
            FROM detections {where_sql}
            ORDER BY time_utc DESC
            LIMIT 100000
        """, tuple(params))
        import csv
        buf = io.StringIO()
        fieldnames = ["time_utc","scan_id","f_center_hz","f_low_hz","f_high_hz","peak_db","noise_db","snr_db","service","region","notes","confidence"]
        w = csv.DictWriter(buf, fieldnames=fieldnames); w.writeheader()
        for r in rows: w.writerow({k:r.get(k,'') for k in fieldnames})
        buf.seek(0)
        return Response(buf.read(), mimetype='text/csv', headers={'Content-Disposition':'attachment; filename=detections.csv'})

    # ---------- Controller proxy API ----------
    @app.get('/ctl/devices')
    def ctl_devices():
        try:
            devs = app._ctl.devices()
            return jsonify(devs)
        except Exception as e:
            # Surface the reason to the frontend as a JSON object (not an empty list)
            msg = str(e)
            hint = "unauthorized" if "401" in msg or "unauthorized" in msg.lower() else "unreachable"
            return jsonify({"error": f"controller_{hint}", "detail": msg})

    @app.get('/api/baselines')
    def api_baselines_list():
        require_auth()
        baselines_payload = {
            "baselines": controller_baselines(),
            "summaries": baseline_summary_map(),
        }
        return jsonify(baselines_payload)

    @app.post('/api/baselines')
    def api_baselines_create():
        require_auth()
        payload = request.get_json(force=True, silent=False) or {}
        try:
            row = create_baseline_entry(payload)
        except ValueError as exc:
            abort(400, description=str(exc))
        except RuntimeError as exc:
            abort(500, description=str(exc))
        summaries = baseline_summary_map()
        return (jsonify({"baseline": row, "summaries": summaries}), 201)

    @app.get('/api/baseline/<int:baseline_id>/tactical')
    def api_baseline_tactical(baseline_id: int):
        require_auth()
        payload = tactical_snapshot_payload(baseline_id)
        if not payload:
            abort(404, description="Baseline not found")
        return jsonify(payload)

    @app.get('/api/baseline/<int:baseline_id>/hotspots')
    def api_baseline_hotspots(baseline_id: int):
        require_auth()
        payload = hotspots_payload(baseline_id)
        if not payload:
            abort(404, description="Baseline not found")
        return jsonify(payload)

    @app.get('/api/baseline/<int:baseline_id>/changes')
    def api_baseline_changes(baseline_id: int):
        require_auth()
        minute_override = request.args.get('minutes', type=int)
        type_tokens = request.args.getlist('type')
        payload = change_events_payload(
            baseline_id,
            window_minutes=minute_override or CHANGE_WINDOW_MINUTES,
            event_types=type_tokens if type_tokens else None,
        )
        if not payload:
            abort(404, description="Baseline not found")
        return jsonify(payload)

    def active_state_payload() -> Dict[str, Any]:
        job = controller_active_job()
        if not job:
            return {"state": "idle"}
        return {"state": "running", "job": job}

    def start_job_response():
        require_auth()
        payload = request.get_json(force=True, silent=False) or {}
        job = start_job_from_payload(payload)
        state_val = job.get('status', 'running') if isinstance(job, dict) else 'running'
        return jsonify({"state": state_val, "job": job})

    @app.get('/api/jobs')
    def api_jobs_list():
        require_auth()
        try:
            jobs = app._ctl.list_jobs()
            return jsonify({"jobs": jobs})
        except Exception as exc:
            abort(502, description=str(exc))

    @app.get('/api/jobs/active')
    def api_jobs_active():
        require_auth()
        return jsonify(active_state_payload())

    @app.post('/api/jobs')
    def api_jobs_create():
        return start_job_response()

    @app.post('/api/scans')
    def api_start_scan():
        return start_job_response()

    @app.get('/api/jobs/<job_id>')
    def api_job_detail(job_id: str):
        require_auth()
        try:
            job = app._ctl.job_detail(job_id)
            return jsonify(job)
        except Exception as exc:
            abort(404, description=str(exc))

    @app.delete('/api/jobs/<job_id>')
    def api_job_delete(job_id: str):
        require_auth()
        data = stop_job_by_id(job_id)
        return jsonify(data)

    @app.delete('/api/scans/active')
    def api_stop_active():
        require_auth()
        job = controller_active_job()
        if not job:
            return jsonify({"ok": True})
        data = stop_job_by_id(str(job.get('id')))
        return jsonify(data)

    @app.get('/api/jobs/<job_id>/logs')
    def api_job_logs(job_id: str):
        require_auth()
        tail = request.args.get('tail', type=int)
        return job_logs_response(job_id, tail)

    @app.get('/api/now')
    def api_now():
        require_auth()
        return jsonify(active_state_payload())

    @app.get('/api/logs')
    def api_logs():
        require_auth()
        job_id = request.args.get('job_id')
        tail = request.args.get('tail', type=int)
        if not job_id:
            job = controller_active_job()
            if not job:
                return Response("", mimetype='text/plain')
            job_id = str(job.get('id'))
        return job_logs_response(job_id, tail)

    @app.get('/api/live/windows')
    def api_live_windows():
        require_auth()
        job_id = request.args.get('job_id')
        tail = request.args.get('tail', type=int)
        limit = request.args.get('limit', type=int) or 100
        limit = max(1, min(500, limit))
        tail = tail if tail and tail > 0 else 5000
        if not job_id:
            job = controller_active_job()
            if not job:
                return jsonify({"windows": []})
            job_id = str(job.get('id'))
        try:
            log_text = app._ctl.job_logs(job_id, tail=tail)
        except Exception as exc:
            abort(502, description=str(exc))
        windows: List[Dict[str, Any]] = []
        for line in log_text.splitlines():
            parsed = parse_window_log_line(line)
            if not parsed:
                continue
            center_hz = float(parsed.get('center_hz', 0.0))
            parsed['center_hz'] = center_hz
            parsed['center_mhz'] = center_hz / 1e6
            windows.append(parsed)
        if not windows:
            return jsonify({"windows": []})
        windows = windows[-limit:]
        return jsonify({"windows": windows})

    return app

# ================================
# CLI
# ================================

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--db', required=True)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=8080)
    return ap.parse_args()

if __name__ == '__main__':
    args = parse_args()
    app = create_app(args.db)
    app.run(host=args.host, port=args.port, threaded=True)

