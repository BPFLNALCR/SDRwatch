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
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from flask import Flask, request, Response, render_template, render_template_string, jsonify, abort, url_for  # type: ignore
from urllib import request as urlreq, parse as urlparse, error as urlerr

# ================================
# Config
# ================================
CHART_HEIGHT_PX = 160
API_TOKEN = os.getenv("SDRWATCH_TOKEN", "")  # page auth (optional)
CONTROL_URL = os.getenv("SDRWATCH_CONTROL_URL", "http://127.0.0.1:8765")
CONTROL_TOKEN = os.getenv("SDRWATCH_CONTROL_TOKEN", "") or os.getenv("SDRWATCH_TOKEN", "")
BASELINE_NEW_THRESHOLD = 0.2

# ================================
# DB helpers
# ================================

def open_db_ro(path: str) -> sqlite3.Connection:
    abspath = os.path.abspath(path)
    con = sqlite3.connect(f"file:{abspath}?mode=ro", uri=True, check_same_thread=False)
    con.execute("PRAGMA busy_timeout=2000;")
    con.row_factory = lambda cur, row: {d[0]: row[i] for i, d in enumerate(cur.description)}
    return con

def q1(con: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()):  # one row
    cur = con.execute(sql, params)
    return cur.fetchone()

def qa(con: sqlite3.Connection, sql: str, params: Tuple[Any, ...] = ()):  # all rows
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
        req = urlreq.Request(url, data=data, headers=headers, method=method.upper())
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

    # Attempt initial connection (tolerates failure if DB is missing)
    _ensure_con()

    def con() -> sqlite3.Connection:
        connection = _ensure_con()
        if connection is None:
            raise RuntimeError("database connection unavailable")
        return connection

    def db_state() -> Tuple[str, str]:
        connection = _ensure_con()
        if connection is None:
            return (
                "unavailable",
                app._db_error or "Database file could not be opened in read-only mode.",
            )
        try:
            cur = connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('scans','detections','baseline')"
            )
            names = set()
            for row in cur.fetchall():
                if isinstance(row, dict):
                    names.add(str(row.get('name', '')))
                else:
                    names.add(str(row[0]))
            required = {"scans", "detections", "baseline"}
            if not required.issubset(names):
                return ("waiting", "")
        except sqlite3.OperationalError as exc:
            app._con = None
            app._db_error = str(exc)
            return (
                "waiting",
                f"Database not initialized yet ({exc}). Start a scan to populate it.",
            )
        return ("ready", "")

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

    def controller_baselines() -> List[Dict[str, Any]]:
        try:
            data = app._ctl.baselines()
        except Exception as exc:
            app.logger.warning("controller /baselines fetch failed: %s", exc)
            return []
        if isinstance(data, list):
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
                bid = int(raw_id)
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

        def _coerce_int(value: Any, field: str) -> int:
            if value in (None, ""):
                raise ValueError(f"{field} is required")
            try:
                return int(float(value))
            except Exception as exc:
                raise ValueError(f"{field} must be a number") from exc

        def _coerce_float(value: Any) -> Optional[float]:
            if value in (None, ""):
                return None
            try:
                return float(value)
            except Exception as exc:
                raise ValueError("Invalid coordinate value") from exc

        freq_start = _coerce_int(data.get("freq_start_hz"), "freq_start_hz")
        freq_stop = _coerce_int(data.get("freq_stop_hz"), "freq_stop_hz")
        if freq_stop <= freq_start:
            raise ValueError("freq_stop_hz must be greater than freq_start_hz")

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
            "freq_start_hz": freq_start,
            "freq_stop_hz": freq_stop,
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
        if state != "ready":
            context = db_waiting_context(state, state_message)
            if request.headers.get("HX-Request"):
                return render_template("partials/dashboard_empty.html", **context)
            return render_template("dashboard.html", **context)

        scans_total = q1(con(), "SELECT COUNT(*) AS c FROM scans")['c'] or 0
        detections_total = q1(con(), "SELECT COUNT(*) AS c FROM detections")['c'] or 0
        baseline_total = q1(con(), "SELECT COUNT(*) AS c FROM baseline")['c'] or 0

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

        conds_count, params_count = detection_predicates(filters, alias="d")
        count_where = " WHERE " + " AND ".join(conds_count) if conds_count else ""
        if count_where:
            filtered_row = q1(
                con(),
                f"SELECT COUNT(*) AS c FROM detections d{count_where}",
                tuple(params_count),
            )
            filtered_detections = filtered_row.get("c") if filtered_row else 0
        else:
            filtered_detections = detections_total
        filtered_detections = int(filtered_detections or 0)

        snr_hist, snr_stats = snr_histogram(con(), filters, bucket_db=snr_bucket_db)
        freq_bins, latest, freq_max = frequency_bins_latest_scan(con(), filters, num_bins=40)
        avg_bins, avg_start_mhz, avg_stop_mhz, avg_max = frequency_bins_all_scans_avg(con(), filters, num_bins=40)
        timeline = timeline_metrics(con(), filters)
        heatmap = coverage_heatmap(con(), filters, max_scans=heatmap_scans, num_bins=heatmap_bins)
        services = [r['service'] for r in qa(con(), "SELECT DISTINCT COALESCE(service,'Unknown') AS service FROM detections ORDER BY service")]
        top_services_data = top_services(con(), filters, limit=10)
        strongest = strongest_signals(con(), filters, limit=10, include_confidence=confidence_available)

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
        )

        if request.headers.get("HX-Request"):
            return render_template("partials/dashboard_content.html", **context)

        return render_template("dashboard.html", **context)

    @app.route('/timeline')
    def timeline_view():
        state, state_message = db_state()
        if state != "ready":
            abort(409, description=state_message or "Database not initialized yet")
        filters, form_defaults = parse_detection_filters(request.args, default_since_hours=168)
        confidence_available = detections_have_confidence()
        filters["__confidence_available"] = confidence_available
        hist, snr_stats = snr_histogram(con(), filters, bucket_db=3)
        timeline_data = timeline_metrics(con(), filters, max_buckets=120)
        conds, params = detection_predicates(filters, alias="d")
        where_sql = " WHERE " + " AND ".join(conds) if conds else ""
        params_tuple = tuple(params)
        total_row = q1(con(), f"SELECT COUNT(*) AS c FROM detections d{where_sql}", params_tuple)
        filtered_detections = int((total_row or {}).get("c") or 0)
        buckets = timeline_data.get("buckets", [])
        bucket_sum = sum(int(b.get("detections") or 0) for b in buckets)
        bucket_count = len(buckets)
        time_span = ""
        if bucket_count:
            newest = buckets[0].get("label", "")
            oldest = buckets[-1].get("label", "")
            if newest and oldest:
                time_span = f"{oldest} → {newest}"
        services = [r['service'] for r in qa(con(), "SELECT DISTINCT COALESCE(service,'Unknown') AS service FROM detections ORDER BY service")]
        return render_template(
            "timeline.html",
            filters=filters,
            form_defaults=form_defaults,
            hist=hist,
            snr_stats=snr_stats,
            timeline=timeline_data,
            services=services,
            detection_total=filtered_detections,
            bucket_sum=bucket_sum,
            bucket_count=bucket_count,
            time_span=time_span,
            confidence_available=confidence_available,
        )

    @app.route('/coverage')
    def coverage_view():
        state, state_message = db_state()
        if state != "ready":
            abort(409, description=state_message or "Database not initialized yet")
        filters, form_defaults = parse_detection_filters(request.args)
        confidence_available = detections_have_confidence()
        filters["__confidence_available"] = confidence_available
        latest_bins, latest_meta, latest_max = frequency_bins_latest_scan(con(), filters, num_bins=60)
        avg_bins, avg_start_mhz, avg_stop_mhz, avg_max = frequency_bins_all_scans_avg(con(), filters, num_bins=60)
        services = [r['service'] for r in qa(con(), "SELECT DISTINCT COALESCE(service,'Unknown') AS service FROM detections ORDER BY service")]
        return render_template(
            "coverage.html",
            filters=filters,
            form_defaults=form_defaults,
            latest_bins=latest_bins,
            latest_meta=latest_meta,
            latest_max=latest_max,
            avg_bins=avg_bins,
            avg_start_mhz=avg_start_mhz,
            avg_stop_mhz=avg_stop_mhz,
            avg_max=avg_max,
            services=services,
            confidence_available=confidence_available,
        )

    @app.route('/detections')
    def detections():
        state, state_message = db_state()
        if state != "ready":
            return render_template("db_waiting.html", **db_waiting_context(state, state_message))
        args = request.args
        filters, form_defaults = parse_detection_filters(args)
        confidence_available = detections_have_confidence()
        filters["__confidence_available"] = confidence_available
        confidence_sql = "d.confidence" if confidence_available else "NULL"
        conds, params = detection_predicates(filters, alias="d")
        where_sql = " WHERE " + " AND ".join(conds) if conds else ""
        params_tuple = tuple(params)
        page = max(1, int(float(args.get('page', 1))))
        page_size = min(200, max(10, int(float(args.get('page_size', 50)))))
        offset = (page - 1) * page_size

        detections_total = q1(con(), "SELECT COUNT(*) AS c FROM detections")['c']
        filtered_row = q1(con(), f"SELECT COUNT(*) AS c FROM detections d{where_sql}", params_tuple) or {"c": 0}
        filtered_total = int(filtered_row.get('c') or 0)

        rows = qa(
            con(),
            f"""
            SELECT d.time_utc, d.scan_id, d.f_center_hz, d.f_low_hz, d.f_high_hz,
                   d.peak_db, d.noise_db, d.snr_db, d.service, d.region, d.notes,
                   {confidence_sql} AS confidence
            FROM detections d
            {where_sql}
            ORDER BY d.time_utc DESC
            LIMIT ? OFFSET ?
        """,
            params_tuple + (page_size, offset),
        )

        freq_values = [r.get('f_center_hz') for r in rows if r.get('f_center_hz') is not None]
        freq_min = int(min(freq_values)) if freq_values else None
        freq_max = int(max(freq_values)) if freq_values else None
        baseline_bins = load_baseline_bins(freq_min, freq_max)
        annotate_baseline_status(rows, baseline_bins, BASELINE_NEW_THRESHOLD)

        spur_bins = load_spur_bins()
        annotate_near_spur(rows, spur_bins)

        services = [r['service'] for r in qa(con(), "SELECT DISTINCT COALESCE(service,'Unknown') AS service FROM detections ORDER BY service")]
        snr_hist, snr_stats = snr_histogram(con(), filters, bucket_db=3)
        timeline = timeline_metrics(con(), filters)
        top_services_data = top_services(con(), filters, limit=8)

        freq_bounds = q1(con(), f"SELECT MIN(d.f_center_hz) AS fmin, MAX(d.f_center_hz) AS fmax FROM detections d{where_sql}", params_tuple)
        unique_services_row = q1(con(), f"SELECT COUNT(DISTINCT COALESCE(d.service,'Unknown')) AS c FROM detections d{where_sql}", params_tuple) if filtered_total else {"c": 0}
        avg_snr_row = q1(con(), f"SELECT AVG(d.snr_db) AS avg_snr FROM detections d{where_sql}", params_tuple) if filtered_total else {"avg_snr": None}
        newest = q1(con(), f"SELECT d.time_utc FROM detections d{where_sql} ORDER BY d.time_utc DESC LIMIT 1", params_tuple)
        oldest = q1(con(), f"SELECT d.time_utc FROM detections d{where_sql} ORDER BY d.time_utc ASC LIMIT 1", params_tuple)

        summary_cards: List[Dict[str, Any]] = []
        summary_cards.append({
            "label": "Filtered detections",
            "value": f"{filtered_total:,}",
            "subtext": (f"{((filtered_total / detections_total) * 100):.1f}% of database" if detections_total else ""),
        })
        unique_services = int(unique_services_row.get('c') or 0)
        summary_cards.append({
            "label": "Unique services",
            "value": f"{unique_services:,}" if unique_services else "0",
            "subtext": "Based on current filters",
        })
        avg_snr_val = avg_snr_row.get('avg_snr') if avg_snr_row else None
        snr_text = "—"
        if snr_stats and snr_stats.get('p50') is not None:
            snr_text = f"{snr_stats['p50']:.1f} dB"
        elif avg_snr_val is not None:
            snr_text = f"{float(avg_snr_val):.1f} dB"
        summary_cards.append({
            "label": "Median SNR",
            "value": snr_text,
            "subtext": (f"p90 {snr_stats['p90']:.1f} dB" if snr_stats and snr_stats.get('p90') is not None else ""),
        })
        freq_span_text = "No detections"
        if freq_bounds and freq_bounds.get('fmin') is not None and freq_bounds.get('fmax') is not None and freq_bounds['fmax'] > freq_bounds['fmin']:
            fmin_mhz = freq_bounds['fmin'] / 1e6
            fmax_mhz = freq_bounds['fmax'] / 1e6
            freq_span_text = f"{fmin_mhz:.3f}–{fmax_mhz:.3f} MHz"
        summary_cards.append({
            "label": "Frequency span",
            "value": freq_span_text,
            "subtext": "From selected detections",
        })
        window_text = "No time range"
        if newest and newest.get('time_utc') and oldest and oldest.get('time_utc'):
            window_text = f"{format_ts_label(oldest['time_utc'])} → {format_ts_label(newest['time_utc'])}"
        summary_cards.append({
            "label": "Time window",
            "value": window_text,
            "subtext": "UTC",
        })

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

        base_args = {k: v for k, v in args.to_dict(flat=True).items() if v not in (None, '')}

        def build_filter_url(overrides: Dict[str, Any], *, reset: bool = False) -> str:
            query = {} if reset else dict(base_args)
            for key, value in overrides.items():
                if value in (None, ''):
                    query.pop(key, None)
                else:
                    query[key] = str(value)
            query.pop('page', None)
            encoded = urlparse.urlencode(query)
            return url_for('detections') + (f"?{encoded}" if encoded else "")

        quick_range_options = []
        for hours, label in [(1, "1h"), (6, "6h"), (24, "24h"), (168, "7d"), (720, "30d")]:
            quick_range_options.append({
                "label": label,
                "href": build_filter_url({"since_hours": hours}),
                "active": filters.get('since_hours') == hours,
            })

        min_snr_filter = filters.get('min_snr')
        def _snr_active(target: float) -> bool:
            return min_snr_filter is not None and abs(float(min_snr_filter) - float(target)) < 1e-6

        quick_snr_options = []
        for snr_val, label in [(3, "≥3 dB"), (6, "≥6 dB"), (10, "≥10 dB"), (15, "≥15 dB")]:
            quick_snr_options.append({
                "label": label,
                "href": build_filter_url({"min_snr": snr_val}),
                "active": _snr_active(snr_val),
            })

        freq_presets = [
            {"label": "FM 88–108 MHz", "min": 88.0, "max": 108.0},
            {"label": "Airband 118–137 MHz", "min": 118.0, "max": 137.0},
            {"label": "UHF Satcom 240–270 MHz", "min": 240.0, "max": 270.0},
            {"label": "ADS-B 1085–1095 MHz", "min": 1085.0, "max": 1095.0},
        ]
        quick_band_options = []
        for preset in freq_presets:
            min_hz = int(preset['min'] * 1e6)
            max_hz = int(preset['max'] * 1e6)
            quick_band_options.append({
                "label": preset['label'],
                "href": build_filter_url({"f_min_mhz": preset['min'], "f_max_mhz": preset['max']}),
                "active": (filters.get('f_min_hz') == min_hz and filters.get('f_max_hz') == max_hz),
            })

        export_params = {k: v for k, v in base_args.items() if k in {"service", "min_snr", "min_conf", "f_min_mhz", "f_max_mhz", "since_hours"}}
        qs = urlparse.urlencode(export_params)

        return render_template(
            "detections.html",
            rows=rows,
            page_num=page,
            page_size=page_size,
            total=filtered_total,
            detections_total=detections_total,
            services=services,
            qs=qs,
            form_defaults=form_defaults,
            active_filters=active_filters,
            summary_cards=summary_cards,
            snr_hist=snr_hist,
            snr_stats=snr_stats,
            timeline=timeline,
            top_services=top_services_data,
            quick_ranges=quick_range_options,
            quick_snrs=quick_snr_options,
            quick_bands=quick_band_options,
            chart_style_attr=f'style="height:{CHART_HEIGHT_PX}px;"',
            clear_filters_url=build_filter_url({}, reset=True),
            confidence_available=confidence_available,
            spur_hint_available=bool(spur_bins),
            baseline_threshold=BASELINE_NEW_THRESHOLD,
            baseline_hint_available=bool(baseline_bins),
        )

    @app.route('/scans')
    def scans():
        state, state_message = db_state()
        if state != "ready":
            return render_template("db_waiting.html", **db_waiting_context(state, state_message))
        args = request.args
        page = max(1, int(float(args.get('page',1))))
        page_size = min(200, max(10, int(float(args.get('page_size',25)))))
        total = q1(con(), "SELECT COUNT(*) AS c FROM scans")['c']
        offset = (page-1)*page_size
        rows = qa(con(), """
            SELECT id, t_start_utc, t_end_utc, f_start_hz, f_stop_hz, step_hz, samp_rate, fft, avg, device, driver, latitude, longitude
            FROM scans
            ORDER BY COALESCE(t_end_utc,t_start_utc) DESC
            LIMIT ? OFFSET ?
        """, (page_size, offset))
        return render_template(
            "scans.html",
            rows=rows, page_num=page, page_size=page_size, total=total, req_args=args
        )

    @app.route('/baseline')
    def baseline():
        state, state_message = db_state()
        if state != "ready":
            return render_template("db_waiting.html", **db_waiting_context(state, state_message))
        args = request.args
        rows: List[Dict[str,Any]] = []
        if args.get('f_mhz') not in (None,''):
            fmhz = float(args.get('f_mhz'))
            window_khz = int(float(args.get('window_khz', 50)))
            center = int(fmhz*1e6)
            half = int(window_khz*1e3)
            rows = qa(con(), """
                SELECT bin_hz, ema_occ, ema_power_db, last_seen_utc, total_obs, hits
                FROM baseline
                WHERE bin_hz BETWEEN ? AND ?
                ORDER BY bin_hz
            """, (center-half, center+half))
        return render_template("baseline.html", rows=rows, req_args=args)

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

