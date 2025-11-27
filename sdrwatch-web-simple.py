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
    }
    form_defaults: Dict[str, str] = {
        "service": "",
        "min_snr": "",
        "f_min_mhz": "",
        "f_max_mhz": "",
        "since_hours": "" if default_since_hours is None else str(default_since_hours),
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

def strongest_signals(con: sqlite3.Connection, filters: Dict[str, Any], limit: int = 10):
    conds, params = detection_predicates(filters, alias="d")
    conds.append("d.snr_db IS NOT NULL")
    where_sql = " WHERE " + " AND ".join(conds)
    params_tuple = tuple(params)
    return qa(
        con,
        f"""
        SELECT d.f_center_hz, d.snr_db, d.service
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

    def start_job(self, device_key: str, label: str, params: Dict[str, Any]):
        return self._req('POST', '/jobs', body={"device_key": device_key, "label": label, "params": params})

    def job_detail(self, job_id: str):
        return self._req('GET', f'/jobs/{job_id}')

    def stop_job(self, job_id: str):
        return self._req('DELETE', f'/jobs/{job_id}')

    def job_logs(self, job_id: str, tail: Optional[int] = None) -> str:
        params = {"tail": int(tail)} if tail else None
        return self._req('GET', f'/jobs/{job_id}/logs', params=params, want_text=True)

# ================================
# Flask app
# ================================

def create_app(db_path: str) -> Flask:
    app = Flask(__name__, template_folder='templates', static_folder='static')
    app._db_path = db_path
    app._db_error: Optional[str] = None
    app._con: Optional[sqlite3.Connection] = None
    app._ctl = ControllerClient(CONTROL_URL, CONTROL_TOKEN)

    def _ensure_con() -> Optional[sqlite3.Connection]:
        if app._con is not None:
            return app._con
        try:
            app._con = open_db_ro(app._db_path)
            app._db_error = None
        except Exception as exc:
            app._con = None
            app._db_error = str(exc)
        return app._con

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

    def start_job_from_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
        device_key = payload.get('device_key')
        if not device_key:
            abort(400, description='device_key is required')
        label = payload.get('label') or 'web'
        params = payload.get('params') or {}
        try:
            return app._ctl.start_job(device_key, label, params)
        except Exception as exc:
            abort(400, description=str(exc))

    def stop_job_by_id(job_id: str) -> Dict[str, Any]:
        try:
            return app._ctl.stop_job(job_id)
        except Exception as exc:
            abort(400, description=str(exc))

    def job_logs_response(job_id: str, tail: Optional[int] = None) -> Response:
        try:
            data = app._ctl.job_logs(job_id, tail=tail)
            return Response(data or "", mimetype='text/plain')
        except Exception as exc:
            abort(404, description=str(exc))

    # ---------- Pages ----------
    @app.get('/control')
    def control():
        return render_template("control.html", db_path=app._db_path)

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
        strongest = strongest_signals(con(), filters, limit=10)

        active_filters: List[Dict[str, str]] = []
        if filters.get('service'):
            active_filters.append({"label": "Service", "value": str(filters['service'])})
        if filters.get('min_snr') is not None:
            active_filters.append({"label": "Min SNR", "value": f"{filters['min_snr']:.1f} dB"})
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
            db_status="ready",
            db_status_message="",
            db_path=app._db_path,
        )

        if request.headers.get("HX-Request"):
            return render_template("partials/dashboard_content.html", **context)

        return render_template("dashboard.html", **context)

    @app.route('/detections')
    def detections():
        state, state_message = db_state()
        if state != "ready":
            return render_template("db_waiting.html", **db_waiting_context(state, state_message))
        args = request.args
        filters, form_defaults = parse_detection_filters(args)
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
                   d.peak_db, d.noise_db, d.snr_db, d.service, d.region, d.notes
            FROM detections d
            {where_sql}
            ORDER BY d.time_utc DESC
            LIMIT ? OFFSET ?
        """,
            params_tuple + (page_size, offset),
        )

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

        export_params = {k: v for k, v in base_args.items() if k in {"service", "min_snr", "f_min_mhz", "f_max_mhz", "since_hours"}}
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
        if args.get('f_min_mhz'): where.append("f_center_hz >= ?"); params.append(int(float(args.get('f_min_mhz'))*1e6))
        if args.get('f_max_mhz'): where.append("f_center_hz <= ?"); params.append(int(float(args.get('f_max_mhz'))*1e6))
        if args.get('since_hours'): where.append("time_utc >= datetime('now', ?)"); params.append(f"-{int(float(args.get('since_hours')))} hours")
        where_sql = (" WHERE "+" AND ".join(where)) if where else ""
        rows = qa(con(), f"""
            SELECT time_utc, scan_id, f_center_hz, f_low_hz, f_high_hz,
                   peak_db, noise_db, snr_db, service, region, notes
            FROM detections {where_sql}
            ORDER BY time_utc DESC
            LIMIT 100000
        """, tuple(params))
        import csv
        buf = io.StringIO()
        fieldnames = ["time_utc","scan_id","f_center_hz","f_low_hz","f_high_hz","peak_db","noise_db","snr_db","service","region","notes"]
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

    def active_state_payload() -> Dict[str, Any]:
        job = controller_active_job()
        if not job:
            return {"state": "idle"}
        return {"state": "running", "job": job}

    def start_job_response():
        require_auth()
        payload = request.get_json(force=True, silent=False) or {}
        job = start_job_from_payload(payload)
        return jsonify({"state": job.get('status', 'running'), "job": job})

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

