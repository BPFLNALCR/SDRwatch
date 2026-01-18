"""
HTML view routes blueprint for SDRwatch Web.

Provides template-rendering endpoints for the main dashboard, control panel,
live view, changes page, spur map, and CSV export.
"""
from __future__ import annotations

import io
import sqlite3
from typing import Any, Dict, List, Optional

from flask import Blueprint, Response, current_app, render_template, request

from sdrwatch_web.baseline_helpers import (
    baseline_summary_map,
    change_events_payload,
    controller_baselines,
    fetch_baseline_record,
    hotspots_payload,
    tactical_snapshot_payload,
)
from sdrwatch_web.charts import (
    coverage_heatmap,
    frequency_bins_all_scans_avg,
    frequency_bins_latest_scan,
    snr_histogram,
    strongest_signals,
    timeline_metrics,
    top_services,
)
from sdrwatch_web.config import (
    ACTIVE_SIGNAL_WINDOW_MINUTES,
    CHANGE_EVENT_LIMIT,
    CHANGE_WINDOW_MINUTES,
    CHART_HEIGHT_PX,
    HOTSPOT_BUCKET_COUNT,
    NEW_SIGNAL_WINDOW_MINUTES,
    POWER_SHIFT_THRESHOLD_DB,
    QUIETED_TIMEOUT_MINUTES,
    TACTICAL_RECENT_MINUTES,
)
from sdrwatch_web.controller import controller_profiles
from sdrwatch_web.db import (
    db_state,
    db_waiting_context,
    detections_have_confidence,
    get_con,
    q1,
    qa,
    table_exists,
)
from sdrwatch_web.filters import detection_predicates, parse_detection_filters
from sdrwatch_web.formatting import (
    format_bandwidth_khz,
    format_change_summary,
    format_freq_label,
    format_ts_label,
)
from sdrwatch_web.spur import annotate_near_spur, load_spur_bins

bp = Blueprint("views", __name__)


# ---------------------------------------------------------------------------
# Dashboard (/)
# ---------------------------------------------------------------------------


@bp.route("/")
def dashboard():
    """Main dashboard page."""
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

    con = get_con()
    app = current_app._get_current_object()

    def safe_table_count(table: str) -> int:
        if not table_exists(table):
            return 0
        try:
            row = q1(con, f"SELECT COUNT(*) AS c FROM {table}")
        except sqlite3.OperationalError:
            return 0
        if row is None:
            return 0
        return int(row.get("c") or 0)

    detections_available = table_exists("detections")
    scans_available = table_exists("scans")

    scans_total = safe_table_count("scans")
    detections_total = safe_table_count("detections")
    baseline_total = safe_table_count("baselines")

    filters, form_defaults = parse_detection_filters(request.args, default_since_hours=168)
    confidence_available = detections_have_confidence()
    filters["__confidence_available"] = confidence_available

    snr_bucket_db = 3
    snr_bucket_raw = request.args.get("snr_bucket_db")
    if snr_bucket_raw:
        try:
            snr_bucket_db = max(1, int(float(snr_bucket_raw)))
        except Exception:
            pass

    heatmap_scans = 20
    heatmap_scans_raw = request.args.get("heatmap_scans")
    if heatmap_scans_raw:
        try:
            heatmap_scans = max(5, min(100, int(float(heatmap_scans_raw))))
        except Exception:
            pass

    heatmap_bins = 36
    heatmap_bins_raw = request.args.get("heatmap_bins")
    if heatmap_bins_raw:
        try:
            heatmap_bins = max(12, min(120, int(float(heatmap_bins_raw))))
        except Exception:
            pass

    try:
        baselines = qa(
            con,
            """
            SELECT id, name, created_at, freq_start_hz, freq_stop_hz,
                   bin_hz, total_windows, location_lat, location_lon, notes
            FROM baselines
            ORDER BY id DESC
            LIMIT 12
            """,
        )
    except sqlite3.OperationalError:
        baselines = []

    baseline_summary_data = baseline_summary_map()
    baseline_id_param = (request.args.get("baseline_id") or "").strip()
    selected_baseline: Optional[Dict[str, Any]] = None

    if baselines:
        if baseline_id_param:
            for row in baselines:
                if str(row.get("id")) == baseline_id_param:
                    selected_baseline = row
                    break
        if selected_baseline is None:
            selected_baseline = baselines[0]

    if baseline_id_param and selected_baseline is None:
        try:
            baseline_id_lookup = int(float(baseline_id_param))
        except Exception:
            baseline_id_lookup = None
        if baseline_id_lookup is not None:
            row = fetch_baseline_record(baseline_id_lookup)
            if row:
                selected_baseline = row
                baselines = [row] + [b for b in baselines if b.get("id") != row.get("id")]

    selected_baseline_id = ""
    selected_baseline_id_int: Optional[int] = None
    if selected_baseline and selected_baseline.get("id") is not None:
        try:
            raw_id_value = selected_baseline.get("id")
            if raw_id_value is not None:
                selected_baseline_id_int = int(float(str(raw_id_value)))
                selected_baseline_id = str(selected_baseline_id_int)
        except Exception:
            selected_baseline_id_int = None
            selected_baseline_id = ""

    dashboard_change_filters = [
        {"key": "ALL", "label": "All"},
        {"key": "NEW_SIGNAL", "label": "New"},
        {"key": "POWER_SHIFT", "label": "Power shifts"},
        {"key": "QUIETED", "label": "Quieted"},
    ]
    change_filter_param = (
        request.args.get("change_filter") or request.args.get("type") or "ALL"
    ).strip().upper()
    valid_filter_keys = {opt["key"] for opt in dashboard_change_filters}
    if change_filter_param not in valid_filter_keys:
        change_filter_param = "ALL"
    requested_event_types = None
    if change_filter_param != "ALL":
        requested_event_types = [change_filter_param]

    tactical_payload = None
    hotspot_payload = None
    change_payload_dashboard = None
    signal_cards: List[Dict[str, Any]] = []

    if selected_baseline_id_int is not None:
        tactical_payload = tactical_snapshot_payload(selected_baseline_id_int)
        hotspot_payload = hotspots_payload(selected_baseline_id_int)
        change_payload_dashboard = change_events_payload(
            selected_baseline_id_int,
            window_minutes=CHANGE_WINDOW_MINUTES,
            event_types=requested_event_types,
        )
        if tactical_payload:
            cards_src = tactical_payload.get("active_signals") or []
            signal_cards = [dict(card) for card in cards_src]
            if signal_cards:
                spur_bins = load_spur_bins()
                annotate_near_spur(signal_cards, spur_bins)

    selected_baseline_summary = None
    if selected_baseline_id_int is not None:
        selected_baseline_summary = baseline_summary_data.get(selected_baseline_id_int)

    snapshot_payload = tactical_payload.get("snapshot") if tactical_payload else None
    selected_baseline_last_update = None
    if selected_baseline_summary and selected_baseline_summary.get("last_update_utc"):
        selected_baseline_last_update = selected_baseline_summary.get("last_update_utc")
    elif snapshot_payload and snapshot_payload.get("last_update"):
        selected_baseline_last_update = snapshot_payload.get("last_update")
    elif snapshot_payload and snapshot_payload.get("latest_update"):
        latest_obj = snapshot_payload.get("latest_update") or {}
        if isinstance(latest_obj, dict):
            selected_baseline_last_update = latest_obj.get("timestamp_utc")

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
                    con,
                    f"SELECT COUNT(*) AS c FROM detections d{count_where}",
                    tuple(params_count),
                )
                filtered_detections = int((filtered_row or {}).get("c") or 0)
            except sqlite3.OperationalError:
                filtered_detections = 0
        else:
            filtered_detections = detections_total

        try:
            snr_hist, snr_stats = snr_histogram(con, filters, bucket_db=snr_bucket_db)
        except sqlite3.OperationalError:
            snr_hist, snr_stats = [], None

        if scans_available:
            try:
                freq_bins, latest, freq_max = frequency_bins_latest_scan(con, filters, num_bins=40)
            except sqlite3.OperationalError:
                freq_bins, latest, freq_max = [], None, 0
            try:
                avg_bins, avg_start_mhz, avg_stop_mhz, avg_max = frequency_bins_all_scans_avg(
                    con, filters, num_bins=40
                )
            except sqlite3.OperationalError:
                avg_bins, avg_start_mhz, avg_stop_mhz, avg_max = [], 0.0, 0.0, 0.0
            try:
                timeline = timeline_metrics(con, filters)
            except sqlite3.OperationalError:
                timeline = empty_timeline_payload()
            try:
                heatmap = coverage_heatmap(con, filters, max_scans=heatmap_scans, num_bins=heatmap_bins)
            except sqlite3.OperationalError:
                heatmap = empty_heatmap_payload()

        try:
            services = [
                r["service"]
                for r in qa(
                    con,
                    "SELECT DISTINCT COALESCE(service,'Unknown') AS service FROM detections ORDER BY service",
                )
            ]
        except sqlite3.OperationalError:
            services = []
        try:
            top_services_data = top_services(con, filters, limit=10)
        except sqlite3.OperationalError:
            top_services_data = []
        try:
            strongest = strongest_signals(con, filters, limit=10, include_confidence=confidence_available)
        except sqlite3.OperationalError:
            strongest = []
    else:
        services = []

    active_filters: List[Dict[str, str]] = []
    if filters.get("service"):
        active_filters.append({"label": "Service", "value": str(filters["service"])})
    if filters.get("min_snr") is not None:
        active_filters.append({"label": "Min SNR", "value": f"{filters['min_snr']:.1f} dB"})
    if confidence_available and filters.get("min_conf") is not None:
        active_filters.append({"label": "Confidence", "value": f"≥ {filters['min_conf']:.2f}"})
    if filters.get("f_min_hz") is not None or filters.get("f_max_hz") is not None:
        lo = filters.get("f_min_hz")
        hi = filters.get("f_max_hz")
        if lo is not None and hi is not None:
            active_filters.append({"label": "Freq", "value": f"{lo/1e6:.3f}–{hi/1e6:.3f} MHz"})
        elif lo is not None:
            active_filters.append({"label": "Freq ≥", "value": f"{lo/1e6:.3f} MHz"})
        elif hi is not None:
            active_filters.append({"label": "Freq ≤", "value": f"{hi/1e6:.3f} MHz"})
    if filters.get("since_hours"):
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
        confidence_available=confidence_available,
        db_status="ready",
        db_status_message="",
        db_path=app._db_path,
        tactical_config=tactical_config,
        baselines=baselines,
        baseline_summaries=baseline_summary_data,
        selected_baseline=selected_baseline,
        selected_baseline_id=selected_baseline_id,
        selected_baseline_summary=selected_baseline_summary,
        selected_baseline_last_update=selected_baseline_last_update,
        selected_tactical=tactical_payload,
        selected_snapshot=snapshot_payload,
        selected_hotspots=hotspot_payload,
        signal_cards=signal_cards,
        dashboard_change_payload=change_payload_dashboard,
        dashboard_change_filter=change_filter_param,
        dashboard_change_filters=dashboard_change_filters,
        change_window_minutes=CHANGE_WINDOW_MINUTES,
        format_ts_label=format_ts_label,
        format_freq_label=format_freq_label,
        format_bandwidth_khz=format_bandwidth_khz,
        format_change_summary=format_change_summary,
    )

    if request.headers.get("HX-Request"):
        return render_template("partials/dashboard_content.html", **context)

    return render_template("dashboard.html", **context)


# ---------------------------------------------------------------------------
# Control panel (/control)
# ---------------------------------------------------------------------------


@bp.get("/control")
def control():
    """Scan control panel page."""
    app = current_app._get_current_object()
    baseline_list = controller_baselines()
    summaries = baseline_summary_map()
    return render_template(
        "control.html",
        db_path=app._db_path,
        profiles=controller_profiles(),
        baselines=baseline_list,
        baseline_summaries=summaries,
    )


# ---------------------------------------------------------------------------
# Live view (/live)
# ---------------------------------------------------------------------------


@bp.route("/live")
def live():
    """Live scan visualization page."""
    app = current_app._get_current_object()
    state, state_message = db_state()
    return render_template(
        "live.html",
        db_status=state,
        db_status_message=state_message,
        db_path=app._db_path,
    )


# ---------------------------------------------------------------------------
# Changes (/changes)
# ---------------------------------------------------------------------------


@bp.route("/changes")
def changes():
    """Change events page."""
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
        context.update({
            "change_config": change_config,
            "baselines": [],
            "selected_baseline": None,
            "selected_baseline_id": "",
            "change_payload": None,
            "active_filter": "ALL",
        })
        return render_template("changes.html", **context)

    con = get_con()
    baselines = qa(
        con,
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
            active_filter="ALL",
            change_config=change_config,
            db_status="ready",
            db_status_message="",
        )

    baseline_id_param = (request.args.get("baseline_id") or "").strip()
    selected_baseline: Optional[Dict[str, Any]] = None

    if baseline_id_param:
        for row in baselines:
            if str(row.get("id")) == baseline_id_param:
                selected_baseline = row
                break
    if selected_baseline is None:
        selected_baseline = baselines[0]

    selected_baseline_id = ""
    selected_baseline_id_int: Optional[int] = None
    if selected_baseline and selected_baseline.get("id") is not None:
        try:
            raw_id = selected_baseline.get("id")
            selected_baseline_id_int = int(raw_id) if raw_id is not None else None
            if selected_baseline_id_int is not None:
                selected_baseline_id = str(selected_baseline_id_int)
        except Exception:
            pass

    type_param = (request.args.get("type") or "ALL").strip().upper()
    valid_types = {"ALL", "NEW_SIGNAL", "QUIETED", "POWER_SHIFT"}
    if type_param not in valid_types:
        type_param = "ALL"

    change_payload = None
    if selected_baseline_id_int is not None:
        event_types = [type_param] if type_param != "ALL" else None
        change_payload = change_events_payload(
            selected_baseline_id_int,
            window_minutes=CHANGE_WINDOW_MINUTES,
            event_types=event_types,
        )

    return render_template(
        "changes.html",
        baselines=baselines,
        selected_baseline=selected_baseline,
        selected_baseline_id=selected_baseline_id,
        change_payload=change_payload,
        active_filter=type_param,
        change_config=change_config,
        db_status="ready",
        db_status_message="",
        format_ts_label=format_ts_label,
        format_freq_label=format_freq_label,
        format_bandwidth_khz=format_bandwidth_khz,
        format_change_summary=format_change_summary,
    )


# ---------------------------------------------------------------------------
# Spur map (/spur-map)
# ---------------------------------------------------------------------------


@bp.route("/spur-map")
def spur_map():
    """Spur calibration map page."""
    app = current_app._get_current_object()
    state, state_message = db_state()

    if state != "ready":
        context = db_waiting_context(state, state_message)
        context["spur_entries"] = []
        return render_template("spur_map.html", **context)

    con = get_con()
    try:
        rows = qa(
            con,
            """
            SELECT bin_hz, mean_power_db, hits, last_seen_utc
            FROM spur_map
            ORDER BY bin_hz
            """,
        )
    except sqlite3.OperationalError:
        rows = []

    return render_template(
        "spur_map.html",
        spur_entries=rows,
        db_status="ready",
        db_status_message="",
        db_path=app._db_path,
        format_ts_label=format_ts_label,
    )


# ---------------------------------------------------------------------------
# CSV export (/export/detections.csv)
# ---------------------------------------------------------------------------


@bp.get("/export/detections.csv")
def export_csv():
    """Export detections as CSV."""
    state, _ = db_state()
    if state != "ready":
        return Response("Database not ready", mimetype="text/plain", status=503)

    con = get_con()
    filters, _ = parse_detection_filters(request.args, default_since_hours=168)
    confidence_available = detections_have_confidence()
    filters["__confidence_available"] = confidence_available

    conds, params = detection_predicates(filters, alias="d")
    where_sql = " WHERE " + " AND ".join(conds) if conds else ""
    confidence_sql = "d.confidence" if confidence_available else "NULL"

    try:
        rows = qa(
            con,
            f"""
            SELECT time_utc, scan_id, f_center_hz, f_low_hz, f_high_hz,
                   peak_db, noise_db, snr_db, service, region, notes,
                   {confidence_sql} AS confidence
            FROM detections d {where_sql}
            ORDER BY time_utc DESC
            LIMIT 100000
            """,
            tuple(params),
        )
    except sqlite3.OperationalError:
        return Response("Query failed", mimetype="text/plain", status=500)

    import csv

    buf = io.StringIO()
    fieldnames = [
        "time_utc",
        "scan_id",
        "f_center_hz",
        "f_low_hz",
        "f_high_hz",
        "peak_db",
        "noise_db",
        "snr_db",
        "service",
        "region",
        "notes",
        "confidence",
    ]
    w = csv.DictWriter(buf, fieldnames=fieldnames)
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fieldnames})
    buf.seek(0)

    return Response(
        buf.read(),
        mimetype="text/csv",
        headers={"Content-Disposition": "attachment; filename=detections.csv"},
    )
