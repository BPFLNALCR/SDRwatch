"""
Debug and observability API blueprint for SDRwatch Web.

Provides endpoints for health checks, database stats, configuration info,
and error tracking. Also includes the /debug dashboard page.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List

from flask import Blueprint, current_app, jsonify, render_template, request

from sdrwatch_web.auth import require_auth
from sdrwatch_web.config import (
    ACTIVE_SIGNAL_WINDOW_MINUTES,
    API_TOKEN,
    CHANGE_EVENT_LIMIT,
    CHANGE_WINDOW_MINUTES,
    CONTROL_TOKEN,
    CONTROL_URL,
    HOTSPOT_BUCKET_COUNT,
    NEW_SIGNAL_WINDOW_MINUTES,
    POWER_SHIFT_THRESHOLD_DB,
    QUIETED_MIN_WINDOWS,
    QUIETED_TIMEOUT_MINUTES,
    TACTICAL_RECENT_MINUTES,
    BAND_SUMMARY_MAX_BANDS,
    BAND_SUMMARY_TARGET_WIDTH_HZ,
)
from sdrwatch_web.controller import get_controller
from sdrwatch_web.db import get_con_optional, q1, qa

bp = Blueprint("api_debug", __name__)


# ---------------------------------------------------------------------------
# In-memory error ring buffer
# ---------------------------------------------------------------------------

_error_ring: List[Dict[str, Any]] = []
_error_ring_max = 100


def capture_error(exc: Exception, path: str, method: str) -> None:
    """
    Capture an exception to the error ring buffer.

    Args:
        exc: The exception that occurred.
        path: Request path.
        method: HTTP method.
    """
    import traceback as tb

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
        "path": path,
        "method": method,
        "error": str(exc),
        "type": type(exc).__name__,
        "traceback": tb.format_exc(),
    }
    _error_ring.append(entry)
    while len(_error_ring) > _error_ring_max:
        _error_ring.pop(0)


def get_error_ring() -> List[Dict[str, Any]]:
    """Get a copy of the error ring buffer."""
    return list(_error_ring)


def clear_error_ring() -> None:
    """Clear the error ring buffer."""
    _error_ring.clear()


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@bp.get("/api/debug/health")
def api_debug_health():
    """Health check endpoint: DB connectivity, controller reachability, WAL size."""
    require_auth()
    app = current_app._get_current_object()

    health: Dict[str, Any] = {
        "status": "ok",
        "db": "unknown",
        "controller": "unknown",
        "wal_size_bytes": None,
    }

    # Check DB
    try:
        connection = get_con_optional()
        if connection is not None:
            connection.execute("SELECT 1")
            health["db"] = "connected"
        else:
            health["db"] = "disconnected"
            health["status"] = "degraded"
    except Exception as exc:
        health["db"] = f"error: {exc}"
        health["status"] = "degraded"

    # Check WAL size
    try:
        db_dir = os.path.dirname(os.path.abspath(app._db_path))
        db_name = os.path.basename(app._db_path)
        wal_path = os.path.join(db_dir, db_name + "-wal")
        if os.path.exists(wal_path):
            health["wal_size_bytes"] = os.path.getsize(wal_path)
    except Exception:
        pass

    # Check controller
    try:
        ctl = get_controller()
        ctl.list_jobs()
        health["controller"] = "reachable"
    except Exception as exc:
        health["controller"] = f"error: {exc}"
        health["status"] = "degraded"

    return jsonify(health)


# ---------------------------------------------------------------------------
# Database stats
# ---------------------------------------------------------------------------


@bp.get("/api/debug/db-stats")
def api_debug_db_stats():
    """Database statistics: table row counts, recent scan_updates timing."""
    require_auth()

    stats: Dict[str, Any] = {
        "tables": {},
        "recent_scans": [],
    }

    try:
        connection = get_con_optional()
        if connection is None:
            return jsonify({"error": "db unavailable"}), 503

        # Table row counts
        tables = [
            "baselines",
            "baseline_detections",
            "baseline_noise",
            "baseline_occupancy",
            "scan_updates",
            "spur_map",
            "baseline_band_summary",
            "baseline_snapshot",
        ]
        for table in tables:
            try:
                row = q1(connection, f"SELECT COUNT(*) AS cnt FROM {table}")
                stats["tables"][table] = int(row.get("cnt", 0)) if row else 0
            except sqlite3.OperationalError:
                stats["tables"][table] = None  # Table doesn't exist

        # Recent scan_updates (last 20)
        try:
            rows = qa(
                connection,
                """
                SELECT id, baseline_id, timestamp_utc, num_hits, num_segments,
                       num_new_signals, num_revisits, num_confirmed, num_false_positive,
                       duration_ms
                FROM scan_updates
                ORDER BY id DESC
                LIMIT 20
                """,
            )
            stats["recent_scans"] = rows
        except sqlite3.OperationalError:
            # duration_ms column might not exist yet
            try:
                rows = qa(
                    connection,
                    """
                    SELECT id, baseline_id, timestamp_utc, num_hits, num_segments,
                           num_new_signals, num_revisits, num_confirmed, num_false_positive
                    FROM scan_updates
                    ORDER BY id DESC
                    LIMIT 20
                    """,
                )
                stats["recent_scans"] = rows
            except sqlite3.OperationalError:
                stats["recent_scans"] = []
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    return jsonify(stats)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@bp.get("/api/debug/config")
def api_debug_config():
    """Runtime configuration: environment variables and constants."""
    require_auth()
    app = current_app._get_current_object()

    config: Dict[str, Any] = {
        "env": {
            "SDRWATCH_CONTROL_URL": CONTROL_URL,
            "SDRWATCH_CONTROL_TOKEN": "***" if CONTROL_TOKEN else "(not set)",
            "SDRWATCH_TOKEN": "***" if API_TOKEN else "(not set)",
            "SDRWATCH_DEBUG": os.environ.get("SDRWATCH_DEBUG", "(not set)"),
            "SDRWATCH_LOG_LEVEL": os.environ.get("SDRWATCH_LOG_LEVEL", "(not set)"),
        },
        "constants": {
            "TACTICAL_RECENT_MINUTES": TACTICAL_RECENT_MINUTES,
            "ACTIVE_SIGNAL_WINDOW_MINUTES": ACTIVE_SIGNAL_WINDOW_MINUTES,
            "HOTSPOT_BUCKET_COUNT": HOTSPOT_BUCKET_COUNT,
            "CHANGE_WINDOW_MINUTES": CHANGE_WINDOW_MINUTES,
            "NEW_SIGNAL_WINDOW_MINUTES": NEW_SIGNAL_WINDOW_MINUTES,
            "POWER_SHIFT_THRESHOLD_DB": POWER_SHIFT_THRESHOLD_DB,
            "QUIETED_TIMEOUT_MINUTES": QUIETED_TIMEOUT_MINUTES,
            "QUIETED_MIN_WINDOWS": QUIETED_MIN_WINDOWS,
            "CHANGE_EVENT_LIMIT": CHANGE_EVENT_LIMIT,
            "BAND_SUMMARY_MAX_BANDS": BAND_SUMMARY_MAX_BANDS,
            "BAND_SUMMARY_TARGET_WIDTH_HZ": BAND_SUMMARY_TARGET_WIDTH_HZ,
        },
        "db_path": app._db_path,
    }
    return jsonify(config)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


@bp.get("/api/debug/errors")
def api_debug_errors():
    """Recent captured errors from the ring buffer."""
    require_auth()

    limit = request.args.get("limit", type=int) or 50
    limit = max(1, min(limit, _error_ring_max))
    errors = list(reversed(_error_ring[-limit:]))
    return jsonify({"errors": errors, "total_captured": len(_error_ring)})


# ---------------------------------------------------------------------------
# Debug dashboard page
# ---------------------------------------------------------------------------


@bp.get("/debug")
def debug_page():
    """Debug dashboard page."""
    require_auth()
    app = current_app._get_current_object()
    return render_template("debug.html", db_path=app._db_path)
