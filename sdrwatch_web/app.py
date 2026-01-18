"""
Application factory for SDRwatch Web.

Wires together blueprints, DB lifecycle, error handling, and request middleware.
"""
from __future__ import annotations

import os
import sqlite3
import traceback as tb
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional, Set

from flask import Flask, g, request

from sdrwatch_web.config import CONTROL_TOKEN, CONTROL_URL
from sdrwatch_web.controller import ControllerClient


def create_app(db_path: str) -> Flask:
    """Create and configure the Flask application."""
    app = Flask(
        __name__,
        template_folder=os.path.join(os.path.dirname(__file__), "..", "templates"),
        static_folder=os.path.join(os.path.dirname(__file__), "..", "static"),
    )

    # ------------------------------------------------------------------
    # App-level state
    # ------------------------------------------------------------------
    app.config["DB_PATH"] = db_path
    app.config["DB_ERROR"] = None
    app.config["DB_CONNECTION"] = None
    app.config["HAS_CONFIDENCE_COLUMN"] = None
    app.config["TABLE_COLUMNS_CACHE"] = {}
    app.config["TABLE_EXISTS_CACHE"] = {}
    app.config["CONTROLLER_CLIENT"] = ControllerClient(CONTROL_URL, CONTROL_TOKEN)

    # ------------------------------------------------------------------
    # DB lifecycle helpers (attached to app for shared use)
    # ------------------------------------------------------------------

    def _open_db_ro(path: str) -> sqlite3.Connection:
        """Open SQLite in read-only mode with dict row factory."""
        abspath = os.path.abspath(path)
        con = sqlite3.connect(
            f"file:{abspath}?mode=ro", uri=True, check_same_thread=False
        )
        con.execute("PRAGMA busy_timeout=2000;")
        con.row_factory = lambda cur, row: {
            d[0]: row[i] for i, d in enumerate(cur.description)
        }
        return con

    def _ensure_con() -> Optional[sqlite3.Connection]:
        """Lazy-open and cache the read-only DB connection."""
        if app.config["DB_CONNECTION"] is not None:
            return app.config["DB_CONNECTION"]
        try:
            app.config["DB_CONNECTION"] = _open_db_ro(app.config["DB_PATH"])
            app.config["DB_ERROR"] = None
            app.config["HAS_CONFIDENCE_COLUMN"] = None
        except Exception as exc:
            app.config["DB_CONNECTION"] = None
            app.config["DB_ERROR"] = str(exc)
            app.config["HAS_CONFIDENCE_COLUMN"] = None
        return app.config["DB_CONNECTION"]

    def _reset_con() -> None:
        """Close and discard the cached DB connection."""
        if app.config["DB_CONNECTION"] is not None:
            try:
                app.config["DB_CONNECTION"].close()
            except Exception:
                pass
        app.config["DB_CONNECTION"] = None
        app.config["TABLE_COLUMNS_CACHE"] = {}
        app.config["TABLE_EXISTS_CACHE"] = {}

    # Attach helpers for external access via extensions dict
    if "sdrwatch_helpers" not in app.extensions:
        app.extensions["sdrwatch_helpers"] = {}
    app.extensions["sdrwatch_helpers"]["ensure_con"] = _ensure_con
    app.extensions["sdrwatch_helpers"]["reset_con"] = _reset_con

    # Attempt initial connection (tolerates failure)
    _ensure_con()

    # ------------------------------------------------------------------
    # Error ring buffer (exposed via api_debug blueprint)
    # ------------------------------------------------------------------
    app.config["ERROR_RING"] = []
    app.config["ERROR_RING_MAX"] = 100

    # ------------------------------------------------------------------
    # Request timing middleware
    # ------------------------------------------------------------------

    @app.before_request
    def log_request_start():
        g.start_time = perf_counter()

    @app.after_request
    def log_request_end(response):
        if hasattr(g, "start_time"):
            duration_ms = (perf_counter() - g.start_time) * 1000
            # Log slow requests (>500ms) or errors at debug level
            if duration_ms > 500 or response.status_code >= 400:
                app.logger.debug(
                    "%s %s -> %d (%.1fms)",
                    request.method,
                    request.path,
                    response.status_code,
                    duration_ms,
                )
        return response

    # ------------------------------------------------------------------
    # Global error handler -> ring buffer
    # ------------------------------------------------------------------

    @app.errorhandler(Exception)
    def capture_error_to_ring(exc):
        entry = {
            "ts": datetime.now(timezone.utc)
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z"),
            "path": request.path,
            "method": request.method,
            "error": str(exc),
            "type": type(exc).__name__,
            "traceback": tb.format_exc(),
        }
        app.config["ERROR_RING"].append(entry)
        while len(app.config["ERROR_RING"]) > app.config["ERROR_RING_MAX"]:
            app.config["ERROR_RING"].pop(0)
        # Re-raise to let Flask handle normally
        raise exc

    # ------------------------------------------------------------------
    # Register blueprints
    # ------------------------------------------------------------------
    from sdrwatch_web.blueprints.api_baselines import bp as api_baselines_bp
    from sdrwatch_web.blueprints.api_debug import bp as api_debug_bp
    from sdrwatch_web.blueprints.api_jobs import bp as api_jobs_bp
    from sdrwatch_web.blueprints.ctl import bp as ctl_bp
    from sdrwatch_web.blueprints.views import bp as views_bp

    app.register_blueprint(api_debug_bp)
    app.register_blueprint(api_jobs_bp)
    app.register_blueprint(api_baselines_bp)
    app.register_blueprint(ctl_bp)
    app.register_blueprint(views_bp)

    return app
