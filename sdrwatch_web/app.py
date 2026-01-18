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
    app._db_path = db_path
    app._db_error = None
    app._con = None
    app._has_confidence_column = None
    app._table_columns_cache = {}
    app._table_exists_cache = {}
    app._ctl = ControllerClient(CONTROL_URL, CONTROL_TOKEN)

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
        if app._con is not None:
            return app._con
        try:
            app._con = _open_db_ro(app._db_path)
            app._db_error = None
            app._has_confidence_column = None
        except Exception as exc:
            app._con = None
            app._db_error = str(exc)
            app._has_confidence_column = None
        return app._con

    def _reset_con() -> None:
        """Close and discard the cached DB connection."""
        if app._con is not None:
            try:
                app._con.close()
            except Exception:
                pass
        app._con = None
        app._table_columns_cache = {}
        app._table_exists_cache = {}

    # Attach helpers for external access
    app._ensure_con = _ensure_con
    app._reset_con = _reset_con

    # Attempt initial connection (tolerates failure)
    _ensure_con()

    # ------------------------------------------------------------------
    # Error ring buffer (exposed via api_debug blueprint)
    # ------------------------------------------------------------------
    app._error_ring = []
    app._error_ring_max = 100

    # ------------------------------------------------------------------
    # Request timing middleware
    # ------------------------------------------------------------------

    @app.before_request
    def log_request_start():
        request._start_time = perf_counter()

    @app.after_request
    def log_request_end(response):
        if hasattr(request, "_start_time"):
            duration_ms = (perf_counter() - request._start_time) * 1000
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
        app._error_ring.append(entry)
        while len(app._error_ring) > app._error_ring_max:
            app._error_ring.pop(0)
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
