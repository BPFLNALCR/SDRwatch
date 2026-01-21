"""
Application factory for SDRwatch Web.

Wires together blueprints, DB lifecycle, error handling, and request middleware.
"""
from __future__ import annotations

import os
import traceback as tb
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Dict, List, Optional, Set

from flask import Flask, g, request

from sdrwatch_web.config import CONTROL_TOKEN, CONTROL_URL
from sdrwatch_web.controller import ControllerClient
from sdrwatch_web.db import init_db


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
    # Initialize database state using the shared db module
    init_db(app, db_path)

    # Legacy keys for backward compatibility with existing code
    app.config["DB_PATH"] = db_path
    app.config["DB_ERROR"] = app.config.get("SDRWATCH_DB_ERROR")
    app.config["DB_CONNECTION"] = app.config.get("SDRWATCH_DB_CON")
    app.config["HAS_CONFIDENCE_COLUMN"] = app.config.get("SDRWATCH_DB_HAS_CONFIDENCE")
    app.config["TABLE_COLUMNS_CACHE"] = app.config.get("SDRWATCH_DB_COLUMNS_CACHE", {})
    app.config["TABLE_EXISTS_CACHE"] = app.config.get("SDRWATCH_DB_EXISTS_CACHE", {})
    app.config["CONTROLLER_CLIENT"] = ControllerClient(CONTROL_URL, CONTROL_TOKEN)

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
    from sdrwatch_web.blueprints.api_signals import bp as api_signals_bp
    from sdrwatch_web.blueprints.ctl import bp as ctl_bp
    from sdrwatch_web.blueprints.views import bp as views_bp

    app.register_blueprint(api_debug_bp)
    app.register_blueprint(api_jobs_bp)
    app.register_blueprint(api_baselines_bp)
    app.register_blueprint(api_signals_bp)
    app.register_blueprint(ctl_bp)
    app.register_blueprint(views_bp)

    return app
