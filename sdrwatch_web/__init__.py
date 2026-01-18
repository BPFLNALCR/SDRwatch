"""
SDRwatch Web — Flask application for baseline-centric spectrum monitoring.

This package provides the Flask web interface that:
- Queries SQLite for baseline dashboards and detection history
- Proxies control actions (jobs/logs/devices) to sdrwatch-control
- Renders HTML views with HTMX-powered interactivity

Usage:
    from sdrwatch_web import create_app
    app = create_app(db_path="sdrwatch.db")
    app.run(host="0.0.0.0", port=8080)
"""
from __future__ import annotations

__version__ = "0.1.0"

# Import create_app so it's accessible from package root
from sdrwatch_web.app import create_app

__all__ = ["create_app", "__version__"]
