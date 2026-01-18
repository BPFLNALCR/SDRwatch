#!/usr/bin/env python3
"""
SDRwatch Web — Entry point.

Thin CLI shim that parses arguments and runs the Flask application.

Run:
    python sdrwatch-web.py --db sdrwatch.db --host 0.0.0.0 --port 8080

Environment:
    SDRWATCH_TOKEN            Protect web /api/* endpoints (optional)
    SDRWATCH_CONTROL_URL      Controller URL (default http://127.0.0.1:8765)
    SDRWATCH_CONTROL_TOKEN    Auth token for controller (fallback: SDRWATCH_TOKEN)
"""
from __future__ import annotations

import argparse


def parse_args():
    ap = argparse.ArgumentParser(
        description="SDRwatch Web — baseline-centric spectrum monitoring UI"
    )
    ap.add_argument(
        "--db",
        required=True,
        help="Path to the SQLite database file (sdrwatch.db)",
    )
    ap.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the web server (default: 0.0.0.0)",
    )
    ap.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)",
    )
    return ap.parse_args()


def main():
    args = parse_args()

    from sdrwatch_web import create_app

    app = create_app(args.db)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
