"""
Authentication helpers for SDRwatch Web.

Provides the require_auth decorator/function for protecting API endpoints
with bearer token authentication.
"""
from __future__ import annotations

from flask import abort, request

from sdrwatch_web.config import API_TOKEN


def require_auth() -> None:
    """
    Check bearer token authentication for the current request.

    If API_TOKEN is not set, authentication is disabled (open access).
    Otherwise, the request must include a valid Authorization header.

    Raises:
        werkzeug.exceptions.Unauthorized: If token is invalid or missing.
    """
    if not API_TOKEN:
        return

    hdr = request.headers.get("Authorization", "")
    if hdr != f"Bearer {API_TOKEN}":
        abort(401)
