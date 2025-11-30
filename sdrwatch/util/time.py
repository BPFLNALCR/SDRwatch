"""Time utilities shared across SDRwatch components."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now_str() -> str:
    """Return the current UTC timestamp as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()
