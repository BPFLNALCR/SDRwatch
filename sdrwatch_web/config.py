"""
Configuration constants and environment parsing for SDRwatch Web.

All SDRWATCH_* environment variables are parsed here and exported as module-level
constants. Blueprints and helpers import from this module rather than reading
os.environ directly.
"""
from __future__ import annotations

import os
from typing import Any


def _int_env(name: str, default: int) -> int:
    """Parse an integer from environment, returning default on missing/invalid."""
    val = os.getenv(name)
    if not val:
        return default
    try:
        return max(1, int(float(val)))
    except Exception:
        return default


def _float_env(name: str, default: float) -> float:
    """Parse a float from environment, returning default on missing/invalid."""
    val = os.getenv(name)
    if not val:
        return default
    try:
        return float(val)
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Authentication tokens
# ---------------------------------------------------------------------------
API_TOKEN: str = os.getenv("SDRWATCH_TOKEN", "")
"""Optional bearer token protecting web /api/* endpoints."""

CONTROL_URL: str = os.getenv("SDRWATCH_CONTROL_URL", "http://127.0.0.1:8765")
"""Base URL of the sdrwatch-control HTTP API."""

CONTROL_TOKEN: str = os.getenv("SDRWATCH_CONTROL_TOKEN", "") or os.getenv("SDRWATCH_TOKEN", "")
"""Bearer token for authenticating to the controller (falls back to API_TOKEN)."""


# ---------------------------------------------------------------------------
# Baseline & tactical thresholds
# ---------------------------------------------------------------------------
BASELINE_NEW_THRESHOLD: float = 0.2
"""Occupancy fraction below which a signal is considered NEW."""

TACTICAL_RECENT_MINUTES: int = _int_env("SDRWATCH_TACTICAL_RECENT_MINUTES", 30)
"""Window for counting recent NEW signals in tactical snapshot."""

ACTIVE_SIGNAL_WINDOW_MINUTES: int = _int_env("SDRWATCH_ACTIVE_SIGNAL_MINUTES", 15)
"""Window for listing active signals in tactical view."""

HOTSPOT_BUCKET_COUNT: int = _int_env("SDRWATCH_HOTSPOT_BUCKETS", 60)
"""Number of frequency buckets for hotspot heatmap."""


# ---------------------------------------------------------------------------
# Change events
# ---------------------------------------------------------------------------
CHANGE_WINDOW_MINUTES: int = _int_env("SDRWATCH_CHANGE_WINDOW_MINUTES", 60)
"""Lookback window for change events (NEW, QUIETED, POWER_SHIFT)."""

NEW_SIGNAL_WINDOW_MINUTES: int = _int_env("SDRWATCH_NEW_SIGNAL_MINUTES", CHANGE_WINDOW_MINUTES)
"""Lookback for NEW_SIGNAL events specifically."""

POWER_SHIFT_THRESHOLD_DB: float = _float_env("SDRWATCH_POWER_SHIFT_THRESHOLD_DB", 6.0)
"""Minimum dB delta to emit a POWER_SHIFT event."""

QUIETED_TIMEOUT_MINUTES: int = _int_env("SDRWATCH_QUIETED_TIMEOUT_MINUTES", 15)
"""Minutes since last_seen to consider a signal quieted."""

QUIETED_MIN_WINDOWS: int = _int_env("SDRWATCH_QUIETED_MIN_WINDOWS", 20)
"""Minimum total_windows before a detection can be marked QUIETED."""

CHANGE_EVENT_LIMIT: int = _int_env("SDRWATCH_CHANGE_EVENT_LIMIT", 120)
"""Maximum change events returned per request."""


# ---------------------------------------------------------------------------
# Band summary
# ---------------------------------------------------------------------------
BAND_SUMMARY_MAX_BANDS: int = _int_env("SDRWATCH_BAND_SUMMARY_MAX_BANDS", 6)
"""Maximum number of bands in a baseline band summary."""

BAND_SUMMARY_TARGET_WIDTH_HZ: float = _float_env("SDRWATCH_BAND_SUMMARY_TARGET_WIDTH_HZ", 10_000_000.0)
"""Target width per band in Hz."""

BAND_SUMMARY_RECENT_MINUTES: int = _int_env("SDRWATCH_BAND_SUMMARY_RECENT_MINUTES", 30)
"""Lookback for recent activity in band summary."""

BAND_SUMMARY_OCC_THRESHOLD: float = _float_env("SDRWATCH_BAND_SUMMARY_OCC_THRESHOLD", BASELINE_NEW_THRESHOLD)
"""Occupancy threshold for band summary classification."""


# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------
CHART_HEIGHT_PX: int = 160
"""Default height in pixels for chart elements."""
