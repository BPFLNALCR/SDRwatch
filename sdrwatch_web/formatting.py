"""
Display formatting helpers for SDRwatch Web.

Provides functions for formatting frequencies, timestamps, bandwidths,
and other values for human-readable display in templates.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional


def format_ts_label(ts: Optional[str]) -> str:
    """
    Format an ISO timestamp for display (e.g., "Jan 15 14:30").

    Args:
        ts: ISO 8601 timestamp string, or None.

    Returns:
        Formatted timestamp or "—" if None/invalid.
    """
    if not ts:
        return "—"
    text = ts.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(text)
        return dt.strftime("%b %d %H:%M")
    except Exception:
        return ts


def format_freq_label(freq_hz: Optional[float]) -> str:
    """
    Format a frequency in Hz for display (e.g., "100.5 MHz").

    Args:
        freq_hz: Frequency in Hz.

    Returns:
        Formatted frequency string with appropriate unit.
    """
    if freq_hz is None:
        return "—"
    try:
        freq = float(freq_hz)
    except (ValueError, TypeError):
        return "—"

    if freq >= 1e9:
        return f"{freq / 1e9:.3f} GHz"
    elif freq >= 1e6:
        return f"{freq / 1e6:.3f} MHz"
    elif freq >= 1e3:
        return f"{freq / 1e3:.1f} kHz"
    else:
        return f"{freq:.0f} Hz"


def format_bandwidth_khz(bw_hz: Optional[float]) -> str:
    """
    Format bandwidth in Hz as kHz display.

    Args:
        bw_hz: Bandwidth in Hz.

    Returns:
        Formatted bandwidth string (e.g., "25.0 kHz").
    """
    if bw_hz is None:
        return "—"
    try:
        bw = float(bw_hz)
    except (ValueError, TypeError):
        return "—"

    if bw >= 1e6:
        return f"{bw / 1e6:.2f} MHz"
    elif bw >= 1e3:
        return f"{bw / 1e3:.1f} kHz"
    else:
        return f"{bw:.0f} Hz"


def format_change_summary(event: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary for a change event.

    Args:
        event: Change event dict with type, f_center_hz, etc.

    Returns:
        Summary string describing the event.
    """
    event_type = str(event.get("type", "")).upper()
    freq_hz = event.get("f_center_hz")
    freq_str = format_freq_label(freq_hz)

    if event_type == "NEW_SIGNAL":
        confidence = event.get("confidence")
        if confidence is not None:
            try:
                conf_pct = float(confidence) * 100
                return f"New signal at {freq_str} (confidence {conf_pct:.0f}%)"
            except (ValueError, TypeError):
                pass
        return f"New signal detected at {freq_str}"

    elif event_type == "QUIETED":
        last_seen = event.get("last_seen_utc")
        if last_seen:
            return f"Signal at {freq_str} went quiet (last seen {format_ts_label(last_seen)})"
        return f"Signal at {freq_str} went quiet"

    elif event_type == "POWER_SHIFT":
        delta_db = event.get("delta_db")
        if delta_db is not None:
            try:
                return f"Power shift at {freq_str} (+{float(delta_db):.1f} dB)"
            except (ValueError, TypeError):
                pass
        return f"Power shift detected at {freq_str}"

    else:
        return f"{event_type} at {freq_str}"


def compute_display_bandwidth_hz(
    baseline_row: Dict[str, Any],
    f_low_hz: Optional[float],
    f_high_hz: Optional[float],
    bandwidth_hz: Optional[float],
) -> Optional[float]:
    """
    Compute display bandwidth, clamping to baseline bin_hz minimum.

    Args:
        baseline_row: Baseline record with bin_hz field.
        f_low_hz: Low frequency edge.
        f_high_hz: High frequency edge.
        bandwidth_hz: Pre-computed bandwidth, if any.

    Returns:
        Display bandwidth in Hz, or None if not computable.
    """
    # Get baseline bin width as minimum displayable bandwidth
    bin_hz = None
    try:
        raw_bin = baseline_row.get("bin_hz")
        if raw_bin is not None:
            bin_hz = float(raw_bin)
    except (ValueError, TypeError):
        bin_hz = None

    # Use provided bandwidth or compute from edges
    if bandwidth_hz is not None:
        try:
            bw = float(bandwidth_hz)
        except (ValueError, TypeError):
            bw = None
    elif f_low_hz is not None and f_high_hz is not None:
        try:
            bw = max(0.0, float(f_high_hz) - float(f_low_hz))
        except (ValueError, TypeError):
            bw = None
    else:
        bw = None

    if bw is None:
        return None

    # Clamp to at least bin_hz
    if bin_hz is not None and bin_hz > 0:
        bw = max(bw, bin_hz)

    return bw


# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------


def now_utc() -> datetime:
    """Get current UTC datetime."""
    return datetime.now(timezone.utc)


def isoformat_utc(dt: datetime) -> str:
    """Format datetime as ISO 8601 with Z suffix."""
    return dt.isoformat(timespec="seconds").replace("+00:00", "Z")


def parse_ts_utc(ts: Optional[str]) -> Optional[datetime]:
    """
    Parse an ISO timestamp string to datetime.

    Args:
        ts: ISO 8601 timestamp string.

    Returns:
        datetime object in UTC, or None if parsing fails.
    """
    if not ts:
        return None
    text = ts.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except Exception:
        return None
