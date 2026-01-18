"""
Query filter parsing and SQL predicate builders for SDRwatch Web.

These helpers parse HTTP query parameters into normalized filter dicts
and build SQL WHERE clauses for detection and scan queries.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


def parse_detection_filters(
    args,
    *,
    default_since_hours: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Parse query params into normalized detection filters and form defaults.

    Args:
        args: Request args (or any dict-like object with .get()).
        default_since_hours: Default lookback hours if not specified.

    Returns:
        Tuple of (filters dict, form_defaults dict for repopulating forms).
    """

    def _clean_str(key: str) -> str:
        val = args.get(key) if hasattr(args, "get") else args.get(key)
        if val is None:
            return ""
        if isinstance(val, str):
            return val.strip()
        return str(val).strip()

    def _coerce_float(text: str) -> Optional[float]:
        if text == "":
            return None
        try:
            return float(text)
        except Exception:
            return None

    filters: Dict[str, Any] = {
        "service": None,
        "min_snr": None,
        "f_min_hz": None,
        "f_max_hz": None,
        "since_hours": default_since_hours,
        "min_conf": None,
    }
    form_defaults: Dict[str, str] = {
        "service": "",
        "min_snr": "",
        "f_min_mhz": "",
        "f_max_mhz": "",
        "since_hours": "" if default_since_hours is None else str(default_since_hours),
        "min_conf": "",
    }

    service = _clean_str("service")
    if service:
        filters["service"] = service
        form_defaults["service"] = service

    min_snr_raw = _clean_str("min_snr")
    min_snr_val = _coerce_float(min_snr_raw)
    if min_snr_val is not None:
        filters["min_snr"] = min_snr_val
    if min_snr_raw:
        form_defaults["min_snr"] = min_snr_raw

    f_min_raw = _clean_str("f_min_mhz")
    f_min_val = _coerce_float(f_min_raw)
    if f_min_val is not None:
        filters["f_min_hz"] = int(f_min_val * 1e6)
    if f_min_raw:
        form_defaults["f_min_mhz"] = f_min_raw

    f_max_raw = _clean_str("f_max_mhz")
    f_max_val = _coerce_float(f_max_raw)
    if f_max_val is not None:
        filters["f_max_hz"] = int(f_max_val * 1e6)
    if f_max_raw:
        form_defaults["f_max_mhz"] = f_max_raw

    since_raw = _clean_str("since_hours")
    if since_raw:
        try:
            since_val = max(1, int(float(since_raw)))
            filters["since_hours"] = since_val
            form_defaults["since_hours"] = str(since_val)
        except Exception:
            if default_since_hours is not None:
                form_defaults["since_hours"] = str(default_since_hours)
    elif default_since_hours is None:
        filters["since_hours"] = None

    min_conf_raw = _clean_str("min_conf")
    min_conf_val = _coerce_float(min_conf_raw)
    if min_conf_val is not None:
        filters["min_conf"] = float(min_conf_val)
    if min_conf_raw:
        form_defaults["min_conf"] = min_conf_raw

    return filters, form_defaults


def detection_predicates(
    filters: Dict[str, Any],
    *,
    alias: str = "d",
) -> Tuple[List[str], List[Any]]:
    """
    Build SQL WHERE conditions for detection queries.

    Args:
        filters: Normalized filter dict from parse_detection_filters().
        alias: Table alias for the detections table.

    Returns:
        Tuple of (list of SQL condition strings, list of parameter values).
    """
    conds: List[str] = []
    params: List[Any] = []

    service = filters.get("service")
    if service:
        conds.append(f"COALESCE({alias}.service,'Unknown') = ?")
        params.append(service)

    min_snr = filters.get("min_snr")
    if min_snr is not None:
        conds.append(f"{alias}.snr_db >= ?")
        params.append(float(min_snr))

    f_min = filters.get("f_min_hz")
    if f_min is not None:
        conds.append(f"{alias}.f_center_hz >= ?")
        params.append(int(f_min))

    f_max = filters.get("f_max_hz")
    if f_max is not None:
        conds.append(f"{alias}.f_center_hz <= ?")
        params.append(int(f_max))

    since_hours = filters.get("since_hours")
    if since_hours is not None and since_hours > 0:
        conds.append(f"{alias}.time_utc >= datetime('now', ?)")
        params.append(f"-{int(since_hours)} hours")

    min_conf = filters.get("min_conf")
    confidence_available = bool(filters.get("__confidence_available"))
    if confidence_available and min_conf is not None:
        conds.append(f"{alias}.confidence >= ?")
        params.append(float(min_conf))

    return conds, params


def scan_predicates(
    filters: Dict[str, Any],
    *,
    alias: str = "s",
) -> Tuple[List[str], List[Any]]:
    """
    Build SQL WHERE conditions for scan queries.

    Args:
        filters: Normalized filter dict from parse_detection_filters().
        alias: Table alias for the scans table.

    Returns:
        Tuple of (list of SQL condition strings, list of parameter values).
    """
    conds: List[str] = []
    params: List[Any] = []

    since_hours = filters.get("since_hours")
    if since_hours is not None and since_hours > 0:
        conds.append(f"COALESCE({alias}.t_end_utc, {alias}.t_start_utc) >= datetime('now', ?)")
        params.append(f"-{int(since_hours)} hours")

    f_min = filters.get("f_min_hz")
    if f_min is not None:
        conds.append(f"{alias}.f_stop_hz >= ?")
        params.append(int(f_min))

    f_max = filters.get("f_max_hz")
    if f_max is not None:
        conds.append(f"{alias}.f_start_hz <= ?")
        params.append(int(f_max))

    return conds, params
