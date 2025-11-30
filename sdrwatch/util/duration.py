"""Duration parsing helpers for CLI arguments."""

from __future__ import annotations

import argparse
from typing import Any, Optional


def parse_duration_to_seconds(spec: Optional[Any]) -> Optional[float]:
    """Parse strings like '30', '10m', '2h', returning seconds as float."""

    if spec is None:
        return None
    if isinstance(spec, (int, float)):
        return float(spec)
    text = str(spec).strip().lower()
    if not text:
        return None
    unit = text[-1]
    if unit.isalpha():
        value_part = text[:-1]
    else:
        unit = "s"
        value_part = text
    try:
        value = float(value_part)
    except ValueError as exc:  # pragma: no cover - argparse guards
        raise argparse.ArgumentTypeError(f"Invalid duration '{spec}'") from exc
    multipliers = {
        "s": 1.0,
        "m": 60.0,
        "h": 3600.0,
        "d": 86400.0,
    }
    if unit not in multipliers:
        raise argparse.ArgumentTypeError(f"Unsupported duration suffix '{unit}'")
    return value * multipliers[unit]
