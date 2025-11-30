"""Baseline model definitions shared across the scanner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BaselineModel:
    """In-memory representation of a baseline row."""

    id: int
    name: str
    freq_start_hz: int
    freq_stop_hz: int
    bin_hz: float
    baseline_version: int
    total_windows: int


# Backwards compatibility: many modules still refer to BaselineContext.
BaselineContext = BaselineModel

__all__ = ["BaselineModel", "BaselineContext"]
