"""Helpers for materialized baseline summary tables."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import List, Tuple


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return max(1, int(float(value)))
    except Exception:
        return default


def _float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    try:
        return float(value)
    except Exception:
        return default


@dataclass
class BandSummaryConfig:
    """Configuration for band summary partitioning and recent windows."""

    max_bands: int = 6
    target_band_width_hz: float = 10_000_000.0
    min_band_width_hz: float = 1_000_000.0
    recent_minutes: int = 30
    occ_threshold_ratio: float = 0.2

    @classmethod
    def from_env(cls) -> "BandSummaryConfig":
        return cls(
            max_bands=_int_env("SDRWATCH_BAND_SUMMARY_MAX_BANDS", 6),
            target_band_width_hz=_float_env("SDRWATCH_BAND_SUMMARY_TARGET_WIDTH_HZ", 10_000_000.0),
            min_band_width_hz=_float_env("SDRWATCH_BAND_SUMMARY_MIN_WIDTH_HZ", 1_000_000.0),
            recent_minutes=_int_env("SDRWATCH_BAND_SUMMARY_RECENT_MINUTES", 30),
            occ_threshold_ratio=_float_env("SDRWATCH_BAND_SUMMARY_OCC_THRESHOLD", 0.2),
        )


def band_partitions(freq_start_hz: float, freq_stop_hz: float, config: BandSummaryConfig) -> Tuple[List[Tuple[float, float]], float]:
    """Compute span partitions for a baseline."""

    span = freq_stop_hz - freq_start_hz
    if span <= 0:
        return [], 0.0
    target_width = config.target_band_width_hz if config.target_band_width_hz > 0 else span
    target_width = max(config.min_band_width_hz, target_width)
    approx_count = max(1, int(math.ceil(span / target_width)))
    band_count = max(1, min(config.max_bands, approx_count))
    band_width = span / band_count if band_count else span
    partitions: List[Tuple[float, float]] = []
    for idx in range(band_count):
        low = freq_start_hz + idx * band_width
        high = freq_start_hz + (idx + 1) * band_width if idx < band_count - 1 else freq_stop_hz
        partitions.append((low, high))
    return partitions, band_width


def tactical_recent_minutes() -> int:
    """Return the configured tactical recent window for snapshot summaries."""

    return _int_env("SDRWATCH_TACTICAL_RECENT_MINUTES", 30)
