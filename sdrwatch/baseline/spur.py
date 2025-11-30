"""Spur calibration and suppression helpers."""

from __future__ import annotations

from typing import Dict, Optional

from sdrwatch.baseline.store import Store
from sdrwatch.detection.types import Segment


class SpurCalibrationTracker:
    """Accumulate spur candidates during calibration sweeps."""

    def __init__(self) -> None:
        self._tracker: Dict[int, Dict[str, float]] = {}

    def observe(self, segments: list[Segment]) -> None:
        for seg in segments:
            entry = self._tracker.setdefault(seg.f_center_hz, {"hits": 0.0, "power_sum": 0.0})
            entry["hits"] += 1.0
            entry["power_sum"] += float(seg.peak_db)

    def persist(self, store: Store, total_windows: int, min_ratio: float = 0.6) -> None:
        if total_windows <= 0:
            return
        min_hits = max(1, int(total_windows * min_ratio))
        for bin_hz, stats in self._tracker.items():
            hits = int(stats.get("hits", 0))
            if hits < min_hits:
                continue
            avg_power = float(stats.get("power_sum", 0.0)) / float(max(hits, 1))
            store.update_spur_bin(bin_hz, avg_power, hits_increment=hits)


class SpurEvaluator:
    """Apply spur suppression heuristics based on the stored spur map."""

    def __init__(
        self,
        store: Store,
        *,
        tolerance_hz: float = 5_000.0,
        margin_db: float = 4.0,
        min_hits: int = 5,
        override_snr: float = 10.0,
        penalty_max: float = 0.35,
    ) -> None:
        self.store = store
        self.tolerance_hz = int(tolerance_hz)
        self.margin_db = float(margin_db)
        self.min_hits = int(min_hits)
        self.override_snr = float(override_snr)
        self.penalty_max = float(penalty_max)

    def should_mask(self, seg: Segment, *, calibration_mode: bool = False) -> bool:
        if calibration_mode:
            return False
        spur = self._lookup(seg.f_center_hz)
        if not spur:
            return False
        _, mean_power_db, hits = spur
        if hits < self.min_hits:
            return False
        if seg.peak_db >= mean_power_db + self.margin_db:
            return False
        if seg.snr_db >= self.override_snr:
            return False
        return True

    def confidence_penalty(self, seg: Segment, *, calibration_mode: bool = False) -> float:
        if calibration_mode:
            return 0.0
        spur = self._lookup(seg.f_center_hz)
        if not spur:
            return 0.0
        _, mean_power_db, hits = spur
        if hits < self.min_hits:
            return 0.0
        diff = float(seg.peak_db - mean_power_db)
        if diff >= self.margin_db + 5.0:
            return 0.05
        if diff >= self.margin_db:
            return min(0.15, self.penalty_max)
        return self.penalty_max

    def _lookup(self, center_hz: int) -> Optional[tuple[int, float, int]]:
        return self.store.lookup_spur(center_hz, self.tolerance_hz)
