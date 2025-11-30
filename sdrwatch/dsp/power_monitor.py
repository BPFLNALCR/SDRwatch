"""Power anomaly tracking helpers."""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


class WindowPowerMonitor:
    """Track rolling power to flag anomalous sweeps."""

    def __init__(self, spike_db: float = 8.0, ema_alpha: float = 0.2, warmup_windows: int = 3):
        self.spike_db = float(spike_db)
        self.ema_alpha = float(np.clip(ema_alpha, 1e-3, 1.0))
        self.warmup_windows = max(0, int(warmup_windows))
        self.ema: Optional[float] = None
        self.count = 0

    def update(self, mean_db: float) -> Tuple[bool, float, float]:
        self.count += 1
        if self.ema is None:
            self.ema = mean_db
            return False, mean_db, 0.0
        delta = float(mean_db - self.ema)
        is_anom = self.count > self.warmup_windows and delta > self.spike_db
        self.ema = (1.0 - self.ema_alpha) * self.ema + self.ema_alpha * mean_db
        return is_anom, float(self.ema), delta
