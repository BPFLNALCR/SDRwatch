"""Clustering helpers that operate directly on PSD arrays."""

from __future__ import annotations

from typing import Tuple

import numpy as np  # type: ignore


def expand_peak_bandwidth(
    psd_db: np.ndarray,
    noise_db: np.ndarray,
    peak_idx: int,
    *,
    floor_margin_db: float,
    peak_drop_db: float,
    max_gap_bins: int,
) -> Tuple[int, int]:
    """Walk away from a peak until energy drops near the noise floor.

    This helper encapsulates the bandwidth shaping heuristics used when turning per-bin
    PSD hits into contiguous segments. It is intentionally stateless so both the main
    sweep path and revisit confirmations can reuse the same logic.
    """

    psd_db = np.asarray(psd_db).astype(np.float64)
    noise_db = np.asarray(noise_db).astype(np.float64)
    N = psd_db.size
    if N == 0:
        return 0, 0
    peak_idx = int(np.clip(peak_idx, 0, max(N - 1, 0)))
    threshold_peak = float(psd_db[peak_idx] - max(0.0, peak_drop_db))

    def walk(direction: int) -> int:
        idx = peak_idx
        best = peak_idx
        gap = 0
        while True:
            nxt = idx + direction
            if nxt < 0 or nxt >= N:
                break
            if noise_db.shape == psd_db.shape:
                local_noise = float(noise_db[nxt])
            else:
                local_noise = float(noise_db[peak_idx])
            floor_threshold = local_noise + float(floor_margin_db)
            threshold = max(floor_threshold, threshold_peak)
            if psd_db[nxt] >= threshold:
                best = nxt
                gap = 0
            else:
                gap += 1
                if gap > max_gap_bins:
                    break
            idx = nxt
        return best

    low_idx = walk(-1)
    high_idx = walk(1)
    return low_idx, high_idx
