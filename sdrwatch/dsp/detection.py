"""Segment detection built on PSD inputs and CFAR masks."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from sdrwatch.detection.types import Segment

from .cfar import cfar_os_mask
from .clustering import expand_peak_bandwidth
from .noise_estimation import robust_noise_floor_db


def detect_segments(
    freqs_hz: np.ndarray,
    psd_db: np.ndarray,
    thresh_db: float,
    guard_bins: int = 1,
    min_width_bins: int = 2,
    cfar_mode: str = "off",
    cfar_train: int = 24,
    cfar_guard: int = 4,
    cfar_quantile: float = 0.75,
    cfar_alpha_db: Optional[float] = None,
    abs_power_floor_db: Optional[float] = None,
    *,
    bandwidth_floor_db: float = 2.0,
    bandwidth_peak_drop_db: float = 18.0,
    bandwidth_gap_hz: float = 15_000.0,
) -> Tuple[List[Segment], np.ndarray, np.ndarray]:
    """Detect contiguous energy segments from a PSD in dB."""
    psd_db = np.asarray(psd_db).astype(np.float64)
    freqs_hz = np.asarray(freqs_hz).astype(np.float64)
    N = psd_db.size
    if N == 0:
        return [], np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)

    if cfar_mode and cfar_mode.lower() != "off":
        alpha_db = float(cfar_alpha_db if cfar_alpha_db is not None else thresh_db)
        above, noise_local_db = cfar_os_mask(psd_db, cfar_train, cfar_guard, cfar_quantile, alpha_db)
        noise_for_snr_db = noise_local_db
    else:
        nf = robust_noise_floor_db(psd_db)
        dynamic = nf + float(thresh_db)
        above = psd_db > dynamic
        noise_for_snr_db = np.full(N, nf, dtype=np.float64)

    if N > 1:
        diffs = np.diff(freqs_hz)
        bin_hz = float(np.median(diffs)) if diffs.size else 0.0
    else:
        bin_hz = 0.0
    gap_bins = max(1, int(round(bandwidth_gap_hz / max(bin_hz, 1.0)))) if bin_hz > 0 else max(1, min_width_bins)

    segs: List[Segment] = []
    i = 0
    while i < N:
        if bool(above[i]):
            start_i = i
            j = i + 1
            gap = 0
            while j < N and (bool(above[j]) or gap < guard_bins):
                if bool(above[j]):
                    gap = 0
                else:
                    gap += 1
                j += 1
            end_i = j
            if (end_i - start_i) >= min_width_bins:
                sl = slice(start_i, end_i)
                peak_idx_local = int(np.argmax(psd_db[sl]))
                peak_idx = start_i + peak_idx_local
                peak_db = float(psd_db[peak_idx])
                noise_db = float(noise_for_snr_db[peak_idx])
                snr_db = float(peak_db - noise_db)
                low_idx, high_idx = expand_peak_bandwidth(
                    psd_db,
                    noise_for_snr_db,
                    peak_idx,
                    floor_margin_db=float(bandwidth_floor_db),
                    peak_drop_db=float(bandwidth_peak_drop_db),
                    max_gap_bins=gap_bins,
                )
                idx_low = min(low_idx, high_idx)
                idx_high = max(low_idx, high_idx)
                f_low = float(freqs_hz[idx_low])
                f_high = float(freqs_hz[idx_high])
                center_idx = (idx_low + idx_high) // 2
                f_center = float(freqs_hz[center_idx])
                bandwidth_hz = max(f_high - f_low + (bin_hz if bin_hz > 0 else 0.0), 0.0)
                segs.append(
                    Segment(
                        f_low_hz=int(round(f_low)),
                        f_high_hz=int(round(f_high)),
                        f_center_hz=int(round(f_center)),
                        peak_db=peak_db,
                        noise_db=noise_db,
                        snr_db=snr_db,
                        bandwidth_hz=float(bandwidth_hz),
                    )
                )
            i = j
        else:
            i += 1

    if abs_power_floor_db is not None:
        floor = float(abs_power_floor_db)
        segs = [seg for seg in segs if seg.peak_db >= floor]

    return segs, above, noise_for_snr_db
