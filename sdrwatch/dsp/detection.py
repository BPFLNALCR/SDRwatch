"""Segment detection built on PSD inputs and CFAR masks."""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from sdrwatch.detection.types import Segment

from .cfar import cfar_os_mask
from .clustering import expand_peak_bandwidth, estimate_bandwidth
from .noise_estimation import robust_noise_floor_db


def split_segment_by_valleys(
    psd_db: np.ndarray,
    noise_db: np.ndarray,
    start_idx: int,
    end_idx: int,
    *,
    drop_db: float,
    noise_margin_db: float,
    min_valley_bins: int,
    min_segment_bins: int,
    min_peak_prominence_db: float,
) -> List[Tuple[int, int]]:
    """Split merged lobes by searching for deep valleys between peaks."""

    start_idx = int(start_idx)
    end_idx = int(end_idx)
    if end_idx - start_idx <= min_segment_bins or drop_db <= 0 and noise_margin_db <= 0:
        return [(start_idx, end_idx)]

    window = np.asarray(psd_db[start_idx:end_idx], dtype=np.float64)
    noise_arr = np.asarray(noise_db, dtype=np.float64)
    if noise_arr.shape != psd_db.shape:
        if noise_arr.size == 1:
            noise_arr = np.full(psd_db.shape, float(noise_arr.reshape(-1)[0]))
        else:
            noise_arr = np.resize(noise_arr, psd_db.shape)
    if window.size <= max(2, min_segment_bins):
        return [(start_idx, end_idx)]

    peaks = _find_local_peaks(window, min_peak_prominence_db)
    if len(peaks) <= 1:
        return [(start_idx, end_idx)]

    min_valley_bins = max(1, int(min_valley_bins))
    min_segment_bins = max(1, int(min_segment_bins))
    drop_db = max(0.0, float(drop_db))
    noise_margin_db = max(0.0, float(noise_margin_db))
    boundaries: List[int] = [start_idx]

    for left_idx, right_idx in zip(peaks[:-1], peaks[1:]):
        if (right_idx - left_idx) < min_valley_bins:
            continue
        valley_slice = slice(start_idx + left_idx, start_idx + right_idx + 1)
        valley_vals = np.asarray(psd_db[valley_slice], dtype=np.float64)
        if valley_vals.size == 0:
            continue
        valley_offset = int(np.argmin(valley_vals))
        split_idx = valley_slice.start + valley_offset
        valley_db = float(psd_db[split_idx])
        left_peak_db = float(psd_db[start_idx + left_idx])
        right_peak_db = float(psd_db[start_idx + right_idx])
        noise_floor = float(noise_arr[split_idx])
        drop_condition = (left_peak_db - valley_db) >= drop_db and (right_peak_db - valley_db) >= drop_db
        noise_condition = valley_db <= (noise_floor + noise_margin_db)
        if not (drop_condition or noise_condition):
            continue
        if (split_idx - boundaries[-1]) < min_segment_bins:
            continue
        if (end_idx - split_idx) < min_segment_bins:
            continue
        boundaries.append(split_idx)

    boundaries.append(end_idx)
    result: List[Tuple[int, int]] = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        if (b - a) >= min_segment_bins:
            result.append((a, b))
    return result or [(start_idx, end_idx)]


def _find_local_peaks(window: np.ndarray, min_prominence_db: float) -> List[int]:
    data = np.asarray(window, dtype=np.float64)
    if data.size == 0:
        return []
    if data.size == 1:
        return [0]
    left = np.r_[data[0], data[:-1]]
    right = np.r_[data[1:], data[-1]]
    mask = (data >= left) & (data >= right)
    peak_indices = np.where(mask)[0]
    if peak_indices.size == 0:
        return [int(np.argmax(data))]
    prominence_threshold = float(np.max(data) - min_prominence_db)
    filtered: List[int] = []
    for idx in peak_indices:
        if data[idx] >= prominence_threshold:
            filtered.append(int(idx))
    if not filtered:
        filtered.append(int(np.argmax(data)))
    return filtered


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
    bandshape_mode: str = "minus6db",
    bandshape_drop_db: Optional[float] = None,
    bandshape_window_bins: int = 24,
    bandshape_curvature_db: float = 3.0,
    bandshape_min_prominence_db: float = 1.0,
    bandshape_polyfit_bins: int = 10,
    split_peak_drop_db: float = 4.0,
    split_noise_margin_db: float = 1.5,
    split_min_valley_bins: int = 2,
    split_min_peak_prominence_db: float = 2.0,
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
                candidate_ranges = split_segment_by_valleys(
                    psd_db,
                    noise_for_snr_db,
                    start_i,
                    end_i,
                    drop_db=float(split_peak_drop_db),
                    noise_margin_db=float(split_noise_margin_db),
                    min_valley_bins=max(1, int(split_min_valley_bins)),
                    min_segment_bins=max(1, min_width_bins),
                    min_peak_prominence_db=float(split_min_peak_prominence_db),
                )
                for seg_start, seg_end in candidate_ranges:
                    if (seg_end - seg_start) < min_width_bins:
                        continue
                    sl = slice(seg_start, seg_end)
                    peak_idx_local = int(np.argmax(psd_db[sl]))
                    peak_idx = seg_start + peak_idx_local
                    peak_db = float(psd_db[peak_idx])
                    noise_db = float(noise_for_snr_db[peak_idx])
                    snr_db = float(peak_db - noise_db)
                    try:
                        est_low, est_high, est_bw = estimate_bandwidth(
                            psd_db,
                            freqs_hz,
                            peak_idx,
                            mode=bandshape_mode,
                            drop_db=bandshape_drop_db,
                            curvature_guard_db=bandshape_curvature_db,
                            max_window_bins=max(3, int(bandshape_window_bins)),
                            min_width_bins=max(min_width_bins, 1),
                            min_prominence_db=bandshape_min_prominence_db,
                            polyfit_window_bins=max(3, int(bandshape_polyfit_bins)),
                        )
                    except Exception:
                        est_low, est_high, est_bw = (np.nan, np.nan, np.nan)

                    fallback_low_idx, fallback_high_idx = expand_peak_bandwidth(
                        psd_db,
                        noise_for_snr_db,
                        peak_idx,
                        floor_margin_db=float(bandwidth_floor_db),
                        peak_drop_db=float(bandwidth_peak_drop_db),
                        max_gap_bins=gap_bins,
                    )
                    fallback_idx_low = min(fallback_low_idx, fallback_high_idx)
                    fallback_idx_high = max(fallback_low_idx, fallback_high_idx)
                    fallback_idx_low = max(fallback_idx_low, seg_start)
                    fallback_idx_high = min(fallback_idx_high, seg_end - 1)
                    fallback_low = float(freqs_hz[fallback_idx_low])
                    fallback_high = float(freqs_hz[fallback_idx_high])
                    fallback_bw = max(fallback_high - fallback_low + (bin_hz if bin_hz > 0 else 0.0), 0.0)

                    if not np.isfinite(est_bw) or est_bw <= 0.0:
                        f_low = fallback_low
                        f_high = fallback_high
                        bandwidth_hz = fallback_bw
                    else:
                        freq_min = float(freqs_hz[seg_start])
                        freq_max = float(freqs_hz[seg_end - 1])
                        f_low = float(np.clip(est_low, freq_min, freq_max))
                        f_high = float(np.clip(est_high, freq_min, freq_max))
                        if f_high <= f_low:
                            f_low = fallback_low
                            f_high = fallback_high
                            bandwidth_hz = fallback_bw
                        else:
                            bandwidth_hz = max(float(est_bw), fallback_bw if fallback_bw > 0 else 0.0)
                    f_center = float((f_low + f_high) / 2.0)
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
