"""Clustering helpers that operate directly on PSD arrays."""

from __future__ import annotations

from typing import Tuple

import numpy as np  # type: ignore


def estimate_bandwidth(
    psd_db: np.ndarray,
    freq_axis_hz: np.ndarray,
    peak_index: int,
    mode: str = "minus6db",
    *,
    drop_db: float | None = None,
    curvature_guard_db: float = 3.0,
    max_window_bins: int = 24,
    min_width_bins: int = 2,
    min_prominence_db: float = 1.0,
    polyfit_window_bins: int = 10,
) -> Tuple[float, float, float]:
    """Estimate the low/high frequency edges of a single PSD peak.

    The resolver walks left/right until the PSD drops below the target threshold or
    the local curvature suggests another lobe, then refines the edges with a
    quadratic fit to reduce bin-quantization noise.
    """

    psd = np.asarray(psd_db, dtype=np.float64)
    freq = np.asarray(freq_axis_hz, dtype=np.float64)
    N = psd.size
    if N == 0 or freq.size != N:
        return 0.0, 0.0, 0.0
    peak_index = int(np.clip(int(peak_index), 0, max(N - 1, 0)))
    peak_db = float(psd[peak_index])
    diffs = np.diff(freq)
    bin_hz = float(np.median(diffs)) if diffs.size else 0.0

    resolved_drop = _resolve_drop_db(mode, drop_db)
    threshold_db = peak_db - resolved_drop
    min_width_bins = max(1, int(min_width_bins))
    max_window_bins = max(max_window_bins, min_width_bins)
    curvature_guard_db = max(0.0, float(curvature_guard_db))
    min_prominence_db = max(0.0, float(min_prominence_db))

    def walk(direction: int) -> int:
        idx = peak_index
        best = peak_index
        prev_val = peak_db
        steps = 0
        while True:
            nxt = idx + direction
            if nxt < 0 or nxt >= N:
                break
            if steps >= max_window_bins:
                break
            val = float(psd[nxt])
            if prev_val - val < -curvature_guard_db:
                break
            best = nxt
            idx = nxt
            steps += 1
            if val <= threshold_db:
                break
            prev_val = val
        return best

    idx_low = walk(-1)
    idx_high = walk(1)

    if idx_high <= idx_low:
        pad = max(1, min_width_bins)
        idx_low = max(0, peak_index - pad)
        idx_high = min(N - 1, peak_index + pad)

    idx_low, idx_high = _enforce_min_width(idx_low, idx_high, peak_index, min_width_bins, N)

    f_low = float(freq[idx_low])
    f_high = float(freq[idx_high])

    if polyfit_window_bins > 0:
        f_low, f_high = _refine_with_polyfit(
            freq,
            psd,
            peak_index,
            f_low,
            f_high,
            resolved_drop,
            min_prominence_db,
            polyfit_window_bins,
        )

    bandwidth_hz = max(f_high - f_low, (bin_hz if bin_hz > 0.0 else 0.0))
    return f_low, f_high, bandwidth_hz


def _resolve_drop_db(mode: str, drop_db: float | None) -> float:
    if drop_db is not None:
        return max(0.1, float(drop_db))
    normalized = (mode or "minus6db").strip().lower()
    mapping = {
        "minus1db": 1.0,
        "minus3db": 3.0,
        "minus6db": 6.0,
        "minus10db": 10.0,
        "minus12db": 12.0,
    }
    if normalized.startswith("custom:"):
        try:
            return max(0.1, float(normalized.split(":", 1)[1]))
        except Exception:
            return 6.0
    return mapping.get(normalized, 6.0)


def _enforce_min_width(idx_low: int, idx_high: int, peak_idx: int, min_width_bins: int, total_bins: int) -> Tuple[int, int]:
    span = idx_high - idx_low
    if span >= min_width_bins:
        return idx_low, idx_high
    needed = min_width_bins - span
    pad_left = needed // 2
    pad_right = needed - pad_left
    idx_low = max(0, idx_low - pad_left)
    idx_high = min(total_bins - 1, idx_high + pad_right)
    if idx_high <= idx_low:
        idx_low = max(0, peak_idx - min_width_bins)
        idx_high = min(total_bins - 1, peak_idx + min_width_bins)
    return idx_low, idx_high


def _refine_with_polyfit(
    freq: np.ndarray,
    psd: np.ndarray,
    peak_idx: int,
    f_low: float,
    f_high: float,
    drop_db: float,
    min_prominence_db: float,
    window_bins: int,
) -> Tuple[float, float]:
    half = max(2, int(window_bins))
    start = max(0, peak_idx - half)
    stop = min(psd.size, peak_idx + half + 1)
    local_freq = freq[start:stop]
    local_psd = psd[start:stop]
    if local_freq.size < 5:
        return f_low, f_high
    prominence = float(np.max(local_psd) - np.min(local_psd))
    if prominence < min_prominence_db:
        return f_low, f_high
    x = local_freq - freq[peak_idx]
    try:
        coeffs = np.polyfit(x, local_psd, 2)
    except np.linalg.LinAlgError:
        return f_low, f_high
    a, b, c = coeffs
    if a >= 0:
        return f_low, f_high
    target = float(psd[peak_idx]) - drop_db
    roots = np.roots([a, b, c - target])
    real_roots = [float(r.real) for r in roots if abs(r.imag) < 1e-3]
    if len(real_roots) < 2:
        return f_low, f_high
    center = freq[peak_idx]
    left_candidates = [center + r for r in real_roots if r < 0]
    right_candidates = [center + r for r in real_roots if r > 0]
    if not left_candidates or not right_candidates:
        return f_low, f_high
    refined_low = max(left_candidates)
    refined_high = min(right_candidates)
    if refined_high <= refined_low:
        return f_low, f_high
    refined_low = max(refined_low, f_low)
    refined_high = min(refined_high, f_high)
    return refined_low, refined_high


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
