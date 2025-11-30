"""CFAR helpers built on top of numpy windowing."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .windowing import _sliding_window_view


def cfar_os_mask(psd_db: np.ndarray, train: int, guard: int, quantile: float, alpha_db: float) -> Tuple[np.ndarray, np.ndarray]:
    """Order-Statistic CFAR (OS-CFAR) operating on 1D PSD arrays."""
    psd_db = np.asarray(psd_db).astype(np.float64)
    N = psd_db.size
    if N == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    psd_lin = np.power(10.0, psd_db / 10.0)
    win = 2 * train + 2 * guard + 1
    if win <= 1:
        noise_db = np.full(N, float(np.median(psd_db)))
        above = psd_db > (noise_db + alpha_db)
        return above, noise_db
    pad = train + guard
    padded = np.pad(psd_lin, (pad, pad), mode="edge")
    windows = _sliding_window_view(padded, win)
    mask = np.ones(win, dtype=bool)
    mask[train : train + 2 * guard + 1] = False
    train_windows = windows[:, mask]
    q = float(np.clip(quantile, 1e-6, 1.0 - 1e-6))
    noise_lin = np.quantile(train_windows, q, axis=1)
    alpha = np.power(10.0, alpha_db / 10.0)
    threshold_lin = noise_lin * alpha
    above = psd_lin > threshold_lin
    noise_db = 10.0 * np.log10(np.maximum(noise_lin, 1e-20))
    return above, noise_db
