"""Window helpers that wrap numpy stride tricks with fallbacks."""

from __future__ import annotations

import numpy as np


def _sliding_window_view(x: np.ndarray, window: int) -> np.ndarray:
    """Return a sliding window view over the last axis with a 1D fallback."""
    try:
        return np.lib.stride_tricks.sliding_window_view(x, window)
    except Exception:
        x = np.asarray(x)
        if x.ndim != 1:
            raise ValueError("sliding window fallback only supports 1D arrays")
        shape = (x.size - window + 1, window)
        if shape[0] <= 0:
            return np.empty((0, window), dtype=x.dtype)
        strides = (x.strides[0], x.strides[0])
        return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
