"""Noise-floor estimation utilities."""

from __future__ import annotations

import numpy as np # type: ignore


def robust_noise_floor_db(psd_db: np.ndarray) -> float:
    """Median + 1.4826 * MAD noise estimator in dB."""
    med = np.median(psd_db)
    mad = np.median(np.abs(psd_db - med))
    return float(med + 1.4826 * mad)
