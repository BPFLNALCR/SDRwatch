"""Numeric helper functions used across DSP logic."""

import numpy as np


def db10(x: np.ndarray) -> np.ndarray:
    """Return 10 * log10(x) with floor to keep inputs positive."""
    return 10.0 * np.log10(np.maximum(x, 1e-20))
