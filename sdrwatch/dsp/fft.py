"""FFT + PSD helper routines for the scanner."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np  # type: ignore

from sdrwatch.util.math import db10

try:  # pragma: no cover - optional dependency
    from scipy.signal import welch  # type: ignore

    HAVE_SCIPY = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_SCIPY = False
    welch = None  # type: ignore


def _welch_psd(samples: np.ndarray, samp_rate: float, fft_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Compute PSD via scipy.signal.welch (complex-friendly)."""
    assert welch is not None  # noqa: S101 - guarded by HAVE_SCIPY
    freqs, psd = welch(  # type: ignore[misc]
        samples,
        fs=samp_rate,
        nperseg=fft_size,
        noverlap=0,
        return_onesided=False,
        scaling="density",
    )
    order = np.argsort(freqs)
    freqs = freqs[order]
    psd = psd[order]
    mid = len(freqs) // 2
    freqs = np.concatenate((freqs[mid:], freqs[:mid]))
    psd = np.concatenate((psd[mid:], psd[:mid]))
    return freqs, psd


def _manual_psd(samples: np.ndarray, samp_rate: float, fft_size: int, avg: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback PSD computation using numpy FFT + Hann windows."""
    seg = fft_size
    windows: List[np.ndarray] = []
    for i in range(avg):
        start = i * seg
        chunk = samples[start : start + seg]
        if len(chunk) < seg:
            break
        X = np.fft.fftshift(np.fft.fft(chunk * np.hanning(seg), n=seg))
        Pxx = (np.abs(X) ** 2) / (seg * samp_rate)
        windows.append(Pxx)
    if not windows:
        trimmed = samples[:fft_size]
        if trimmed.size < fft_size:
            pad = np.zeros(fft_size, dtype=trimmed.dtype)
            pad[: trimmed.size] = trimmed
            trimmed = pad
        X = np.fft.fftshift(np.fft.fft(trimmed * np.hanning(fft_size), n=fft_size))
        Pxx = (np.abs(X) ** 2) / (fft_size * samp_rate)
        windows = [Pxx]
    psd = np.mean(np.vstack(windows), axis=0)
    freqs = np.linspace(-samp_rate / 2, samp_rate / 2, len(psd), endpoint=False)
    return freqs, psd


def compute_psd_db(samples: np.ndarray, samp_rate: float, fft_size: int, avg: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (freqs, psd_db) for complex IQ input samples."""
    if HAVE_SCIPY:
        freqs, psd = _welch_psd(samples, samp_rate, fft_size)
    else:
        freqs, psd = _manual_psd(samples, samp_rate, fft_size, avg)
    psd_db = db10(psd)
    return freqs, psd_db
