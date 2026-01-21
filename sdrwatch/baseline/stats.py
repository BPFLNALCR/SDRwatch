"""Baseline statistics update helpers."""

from __future__ import annotations

import sys
from typing import Tuple

import numpy as np  # type: ignore

from sdrwatch.baseline.model import BaselineContext
from sdrwatch.baseline.store import Store
from sdrwatch.util.time import utc_now_str


class BaselineStatsUpdater:
    """Apply per-window baseline statistics and span maintenance."""

    def __init__(self, store: Store, baseline_ctx: BaselineContext, *, sweep_bin_hz: float) -> None:
        self.store = store
        self.baseline_ctx = baseline_ctx
        self.sweep_bin_hz = float(sweep_bin_hz)
        self._warned_bin_mismatch = False
        self._warned_bin_invalid = False
        self._warned_span = False
        self._total_dwell_ms: float = 0.0  # Track dwell time accumulated this sweep

    def update_window(
        self,
        rf_freqs: np.ndarray,
        psd_db: np.ndarray,
        noise_per_bin_db: np.ndarray,
        occupied_mask: np.ndarray,
        *,
        dwell_ms: float = 0.0,
    ) -> bool:
        """Update per-bin EMA stats for a single sweep window.
        
        Args:
            rf_freqs: RF frequencies for each bin (Hz).
            psd_db: Power spectral density in dB for each bin.
            noise_per_bin_db: Noise floor estimate in dB for each bin.
            occupied_mask: Boolean mask indicating which bins are occupied.
            dwell_ms: Dwell time in milliseconds for this window. If not provided,
                      time-based duty cycle will fall back to window-based ratio.
        """

        if self.baseline_ctx.bin_hz <= 0:
            self._warn_invalid_bin()
            return False
        self._warn_if_bin_mismatch()
        valid_mask = (rf_freqs >= self.baseline_ctx.freq_start_hz) & (rf_freqs <= self.baseline_ctx.freq_stop_hz)
        if not bool(np.any(valid_mask)):
            self._warn_span_outside(rf_freqs)
            return False

        bin_positions = (rf_freqs - self.baseline_ctx.freq_start_hz) / max(self.baseline_ctx.bin_hz, 1.0)
        bin_indices = np.rint(bin_positions[valid_mask]).astype(int)
        noise_vec = noise_per_bin_db[valid_mask].astype(float)
        power_vec = psd_db[valid_mask].astype(float)
        occupied_vec = occupied_mask[valid_mask]
        window_ts = utc_now_str()

        self.store.begin()
        self.store.update_baseline_stats(
            self.baseline_ctx.id,
            bin_indices,
            noise_floor_db=noise_vec,
            power_db=power_vec,
            occupied_mask=occupied_vec,
            timestamp_utc=window_ts,
            dwell_ms=dwell_ms,
        )
        total_windows = self.store.increment_baseline_windows(self.baseline_ctx.id, 1)
        if dwell_ms > 0:
            total_observed_ms = self.store.increment_baseline_observed_ms(self.baseline_ctx.id, dwell_ms)
            self.baseline_ctx.total_observed_ms = total_observed_ms
            self._total_dwell_ms += dwell_ms
        self.store.commit()
        self.baseline_ctx.total_windows = total_windows
        return True

    def update_span(self, planned: Tuple[int, int]) -> None:
        """Expand the baseline span to cover the planned sweep if needed."""

        start, stop = (int(planned[0]), int(planned[1]))
        updated_start, updated_stop = self.store.update_baseline_span(self.baseline_ctx.id, start, stop)
        if updated_start is not None:
            self.baseline_ctx.freq_start_hz = updated_start
        if updated_stop is not None:
            self.baseline_ctx.freq_stop_hz = updated_stop

    # -----------------
    # Warning helpers
    # -----------------

    def _warn_if_bin_mismatch(self) -> None:
        if self._warned_bin_mismatch:
            return
        diff = abs(self.sweep_bin_hz - self.baseline_ctx.bin_hz)
        if diff <= max(1.0, self.baseline_ctx.bin_hz * 0.05):
            return
        self._warned_bin_mismatch = True
        msg = (
            f"[baseline] WARNING: sweep bin {self.sweep_bin_hz:.2f} Hz differs from baseline bin "
            f"{self.baseline_ctx.bin_hz:.2f} Hz"
        )
        print(msg, file=sys.stderr)

    def _warn_invalid_bin(self) -> None:
        if self._warned_bin_invalid:
            return
        self._warned_bin_invalid = True
        print("[baseline] bin_hz invalid; skipping stats update", file=sys.stderr)

    def _warn_span_outside(self, rf_freqs: np.ndarray) -> None:
        if self._warned_span:
            return
        self._warned_span = True
        low = float(np.min(rf_freqs)) if rf_freqs.size else 0.0
        high = float(np.max(rf_freqs)) if rf_freqs.size else 0.0
        ctx = self.baseline_ctx
        msg = (
            f"[baseline] sweep window {low/1e6:.3f}-{high/1e6:.3f} MHz outside baseline span "
            f"{ctx.freq_start_hz/1e6:.3f}-{ctx.freq_stop_hz/1e6:.3f} MHz"
        )
        print(msg, file=sys.stderr)
