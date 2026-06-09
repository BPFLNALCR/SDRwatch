"""Deterministic no-hardware SDR source for development and testing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np  # type: ignore


@dataclass(frozen=True)
class SyntheticSignal:
    """Absolute-frequency synthetic signal definition."""

    center_hz: float
    base_amplitude: float
    offsets_hz: Sequence[float]
    weights: Sequence[float]
    mode: Literal["stable", "intermittent", "power_shift"] = "stable"


class SimulatedSource:
    """Generate deterministic complex IQ samples without SDR hardware."""

    _NOISE_STD = 0.004
    _POWER_SHIFT_CYCLE = (0.45, 1.0, 0.65, 0.25)
    _SIGNALS = (
        SyntheticSignal(
            center_hz=88_300_000.0,
            base_amplitude=0.22,
            offsets_hz=(-75_000.0, -25_000.0, 0.0, 25_000.0, 75_000.0),
            weights=(0.18, 0.45, 1.0, 0.45, 0.18),
            mode="stable",
        ),
        SyntheticSignal(
            center_hz=89_100_000.0,
            base_amplitude=0.18,
            offsets_hz=(-60_000.0, -20_000.0, 0.0, 20_000.0, 60_000.0),
            weights=(0.2, 0.5, 1.0, 0.5, 0.2),
            mode="power_shift",
        ),
        SyntheticSignal(
            center_hz=90_500_000.0,
            base_amplitude=0.16,
            offsets_hz=(-55_000.0, -18_000.0, 0.0, 18_000.0, 55_000.0),
            weights=(0.18, 0.45, 1.0, 0.45, 0.18),
            mode="intermittent",
        ),
        SyntheticSignal(
            center_hz=91_100_000.0,
            base_amplitude=0.2,
            offsets_hz=(-65_000.0, -22_000.0, 0.0, 22_000.0, 65_000.0),
            weights=(0.18, 0.5, 1.0, 0.5, 0.18),
            mode="stable",
        ),
        SyntheticSignal(
            center_hz=98_700_000.0,
            base_amplitude=0.19,
            offsets_hz=(-70_000.0, -24_000.0, 0.0, 24_000.0, 70_000.0),
            weights=(0.18, 0.45, 1.0, 0.45, 0.18),
            mode="stable",
        ),
        SyntheticSignal(
            center_hz=103_900_000.0,
            base_amplitude=0.21,
            offsets_hz=(-75_000.0, -25_000.0, 0.0, 25_000.0, 75_000.0),
            weights=(0.18, 0.45, 1.0, 0.45, 0.18),
            mode="stable",
        ),
    )

    def __init__(self, samp_rate: float, gain: str | float, *, seed: int = 13_037) -> None:
        self.samp_rate = float(samp_rate)
        self.gain = gain
        self.device = "Simulated SDR"
        self._rng = np.random.RandomState(int(seed))
        self._current_center_hz = 0.0
        self._last_center_hz: float | None = None
        self._sample_cursor = 0
        self._sweep_index = 0
        self._window_index = -1

    def tune(self, center_hz: float) -> None:
        center_hz = float(center_hz)
        if self._last_center_hz is not None and center_hz + 1.0 < self._last_center_hz:
            self._sweep_index += 1
            self._window_index = 0
        elif self._last_center_hz is None:
            self._window_index = 0
        else:
            self._window_index += 1
        self._current_center_hz = center_hz
        self._last_center_hz = center_hz
        self._sample_cursor = 0

    def _signal_amplitude(self, signal: SyntheticSignal) -> float:
        if signal.mode == "stable":
            return signal.base_amplitude
        if signal.mode == "intermittent":
            return signal.base_amplitude if (self._sweep_index % 2 == 0) else 0.0
        scale = self._POWER_SHIFT_CYCLE[self._sweep_index % len(self._POWER_SHIFT_CYCLE)]
        return signal.base_amplitude * scale

    def _tone_series(self, tone_hz: float, amplitude: float, sample_idx: np.ndarray) -> np.ndarray:
        phase = 2.0 * np.pi * tone_hz * sample_idx / self.samp_rate
        return (amplitude * np.exp(1j * phase)).astype(np.complex64)

    def read(self, count: int) -> np.ndarray:
        count = int(count)
        if count <= 0:
            return np.zeros(0, dtype=np.complex64)

        real = self._rng.normal(0.0, self._NOISE_STD, count)
        imag = self._rng.normal(0.0, self._NOISE_STD, count)
        samples = (real + 1j * imag).astype(np.complex64)

        if self._last_center_hz is None:
            self._sample_cursor += count
            return samples

        half_span_hz = self.samp_rate / 2.0
        sample_idx = np.arange(self._sample_cursor, self._sample_cursor + count, dtype=np.float64)
        for signal in self._SIGNALS:
            amplitude = self._signal_amplitude(signal)
            if amplitude <= 0.0:
                continue
            rel_center_hz = signal.center_hz - self._current_center_hz
            if abs(rel_center_hz) > half_span_hz + max(abs(offset) for offset in signal.offsets_hz):
                continue
            for offset_hz, weight in zip(signal.offsets_hz, signal.weights, strict=True):
                tone_hz = rel_center_hz + float(offset_hz)
                if abs(tone_hz) > half_span_hz:
                    continue
                samples += self._tone_series(tone_hz, amplitude * float(weight), sample_idx)

        self._sample_cursor += count
        return samples.astype(np.complex64, copy=False)

    def close(self) -> None:
        return None