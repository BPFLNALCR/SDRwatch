"""Window scheduling helpers for sweep orchestrations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, List


@dataclass(frozen=True)
class SweepWindow:
    """Single sweep window emitted by the scheduler."""

    index: int
    center_hz: float
    start_hz: float
    stop_hz: float


class WindowScheduler:
    """Generate ordered sweep windows for a given span."""

    def __init__(self, start_hz: float, stop_hz: float, step_hz: float) -> None:
        if step_hz <= 0:
            raise ValueError("step_hz must be positive")
        if stop_hz < start_hz:
            raise ValueError("stop_hz must be >= start_hz")
        self._start_hz = float(start_hz)
        self._stop_hz = float(stop_hz)
        self._step_hz = float(step_hz)

    def __iter__(self) -> Iterator[SweepWindow]:
        center = self._start_hz
        idx = 0
        while center <= self._stop_hz + 1e-6:
            half_span = self._step_hz / 2.0
            yield SweepWindow(index=idx, center_hz=center, start_hz=center - half_span, stop_hz=center + half_span)
            idx += 1
            center += self._step_hz

    def windows(self) -> List[SweepWindow]:
        """Eagerly materialize the scheduled windows."""

        return list(iter(self))

    @property
    def count(self) -> int:
        """Return the number of windows implied by the schedule."""

        span = max(self._stop_hz - self._start_hz, 0.0)
        return int(span // self._step_hz) + 1
