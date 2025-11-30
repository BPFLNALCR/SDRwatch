"""Native librtlsdr (pyrtlsdr) driver wrapper."""

from __future__ import annotations

from typing import Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    from rtlsdr import RtlSdr  # type: ignore

    HAVE_RTLSDR = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_RTLSDR = False
    RtlSdr = None  # type: ignore


class RTLSDRSource:
    """Convenience wrapper around pyrtlsdr.RtlSdr."""

    def __init__(self, samp_rate: float, gain: str | float, *, device_index: Optional[int] = None, serial_number: Optional[str] = None):
        if not HAVE_RTLSDR:
            raise RuntimeError("pyrtlsdr not available")
        if serial_number:
            self.dev = RtlSdr(serial_number=str(serial_number))  # type: ignore[call-arg]
        elif device_index is not None:
            self.dev = RtlSdr(device_index=int(device_index))  # type: ignore[call-arg]
        else:
            self.dev = RtlSdr()  # type: ignore[call-arg]
        self.dev.sample_rate = samp_rate
        if isinstance(gain, str) and gain == "auto":
            self.dev.gain = "auto"
        else:
            self.dev.gain = float(gain)

    def tune(self, center_hz: float) -> None:
        self.dev.center_freq = center_hz

    def read(self, count: int) -> np.ndarray:
        return self.dev.read_samples(count)

    def close(self) -> None:
        try:
            self.dev.close()
        except Exception:
            pass
