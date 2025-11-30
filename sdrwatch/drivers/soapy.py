"""SoapySDR-backed SDR source wrapper."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

try:  # pragma: no cover - optional dependency
    import SoapySDR  # type: ignore
    from SoapySDR import SOAPY_SDR_CF32, SOAPY_SDR_RX  # type: ignore

    HAVE_SOAPY = True
except Exception:  # pragma: no cover - optional dependency
    HAVE_SOAPY = False
    SoapySDR = None  # type: ignore
    SOAPY_SDR_CF32 = 0  # type: ignore
    SOAPY_SDR_RX = 0  # type: ignore


class SDRSource:
    """Thin convenience wrapper around SoapySDR.Device."""

    def __init__(self, driver: str, samp_rate: float, gain: str | float, soapy_args: Optional[Dict[str, str]] = None):
        if not HAVE_SOAPY:
            raise RuntimeError("SoapySDR not available")
        dev_args: Dict[str, str] = {"driver": driver}
        if soapy_args:
            dev_args.update({str(k): str(v) for k, v in soapy_args.items()})
        self.dev = SoapySDR.Device(dev_args)  # type: ignore[call-arg]
        self.dev.setSampleRate(SOAPY_SDR_RX, 0, samp_rate)
        if isinstance(gain, str) and gain == "auto":
            try:
                self.dev.setGainMode(SOAPY_SDR_RX, 0, True)
            except Exception:
                pass
        else:
            self.dev.setGain(SOAPY_SDR_RX, 0, float(gain))
        self.stream = self.dev.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CF32)
        self.dev.activateStream(self.stream)

    def tune(self, center_hz: float) -> None:
        self.dev.setFrequency(SOAPY_SDR_RX, 0, center_hz)

    def read(self, count: int) -> np.ndarray:
        buffs: List[np.ndarray] = []
        got = 0
        while got < count:
            sr = int(min(8192, count - got))
            buff = np.empty(sr, dtype=np.complex64)
            st = self.dev.readStream(self.stream, [buff], sr)
            n = getattr(st, "ret", st)
            if isinstance(n, tuple):
                n = n[0]
            if isinstance(n, (list, np.ndarray)):
                n = int(n[0])
            if int(n) > 0:
                buffs.append(buff[: int(n)])
                got += int(n)
            else:
                time.sleep(0.001)
        if not buffs:
            return np.zeros(count, dtype=np.complex64)
        return np.concatenate(buffs)

    def close(self) -> None:
        try:
            self.dev.deactivateStream(self.stream)
            self.dev.closeStream(self.stream)
        except Exception:
            pass
