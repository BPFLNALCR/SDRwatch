"""Native librtlsdr (pyrtlsdr) driver wrapper."""

from __future__ import annotations

import importlib
import sys
import types
from importlib import metadata as importlib_metadata
from typing import Dict, List, Optional

import numpy as np # type: ignore

from sdrwatch.util.logging import get_logger

_log = get_logger(__name__)


def _install_pkg_resources_shim() -> None:
    """Provide a minimal pkg_resources shim for pyrtlsdr on Python 3.13+ environments.

    Some pyrtlsdr releases import pkg_resources only to query distribution metadata.
    Modern environments may omit pkg_resources even when setuptools is present/trimmed.
    """
    if "pkg_resources" in sys.modules:
        return
    shim = types.ModuleType("pkg_resources")

    class DistributionNotFound(Exception):
        pass

    class Distribution:
        def __init__(self, version: str):
            self.version = version

    def get_distribution(name: str) -> Distribution:
        try:
            version = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError as exc:  # pragma: no cover - env dependent
            raise DistributionNotFound(str(exc))
        return Distribution(version)

    shim.DistributionNotFound = DistributionNotFound  # type: ignore[attr-defined]
    shim.get_distribution = get_distribution  # type: ignore[attr-defined]
    sys.modules["pkg_resources"] = shim


def _import_rtlsdr_module():
    """Import pyrtlsdr with a compatibility fallback for missing pkg_resources."""
    try:
        return importlib.import_module("rtlsdr"), None
    except ModuleNotFoundError as exc:
        if exc.name != "pkg_resources":
            return None, exc
        try:
            _install_pkg_resources_shim()
            return importlib.import_module("rtlsdr"), None
        except Exception as inner_exc:  # pragma: no cover - env dependent
            return None, inner_exc
    except Exception as exc:  # pragma: no cover - env dependent
        return None, exc

try:  # pragma: no cover - optional dependency
    _rtlsdr_mod, _rtlsdr_err = _import_rtlsdr_module()
    if _rtlsdr_mod is None:
        raise _rtlsdr_err if _rtlsdr_err else RuntimeError("failed to import pyrtlsdr")
    RtlSdr = getattr(_rtlsdr_mod, "RtlSdr")  # type: ignore[misc]
    HAVE_RTLSDR = True
    RTLSDR_IMPORT_ERROR: Optional[str] = None
except Exception as exc:  # pragma: no cover - optional dependency
    HAVE_RTLSDR = False
    RTLSDR_IMPORT_ERROR = str(exc)
    RtlSdr = None  # type: ignore


def enumerate_rtlsdr_devices(max_devices: int = 8) -> List[Dict[str, Optional[str]]]:
    """Return lightweight RTL device metadata using native pyrtlsdr backend."""
    if not HAVE_RTLSDR:
        return []
    devices: List[Dict[str, Optional[str]]] = []
    for i in range(max_devices):
        try:
            sdr = RtlSdr(i)  # type: ignore[call-arg]
            serial = getattr(sdr, "serial_number", None)
            devices.append({"index": str(i), "serial": serial})
            sdr.close()
            del sdr
        except Exception:
            break
    return devices


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
            _log.debug("rtlsdr close error (ignored)", exc_info=True)
