"""Shared fake-device helpers for multi-RTL controller tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


def rtl_device(control: Any, index: int, serial: str | None = None, *, label: str | None = None) -> Any:
    return control.Device(
        key=f"rtl:{index}",
        kind="rtlsdr",
        label=label or f"RTL-SDR #{index}" + (f" (SN {serial})" if serial else ""),
        extra={"index": index, "serial": serial},
    )


def rtl_devices(control: Any, serials: Iterable[str | None]) -> list[Any]:
    return [rtl_device(control, idx, serial) for idx, serial in enumerate(serials)]


def unsupported_device(control: Any, key: str, kind: str, *, serial: str | None = None) -> Any:
    return control.Device(
        key=key,
        kind=kind,
        label=f"{kind.upper()} test device",
        extra={"serial": serial},
    )


def fake_lock_owner(path: Path, owner: str, *, age_seconds: float | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(owner, encoding="utf-8")
    if age_seconds is not None:
        import os
        import time

        ts = time.time() - age_seconds
        os.utime(path, (ts, ts))


@dataclass
class FakeProcess:
    pid: int = 4321
    returncode: int = 0
    waited: bool = False

    def wait(self) -> int:
        self.waited = True
        return self.returncode


class FakePopen:
    """subprocess.Popen stand-in that records the latest command."""

    calls: list[dict[str, Any]] = []
    next_pid = 4321

    def __new__(cls, cmd: list[str], **kwargs: Any) -> FakeProcess:
        proc = FakeProcess(pid=cls.next_pid)
        cls.next_pid += 1
        cls.calls.append({"cmd": list(cmd), "kwargs": dict(kwargs), "process": proc})
        return proc
