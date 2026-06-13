"""Shared helpers for no-hardware controller command tests."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def load_control_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "sdrwatch-control.py"
    spec = importlib.util.spec_from_file_location("sdrwatch_control_for_tests", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def build_scanner_cmd(params: dict[str, Any], *, device_key: str = "rtl:0") -> list[str]:
    control = load_control_module()
    manager = object.__new__(control.JobManager)
    return manager._build_cmd(
        script_path=None,
        device_key=device_key,
        baseline_id=1,
        args=params,
    )
