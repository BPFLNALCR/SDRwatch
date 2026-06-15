"""Shared helpers for no-hardware controller command tests."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any


def load_control_module(control_base: Path | None = None, *, module_name: str | None = None) -> Any:
    old_base = os.environ.get("SDRWATCH_CONTROL_BASE")
    if control_base is not None:
        os.environ["SDRWATCH_CONTROL_BASE"] = str(control_base)
    path = Path(__file__).resolve().parents[1] / "sdrwatch-control.py"
    spec_name = module_name or (
        "sdrwatch_control_for_tests"
        if control_base is None
        else f"sdrwatch_control_for_tests_{abs(hash(str(control_base))) & 0xFFFFFFFF:x}"
    )
    spec = importlib.util.spec_from_file_location(spec_name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    try:
        spec.loader.exec_module(module)
    finally:
        if control_base is not None:
            if old_base is None:
                os.environ.pop("SDRWATCH_CONTROL_BASE", None)
            else:
                os.environ["SDRWATCH_CONTROL_BASE"] = old_base
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
