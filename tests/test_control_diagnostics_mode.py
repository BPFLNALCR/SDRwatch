"""Controller diagnostic mode tests that do not require SDR hardware."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any


def _load_control_module() -> Any:
    path = Path(__file__).resolve().parents[1] / "sdrwatch-control.py"
    spec = importlib.util.spec_from_file_location("sdrwatch_control_for_tests", path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_diagnostics_mode_generates_safe_per_job_path() -> None:
    control = _load_control_module()

    params = control.normalize_diagnostic_params(
        "job/with unsafe chars",
        {"start": 88000000, "diagnostics_mode": True},
    )

    assert params["diagnostic_jsonl"].endswith("job-with-unsafe-chars.diagnostic.jsonl")
    assert "/" not in Path(params["diagnostic_jsonl"]).name


def test_explicit_diagnostic_jsonl_is_preserved() -> None:
    control = _load_control_module()

    params = control.normalize_diagnostic_params(
        "abc123",
        {"diagnostics_mode": True, "diagnostic_jsonl": "custom.jsonl"},
    )

    assert params["diagnostic_jsonl"] == "custom.jsonl"


def test_disabled_diagnostics_do_not_add_path() -> None:
    control = _load_control_module()

    params = control.normalize_diagnostic_params("abc123", {"start": 88000000})

    assert "diagnostic_jsonl" not in params


def test_generated_diagnostic_path_reaches_scanner_command() -> None:
    control = _load_control_module()
    params = control.normalize_diagnostic_params("abc123", {"diagnostics_mode": True})
    manager = object.__new__(control.JobManager)

    cmd = manager._build_cmd(
        script_path=None,
        device_key="rtl:0",
        baseline_id=1,
        args=params,
    )

    assert "--diagnostic-jsonl" in cmd
    path = cmd[cmd.index("--diagnostic-jsonl") + 1]
    assert path.endswith("abc123.diagnostic.jsonl")
