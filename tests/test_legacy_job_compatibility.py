"""Legacy single-device job compatibility tests."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pytest

from tests.helpers_control import load_control_module
from tests.helpers_multi_rtl import FakePopen, rtl_device


def _cmd_value(cmd: list[str], flag: str) -> str:
    index = cmd.index(flag)
    return cmd[index + 1]


def test_legacy_single_device_job_uses_native_rtl_without_role_flags(tmp_path, monkeypatch: Any) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    FakePopen.calls = []
    monkeypatch.setattr(control.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(control, "discover_devices", lambda: [rtl_device(control, 0, "00000001")])

    job = manager.start_job(
        device_key="rtl:0",
        label="web",
        baseline_id=1,
        sdrwatch_args={"start": 88_000_000, "stop": 108_000_000, "profile": "fm_broadcast"},
    )

    assert FakePopen.calls
    cmd = FakePopen.calls[-1]["cmd"]
    assert _cmd_value(cmd, "--driver") == "rtlsdr_native"
    assert _cmd_value(cmd, "--device-key") == "rtl:0"
    assert _cmd_value(cmd, "--profile") == "fm_broadcast"
    for flag in ("--role-run-id", "--receiver-role", "--role-lane", "--source-task"):
        assert flag not in cmd

    payload = asdict(job)
    assert payload["device_key"] == "rtl:0"
    assert payload["receiver_role"] is None
    assert payload["role_lane"] is None
    assert payload["role_run_id"] is None
    assert payload["source_task"] is None


def test_legacy_jobs_route_preserves_request_shape_and_null_role_metadata(tmp_path, monkeypatch: Any) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    FakePopen.calls = []
    monkeypatch.setattr(control.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(control, "discover_devices", lambda: [rtl_device(control, 0, "00000001")])
    app = control.make_app(manager)

    response = app.test_client().post(
        "/jobs",
        json={
            "device_key": "rtl:0",
            "label": "web",
            "baseline_id": 1,
            "params": {"start": 88_000_000, "stop": 108_000_000},
        },
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["device_key"] == "rtl:0"
    assert data["label"] == "web"
    assert data["baseline_id"] == 1
    assert data["receiver_role"] is None
    assert data["role_lane"] is None
    assert data["role_run_id"] is None
    assert data["source_task"] is None


@pytest.mark.parametrize("device_key", ["hackrf:0", "airspy:0", "soapy:0"])
def test_unsupported_legacy_job_devices_are_rejected_before_spawn(tmp_path, monkeypatch: Any, device_key: str) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    FakePopen.calls = []
    monkeypatch.setattr(control.subprocess, "Popen", FakePopen)

    with pytest.raises(control.UnsupportedBackendError) as excinfo:
        manager.start_job(
            device_key=device_key,
            label="web",
            baseline_id=1,
            sdrwatch_args={"start": 88_000_000, "stop": 108_000_000},
        )

    assert excinfo.value.payload["spawned"] is False
    assert FakePopen.calls == []
