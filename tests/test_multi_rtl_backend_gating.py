"""Backend gating tests for RTL-only scanner execution."""

from __future__ import annotations

import pytest

from tests.helpers_control import load_control_module
from tests.helpers_multi_rtl import FakePopen


@pytest.mark.parametrize("device_key", ["hackrf:0", "airspy:0", "soapy:0"])
def test_unsupported_device_starts_are_rejected_before_spawn(tmp_path, monkeypatch, device_key) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    monkeypatch.setattr(control.subprocess, "Popen", FakePopen)

    with pytest.raises(control.UnsupportedBackendError) as excinfo:
        manager.start_job(
            device_key=device_key,
            label="bad",
            baseline_id=1,
            sdrwatch_args={"start": 88e6, "stop": 90e6},
        )

    payload = excinfo.value.payload
    assert payload["error"] == "unsupported_backend"
    assert payload["spawned"] is False
    assert payload["supported_backends"] == ["rtlsdr_native"]
    assert FakePopen.calls == []


def test_explicit_non_native_driver_is_rejected_before_spawn(tmp_path, monkeypatch) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    monkeypatch.setattr(control.subprocess, "Popen", FakePopen)

    with pytest.raises(control.UnsupportedBackendError) as excinfo:
        manager.start_job(
            device_key="rtl:0",
            label="bad",
            baseline_id=1,
            sdrwatch_args={"driver": "soapy", "start": 88e6, "stop": 90e6},
        )

    assert excinfo.value.payload["requested_backend"] == "soapy"
    assert excinfo.value.payload["spawned"] is False
    assert FakePopen.calls == []


def test_unsupported_backend_route_payload_reports_not_spawned(tmp_path) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    app = control.make_app(manager)

    response = app.test_client().post(
        "/jobs",
        json={"device_key": "hackrf:0", "baseline_id": 1, "label": "bad", "params": {}},
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "unsupported_backend"
    assert response.get_json()["spawned"] is False
