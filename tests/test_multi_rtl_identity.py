"""Receiver identity resolution tests for multi-RTL inventory."""

from __future__ import annotations

from tests.helpers_control import load_control_module
from tests.helpers_multi_rtl import rtl_devices


def test_unique_serial_produces_stable_persistent_identity(tmp_path, monkeypatch) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    monkeypatch.setattr(control, "discover_devices", lambda: rtl_devices(control, ["00000001"]))

    device = manager.hardware_inventory()["devices"][0]

    assert device["device_identity"] == "rtl:serial:00000001"
    assert device["identity_confidence"] == "stable"
    assert device["identity_scope"] == "persistent"
    assert device["warnings"] == []


def test_missing_and_duplicate_serials_are_unstable(tmp_path, monkeypatch) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    monkeypatch.setattr(control, "discover_devices", lambda: rtl_devices(control, [None, "DUP", "DUP"]))

    devices = {row["legacy_device_key"]: row for row in manager.hardware_inventory()["devices"]}

    assert devices["rtl:0"]["device_identity"] == "rtl:index:0"
    assert devices["rtl:0"]["identity_confidence"] == "index_only"
    assert "index_only_identity" in [w["code"] for w in devices["rtl:0"]["warnings"]]
    assert devices["rtl:1"]["identity_confidence"] == "ambiguous"
    assert devices["rtl:2"]["identity_confidence"] == "ambiguous"
    assert "duplicate_serial" in [w["code"] for w in devices["rtl:1"]["warnings"]]
