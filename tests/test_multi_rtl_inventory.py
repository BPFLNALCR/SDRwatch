"""No-hardware inventory and capability tests for multi-RTL mode."""

from __future__ import annotations

from dataclasses import asdict

from tests.helpers_control import load_control_module
from tests.helpers_multi_rtl import rtl_devices, unsupported_device


def _inventory_for(tmp_path, monkeypatch, serials):
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    monkeypatch.setattr(control, "discover_devices", lambda: rtl_devices(control, serials))
    return control, manager.hardware_inventory()


def test_zero_one_two_three_rtl_inventory_tiers(tmp_path, monkeypatch) -> None:
    expected = [
        ([], "0", 0),
        (["00000001"], "1", 1),
        (["00000001", "00000002"], "2", 2),
        (["00000001", "00000002", "00000003"], "2_plus", 3),
    ]

    for idx, (serials, tier, count) in enumerate(expected):
        _control, payload = _inventory_for(tmp_path / str(idx), monkeypatch, serials)
        assert payload["capability"]["tier"] == tier
        assert payload["capability"]["runnable_rtl_count"] == count
        assert len(payload["devices"]) == count


def test_missing_duplicate_and_index_only_identity_warnings(tmp_path, monkeypatch) -> None:
    _control, payload = _inventory_for(tmp_path, monkeypatch, [None, "DUP", "DUP", "UNIQUE"])
    by_key = {row["legacy_device_key"]: row for row in payload["devices"]}

    missing = by_key["rtl:0"]
    assert missing["identity_confidence"] == "index_only"
    assert missing["identity_scope"] == "session"
    assert missing["device_identity"] == "rtl:index:0"
    assert "missing_serial" in [w["code"] for w in missing["warnings"]]
    assert "index_only_identity" in [w["code"] for w in missing["warnings"]]

    duplicate = by_key["rtl:1"]
    assert duplicate["identity_confidence"] == "ambiguous"
    assert duplicate["identity_scope"] == "session"
    assert "duplicate_serial" in [w["code"] for w in duplicate["warnings"]]

    stable = by_key["rtl:3"]
    assert stable["identity_confidence"] == "stable"
    assert stable["identity_scope"] == "persistent"
    assert stable["device_identity"] == "rtl:serial:UNIQUE"
    assert stable["warnings"] == []


def test_unsupported_hardware_classes_are_non_runnable(tmp_path, monkeypatch) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    monkeypatch.setattr(
        control,
        "discover_devices",
        lambda: [
            *rtl_devices(control, ["00000001"]),
            unsupported_device(control, "hackrf:0", "hackrf"),
            unsupported_device(control, "airspy:0", "airspy"),
            unsupported_device(control, "soapy:0", "soapy"),
        ],
    )

    payload = manager.hardware_inventory()

    assert payload["devices"][0]["runnable_backend"] == "rtlsdr_native"
    unsupported = {row["hardware_kind"]: row for row in payload["unsupported_hardware"]}
    for kind in ("hackrf", "airspy", "soapy"):
        assert unsupported[kind]["runnable"] is False
        assert unsupported[kind]["runnable_backend"] is None
        assert unsupported[kind]["support_state"] in {"planned", "unsupported"}
    assert unsupported["hackrf"]["detected"] is True


def test_hardware_inventory_route_and_legacy_devices_compatibility(tmp_path, monkeypatch) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    devices = rtl_devices(control, ["00000001", "00000002"])
    monkeypatch.setattr(control, "discover_devices", lambda: devices)
    app = control.make_app(manager)

    inventory_response = app.test_client().get("/hardware/inventory")
    assert inventory_response.status_code == 200
    payload = inventory_response.get_json()
    assert payload["capability"]["tier"] == "2"

    devices_response = app.test_client().get("/devices")
    assert devices_response.status_code == 200
    assert devices_response.get_json() == [asdict(dev) for dev in devices]
