"""Three-RTL guard/rover/reference role-run tests."""

from __future__ import annotations

import pytest

from tests.helpers_control import load_control_module
from tests.helpers_multi_rtl import FakePopen, rtl_devices, unsupported_device


def _three_role_manager(tmp_path, monkeypatch):
    FakePopen.calls.clear()
    FakePopen.next_pid = 4321
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    monkeypatch.setattr(control, "discover_devices", lambda: rtl_devices(control, ["S1", "S2", "S3"]))
    monkeypatch.setattr(control.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(control.time, "sleep", lambda _seconds: None)
    manager._spawn_reaper = lambda *_args, **_kwargs: None
    return control, manager


def _assign_three_lanes(manager) -> None:
    manager.set_role_assignment(
        "guard_primary",
        {
            "device_identity": "rtl:serial:S1",
            "role": "GUARD",
            "task": {"source_task": "guard_window", "start_hz": 101_100_000, "stop_hz": 103_500_000},
        },
    )
    manager.set_role_assignment(
        "guard_secondary",
        {
            "device_identity": "rtl:serial:S2",
            "role": "GUARD",
            "task": {"source_task": "guard_window", "start_hz": 460_000_000, "stop_hz": 462_400_000},
        },
    )
    manager.set_role_assignment(
        "reference",
        {
            "device_identity": "rtl:serial:S3",
            "role": "REFERENCE",
            "task": {"source_task": "reference_window", "start_hz": 98_000_000, "stop_hz": 100_400_000},
        },
    )


def test_three_rtl_inventory_supports_tier_2_plus_and_role_lanes(tmp_path, monkeypatch) -> None:
    _control, manager = _three_role_manager(tmp_path, monkeypatch)

    inventory = manager.hardware_inventory()
    assignments = manager.list_role_assignments()

    assert inventory["capability"]["tier"] == "2_plus"
    assert inventory["capability"]["max_role_count"] == 3
    assert set(inventory["capability"]["supported_roles"]) == {"GUARD", "ROVER", "REFERENCE"}
    assert assignments["supported_role_lanes"] == {
        "guard_primary": "Friendly Guard",
        "guard_secondary": "Watchlist Guard",
        "reference": "Reference",
        "rover": "Rover",
    }


def test_two_guard_plus_rover_start_uses_three_distinct_receivers(tmp_path, monkeypatch) -> None:
    _control, manager = _three_role_manager(tmp_path, monkeypatch)
    _assign_three_lanes(manager)
    manager.clear_role_assignment("reference")
    manager.set_role_assignment(
        "rover",
        {
            "device_identity": "rtl:serial:S3",
            "role": "ROVER",
            "task": {"source_task": "rover_sweep", "start_hz": 88_000_000, "stop_hz": 108_000_000},
        },
    )

    response = manager.start_role_run(
        {"baseline_id": 1, "role_lanes": ["guard_primary", "guard_secondary", "rover"]}
    )
    children = response["role_run"]["child_jobs"]

    assert response["role_run"]["status"] == "running"
    assert {child["role_lane"] for child in children} == {"guard_primary", "guard_secondary", "rover"}
    assert {child["receiver_role"] for child in children} == {"GUARD", "ROVER"}
    assert {child["device_key"] for child in children} == {"rtl:0", "rtl:1", "rtl:2"}


def test_two_guard_plus_reference_start_marks_reference_context_task(tmp_path, monkeypatch) -> None:
    _control, manager = _three_role_manager(tmp_path, monkeypatch)
    _assign_three_lanes(manager)

    response = manager.start_role_run(
        {"baseline_id": 1, "role_lanes": ["guard_primary", "guard_secondary", "reference"]}
    )
    children = response["role_run"]["child_jobs"]
    jobs_by_lane = {child["role_lane"]: manager.jobs[child["job_id"]] for child in children}

    assert response["role_run"]["active_device_count"] == 3
    assert jobs_by_lane["guard_primary"].source_task == "guard_window"
    assert jobs_by_lane["guard_secondary"].source_task == "guard_window"
    assert jobs_by_lane["reference"].receiver_role == "REFERENCE"
    assert jobs_by_lane["reference"].source_task == "reference_window"
    assert jobs_by_lane["reference"].device_index == 2
    assert jobs_by_lane["reference"].active_role_count == 3


def test_role_run_rejects_duplicate_physical_receiver_across_three_lanes(tmp_path, monkeypatch) -> None:
    control, manager = _three_role_manager(tmp_path, monkeypatch)
    _assign_three_lanes(manager)
    duplicate = rtl_devices(control, ["S1", "S2", "S3"])
    duplicate[1].extra["serial"] = "S1"
    monkeypatch.setattr(control, "discover_devices", lambda: duplicate)

    with pytest.raises(RuntimeError, match="not currently detected|same physical receiver|ambiguous"):
        manager.start_role_run({"baseline_id": 1, "role_lanes": ["guard_primary", "guard_secondary", "reference"]})


def test_reference_child_command_uses_parked_window_without_fusion_flags(tmp_path, monkeypatch) -> None:
    _control, manager = _three_role_manager(tmp_path, monkeypatch)
    _assign_three_lanes(manager)

    response = manager.start_role_run({"baseline_id": 1, "role_lanes": ["reference"]})
    child = response["role_run"]["child_jobs"][0]
    cmd = FakePopen.calls[-1]["cmd"]

    assert child["receiver_role"] == "REFERENCE"
    assert "--receiver-role" in cmd
    assert cmd[cmd.index("--receiver-role") + 1] == "REFERENCE"
    assert "--source-task" in cmd
    assert cmd[cmd.index("--source-task") + 1] == "reference_window"
    assert "--device-index" in cmd
    assert cmd[cmd.index("--device-index") + 1] == "2"
    assert "--enable-fusion" not in cmd
    assert "--environmental-correction" not in cmd


def test_three_device_inventory_keeps_future_hardware_non_runnable(tmp_path, monkeypatch) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    monkeypatch.setattr(
        control,
        "discover_devices",
        lambda: [
            *rtl_devices(control, ["S1", "S2", "S3"]),
            unsupported_device(control, "airspy:0", "airspy"),
            unsupported_device(control, "hackrf:0", "hackrf"),
            unsupported_device(control, "soapy:0", "soapy"),
        ],
    )

    payload = manager.hardware_inventory()

    assert payload["capability"]["tier"] == "2_plus"
    assert all(row["runnable"] is False for row in payload["unsupported_hardware"])
    assert {row["hardware_kind"] for row in payload["unsupported_hardware"]} == {"airspy", "hackrf", "soapy"}
