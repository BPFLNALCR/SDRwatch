"""Grouped GUARD+ROVER role-run tests."""

from __future__ import annotations

import pytest

from tests.helpers_control import load_control_module
from tests.helpers_multi_rtl import FakePopen, rtl_devices


def _two_role_manager(tmp_path, monkeypatch):
    FakePopen.calls.clear()
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    monkeypatch.setattr(control, "discover_devices", lambda: rtl_devices(control, ["S1", "S2"]))
    monkeypatch.setattr(control.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(control.time, "sleep", lambda _seconds: None)
    manager._spawn_reaper = lambda *_args, **_kwargs: None
    manager.set_role_assignment(
        "guard_primary",
        {
            "device_identity": "rtl:serial:S1",
            "role": "GUARD",
            "task": {"source_task": "guard_window", "start_hz": 101_100_000, "stop_hz": 103_500_000},
        },
    )
    manager.set_role_assignment(
        "rover",
        {
            "device_identity": "rtl:serial:S2",
            "role": "ROVER",
            "task": {"source_task": "rover_sweep", "start_hz": 88_000_000, "stop_hz": 108_000_000},
        },
    )
    return control, manager


def test_role_run_controller_contract_routes(tmp_path, monkeypatch) -> None:
    control, manager = _two_role_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(control, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(control.os, "kill", lambda _pid, _sig: None)
    app = control.make_app(manager)
    client = app.test_client()

    start = client.post("/role-runs", json={"baseline_id": 1, "role_lanes": ["guard_primary", "rover"]})
    assert start.status_code == 201
    role_run_id = start.get_json()["role_run"]["role_run_id"]

    listed = client.get("/role-runs")
    assert listed.status_code == 200
    assert listed.get_json()["role_runs"][0]["role_run_id"] == role_run_id

    detail = client.get(f"/role-runs/{role_run_id}")
    assert detail.status_code == 200
    assert len(detail.get_json()["role_run"]["child_jobs"]) == 2

    stopped = client.delete(f"/role-runs/{role_run_id}")
    assert stopped.status_code == 200
    assert stopped.get_json()["role_run"]["status"] == "finished"


def test_guard_rover_grouped_start_uses_distinct_current_devices(tmp_path, monkeypatch) -> None:
    _control, manager = _two_role_manager(tmp_path, monkeypatch)

    response = manager.start_role_run({"baseline_id": 1, "role_lanes": ["guard_primary", "rover"]})
    children = response["role_run"]["child_jobs"]

    assert response["role_run"]["status"] == "running"
    assert {child["receiver_role"] for child in children} == {"GUARD", "ROVER"}
    assert {child["device_key"] for child in children} == {"rtl:0", "rtl:1"}
    assert {child["device_identity"] for child in children} == {"rtl:serial:S1", "rtl:serial:S2"}


def test_serial_assignments_refresh_to_current_runtime_index(tmp_path, monkeypatch) -> None:
    control, manager = _two_role_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(control, "discover_devices", lambda: rtl_devices(control, ["S2", "S1"]))

    response = manager.start_role_run({"baseline_id": 1, "role_lanes": ["guard_primary"]})
    child = response["role_run"]["child_jobs"][0]
    job = manager.jobs[child["job_id"]]

    assert child["device_key"] == "rtl:1"
    assert job.device_index == 1


def test_role_run_rejects_missing_duplicated_or_ambiguous_serial(tmp_path, monkeypatch) -> None:
    control, manager = _two_role_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(control, "discover_devices", lambda: rtl_devices(control, ["S2", "S2"]))

    with pytest.raises(RuntimeError):
        manager.start_role_run({"baseline_id": 1, "role_lanes": ["guard_primary"]})


def test_partial_startup_degrades_when_one_child_starts(tmp_path, monkeypatch) -> None:
    _control, manager = _two_role_manager(tmp_path, monkeypatch)
    original_start_job = manager.start_job
    calls = {"count": 0}

    def flaky_start_job(**kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise RuntimeError("device locked")
        return original_start_job(**kwargs)

    manager.start_job = flaky_start_job

    response = manager.start_role_run({"baseline_id": 1, "role_lanes": ["guard_primary", "rover"]})
    role_run = response["role_run"]

    assert role_run["status"] == "degraded"
    assert len(role_run["child_jobs"]) == 2
    assert role_run["child_jobs"][1]["error_message"] == "device locked"


def test_no_child_start_returns_conventional_error(tmp_path, monkeypatch) -> None:
    _control, manager = _two_role_manager(tmp_path, monkeypatch)
    manager.start_job = lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("no spawn"))

    with pytest.raises(RuntimeError):
        manager.start_role_run({"baseline_id": 1, "role_lanes": ["guard_primary"]})


def test_grouped_stop_stops_children_and_releases_locks(tmp_path, monkeypatch) -> None:
    control, manager = _two_role_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(control, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(control.os, "kill", lambda _pid, _sig: None)
    response = manager.start_role_run({"baseline_id": 1, "role_lanes": ["guard_primary", "rover"]})
    role_run_id = response["role_run"]["role_run_id"]

    stopped = manager.stop_role_run(role_run_id)["role_run"]

    assert stopped["status"] == "finished"
    assert all(child["status"] == "finished" for child in stopped["child_jobs"])
    assert not manager._lock_path("rtl:0").exists()
    assert not manager._lock_path("rtl:1").exists()


def test_child_direct_stop_degrades_parent_role_run(tmp_path, monkeypatch) -> None:
    control, manager = _two_role_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(control, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(control.os, "kill", lambda _pid, _sig: None)
    response = manager.start_role_run({"baseline_id": 1, "role_lanes": ["guard_primary", "rover"]})
    role_run_id = response["role_run"]["role_run_id"]
    first_job_id = response["role_run"]["child_jobs"][0]["job_id"]

    manager.stop_job(first_job_id)
    refreshed = manager.get_role_run(role_run_id)["role_run"]

    assert refreshed["status"] == "degraded"
