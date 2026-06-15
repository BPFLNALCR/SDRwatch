"""One-device GUARD role-run tests."""

from __future__ import annotations

from tests.helpers_control import load_control_module
from tests.helpers_multi_rtl import FakePopen, rtl_devices


def _guard_manager(tmp_path, monkeypatch):
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    monkeypatch.setattr(control, "discover_devices", lambda: rtl_devices(control, ["S1"]))
    monkeypatch.setattr(control.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(control.time, "sleep", lambda _seconds: None)
    manager._spawn_reaper = lambda *_args, **_kwargs: None
    manager.set_role_assignment(
        "guard_primary",
        {
            "device_identity": "rtl:serial:S1",
            "role": "GUARD",
            "task": {
                "source_task": "guard_window",
                "start_hz": 101_100_000,
                "stop_hz": 103_500_000,
                "profile": "fm_broadcast",
            },
        },
    )
    return control, manager


def test_one_device_guard_role_run_starts_child_job(tmp_path, monkeypatch) -> None:
    _control, manager = _guard_manager(tmp_path, monkeypatch)

    response = manager.start_role_run(
        {
            "label": "guard",
            "baseline_id": 1,
            "role_lanes": ["guard_primary"],
            "params": {"diagnostics_mode": True},
        }
    )

    role_run = response["role_run"]
    assert role_run["status"] == "running"
    assert role_run["active_device_count"] == 1
    assert role_run["child_jobs"][0]["receiver_role"] == "GUARD"
    assert role_run["child_jobs"][0]["device_identity"] == "rtl:serial:S1"
    assert role_run["child_jobs"][0]["status"] == "running"


def test_guard_role_uses_narrow_window_parameters(tmp_path, monkeypatch) -> None:
    _control, manager = _guard_manager(tmp_path, monkeypatch)

    response = manager.start_role_run({"baseline_id": 1, "role_lanes": ["guard_primary"]})
    job = manager.jobs[response["role_run"]["child_jobs"][0]["job_id"]]

    assert job.params["start"] == 101_100_000
    assert job.params["stop"] == 103_500_000
    assert job.params["step"] == 2_400_000
    assert job.params["profile"] == "fm_broadcast"
    assert job.params["source_task"] == "guard_window"
    assert "--driver" in job.cmd
    assert job.cmd[job.cmd.index("--driver") + 1] == "rtlsdr_native"


def test_guard_child_job_metadata_is_serialized(tmp_path, monkeypatch) -> None:
    _control, manager = _guard_manager(tmp_path, monkeypatch)
    response = manager.start_role_run({"baseline_id": 1, "role_lanes": ["guard_primary"]})
    child = response["role_run"]["child_jobs"][0]
    job = manager.jobs[child["job_id"]]

    assert job.receiver_role == "GUARD"
    assert job.role_lane == "guard_primary"
    assert job.role_run_id == response["role_run"]["role_run_id"]
    assert job.source_task == "guard_window"
    assert job.device_identity == "rtl:serial:S1"
    assert job.device_serial == "S1"
    assert job.device_index == 0
    assert job.identity_confidence == "stable"
    assert job.active_device_count == 1
    assert job.active_role_count == 1


def test_guard_metadata_does_not_change_existing_fm_or_persistence_params(tmp_path, monkeypatch) -> None:
    _control, manager = _guard_manager(tmp_path, monkeypatch)
    response = manager.start_role_run(
        {
            "baseline_id": 1,
            "role_lanes": ["guard_primary"],
            "params": {"persistence_min_sweep_loops": 3, "two_pass": True},
        }
    )
    job = manager.jobs[response["role_run"]["child_jobs"][0]["job_id"]]

    assert job.params["profile"] == "fm_broadcast"
    assert job.params["persistence_min_sweep_loops"] == 3
    assert job.params["two_pass"] is True
    assert "--persistence-min-sweep-loops" in job.cmd
    assert "--two-pass" in job.cmd
