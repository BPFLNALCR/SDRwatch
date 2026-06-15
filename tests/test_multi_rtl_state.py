"""No-hardware tests for role-aware controller state."""

from __future__ import annotations

import json

from tests.helpers_control import load_control_module


def test_controller_state_migrates_missing_role_keys(tmp_path) -> None:
    base = tmp_path / "control"
    base.mkdir()
    (base / "state.json").write_text(
        json.dumps({"jobs": {}}),
        encoding="utf-8",
    )
    control = load_control_module(base)

    manager = control.JobManager()

    assert manager.role_assignments == {}
    assert manager.role_runs == {}
    assert isinstance(manager.role_assignment_session_epoch, str)
    persisted = json.loads((base / "state.json").read_text(encoding="utf-8"))
    assert persisted["role_assignments"] == {}
    assert persisted["role_runs"] == {}
    assert persisted["role_assignment_session_epoch"] == manager.role_assignment_session_epoch


def test_role_aware_job_metadata_serializes(tmp_path) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    job = control.Job(
        id="job-guard",
        created_ts=1.0,
        label="guard",
        device_key="rtl:0",
        baseline_id=1,
        status="running",
        pid=123,
        cmd=["python", "-m", "sdrwatch.cli"],
        log_path="job.log",
        params={},
        role_run_id="rr-1",
        receiver_role="GUARD",
        role_lane="guard_primary",
        source_task="guard_window",
        device_identity="rtl:serial:00000001",
        device_serial="00000001",
        device_index=0,
        identity_confidence="stable",
        active_device_count=1,
        active_role_count=1,
    )

    manager.jobs[job.id] = job
    manager._persist()
    payload = json.loads(control.STATE_PATH.read_text(encoding="utf-8"))

    stored = payload["jobs"]["job-guard"]
    assert stored["receiver_role"] == "GUARD"
    assert stored["role_lane"] == "guard_primary"
    assert stored["role_run_id"] == "rr-1"
    assert stored["source_task"] == "guard_window"
    assert stored["device_identity"] == "rtl:serial:00000001"
    assert stored["device_serial"] == "00000001"
    assert stored["device_index"] == 0
    assert stored["identity_confidence"] == "stable"
    assert stored["active_device_count"] == 1
    assert stored["active_role_count"] == 1
