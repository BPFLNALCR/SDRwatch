"""No-hardware lock and lifecycle tests for multi-RTL controller safety."""

from __future__ import annotations

import json
import threading
import time

from tests.helpers_control import load_control_module
from tests.helpers_multi_rtl import FakeProcess, fake_lock_owner


def test_atomic_lock_acquisition_allows_only_one_same_device_owner(tmp_path) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    results: list[str] = []

    def claim(owner: str) -> None:
        try:
            manager._acquire_device("rtl:0", owner=owner, metadata={"job_id": owner})
            results.append(owner)
        except RuntimeError:
            pass

    threads = [threading.Thread(target=claim, args=(f"job-{idx}",)) for idx in range(12)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(results) == 1
    payload = json.loads(manager._lock_path("rtl:0").read_text(encoding="utf-8"))
    assert payload["job_id"] == results[0]


def test_distinct_device_locks_can_be_claimed_concurrently(tmp_path) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()

    manager._acquire_device("rtl:0", owner="job-0", metadata={"job_id": "job-0"})
    manager._acquire_device("rtl:1", owner="job-1", metadata={"job_id": "job-1"})

    assert manager._lock_path("rtl:0").exists()
    assert manager._lock_path("rtl:1").exists()


def test_stale_lock_for_dead_known_owner_is_replaced(tmp_path) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    manager.jobs["dead-job"] = control.Job(
        id="dead-job",
        created_ts=1.0,
        label="dead",
        device_key="rtl:0",
        baseline_id=1,
        status="finished",
        pid=None,
        cmd=[],
        log_path="dead.log",
        params={},
    )
    fake_lock_owner(manager._lock_path("rtl:0"), "dead-job")

    manager._acquire_device("rtl:0", owner="new-job", metadata={"job_id": "new-job"})

    payload = json.loads(manager._lock_path("rtl:0").read_text(encoding="utf-8"))
    assert payload["job_id"] == "new-job"


def test_stale_lock_for_unknown_old_owner_is_replaced(tmp_path) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    fake_lock_owner(manager._lock_path("rtl:0"), "unknown-job", age_seconds=7200)

    manager._acquire_device("rtl:0", owner="new-job", metadata={"job_id": "new-job"})

    payload = json.loads(manager._lock_path("rtl:0").read_text(encoding="utf-8"))
    assert payload["job_id"] == "new-job"


def test_startup_reconciliation_releases_dead_persisted_job_lock(tmp_path, monkeypatch) -> None:
    base = tmp_path / "control"
    control = load_control_module(base, module_name="control_lifecycle_first")
    control.ensure_dirs()
    state = {
        "jobs": {
            "dead-job": {
                "id": "dead-job",
                "created_ts": 1.0,
                "label": "dead",
                "device_key": "rtl:0",
                "baseline_id": 1,
                "status": "running",
                "pid": 999999,
                "cmd": [],
                "log_path": "dead.log",
                "params": {},
            }
        }
    }
    control.write_state(state)
    fake_lock_owner(control.LOCKS_DIR / "rtl_0.lock", "dead-job")
    monkeypatch.setattr(control, "pid_alive", lambda _pid: False)

    manager = control.JobManager()

    assert manager.jobs["dead-job"].status == "finished"
    assert not manager._lock_path("rtl:0").exists()


def test_process_reaper_releases_lock_and_sets_terminal_status(tmp_path) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    job = control.Job(
        id="job-1",
        created_ts=1.0,
        label="scan",
        device_key="rtl:0",
        baseline_id=1,
        status="running",
        pid=123,
        cmd=[],
        log_path="job.log",
        params={},
    )
    manager.jobs[job.id] = job
    manager._acquire_device("rtl:0", owner=job.id, metadata={"job_id": job.id})

    manager._spawn_reaper(job.id, FakeProcess(returncode=0), "rtl:0")
    deadline = time.time() + 2.0
    while time.time() < deadline and manager.jobs[job.id].status == "running":
        time.sleep(0.01)

    assert manager.jobs[job.id].status == "finished"
    assert not manager._lock_path("rtl:0").exists()


def test_stop_job_finishes_without_status_drift(tmp_path, monkeypatch) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    job = control.Job(
        id="job-1",
        created_ts=1.0,
        label="scan",
        device_key="rtl:0",
        baseline_id=1,
        status="running",
        pid=123,
        cmd=[],
        log_path="job.log",
        params={},
    )
    manager.jobs[job.id] = job
    manager._acquire_device("rtl:0", owner=job.id, metadata={"job_id": job.id})
    monkeypatch.setattr(control, "pid_alive", lambda _pid: False)
    monkeypatch.setattr(control.os, "kill", lambda _pid, _sig: None)

    stopped = manager.stop_job(job.id, wait=0.01)

    assert stopped.status == "finished"
    assert not manager._lock_path("rtl:0").exists()


def test_failed_pre_spawn_start_never_acquires_unsupported_lock(tmp_path) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()

    try:
        manager.start_job(device_key="soapy:0", label="bad", baseline_id=1, sdrwatch_args={})
    except control.UnsupportedBackendError:
        pass
    else:
        raise AssertionError("unsupported backend should be rejected")

    assert not manager._lock_path("soapy:0").exists()
