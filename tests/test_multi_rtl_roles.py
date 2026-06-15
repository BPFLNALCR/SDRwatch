"""Manual receiver role assignment tests."""

from __future__ import annotations

from tests.helpers_control import load_control_module
from tests.helpers_multi_rtl import rtl_devices


def test_role_assignment_set_list_and_clear_for_all_roles(tmp_path, monkeypatch) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    monkeypatch.setattr(control, "discover_devices", lambda: rtl_devices(control, ["S1", "S2", "S3"]))

    manager.set_role_assignment("guard_primary", {"device_identity": "rtl:serial:S1", "role": "GUARD"})
    manager.set_role_assignment("rover", {"device_identity": "rtl:serial:S2", "role": "ROVER"})
    manager.set_role_assignment("reference", {"device_identity": "rtl:serial:S3", "role": "REFERENCE"})

    assignments = {row["role_lane"]: row for row in manager.list_role_assignments()["assignments"]}
    assert assignments["guard_primary"]["role"] == "GUARD"
    assert assignments["rover"]["role"] == "ROVER"
    assert assignments["reference"]["role"] == "REFERENCE"

    cleared = manager.clear_role_assignment("rover")
    assert cleared == {"cleared": True, "role_lane": "rover"}
    remaining = {row["role_lane"] for row in manager.list_role_assignments()["assignments"]}
    assert remaining == {"guard_primary", "reference"}


def test_stable_serial_assignment_survives_restart(tmp_path, monkeypatch) -> None:
    base = tmp_path / "control"
    control = load_control_module(base, module_name="control_roles_first")
    manager = control.JobManager()
    monkeypatch.setattr(control, "discover_devices", lambda: rtl_devices(control, ["S1"]))

    manager.set_role_assignment("guard_primary", {"device_identity": "rtl:serial:S1", "role": "GUARD"})

    restarted = load_control_module(base, module_name="control_roles_restarted")
    restarted_manager = restarted.JobManager()
    assignments = restarted_manager.list_role_assignments()["assignments"]
    assert assignments[0]["device_identity"] == "rtl:serial:S1"
    assert assignments[0]["assignment_scope"] == "persistent"


def test_index_only_assignment_requires_ack_and_is_session_scoped(tmp_path, monkeypatch) -> None:
    base = tmp_path / "control"
    control = load_control_module(base, module_name="control_roles_index_first")
    manager = control.JobManager()
    monkeypatch.setattr(control, "discover_devices", lambda: rtl_devices(control, [None]))

    try:
        manager.set_role_assignment("guard_primary", {"device_identity": "rtl:index:0", "role": "GUARD"})
    except control.RoleAssignmentError as exc:
        assert exc.status_code == 409
        assert exc.payload["error"] == "identity_warning_requires_acknowledgement"
    else:
        raise AssertionError("index-only assignment should require acknowledgement")

    manager.set_role_assignment(
        "guard_primary",
        {
            "device_identity": "rtl:index:0",
            "role": "GUARD",
            "acknowledge_identity_warning": True,
        },
    )
    assignment = manager.list_role_assignments()["assignments"][0]
    assert assignment["assignment_scope"] == "session"

    restarted = load_control_module(base, module_name="control_roles_index_restarted")
    restarted_manager = restarted.JobManager()
    assert restarted_manager.list_role_assignments()["assignments"] == []


def test_duplicate_device_assignment_is_rejected(tmp_path, monkeypatch) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    monkeypatch.setattr(control, "discover_devices", lambda: rtl_devices(control, ["S1"]))
    manager.set_role_assignment("guard_primary", {"device_identity": "rtl:serial:S1", "role": "GUARD"})

    try:
        manager.set_role_assignment("rover", {"device_identity": "rtl:serial:S1", "role": "ROVER"})
    except control.RoleAssignmentError as exc:
        assert exc.status_code == 409
        assert exc.payload["error"] == "duplicate_device_assignment"
    else:
        raise AssertionError("duplicate assignment should be rejected")


def test_role_assignment_controller_routes(tmp_path, monkeypatch) -> None:
    control = load_control_module(tmp_path / "control")
    manager = control.JobManager()
    monkeypatch.setattr(control, "discover_devices", lambda: rtl_devices(control, ["S1"]))
    app = control.make_app(manager)
    client = app.test_client()

    set_response = client.put(
        "/role-assignments/guard_primary",
        json={"device_identity": "rtl:serial:S1", "role": "GUARD"},
    )
    assert set_response.status_code == 200
    assert set_response.get_json()["assignment"]["role_lane"] == "guard_primary"

    list_response = client.get("/role-assignments")
    assert list_response.status_code == 200
    assert list_response.get_json()["assignments"][0]["device_identity"] == "rtl:serial:S1"

    clear_response = client.delete("/role-assignments/guard_primary")
    assert clear_response.status_code == 200
    assert clear_response.get_json() == {"cleared": True, "role_lane": "guard_primary"}
