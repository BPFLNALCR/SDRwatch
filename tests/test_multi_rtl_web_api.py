"""Web API proxy tests for multi-RTL controller surfaces."""

from __future__ import annotations

from typing import Any

from sdrwatch_web import create_app


class FakeController:
    def __init__(self) -> None:
        self.inventory_payload = {
            "capability": {
                "tier": "1",
                "label": "Tier 1 - Basic RTL",
                "runnable_rtl_count": 1,
                "max_role_count": 1,
                "supported_roles": ["GUARD"],
                "warnings": [],
            },
            "devices": [
                {
                    "device_identity": "rtl:serial:00000001",
                    "legacy_device_key": "rtl:0",
                    "device_kind": "rtlsdr",
                    "runnable": True,
                    "runnable_backend": "rtlsdr_native",
                }
            ],
            "unsupported_hardware": [],
        }

    def hardware_inventory(self) -> dict[str, Any]:
        return self.inventory_payload

    def devices(self) -> list[dict[str, Any]]:
        return [{"key": "rtl:0", "label": "RTL-SDR #0", "kind": "rtlsdr"}]

    def list_role_assignments(self) -> dict[str, Any]:
        return {
            "assignments": [
                {
                    "assignment_id": "assign-guard-primary",
                    "role": "GUARD",
                    "role_lane": "guard_primary",
                    "device_identity": "rtl:serial:00000001",
                }
            ]
        }

    def set_role_assignment(self, role_lane: str, payload: dict[str, Any]) -> dict[str, Any]:
        return {"assignment": {"role_lane": role_lane, **payload}}

    def clear_role_assignment(self, role_lane: str) -> dict[str, Any]:
        return {"cleared": True, "role_lane": role_lane}

    def start_role_run(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"role_run": {"role_run_id": "rr-1", "status": "running", "requested_roles": payload.get("role_lanes")}}

    def list_role_runs(self) -> dict[str, Any]:
        return {"role_runs": [{"role_run_id": "rr-1", "status": "running"}]}

    def get_role_run(self, role_run_id: str) -> dict[str, Any]:
        return {"role_run": {"role_run_id": role_run_id, "status": "running"}}

    def stop_role_run(self, role_run_id: str) -> dict[str, Any]:
        return {"role_run": {"role_run_id": role_run_id, "status": "finished"}}


def _app(tmp_path):
    db_path = tmp_path / "sdrwatch.db"
    app = create_app(str(db_path))
    fake = FakeController()
    app.extensions["sdrwatch_controller"] = fake
    return app, fake


def test_api_hardware_inventory_proxy_and_ctl_devices_compatibility(tmp_path) -> None:
    app, fake = _app(tmp_path)
    client = app.test_client()

    inventory_response = client.get("/api/hardware/inventory")
    assert inventory_response.status_code == 200
    assert inventory_response.get_json() == fake.inventory_payload

    ctl_response = client.get("/ctl/devices")
    assert ctl_response.status_code == 200
    assert ctl_response.get_json() == fake.devices()


def test_api_hardware_inventory_honors_web_token_auth(tmp_path, monkeypatch) -> None:
    import sdrwatch_web.auth as auth

    app, _fake = _app(tmp_path)
    monkeypatch.setattr(auth, "API_TOKEN", "secret-token")

    assert app.test_client().get("/api/hardware/inventory").status_code == 401
    authed = app.test_client().get(
        "/api/hardware/inventory",
        headers={"Authorization": "Bearer secret-token"},
    )
    assert authed.status_code == 200


def test_api_role_assignment_proxy_set_list_and_clear(tmp_path) -> None:
    app, _fake = _app(tmp_path)
    client = app.test_client()

    list_response = client.get("/api/role-assignments")
    assert list_response.status_code == 200
    assert list_response.get_json()["assignments"][0]["role_lane"] == "guard_primary"

    set_response = client.put(
        "/api/role-assignments/rover",
        json={"device_identity": "rtl:serial:00000002", "role": "ROVER"},
    )
    assert set_response.status_code == 200
    assert set_response.get_json()["assignment"]["role_lane"] == "rover"

    clear_response = client.delete("/api/role-assignments/rover")
    assert clear_response.status_code == 200
    assert clear_response.get_json() == {"cleared": True, "role_lane": "rover"}


def test_api_role_run_proxy_start_list_detail_and_stop(tmp_path) -> None:
    app, _fake = _app(tmp_path)
    client = app.test_client()

    start = client.post("/api/role-runs", json={"baseline_id": 1, "role_lanes": ["guard_primary", "rover"]})
    assert start.status_code == 201
    assert start.get_json()["role_run"]["role_run_id"] == "rr-1"

    listed = client.get("/api/role-runs")
    assert listed.status_code == 200
    assert listed.get_json()["role_runs"][0]["role_run_id"] == "rr-1"

    detail = client.get("/api/role-runs/rr-1")
    assert detail.status_code == 200
    assert detail.get_json()["role_run"]["status"] == "running"

    stopped = client.delete("/api/role-runs/rr-1")
    assert stopped.status_code == 200
    assert stopped.get_json()["role_run"]["status"] == "finished"
