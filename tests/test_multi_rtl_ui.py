"""Control-page coverage for hardware-aware multi-RTL operator UI."""

from __future__ import annotations

from pathlib import Path

from tests.test_control_page_scan_settings import _control_html, _function_block


def test_control_page_renders_hardware_inventory_and_tier_panel(tmp_path: Path) -> None:
    html = _control_html(tmp_path)

    for text in ("Receiver Hardware", "Capability tier", "Detected receivers"):
        assert text in html
    for element_id in (
        "hardwareTierPill",
        "hardwareInventoryBody",
        "hardwareRefreshBtn",
        "unsupportedHardwareBody",
    ):
        assert f'id="{element_id}"' in html

    load_function = _function_block(html, "loadHardwareInventory")
    render_function = _function_block(html, "renderHardwareInventory")
    assert "fetch('/api/hardware/inventory'" in load_function
    assert "capability.tier" in render_function
    assert "identity_confidence" in render_function


def test_control_page_renders_identity_warnings_and_role_assignment_controls(tmp_path: Path) -> None:
    html = _control_html(tmp_path)

    for text in ("Manual roles", "Friendly Guard", "Watchlist Guard", "Rover", "Reference"):
        assert text in html
    for element_id in (
        "roleAssignmentContainer",
        "roleAssignmentStatus",
    ):
        assert f'id="{element_id}"' in html

    render_function = _function_block(html, "renderRoleAssignments")
    assign_function = _function_block(html, "assignRoleLane")
    clear_function = _function_block(html, "clearRoleLane")
    assert "roleLane_${escapeHtml(lane.id)}" in render_function
    for lane in ("guard_primary", "guard_secondary", "rover", "reference"):
        assert f"id: '{lane}'" in html
    assert "warningText" in render_function
    assert "acknowledge_identity_warning: true" in assign_function
    assert "fetch(`/api/role-assignments/${encodeURIComponent(lane)}`" in assign_function
    assert "method: 'PUT'" in assign_function
    assert "method: 'DELETE'" in clear_function


def test_control_page_renders_role_run_status_and_grouped_stop_controls(tmp_path: Path) -> None:
    html = _control_html(tmp_path)

    for text in ("Role-aware runs", "Start assigned roles"):
        assert text in html
    for element_id in ("startRoleRunBtn", "roleRunStatus", "roleRunsContainer"):
        assert f'id="{element_id}"' in html

    start_function = _function_block(html, "startAssignedRoleRun")
    render_function = _function_block(html, "renderRoleRuns")
    stop_function = _function_block(html, "stopRoleRun")
    assert "fetch('/api/role-runs'" in start_function
    assert "method: 'POST'" in start_function
    assert "child_jobs" in render_function
    assert "stopRoleRun" in render_function
    assert "method: 'DELETE'" in stop_function


def test_control_page_keeps_existing_scan_controls_with_multi_rtl_panel(tmp_path: Path) -> None:
    html = _control_html(tmp_path)

    assert 'id="scan_preset"' in html
    assert '<option value="rtl_v4_discovery" selected>RTL-SDR v4 Discovery</option>' in html
    assert '<option value="fm_validation">FM Validation</option>' in html
    assert 'id="diagnostics_mode"' in html
    assert 'id="device_key"' in html
    assert "function buildScanJobPayload()" in html
    assert "fetch('/api/jobs'" in html
