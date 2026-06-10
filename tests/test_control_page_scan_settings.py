"""No-hardware control page scan settings tests."""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path
from typing import Any, Dict

from sdrwatch_web import create_app


def _create_temp_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE baselines (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL,
                freq_start_hz INTEGER NOT NULL,
                freq_stop_hz INTEGER NOT NULL,
                bin_hz REAL NOT NULL,
                total_windows INTEGER NOT NULL DEFAULT 0,
                location_lat REAL,
                location_lon REAL,
                sdr_serial TEXT,
                antenna TEXT,
                notes TEXT
            );
            CREATE TABLE monitoring_zones (
                id INTEGER PRIMARY KEY,
                baseline_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                f_start_hz INTEGER NOT NULL,
                f_stop_hz INTEGER NOT NULL,
                category TEXT,
                priority INTEGER DEFAULT 100,
                enabled INTEGER DEFAULT 1,
                is_preset INTEGER DEFAULT 0,
                created_at TEXT NOT NULL
            );
            CREATE TABLE friendly_signals (
                id INTEGER PRIMARY KEY,
                baseline_id INTEGER NOT NULL,
                f_center_hz INTEGER NOT NULL,
                f_tolerance_hz INTEGER DEFAULT 5000,
                label TEXT NOT NULL,
                notes TEXT,
                source TEXT,
                created_at TEXT NOT NULL
            );
            """
        )
        conn.execute(
            """
            INSERT INTO baselines (
                id, name, created_at, freq_start_hz, freq_stop_hz,
                bin_hz, total_windows, antenna
            ) VALUES (1, 'Test Site', '2026-06-10T00:00:00Z',
                88000000, 108000000, 1000.0, 4, 'Discone')
            """
        )
        conn.execute(
            """
            INSERT INTO monitoring_zones (
                id, baseline_id, name, description, f_start_hz, f_stop_hz,
                category, priority, enabled, is_preset, created_at
            ) VALUES (1, 1, 'FM Broadcast', 'FM broadcast band',
                88000000, 108000000, 'broadcast', 10, 1, 1,
                '2026-06-10T00:00:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO monitoring_zones (
                id, baseline_id, name, description, f_start_hz, f_stop_hz,
                category, priority, enabled, is_preset, created_at
            ) VALUES (2, 1, 'Airband', 'Aviation voice',
                118000000, 137000000, 'aviation', 20, 0, 1,
                '2026-06-10T00:00:00Z')
            """
        )
        conn.commit()
    finally:
        conn.close()


class FakeController:
    def __init__(self, include_cmd: bool = True) -> None:
        self.started_payload: Dict[str, Any] | None = None
        self.include_cmd = include_cmd
        self.job = {
            "id": "abc123def456",
            "created_ts": 1781020800.0,
            "label": "web",
            "device_key": "rtl:0",
            "baseline_id": 1,
            "status": "running",
            "pid": 123,
            "params": {"start": 88000000, "stop": 108000000},
            "log_path": "scanner.log",
        }
        if include_cmd:
            self.job["cmd"] = ["python", "-m", "sdrwatch.cli", "--start", "88000000"]

    def baselines(self) -> list[Dict[str, Any]]:
        return [
            {
                "id": 1,
                "name": "Test Site",
                "antenna": "Discone",
                "freq_start_hz": 88000000,
                "freq_stop_hz": 108000000,
                "bin_hz": 1000.0,
                "total_windows": 4,
            }
        ]

    def profiles(self) -> Dict[str, Any]:
        return {"profiles": [{"name": "fm_broadcast"}]}

    def devices(self) -> list[Dict[str, Any]]:
        return [{"key": "rtl:0", "label": "RTL-SDR #0", "kind": "rtlsdr"}]

    def list_jobs(self) -> list[Dict[str, Any]]:
        return [self.job]

    def job_detail(self, job_id: str) -> Dict[str, Any]:
        if job_id != self.job["id"]:
            raise RuntimeError("not found")
        return self.job

    def job_logs(self, job_id: str, tail: int | None = None) -> str:
        return "controller log\n"

    def start_job(self, device_key: str, label: str, baseline_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        self.started_payload = {
            "device_key": device_key,
            "label": label,
            "baseline_id": baseline_id,
            "params": dict(params),
        }
        response = dict(self.job)
        response["device_key"] = device_key
        response["label"] = label
        response["baseline_id"] = baseline_id
        response["params"] = dict(params)
        return response

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        response = dict(self.job)
        response["status"] = "stopped"
        return response


def _app(tmp_path: Path, include_cmd: bool = True):
    db_path = tmp_path / "sdrwatch.db"
    _create_temp_db(db_path)
    app = create_app(str(db_path))
    fake = FakeController(include_cmd=include_cmd)
    app.extensions["sdrwatch_controller"] = fake
    return app, fake


def _control_html(tmp_path: Path, include_cmd: bool = True) -> str:
    app, _fake = _app(tmp_path, include_cmd=include_cmd)
    response = app.test_client().get("/control")
    assert response.status_code == 200
    return response.get_data(as_text=True)


def _tag(html: str, element_id: str) -> str:
    match = re.search(rf"<[^>]+id=\"{re.escape(element_id)}\"[^>]*>", html)
    assert match, f"Missing element #{element_id}"
    return match.group(0)


def test_basic_controls_render_as_primary_scan_workflow(tmp_path: Path) -> None:
    html = _control_html(tmp_path)

    for text in (
        "Basic controls",
        "Monitoring Location",
        "SDR Device",
        "Monitoring Zones",
        "Diagnostics mode",
        "Start Monitoring",
        "Stop",
        "Live logs",
    ):
        assert text in html

    for element_id in (
        "basicControls",
        "baseline_select",
        "device_key",
        "run_mode",
        "duration_value",
        "diagnostics_mode",
        "startBtn",
        "stopBtn",
        "log",
    ):
        assert f'id="{element_id}"' in html


def test_api_jobs_preserves_basic_payload_and_diagnostics_mode(tmp_path: Path) -> None:
    app, fake = _app(tmp_path)
    response = app.test_client().post(
        "/api/jobs",
        json={
            "device_key": "rtl:0",
            "label": "web",
            "baseline_id": 1,
            "params": {
                "start": 88000000,
                "stop": 108000000,
                "diagnostics_mode": True,
            },
        },
    )

    assert response.status_code == 200
    assert fake.started_payload == {
        "device_key": "rtl:0",
        "label": "web",
        "baseline_id": 1,
        "params": {
            "start": 88000000,
            "stop": 108000000,
            "diagnostics_mode": True,
        },
    }


def test_tuning_controls_are_bounded_and_described(tmp_path: Path) -> None:
    html = _control_html(tmp_path)

    assert "Tuning controls" in html
    for element_id in ("threshold_db", "guard_bins", "min_width_bins", "cfar_alpha_db", "cfar_quantile"):
        tag = _tag(html, element_id)
        assert 'type="range"' in tag
        assert f'id="{element_id}_value"' in html

    for element_id in ("cfar", "fft", "avg", "samp_rate", "gain_mode", "gain"):
        assert f'id="{element_id}"' in html

    for help_text in (
        "Detection threshold above the learned noise floor.",
        "Below-threshold bins allowed inside one detection.",
        "Minimum contiguous bins before a detection is reported.",
        "Automatic gain is safest for normal scans.",
    ):
        assert help_text in html


def test_scan_form_and_numeric_controls_disable_autocomplete(tmp_path: Path) -> None:
    html = _control_html(tmp_path)
    assert 'id="scanForm"' in html
    assert 'autocomplete="off"' in _tag(html, "scanForm")

    for element_id in (
        "threshold_db",
        "guard_bins",
        "min_width_bins",
        "cfar_alpha_db",
        "cfar_quantile",
        "cluster_merge_hz",
        "max_detection_width_hz",
        "max_detection_width_ratio",
        "new_ema_occ",
        "persistence_hit_ratio",
        "persistence_min_seconds",
        "persistence_min_hits",
        "persistence_min_windows",
        "revisit_fft",
        "revisit_avg",
        "revisit_margin_hz",
        "revisit_span_limit_hz",
        "revisit_max_bands",
        "revisit_floor_threshold_db",
        "latitude",
        "longitude",
    ):
        assert 'autocomplete="off"' in _tag(html, element_id)


def test_generated_settings_builder_keeps_existing_param_names(tmp_path: Path) -> None:
    html = _control_html(tmp_path)

    assert "function buildScanJobPayload()" in html
    for param_name in (
        "threshold_db",
        "guard_bins",
        "min_width_bins",
        "cfar",
        "cfar_alpha_db",
        "cfar_quantile",
        "fft",
        "avg",
        "samp_rate",
        "gain",
    ):
        assert f"params.{param_name}" in html


def test_expert_controls_render_required_settings(tmp_path: Path) -> None:
    html = _control_html(tmp_path)

    assert "Expert controls" in html
    for element_id in (
        "cluster_merge_hz",
        "max_detection_width_hz",
        "max_detection_width_ratio",
        "new_ema_occ",
        "persistence_mode",
        "persistence_hit_ratio",
        "persistence_min_seconds",
        "persistence_min_hits",
        "persistence_min_windows",
        "two_pass",
        "revisit_fft",
        "revisit_avg",
        "revisit_margin_hz",
        "revisit_span_limit_hz",
        "revisit_max_bands",
        "revisit_floor_threshold_db",
        "db",
        "jsonl",
        "diagnostic_jsonl",
    ):
        assert f'id="{element_id}"' in html


def test_reset_and_copy_controls_render(tmp_path: Path) -> None:
    html = _control_html(tmp_path)

    assert "Reset to safe defaults" in html
    assert "Copy current scan settings" in html
    assert 'id="resetDefaultsBtn"' in html
    assert 'id="copySettingsBtn"' in html
    assert 'id="scanSettingsStatus"' in html
    assert "const SAFE_SCAN_DEFAULTS" in html
    assert "function resetScanDefaults()" in html
    assert "function copyCurrentScanSettings()" in html


def test_internal_debug_command_copy_is_labeled_and_uses_job_metadata(tmp_path: Path) -> None:
    html = _control_html(tmp_path)

    assert "Copy generated scanner command (internal/debug)" in html
    assert 'id="copyScannerCommandBtn"' in html
    assert 'id="scannerCommandStatus"' in html
    assert "function getGeneratedScannerCommandText()" in html
    assert "activeJob?.cmd" in html

    no_cmd_dir = tmp_path / "no_cmd"
    no_cmd_dir.mkdir()
    html_without_cmd = _control_html(no_cmd_dir, include_cmd=False)
    assert "disabled" in _tag(html_without_cmd, "copyScannerCommandBtn")
