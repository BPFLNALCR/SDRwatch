"""No-hardware diagnostics bundle and web API tests."""

from __future__ import annotations

import json
import sqlite3
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

from sdrwatch_web import create_app
from sdrwatch_web.diagnostics import DiagnosticBundleBounds, build_diagnostic_bundle


def _write_text(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


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
                bin_hz REAL NOT NULL
            );
            CREATE TABLE baseline_detections (
                id INTEGER PRIMARY KEY,
                baseline_id INTEGER NOT NULL,
                f_low_hz INTEGER NOT NULL,
                f_high_hz INTEGER NOT NULL,
                f_center_hz INTEGER NOT NULL,
                first_seen_utc TEXT NOT NULL,
                last_seen_utc TEXT NOT NULL,
                total_hits INTEGER NOT NULL,
                total_windows INTEGER NOT NULL,
                confidence REAL NOT NULL,
                label TEXT,
                classification TEXT DEFAULT 'unknown',
                selected INTEGER DEFAULT 0
            );
            CREATE TABLE scan_updates (
                id INTEGER PRIMARY KEY,
                baseline_id INTEGER NOT NULL,
                timestamp_utc TEXT NOT NULL,
                num_hits INTEGER NOT NULL,
                num_segments INTEGER NOT NULL,
                num_new_signals INTEGER NOT NULL,
                num_revisits INTEGER NOT NULL DEFAULT 0,
                num_confirmed INTEGER NOT NULL DEFAULT 0,
                num_false_positive INTEGER NOT NULL DEFAULT 0,
                duration_ms INTEGER
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
            "INSERT INTO baselines VALUES (1, 'Test Site', '2026-06-09T00:00:00Z', 88000000, 108000000, 1000.0)"
        )
        conn.execute(
            """
            INSERT INTO baseline_detections (
                id, baseline_id, f_low_hz, f_high_hz, f_center_hz,
                first_seen_utc, last_seen_utc, total_hits, total_windows,
                confidence, label, classification, selected
            ) VALUES (1, 1, 100000000, 100200000, 100100000,
                '2026-06-09T00:00:00Z', '2026-06-09T00:01:00Z',
                3, 5, 0.9, 'Test signal', 'unknown', 1)
            """
        )
        conn.execute(
            """
            INSERT INTO baseline_detections (
                id, baseline_id, f_low_hz, f_high_hz, f_center_hz,
                first_seen_utc, last_seen_utc, total_hits, total_windows,
                confidence, label, classification, selected
            ) VALUES (2, 1, 101000000, 101200000, 101100000,
                '2026-06-09T00:00:00Z', '2026-06-09T00:02:00Z',
                2, 5, 0.8, 'Second signal', 'unknown', 0)
            """
        )
        conn.execute(
            """
            INSERT INTO scan_updates (
                id, baseline_id, timestamp_utc, num_hits, num_segments,
                num_new_signals, duration_ms
            ) VALUES (1, 1, '2026-06-09T00:01:00Z', 3, 1, 1, 120)
            """
        )
        conn.execute(
            """
            INSERT INTO monitoring_zones (
                id, baseline_id, name, description, f_start_hz, f_stop_hz,
                category, priority, enabled, is_preset, created_at
            ) VALUES (1, 1, 'FM', 'FM broadcast', 88000000, 108000000,
                'broadcast', 10, 1, 1, '2026-06-09T00:00:00Z')
            """
        )
        conn.execute(
            """
            INSERT INTO friendly_signals (
                id, baseline_id, f_center_hz, f_tolerance_hz, label,
                notes, source, created_at
            ) VALUES (1, 1, 100100000, 10000, 'Known carrier',
                'Expected local signal', 'user', '2026-06-09T00:00:00Z')
            """
        )
        conn.commit()
    finally:
        conn.close()


def _job(tmp_path: Path, diagnostic_path: Path, log_path: Path, status: str = "finished") -> Dict[str, Any]:
    return {
        "id": "abc123def456",
        "created_ts": 1781020800.0,
        "label": "web",
        "device_key": "rtl:0",
        "baseline_id": 1,
        "status": status,
        "pid": None,
        "cmd": ["python", "-m", "sdrwatch.cli", "--diagnostic-jsonl", str(diagnostic_path)],
        "log_path": str(log_path),
        "params": {
            "start": 88000000,
            "stop": 108000000,
            "diagnostics_mode": True,
            "diagnostic_jsonl": str(diagnostic_path),
            "db": str(tmp_path / "sdrwatch.db"),
        },
    }


def _zip_entries(content: bytes) -> tuple[zipfile.ZipFile, BytesIO]:
    buffer = BytesIO(content)
    return zipfile.ZipFile(buffer), buffer


def test_diagnostic_bundle_contains_available_evidence(tmp_path: Path) -> None:
    db_path = tmp_path / "sdrwatch.db"
    log_path = tmp_path / "scanner.log"
    diag_path = tmp_path / "diagnostic.jsonl"
    _create_temp_db(db_path)
    _write_text(log_path, ["log one", "log two"])
    _write_text(diag_path, ['{"event": 1}', '{"event": 2}'])

    bundle = build_diagnostic_bundle(
        job=_job(tmp_path, diag_path, log_path),
        db_path=str(db_path),
        bounds=DiagnosticBundleBounds(log_tail_lines=20, diagnostic_tail_lines=20, row_limit=20),
    )

    zf, buffer = _zip_entries(bundle.content)
    try:
        names = set(zf.namelist())
        assert "manifest.json" in names
        assert "NOTES.md" in names
        assert "job/job.json" in names
        assert "job/params.json" in names
        assert "job/scanner-command.txt" in names
        assert "logs/scanner-log-tail.txt" in names
        assert "diagnostics/diagnostic-jsonl-tail.jsonl" in names
        assert "database/baseline.json" in names
        assert "database/baseline-detections.json" in names
        assert "database/scan-updates.json" in names
        assert "database/monitoring-zones.json" in names
        assert "database/friendly-signals.json" in names
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["job_id"] == "abc123def456"
        assert manifest["missing"] == []
    finally:
        zf.close()
        buffer.close()


def test_diagnostic_bundle_records_truncation_and_missing_evidence(tmp_path: Path) -> None:
    db_path = tmp_path / "sdrwatch.db"
    log_path = tmp_path / "missing.log"
    diag_path = tmp_path / "diagnostic.jsonl"
    _create_temp_db(db_path)
    _write_text(diag_path, ['{"event": 1}', '{"event": 2}', '{"event": 3}'])

    bundle = build_diagnostic_bundle(
        job=_job(tmp_path, diag_path, log_path),
        db_path=str(db_path),
        scanner_log_text="line1\nline2\nline3\n",
        bounds=DiagnosticBundleBounds(log_tail_lines=2, diagnostic_tail_lines=2, row_limit=1),
    )

    zf, buffer = _zip_entries(bundle.content)
    try:
        manifest = json.loads(zf.read("manifest.json"))
        truncated = {item["category"] for item in manifest["truncated"]}
        assert "scanner_log" in truncated
        assert "diagnostic_jsonl" in truncated
        assert "baseline_detections" in truncated
        assert zf.read("logs/scanner-log-tail.txt").decode("utf-8") == "line2\nline3\n"
    finally:
        zf.close()
        buffer.close()


def test_notes_template_contains_required_prompts_and_problem_types(tmp_path: Path) -> None:
    db_path = tmp_path / "sdrwatch.db"
    log_path = tmp_path / "scanner.log"
    diag_path = tmp_path / "diagnostic.jsonl"
    _create_temp_db(db_path)
    _write_text(log_path, ["log"])
    _write_text(diag_path, ['{"event": 1}'])

    bundle = build_diagnostic_bundle(job=_job(tmp_path, diag_path, log_path), db_path=str(db_path))

    zf, buffer = _zip_entries(bundle.content)
    try:
        notes = zf.read("NOTES.md").decode("utf-8")
        for text in (
            "Expected behavior",
            "Actual behavior",
            "Frequency or band affected",
            "Problem type",
            "false positive",
            "false negative",
            "wrong bandwidth",
            "wrong center",
            "merged signals",
            "split signals",
            "unstable baseline",
            "other",
        ):
            assert text in notes
    finally:
        zf.close()
        buffer.close()


class FakeController:
    def __init__(self, job: Dict[str, Any], log_text: str = "controller log\n") -> None:
        self.job = job
        self.log_text = log_text
        self.started_payload: Dict[str, Any] | None = None

    def start_job(self, device_key: str, label: str, baseline_id: int, params: Dict[str, Any]) -> Dict[str, Any]:
        self.started_payload = {
            "device_key": device_key,
            "label": label,
            "baseline_id": baseline_id,
            "params": dict(params),
        }
        response = dict(self.job)
        response["params"] = dict(params)
        if params.get("diagnostics_mode") and not response["params"].get("diagnostic_jsonl"):
            response["params"]["diagnostic_jsonl"] = "/tmp/sdrwatch-control/diagnostics/abc123def456.diagnostic.jsonl"
        return response

    def list_jobs(self) -> list[Dict[str, Any]]:
        return [self.job]

    def job_detail(self, job_id: str) -> Dict[str, Any]:
        if job_id != self.job["id"]:
            raise RuntimeError("not found")
        return self.job

    def job_logs(self, job_id: str, tail: int | None = None) -> str:
        if job_id != self.job["id"]:
            raise RuntimeError("not found")
        if tail:
            return "\n".join(self.log_text.splitlines()[-tail:]) + "\n"
        return self.log_text


def test_api_jobs_accepts_diagnostics_mode_without_manual_path(tmp_path: Path) -> None:
    db_path = tmp_path / "sdrwatch.db"
    log_path = tmp_path / "scanner.log"
    diag_path = tmp_path / "diagnostic.jsonl"
    _create_temp_db(db_path)
    _write_text(log_path, ["log"])
    _write_text(diag_path, ['{"event": 1}'])
    fake = FakeController(_job(tmp_path, diag_path, log_path, status="running"))
    app = create_app(str(db_path))
    app.extensions["sdrwatch_controller"] = fake

    client = app.test_client()
    response = client.post(
        "/api/jobs",
        json={
            "device_key": "rtl:0",
            "label": "web",
            "baseline_id": 1,
            "params": {"start": 88000000, "stop": 108000000, "diagnostics_mode": True},
        },
    )

    assert response.status_code == 200
    data = response.get_json()
    assert fake.started_payload is not None
    assert fake.started_payload["params"]["diagnostics_mode"] is True
    assert "diagnostic_jsonl" not in fake.started_payload["params"]
    assert data["job"]["params"]["diagnostic_jsonl"].endswith(".diagnostic.jsonl")


def test_api_jobs_preserves_non_diagnostic_start_payload(tmp_path: Path) -> None:
    db_path = tmp_path / "sdrwatch.db"
    log_path = tmp_path / "scanner.log"
    diag_path = tmp_path / "diagnostic.jsonl"
    _create_temp_db(db_path)
    _write_text(log_path, ["log"])
    _write_text(diag_path, ['{"event": 1}'])
    fake = FakeController(_job(tmp_path, diag_path, log_path, status="running"))
    app = create_app(str(db_path))
    app.extensions["sdrwatch_controller"] = fake

    client = app.test_client()
    response = client.post(
        "/api/jobs",
        json={
            "device_key": "rtl:0",
            "label": "web",
            "baseline_id": 1,
            "params": {"start": 88000000, "stop": 108000000},
        },
    )

    assert response.status_code == 200
    assert fake.started_payload is not None
    assert "diagnostics_mode" not in fake.started_payload["params"]
    assert "diagnostic_jsonl" not in response.get_json()["job"]["params"]


def test_api_diagnostic_bundle_download_and_auth(tmp_path: Path, monkeypatch: Any) -> None:
    db_path = tmp_path / "sdrwatch.db"
    log_path = tmp_path / "scanner.log"
    diag_path = tmp_path / "diagnostic.jsonl"
    _create_temp_db(db_path)
    _write_text(log_path, ["log"])
    _write_text(diag_path, ['{"event": 1}'])
    fake = FakeController(_job(tmp_path, diag_path, log_path))
    app = create_app(str(db_path))
    app.extensions["sdrwatch_controller"] = fake
    monkeypatch.setattr("sdrwatch_web.auth.API_TOKEN", "secret")

    client = app.test_client()
    assert client.get("/api/jobs/abc123def456/diagnostic-bundle").status_code == 401

    response = client.get(
        "/api/jobs/abc123def456/diagnostic-bundle",
        headers={"Authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert response.mimetype == "application/zip"
    assert "sdrwatch-diagnostics-abc123def456.zip" in response.headers["Content-Disposition"]
    zf, buffer = _zip_entries(response.data)
    try:
        assert "manifest.json" in zf.namelist()
    finally:
        zf.close()
        buffer.close()


def test_bundle_export_works_for_active_recent_and_finished_jobs_without_hardware(tmp_path: Path) -> None:
    db_path = tmp_path / "sdrwatch.db"
    _create_temp_db(db_path)
    for status in ("running", "stopped", "finished"):
        log_path = tmp_path / f"{status}.log"
        diag_path = tmp_path / f"{status}.jsonl"
        _write_text(log_path, [f"{status} log"])
        _write_text(diag_path, [f'{{"status": "{status}"}}'])
        bundle = build_diagnostic_bundle(
            job=_job(tmp_path, diag_path, log_path, status=status),
            db_path=str(db_path),
        )
        assert bundle.manifest["job_status_at_export"] == status
        assert bundle.content.startswith(b"PK")
