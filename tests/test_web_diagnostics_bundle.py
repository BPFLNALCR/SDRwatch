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


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return None


def _append_span_violation(
    errors: list[str],
    *,
    label: str,
    low: Any,
    center: Any,
    high: Any,
) -> None:
    low_int = _coerce_int(low)
    center_int = _coerce_int(center)
    high_int = _coerce_int(high)
    if low_int is None or center_int is None or high_int is None:
        return
    if not (low_int <= center_int <= high_int):
        errors.append(f"{label}: expected {low_int} <= {center_int} <= {high_int}")


def _collect_span_violations(record: Any, label: str) -> list[str]:
    if isinstance(record, list):
        errors: list[str] = []
        for index, item in enumerate(record):
            errors.extend(_collect_span_violations(item, f"{label}[{index}]"))
        return errors
    if not isinstance(record, dict):
        return []

    errors: list[str] = []

    _append_span_violation(
        errors,
        label=f"{label}.persisted",
        low=record.get("f_low_hz"),
        center=record.get("f_center_hz"),
        high=record.get("f_high_hz"),
    )
    _append_span_violation(
        errors,
        label=f"{label}.raw",
        low=record.get("raw_low_hz"),
        center=record.get("raw_center_hz"),
        high=record.get("raw_high_hz"),
    )
    _append_span_violation(
        errors,
        label=f"{label}.measured",
        low=record.get("measured_low_hz"),
        center=record.get("measured_center_hz"),
        high=record.get("measured_high_hz"),
    )
    _append_span_violation(
        errors,
        label=f"{label}.match",
        low=record.get("match_low_hz"),
        center=record.get("match_center_hz"),
        high=record.get("match_high_hz"),
    )
    _append_span_violation(
        errors,
        label=f"{label}.display",
        low=record.get("display_low_hz"),
        center=record.get("display_center_hz"),
        high=record.get("display_high_hz"),
    )

    nested_specs = [
        ("raw_segment", "raw_low_hz", "raw_center_hz", "raw_high_hz"),
        ("match_span", "match_low_hz", "match_center_hz", "match_high_hz"),
        ("display_span", "display_low_hz", "display_center_hz", "display_high_hz"),
        ("measured_span", "measured_low_hz", "measured_center_hz", "measured_high_hz"),
    ]
    for key, top_level_low_key, top_level_center_key, top_level_high_key in nested_specs:
        nested = record.get(key)
        if not isinstance(nested, dict):
            continue
        low = nested.get("low_hz", nested.get("f_low_hz", record.get(top_level_low_key)))
        high = nested.get("high_hz", nested.get("f_high_hz", record.get(top_level_high_key)))
        center = record.get(top_level_center_key)
        if center in (None, ""):
            center = nested.get("center_hz", nested.get("f_center_hz", nested.get(top_level_center_key)))
        _append_span_violation(errors, label=f"{label}.{key}", low=low, center=center, high=high)

    for key, value in record.items():
        if isinstance(value, (dict, list)):
            errors.extend(_collect_span_violations(value, f"{label}.{key}"))
    return errors


def _bundle_span_violations(zf: zipfile.ZipFile) -> list[str]:
    errors: list[str] = []
    names = set(zf.namelist())

    if "database/baseline-detections.json" in names:
        baseline_rows = json.loads(zf.read("database/baseline-detections.json"))
        errors.extend(_collect_span_violations(baseline_rows, "database/baseline-detections.json"))

    if "diagnostics/diagnostic-jsonl-tail.jsonl" in names:
        text = zf.read("diagnostics/diagnostic-jsonl-tail.jsonl").decode("utf-8")
        for line_no, line in enumerate(text.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            errors.extend(_collect_span_violations(payload, f"diagnostics/diagnostic-jsonl-tail.jsonl:{line_no}"))

    for name in sorted(names):
        if not name.startswith("diagnostics/") or "characterization" not in name or name.endswith("diagnostic-jsonl-tail.jsonl"):
            continue
        if name.endswith(".json"):
            payload = json.loads(zf.read(name))
            errors.extend(_collect_span_violations(payload, name))
        elif name.endswith(".jsonl"):
            text = zf.read(name).decode("utf-8")
            for line_no, line in enumerate(text.splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                errors.extend(_collect_span_violations(payload, f"{name}:{line_no}"))

    return errors


def test_diagnostic_bundle_contains_available_evidence(tmp_path: Path) -> None:
    db_path = tmp_path / "sdrwatch.db"
    log_path = tmp_path / "scanner.log"
    diag_path = tmp_path / "diagnostic.jsonl"
    _create_temp_db(db_path)
    _write_text(log_path, ["log one", "log two"])
    _write_text(
        diag_path,
        [
            '{"event": "detection_window", "tuning_params": {"profile": "fm_broadcast", "two_pass": true}}',
            '{"event": "persistence_decision", "action": "insert"}',
            '{"event": "persistence_decision", "action": "update"}',
            '{"event": "width_decision", "stage": "shape_display", "was_floored": true}',
            '{"event": "revisit_queue", "action": "queued"}',
        ],
    )

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
        assert "diagnostics/decision-summary.json" in names
        summary = json.loads(zf.read("diagnostics/decision-summary.json"))
        assert summary["event_counts"]["persistence_decision"] == 2
        assert summary["persistence_actions"] == {"insert": 1, "update": 1}
        assert summary["width_stages"] == {"shape_display": 1}
        assert summary["revisit_events"] == {"revisit_queue": 1}
        assert summary["effective_settings"]["profile"] == "fm_broadcast"
        assert summary["effective_settings"]["two_pass"] is True
        manifest = json.loads(zf.read("manifest.json"))
        assert manifest["job_id"] == "abc123def456"
        assert manifest["missing"] == []
    finally:
        zf.close()
        buffer.close()


def test_diagnostic_bundle_decision_summary_includes_structured_aggregate_counts(tmp_path: Path) -> None:
    db_path = tmp_path / "sdrwatch.db"
    log_path = tmp_path / "scanner.log"
    diag_path = tmp_path / "diagnostic.jsonl"
    _create_temp_db(db_path)
    _write_text(log_path, ["log"])
    _write_text(
        diag_path,
        [
            '{"event": "segment_inventory", "num_segments": 2}',
            '{"event": "cluster_emit", "center_hz": 100100000}',
            '{"event": "cluster_reject", "center_hz": 100300000}',
            '{"event": "persistence_decision", "action": "match"}',
            '{"event": "persistence_decision", "action": "no_match"}',
            '{"event": "persistence_decision", "action": "cross_sweep_promote"}',
            '{"event": "width_decision", "stage": "shape_match"}',
            '{"event": "revisit_queue", "action": "queued"}',
            '{"event": "revisit_result", "matched": true}',
            '{"event": "characterization_record", "source_pass": "coarse"}',
        ],
    )

    bundle = build_diagnostic_bundle(
        job=_job(tmp_path, diag_path, log_path),
        db_path=str(db_path),
        bounds=DiagnosticBundleBounds(log_tail_lines=20, diagnostic_tail_lines=20, row_limit=20),
    )

    zf, buffer = _zip_entries(bundle.content)
    try:
        summary = json.loads(zf.read("diagnostics/decision-summary.json"))
        aggregates = summary["aggregate_counts"]
        assert aggregates["segment_inventory_count"] == 1
        assert aggregates["cluster_emitted_count"] == 1
        assert aggregates["cluster_rejected_count"] == 1
        assert aggregates["persistence_match_count"] == 1
        assert aggregates["persistence_no_match_count"] == 1
        assert aggregates["persistence_cross_sweep_promote_count"] == 1
        assert aggregates["width_decision_count"] == 1
        assert aggregates["revisit_queued_count"] == 1
        assert aggregates["revisit_result_count"] == 1
        assert aggregates["characterization_record_count"] == 1
        assert summary["truncated"] is False
    finally:
        zf.close()
        buffer.close()


def test_diagnostic_bundle_exported_centers_stay_within_exported_spans(tmp_path: Path) -> None:
    db_path = tmp_path / "sdrwatch.db"
    log_path = tmp_path / "scanner.log"
    diag_path = tmp_path / "diagnostic.jsonl"
    _create_temp_db(db_path)
    _write_text(log_path, ["log"])
    _write_text(
        diag_path,
        [
            json.dumps(
                {
                    "event": "characterization_record",
                    "raw_low_hz": 100_099_000,
                    "raw_center_hz": 100_100_000,
                    "raw_high_hz": 100_101_000,
                    "measured_low_hz": 100_020_000,
                    "measured_center_hz": 100_100_000,
                    "measured_high_hz": 100_180_000,
                    "match_low_hz": 100_060_000,
                    "match_center_hz": 100_100_000,
                    "match_high_hz": 100_140_000,
                }
            )
        ],
    )

    bundle = build_diagnostic_bundle(
        job=_job(tmp_path, diag_path, log_path),
        db_path=str(db_path),
        bounds=DiagnosticBundleBounds(log_tail_lines=20, diagnostic_tail_lines=20, row_limit=20),
    )

    zf, buffer = _zip_entries(bundle.content)
    try:
        assert _bundle_span_violations(zf) == []
    finally:
        zf.close()
        buffer.close()


def test_diagnostic_bundle_exports_bounded_characterization_summary(tmp_path: Path) -> None:
    db_path = tmp_path / "sdrwatch.db"
    log_path = tmp_path / "scanner.log"
    diag_path = tmp_path / "diagnostic.jsonl"
    _create_temp_db(db_path)
    _write_text(log_path, ["log"])
    _write_text(
        diag_path,
        [
            json.dumps(
                {
                    "event": "characterization_record",
                    "source_pass": "coarse",
                    "raw_low_hz": 100_099_000,
                    "raw_center_hz": 100_100_000,
                    "raw_high_hz": 100_101_000,
                    "raw_bandwidth_hz": 2_000.0,
                    "measured_low_hz": 100_060_000,
                    "measured_center_hz": 100_100_000,
                    "measured_high_hz": 100_140_000,
                    "measured_bandwidth_hz": 80_000.0,
                    "measured_bandwidth_confidence": 0.7,
                    "match_low_hz": 100_060_000,
                    "match_center_hz": 100_100_000,
                    "match_high_hz": 100_140_000,
                    "match_bandwidth_hz": 80_000.0,
                    "display_low_hz": 100_000_000,
                    "stable_center_hz": 100_100_000,
                    "center_delta_hz": 0,
                    "display_center_hz": 100_100_000,
                    "display_high_hz": 100_200_000,
                    "display_bandwidth_hz": 200_000.0,
                    "characterization_confidence": 0.8,
                    "characterization_method": "coarse_cluster_span",
                    "classification_candidate": "unknown",
                    "classification_evidence": [],
                    "evidence_sources": ["coarse_cluster"],
                    "bandplan_service": "FM Broadcast",
                    "profile_context": "fm_broadcast",
                }
            ),
            json.dumps(
                {
                    "event": "characterization_record",
                    "source_pass": "revisit",
                    "raw_low_hz": 100_299_000,
                    "raw_center_hz": 100_300_000,
                    "raw_high_hz": 100_301_000,
                    "raw_bandwidth_hz": 2_000.0,
                    "measured_low_hz": 100_260_000,
                    "measured_center_hz": 100_300_000,
                    "measured_high_hz": 100_340_000,
                    "measured_bandwidth_hz": 80_000.0,
                    "measured_bandwidth_confidence": 0.9,
                    "match_low_hz": 100_260_000,
                    "match_center_hz": 100_300_000,
                    "match_high_hz": 100_340_000,
                    "match_bandwidth_hz": 80_000.0,
                    "display_low_hz": 100_200_000,
                    "stable_center_hz": 100_292_000,
                    "center_delta_hz": 8_000,
                    "display_center_hz": 100_300_000,
                    "display_high_hz": 100_400_000,
                    "display_bandwidth_hz": 200_000.0,
                    "characterization_confidence": 0.9,
                    "characterization_method": "revisit_refinement",
                    "classification_candidate": "unknown",
                    "classification_evidence": [],
                    "evidence_sources": ["revisit_confirmation"],
                    "bandplan_service": "FM Broadcast",
                    "profile_context": "fm_broadcast",
                }
            ),
        ],
    )

    bundle = build_diagnostic_bundle(
        job=_job(tmp_path, diag_path, log_path),
        db_path=str(db_path),
        bounds=DiagnosticBundleBounds(log_tail_lines=20, diagnostic_tail_lines=20, row_limit=1),
    )

    zf, buffer = _zip_entries(bundle.content)
    try:
        summary = json.loads(zf.read("diagnostics/characterization-summary.json"))
        assert summary["record_count"] == 2
        assert summary["source_pass_counts"] == {"coarse": 1, "revisit": 1}
        assert summary["truncated"] is True
        assert len(summary["records"]) == 1
        [sample] = summary["records"]
        assert sample["measured_characterization"]["occupied_bandwidth_hz"] == 80_000.0
        assert "stable_center_hz" in sample["measured_characterization"]
        assert "center_delta_hz" in sample["measured_characterization"]
        assert sample["display_span"]["bandwidth_hz"] == 200_000.0
        assert sample["context"]["profile_context"] == "fm_broadcast"
    finally:
        zf.close()
        buffer.close()


def test_diagnostic_bundle_span_scan_reports_invalid_exported_centers() -> None:
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "database/baseline-detections.json",
            json.dumps(
                [
                    {
                        "id": 1,
                        "baseline_id": 1,
                        "f_low_hz": 100_000_000,
                        "f_center_hz": 100_250_000,
                        "f_high_hz": 100_200_000,
                    }
                ]
            ),
        )
        zf.writestr(
            "diagnostics/characterization-records.json",
            json.dumps(
                [
                    {
                        "raw_low_hz": 100_099_000,
                        "raw_center_hz": 100_102_000,
                        "raw_high_hz": 100_101_000,
                        "measured_low_hz": 100_020_000,
                        "measured_center_hz": 100_200_000,
                        "measured_high_hz": 100_180_000,
                        "match_low_hz": 100_060_000,
                        "match_center_hz": 100_150_000,
                        "match_high_hz": 100_140_000,
                    }
                ]
            ),
        )

    zf, zip_buffer = _zip_entries(buffer.getvalue())
    try:
        errors = _bundle_span_violations(zf)
        assert any("database/baseline-detections.json[0].persisted" in error for error in errors)
        assert any("diagnostics/characterization-records.json[0].raw" in error for error in errors)
        assert any("diagnostics/characterization-records.json[0].measured" in error for error in errors)
        assert any("diagnostics/characterization-records.json[0].match" in error for error in errors)
    finally:
        zf.close()
        zip_buffer.close()
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


def test_diagnostic_bundle_decision_summary_marks_empty_tail(tmp_path: Path) -> None:
    db_path = tmp_path / "sdrwatch.db"
    log_path = tmp_path / "scanner.log"
    diag_path = tmp_path / "diagnostic.jsonl"
    _create_temp_db(db_path)
    _write_text(log_path, ["log"])
    _write_text(diag_path, ['{"event": "detection_window", "tuning_params": {"two_pass": false}}'])

    bundle = build_diagnostic_bundle(
        job=_job(tmp_path, diag_path, log_path),
        db_path=str(db_path),
        bounds=DiagnosticBundleBounds(log_tail_lines=20, diagnostic_tail_lines=20, row_limit=20),
    )

    zf, buffer = _zip_entries(bundle.content)
    try:
        summary = json.loads(zf.read("diagnostics/decision-summary.json"))
        assert summary["event_counts"] == {"detection_window": 1}
        assert summary["persistence_actions"] == {}
        assert summary["effective_settings"]["two_pass"] is False
        manifest = json.loads(zf.read("manifest.json"))
        assert {"category": "decision_evidence", "reason": "no decision events in diagnostic JSONL tail"} in manifest[
            "missing"
        ]
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


def test_bundle_summarizes_role_aware_telemetry_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "sdrwatch.db"
    log_path = tmp_path / "scanner.log"
    diag_path = tmp_path / "role.diagnostic.jsonl"
    _create_temp_db(db_path)
    _write_text(log_path, ["log"])
    _write_text(
        diag_path,
        [
            json.dumps(
                {
                    "event": "detection_window",
                    "job_id": "job-1",
                    "role_run_id": "rr-1",
                    "receiver_role": "GUARD",
                    "role_lane": "guard_primary",
                    "device_identity": "rtl:serial:S1",
                    "device_key": "rtl:0",
                    "device_serial": "S1",
                    "timing": {"tune_ms": 1.0, "read_ms": 2.0, "total_window_ms": 3.0},
                    "unavailable_fields": ["dropped_reads"],
                }
            ),
            json.dumps(
                {
                    "event": "resource_telemetry",
                    "job_id": "job-1",
                    "role_run_id": "rr-1",
                    "receiver_role": "GUARD",
                    "pid": 1234,
                    "rss_memory_bytes": None,
                    "unavailable_fields": ["rss_memory_bytes"],
                }
            ),
        ],
    )

    bundle = build_diagnostic_bundle(job=_job(tmp_path, diag_path, log_path), db_path=str(db_path))

    role_summary = bundle.manifest["role_telemetry_summary"]
    assert role_summary["roles"] == ["GUARD"]
    assert role_summary["devices"] == ["rtl:serial:S1"]
    assert role_summary["jobs"] == ["job-1"]
    assert role_summary["role_run_ids"] == ["rr-1"]
    assert role_summary["timing_fields"]["tune_ms"] == 1
    assert role_summary["resource_telemetry_count"] == 1
    assert "dropped_reads" in role_summary["unavailable_fields"]


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
