"""Diagnostic bundle creation for GUI-driven SDRwatch troubleshooting."""

from __future__ import annotations

import io
import json
import os
import shlex
import sqlite3
import zipfile
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from sdrwatch.util.detection_diagnostics import (
    build_device_telemetry_snapshot,
    build_effective_parameter_manifest,
    summarize_characterization_records,
)
from sdrwatch_web.config import (
    DIAGNOSTIC_BUNDLE_FILENAME_PREFIX,
    DIAGNOSTIC_BUNDLE_JSONL_TAIL_LINES,
    DIAGNOSTIC_BUNDLE_LOG_TAIL_LINES,
    DIAGNOSTIC_BUNDLE_MAX_JSONL_TAIL_LINES,
    DIAGNOSTIC_BUNDLE_MAX_LOG_TAIL_LINES,
    DIAGNOSTIC_BUNDLE_MAX_ROW_LIMIT,
    DIAGNOSTIC_BUNDLE_ROW_LIMIT,
)


@dataclass(frozen=True)
class DiagnosticBundleBounds:
    log_tail_lines: int = DIAGNOSTIC_BUNDLE_LOG_TAIL_LINES
    diagnostic_tail_lines: int = DIAGNOSTIC_BUNDLE_JSONL_TAIL_LINES
    row_limit: int = DIAGNOSTIC_BUNDLE_ROW_LIMIT

    @classmethod
    def from_values(
        cls,
        *,
        log_tail_lines: Any = None,
        diagnostic_tail_lines: Any = None,
        row_limit: Any = None,
    ) -> "DiagnosticBundleBounds":
        return cls(
            log_tail_lines=_bounded_int(
                log_tail_lines,
                DIAGNOSTIC_BUNDLE_LOG_TAIL_LINES,
                DIAGNOSTIC_BUNDLE_MAX_LOG_TAIL_LINES,
            ),
            diagnostic_tail_lines=_bounded_int(
                diagnostic_tail_lines,
                DIAGNOSTIC_BUNDLE_JSONL_TAIL_LINES,
                DIAGNOSTIC_BUNDLE_MAX_JSONL_TAIL_LINES,
            ),
            row_limit=_bounded_int(
                row_limit,
                DIAGNOSTIC_BUNDLE_ROW_LIMIT,
                DIAGNOSTIC_BUNDLE_MAX_ROW_LIMIT,
            ),
        )


@dataclass(frozen=True)
class DiagnosticBundle:
    filename: str
    content: bytes
    manifest: Dict[str, Any]


class BundleManifest:
    def __init__(self, *, job: Dict[str, Any], bounds: DiagnosticBundleBounds) -> None:
        self.data: Dict[str, Any] = {
            "bundle_version": 1,
            "created_at": _utc_now(),
            "job_id": str(job.get("id") or "unknown"),
            "job_status_at_export": job.get("status") or "unknown",
            "bounds": {
                "log_tail_lines": bounds.log_tail_lines,
                "diagnostic_tail_lines": bounds.diagnostic_tail_lines,
                "row_limit": bounds.row_limit,
            },
            "included": [],
            "missing": [],
            "truncated": [],
        }

    def included(self, path: str) -> None:
        self.data["included"].append(path)

    def missing(self, category: str, reason: str) -> None:
        self.data["missing"].append({"category": category, "reason": reason})

    def truncated(self, category: str, reason: str, **extra: Any) -> None:
        entry = {"category": category, "reason": reason}
        entry.update(extra)
        self.data["truncated"].append(entry)


def _bounded_int(value: Any, default: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(1, min(parsed, maximum))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, default=str) + "\n").encode("utf-8")


def _row_to_dict(row: Any) -> Dict[str, Any]:
    if isinstance(row, dict):
        return dict(row)
    if isinstance(row, sqlite3.Row):
        return {key: row[key] for key in row.keys()}
    return dict(row)


def tail_text(text: str, limit: int) -> Tuple[str, int, bool]:
    lines = text.splitlines()
    truncated = len(lines) > limit
    selected = lines[-limit:] if truncated else lines
    payload = "\n".join(selected)
    if payload:
        payload += "\n"
    return payload, len(selected), truncated


def tail_file(path: str, limit: int) -> Tuple[Optional[str], int, bool, Optional[str]]:
    if not path:
        return None, 0, False, "path not configured"
    try:
        count = 0
        lines: deque[str] = deque(maxlen=limit)
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                count += 1
                lines.append(line)
        return "".join(lines), min(count, limit), count > limit, None
    except FileNotFoundError:
        return None, 0, False, f"file not found: {path}"
    except OSError as exc:
        return None, 0, False, str(exc)


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND lower(name)=lower(?)",
        (table,),
    ).fetchone()
    return bool(row)


def _query_rows(
    conn: sqlite3.Connection,
    table: str,
    sql: str,
    params: Iterable[Any],
    limit: int,
) -> Tuple[Optional[List[Dict[str, Any]]], bool, Optional[str]]:
    if not _table_exists(conn, table):
        return None, False, "table missing"
    try:
        rows = [_row_to_dict(row) for row in conn.execute(sql, tuple(params) + (limit + 1,)).fetchall()]
    except sqlite3.Error as exc:
        return None, False, str(exc)
    truncated = len(rows) > limit
    return rows[:limit], truncated, None


def _open_db(db_path: str) -> Tuple[Optional[sqlite3.Connection], Optional[str]]:
    if not db_path:
        return None, "database path not configured"
    abspath = os.path.abspath(db_path)
    if not os.path.exists(abspath):
        return None, f"database file not found: {db_path}"
    try:
        conn = sqlite3.connect(f"file:{abspath}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn, None
    except sqlite3.Error as exc:
        return None, str(exc)


def render_operator_notes_template(job: Dict[str, Any]) -> str:
    job_id = str(job.get("id") or "unknown")
    baseline_id = job.get("baseline_id")
    return f"""# SDRwatch Diagnostic Notes

Job ID: {job_id}
Baseline ID: {baseline_id if baseline_id is not None else "unknown"}

## Expected behavior


## Actual behavior


## Frequency or band affected


## Problem type

- [ ] false positive
- [ ] false negative
- [ ] wrong bandwidth
- [ ] wrong center
- [ ] merged signals
- [ ] split signals
- [ ] unstable baseline
- [ ] other

## Additional notes


"""


def _add_bytes(zf: zipfile.ZipFile, manifest: BundleManifest, path: str, data: bytes) -> None:
    zf.writestr(path, data)
    manifest.included(path)


def _add_json(zf: zipfile.ZipFile, manifest: BundleManifest, path: str, value: Any) -> None:
    _add_bytes(zf, manifest, path, _json_bytes(value))


def _job_params(job: Dict[str, Any]) -> Dict[str, Any]:
    params = job.get("params")
    return dict(params) if isinstance(params, dict) else {}


def _diagnostic_path(job: Dict[str, Any]) -> str:
    params = _job_params(job)
    value = params.get("diagnostic_jsonl") or job.get("diagnostic_jsonl")
    return str(value or "")


def _count_key(counter: Dict[str, int], key: Any) -> None:
    if key in (None, ""):
        key = "unknown"
    key_text = str(key)
    counter[key_text] = counter.get(key_text, 0) + 1


def _summarize_decision_tail(text: str) -> Tuple[Dict[str, Any], bool]:
    summary: Dict[str, Any] = {
        "event_counts": {},
        "persistence_actions": {},
        "width_stages": {},
        "revisit_events": {},
        "effective_settings": {},
        "aggregate_counts": {
            "segment_inventory_count": 0,
            "cluster_emitted_count": 0,
            "cluster_rejected_count": 0,
            "persistence_match_count": 0,
            "persistence_no_match_count": 0,
            "persistence_cross_sweep_promote_count": 0,
            "width_decision_count": 0,
            "revisit_queued_count": 0,
            "revisit_result_count": 0,
            "characterization_record_count": 0,
        },
        "parse_errors": 0,
        "truncated": False,
    }
    has_decision_evidence = False
    effective_settings: Dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            summary["parse_errors"] += 1
            continue
        if not isinstance(record, dict):
            continue
        event = record.get("event")
        if event in (None, ""):
            continue
        event_name = str(event)
        _count_key(summary["event_counts"], event_name)
        aggregates = summary["aggregate_counts"]
        if event_name == "segment_inventory":
            aggregates["segment_inventory_count"] += 1
        elif event_name == "cluster_emit":
            aggregates["cluster_emitted_count"] += 1
            has_decision_evidence = True
        elif event_name == "cluster_reject":
            aggregates["cluster_rejected_count"] += 1
            has_decision_evidence = True
        elif event_name == "width_decision":
            aggregates["width_decision_count"] += 1
        elif event_name == "revisit_queue":
            aggregates["revisit_queued_count"] += 1
        elif event_name == "revisit_result":
            aggregates["revisit_result_count"] += 1
        elif event_name == "characterization_record":
            aggregates["characterization_record_count"] += 1
        if event_name == "detection_window" and isinstance(record.get("tuning_params"), dict):
            effective_settings = dict(record["tuning_params"])
        elif event_name == "sweep_start" and isinstance(record.get("params"), dict):
            effective_settings = dict(record["params"])
        elif event_name == "persistence_decision":
            has_decision_evidence = True
            action = record.get("action")
            _count_key(summary["persistence_actions"], action)
            if action == "match":
                aggregates["persistence_match_count"] += 1
            elif action in {"no_match", "cross_sweep_no_match"}:
                aggregates["persistence_no_match_count"] += 1
            elif action == "cross_sweep_promote":
                aggregates["persistence_cross_sweep_promote_count"] += 1
        elif event_name == "width_decision":
            has_decision_evidence = True
            _count_key(summary["width_stages"], record.get("stage"))
        elif event_name.startswith("revisit_"):
            has_decision_evidence = True
            _count_key(summary["revisit_events"], event_name)
    summary["effective_settings"] = effective_settings
    return summary, has_decision_evidence


def _clean_job_record(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in record.items()
        if key not in {"event", "ts", "run_id", "sweep_id", "time_utc"}
    }


def _extract_job_level_diagnostics(text: str) -> Dict[str, Any]:
    extracted: Dict[str, Any] = {"effective_parameters": None, "device_telemetry": None, "parse_errors": 0}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            extracted["parse_errors"] += 1
            continue
        if not isinstance(record, dict):
            continue
        event = record.get("event")
        if event == "effective_parameters":
            extracted["effective_parameters"] = _clean_job_record(record)
        elif event == "device_telemetry":
            extracted["device_telemetry"] = _clean_job_record(record)
    return extracted


def _summarize_characterization_tail(text: str, sample_limit: int) -> Dict[str, Any]:
    parse_errors = 0
    records: List[Dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            parse_errors += 1
            continue
        if isinstance(record, dict):
            records.append(record)
    summary = summarize_characterization_records(records, sample_limit=sample_limit)
    summary["parse_errors"] = parse_errors
    return summary


def _add_log_evidence(
    zf: zipfile.ZipFile,
    manifest: BundleManifest,
    job: Dict[str, Any],
    scanner_log_text: Optional[str],
    bounds: DiagnosticBundleBounds,
) -> None:
    if scanner_log_text is not None:
        text, count, truncated = tail_text(scanner_log_text, bounds.log_tail_lines)
        if truncated:
            manifest.truncated("scanner_log", "tail limited", included_lines=count)
        _add_bytes(zf, manifest, "logs/scanner-log-tail.txt", text.encode("utf-8"))
        return

    text, count, truncated, error = tail_file(str(job.get("log_path") or ""), bounds.log_tail_lines)
    if error:
        manifest.missing("scanner_log", error)
        return
    if truncated:
        manifest.truncated("scanner_log", "tail limited", included_lines=count)
    _add_bytes(zf, manifest, "logs/scanner-log-tail.txt", (text or "").encode("utf-8"))


def _add_diagnostic_jsonl(
    zf: zipfile.ZipFile,
    manifest: BundleManifest,
    job: Dict[str, Any],
    bounds: DiagnosticBundleBounds,
) -> Dict[str, Any]:
    path = _diagnostic_path(job)
    text, count, truncated, error = tail_file(path, bounds.diagnostic_tail_lines)
    if error:
        manifest.missing("diagnostic_jsonl", error)
        manifest.missing("decision_evidence", "diagnostic JSONL unavailable")
        return {"effective_parameters": None, "device_telemetry": None}
    if truncated:
        manifest.truncated("diagnostic_jsonl", "tail limited", included_lines=count)
        manifest.truncated("decision_evidence", "diagnostic tail limited", included_lines=count)
    text = text or ""
    _add_bytes(zf, manifest, "diagnostics/diagnostic-jsonl-tail.jsonl", text.encode("utf-8"))
    summary, has_decision_evidence = _summarize_decision_tail(text)
    summary["truncated"] = bool(truncated)
    _add_json(zf, manifest, "diagnostics/decision-summary.json", summary)
    job_level = _extract_job_level_diagnostics(text)
    characterization_summary = _summarize_characterization_tail(text, bounds.row_limit)
    if characterization_summary.get("truncated"):
        manifest.truncated(
            "characterization_summary",
            "sample limited",
            included_records=min(
                int(characterization_summary.get("record_count", 0) or 0),
                bounds.row_limit,
            ),
        )
    _add_json(zf, manifest, "diagnostics/characterization-summary.json", characterization_summary)
    if not has_decision_evidence:
        manifest.missing("decision_evidence", "no decision events in diagnostic JSONL tail")
    return job_level


def _add_database_evidence(
    zf: zipfile.ZipFile,
    manifest: BundleManifest,
    *,
    db_path: str,
    job: Dict[str, Any],
    bounds: DiagnosticBundleBounds,
) -> None:
    baseline_id = job.get("baseline_id") or _job_params(job).get("baseline_id")
    if baseline_id is None:
        for category in (
            "baseline",
            "baseline_detections",
            "scan_updates",
            "monitoring_zones",
            "friendly_signals",
        ):
            manifest.missing(category, "job has no baseline_id")
        return

    conn, error = _open_db(db_path)
    if error or conn is None:
        for category in (
            "baseline",
            "baseline_detections",
            "scan_updates",
            "monitoring_zones",
            "friendly_signals",
        ):
            manifest.missing(category, error or "database unavailable")
        return

    try:
        if _table_exists(conn, "baselines"):
            row = conn.execute("SELECT * FROM baselines WHERE id = ?", (baseline_id,)).fetchone()
            if row:
                _add_json(zf, manifest, "database/baseline.json", _row_to_dict(row))
            else:
                manifest.missing("baseline", f"baseline {baseline_id} not found")
        else:
            manifest.missing("baseline", "table missing")

        queries = [
            (
                "baseline_detections",
                "database/baseline-detections.json",
                """
                SELECT * FROM baseline_detections
                WHERE baseline_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (baseline_id,),
            ),
            (
                "scan_updates",
                "database/scan-updates.json",
                """
                SELECT * FROM scan_updates
                WHERE baseline_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (baseline_id,),
            ),
            (
                "monitoring_zones",
                "database/monitoring-zones.json",
                """
                SELECT * FROM monitoring_zones
                WHERE baseline_id = ? AND enabled = 1
                ORDER BY priority, f_start_hz
                LIMIT ?
                """,
                (baseline_id,),
            ),
            (
                "friendly_signals",
                "database/friendly-signals.json",
                """
                SELECT * FROM friendly_signals
                WHERE baseline_id = ?
                ORDER BY f_center_hz
                LIMIT ?
                """,
                (baseline_id,),
            ),
        ]
        for table, archive_path, sql, params in queries:
            rows, truncated, query_error = _query_rows(conn, table, sql, params, bounds.row_limit)
            if query_error:
                manifest.missing(table, query_error)
                continue
            if rows == []:
                manifest.missing(table, "no records configured")
            if truncated:
                manifest.truncated(table, "row limited", included_rows=bounds.row_limit)
            _add_json(zf, manifest, archive_path, rows or [])
    finally:
        conn.close()


def build_diagnostic_bundle(
    *,
    job: Dict[str, Any],
    db_path: str,
    scanner_log_text: Optional[str] = None,
    bounds: Optional[DiagnosticBundleBounds] = None,
) -> DiagnosticBundle:
    """Build a bounded local diagnostic bundle archive for a job."""
    bounds = bounds or DiagnosticBundleBounds()
    manifest = BundleManifest(job=job, bounds=bounds)
    job_id = str(job.get("id") or "unknown")
    filename = f"{DIAGNOSTIC_BUNDLE_FILENAME_PREFIX}-{job_id}.zip"

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        _add_json(zf, manifest, "job/job.json", job)
        _add_json(zf, manifest, "job/params.json", _job_params(job))
        cmd = job.get("cmd") or []
        if isinstance(cmd, list):
            command_text = shlex.join(str(part) for part in cmd)
        else:
            command_text = str(cmd)
        _add_bytes(zf, manifest, "job/scanner-command.txt", (command_text + "\n").encode("utf-8"))
        _add_log_evidence(zf, manifest, job, scanner_log_text, bounds)
        job_level = _add_diagnostic_jsonl(zf, manifest, job, bounds)
        params = _job_params(job)
        device_telemetry = job_level.get("device_telemetry")
        if device_telemetry is None:
            device_telemetry = build_device_telemetry_snapshot(
                type("DiagnosticArgs", (), params)(),
                None,
                device_key=str(job.get("device_key") or params.get("device_key") or ""),
            )
        effective_parameters = job_level.get("effective_parameters")
        if effective_parameters is None:
            effective_parameters = build_effective_parameter_manifest(
                type("DiagnosticArgs", (), params)(),
                job_id=str(job.get("id") or "unknown"),
                device_telemetry=device_telemetry,
            )
        manifest.data["effective_parameters"] = effective_parameters
        _add_json(zf, manifest, "job/effective-parameters.json", effective_parameters)
        if device_telemetry is not None:
            manifest.data["device_telemetry"] = device_telemetry
            _add_json(zf, manifest, "job/device-telemetry.json", device_telemetry)
        _add_database_evidence(zf, manifest, db_path=db_path, job=job, bounds=bounds)
        _add_bytes(zf, manifest, "NOTES.md", render_operator_notes_template(job).encode("utf-8"))
        _add_json(zf, manifest, "manifest.json", manifest.data)

    return DiagnosticBundle(filename=filename, content=buffer.getvalue(), manifest=manifest.data)
