"""
Job management API blueprint for SDRwatch Web.

Provides endpoints for listing, starting, stopping jobs, and streaming logs.
Also includes the /api/scans aliases and /api/now, /api/logs endpoints.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from flask import Blueprint, Response, abort, jsonify, request

from sdrwatch_web.auth import require_auth
from sdrwatch_web.controller import controller_active_job, get_controller

bp = Blueprint("api_jobs", __name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def parse_window_log_line(line: str) -> Optional[Dict[str, Any]]:
    """Parse a scanner window log line into a structured dict."""
    prefix = "[scan] window"
    if not isinstance(line, str):
        return None
    text = line.strip()
    if not text.startswith(prefix):
        return None
    payload = text[len(prefix):].strip()
    if not payload:
        return None

    result: Dict[str, Any] = {"raw": line.rstrip("\n")}
    required_keys = {"center_hz", "det_count", "mean_db", "p90_db", "anomalous"}
    seen: Dict[str, Any] = {}

    for chunk in payload.split():
        if "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        key = key.strip()
        value = value.strip()

        if key == "center_hz":
            try:
                seen[key] = float(value)
            except Exception:
                return None
        elif key == "det_count":
            try:
                seen[key] = int(float(value))
            except Exception:
                return None
        elif key in {"mean_db", "p90_db"}:
            try:
                seen[key] = float(value)
            except Exception:
                return None
        elif key == "anomalous":
            try:
                seen[key] = bool(int(float(value)))
            except Exception:
                if value.lower() in {"true", "false"}:
                    seen[key] = value.lower() == "true"
                else:
                    return None

    if not required_keys.issubset(seen.keys()):
        return None

    result.update(seen)
    return result


def active_state_payload() -> Dict[str, Any]:
    """Build the active state payload (running job or idle)."""
    job = controller_active_job()
    if not job:
        return {"state": "idle"}
    return {"state": "running", "job": job}


def start_job_from_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Validate and start a job from request payload."""
    device_key = payload.get("device_key")
    if not device_key:
        abort(400, description="device_key is required")
        return None  # Unreachable but satisfies type checker

    label = str(payload.get("label") or "web")
    params = payload.get("params") or {}
    baseline_id_raw = payload.get("baseline_id")

    if baseline_id_raw is None:
        abort(400, description="baseline_id is required; create or select a baseline first.")

    if isinstance(baseline_id_raw, str):
        candidate = baseline_id_raw.strip()
        if not candidate:
            abort(400, description="baseline_id is required; create or select a baseline first.")
        baseline_id_token: Any = candidate
    else:
        baseline_id_token = baseline_id_raw

    try:
        baseline_id_int = int(baseline_id_token)
    except Exception as exc:
        abort(400, description=f"invalid baseline_id: {exc}")
        return None  # Unreachable but satisfies type checker

    ctl = get_controller()
    try:
        return ctl.start_job(str(device_key), label, baseline_id_int, params)
    except Exception as exc:
        abort(400, description=str(exc))


def stop_job_by_id(job_id: str) -> Dict[str, Any]:
    """Stop a job by ID."""
    ctl = get_controller()
    try:
        return ctl.stop_job(job_id)
    except Exception as exc:
        abort(400, description=str(exc))
        return {"error": str(exc)}  # Unreachable but satisfies type checker


def job_logs_response(job_id: str, tail: Optional[int] = None) -> Response:
    """Build a Response with job logs."""
    ctl = get_controller()
    try:
        data = ctl.job_logs(job_id, tail=tail)
        return Response(data or "", mimetype="text/plain")
    except Exception as exc:
        abort(404, description=str(exc))


def start_job_response():
    """Handle job start request and return response."""
    require_auth()
    payload = request.get_json(force=True, silent=False) or {}
    job = start_job_from_payload(payload)
    state_val = job.get("status", "running") if isinstance(job, dict) else "running"
    return jsonify({"state": state_val, "job": job})


# ---------------------------------------------------------------------------
# Job list
# ---------------------------------------------------------------------------


@bp.get("/api/jobs")
def api_jobs_list():
    """List all jobs."""
    require_auth()
    ctl = get_controller()
    try:
        jobs = ctl.list_jobs()
        return jsonify({"jobs": jobs})
    except Exception as exc:
        abort(502, description=str(exc))


# ---------------------------------------------------------------------------
# Active job
# ---------------------------------------------------------------------------


@bp.get("/api/jobs/active")
def api_jobs_active():
    """Get the currently active job."""
    require_auth()
    return jsonify(active_state_payload())


# ---------------------------------------------------------------------------
# Start job
# ---------------------------------------------------------------------------


@bp.post("/api/jobs")
def api_jobs_create():
    """Start a new job."""
    return start_job_response()


@bp.post("/api/scans")
def api_start_scan():
    """Start a new scan (alias for POST /api/jobs)."""
    return start_job_response()


# ---------------------------------------------------------------------------
# Job detail
# ---------------------------------------------------------------------------


@bp.get("/api/jobs/<job_id>")
def api_job_detail(job_id: str):
    """Get details for a specific job."""
    require_auth()
    ctl = get_controller()
    try:
        job = ctl.job_detail(job_id)
        return jsonify(job)
    except Exception as exc:
        abort(404, description=str(exc))


# ---------------------------------------------------------------------------
# Stop job
# ---------------------------------------------------------------------------


@bp.delete("/api/jobs/<job_id>")
def api_job_delete(job_id: str):
    """Stop a job by ID."""
    require_auth()
    data = stop_job_by_id(job_id)
    return jsonify(data)


@bp.delete("/api/scans/active")
def api_stop_active():
    """Stop the currently active scan."""
    require_auth()
    job = controller_active_job()
    if not job:
        return jsonify({"ok": True})
    data = stop_job_by_id(str(job.get("id")))
    return jsonify(data)


# ---------------------------------------------------------------------------
# Job logs
# ---------------------------------------------------------------------------


@bp.get("/api/jobs/<job_id>/logs")
def api_job_logs(job_id: str):
    """Get logs for a specific job."""
    require_auth()
    tail = request.args.get("tail", type=int)
    return job_logs_response(job_id, tail)


# ---------------------------------------------------------------------------
# Live state and logs
# ---------------------------------------------------------------------------


@bp.get("/api/now")
def api_now():
    """Get current active state (alias for /api/jobs/active)."""
    require_auth()
    return jsonify(active_state_payload())


@bp.get("/api/logs")
def api_logs():
    """Get logs for the active job or specified job_id."""
    require_auth()
    job_id = request.args.get("job_id")
    tail = request.args.get("tail", type=int)

    if not job_id:
        job = controller_active_job()
        if not job:
            return Response("", mimetype="text/plain")
        job_id = str(job.get("id"))

    return job_logs_response(job_id, tail)


@bp.get("/api/live/windows")
def api_live_windows():
    """Get parsed window log lines for live visualization."""
    require_auth()

    job_id = request.args.get("job_id")
    tail = request.args.get("tail", type=int)
    limit = request.args.get("limit", type=int) or 100
    limit = max(1, min(500, limit))
    tail = tail if tail and tail > 0 else 5000

    if not job_id:
        job = controller_active_job()
        if not job:
            return jsonify({"windows": []})
        job_id = str(job.get("id"))

    ctl = get_controller()
    try:
        log_text = ctl.job_logs(job_id, tail=tail)
    except Exception as exc:
        abort(502, description=str(exc))

    windows: List[Dict[str, Any]] = []
    for line in log_text.splitlines():
        parsed = parse_window_log_line(line)
        if not parsed:
            continue
        center_hz = float(parsed.get("center_hz", 0.0))
        parsed["center_hz"] = center_hz
        parsed["center_mhz"] = center_hz / 1e6
        windows.append(parsed)

    if not windows:
        return jsonify({"windows": []})

    windows = windows[-limit:]
    return jsonify({"windows": windows})
