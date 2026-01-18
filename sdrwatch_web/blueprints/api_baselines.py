"""
Baseline API blueprint for SDRwatch Web.

Provides endpoints for baseline CRUD, tactical snapshots, hotspots,
and change events.
"""
from __future__ import annotations

from flask import Blueprint, abort, jsonify, request

from sdrwatch_web.auth import require_auth
from sdrwatch_web.baseline_helpers import (
    baseline_summary_map,
    change_events_payload,
    controller_baselines,
    hotspots_payload,
    tactical_snapshot_payload,
)
from sdrwatch_web.config import CHANGE_WINDOW_MINUTES
from sdrwatch_web.schema import create_baseline_entry

bp = Blueprint("api_baselines", __name__)


# ---------------------------------------------------------------------------
# Baseline list
# ---------------------------------------------------------------------------


@bp.get("/api/baselines")
def api_baselines_list():
    """List all baselines with summaries."""
    require_auth()
    baselines_payload = {
        "baselines": controller_baselines(),
        "summaries": baseline_summary_map(),
    }
    return jsonify(baselines_payload)


# ---------------------------------------------------------------------------
# Create baseline
# ---------------------------------------------------------------------------


@bp.post("/api/baselines")
def api_baselines_create():
    """Create a new baseline."""
    require_auth()
    payload = request.get_json(force=True, silent=False) or {}

    try:
        row = create_baseline_entry(payload)
    except ValueError as exc:
        abort(400, description=str(exc))
    except RuntimeError as exc:
        abort(500, description=str(exc))

    summaries = baseline_summary_map()
    return (jsonify({"baseline": row, "summaries": summaries}), 201)


# ---------------------------------------------------------------------------
# Tactical snapshot
# ---------------------------------------------------------------------------


@bp.get("/api/baseline/<int:baseline_id>/tactical")
def api_baseline_tactical(baseline_id: int):
    """Get tactical snapshot for a baseline."""
    require_auth()
    payload = tactical_snapshot_payload(baseline_id)
    if not payload:
        abort(404, description="Baseline not found")
    return jsonify(payload)


# ---------------------------------------------------------------------------
# Hotspots
# ---------------------------------------------------------------------------


@bp.get("/api/baseline/<int:baseline_id>/hotspots")
def api_baseline_hotspots(baseline_id: int):
    """Get hotspot heatmap data for a baseline."""
    require_auth()
    payload = hotspots_payload(baseline_id)
    if not payload:
        abort(404, description="Baseline not found")
    return jsonify(payload)


# ---------------------------------------------------------------------------
# Change events
# ---------------------------------------------------------------------------


@bp.get("/api/baseline/<int:baseline_id>/changes")
def api_baseline_changes(baseline_id: int):
    """Get change events for a baseline."""
    require_auth()
    minute_override = request.args.get("minutes", type=int)
    type_tokens = request.args.getlist("type")

    payload = change_events_payload(
        baseline_id,
        window_minutes=minute_override or CHANGE_WINDOW_MINUTES,
        event_types=type_tokens if type_tokens else None,
    )
    if not payload:
        abort(404, description="Baseline not found")
    return jsonify(payload)
