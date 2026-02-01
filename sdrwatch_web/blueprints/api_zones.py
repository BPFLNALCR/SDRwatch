"""
Monitoring zones and friendly signals API blueprint for SDRwatch Web.

Provides endpoints for managing monitoring zones (frequency ranges to scan)
and friendly signals (known-good frequencies to suppress from alerts).
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from flask import Blueprint, abort, current_app, jsonify, request

from sdrwatch_web.auth import require_auth
from sdrwatch_web.db import get_con, q1, qa
from sdrwatch_web.schema import PRESET_MONITORING_ZONES, ZONE_CATEGORIES, seed_preset_zones

bp = Blueprint("api_zones", __name__)


# ---------------------------------------------------------------------------
# Bandplan Detection
# ---------------------------------------------------------------------------


@bp.get("/api/bandplans")
def api_bandplans():
    """
    Detect available bandplan CSV files.
    
    Scans for files matching bandplan_*.csv in the project directory.
    Returns a list of bandplan files with display names derived from filenames.
    """
    require_auth()
    bandplans = []
    
    # Look in multiple possible locations
    search_paths = []
    
    # 1. Project root (where sdrwatch-control.py is)
    project_root = Path(__file__).resolve().parent.parent.parent
    search_paths.append(project_root)
    
    # 2. Current working directory
    search_paths.append(Path.cwd())
    
    # 3. /opt/sdrwatch if it exists
    opt_path = Path("/opt/sdrwatch")
    if opt_path.exists():
        search_paths.append(opt_path)
    
    seen = set()
    for base in search_paths:
        for f in base.glob("bandplan_*.csv"):
            if f.name in seen:
                continue
            seen.add(f.name)
            
            # Derive display name from filename
            # bandplan_eu.csv -> "EU"
            # bandplan_us_na.csv -> "US NA"
            name_part = f.stem.replace("bandplan_", "").replace("_", " ").upper()
            bandplans.append({
                "filename": f.name,
                "path": str(f),
                "display_name": name_part or f.stem,
            })
    
    # Sort by display name
    bandplans.sort(key=lambda x: x["display_name"])
    
    return jsonify({"bandplans": bandplans})


# ---------------------------------------------------------------------------
# Helper to get writable DB connection
# ---------------------------------------------------------------------------


def _get_writable_db():
    """Get a writable database connection."""
    from flask import current_app

    db_path = current_app.config.get("SDRWATCH_DB_PATH") or current_app.config.get("DB_PATH")
    if not db_path:
        abort(500, description="No database path configured")

    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        abort(500, description=f"Database error: {e}")


# ---------------------------------------------------------------------------
# Monitoring Zones
# ---------------------------------------------------------------------------


@bp.get("/api/zones/categories")
def api_zone_categories():
    """Get available zone categories for grouping."""
    require_auth()
    return jsonify({"categories": ZONE_CATEGORIES})


@bp.get("/api/zones/presets")
def api_zone_presets():
    """Get the list of preset monitoring zones (for reference)."""
    require_auth()
    return jsonify({
        "presets": PRESET_MONITORING_ZONES,
        "categories": ZONE_CATEGORIES,
    })


@bp.get("/api/baseline/<int:baseline_id>/zones")
def api_zones_list(baseline_id: int):
    """List all monitoring zones for a baseline."""
    require_auth()
    con = get_con()

    # Verify baseline exists
    baseline = q1(con, "SELECT id, name FROM baselines WHERE id = ?", (baseline_id,))
    if not baseline:
        abort(404, description="Baseline not found")

    try:
        zones = qa(
            con,
            """
            SELECT id, baseline_id, name, description, f_start_hz, f_stop_hz,
                   category, priority, enabled, is_preset, created_at
            FROM monitoring_zones
            WHERE baseline_id = ?
            ORDER BY priority, f_start_hz
            """,
            (baseline_id,),
        )
    except sqlite3.OperationalError as e:
        abort(500, description=f"Database error: {e}")

    return jsonify({
        "baseline_id": baseline_id,
        "zones": zones,
        "categories": ZONE_CATEGORIES,
    })


@bp.post("/api/baseline/<int:baseline_id>/zones")
def api_zones_create(baseline_id: int):
    """Create a new monitoring zone for a baseline."""
    require_auth()
    payload = request.get_json(force=True, silent=False) or {}

    name = (payload.get("name") or "").strip()
    if not name:
        abort(400, description="name is required")

    f_start_hz = payload.get("f_start_hz")
    f_stop_hz = payload.get("f_stop_hz")

    if f_start_hz is None or f_stop_hz is None:
        abort(400, description="f_start_hz and f_stop_hz are required")

    try:
        f_start_hz = int(f_start_hz)
        f_stop_hz = int(f_stop_hz)
    except (TypeError, ValueError):
        abort(400, description="f_start_hz and f_stop_hz must be integers")

    if f_start_hz >= f_stop_hz:
        abort(400, description="f_start_hz must be less than f_stop_hz")

    conn = _get_writable_db()
    try:
        # Verify baseline exists
        cursor = conn.execute("SELECT id FROM baselines WHERE id = ?", (baseline_id,))
        if cursor.fetchone() is None:
            abort(404, description="Baseline not found")

        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat() + "Z"
        cursor = conn.execute(
            """
            INSERT INTO monitoring_zones (
                baseline_id, name, description, f_start_hz, f_stop_hz,
                category, priority, enabled, is_preset, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
            """,
            (
                baseline_id,
                name,
                payload.get("description", ""),
                f_start_hz,
                f_stop_hz,
                payload.get("category", "custom"),
                payload.get("priority", 100),
                1 if payload.get("enabled", True) else 0,
                now,
            ),
        )
        zone_id = cursor.lastrowid
        conn.commit()

        row = conn.execute(
            """
            SELECT id, baseline_id, name, description, f_start_hz, f_stop_hz,
                   category, priority, enabled, is_preset, created_at
            FROM monitoring_zones WHERE id = ?
            """,
            (zone_id,),
        ).fetchone()
    finally:
        conn.close()

    return (jsonify({"zone": dict(row) if row else None}), 201)


@bp.patch("/api/zones/<int:zone_id>")
def api_zones_update(zone_id: int):
    """Update a monitoring zone."""
    require_auth()
    payload = request.get_json(force=True, silent=False) or {}

    conn = _get_writable_db()
    try:
        # Verify zone exists
        cursor = conn.execute(
            "SELECT id, baseline_id FROM monitoring_zones WHERE id = ?",
            (zone_id,),
        )
        existing = cursor.fetchone()
        if existing is None:
            abort(404, description="Zone not found")

        updates = []
        params = []

        if "name" in payload:
            name = (payload["name"] or "").strip()
            if not name:
                abort(400, description="name cannot be empty")
            updates.append("name = ?")
            params.append(name)

        if "description" in payload:
            updates.append("description = ?")
            params.append(payload["description"] or "")

        if "f_start_hz" in payload:
            updates.append("f_start_hz = ?")
            params.append(int(payload["f_start_hz"]))

        if "f_stop_hz" in payload:
            updates.append("f_stop_hz = ?")
            params.append(int(payload["f_stop_hz"]))

        if "category" in payload:
            updates.append("category = ?")
            params.append(payload["category"] or "custom")

        if "priority" in payload:
            updates.append("priority = ?")
            params.append(int(payload["priority"]))

        if "enabled" in payload:
            updates.append("enabled = ?")
            params.append(1 if payload["enabled"] else 0)

        if not updates:
            abort(400, description="No fields to update")

        params.append(zone_id)
        conn.execute(
            f"UPDATE monitoring_zones SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )
        conn.commit()

        row = conn.execute(
            """
            SELECT id, baseline_id, name, description, f_start_hz, f_stop_hz,
                   category, priority, enabled, is_preset, created_at
            FROM monitoring_zones WHERE id = ?
            """,
            (zone_id,),
        ).fetchone()
    finally:
        conn.close()

    return jsonify({"zone": dict(row) if row else None})


@bp.delete("/api/zones/<int:zone_id>")
def api_zones_delete(zone_id: int):
    """Delete a monitoring zone."""
    require_auth()

    conn = _get_writable_db()
    try:
        cursor = conn.execute(
            "SELECT id FROM monitoring_zones WHERE id = ?",
            (zone_id,),
        )
        if cursor.fetchone() is None:
            abort(404, description="Zone not found")

        conn.execute("DELETE FROM monitoring_zones WHERE id = ?", (zone_id,))
        conn.commit()
    finally:
        conn.close()

    return jsonify({"deleted": zone_id})


@bp.post("/api/baseline/<int:baseline_id>/zones/reset-presets")
def api_zones_reset_presets(baseline_id: int):
    """Reset preset zones to defaults (deletes presets, re-seeds them)."""
    require_auth()

    conn = _get_writable_db()
    try:
        # Verify baseline exists
        cursor = conn.execute("SELECT id FROM baselines WHERE id = ?", (baseline_id,))
        if cursor.fetchone() is None:
            abort(404, description="Baseline not found")

        # Delete existing preset zones
        conn.execute(
            "DELETE FROM monitoring_zones WHERE baseline_id = ? AND is_preset = 1",
            (baseline_id,),
        )

        # Re-seed
        count = seed_preset_zones(conn, baseline_id)
        conn.commit()
    finally:
        conn.close()

    return jsonify({"reset": True, "zones_created": count})


@bp.post("/api/baseline/<int:baseline_id>/zones/bulk-enable")
def api_zones_bulk_enable(baseline_id: int):
    """Enable or disable multiple zones at once."""
    require_auth()
    payload = request.get_json(force=True, silent=False) or {}

    zone_ids = payload.get("zone_ids", [])
    enabled = payload.get("enabled", True)

    if not zone_ids or not isinstance(zone_ids, list):
        abort(400, description="zone_ids must be a non-empty list")

    conn = _get_writable_db()
    try:
        placeholders = ",".join("?" * len(zone_ids))
        conn.execute(
            f"""
            UPDATE monitoring_zones
            SET enabled = ?
            WHERE baseline_id = ? AND id IN ({placeholders})
            """,
            [1 if enabled else 0, baseline_id] + zone_ids,
        )
        conn.commit()
    finally:
        conn.close()

    return jsonify({"updated": len(zone_ids), "enabled": enabled})


# ---------------------------------------------------------------------------
# Friendly Signals
# ---------------------------------------------------------------------------


@bp.get("/api/baseline/<int:baseline_id>/friendly")
def api_friendly_list(baseline_id: int):
    """List all friendly signals for a baseline."""
    require_auth()
    con = get_con()

    # Verify baseline exists
    baseline = q1(con, "SELECT id, name FROM baselines WHERE id = ?", (baseline_id,))
    if not baseline:
        abort(404, description="Baseline not found")

    try:
        signals = qa(
            con,
            """
            SELECT id, baseline_id, f_center_hz, f_tolerance_hz, label, notes, source, created_at
            FROM friendly_signals
            WHERE baseline_id = ?
            ORDER BY f_center_hz
            """,
            (baseline_id,),
        )
    except sqlite3.OperationalError as e:
        abort(500, description=f"Database error: {e}")

    return jsonify({
        "baseline_id": baseline_id,
        "friendly_signals": signals,
    })


@bp.post("/api/baseline/<int:baseline_id>/friendly")
def api_friendly_create(baseline_id: int):
    """Add a friendly signal to a baseline."""
    require_auth()
    payload = request.get_json(force=True, silent=False) or {}

    label = (payload.get("label") or "").strip()
    if not label:
        abort(400, description="label is required")

    f_center_hz = payload.get("f_center_hz")
    if f_center_hz is None:
        abort(400, description="f_center_hz is required")

    try:
        f_center_hz = int(f_center_hz)
    except (TypeError, ValueError):
        abort(400, description="f_center_hz must be an integer")

    f_tolerance_hz = payload.get("f_tolerance_hz", 5000)
    try:
        f_tolerance_hz = int(f_tolerance_hz)
    except (TypeError, ValueError):
        f_tolerance_hz = 5000

    conn = _get_writable_db()
    try:
        # Verify baseline exists
        cursor = conn.execute("SELECT id FROM baselines WHERE id = ?", (baseline_id,))
        if cursor.fetchone() is None:
            abort(404, description="Baseline not found")

        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat() + "Z"
        cursor = conn.execute(
            """
            INSERT INTO friendly_signals (
                baseline_id, f_center_hz, f_tolerance_hz, label, notes, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                baseline_id,
                f_center_hz,
                f_tolerance_hz,
                label,
                payload.get("notes", ""),
                payload.get("source", "user"),
                now,
            ),
        )
        signal_id = cursor.lastrowid
        conn.commit()

        row = conn.execute(
            """
            SELECT id, baseline_id, f_center_hz, f_tolerance_hz, label, notes, source, created_at
            FROM friendly_signals WHERE id = ?
            """,
            (signal_id,),
        ).fetchone()
    finally:
        conn.close()

    return (jsonify({"friendly_signal": dict(row) if row else None}), 201)


@bp.patch("/api/friendly/<int:signal_id>")
def api_friendly_update(signal_id: int):
    """Update a friendly signal."""
    require_auth()
    payload = request.get_json(force=True, silent=False) or {}

    conn = _get_writable_db()
    try:
        cursor = conn.execute(
            "SELECT id FROM friendly_signals WHERE id = ?",
            (signal_id,),
        )
        if cursor.fetchone() is None:
            abort(404, description="Friendly signal not found")

        updates = []
        params = []

        if "label" in payload:
            label = (payload["label"] or "").strip()
            if not label:
                abort(400, description="label cannot be empty")
            updates.append("label = ?")
            params.append(label)

        if "f_center_hz" in payload:
            updates.append("f_center_hz = ?")
            params.append(int(payload["f_center_hz"]))

        if "f_tolerance_hz" in payload:
            updates.append("f_tolerance_hz = ?")
            params.append(int(payload["f_tolerance_hz"]))

        if "notes" in payload:
            updates.append("notes = ?")
            params.append(payload["notes"] or "")

        if "source" in payload:
            updates.append("source = ?")
            params.append(payload["source"] or "user")

        if not updates:
            abort(400, description="No fields to update")

        params.append(signal_id)
        conn.execute(
            f"UPDATE friendly_signals SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )
        conn.commit()

        row = conn.execute(
            """
            SELECT id, baseline_id, f_center_hz, f_tolerance_hz, label, notes, source, created_at
            FROM friendly_signals WHERE id = ?
            """,
            (signal_id,),
        ).fetchone()
    finally:
        conn.close()

    return jsonify({"friendly_signal": dict(row) if row else None})


@bp.delete("/api/friendly/<int:signal_id>")
def api_friendly_delete(signal_id: int):
    """Delete a friendly signal."""
    require_auth()

    conn = _get_writable_db()
    try:
        cursor = conn.execute(
            "SELECT id FROM friendly_signals WHERE id = ?",
            (signal_id,),
        )
        if cursor.fetchone() is None:
            abort(404, description="Friendly signal not found")

        conn.execute("DELETE FROM friendly_signals WHERE id = ?", (signal_id,))
        conn.commit()
    finally:
        conn.close()

    return jsonify({"deleted": signal_id})


@bp.post("/api/baseline/<int:baseline_id>/friendly/from-detection/<int:detection_id>")
def api_friendly_from_detection(baseline_id: int, detection_id: int):
    """
    Create a friendly signal entry from an existing detection.
    
    This allows users to mark a detected signal as "friendly" so it
    won't be flagged as suspicious in future scans.
    """
    require_auth()
    payload = request.get_json(force=True, silent=True) or {}

    conn = _get_writable_db()
    try:
        # Get the detection
        cursor = conn.execute(
            """
            SELECT id, baseline_id, f_center_hz, f_low_hz, f_high_hz, service, label
            FROM baseline_detections
            WHERE id = ? AND baseline_id = ?
            """,
            (detection_id, baseline_id),
        )
        detection = cursor.fetchone()
        if detection is None:
            abort(404, description="Detection not found")

        detection = dict(detection)
        f_center = detection["f_center_hz"]
        f_low = detection.get("f_low_hz") or f_center
        f_high = detection.get("f_high_hz") or f_center
        bandwidth = f_high - f_low
        tolerance = max(5000, bandwidth // 2)  # At least 5kHz, or half bandwidth

        # Use provided label or derive from detection
        label = (payload.get("label") or "").strip()
        if not label:
            label = detection.get("label") or detection.get("service") or f"Signal @ {f_center/1e6:.3f} MHz"

        now = datetime.now(timezone.utc).replace(microsecond=0).isoformat() + "Z"
        cursor = conn.execute(
            """
            INSERT INTO friendly_signals (
                baseline_id, f_center_hz, f_tolerance_hz, label, notes, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                baseline_id,
                f_center,
                tolerance,
                label,
                payload.get("notes", f"Auto-created from detection #{detection_id}"),
                "detection",
                now,
            ),
        )
        signal_id = cursor.lastrowid
        conn.commit()

        row = conn.execute(
            """
            SELECT id, baseline_id, f_center_hz, f_tolerance_hz, label, notes, source, created_at
            FROM friendly_signals WHERE id = ?
            """,
            (signal_id,),
        ).fetchone()
    finally:
        conn.close()

    return (jsonify({"friendly_signal": dict(row) if row else None}), 201)
