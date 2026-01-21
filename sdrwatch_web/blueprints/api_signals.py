"""
Signal management API endpoints for SDRwatch Web.

Provides REST API for viewing and updating signal classification,
labels, notes, and user-corrected bandwidth.
"""
from __future__ import annotations

import sqlite3
from typing import Any, Dict, List, Optional

from flask import Blueprint, current_app, jsonify, request

from sdrwatch_web.db import get_con, get_con_optional, q1, qa, table_exists
from sdrwatch_web.formatting import (
    compute_display_bandwidth_hz,
    format_bandwidth_khz,
    format_freq_label,
)
from sdrwatch_web.baseline_helpers import fetch_baseline_record


bp = Blueprint("api_signals", __name__, url_prefix="/api/signals")


# ---------------------------------------------------------------------------
# Signal ID formatting
# ---------------------------------------------------------------------------


def format_signal_id(detection_id: int) -> str:
    """
    Format a detection ID as a human-friendly signal identifier.

    Args:
        detection_id: Integer primary key from baseline_detections.

    Returns:
        String like "SIG-0042".
    """
    return f"SIG-{detection_id:04d}"


def parse_signal_id(signal_id: str) -> Optional[int]:
    """
    Parse a human-friendly signal ID back to integer.

    Args:
        signal_id: String like "SIG-0042" or just "42".

    Returns:
        Integer detection ID or None if invalid.
    """
    if not signal_id:
        return None
    cleaned = signal_id.strip().upper()
    if cleaned.startswith("SIG-"):
        cleaned = cleaned[4:]
    try:
        return int(cleaned)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Classification constants
# ---------------------------------------------------------------------------


VALID_CLASSIFICATIONS = {"friendly", "ambient", "hostile", "unknown"}


# ---------------------------------------------------------------------------
# GET /api/signals - List signals with optional filters
# ---------------------------------------------------------------------------


@bp.get("")
def list_signals():
    """
    List signals with optional filters.

    Query params:
        baseline_id: Filter by baseline (required)
        selected: Filter to selected signals only (optional, "1" or "true")
        classification: Filter by classification (optional)
        limit: Max results (default 100)

    Returns:
        JSON array of signal objects.
    """
    con = get_con_optional()
    if con is None:
        return jsonify({"error": "database not available"}), 503

    baseline_id_raw = request.args.get("baseline_id", "").strip()
    if not baseline_id_raw:
        return jsonify({"error": "baseline_id is required"}), 400

    try:
        baseline_id = int(baseline_id_raw)
    except ValueError:
        return jsonify({"error": "invalid baseline_id"}), 400

    baseline = fetch_baseline_record(baseline_id)
    if not baseline:
        return jsonify({"error": "baseline not found"}), 404

    conditions = ["baseline_id = ?"]
    params: List[Any] = [baseline_id]

    selected_param = request.args.get("selected", "").strip().lower()
    if selected_param in ("1", "true"):
        conditions.append("selected = 1")

    classification_param = request.args.get("classification", "").strip().lower()
    if classification_param and classification_param in VALID_CLASSIFICATIONS:
        conditions.append("classification = ?")
        params.append(classification_param)

    limit = 100
    limit_raw = request.args.get("limit", "").strip()
    if limit_raw:
        try:
            limit = max(1, min(500, int(limit_raw)))
        except ValueError:
            pass

    where_clause = " AND ".join(conditions)
    try:
        rows = qa(
            con,
            f"""
            SELECT id, baseline_id, f_low_hz, f_high_hz, f_center_hz,
                   first_seen_utc, last_seen_utc, total_hits, total_windows,
                   confidence, label, classification, user_bw_hz, notes, selected
            FROM baseline_detections
            WHERE {where_clause}
            ORDER BY last_seen_utc DESC
            LIMIT ?
            """,
            tuple(params) + (limit,),
        )
    except sqlite3.OperationalError as e:
        return jsonify({"error": f"query failed: {e}"}), 500

    signals = []
    for row in rows:
        det_id = row.get("id")
        f_low = row.get("f_low_hz")
        f_high = row.get("f_high_hz")
        bw = (f_high - f_low) if f_low is not None and f_high is not None else None
        user_bw = row.get("user_bw_hz")

        display_bw = compute_display_bandwidth_hz(
            baseline_row=baseline,
            f_low_hz=float(f_low) if f_low else None,
            f_high_hz=float(f_high) if f_high else None,
            bandwidth_hz=bw,
        )

        signals.append({
            "id": det_id,
            "signal_id": format_signal_id(det_id) if det_id else None,
            "baseline_id": row.get("baseline_id"),
            "f_center_hz": row.get("f_center_hz"),
            "f_low_hz": f_low,
            "f_high_hz": f_high,
            "bandwidth_hz": bw,
            "bandwidth_hz_display": user_bw if user_bw else display_bw,
            "user_bw_hz": user_bw,
            "first_seen_utc": row.get("first_seen_utc"),
            "last_seen_utc": row.get("last_seen_utc"),
            "total_hits": row.get("total_hits"),
            "total_windows": row.get("total_windows"),
            "confidence": row.get("confidence"),
            "label": row.get("label"),
            "classification": row.get("classification") or "unknown",
            "notes": row.get("notes"),
            "selected": bool(row.get("selected")),
        })

    return jsonify(signals)


# ---------------------------------------------------------------------------
# GET /api/signals/<id> - Get single signal detail
# ---------------------------------------------------------------------------


@bp.get("/<signal_id>")
def get_signal(signal_id: str):
    """
    Get detailed information about a single signal.

    Args:
        signal_id: Signal ID (e.g., "SIG-0042" or "42").

    Returns:
        JSON signal object with full detail.
    """
    con = get_con_optional()
    if con is None:
        return jsonify({"error": "database not available"}), 503

    det_id = parse_signal_id(signal_id)
    if det_id is None:
        return jsonify({"error": "invalid signal_id"}), 400

    try:
        row = q1(
            con,
            """
            SELECT id, baseline_id, f_low_hz, f_high_hz, f_center_hz,
                   first_seen_utc, last_seen_utc, total_hits, total_windows,
                   confidence, label, classification, user_bw_hz, notes, selected
            FROM baseline_detections
            WHERE id = ?
            """,
            (det_id,),
        )
    except sqlite3.OperationalError as e:
        return jsonify({"error": f"query failed: {e}"}), 500

    if not row:
        return jsonify({"error": "signal not found"}), 404

    baseline_id = row.get("baseline_id")
    baseline = fetch_baseline_record(baseline_id) if baseline_id else None

    f_low = row.get("f_low_hz")
    f_high = row.get("f_high_hz")
    bw = (f_high - f_low) if f_low is not None and f_high is not None else None
    user_bw = row.get("user_bw_hz")

    display_bw = compute_display_bandwidth_hz(
        baseline_row=baseline,
        f_low_hz=float(f_low) if f_low else None,
        f_high_hz=float(f_high) if f_high else None,
        bandwidth_hz=bw,
    ) if baseline else bw

    total_hits = row.get("total_hits") or 0
    total_windows = row.get("total_windows") or 0
    hit_ratio = total_hits / total_windows if total_windows > 0 else 0.0

    f_center_hz = row.get("f_center_hz")
    return jsonify({
        "id": det_id,
        "signal_id": format_signal_id(det_id),
        "baseline_id": baseline_id,
        "baseline_name": baseline.get("name") if baseline else None,
        "f_center_hz": f_center_hz,
        "f_center_mhz": float(f_center_hz) / 1e6 if f_center_hz is not None else None,
        "f_low_hz": f_low,
        "f_high_hz": f_high,
        "bandwidth_hz": bw,
        "bandwidth_hz_display": user_bw if user_bw else display_bw,
        "user_bw_hz": user_bw,
        "first_seen_utc": row.get("first_seen_utc"),
        "last_seen_utc": row.get("last_seen_utc"),
        "total_hits": total_hits,
        "total_windows": total_windows,
        "hit_ratio": hit_ratio,
        "confidence": row.get("confidence"),
        "label": row.get("label"),
        "classification": row.get("classification") or "unknown",
        "notes": row.get("notes"),
        "selected": bool(row.get("selected")),
    })


# ---------------------------------------------------------------------------
# PATCH /api/signals/<id> - Update signal classification/metadata
# ---------------------------------------------------------------------------


@bp.patch("/<signal_id>")
def update_signal(signal_id: str):
    """
    Update signal classification, label, notes, user_bw_hz, or selected state.

    Args:
        signal_id: Signal ID (e.g., "SIG-0042" or "42").

    Request body (JSON):
        label: Optional string label
        classification: One of friendly/ambient/hostile/unknown
        user_bw_hz: User-corrected bandwidth in Hz (display only)
        notes: Freeform notes text
        selected: Boolean selection state

    Returns:
        Updated signal object.
    """
    con = get_con_optional()
    if con is None:
        return jsonify({"error": "database not available"}), 503

    det_id = parse_signal_id(signal_id)
    if det_id is None:
        return jsonify({"error": "invalid signal_id"}), 400

    # Verify signal exists
    try:
        existing = q1(
            con,
            "SELECT id, baseline_id FROM baseline_detections WHERE id = ?",
            (det_id,),
        )
    except sqlite3.OperationalError as e:
        return jsonify({"error": f"query failed: {e}"}), 500

    if not existing:
        return jsonify({"error": "signal not found"}), 404

    data = request.get_json(silent=True) or {}

    updates = []
    params: List[Any] = []

    # Label
    if "label" in data:
        label_val = data.get("label")
        if label_val is not None:
            label_val = str(label_val).strip()[:64] or None  # Max 64 chars
        updates.append("label = ?")
        params.append(label_val)

    # Classification
    if "classification" in data:
        classification_val = str(data.get("classification") or "unknown").strip().lower()
        if classification_val not in VALID_CLASSIFICATIONS:
            classification_val = "unknown"
        updates.append("classification = ?")
        params.append(classification_val)

    # User-corrected bandwidth
    if "user_bw_hz" in data:
        user_bw_val = data.get("user_bw_hz")
        if user_bw_val is not None:
            try:
                user_bw_val = int(user_bw_val)
                if user_bw_val <= 0:
                    user_bw_val = None
            except (ValueError, TypeError):
                user_bw_val = None
        updates.append("user_bw_hz = ?")
        params.append(user_bw_val)

    # Notes
    if "notes" in data:
        notes_val = data.get("notes")
        if notes_val is not None:
            notes_val = str(notes_val).strip()[:1024] or None  # Max 1024 chars
        updates.append("notes = ?")
        params.append(notes_val)

    # Selected
    if "selected" in data:
        selected_val = 1 if data.get("selected") else 0
        updates.append("selected = ?")
        params.append(selected_val)

    if not updates:
        return jsonify({"error": "no valid fields to update"}), 400

    params.append(det_id)

    # Need writable connection for update
    db_path = current_app.config.get('SDRWATCH_DB_PATH') or current_app.config.get('DB_PATH')
    if not db_path:
        return jsonify({"error": "database path not configured"}), 500

    try:
        write_con = sqlite3.connect(db_path)
        write_con.row_factory = sqlite3.Row
        write_con.execute(
            f"UPDATE baseline_detections SET {', '.join(updates)} WHERE id = ?",
            tuple(params),
        )
        write_con.commit()
        write_con.close()
    except sqlite3.Error as e:
        return jsonify({"error": f"update failed: {e}"}), 500

    # Return updated signal
    return get_signal(signal_id)


# ---------------------------------------------------------------------------
# POST /api/signals/<id>/toggle-selected - Quick toggle selection
# ---------------------------------------------------------------------------


@bp.post("/<signal_id>/toggle-selected")
def toggle_selected(signal_id: str):
    """
    Toggle the selected state of a signal.

    Args:
        signal_id: Signal ID (e.g., "SIG-0042" or "42").

    Returns:
        Updated signal object with new selected state.
    """
    con = get_con_optional()
    if con is None:
        return jsonify({"error": "database not available"}), 503

    det_id = parse_signal_id(signal_id)
    if det_id is None:
        return jsonify({"error": "invalid signal_id"}), 400

    try:
        existing = q1(
            con,
            "SELECT id, selected FROM baseline_detections WHERE id = ?",
            (det_id,),
        )
    except sqlite3.OperationalError as e:
        return jsonify({"error": f"query failed: {e}"}), 500

    if not existing:
        return jsonify({"error": "signal not found"}), 404

    new_selected = 0 if existing.get("selected") else 1

    db_path = current_app.config.get('SDRWATCH_DB_PATH') or current_app.config.get('DB_PATH')
    if not db_path:
        return jsonify({"error": "database path not configured"}), 500

    try:
        write_con = sqlite3.connect(db_path)
        write_con.execute(
            "UPDATE baseline_detections SET selected = ? WHERE id = ?",
            (new_selected, det_id),
        )
        write_con.commit()
        write_con.close()
    except sqlite3.Error as e:
        return jsonify({"error": f"update failed: {e}"}), 500

    return get_signal(signal_id)


# ---------------------------------------------------------------------------
# GET /api/signals/selected - Get all selected signals across baselines
# ---------------------------------------------------------------------------


@bp.get("/selected")
def get_selected_signals():
    """
    Get all selected (highlighted) signals across all baselines.

    Returns:
        JSON array of selected signal objects.
    """
    con = get_con_optional()
    if con is None:
        return jsonify({"error": "database not available"}), 503

    try:
        rows = qa(
            con,
            """
            SELECT bd.id, bd.baseline_id, bd.f_center_hz, bd.f_low_hz, bd.f_high_hz,
                   bd.first_seen_utc, bd.last_seen_utc, bd.total_hits, bd.total_windows,
                   bd.confidence, bd.label, bd.classification, bd.user_bw_hz, bd.notes,
                   bd.selected, b.name AS baseline_name
            FROM baseline_detections bd
            LEFT JOIN baselines b ON bd.baseline_id = b.id
            WHERE bd.selected = 1
            ORDER BY bd.last_seen_utc DESC
            LIMIT 100
            """,
        )
    except sqlite3.OperationalError as e:
        return jsonify({"error": f"query failed: {e}"}), 500

    signals = []
    for row in rows:
        det_id = row.get("id")
        f_low = row.get("f_low_hz")
        f_high = row.get("f_high_hz")
        bw = (f_high - f_low) if f_low is not None and f_high is not None else None

        signals.append({
            "id": det_id,
            "signal_id": format_signal_id(det_id) if det_id else None,
            "baseline_id": row.get("baseline_id"),
            "baseline_name": row.get("baseline_name"),
            "f_center_hz": row.get("f_center_hz"),
            "bandwidth_hz": bw,
            "first_seen_utc": row.get("first_seen_utc"),
            "last_seen_utc": row.get("last_seen_utc"),
            "total_hits": row.get("total_hits"),
            "total_windows": row.get("total_windows"),
            "confidence": row.get("confidence"),
            "label": row.get("label"),
            "classification": row.get("classification") or "unknown",
            "notes": row.get("notes"),
            "selected": True,
        })

    return jsonify(signals)
