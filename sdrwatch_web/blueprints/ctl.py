"""
Controller proxy blueprint for SDRwatch Web.

Provides endpoints that proxy requests directly to the controller.
"""
from __future__ import annotations

from flask import Blueprint, jsonify, request

from sdrwatch_web.controller import get_controller

bp = Blueprint("ctl", __name__)


@bp.get("/ctl/devices")
def ctl_devices():
    """Proxy device list from controller."""
    ctl = get_controller()
    try:
        devs = ctl.devices()
        return jsonify(devs)
    except Exception as e:
        # Surface the reason to the frontend as a JSON object (not an empty list)
        msg = str(e)
        hint = "unauthorized" if "401" in msg or "unauthorized" in msg.lower() else "unreachable"
        return jsonify({"error": f"controller_{hint}", "detail": msg})


@bp.get("/ctl/discovery-debug")
def ctl_discovery_debug():
    """Proxy controller discovery debug events for troubleshooting enumeration failures."""
    ctl = get_controller()
    limit = request.args.get("limit", default=50, type=int)
    try:
        payload = ctl.discovery_debug(limit=limit)
        return jsonify(payload)
    except Exception as e:
        msg = str(e)
        hint = "unauthorized" if "401" in msg or "unauthorized" in msg.lower() else "unreachable"
        return jsonify({"error": f"controller_{hint}", "detail": msg})
