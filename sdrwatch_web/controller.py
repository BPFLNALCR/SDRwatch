"""
Controller HTTP client for SDRwatch Web.

Provides the ControllerClient class that proxies requests to sdrwatch-control,
plus helper functions for common controller interactions.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from urllib import error as urlerr
from urllib import parse as urlparse
from urllib import request as urlreq

from flask import current_app

from sdrwatch_web.config import CONTROL_TOKEN, CONTROL_URL


class ControllerClient:
    """
    HTTP client for communicating with sdrwatch-control.

    Wraps the controller's REST API with typed methods for device discovery,
    job management, profiles, and baselines.
    """

    def __init__(self, base_url: str, token: str = "") -> None:
        """
        Initialize the controller client.

        Args:
            base_url: Base URL of the controller (e.g., http://127.0.0.1:8765).
            token: Optional bearer token for authentication.
        """
        self.base = base_url.rstrip('/')
        self.token = token

    def _req(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None,
        want_text: bool = False,
    ) -> Any:
        """
        Make an HTTP request to the controller.

        Args:
            method: HTTP method (GET, POST, DELETE, etc.).
            path: URL path (e.g., /jobs).
            params: Optional query parameters.
            body: Optional JSON body for POST/PUT requests.
            want_text: If True, return raw text instead of parsing JSON.

        Returns:
            Parsed JSON response or raw text.

        Raises:
            RuntimeError: On HTTP errors or connection failures.
        """
        url = self.base + path
        if params:
            q = urlparse.urlencode(params)
            url += ("?" + q)

        data = None
        headers = {"Content-Type": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        if body is not None:
            data = json.dumps(body).encode('utf-8')

        req = urlreq.Request(url, data, headers=headers, method=method.upper())
        try:
            with urlreq.urlopen(req, timeout=10) as resp:
                ct = resp.headers.get('Content-Type', '')
                raw = resp.read()
                if want_text or not ct.startswith('application/json'):
                    return raw.decode('utf-8', errors='replace')
                return json.loads(raw.decode('utf-8'))
        except urlerr.HTTPError as e:
            raise RuntimeError(f"controller HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}")
        except Exception as e:
            raise RuntimeError(str(e))

    # -----------------------------------------------------------------------
    # Device discovery
    # -----------------------------------------------------------------------

    def devices(self) -> Any:
        """List available SDR devices."""
        return self._req('GET', '/devices')

    # -----------------------------------------------------------------------
    # Job management
    # -----------------------------------------------------------------------

    def list_jobs(self) -> List[Dict[str, Any]]:
        """List all jobs (running and completed)."""
        return self._req('GET', '/jobs')

    def start_job(
        self,
        device_key: str,
        label: str,
        baseline_id: int,
        params: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Start a new scan job.

        Args:
            device_key: SDR device identifier (e.g., "rtl:0").
            label: Human-readable job label.
            baseline_id: ID of the baseline to scan against.
            params: Scan parameters (mirrors CLI flags).

        Returns:
            Job creation response from controller.
        """
        body = {
            "device_key": device_key,
            "label": label,
            "baseline_id": int(baseline_id),
            "params": params,
        }
        return self._req('POST', '/jobs', body=body)

    def job_detail(self, job_id: str) -> Dict[str, Any]:
        """Get details for a specific job."""
        return self._req('GET', f'/jobs/{job_id}')

    def stop_job(self, job_id: str) -> Dict[str, Any]:
        """Stop a running job."""
        return self._req('DELETE', f'/jobs/{job_id}')

    def job_logs(self, job_id: str, tail: Optional[int] = None) -> str:
        """
        Get log output for a job.

        Args:
            job_id: Job identifier.
            tail: Optional number of lines from end.

        Returns:
            Log text content.
        """
        params = {"tail": int(tail)} if tail else None
        return self._req('GET', f'/jobs/{job_id}/logs', params=params, want_text=True)

    # -----------------------------------------------------------------------
    # Profiles & baselines
    # -----------------------------------------------------------------------

    def profiles(self) -> Any:
        """Get available scan profiles."""
        return self._req('GET', '/profiles')

    def baselines(self) -> Any:
        """Get list of baselines from controller."""
        return self._req('GET', '/baselines')


def get_controller() -> ControllerClient:
    """
    Get the ControllerClient instance from the current Flask app.

    Returns:
        ControllerClient instance stored on the app.
    """
    return current_app.extensions['sdrwatch_controller']


def init_controller(app) -> None:
    """
    Initialize the controller client on the Flask app.

    Args:
        app: Flask application instance.
    """
    if not hasattr(app, 'extensions'):
        app.extensions = {}
    app.extensions['sdrwatch_controller'] = ControllerClient(CONTROL_URL, CONTROL_TOKEN)


def controller_active_job() -> Optional[Dict[str, Any]]:
    """
    Get the currently running job, if any.

    Returns:
        Job dict if a job is running, None otherwise.
    """
    ctl = get_controller()
    try:
        jobs = ctl.list_jobs()
    except Exception:
        return None

    running = [j for j in jobs if str(j.get("status", "")).lower() == "running"]
    running.sort(key=lambda j: float(j.get("created_ts") or 0.0), reverse=True)
    return running[0] if running else None


def controller_profiles() -> List[Dict[str, Any]]:
    """
    Fetch scan profiles from the controller.

    Returns:
        List of profile dicts, or empty list on error.
    """
    ctl = get_controller()
    try:
        data = ctl.profiles()
    except Exception as exc:
        current_app.logger.warning("controller /profiles fetch failed: %s", exc)
        return []

    if isinstance(data, dict):
        profiles = data.get("profiles")
        if isinstance(profiles, list):
            return profiles
        current_app.logger.warning("controller /profiles payload missing list: %s", data)
        return []

    current_app.logger.warning("controller /profiles unexpected payload type: %r", type(data))
    return []
