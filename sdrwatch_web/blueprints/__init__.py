"""
Blueprints package for SDRwatch Web.

This package contains Flask blueprints that organize routes by function:
- api_debug: Debug and observability endpoints (/api/debug/*, /debug)
- api_jobs: Job management endpoints (/api/jobs/*, /api/scans/*, /api/logs)
- api_baselines: Baseline CRUD and analytics (/api/baselines, /api/baseline/<id>/*)
- views: HTML template routes (/, /control, /live, /changes, /spur-map)
- ctl: Controller proxy routes (/ctl/devices)
"""
from __future__ import annotations
