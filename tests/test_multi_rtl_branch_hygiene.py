"""Spec Kit branch/artifact hygiene checks for the multi-RTL feature."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FEATURE_DIR = "specs/008-multi-rtl-guard-rover"


def test_feature_pointer_uses_existing_008_directory() -> None:
    payload = json.loads((ROOT / ".specify" / "feature.json").read_text(encoding="utf-8"))

    assert payload["feature_directory"] == FEATURE_DIR
    assert (ROOT / FEATURE_DIR / "spec.md").exists()
    assert (ROOT / FEATURE_DIR / "tasks.md").exists()


def test_agent_context_points_to_008_plan() -> None:
    text = (ROOT / "AGENTS.md").read_text(encoding="utf-8")

    assert f"{FEATURE_DIR}/plan.md" in text
    assert "web GUI" in text
    assert "controller job lifecycle" in text
