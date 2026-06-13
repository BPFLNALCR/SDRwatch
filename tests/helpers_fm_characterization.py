"""Shared no-hardware characterization fixtures and assertions."""

from __future__ import annotations

from typing import Any, Mapping

from sdrwatch.detection.types import CharacterizationEvidence, CharacterizationSpan


def make_characterization_evidence(
    *,
    source_pass: str = "coarse",
    detection_id: int | None = 1,
    baseline_id: int = 1,
    bandplan_service: str = "FM Broadcast",
    profile_context: str = "fm_broadcast",
) -> CharacterizationEvidence:
    return CharacterizationEvidence(
        detection_id=detection_id,
        baseline_id=baseline_id,
        source_pass=source_pass,
        raw_segment=CharacterizationSpan(
            low_hz=100_099_000,
            high_hz=100_101_000,
            center_hz=100_100_000,
            bandwidth_hz=2_000.0,
        ),
        measured_span=CharacterizationSpan(
            low_hz=100_060_000,
            high_hz=100_140_000,
            center_hz=100_100_000,
            bandwidth_hz=80_000.0,
        ),
        match_span=CharacterizationSpan(
            low_hz=100_060_000,
            high_hz=100_140_000,
            center_hz=100_100_000,
            bandwidth_hz=80_000.0,
        ),
        display_span=CharacterizationSpan(
            low_hz=100_000_000,
            high_hz=100_200_000,
            center_hz=100_100_000,
            bandwidth_hz=200_000.0,
        ),
        peak_db=-35.0,
        noise_db=-80.0,
        snr_db=24.0,
        measured_bandwidth_confidence=0.7,
        characterization_confidence=0.8,
        characterization_method="coarse_cluster_span",
        center_stability_hz=0.0,
        bandwidth_stability_hz=0.0,
        revisit_measurement_count=0,
        coarse_measurement_count=3,
        classification_candidate="unknown",
        classification_evidence=[],
        evidence_sources=["coarse_cluster"],
        bandplan_service=bandplan_service,
        bandplan_region="Global",
        bandplan_notes="88-108 MHz Radio",
        profile_context=profile_context,
        context_only=False,
    )


def assert_characterization_record_shape(record: Mapping[str, Any]) -> None:
    required_fields = {
        "event",
        "source_pass",
        "baseline_id",
        "raw_low_hz",
        "raw_high_hz",
        "raw_center_hz",
        "raw_bandwidth_hz",
        "measured_low_hz",
        "measured_high_hz",
        "measured_center_hz",
        "measured_bandwidth_hz",
        "measured_bandwidth_confidence",
        "match_low_hz",
        "match_high_hz",
        "match_center_hz",
        "match_bandwidth_hz",
        "display_low_hz",
        "display_high_hz",
        "display_center_hz",
        "display_bandwidth_hz",
        "characterization_confidence",
        "characterization_method",
        "classification_candidate",
        "classification_evidence",
        "evidence_sources",
        "bandplan_service",
        "profile_context",
        "raw_segment",
        "measured_span",
        "match_span",
        "display_span",
    }
    missing = required_fields.difference(record.keys())
    assert not missing
    assert record["event"] == "characterization_record"
    assert record["raw_low_hz"] <= record["raw_center_hz"] <= record["raw_high_hz"]
    assert record["measured_low_hz"] <= record["measured_center_hz"] <= record["measured_high_hz"]
    assert record["match_low_hz"] <= record["match_center_hz"] <= record["match_high_hz"]
    assert record["display_low_hz"] <= record["display_center_hz"] <= record["display_high_hz"]
    assert record["raw_bandwidth_hz"] <= record["measured_bandwidth_hz"] <= record["display_bandwidth_hz"]

