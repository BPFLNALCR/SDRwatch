"""Foundational characterization record tests."""

from __future__ import annotations

from sdrwatch.util.detection_diagnostics import build_characterization_record

from tests.helpers_fm_characterization import (
    assert_characterization_record_shape,
    make_characterization_evidence,
)
from tests.helpers_fm_detection import ListLogger, ingest_segments, make_engine, make_segment


def test_build_characterization_record_uses_explicit_separate_span_fields() -> None:
    record = build_characterization_record(evidence=make_characterization_evidence())

    assert_characterization_record_shape(record)
    assert record["bandplan_service"] == "FM Broadcast"
    assert record["profile_context"] == "fm_broadcast"
    assert record["classification_candidate"] == "unknown"
    assert record["raw_bandwidth_hz"] < record["display_bandwidth_hz"]


def test_engine_emits_characterization_record_without_changing_match_span_behavior(tmp_path) -> None:
    logger = ListLogger()
    engine, store, ctx, logger = make_engine(tmp_path, logger=logger)

    engine.ingest(0, [make_segment(100_100_000, width_hz=2_000)])

    [record] = logger.events("characterization_record")
    [detection] = store.load_baseline_detections(ctx.id)
    assert_characterization_record_shape(record)
    assert record["match_low_hz"] == detection.f_low_hz
    assert record["match_high_hz"] == detection.f_high_hz
    assert record["match_center_hz"] == detection.f_center_hz
    assert record["display_bandwidth_hz"] > record["measured_bandwidth_hz"]


def test_wide_spiky_fm_like_signal_keeps_bounded_cards_and_separate_measured_width(tmp_path) -> None:
    logger = ListLogger()
    engine, store, ctx, logger = make_engine(tmp_path, logger=logger)
    centers = [
        100_060_000,
        100_075_000,
        100_090_000,
        100_105_000,
        100_120_000,
        100_095_000,
        100_080_000,
        100_110_000,
    ]

    ingest_segments(engine, centers, width_hz=2_000)

    detections = store.load_baseline_detections(ctx.id)
    records = logger.events("characterization_record")
    assert len(detections) <= 2
    assert records
    assert all(record["measured_bandwidth_hz"] < record["display_bandwidth_hz"] for record in records)
    assert all(record["match_bandwidth_hz"] <= record["display_bandwidth_hz"] for record in records)
