"""Persistence and width decision diagnostic coverage."""

from __future__ import annotations

from tests.helpers_fm_detection import ListLogger, ingest_segments, make_engine, make_segment


def test_persistence_logs_insert_update_no_match_and_width_decisions(tmp_path) -> None:
    logger = ListLogger()
    engine, store, ctx, logger = make_engine(tmp_path, logger=logger)

    ingest_segments(engine, [100_100_000, 100_130_000, 100_300_000], width_hz=2_000)

    detections = store.load_baseline_detections(ctx.id)
    assert len(detections) == 2
    actions = [record["action"] for record in logger.events("persistence_decision")]
    assert "insert" in actions
    assert "update" in actions
    assert "no_match" in actions

    width_events = logger.events("width_decision")
    assert {record["stage"] for record in width_events} >= {"shape_match", "shape_display"}
    assert any(record["was_floored"] for record in width_events)
    assert all(record["output_width_hz"] <= 270_000.0 for record in width_events)


def test_persistence_logs_missing_decision(tmp_path) -> None:
    logger = ListLogger()
    engine, store, ctx, logger = make_engine(tmp_path, logger=logger)
    engine.ingest(0, [make_segment(100_100_000, width_hz=2_000)])
    logger.records.clear()
    engine.persistence._seen_persistent.clear()
    engine.persistence._revisit_tags = []

    engine.finalize_coarse_pass()

    actions = [record["action"] for record in logger.events("persistence_decision")]
    assert "missing" in actions
