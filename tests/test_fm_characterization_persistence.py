"""Characterization persistence regression tests."""

from __future__ import annotations

from sdrwatch.detection.types import RevisitTag

from tests.helpers_fm_detection import ListLogger, make_engine, make_segment


def test_tiny_fft_fragment_does_not_become_fake_measured_fm_bandwidth(tmp_path) -> None:
    logger = ListLogger()
    engine, _store, _ctx, logger = make_engine(tmp_path, logger=logger)

    engine.ingest(0, [make_segment(100_100_000, width_hz=2_000)])

    [record] = logger.events("characterization_record")
    assert record["raw_bandwidth_hz"] == 2_000.0
    assert record["measured_bandwidth_hz"] == 2_000.0
    assert record["match_bandwidth_hz"] == 80_000.0
    assert record["display_bandwidth_hz"] == 200_000.0


def test_revisit_characterization_records_show_revisit_contribution_and_stable_center(tmp_path) -> None:
    logger = ListLogger()
    engine, store, ctx, logger = make_engine(tmp_path, logger=logger)

    engine.ingest(0, [make_segment(100_100_000, width_hz=2_000)])
    engine.flush()
    [detection] = store.load_baseline_detections(ctx.id)

    tag = RevisitTag(
        tag_id="rv-test",
        detection_id=detection.id,
        f_center_hz=detection.f_center_hz,
        f_low_hz=detection.f_low_hz,
        f_high_hz=detection.f_high_hz,
        reason="new",
        created_utc="2026-06-13T00:00:00Z",
    )

    engine.apply_revisit_confirmation(tag, make_segment(100_128_000, width_hz=12_000, snr_db=28.0))

    records = logger.events("characterization_record")
    revisit_records = [record for record in records if record["source_pass"] == "revisit"]

    assert revisit_records
    revisit_record = revisit_records[-1]
    assert revisit_record["revisit_measurement_count"] == 1
    assert revisit_record["stable_center_hz"] == revisit_record["display_center_hz"]
    assert revisit_record["measured_center_hz"] != revisit_record["stable_center_hz"]
    assert revisit_record["center_delta_hz"] == (
        revisit_record["measured_center_hz"] - revisit_record["stable_center_hz"]
    )
    assert revisit_record["center_stability_hz"] > 0.0


def test_single_wide_revisit_observation_does_not_permanently_ratchet_card_width(tmp_path) -> None:
    logger = ListLogger()
    engine, store, ctx, logger = make_engine(tmp_path, logger=logger)

    engine.ingest(0, [make_segment(100_100_000, width_hz=2_000)])
    engine.flush()
    [detection] = store.load_baseline_detections(ctx.id)
    original_width_hz = detection.f_high_hz - detection.f_low_hz

    tag = RevisitTag(
        tag_id="rv-wide",
        detection_id=detection.id,
        f_center_hz=detection.f_center_hz,
        f_low_hz=detection.f_low_hz,
        f_high_hz=detection.f_high_hz,
        reason="new",
        created_utc="2026-06-13T00:00:00Z",
    )
    engine.apply_revisit_confirmation(tag, make_segment(100_100_000, width_hz=800_000, snr_db=30.0))

    [after_wide] = store.load_baseline_detections(ctx.id)
    assert after_wide.f_high_hz - after_wide.f_low_hz <= 270_000

    tag = RevisitTag(
        tag_id="rv-normal",
        detection_id=detection.id,
        f_center_hz=after_wide.f_center_hz,
        f_low_hz=after_wide.f_low_hz,
        f_high_hz=after_wide.f_high_hz,
        reason="new",
        created_utc="2026-06-13T00:00:01Z",
    )
    engine.apply_revisit_confirmation(tag, make_segment(100_102_000, width_hz=4_000, snr_db=28.0))

    [after_normal] = store.load_baseline_detections(ctx.id)
    assert after_normal.f_high_hz - after_normal.f_low_hz <= 270_000
    assert after_normal.f_high_hz - after_normal.f_low_hz <= max(270_000, original_width_hz)
    assert any(record.get("outlier_rejected") for record in logger.events("width_decision"))
