"""FM-like persistence stability tests."""

from __future__ import annotations

from tests.helpers_fm_detection import ListLogger, fm_args, ingest_segments, make_engine, make_segment


def test_wide_spiky_fm_like_signal_updates_bounded_cards(tmp_path) -> None:
    engine, store, ctx, logger = make_engine(tmp_path)
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
    assert len(detections) <= 2
    assert sum(det.total_hits for det in detections) >= 5
    assert all(70_000 <= det.f_high_hz - det.f_low_hz <= 270_000 for det in detections)
    assert {record["action"] for record in logger.events("persistence_decision")} >= {"insert", "update"}


def test_multiple_separated_fm_like_signals_remain_separate(tmp_path) -> None:
    engine, store, ctx, _logger = make_engine(tmp_path)

    ingest_segments(engine, [100_100_000, 100_110_000, 100_300_000, 100_310_000], width_hz=2_000)

    centers = sorted(det.f_center_hz for det in store.load_baseline_detections(ctx.id))
    assert len(centers) == 2
    assert centers[1] - centers[0] > 120_000


def test_fm_broadcast_profile_keeps_close_but_separable_cards_bounded(tmp_path) -> None:
    engine, store, ctx, logger = make_engine(tmp_path)

    ingest_segments(
        engine,
        [
            100_100_000,
            100_112_000,
            100_240_000,
            100_252_000,
            100_102_000,
            100_242_000,
        ],
        width_hz=2_000,
    )

    detections = sorted(store.load_baseline_detections(ctx.id), key=lambda det: det.f_center_hz)
    assert len(detections) == 2
    assert detections[1].f_center_hz - detections[0].f_center_hz > 80_000
    assert all(70_000 <= det.f_high_hz - det.f_low_hz <= 270_000 for det in detections)
    assert {record["action"] for record in logger.events("persistence_decision")} >= {"insert"}


def test_repeated_nearby_fm_fragments_clear_missing_and_update_existing_row(tmp_path) -> None:
    logger = ListLogger()
    engine, store, ctx, logger = make_engine(tmp_path, logger=logger)
    existing_id = store.insert_baseline_detection(
        ctx.id,
        100_060_000,
        100_140_000,
        100_100_000,
        "2026-06-12T00:00:00Z",
        "2026-06-12T00:00:01Z",
        1,
        1,
        0.5,
        missing_since_utc="2026-06-12T00:00:03Z",
    )
    store.con.commit()
    engine.persistence._persisted = store.load_baseline_detections(ctx.id)

    engine.ingest(0, [make_segment(100_145_000, width_hz=2_000)])

    detections = store.load_baseline_detections(ctx.id)
    assert [det.id for det in detections] == [existing_id]
    assert detections[0].missing_since_utc is None
    assert detections[0].total_hits == 2
    actions = [record["action"] for record in logger.events("persistence_decision")]
    assert "missing_cleared" in actions
    assert "update" in actions


def test_width_cap_prevents_fm_row_from_ballooning(tmp_path) -> None:
    args = fm_args(match_bandwidth_pad_hz=200_000.0, min_match_bandwidth_hz=80_000.0)
    engine, store, ctx, _logger = make_engine(tmp_path, args=args)

    engine.ingest(0, [make_segment(100_100_000, width_hz=20_000)])

    [det] = store.load_baseline_detections(ctx.id)
    assert det.f_high_hz - det.f_low_hz == 270_000
