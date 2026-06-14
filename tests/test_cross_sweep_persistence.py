"""Cross-sweep persistence contract tests."""

from __future__ import annotations

from tests.helpers_fm_detection import ListLogger, fm_args, make_engine, make_segment


def _ingest_sweep(engine, sweep_loop_id: int, centers_hz: list[int]) -> None:
    for window_idx, center_hz in enumerate(centers_hz):
        engine.ingest(window_idx, [make_segment(center_hz, width_hz=2_000)], sweep_loop_id=sweep_loop_id)
    engine.flush()


def test_stable_once_per_loop_signal_promotes_after_required_sweep_observations(tmp_path) -> None:
    logger = ListLogger()
    args = fm_args(
        persistence_min_hits=2,
        persistence_min_windows=2,
        persistence_min_sweep_loops=3,
    )
    engine, store, ctx, logger = make_engine(tmp_path, args=args, logger=logger)

    _ingest_sweep(engine, 1, [100_100_000])
    assert store.load_baseline_detections(ctx.id) == []

    _ingest_sweep(engine, 2, [100_101_000])
    assert store.load_baseline_detections(ctx.id) == []

    _ingest_sweep(engine, 3, [100_099_000])

    detections = store.load_baseline_detections(ctx.id)
    assert len(detections) == 1
    [detection] = detections
    assert abs(detection.f_center_hz - 100_100_000) <= 2_000
    assert detection.total_hits >= 3
    promotions = [
        record for record in logger.events("persistence_decision") if record.get("action") == "cross_sweep_promote"
    ]
    assert len(promotions) == 1
    assert promotions[0]["observation_loop_count"] == 3
    assert promotions[0]["required_loop_count"] == 3


def test_same_loop_repeated_windows_do_not_satisfy_multi_loop_threshold(tmp_path) -> None:
    logger = ListLogger()
    args = fm_args(
        persistence_min_hits=1,
        persistence_min_windows=1,
        persistence_min_sweep_loops=2,
    )
    engine, store, ctx, logger = make_engine(tmp_path, args=args, logger=logger)

    engine.ingest(0, [make_segment(100_100_000, width_hz=2_000)], sweep_loop_id=1)
    engine.ingest(1, [make_segment(100_101_000, width_hz=2_000)], sweep_loop_id=1)
    engine.ingest(2, [make_segment(100_099_000, width_hz=2_000)], sweep_loop_id=1)
    engine.flush()

    assert store.load_baseline_detections(ctx.id) == []
    promotions = [
        record for record in logger.events("persistence_decision") if record.get("action") == "cross_sweep_promote"
    ]
    assert promotions == []
    observations = [record for record in logger.events("cross_sweep_observation")]
    assert observations
    assert max(record["observation_loop_count"] for record in observations) == 1


def test_cross_sweep_matching_keeps_nearby_compatible_candidates_separate(tmp_path) -> None:
    logger = ListLogger()
    args = fm_args(
        persistence_min_hits=2,
        persistence_min_windows=2,
        persistence_min_sweep_loops=2,
    )
    engine, store, ctx, _logger = make_engine(tmp_path, args=args, logger=logger)

    _ingest_sweep(engine, 1, [100_100_000, 100_300_000])
    assert store.load_baseline_detections(ctx.id) == []

    _ingest_sweep(engine, 2, [100_101_000, 100_299_000])

    detections = sorted(store.load_baseline_detections(ctx.id), key=lambda det: det.f_center_hz)
    assert len(detections) == 2
    assert detections[1].f_center_hz - detections[0].f_center_hz > 120_000
    assert all((det.f_high_hz - det.f_low_hz) <= 270_000 for det in detections)
