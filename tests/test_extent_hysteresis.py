from sdrwatch.baseline.persistence import BaselinePersistence, EdgeCounters
from sdrwatch.detection.types import RevisitTag

from tests.helpers_fm_detection import fm_args, make_engine, make_segment, make_store


def test_left_edge_expansion_requires_multiple_observations() -> None:
    counters = EdgeCounters()
    current = 100_000_000
    proposed = 99_900_000
    epsilon = 1_000
    expand_threshold = 2
    shrink_threshold = 2

    value = BaselinePersistence._update_edge_with_hysteresis(
        counters,
        current,
        proposed,
        epsilon,
        expand_threshold,
        shrink_threshold,
        direction="left",
    )
    assert value == current
    value = BaselinePersistence._update_edge_with_hysteresis(
        counters,
        current,
        proposed,
        epsilon,
        expand_threshold,
        shrink_threshold,
        direction="left",
    )
    assert value == proposed


def test_right_edge_shrink_waits_for_threshold() -> None:
    counters = EdgeCounters()
    current = 101_000_000
    proposed = 100_500_000
    epsilon = 500
    expand_threshold = 2
    shrink_threshold = 3

    for _ in range(shrink_threshold - 1):
        value = BaselinePersistence._update_edge_with_hysteresis(
            counters,
            current,
            proposed,
            epsilon,
            expand_threshold,
            shrink_threshold,
            direction="right",
        )
        assert value == current
    value = BaselinePersistence._update_edge_with_hysteresis(
        counters,
        current,
        proposed,
        epsilon,
        expand_threshold,
        shrink_threshold,
        direction="right",
    )
    assert value == proposed


def _make_persistence_stub(alpha: float = 0.5, outlier_ratio: float = 4.0):
    stub = object.__new__(BaselinePersistence)
    stub.min_detection_width_hz = 1.0
    stub.width_ema_alpha = alpha
    stub.width_outlier_ratio = outlier_ratio
    stub.max_detection_width_hz = 0.0
    return stub


def test_width_ema_moves_toward_measurement() -> None:
    stub = _make_persistence_stub(alpha=0.5)
    result = stub._blend_width_ema(1000.0, 1200.0)
    assert result == 1100.0


def test_width_ema_rejects_outliers() -> None:
    stub = _make_persistence_stub(alpha=0.5, outlier_ratio=3.0)
    result = stub._blend_width_ema(1000.0, 4000.0)
    assert result == 1000.0


def test_width_ema_applies_min_detection_width_floor() -> None:
    stub = _make_persistence_stub(alpha=0.5)
    stub.min_detection_width_hz = 80_000.0

    result = stub._blend_width_ema(10_000.0, 2_000.0)

    assert result == 80_000.0


def test_width_ema_respects_configured_max_detection_width() -> None:
    stub = _make_persistence_stub(alpha=1.0, outlier_ratio=10.0)
    stub.max_detection_width_hz = 270_000.0

    result = stub._blend_width_ema(200_000.0, 400_000.0)

    assert result == 270_000.0


def test_bounded_center_step_damps_one_off_fragment_jump() -> None:
    result = BaselinePersistence._bounded_center_step(
        current_hz=100_100_000,
        target_hz=100_160_000,
        max_step_hz=15_000.0,
        epsilon_hz=500,
    )

    assert result == 100_115_000


def _insert_detection(store, ctx, *, low: int, center: int, high: int) -> int:
    detection_id = store.insert_baseline_detection(
        ctx.id,
        low,
        high,
        center,
        "2026-06-12T00:00:00Z",
        "2026-06-12T00:00:01Z",
        1,
        1,
        0.5,
    )
    store.con.commit()
    return detection_id


def test_upsert_update_keeps_persisted_center_inside_hysteresis_held_edges(tmp_path) -> None:
    args = fm_args(center_match_hz=300_000.0, extent_expand_hysteresis=3, extent_shrink_hysteresis=3)
    engine, store, ctx, _logger = make_engine(tmp_path, args=args)
    detection_id = _insert_detection(
        store,
        ctx,
        low=100_000_000,
        center=100_040_000,
        high=100_080_000,
    )
    engine.persistence._persisted = store.load_baseline_detections(ctx.id)

    engine.ingest(0, [make_segment(100_200_000, width_hz=80_000)])

    [det] = [row for row in store.load_baseline_detections(ctx.id) if row.id == detection_id]
    assert det.f_low_hz <= det.f_center_hz <= det.f_high_hz


def test_revisit_confirmation_keeps_persisted_center_inside_hysteresis_held_edges(tmp_path) -> None:
    args = fm_args(extent_expand_hysteresis=3, extent_shrink_hysteresis=3, revisit_span_limit_hz=0.0)
    engine, store, ctx, _logger = make_engine(tmp_path, args=args)
    detection_id = _insert_detection(
        store,
        ctx,
        low=100_000_000,
        center=100_040_000,
        high=100_080_000,
    )
    engine.persistence._persisted = store.load_baseline_detections(ctx.id)
    tag = RevisitTag(
        tag_id="rv-test",
        detection_id=detection_id,
        f_center_hz=100_200_000,
        f_low_hz=100_160_000,
        f_high_hz=100_240_000,
        reason="new",
        created_utc="2026-06-12T00:00:02Z",
    )

    engine.apply_revisit_confirmation(tag, make_segment(100_200_000, width_hz=80_000))

    [det] = [row for row in store.load_baseline_detections(ctx.id) if row.id == detection_id]
    assert det.f_low_hz <= det.f_center_hz <= det.f_high_hz


def test_revisit_confirmation_clips_persisted_center_to_baseline_edge(tmp_path) -> None:
    store, ctx = make_store(tmp_path, start_hz=100_000_000, stop_hz=100_150_000)
    args = fm_args(extent_expand_hysteresis=3, extent_shrink_hysteresis=3, revisit_span_limit_hz=0.0)
    engine, _store, _ctx, _logger = make_engine(tmp_path, args=args)
    engine.store = store
    engine.baseline_ctx = ctx
    engine.persistence.store = store
    engine.persistence.baseline_ctx = ctx
    detection_id = _insert_detection(
        store,
        ctx,
        low=100_070_000,
        center=100_110_000,
        high=100_140_000,
    )
    engine.persistence._persisted = store.load_baseline_detections(ctx.id)
    tag = RevisitTag(
        tag_id="rv-edge",
        detection_id=detection_id,
        f_center_hz=100_250_000,
        f_low_hz=100_210_000,
        f_high_hz=100_290_000,
        reason="new",
        created_utc="2026-06-12T00:00:02Z",
    )

    engine.apply_revisit_confirmation(tag, make_segment(100_250_000, width_hz=80_000))

    [det] = [row for row in store.load_baseline_detections(ctx.id) if row.id == detection_id]
    assert ctx.freq_start_hz <= det.f_center_hz <= ctx.freq_stop_hz
    assert det.f_low_hz <= det.f_center_hz <= det.f_high_hz
