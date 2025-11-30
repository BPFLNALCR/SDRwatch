from sdrwatch.baseline.persistence import BaselinePersistence, EdgeCounters


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
