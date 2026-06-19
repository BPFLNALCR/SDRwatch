"""Focused no-hardware tests for profile-governed signal span policy."""

from __future__ import annotations

from types import SimpleNamespace

from sdrwatch.detection.engine import DetectionEngine
from sdrwatch.detection.span_policy import resolve_signal_span_policy
from sdrwatch.io.bandplan import Bandplan

from tests.helpers_fm_detection import FM_BIN_HZ, ListLogger, fm_args, make_segment, make_store, narrow_args


def broad_continuous_policy_args(**overrides):
    return fm_args(**overrides)


def narrowband_policy_args(**overrides):
    return narrow_args(
        min_match_bandwidth_hz=12_500.0,
        min_identity_bandwidth_hz=12_500.0,
        min_persist_bandwidth_hz=12_500.0,
        min_display_bandwidth_hz=25_000.0,
        **overrides,
    )


def unknown_discovery_policy_args(**overrides):
    return narrow_args(**overrides)


def guard_event_policy_args(**overrides):
    return narrow_args(
        persistence_min_hits=1,
        persistence_min_windows=1,
        persistence_min_sweep_loops=1,
        min_match_bandwidth_hz=6_250.0,
        min_identity_bandwidth_hz=6_250.0,
        min_persist_bandwidth_hz=6_250.0,
        **overrides,
    )


def test_policy_defaults_identity_and_persist_floor_from_min_match_bandwidth() -> None:
    policy = resolve_signal_span_policy(
        SimpleNamespace(
            profile="broad_fixture",
            min_match_bandwidth_hz=80_000.0,
            max_persist_width_hz=270_000.0,
        )
    )

    assert policy.min_identity_bandwidth_hz == 80_000.0
    assert policy.min_persist_bandwidth_hz == 80_000.0
    assert policy.max_persist_bandwidth_hz == 270_000.0


def test_unknown_discovery_without_floors_preserves_unset_policy_widths() -> None:
    policy = resolve_signal_span_policy(unknown_discovery_policy_args())

    assert policy.min_identity_bandwidth_hz is None
    assert policy.min_persist_bandwidth_hz is None
    assert policy.min_display_bandwidth_hz is None


def test_narrowband_and_guard_event_fixtures_keep_small_policy_floors() -> None:
    narrow_policy = resolve_signal_span_policy(narrowband_policy_args())
    guard_policy = resolve_signal_span_policy(guard_event_policy_args())

    assert narrow_policy.min_identity_bandwidth_hz == 12_500.0
    assert narrow_policy.min_display_bandwidth_hz == 25_000.0
    assert guard_policy.min_persist_bandwidth_hz == 6_250.0


def test_negative_policy_values_resolve_to_unset_and_are_diagnosable() -> None:
    policy = resolve_signal_span_policy(
        SimpleNamespace(
            min_match_bandwidth_hz=-1.0,
            min_identity_bandwidth_hz=-2.0,
            min_persist_bandwidth_hz=-3.0,
            max_persist_bandwidth_hz=-4.0,
        )
    )

    assert policy.min_identity_bandwidth_hz is None
    assert policy.min_persist_bandwidth_hz is None
    assert "min_identity_bandwidth_hz" in policy.invalid_fields
    assert "min_persist_bandwidth_hz" in policy.invalid_fields


def test_scan_edge_clipping_reports_policy_floor_exception(tmp_path) -> None:
    store, ctx = make_store(tmp_path, start_hz=100_000_000, stop_hz=100_050_000)
    logger = ListLogger()
    args = broad_continuous_policy_args()
    engine = DetectionEngine(
        store=store,
        bandplan=Bandplan(None),
        args=args,
        bin_hz=FM_BIN_HZ,
        baseline_ctx=ctx,
        min_hits=1,
        min_windows=1,
        logger=logger,
    )

    engine.ingest(0, [make_segment(100_005_000, width_hz=2_000)])

    [detection] = store.load_baseline_detections(ctx.id)
    assert detection.f_high_hz - detection.f_low_hz < 80_000
    decisions = [
        record
        for record in logger.events("width_decision")
        if record.get("stage") == "persisted_card_coarse_insert"
    ]
    assert decisions
    assert decisions[-1]["baseline_clipped"] is True
    assert decisions[-1]["clip_reason"] == "scan_edge"
