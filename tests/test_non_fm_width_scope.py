"""Regression tests for non-FM width behavior."""

from __future__ import annotations

from sdrwatch.detection.engine import DetectionEngine
from sdrwatch.io.bandplan import Bandplan

from tests.helpers_fm_detection import FM_BIN_HZ, ListLogger, make_segment, make_store, narrow_args


def test_narrow_non_fm_signal_remains_narrow_without_fm_profile(tmp_path) -> None:
    store, ctx = make_store(tmp_path, start_hz=450_000_000, stop_hz=451_000_000)
    logger = ListLogger()
    args = narrow_args()
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

    engine.ingest(0, [make_segment(450_500_000, width_hz=2_000)])

    [det] = store.load_baseline_detections(ctx.id)
    assert det.f_high_hz - det.f_low_hz < 5_000
    width_events = logger.events("width_decision")
    assert width_events
    assert all(record["was_floored"] is False for record in width_events)
