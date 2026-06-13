"""Characterization persistence regression tests."""

from __future__ import annotations

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
