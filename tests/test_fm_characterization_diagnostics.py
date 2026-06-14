"""Characterization diagnostics summary tests."""

from __future__ import annotations

from pathlib import Path

from sdrwatch.detection.types import RevisitTag
from sdrwatch.util.detection_diagnostics import (
    build_characterization_record,
    summarize_characterization_records,
)
from sdrwatch.util.scan_logger import ScanLogger

from tests.helpers_fm_characterization import make_characterization_evidence
from tests.helpers_fm_detection import ListLogger, fm_args, load_jsonl_records, make_engine, make_segment


def test_characterization_summary_keeps_context_and_measured_fields_separate() -> None:
    records = [
        build_characterization_record(evidence=make_characterization_evidence()),
        build_characterization_record(
            evidence=make_characterization_evidence(
                source_pass="revisit",
                detection_id=2,
            )
        ),
    ]

    summary = summarize_characterization_records(records, sample_limit=1)

    assert summary["record_count"] == 2
    assert summary["source_pass_counts"] == {"coarse": 1, "revisit": 1}
    assert summary["characterization_methods"] == {"coarse_cluster_span": 2}
    assert summary["classification_candidates"] == {"unknown": 2}
    assert summary["truncated"] is True
    assert len(summary["records"]) == 1
    [sample] = summary["records"]
    assert sample["raw_segment"]["bandwidth_hz"] == 2_000.0
    assert sample["measured_characterization"]["occupied_bandwidth_hz"] == 80_000.0
    assert sample["measured_characterization"]["stable_center_hz"] == 100_100_000
    assert sample["measured_characterization"]["center_delta_hz"] == 0
    assert sample["display_span"]["bandwidth_hz"] == 200_000.0
    assert sample["match_span"]["bandwidth_hz"] == 80_000.0
    assert sample["match_span"]["bandwidth_hz"] < sample["display_span"]["bandwidth_hz"]
    assert sample["raw_segment"]["bandwidth_hz"] < sample["measured_characterization"]["occupied_bandwidth_hz"]
    assert sample["context"]["bandplan_service"] == "FM Broadcast"
    assert sample["context"]["profile_context"] == "fm_broadcast"
    assert sample["classification"]["candidate"] == "unknown"
    assert sample["classification"]["context_only"] is False


def test_make_engine_can_seed_isolated_characterization_jsonl_paths(tmp_path) -> None:
    args = fm_args()

    engine, _store, _ctx, _logger = make_engine(tmp_path, args=args, isolated_jsonl=True)

    assert engine.args.jsonl.endswith("signals.jsonl")
    assert engine.args.diagnostic_jsonl.endswith("diagnostic.jsonl")


def test_scan_logger_mirror_carries_characterization_records_to_diagnostic_jsonl(tmp_path: Path) -> None:
    diagnostic_jsonl = tmp_path / "diagnostic.jsonl"
    logger = ScanLogger(tmp_path / "scan.log", mirror_paths=[diagnostic_jsonl])
    engine, _store, _ctx, _logger = make_engine(tmp_path, logger=logger)

    engine.ingest(0, [make_segment(100_100_000, width_hz=2_000)])

    records = load_jsonl_records(diagnostic_jsonl)
    characterization_records = [record for record in records if record.get("event") == "characterization_record"]
    assert len(characterization_records) == 1
    summary = summarize_characterization_records(characterization_records, sample_limit=5)
    assert summary["record_count"] == 1
    assert summary["records"][0]["measured_characterization"]["occupied_bandwidth_hz"] == 2_000.0
    assert summary["records"][0]["measured_characterization"]["stable_center_hz"] == 100_100_000


def test_characterization_summary_counts_revisit_records_and_nonzero_revisit_measurements(tmp_path: Path) -> None:
    logger = ListLogger()
    engine, store, ctx, logger = make_engine(tmp_path, logger=logger)

    engine.ingest(0, [make_segment(100_100_000, width_hz=2_000)])
    engine.flush()
    [detection] = store.load_baseline_detections(ctx.id)

    tag = RevisitTag(
        tag_id="rv-diag",
        detection_id=detection.id,
        f_center_hz=detection.f_center_hz,
        f_low_hz=detection.f_low_hz,
        f_high_hz=detection.f_high_hz,
        reason="new",
        created_utc="2026-06-13T00:00:00Z",
    )
    engine.apply_revisit_confirmation(tag, make_segment(100_118_000, width_hz=10_000, snr_db=26.0))

    summary = summarize_characterization_records(logger.events("characterization_record"), sample_limit=10)
    assert summary["source_pass_counts"] == {"coarse": 1, "revisit": 1}
    assert summary["characterization_methods"] == {"coarse_cluster_span": 1, "revisit_refinement": 1}
    revisit_samples = [record for record in summary["records"] if record["source_pass"] == "revisit"]
    assert revisit_samples
    assert revisit_samples[0]["measured_characterization"]["revisit_measurement_count"] == 1
    assert revisit_samples[0]["measured_characterization"]["center_stability_hz"] > 0.0
