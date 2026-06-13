"""Characterization diagnostics summary tests."""

from __future__ import annotations

from sdrwatch.util.detection_diagnostics import (
    build_characterization_record,
    summarize_characterization_records,
)

from tests.helpers_fm_characterization import make_characterization_evidence
from tests.helpers_fm_detection import fm_args, make_engine


def test_characterization_summary_keeps_context_and_measured_fields_separate() -> None:
    records = [
        build_characterization_record(evidence=make_characterization_evidence()),
        build_characterization_record(evidence=make_characterization_evidence(source_pass="revisit", detection_id=2)),
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
    assert sample["display_span"]["bandwidth_hz"] == 200_000.0
    assert sample["context"]["bandplan_service"] == "FM Broadcast"
    assert sample["classification"]["candidate"] == "unknown"


def test_make_engine_can_seed_isolated_characterization_jsonl_paths(tmp_path) -> None:
    args = fm_args()

    engine, _store, _ctx, _logger = make_engine(tmp_path, args=args, isolated_jsonl=True)

    assert engine.args.jsonl.endswith("signals.jsonl")
    assert engine.args.diagnostic_jsonl.endswith("diagnostic.jsonl")
