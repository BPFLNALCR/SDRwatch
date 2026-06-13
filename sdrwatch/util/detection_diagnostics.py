"""JSONL diagnostics for scan detection tuning."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from sdrwatch.detection.types import CharacterizationEvidence, CharacterizationSpan, Segment
from sdrwatch.util.time import utc_now_str


def segment_to_dict(seg: Segment) -> Dict[str, Any]:
    return {
        "f_low_hz": seg.f_low_hz,
        "f_high_hz": seg.f_high_hz,
        "f_center_hz": seg.f_center_hz,
        "bandwidth_hz": seg.bandwidth_hz,
        "peak_db": seg.peak_db,
        "noise_db": seg.noise_db,
        "snr_db": seg.snr_db,
    }


def build_characterization_record(*, evidence: CharacterizationEvidence) -> Dict[str, Any]:
    record = evidence.to_record()
    record.setdefault("time_utc", utc_now_str())
    return record


def _count_key(counter: Dict[str, int], key: Any) -> None:
    key_text = str(key or "unknown")
    counter[key_text] = counter.get(key_text, 0) + 1


def _span_from_record(record: Dict[str, Any], *, prefix: str, nested_key: str) -> CharacterizationSpan:
    nested = record.get(nested_key)
    if isinstance(nested, dict):
        low = nested.get("low_hz", nested.get("f_low_hz", record.get(f"{prefix}_low_hz", 0)))
        high = nested.get("high_hz", nested.get("f_high_hz", record.get(f"{prefix}_high_hz", 0)))
        center = nested.get("center_hz", nested.get("f_center_hz", record.get(f"{prefix}_center_hz", 0)))
        bandwidth = nested.get("bandwidth_hz", record.get(f"{prefix}_bandwidth_hz", 0.0))
    else:
        low = record.get(f"{prefix}_low_hz", 0)
        high = record.get(f"{prefix}_high_hz", 0)
        center = record.get(f"{prefix}_center_hz", 0)
        bandwidth = record.get(f"{prefix}_bandwidth_hz", 0.0)
    return CharacterizationSpan.from_bounds(
        low_hz=int(low),
        high_hz=int(high),
        center_hz=int(center),
        bandwidth_hz=float(bandwidth),
    )


def summarize_characterization_record(record: Dict[str, Any]) -> Dict[str, Any]:
    evidence = CharacterizationEvidence(
        detection_id=record.get("detection_id"),
        baseline_id=int(record.get("baseline_id", 0) or 0),
        source_pass=str(record.get("source_pass") or "unknown"),
        raw_segment=_span_from_record(record, prefix="raw", nested_key="raw_segment"),
        measured_span=_span_from_record(record, prefix="measured", nested_key="measured_span"),
        match_span=_span_from_record(record, prefix="match", nested_key="match_span"),
        display_span=_span_from_record(record, prefix="display", nested_key="display_span"),
        peak_db=float(record.get("peak_db", 0.0) or 0.0),
        noise_db=float(record.get("noise_db", 0.0) or 0.0),
        snr_db=float(record.get("snr_db", 0.0) or 0.0),
        measured_bandwidth_confidence=float(record.get("measured_bandwidth_confidence", 0.0) or 0.0),
        characterization_confidence=float(record.get("characterization_confidence", 0.0) or 0.0),
        characterization_method=str(record.get("characterization_method") or "unknown"),
        center_stability_hz=float(record.get("center_stability_hz", 0.0) or 0.0),
        bandwidth_stability_hz=float(record.get("bandwidth_stability_hz", 0.0) or 0.0),
        revisit_measurement_count=int(record.get("revisit_measurement_count", 0) or 0),
        coarse_measurement_count=int(record.get("coarse_measurement_count", 0) or 0),
        classification_candidate=str(record.get("classification_candidate") or "unknown"),
        classification_evidence=list(record.get("classification_evidence") or []),
        evidence_sources=list(record.get("evidence_sources") or []),
        bandplan_service=record.get("bandplan_service"),
        bandplan_region=record.get("bandplan_region"),
        bandplan_notes=record.get("bandplan_notes"),
        profile_context=record.get("profile_context"),
        context_only=bool(record.get("context_only", False)),
    )
    return evidence.to_summary()


def summarize_characterization_records(
    records: Iterable[Dict[str, Any]],
    *,
    sample_limit: int = 10,
) -> Dict[str, Any]:
    limit = max(1, int(sample_limit))
    summary: Dict[str, Any] = {
        "record_count": 0,
        "source_pass_counts": {},
        "characterization_methods": {},
        "classification_candidates": {},
        "truncated": False,
        "records": [],
    }
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("event") != "characterization_record":
            continue
        summary["record_count"] += 1
        _count_key(summary["source_pass_counts"], record.get("source_pass"))
        _count_key(summary["characterization_methods"], record.get("characterization_method"))
        _count_key(summary["classification_candidates"], record.get("classification_candidate"))
        if len(summary["records"]) < limit:
            summary["records"].append(summarize_characterization_record(record))
    summary["truncated"] = summary["record_count"] > len(summary["records"])
    return summary


def build_window_record(
    *,
    sweep_id: Optional[int],
    window_idx: int,
    center_hz: float,
    window_low_hz: float,
    window_high_hz: float,
    profile: Optional[str],
    baseline_id: Optional[int],
    tuning_params: Dict[str, Any],
    detection_diagnostics: Dict[str, Any],
    accepted_hits: int,
    spur_ignored: int,
    promoted: int,
    new_signals: int,
    anomalous_power: bool,
    emitted_segments: list[Segment],
) -> Dict[str, Any]:
    return {
        "time_utc": utc_now_str(),
        "event": "detection_window",
        "sweep_id": sweep_id,
        "window_idx": window_idx,
        "baseline_id": baseline_id,
        "center_hz": float(center_hz),
        "window_low_hz": float(window_low_hz),
        "window_high_hz": float(window_high_hz),
        "profile": profile,
        "tuning_params": tuning_params,
        "threshold_info": detection_diagnostics.get("threshold_info", {}),
        "bins_above_threshold": int(detection_diagnostics.get("bins_above_threshold", 0) or 0),
        "raw_candidate_segment_count": int(
            detection_diagnostics.get("raw_candidate_segment_count", 0) or 0
        ),
        "final_emitted_segment_count": int(
            detection_diagnostics.get("final_emitted_segment_count", len(emitted_segments)) or 0
        ),
        "accepted_hits": int(accepted_hits),
        "spur_ignored": int(spur_ignored),
        "promoted": int(promoted),
        "new_signals": int(new_signals),
        "anomalous_power": bool(anomalous_power),
        "segments": detection_diagnostics.get("segments", [segment_to_dict(seg) for seg in emitted_segments]),
    }


class DetectionDiagnosticWriter:
    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        if not self.path.is_absolute():
            self.path = (Path.cwd() / self.path).absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:
            print(f"[detection_diagnostics] write failed to {self.path}: {exc}", file=sys.stderr)
