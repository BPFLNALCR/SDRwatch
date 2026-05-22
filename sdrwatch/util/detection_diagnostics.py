"""JSONL diagnostics for scan detection tuning."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from sdrwatch.detection.types import Segment
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
