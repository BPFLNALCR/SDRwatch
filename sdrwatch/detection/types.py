"""Dataclasses shared across detection, baseline, and sweeper layers."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set


@dataclass
class Segment:
    f_low_hz: int
    f_high_hz: int
    f_center_hz: int
    peak_db: float
    noise_db: float
    snr_db: float
    bandwidth_hz: float = 0.0


@dataclass(frozen=True)
class CharacterizationSpan:
    low_hz: int
    high_hz: int
    center_hz: int
    bandwidth_hz: float

    @classmethod
    def from_bounds(
        cls,
        *,
        low_hz: int,
        high_hz: int,
        center_hz: int,
        bandwidth_hz: float,
        min_bandwidth_hz: float = 1.0,
    ) -> "CharacterizationSpan":
        low = int(low_hz)
        high = int(high_hz)
        if high < low:
            low, high = high, low
        center = int(center_hz)
        if center < low:
            low = center
        if center > high:
            high = center
        width = max(float(bandwidth_hz), float(min_bandwidth_hz))
        return cls(low_hz=low, high_hz=high, center_hz=center, bandwidth_hz=width)

    def prefixed_fields(self, prefix: str) -> Dict[str, Any]:
        return {
            f"{prefix}_low_hz": self.low_hz,
            f"{prefix}_high_hz": self.high_hz,
            f"{prefix}_center_hz": self.center_hz,
            f"{prefix}_bandwidth_hz": self.bandwidth_hz,
        }

    def summary_dict(self) -> Dict[str, Any]:
        return {
            "low_hz": self.low_hz,
            "high_hz": self.high_hz,
            "center_hz": self.center_hz,
            "bandwidth_hz": self.bandwidth_hz,
        }


@dataclass(frozen=True)
class CharacterizationEvidence:
    detection_id: Optional[int]
    baseline_id: int
    source_pass: str
    raw_segment: CharacterizationSpan
    measured_span: CharacterizationSpan
    match_span: CharacterizationSpan
    display_span: CharacterizationSpan
    peak_db: float
    noise_db: float
    snr_db: float
    measured_bandwidth_confidence: float
    characterization_confidence: float
    characterization_method: str
    center_stability_hz: float = 0.0
    bandwidth_stability_hz: float = 0.0
    revisit_measurement_count: int = 0
    coarse_measurement_count: int = 1
    classification_candidate: str = "unknown"
    classification_evidence: List[str] = field(default_factory=list)
    evidence_sources: List[str] = field(default_factory=list)
    bandplan_service: Optional[str] = None
    bandplan_region: Optional[str] = None
    bandplan_notes: Optional[str] = None
    profile_context: Optional[str] = None
    context_only: bool = False

    def to_record(self) -> Dict[str, Any]:
        record: Dict[str, Any] = {
            "event": "characterization_record",
            "detection_id": self.detection_id,
            "baseline_id": self.baseline_id,
            "source_pass": self.source_pass,
            **self.raw_segment.prefixed_fields("raw"),
            **self.measured_span.prefixed_fields("measured"),
            **self.match_span.prefixed_fields("match"),
            **self.display_span.prefixed_fields("display"),
            "peak_db": self.peak_db,
            "noise_db": self.noise_db,
            "snr_db": self.snr_db,
            "measured_bandwidth_confidence": self.measured_bandwidth_confidence,
            "characterization_confidence": self.characterization_confidence,
            "characterization_method": self.characterization_method,
            "center_stability_hz": self.center_stability_hz,
            "bandwidth_stability_hz": self.bandwidth_stability_hz,
            "revisit_measurement_count": self.revisit_measurement_count,
            "coarse_measurement_count": self.coarse_measurement_count,
            "classification_candidate": self.classification_candidate,
            "classification_evidence": list(self.classification_evidence),
            "evidence_sources": list(self.evidence_sources),
            "bandplan_service": self.bandplan_service,
            "bandplan_region": self.bandplan_region,
            "bandplan_notes": self.bandplan_notes,
            "profile_context": self.profile_context,
            "context_only": bool(self.context_only),
            "raw_segment": self.raw_segment.summary_dict(),
            "measured_span": self.measured_span.summary_dict(),
            "match_span": self.match_span.summary_dict(),
            "display_span": self.display_span.summary_dict(),
        }
        return record

    def to_summary(self) -> Dict[str, Any]:
        return {
            "signal_id": self.detection_id,
            "baseline_id": self.baseline_id,
            "source_pass": self.source_pass,
            "raw_segment": self.raw_segment.summary_dict(),
            "measured_characterization": {
                "center_hz": self.measured_span.center_hz,
                "occupied_bandwidth_hz": self.measured_span.bandwidth_hz,
                "bandwidth_confidence": self.measured_bandwidth_confidence,
                "characterization_confidence": self.characterization_confidence,
                "characterization_method": self.characterization_method,
                "center_stability_hz": self.center_stability_hz,
                "bandwidth_stability_hz": self.bandwidth_stability_hz,
                "coarse_measurement_count": self.coarse_measurement_count,
                "revisit_measurement_count": self.revisit_measurement_count,
                "peak_db": self.peak_db,
                "noise_db": self.noise_db,
                "snr_db": self.snr_db,
            },
            "match_span": self.match_span.summary_dict(),
            "display_span": self.display_span.summary_dict(),
            "context": {
                "bandplan_service": self.bandplan_service,
                "bandplan_region": self.bandplan_region,
                "bandplan_notes": self.bandplan_notes,
                "profile_context": self.profile_context,
            },
            "classification": {
                "candidate": self.classification_candidate,
                "evidence": list(self.classification_evidence),
                "evidence_sources": list(self.evidence_sources),
                "context_only": bool(self.context_only),
            },
        }


@dataclass
class DetectionCluster:
    f_low_hz: int
    f_high_hz: int
    first_seen_ts: str
    last_seen_ts: str
    first_window: int
    last_window: int
    hits: int = 0
    windows: Set[int] = field(default_factory=set)
    best_seg: Segment = field(default_factory=lambda: Segment(0, 0, 0, -999.0, -999.0, -999.0, 0.0))
    emitted: bool = False
    center_weight_sum: float = 0.0
    center_weight_total: float = 0.0


@dataclass
class PersistentDetection:
    id: int
    baseline_id: int
    f_low_hz: int
    f_high_hz: int
    f_center_hz: int
    first_seen_utc: str
    last_seen_utc: str
    total_hits: int
    total_windows: int
    confidence: float
    missing_since_utc: Optional[str] = None
    peak_db: Optional[float] = None
    noise_db: Optional[float] = None
    snr_db: Optional[float] = None
    service: Optional[str] = None
    region: Optional[str] = None
    bandplan_notes: Optional[str] = None


@dataclass
class RevisitTag:
    tag_id: str
    detection_id: Optional[int]
    f_center_hz: int
    f_low_hz: int
    f_high_hz: int
    reason: str  # "new", "missing"
    created_utc: str
