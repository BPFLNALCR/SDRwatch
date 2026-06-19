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
    stable_center_hz: int
    center_delta_hz: int
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
    bandwidth_interpretation: str = "threshold_fragment"
    width_floor_applied_hz: float = 0.0
    persist_width_floor_applied_hz: float = 0.0
    persisted_card_bandwidth_hz: Optional[float] = None
    baseline_clipped: bool = False
    clip_reason: Optional[str] = None

    def to_record(self) -> Dict[str, Any]:
        persisted_card_bandwidth = (
            float(self.persisted_card_bandwidth_hz)
            if self.persisted_card_bandwidth_hz is not None
            else float(self.match_span.bandwidth_hz)
        )
        record: Dict[str, Any] = {
            "event": "characterization_record",
            "detection_id": self.detection_id,
            "baseline_id": self.baseline_id,
            "source_pass": self.source_pass,
            **self.raw_segment.prefixed_fields("raw"),
            **self.measured_span.prefixed_fields("measured"),
            **self.match_span.prefixed_fields("match"),
            **self.display_span.prefixed_fields("display"),
            "raw_fragment_bandwidth_hz": self.raw_segment.bandwidth_hz,
            "raw_fragment_center_hz": self.raw_segment.center_hz,
            "measured_occupied_bandwidth_hz": self.measured_span.bandwidth_hz,
            "identity_match_bandwidth_hz": self.match_span.bandwidth_hz,
            "persisted_card_bandwidth_hz": persisted_card_bandwidth,
            "bandwidth_interpretation": self.bandwidth_interpretation,
            "width_floor_applied_hz": float(self.width_floor_applied_hz),
            "persist_width_floor_applied_hz": float(self.persist_width_floor_applied_hz),
            "baseline_clipped": bool(self.baseline_clipped),
            "clip_reason": self.clip_reason,
            "stable_center_hz": self.stable_center_hz,
            "center_delta_hz": self.center_delta_hz,
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
        persisted_card_bandwidth = (
            float(self.persisted_card_bandwidth_hz)
            if self.persisted_card_bandwidth_hz is not None
            else float(self.match_span.bandwidth_hz)
        )
        return {
            "signal_id": self.detection_id,
            "baseline_id": self.baseline_id,
            "source_pass": self.source_pass,
            "raw_segment": {
                **self.raw_segment.summary_dict(),
                "raw_fragment_bandwidth_hz": self.raw_segment.bandwidth_hz,
                "raw_fragment_center_hz": self.raw_segment.center_hz,
                "bandwidth_interpretation": self.bandwidth_interpretation,
            },
            "measured_characterization": {
                "center_hz": self.measured_span.center_hz,
                "stable_center_hz": self.stable_center_hz,
                "center_delta_hz": self.center_delta_hz,
                "occupied_bandwidth_hz": self.measured_span.bandwidth_hz,
                "measured_occupied_bandwidth_hz": self.measured_span.bandwidth_hz,
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
            "match_span": {
                **self.match_span.summary_dict(),
                "identity_match_bandwidth_hz": self.match_span.bandwidth_hz,
                "width_floor_applied_hz": float(self.width_floor_applied_hz),
            },
            "persisted_card_span": {
                "bandwidth_hz": persisted_card_bandwidth,
                "persisted_card_bandwidth_hz": persisted_card_bandwidth,
                "persist_width_floor_applied_hz": float(self.persist_width_floor_applied_hz),
                "baseline_clipped": bool(self.baseline_clipped),
                "clip_reason": self.clip_reason,
            },
            "display_span": self.display_span.summary_dict(),
            "bandwidth_interpretation": self.bandwidth_interpretation,
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
    sweep_loop_ids: Set[int] = field(default_factory=set)
    best_seg: Segment = field(default_factory=lambda: Segment(0, 0, 0, -999.0, -999.0, -999.0, 0.0))
    emitted: bool = False
    center_weight_sum: float = 0.0
    center_weight_total: float = 0.0


@dataclass(frozen=True)
class CrossSweepObservation:
    sweep_loop_id: int
    window_idx: int
    center_hz: int
    raw_low_hz: int
    raw_high_hz: int
    raw_bandwidth_hz: float
    match_low_hz: int
    match_high_hz: int
    match_bandwidth_hz: float
    measured_bandwidth_hz: float
    source_pass: str
    snr_db: float
    peak_db: float
    noise_db: float
    observed_at_utc: str


@dataclass
class CrossSweepCandidateState:
    candidate_id: str
    baseline_id: int
    first_observed_sweep_id: int
    last_observed_sweep_id: int
    observation_count: int
    observed_sweep_ids: Set[int]
    stable_center_hz: int
    match_low_hz: int
    match_high_hz: int
    match_bandwidth_hz: float
    measured_bandwidth_hz: float
    last_raw_segment: Segment
    last_observation: CrossSweepObservation
    promotion_ready: bool = False
    rejection_reason: Optional[str] = None

    @property
    def observation_loop_count(self) -> int:
        return len(self.observed_sweep_ids)


@dataclass(frozen=True)
class DeviceTelemetrySnapshot:
    event: str
    device_key: Optional[str]
    device_kind: Optional[str]
    device_index: Optional[int]
    device_serial: Optional[str]
    device_label: Optional[str]
    device_tuner: Optional[str]
    driver: Optional[str]
    requested_gain: Optional[str]
    actual_gain: Optional[float]
    gain_mode: str
    supported_gains: Optional[List[float]]
    sample_rate_hz: Optional[float]
    actual_sample_rate_hz: Optional[float]
    fft: Optional[int]
    bin_width_hz: Optional[float]
    selected_profile: Optional[str]
    unavailable_fields: List[str] = field(default_factory=list)

    def to_record(self) -> Dict[str, Any]:
        return {
            "event": self.event,
            "device_key": self.device_key,
            "device_kind": self.device_kind,
            "device_index": self.device_index,
            "device_serial": self.device_serial,
            "device_label": self.device_label,
            "device_tuner": self.device_tuner,
            "driver": self.driver,
            "requested_gain": self.requested_gain,
            "actual_gain": self.actual_gain,
            "gain_mode": self.gain_mode,
            "supported_gains": self.supported_gains,
            "sample_rate_hz": self.sample_rate_hz,
            "actual_sample_rate_hz": self.actual_sample_rate_hz,
            "fft": self.fft,
            "bin_width_hz": self.bin_width_hz,
            "selected_profile": self.selected_profile,
            "unavailable_fields": list(self.unavailable_fields),
        }


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
