"""Dataclasses shared across detection, baseline, and sweeper layers."""

from dataclasses import dataclass, field
from typing import Optional, Set


@dataclass
class Segment:
    f_low_hz: int
    f_high_hz: int
    f_center_hz: int
    peak_db: float
    noise_db: float
    snr_db: float
    bandwidth_hz: float = 0.0


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
