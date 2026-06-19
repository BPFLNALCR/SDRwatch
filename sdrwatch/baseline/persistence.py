"""Baseline persistence helpers for detections, JSONL emission, and revisit scheduling."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional, Sequence, Tuple

from sdrwatch.baseline.store import BaselineContext, Store
from sdrwatch.detection.span_policy import SignalSpanPolicy, resolve_signal_span_policy
from sdrwatch.detection.types import DetectionCluster, PersistentDetection, RevisitTag, Segment
from sdrwatch.util.time import utc_now_str


@dataclass
class PersistResult:
    is_new: bool
    occ_ratio: Optional[float]
    detection: Optional[PersistentDetection] = None
    characterization: Optional["CharacterizationSnapshot"] = None


@dataclass(frozen=True)
class CharacterizationSnapshot:
    stable_center_hz: int
    center_delta_hz: int
    center_stability_hz: float
    bandwidth_stability_hz: float
    coarse_measurement_count: int
    revisit_measurement_count: int


@dataclass
class CharacterizationTracker:
    stable_center_hz: int
    recent_measured_centers_hz: List[int] = field(default_factory=list)
    recent_measured_bandwidths_hz: List[float] = field(default_factory=list)
    coarse_measurement_count: int = 0
    revisit_measurement_count: int = 0


@dataclass(frozen=True)
class RevisitConfirmationResult:
    detection: PersistentDetection
    characterization: CharacterizationSnapshot


@dataclass
class EdgeCounters:
    expand: int = 0
    shrink: int = 0


@dataclass
class ExtentHysteresisState:
    left: EdgeCounters = field(default_factory=EdgeCounters)
    right: EdgeCounters = field(default_factory=EdgeCounters)


class BaselinePersistence:
    """Manage persistent detections plus downstream emission hooks."""

    def __init__(
        self,
        *,
        store: Store,
        baseline_ctx: BaselineContext,
        args,
        bin_hz: float,
        freq_merge_hz: float,
        center_match_hz: float,
        max_detection_width_ratio: float,
        max_detection_width_hz: float,
        logger=None,
        revisit_margin_hz: float,
        revisit_span_limit_hz: float,
        signal_span_policy: Optional[SignalSpanPolicy] = None,
    ) -> None:
        self.store = store
        self.baseline_ctx = baseline_ctx
        self.args = args
        self.bin_hz = float(bin_hz)
        self.freq_merge_hz = float(freq_merge_hz)
        self.center_match_hz = float(center_match_hz)
        self.max_detection_width_ratio = float(max_detection_width_ratio)
        self.max_detection_width_hz = float(max_detection_width_hz)
        self.signal_span_policy = signal_span_policy or resolve_signal_span_policy(args)
        self.min_persist_bandwidth_hz = float(self.signal_span_policy.min_persist_bandwidth_hz or 0.0)
        self.max_persist_bandwidth_hz = float(
            self.signal_span_policy.max_persist_bandwidth_hz or self.max_detection_width_hz or 0.0
        )
        self.min_detection_width_hz = float(
            getattr(args, "min_detection_width_hz", max(self.bin_hz, 1.0)) or max(self.bin_hz, 1.0)
        )
        self.width_ema_alpha = float(getattr(args, "width_ema_alpha", 0.25) or 0.25)
        self.width_outlier_ratio = float(getattr(args, "width_outlier_ratio", 4.0) or 4.0)
        self.center_history_limit = max(3, int(getattr(args, "center_history_limit", 5) or 5))
        self.center_deadband_hz = max(1, int(round(self.bin_hz)))
        default_center_step = self.center_match_hz / 4.0 if self.center_match_hz > 0.0 else max(self.min_detection_width_hz, self.bin_hz)
        if self.max_detection_width_hz > 0.0:
            default_center_step = min(default_center_step, self.max_detection_width_hz / 4.0)
        self.center_max_step_hz = max(float(self.center_deadband_hz), float(default_center_step))
        self.revisit_center_max_step_hz = max(
            self.center_max_step_hz,
            min(float(self.center_match_hz), self.center_max_step_hz * 1.5) if self.center_match_hz > 0.0 else self.center_max_step_hz,
        )
        self.logger = logger
        self.profile_name = getattr(args, "profile", None)
        self.jsonl_path = getattr(args, "jsonl", None)
        self.notify_enabled = bool(getattr(args, "notify", False))
        self.two_pass_enabled = bool(getattr(args, "two_pass", False))
        self.revisit_margin_hz = float(revisit_margin_hz)
        self.revisit_span_limit_hz = float(revisit_span_limit_hz)
        self.extent_expand_hysteresis = max(1, int(getattr(args, "extent_expand_hysteresis", 2) or 2))
        self.extent_shrink_hysteresis = max(1, int(getattr(args, "extent_shrink_hysteresis", 3) or 3))
        self._persisted: List[PersistentDetection] = store.load_baseline_detections(baseline_ctx.id)
        self._seen_persistent: set[int] = set()
        self._revisit_tags: List[RevisitTag] = []
        self._tag_counter = 0
        self._extent_state: Dict[int, ExtentHysteresisState] = {det.id: ExtentHysteresisState() for det in self._persisted}
        self._characterization_state: Dict[int, CharacterizationTracker] = {
            det.id: CharacterizationTracker(stable_center_hz=int(det.f_center_hz)) for det in self._persisted
        }

    # -----------------
    # Public interface
    # -----------------

    def persist_detection(
        self,
        *,
        cluster: DetectionCluster,
        combined_seg: Segment,
        emit_seg: Optional[Segment] = None,
        confidence: float,
        window_ratio: float,
        duration_seconds: float,
        persistence_mode: str,
        service: Optional[str],
        region: Optional[str],
        notes: Optional[str],
    ) -> PersistResult:
        is_new_detection, persisted_detection, characterization = self._upsert_detection(
            cluster, combined_seg, confidence,
            service=service, region=region, bandplan_notes=notes
        )
        occ_ratio = self._lookup_occ_ratio(combined_seg.f_center_hz)
        occ_threshold = float(getattr(self.args, "new_ema_occ", 0.02) or 0.02)
        is_new_flag = bool(is_new_detection or (occ_ratio is not None and occ_ratio < occ_threshold))
        emit_seg = emit_seg or combined_seg
        record = {
            "baseline_id": self.baseline_ctx.id,
            "time_utc": utc_now_str(),
            "f_center_hz": emit_seg.f_center_hz,
            "f_low_hz": emit_seg.f_low_hz,
            "f_high_hz": emit_seg.f_high_hz,
            "bandwidth_hz": emit_seg.bandwidth_hz,
            "peak_db": emit_seg.peak_db,
            "noise_db": emit_seg.noise_db,
            "snr_db": emit_seg.snr_db,
            "service": service,
            "region": region,
            "notes": notes,
            "is_new": is_new_flag,
            "confidence": confidence,
            "window_ratio": window_ratio,
            "duration_s": duration_seconds,
            "persistence_mode": persistence_mode,
        }
        if self.profile_name:
            record["profile"] = self.profile_name
        self._emit_jsonl(record)
        if is_new_flag:
            body = (
                f"{combined_seg.f_center_hz/1e6:.6f} MHz; "
                f"SNR {combined_seg.snr_db:.1f} dB; {service or 'Unknown'} {region or ''}"
            )
            self._maybe_notify("SDRWatch: New signal", body)
        return PersistResult(
            is_new=is_new_flag,
            occ_ratio=occ_ratio,
            detection=persisted_detection,
            characterization=characterization,
        )

    def finalize_coarse_pass(self) -> List[RevisitTag]:
        missing_ts = utc_now_str()
        to_mark: List[PersistentDetection] = []
        for det in self._persisted:
            if det.id in self._seen_persistent:
                continue
            to_mark.append(det)
            det.missing_since_utc = det.missing_since_utc or missing_ts
            self._log(
                "persist_missing",
                detection_id=det.id,
                baseline_id=det.baseline_id,
                center_hz=det.f_center_hz,
                width_hz=max(det.f_high_hz - det.f_low_hz, 0),
            )
            self._log(
                "persistence_decision",
                action="missing",
                detection_id=det.id,
                baseline_id=det.baseline_id,
                center_hz=det.f_center_hz,
                width_hz=max(det.f_high_hz - det.f_low_hz, 0),
            )
            if self.two_pass_enabled:
                self._schedule_revisit(
                    detection_id=det.id,
                    seg=Segment(
                        f_low_hz=det.f_low_hz,
                        f_high_hz=det.f_high_hz,
                        f_center_hz=det.f_center_hz,
                        peak_db=0.0,
                        noise_db=0.0,
                        snr_db=0.0,
                        bandwidth_hz=float(det.f_high_hz - det.f_low_hz),
                    ),
                    reason="missing",
                )
        if to_mark:
            self.store.begin()
            try:
                for det in to_mark:
                    self.store.mark_detection_missing(det.id, det.baseline_id, missing_ts)
                self.store.commit()
            except Exception:
                self.store.rollback()
                raise
        tags = self._filter_tags(self._revisit_tags) if self.two_pass_enabled else []
        self._revisit_tags = []
        self._log(
            "sweep_finalize",
            seen_persistent=len(self._seen_persistent),
            missing_marked=len(to_mark),
            tags_emitted=len(tags),
        )
        self._seen_persistent.clear()
        return tags

    def apply_revisit_confirmation(self, tag: RevisitTag, seg: Segment) -> Optional[RevisitConfirmationResult]:
        det = self._find_persistent_by_id(tag.detection_id)
        if det is None:
            return None
        seg = self._constrain_revisit_segment(det, seg)
        characterization = self._record_characterization_measurement(
            det,
            measured_center_hz=int(seg.f_center_hz),
            measured_bandwidth_hz=max(float(seg.f_high_hz - seg.f_low_hz), self.bin_hz),
            source_pass="revisit",
            stable_center_hz=None if self._center_smoothing_enabled() else int(seg.f_center_hz),
        )
        # Revisit confirmations should not permanently "ratchet" extents wider via
        # min/max unioning. Use the same width-EMA + hysteresis logic as coarse
        # persistence so widths can converge and stay bounded.
        prev_width = max(float(det.f_high_hz - det.f_low_hz), self.bin_hz)
        measured_width = max(float(seg.f_high_hz - seg.f_low_hz), self.bin_hz)
        target_width = self._blend_width_ema(prev_width, measured_width)
        half = target_width / 2.0
        stable_center_hz = characterization.stable_center_hz
        proposed_low = int(round(float(stable_center_hz) - half))
        proposed_high = int(round(float(stable_center_hz) + half))
        baseline_low = self.baseline_ctx.freq_start_hz
        baseline_high = self.baseline_ctx.freq_stop_hz
        proposed_low = max(proposed_low, baseline_low)
        proposed_high = min(proposed_high, baseline_high)
        if proposed_high <= proposed_low:
            min_width = int(max(1.0, self.min_detection_width_hz))
            proposed_high = min(baseline_high, proposed_low + min_width)
            if proposed_high <= proposed_low:
                proposed_low = max(baseline_low, proposed_high - min_width)
        new_low, new_high = self._apply_extent_hysteresis(det, proposed_low, proposed_high)
        det.f_low_hz = new_low
        det.f_high_hz = new_high
        det.f_center_hz = int(stable_center_hz)
        self._apply_persisted_span_policy(det, stage="revisit_confirmation")
        det.last_seen_utc = utc_now_str()
        det.missing_since_utc = None
        self.store.begin()
        try:
            self.store.update_baseline_detection(det)
            self.store.commit()
        except Exception:
            self.store.rollback()
            raise
        self._log(
            "revisit_apply",
            action="confirmed",
            tag_id=tag.tag_id,
            detection_id=det.id,
            center_hz=det.f_center_hz,
            measured_center_hz=int(seg.f_center_hz),
            stable_center_hz=stable_center_hz,
            center_delta_hz=characterization.center_delta_hz,
            width_hz=max(det.f_high_hz - det.f_low_hz, 0),
            persisted_card_bandwidth_hz=max(det.f_high_hz - det.f_low_hz, 0),
            persist_width_floor_applied_hz=self._persist_width_floor_applied_value(
                measured_width,
                det.f_high_hz - det.f_low_hz,
            ),
        )
        return RevisitConfirmationResult(detection=det, characterization=characterization)

    def apply_revisit_miss(self, tag: RevisitTag) -> None:
        det = self._find_persistent_by_id(tag.detection_id)
        if det is None:
            return
        timestamp = utc_now_str()
        if tag.reason == "new":
            self.store.begin()
            try:
                self.store.delete_baseline_detection(det.id, det.baseline_id)
                self.store.commit()
            except Exception:
                self.store.rollback()
                raise
            self._persisted = [d for d in self._persisted if d.id != det.id]
            self._extent_state.pop(det.id, None)
            self._characterization_state.pop(det.id, None)
            self._log(
                "revisit_apply",
                action="pruned",
                tag_id=tag.tag_id,
                detection_id=det.id,
                reason=tag.reason,
                center_hz=det.f_center_hz,
            )
            return
        det.missing_since_utc = det.missing_since_utc or timestamp
        self.store.begin()
        try:
            self.store.mark_detection_missing(det.id, det.baseline_id, timestamp)
            self.store.commit()
        except Exception:
            self.store.rollback()
            raise
        self._log(
            "revisit_apply",
            action="marked_missing",
            tag_id=tag.tag_id,
            detection_id=det.id,
            reason=tag.reason,
            center_hz=det.f_center_hz,
        )

    # -----------------
    # Internal helpers
    # -----------------

    def _upsert_detection(
        self,
        cluster: DetectionCluster,
        seg: Segment,
        confidence: float,
        service: Optional[str] = None,
        region: Optional[str] = None,
        bandplan_notes: Optional[str] = None,
    ) -> Tuple[bool, PersistentDetection, CharacterizationSnapshot]:
        timestamp = utc_now_str()
        self.store.begin()
        try:
            match = self._match_persistent(seg)
            cluster_center_hz = seg.f_center_hz
            measured_bandwidth_hz = max(float(cluster.f_high_hz - cluster.f_low_hz), self.bin_hz)
            if match:
                characterization = self._record_characterization_measurement(
                    match,
                    measured_center_hz=cluster_center_hz,
                    measured_bandwidth_hz=measured_bandwidth_hz,
                    source_pass="coarse",
                    stable_center_hz=(
                        None
                        if self._center_smoothing_enabled()
                        else self._blend_centers(
                            match.f_center_hz,
                            match.total_hits,
                            cluster_center_hz,
                            cluster.hits,
                        )
                    ),
                )
                blended_center = characterization.stable_center_hz
                prev_width = max(float(match.f_high_hz - match.f_low_hz), self.bin_hz)
                cluster_width = max(float(cluster.f_high_hz - cluster.f_low_hz), self.bin_hz)
                target_width = self._blend_width_ema(prev_width, cluster_width)
                half = target_width / 2.0
                new_low = int(round(blended_center - half))
                new_high = int(round(blended_center + half))
                baseline_low = self.baseline_ctx.freq_start_hz
                baseline_high = self.baseline_ctx.freq_stop_hz
                new_low = max(new_low, baseline_low)
                new_high = min(new_high, baseline_high)
                if new_high <= new_low:
                    min_width = int(max(1.0, self.min_detection_width_hz))
                    new_high = min(baseline_high, new_low + min_width)
                    if new_high <= new_low:
                        new_low = max(baseline_low, new_high - min_width)
                new_low, new_high = self._apply_extent_hysteresis(match, new_low, new_high)
                match.f_low_hz = new_low
                match.f_high_hz = new_high
                match.f_center_hz = blended_center
                self._apply_persisted_span_policy(match, stage="coarse_update")
                match.last_seen_utc = timestamp
                match.total_hits += cluster.hits
                match.total_windows += len(cluster.windows)
                match.confidence = confidence
                # Update power metrics with latest values
                match.peak_db = seg.peak_db
                match.noise_db = seg.noise_db
                match.snr_db = seg.snr_db
                # Update bandplan info if provided
                if service is not None:
                    match.service = service
                if region is not None:
                    match.region = region
                if bandplan_notes is not None:
                    match.bandplan_notes = bandplan_notes
                self.store.update_baseline_detection(match)
                self._log(
                    "persistence_decision",
                    action="update",
                    detection_id=match.id,
                    baseline_id=self.baseline_ctx.id,
                    center_hz=match.f_center_hz,
                    measured_center_hz=cluster_center_hz,
                    stable_center_hz=characterization.stable_center_hz,
                    center_delta_hz=characterization.center_delta_hz,
                    width_hz=max(match.f_high_hz - match.f_low_hz, 0),
                    persisted_card_bandwidth_hz=max(match.f_high_hz - match.f_low_hz, 0),
                    persist_width_floor_applied_hz=self._persist_width_floor_applied_value(
                        cluster_width,
                        match.f_high_hz - match.f_low_hz,
                    ),
                    hits=match.total_hits,
                    windows=match.total_windows,
                    confidence=confidence,
                )
                is_new = False
                persisted_detection = match
            else:
                insert_low, insert_high, insert_clipped, insert_clip_reason = self._policy_span_for(
                    center_hz=cluster_center_hz,
                    width_hz=max(float(cluster.f_high_hz - cluster.f_low_hz), self.bin_hz),
                    stage="coarse_insert",
                    detection_id=None,
                )
                detection_id = self.store.insert_baseline_detection(
                    self.baseline_ctx.id,
                    insert_low,
                    insert_high,
                    cluster_center_hz,
                    cluster.first_seen_ts,
                    cluster.last_seen_ts,
                    cluster.hits,
                    len(cluster.windows),
                    confidence,
                    peak_db=seg.peak_db,
                    noise_db=seg.noise_db,
                    snr_db=seg.snr_db,
                    service=service,
                    region=region,
                    bandplan_notes=bandplan_notes,
                )
                new_det = PersistentDetection(
                    id=detection_id,
                    baseline_id=self.baseline_ctx.id,
                    f_low_hz=insert_low,
                    f_high_hz=insert_high,
                    f_center_hz=cluster_center_hz,
                    first_seen_utc=cluster.first_seen_ts,
                    last_seen_utc=cluster.last_seen_ts,
                    total_hits=cluster.hits,
                    total_windows=len(cluster.windows),
                    confidence=confidence,
                    peak_db=seg.peak_db,
                    noise_db=seg.noise_db,
                    snr_db=seg.snr_db,
                    service=service,
                    region=region,
                    bandplan_notes=bandplan_notes,
                )
                self._persisted.append(new_det)
                self._seen_persistent.add(detection_id)
                self._extent_state[detection_id] = ExtentHysteresisState()
                characterization = self._record_characterization_measurement(
                    new_det,
                    measured_center_hz=cluster_center_hz,
                    measured_bandwidth_hz=measured_bandwidth_hz,
                    source_pass="coarse",
                    stable_center_hz=cluster_center_hz,
                )
                self._log(
                    "persistence_decision",
                    action="insert",
                    detection_id=detection_id,
                    baseline_id=self.baseline_ctx.id,
                    center_hz=cluster_center_hz,
                    stable_center_hz=characterization.stable_center_hz,
                    center_delta_hz=characterization.center_delta_hz,
                    width_hz=max(cluster.f_high_hz - cluster.f_low_hz, 0),
                    persisted_card_bandwidth_hz=max(insert_high - insert_low, 0),
                    persist_width_floor_applied_hz=self._persist_width_floor_applied_value(
                        measured_bandwidth_hz,
                        insert_high - insert_low,
                    ),
                    baseline_clipped=insert_clipped,
                    clip_reason=insert_clip_reason,
                    hits=cluster.hits,
                    windows=len(cluster.windows),
                    confidence=confidence,
                )
                is_new = True
                persisted_detection = new_det
                if self.two_pass_enabled:
                    self._schedule_revisit(detection_id=detection_id, seg=seg, reason="new")
            self.store.commit()
            return is_new, persisted_detection, characterization
        except Exception:
            self.store.rollback()
            raise

    def _match_persistent(self, seg: Segment) -> Optional[PersistentDetection]:
        for det in self._persisted:
            # Use an effective span for overlap checks. If a stored persistent
            # detection has become too wide (e.g., from historical settings),
            # treating it as an "overlap" region can cause it to absorb unrelated
            # signals. When a max width is configured, cap the span used for
            # matching around the stored center.
            det_low = det.f_low_hz
            det_high = det.f_high_hz
            if self.max_detection_width_hz > 0.0:
                max_w = float(self.max_detection_width_hz)
                det_w = float(det_high - det_low)
                if det_w > max_w:
                    half = max_w / 2.0
                    det_low = int(round(float(det.f_center_hz) - half))
                    det_high = int(round(float(det.f_center_hz) + half))
                    det_low = max(det_low, self.baseline_ctx.freq_start_hz)
                    det_high = min(det_high, self.baseline_ctx.freq_stop_hz)
            spans_overlap = not (
                seg.f_high_hz < (det_low - self.freq_merge_hz)
                or seg.f_low_hz > (det_high + self.freq_merge_hz)
            )
            center_close = abs(seg.f_center_hz - det.f_center_hz) <= self.center_match_hz
            if spans_overlap or center_close:
                width_det = max(float(det.f_high_hz - det.f_low_hz), self.bin_hz)
                width_seg = max(float(seg.f_high_hz - seg.f_low_hz), self.bin_hz)
                max_ratio = float(self.max_detection_width_ratio)
                if width_det > 0.0 and width_seg > width_det * max_ratio:
                    self._log(
                        "persist_width_reject",
                        detection_id=det.id,
                        baseline_id=self.baseline_ctx.id,
                        width_det_hz=width_det,
                        width_seg_hz=width_seg,
                        max_ratio=max_ratio,
                    )
                    self._log(
                        "persistence_decision",
                        action="width_reject",
                        detection_id=det.id,
                        baseline_id=self.baseline_ctx.id,
                        center_hz=seg.f_center_hz,
                        width_det_hz=width_det,
                        width_seg_hz=width_seg,
                        max_ratio=max_ratio,
                        reason="segment width exceeds max ratio",
                    )
                    continue
                self._seen_persistent.add(det.id)
                if det.missing_since_utc:
                    det.missing_since_utc = None
                    self.store.clear_detection_missing(det.id, det.baseline_id)
                    self._log(
                        "persistence_decision",
                        action="missing_cleared",
                        detection_id=det.id,
                        baseline_id=self.baseline_ctx.id,
                        center_hz=det.f_center_hz,
                    )
                self._log(
                    "persist_match",
                    detection_id=det.id,
                    baseline_id=self.baseline_ctx.id,
                    center_delta_hz=int(seg.f_center_hz - det.f_center_hz),
                    spans_overlap=spans_overlap,
                    center_close=center_close,
                    seg_width_hz=max(seg.bandwidth_hz, 0.0),
                    persisted_width_hz=max(det.f_high_hz - det.f_low_hz, 0),
                )
                self._log(
                    "persistence_decision",
                    action="match",
                    detection_id=det.id,
                    baseline_id=self.baseline_ctx.id,
                    center_hz=det.f_center_hz,
                    candidate_center_hz=seg.f_center_hz,
                    center_delta_hz=int(seg.f_center_hz - det.f_center_hz),
                    spans_overlap=spans_overlap,
                    center_close=center_close,
                    seg_width_hz=max(seg.bandwidth_hz, 0.0),
                    persisted_width_hz=max(det.f_high_hz - det.f_low_hz, 0),
                )
                return det
        self._log(
            "persist_no_match",
            baseline_id=self.baseline_ctx.id,
            center_hz=seg.f_center_hz,
            width_hz=max(seg.bandwidth_hz, 0.0),
        )
        self._log(
            "persistence_decision",
            action="no_match",
            baseline_id=self.baseline_ctx.id,
            center_hz=seg.f_center_hz,
            width_hz=max(seg.bandwidth_hz, 0.0),
        )
        return None

    def _lookup_occ_ratio(self, freq_hz: int) -> Optional[float]:
        """Return the duty cycle / occupancy ratio for a frequency.
        
        Uses time-based duty cycle (occupied_ms / observed_ms) when available,
        falling back to window-based occ_ratio for backward compatibility.
        This provides more accurate detection of intermittent/bursty signals.
        """
        bin_index = self._bin_index_for_freq(freq_hz)
        if bin_index is None:
            return None
        return self.store.baseline_duty_cycle(self.baseline_ctx.id, bin_index)

    def _center_smoothing_enabled(self) -> bool:
        return bool(self.signal_span_policy.center_smoothing_enabled)

    def _tracker_for(self, det: PersistentDetection) -> CharacterizationTracker:
        return self._characterization_state.setdefault(
            det.id,
            CharacterizationTracker(stable_center_hz=int(det.f_center_hz)),
        )

    def _record_characterization_measurement(
        self,
        det: PersistentDetection,
        *,
        measured_center_hz: int,
        measured_bandwidth_hz: float,
        source_pass: str,
        stable_center_hz: Optional[int],
    ) -> CharacterizationSnapshot:
        tracker = self._tracker_for(det)
        tracker.recent_measured_centers_hz.append(int(measured_center_hz))
        tracker.recent_measured_bandwidths_hz.append(max(float(measured_bandwidth_hz), self.bin_hz))
        if len(tracker.recent_measured_centers_hz) > self.center_history_limit:
            tracker.recent_measured_centers_hz = tracker.recent_measured_centers_hz[-self.center_history_limit :]
        if len(tracker.recent_measured_bandwidths_hz) > self.center_history_limit:
            tracker.recent_measured_bandwidths_hz = tracker.recent_measured_bandwidths_hz[-self.center_history_limit :]

        if source_pass == "revisit":
            tracker.revisit_measurement_count += 1
        else:
            tracker.coarse_measurement_count += 1

        if stable_center_hz is None:
            target_center_hz = int(round(median(tracker.recent_measured_centers_hz)))
            step_limit_hz = (
                self.revisit_center_max_step_hz if source_pass == "revisit" else self.center_max_step_hz
            )
            stable_center_hz = self._bounded_center_step(
                current_hz=int(tracker.stable_center_hz),
                target_hz=target_center_hz,
                max_step_hz=step_limit_hz,
                epsilon_hz=self.center_deadband_hz,
            )
        tracker.stable_center_hz = int(stable_center_hz)

        center_stability_hz = 0.0
        if len(tracker.recent_measured_centers_hz) > 1:
            center_stability_hz = float(
                max(tracker.recent_measured_centers_hz) - min(tracker.recent_measured_centers_hz)
            )
        bandwidth_stability_hz = 0.0
        if len(tracker.recent_measured_bandwidths_hz) > 1:
            bandwidth_stability_hz = float(
                max(tracker.recent_measured_bandwidths_hz) - min(tracker.recent_measured_bandwidths_hz)
            )
        return CharacterizationSnapshot(
            stable_center_hz=int(tracker.stable_center_hz),
            center_delta_hz=int(int(measured_center_hz) - int(tracker.stable_center_hz)),
            center_stability_hz=center_stability_hz,
            bandwidth_stability_hz=bandwidth_stability_hz,
            coarse_measurement_count=int(tracker.coarse_measurement_count),
            revisit_measurement_count=int(tracker.revisit_measurement_count),
        )

    def _bin_index_for_freq(self, freq_hz: int) -> Optional[int]:
        if freq_hz < self.baseline_ctx.freq_start_hz or freq_hz > self.baseline_ctx.freq_stop_hz:
            return None
        offset = (freq_hz - self.baseline_ctx.freq_start_hz) / max(self.baseline_ctx.bin_hz, 1.0)
        return int(round(offset))

    def _schedule_revisit(self, *, detection_id: Optional[int], seg: Segment, reason: str) -> None:
        if not self.two_pass_enabled:
            return
        margin = max(self.revisit_margin_hz, float(seg.bandwidth_hz or self.bin_hz))
        low = int(max(seg.f_low_hz - margin, 0))
        high = int(seg.f_high_hz + margin)
        tag_id = f"rv{self.baseline_ctx.id}_{self._tag_counter}"
        self._tag_counter += 1
        tag = RevisitTag(
            tag_id=tag_id,
            detection_id=detection_id,
            f_center_hz=int(seg.f_center_hz),
            f_low_hz=low,
            f_high_hz=high,
            reason=reason,
            created_utc=utc_now_str(),
        )
        blocked = reason != "missing" and self._tag_overlaps_known(tag)
        if blocked:
            self._log(
                "revisit_queue",
                action="skipped_overlap",
                tag_id=tag.tag_id,
                detection_id=detection_id,
                reason=reason,
                center_hz=tag.f_center_hz,
                width_hz=max(tag.f_high_hz - tag.f_low_hz, 0),
            )
            return
        self._revisit_tags.append(tag)
        self._log(
            "revisit_queue",
            action="queued",
            tag_id=tag.tag_id,
            detection_id=detection_id,
            reason=reason,
            center_hz=tag.f_center_hz,
            width_hz=max(tag.f_high_hz - tag.f_low_hz, 0),
        )

    def _tag_overlaps_known(self, tag: RevisitTag) -> bool:
        for det in self._persisted:
            if det.id == tag.detection_id:
                continue
            if det.missing_since_utc:
                continue
            if not (tag.f_high_hz < det.f_low_hz or tag.f_low_hz > det.f_high_hz):
                return True
        return False

    def _find_persistent_by_id(self, detection_id: Optional[int]) -> Optional[PersistentDetection]:
        if detection_id is None:
            return None
        for det in self._persisted:
            if det.id == detection_id:
                return det
        return None

    def _constrain_revisit_segment(self, det: PersistentDetection, seg: Segment) -> Segment:
        limit = float(self.revisit_span_limit_hz)
        span_width = float(seg.f_high_hz - seg.f_low_hz)
        if limit <= 0.0 or span_width <= limit:
            return seg
        anchor = det.f_center_hz if det else seg.f_center_hz
        half = limit / 2.0
        low = int(round(anchor - half))
        high = int(round(anchor + half))
        if low < seg.f_low_hz:
            shift = seg.f_low_hz - low
            low += shift
            high += shift
        if high > seg.f_high_hz:
            shift = high - seg.f_high_hz
            high -= shift
            low -= shift
        low = max(low, seg.f_low_hz)
        high = min(high, seg.f_high_hz)
        if high <= low:
            low = seg.f_low_hz
            high = min(seg.f_high_hz, seg.f_low_hz + int(limit))
        seg.f_low_hz = low
        seg.f_high_hz = high
        seg.f_center_hz = int(round((low + high) / 2.0))
        seg.bandwidth_hz = max(float(seg.f_high_hz - seg.f_low_hz), self.bin_hz)
        self._log(
            "revisit_trim",
            detection_id=(det.id if det else None),
            original_width_hz=span_width,
            trimmed_width_hz=float(seg.bandwidth_hz),
            anchor_hz=anchor,
        )
        return seg

    def _filter_tags(self, tags: Sequence[RevisitTag]) -> List[RevisitTag]:
        seen: set[str] = set()
        filtered: List[RevisitTag] = []
        dup_dropped = 0
        overlap_dropped = 0
        for tag in tags:
            key = f"{tag.detection_id}:{tag.f_center_hz}:{tag.reason}"
            if key in seen:
                dup_dropped += 1
                continue
            if tag.reason != "missing" and self._tag_overlaps_known(tag):
                overlap_dropped += 1
                continue
            seen.add(key)
            filtered.append(tag)
        self._log(
            "revisit_filter_summary",
            input=len(tags),
            output=len(filtered),
            duplicates=dup_dropped,
            overlap_blocked=overlap_dropped,
        )
        return filtered

    @staticmethod
    def _bounded_center_step(*, current_hz: int, target_hz: int, max_step_hz: float, epsilon_hz: int) -> int:
        delta_hz = int(target_hz) - int(current_hz)
        if abs(delta_hz) <= int(epsilon_hz):
            return int(current_hz)
        step_limit_hz = max(float(max_step_hz), float(epsilon_hz))
        if delta_hz > step_limit_hz:
            delta_hz = int(round(step_limit_hz))
        elif delta_hz < -step_limit_hz:
            delta_hz = -int(round(step_limit_hz))
        return int(current_hz) + int(delta_hz)

    def _blend_centers(self, center_a: int, weight_a: int, center_b: int, weight_b: int) -> int:
        wa = max(1, int(weight_a))
        wb = max(1, int(weight_b))
        return int(round((center_a * wa + center_b * wb) / float(wa + wb)))

    def _extent_state_for(self, det_id: int) -> ExtentHysteresisState:
        return self._extent_state.setdefault(det_id, ExtentHysteresisState())

    def _apply_extent_hysteresis(self, det: PersistentDetection, proposed_low: int, proposed_high: int) -> Tuple[int, int]:
        state = self._extent_state_for(det.id)
        epsilon = max(1, int(round(self.bin_hz)))
        low = self._update_edge_with_hysteresis(
            state.left,
            det.f_low_hz,
            proposed_low,
            epsilon,
            self.extent_expand_hysteresis,
            self.extent_shrink_hysteresis,
            direction="left",
        )
        high = self._update_edge_with_hysteresis(
            state.right,
            det.f_high_hz,
            proposed_high,
            epsilon,
            self.extent_expand_hysteresis,
            self.extent_shrink_hysteresis,
            direction="right",
        )
        low = max(low, self.baseline_ctx.freq_start_hz)
        high = min(high, self.baseline_ctx.freq_stop_hz)
        if high <= low:
            high = min(self.baseline_ctx.freq_stop_hz, low + epsilon)
        return low, high

    def _enforce_persisted_span_invariant(self, det: PersistentDetection) -> None:
        baseline_low = int(self.baseline_ctx.freq_start_hz)
        baseline_high = int(self.baseline_ctx.freq_stop_hz)
        if baseline_high < baseline_low:
            baseline_low, baseline_high = baseline_high, baseline_low

        center = int(det.f_center_hz)
        center = min(max(center, baseline_low), baseline_high)

        low = int(det.f_low_hz)
        high = int(det.f_high_hz)
        if high < low:
            low, high = high, low
        low = max(low, baseline_low)
        high = min(high, baseline_high)

        if center < low:
            low = center
        if center > high:
            high = center

        min_width = int(round(self.min_persist_bandwidth_hz)) if self.min_persist_bandwidth_hz > 0.0 else 0
        if min_width > 0 and high - low < min_width:
            half = min_width / 2.0
            low = int(round(center - half))
            high = int(round(center + half))
            low = max(low, baseline_low)
            high = min(high, baseline_high)
            if center < low:
                low = center
            if center > high:
                high = center

        effective_max_width = self.max_persist_bandwidth_hz or self.max_detection_width_hz
        max_width = int(round(effective_max_width)) if effective_max_width > 0.0 else 0
        if max_width > 0 and high - low > max_width:
            half = max_width // 2
            low = max(center - half, baseline_low)
            high = min(low + max_width, baseline_high)
            if center > high:
                high = center
                low = max(baseline_low, high - max_width)
            if center < low:
                low = center
                high = min(baseline_high, low + max_width)

        if high < low:
            high = low

        det.f_low_hz = int(low)
        det.f_high_hz = int(high)
        det.f_center_hz = int(center)

    def _persist_min_width_hz(self) -> float:
        return max(
            float(getattr(self, "bin_hz", 1.0) or 1.0),
            float(getattr(self, "min_detection_width_hz", 1.0) or 1.0),
            float(getattr(self, "min_persist_bandwidth_hz", 0.0) or 0.0),
        )

    def _policy_span_for(
        self,
        *,
        center_hz: int,
        width_hz: float,
        stage: str,
        detection_id: Optional[int],
    ) -> Tuple[int, int, bool, Optional[str]]:
        baseline_low = int(self.baseline_ctx.freq_start_hz)
        baseline_high = int(self.baseline_ctx.freq_stop_hz)
        center = min(max(int(center_hz), baseline_low), baseline_high)
        requested_width = max(float(width_hz), self._persist_min_width_hz())
        if self.max_persist_bandwidth_hz > 0.0:
            requested_width = min(requested_width, self.max_persist_bandwidth_hz)
        half = requested_width / 2.0
        raw_low = int(round(center - half))
        raw_high = int(round(center + half))
        low = max(raw_low, baseline_low)
        high = min(raw_high, baseline_high)
        if high <= low:
            min_width = int(max(1.0, self.bin_hz))
            high = min(baseline_high, low + min_width)
            if high <= low:
                low = max(baseline_low, high - min_width)
        clipped = bool(low != raw_low or high != raw_high)
        output_width = max(float(high - low), 0.0)
        floor_applied = self._persist_width_floor_applied_value(width_hz, output_width)
        self._log(
            "width_decision",
            stage=f"persisted_card_{stage}",
            detection_id=detection_id,
            baseline_id=self.baseline_ctx.id,
            center_hz=center,
            input_width_hz=float(width_hz),
            output_width_hz=output_width,
            min_width_hz=float(self._persist_min_width_hz()),
            min_persist_bandwidth_hz=(
                float(self.min_persist_bandwidth_hz) if self.min_persist_bandwidth_hz > 0.0 else None
            ),
            max_width_hz=(float(self.max_persist_bandwidth_hz) if self.max_persist_bandwidth_hz > 0.0 else None),
            persist_width_floor_applied_hz=floor_applied,
            baseline_clipped=clipped,
            clip_reason=("scan_edge" if clipped else None),
            was_floored=floor_applied > 0.0,
            was_clamped=bool(self.max_persist_bandwidth_hz > 0.0 and float(width_hz) > self.max_persist_bandwidth_hz),
        )
        return low, high, clipped, ("scan_edge" if clipped else None)

    def _apply_persisted_span_policy(self, det: PersistentDetection, *, stage: str) -> Tuple[bool, Optional[str]]:
        low, high, clipped, clip_reason = self._policy_span_for(
            center_hz=int(det.f_center_hz),
            width_hz=max(float(det.f_high_hz - det.f_low_hz), self.bin_hz),
            stage=stage,
            detection_id=det.id,
        )
        det.f_low_hz = low
        det.f_high_hz = high
        self._enforce_persisted_span_invariant(det)
        return clipped, clip_reason

    def _persist_width_floor_applied_value(self, measured_width_hz: float, output_width_hz: float) -> float:
        floor = float(self.min_persist_bandwidth_hz or 0.0)
        if floor <= 0.0:
            return 0.0
        measured = max(float(measured_width_hz), self.bin_hz)
        if measured >= floor:
            return 0.0
        if output_width_hz < floor:
            return 0.0
        return float(floor - measured)

    def _blend_width_ema(self, prev_width: float, measured_width: float) -> float:
        original_prev_width = float(prev_width)
        original_measured_width = float(measured_width)
        min_width_hz = self._persist_min_width_hz()
        if prev_width <= 0.0:
            prev_width = max(min_width_hz, measured_width)
        measurement = max(measured_width, min_width_hz)
        was_floored = measurement != measured_width
        outlier_ratio = float(self.width_outlier_ratio)
        outlier_rejected = False
        if prev_width > 0.0 and outlier_ratio > 0.0:
            ratio = measurement / prev_width
            # Reject only large *expansion* outliers. Shrink outliers are allowed
            # so oversized persisted spans can converge back down over time.
            if ratio > outlier_ratio:
                measurement = prev_width
                outlier_rejected = True
        alpha = float(min(max(self.width_ema_alpha, 0.01), 1.0))
        blended = prev_width + alpha * (measurement - prev_width)
        blended = max(blended, min_width_hz)
        was_clamped = False
        max_width_hz = self.max_persist_bandwidth_hz or self.max_detection_width_hz
        if max_width_hz > 0.0 and blended > max_width_hz:
            blended = max_width_hz
            was_clamped = True
        self._log(
            "width_decision",
            stage="persistence_width_ema",
            previous_width_hz=original_prev_width,
            input_width_hz=original_measured_width,
            output_width_hz=blended,
            min_width_hz=float(min_width_hz),
            min_persist_bandwidth_hz=(
                float(self.min_persist_bandwidth_hz) if self.min_persist_bandwidth_hz > 0.0 else None
            ),
            persist_width_floor_applied_hz=self._persist_width_floor_applied_value(
                original_measured_width,
                blended,
            ),
            max_width_hz=float(max_width_hz),
            alpha=alpha,
            outlier_ratio=outlier_ratio,
            was_floored=was_floored,
            was_clamped=was_clamped,
            outlier_rejected=outlier_rejected,
        )
        return blended

    @staticmethod
    def _update_edge_with_hysteresis(
        counters: EdgeCounters,
        current: int,
        proposed: int,
        epsilon: int,
        expand_threshold: int,
        shrink_threshold: int,
        *,
        direction: str,
    ) -> int:
        updated = current
        if direction == "left":
            expand_condition = proposed < current - epsilon
            shrink_condition = proposed > current + epsilon
        else:
            expand_condition = proposed > current + epsilon
            shrink_condition = proposed < current - epsilon

        if expand_condition:
            counters.expand += 1
            if counters.expand >= expand_threshold:
                updated = proposed
                counters.expand = 0
                counters.shrink = 0
        else:
            counters.expand = 0

        if shrink_condition:
            counters.shrink += 1
            if counters.shrink >= shrink_threshold:
                updated = proposed
                counters.shrink = 0
                counters.expand = 0
        else:
            counters.shrink = 0

        return updated

    def _emit_jsonl(self, record: dict) -> None:
        path = self.jsonl_path
        if not path:
            return
        try:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
        except Exception:
            pass

    def _maybe_notify(self, title: str, body: str) -> None:
        if not self.notify_enabled:
            return
        try:
            subprocess.Popen(["notify-send", title, body])
        except Exception:
            pass

    def _log(self, event: str, **fields) -> None:
        logger = getattr(self, "logger", None)
        if not logger:
            return
        payload = dict(fields)
        profile_name = getattr(self, "profile_name", None)
        if profile_name:
            payload.setdefault("profile", profile_name)
        logger.log(event, **payload)
