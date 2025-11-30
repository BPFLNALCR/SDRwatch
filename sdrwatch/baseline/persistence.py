"""Baseline persistence helpers for detections, JSONL emission, and revisit scheduling."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from sdrwatch.baseline.store import BaselineContext, Store
from sdrwatch.detection.types import DetectionCluster, PersistentDetection, RevisitTag, Segment
from sdrwatch.util.time import utc_now_str


@dataclass
class PersistResult:
    is_new: bool
    occ_ratio: Optional[float]


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
    ) -> None:
        self.store = store
        self.baseline_ctx = baseline_ctx
        self.args = args
        self.bin_hz = float(bin_hz)
        self.freq_merge_hz = float(freq_merge_hz)
        self.center_match_hz = float(center_match_hz)
        self.max_detection_width_ratio = float(max_detection_width_ratio)
        self.max_detection_width_hz = float(max_detection_width_hz)
        self.logger = logger
        self.profile_name = getattr(args, "profile", None)
        self.jsonl_path = getattr(args, "jsonl", None)
        self.notify_enabled = bool(getattr(args, "notify", False))
        self.two_pass_enabled = bool(getattr(args, "two_pass", False))
        self.revisit_margin_hz = float(revisit_margin_hz)
        self.revisit_span_limit_hz = float(revisit_span_limit_hz)
        self._persisted: List[PersistentDetection] = store.load_baseline_detections(baseline_ctx.id)
        self._seen_persistent: set[int] = set()
        self._revisit_tags: List[RevisitTag] = []
        self._tag_counter = 0

    # -----------------
    # Public interface
    # -----------------

    def persist_detection(
        self,
        *,
        cluster: DetectionCluster,
        combined_seg: Segment,
        confidence: float,
        window_ratio: float,
        duration_seconds: float,
        persistence_mode: str,
        service: Optional[str],
        region: Optional[str],
        notes: Optional[str],
    ) -> PersistResult:
        is_new_detection = self._upsert_detection(cluster, combined_seg, confidence)
        occ_ratio = self._lookup_occ_ratio(combined_seg.f_center_hz)
        occ_threshold = float(getattr(self.args, "new_ema_occ", 0.02) or 0.02)
        is_new_flag = bool(is_new_detection or (occ_ratio is not None and occ_ratio < occ_threshold))
        record = {
            "baseline_id": self.baseline_ctx.id,
            "time_utc": utc_now_str(),
            "f_center_hz": combined_seg.f_center_hz,
            "f_low_hz": combined_seg.f_low_hz,
            "f_high_hz": combined_seg.f_high_hz,
            "bandwidth_hz": combined_seg.bandwidth_hz,
            "peak_db": combined_seg.peak_db,
            "noise_db": combined_seg.noise_db,
            "snr_db": combined_seg.snr_db,
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
        return PersistResult(is_new=is_new_flag, occ_ratio=occ_ratio)

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

    def apply_revisit_confirmation(self, tag: RevisitTag, seg: Segment) -> None:
        det = self._find_persistent_by_id(tag.detection_id)
        if det is None:
            return
        seg = self._constrain_revisit_segment(det, seg)
        det.f_low_hz = min(det.f_low_hz, seg.f_low_hz)
        det.f_high_hz = max(det.f_high_hz, seg.f_high_hz)
        det.f_center_hz = int(seg.f_center_hz)
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
            width_hz=max(det.f_high_hz - det.f_low_hz, 0),
        )

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

    def _upsert_detection(self, cluster: DetectionCluster, seg: Segment, confidence: float) -> bool:
        timestamp = utc_now_str()
        self.store.begin()
        try:
            match = self._match_persistent(seg)
            cluster_center_hz = seg.f_center_hz
            if match:
                blended_center = self._blend_centers(
                    match.f_center_hz,
                    match.total_hits,
                    cluster_center_hz,
                    cluster.hits,
                )
                prev_width = max(float(match.f_high_hz - match.f_low_hz), self.bin_hz)
                cluster_width = max(float(cluster.f_high_hz - cluster.f_low_hz), self.bin_hz)
                alpha = 0.25
                target_width = prev_width + alpha * (cluster_width - prev_width)
                width_cap = self.max_detection_width_hz
                if width_cap > 0.0 and target_width > width_cap:
                    target_width = width_cap
                half = target_width / 2.0
                new_low = int(round(blended_center - half))
                new_high = int(round(blended_center + half))
                baseline_low = self.baseline_ctx.freq_start_hz
                baseline_high = self.baseline_ctx.freq_stop_hz
                new_low = max(new_low, baseline_low)
                new_high = min(new_high, baseline_high)
                if new_high <= new_low:
                    min_width = int(max(1.0, self.bin_hz))
                    new_high = min(baseline_high, new_low + min_width)
                    if new_high <= new_low:
                        new_low = max(baseline_low, new_high - min_width)
                match.f_low_hz = new_low
                match.f_high_hz = new_high
                match.f_center_hz = blended_center
                match.last_seen_utc = timestamp
                match.total_hits += cluster.hits
                match.total_windows += len(cluster.windows)
                match.confidence = confidence
                self.store.update_baseline_detection(match)
                is_new = False
            else:
                detection_id = self.store.insert_baseline_detection(
                    self.baseline_ctx.id,
                    cluster.f_low_hz,
                    cluster.f_high_hz,
                    cluster_center_hz,
                    cluster.first_seen_ts,
                    cluster.last_seen_ts,
                    cluster.hits,
                    len(cluster.windows),
                    confidence,
                )
                new_det = PersistentDetection(
                    id=detection_id,
                    baseline_id=self.baseline_ctx.id,
                    f_low_hz=cluster.f_low_hz,
                    f_high_hz=cluster.f_high_hz,
                    f_center_hz=cluster_center_hz,
                    first_seen_utc=cluster.first_seen_ts,
                    last_seen_utc=cluster.last_seen_ts,
                    total_hits=cluster.hits,
                    total_windows=len(cluster.windows),
                    confidence=confidence,
                )
                self._persisted.append(new_det)
                self._seen_persistent.add(detection_id)
                is_new = True
                if self.two_pass_enabled:
                    self._schedule_revisit(detection_id=detection_id, seg=seg, reason="new")
            self.store.commit()
            return is_new
        except Exception:
            self.store.rollback()
            raise

    def _match_persistent(self, seg: Segment) -> Optional[PersistentDetection]:
        for det in self._persisted:
            spans_overlap = not (
                seg.f_high_hz < (det.f_low_hz - self.freq_merge_hz)
                or seg.f_low_hz > (det.f_high_hz + self.freq_merge_hz)
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
                    continue
                self._seen_persistent.add(det.id)
                if det.missing_since_utc:
                    det.missing_since_utc = None
                    self.store.clear_detection_missing(det.id, det.baseline_id)
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
                return det
        self._log(
            "persist_no_match",
            baseline_id=self.baseline_ctx.id,
            center_hz=seg.f_center_hz,
            width_hz=max(seg.bandwidth_hz, 0.0),
        )
        return None

    def _lookup_occ_ratio(self, freq_hz: int) -> Optional[float]:
        bin_index = self._bin_index_for_freq(freq_hz)
        if bin_index is None:
            return None
        return self.store.baseline_occ_ratio(self.baseline_ctx.id, bin_index)

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

    def _blend_centers(self, center_a: int, weight_a: int, center_b: int, weight_b: int) -> int:
        wa = max(1, int(weight_a))
        wb = max(1, int(weight_b))
        return int(round((center_a * wa + center_b * wb) / float(wa + wb)))

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
        if not self.logger:
            return
        payload = dict(fields)
        if self.profile_name:
            payload.setdefault("profile", self.profile_name)
        self.logger.log(event, **payload)
