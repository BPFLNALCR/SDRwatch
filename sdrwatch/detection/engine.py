"""Detection engine module coordinating segments with persistent baselines."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from sdrwatch.baseline.persistence import BaselinePersistence
from sdrwatch.baseline.store import BaselineContext, Store
from sdrwatch.baseline.spur import SpurEvaluator
from sdrwatch.detection.types import DetectionCluster, PersistentDetection, RevisitTag, Segment
from sdrwatch.util.time import utc_now_str

if TYPE_CHECKING:  # pragma: no cover - type hint only
    from sdrwatch import Bandplan, ScanLogger


class DetectionEngine:
    def __init__(
        self,
        store: Store,
        bandplan: "Bandplan",
        args,
        *,
        bin_hz: float,
        baseline_ctx: BaselineContext,
        min_hits: int = 2,
        min_windows: int = 2,
        max_gap_windows: int = 3,
        freq_merge_hz: Optional[float] = None,
        logger: Optional["ScanLogger"] = None,
    ):
        self.store = store
        self.bandplan = bandplan
        self.args = args
        self.bin_hz = float(bin_hz)
        self.baseline_ctx = baseline_ctx
        self.min_hits = max(1, int(min_hits))
        self.min_windows = max(1, int(min_windows))
        self.max_gap_windows = max(1, int(max_gap_windows))
        merge_override = getattr(args, "cluster_merge_hz", None)
        merge_override_val: Optional[float]
        try:
            merge_override_val = float(merge_override) if merge_override not in (None, "") else None
        except Exception:
            merge_override_val = None
        if merge_override_val is not None and merge_override_val <= 0.0:
            merge_override_val = None
        if merge_override_val is not None:
            freq_merge_val = merge_override_val
        elif freq_merge_hz is not None:
            freq_merge_val = float(freq_merge_hz)
        else:
            freq_merge_val = max(self.bin_hz * 2.0, 25_000.0)
        self.freq_merge_hz = float(freq_merge_val)
        self.center_match_hz = max(self.bin_hz * 2.0, self.freq_merge_hz / 2.0)
        raw_mode = str(getattr(args, "persistence_mode", "hits") or "hits").lower()
        self.persistence_mode = raw_mode if raw_mode in {"hits", "duration", "both"} else "hits"
        raw_ratio = getattr(args, "persistence_hit_ratio", 0.0)
        ratio_val = 0.0 if raw_ratio is None else float(raw_ratio)
        self.persistence_hit_ratio = float(np.clip(ratio_val, 0.0, 1.0))
        raw_duration = getattr(args, "persistence_min_seconds", 0.0)
        self.persistence_min_seconds = float(max(0.0, float(raw_duration if raw_duration is not None else 0.0)))
        self.min_width_hz = max(self.bin_hz * float(args.min_width_bins), self.bin_hz)
        raw_width_ratio = getattr(args, "max_detection_width_ratio", None)
        try:
            width_ratio_val = float(raw_width_ratio if raw_width_ratio is not None else 3.0)
        except Exception:
            width_ratio_val = 3.0
        if width_ratio_val < 1.0:
            width_ratio_val = 1.0
        self.max_detection_width_ratio = width_ratio_val
        raw_width_cap = getattr(args, "max_detection_width_hz", None)
        try:
            width_cap_val = float(raw_width_cap if raw_width_cap is not None else 0.0)
        except Exception:
            width_cap_val = 0.0
        self.max_detection_width_hz = max(0.0, width_cap_val)
        self.clusters: List[DetectionCluster] = []
        self._last_window_idx = -1
        self.spur_tolerance_hz = 5_000.0
        self.spur_margin_db = 4.0
        self.spur_min_hits = 5
        self.spur_override_snr = 10.0
        self.spur_penalty_max = 0.35
        self._pending_emits = 0
        self._pending_new_signals = 0
        self._persisted: List[PersistentDetection] = self.store.load_baseline_detections(self.baseline_ctx.id)
        self._seen_persistent: Set[int] = set()
        self.two_pass_enabled = bool(getattr(args, "two_pass", False))
        self.revisit_margin_hz = float(
            getattr(args, "revisit_margin_hz", max(self.freq_merge_hz, 25_000.0)) or max(self.freq_merge_hz, 25_000.0)
        )
        raw_span_limit = getattr(args, "revisit_span_limit_hz", None)
        try:
            span_limit = float(raw_span_limit) if raw_span_limit not in (None, "") else 0.0
        except Exception:
            span_limit = 0.0
        self.revisit_span_limit_hz = max(0.0, span_limit)
        self._revisit_tags: List[RevisitTag] = []
        self._tag_counter = 0
        self.logger = logger
        self.profile_name = getattr(args, "profile", None)
        self.bandwidth_pad_hz = max(0.0, float(getattr(args, "bandwidth_pad_hz", 0.0) or 0.0))
        self.min_emit_bandwidth_hz = max(0.0, float(getattr(args, "min_emit_bandwidth_hz", 0.0) or 0.0))
        raw_hit_norm = getattr(args, "confidence_hit_normalizer", None)
        raw_duration_norm = getattr(args, "confidence_duration_norm", None)
        raw_bias = getattr(args, "confidence_bias", None)
        self.conf_hit_normalizer = max(1.0, float(raw_hit_norm if raw_hit_norm not in (None, 0) else 6.0))
        self.conf_duration_norm = max(1.0, float(raw_duration_norm if raw_duration_norm not in (None, 0) else 8.0))
        self.confidence_bias = float(raw_bias if raw_bias is not None else 0.0)

    def _log(self, event: str, **fields: Any) -> None:
        if not self.logger:
            return
        payload = dict(fields)
        if self.profile_name:
            payload.setdefault("profile", self.profile_name)
        self.logger.log(event, **payload)

    def ingest(self, window_idx: int, segments: List[Segment]) -> Tuple[int, int, int, int]:
        self._last_window_idx = max(self._last_window_idx, window_idx)
        accepted = 0
        spur_ignored = 0
        if not segments:
            self._prune_clusters(window_idx)
            emitted, new_emitted = self._drain_pending_emits()
            return accepted, spur_ignored, emitted, new_emitted
        timestamp = utc_now_str()
        for seg in segments:
            if self._spur_should_ignore(seg):
                spur_ignored += 1
                continue
            self._record_hit(window_idx, seg, timestamp)
            accepted += 1
        self._prune_clusters(window_idx)
        emitted, new_emitted = self._drain_pending_emits()
        return accepted, spur_ignored, emitted, new_emitted

    def flush(self) -> Tuple[int, int]:
        self._prune_clusters(self._last_window_idx if self._last_window_idx >= 0 else 0, force=True)
        return self._drain_pending_emits()

    def _record_hit(self, window_idx: int, seg: Segment, timestamp: str):
        cluster = self._find_cluster(seg)
        if cluster is None:
            cluster = DetectionCluster(
                f_low_hz=seg.f_low_hz,
                f_high_hz=seg.f_high_hz,
                first_seen_ts=timestamp,
                last_seen_ts=timestamp,
                first_window=window_idx,
                last_window=window_idx,
                hits=1,
                windows={window_idx},
                best_seg=seg,
            )
            self.clusters.append(cluster)
        else:
            cluster.f_low_hz = min(cluster.f_low_hz, seg.f_low_hz)
            cluster.f_high_hz = max(cluster.f_high_hz, seg.f_high_hz)
            cluster.last_seen_ts = timestamp
            cluster.last_window = window_idx
            cluster.hits += 1
            cluster.windows.add(window_idx)
            if seg.snr_db >= cluster.best_seg.snr_db:
                cluster.best_seg = seg

        self._update_cluster_center(cluster, seg)
        self._maybe_emit_cluster(cluster)

    def _segment_weight(self, seg: Segment) -> float:
        try:
            return float(max(1e-3, 10.0 ** (seg.snr_db / 10.0)))
        except Exception:
            return 1.0

    def _update_cluster_center(self, cluster: DetectionCluster, seg: Segment) -> None:
        weight = self._segment_weight(seg)
        cluster.center_weight_sum += weight * float(seg.f_center_hz)
        cluster.center_weight_total += weight

    def _cluster_center_hz(self, cluster: DetectionCluster) -> int:
        if cluster.center_weight_total <= 0.0:
            return int((cluster.f_low_hz + cluster.f_high_hz) / 2)
        return int(round(cluster.center_weight_sum / cluster.center_weight_total))

    def _shape_emit_span(self, center_hz: int, raw_low: int, raw_high: int) -> Tuple[int, int]:
        width = max(float(raw_high - raw_low), self.bin_hz)
        if self.bandwidth_pad_hz > 0.0:
            width += self.bandwidth_pad_hz * 2.0
        min_emit = self.min_emit_bandwidth_hz
        if min_emit > 0.0 and width < min_emit:
            width = min_emit
        half = width / 2.0
        low = int(round(center_hz - half))
        high = int(round(center_hz + half))
        low = max(low, self.baseline_ctx.freq_start_hz)
        high = min(high, self.baseline_ctx.freq_stop_hz)
        if high <= low:
            high = low + int(max(1.0, self.bin_hz))
        return low, high

    def _blend_centers(self, center_a: int, weight_a: int, center_b: int, weight_b: int) -> int:
        wa = max(1, int(weight_a))
        wb = max(1, int(weight_b))
        return int(round((center_a * wa + center_b * wb) / float(wa + wb)))

    def _cluster_window_ratio(self, cluster: DetectionCluster) -> float:
        span_windows = max(cluster.last_window - cluster.first_window + 1, 1)
        return float(len(cluster.windows)) / float(span_windows)

    def _cluster_duration_seconds(self, cluster: DetectionCluster) -> float:
        try:
            t0 = self._parse_timestamp(cluster.first_seen_ts)
            t1 = self._parse_timestamp(cluster.last_seen_ts)
        except Exception:
            return 0.0
        return max(0.0, (t1 - t0).total_seconds())

    def _parse_timestamp(self, text: str) -> datetime:
        if not text:
            raise ValueError("empty timestamp")
        cleaned = text.strip()
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        return datetime.fromisoformat(cleaned)

    def _find_cluster(self, seg: Segment) -> Optional[DetectionCluster]:
        for cluster in self.clusters:
            if self._segments_overlap(cluster, seg):
                return cluster
        return None

    def _segments_overlap(self, cluster: DetectionCluster, seg: Segment) -> bool:
        return not (
            seg.f_high_hz < (cluster.f_low_hz - self.freq_merge_hz)
            or seg.f_low_hz > (cluster.f_high_hz + self.freq_merge_hz)
        )

    def _maybe_emit_cluster(self, cluster: DetectionCluster):
        if cluster.emitted:
            return
        qualifies, reasons = self._cluster_gate_status(cluster)
        if not qualifies:
            best_seg = cluster.best_seg
            self._log(
                "cluster_reject",
                baseline_id=self.baseline_ctx.id,
                center_hz=self._cluster_center_hz(cluster),
                width_hz=max(float(cluster.f_high_hz - cluster.f_low_hz), 0.0),
                hits=cluster.hits,
                windows=len(cluster.windows),
                window_ratio=self._cluster_window_ratio(cluster),
                duration_s=self._cluster_duration_seconds(cluster),
                snr_db=best_seg.snr_db,
                peak_db=best_seg.peak_db,
                noise_db=best_seg.noise_db,
                min_width_hz=self.min_width_hz,
                reasons=reasons,
            )
            return
        self._emit_detection(cluster)

    def _cluster_gate_status(self, cluster: DetectionCluster) -> Tuple[bool, List[str]]:
        reasons: List[str] = []
        width_hz = float(cluster.f_high_hz - cluster.f_low_hz)
        if width_hz < self.min_width_hz:
            reasons.append(f"width={width_hz:.1f} < min_width={self.min_width_hz:.1f}")
        if cluster.hits < self.min_hits:
            reasons.append(f"hits={cluster.hits} < min_hits={self.min_hits}")
        win_count = len(cluster.windows)
        if win_count < self.min_windows:
            reasons.append(f"windows={win_count} < min_windows={self.min_windows}")
        ratio_threshold = float(self.persistence_hit_ratio)
        ratio_value = self._cluster_window_ratio(cluster)
        ratio_ok = True if ratio_threshold <= 0.0 else (ratio_value >= ratio_threshold)
        duration_threshold = float(self.persistence_min_seconds)
        duration_value = self._cluster_duration_seconds(cluster)
        duration_ok = True if duration_threshold <= 0.0 else (duration_value >= duration_threshold)
        mode = self.persistence_mode
        if mode == "duration":
            if not duration_ok:
                reasons.append(
                    f"duration={duration_value:.2f}s < min_duration={duration_threshold:.2f}s"
                )
        elif mode == "both":
            if not ratio_ok:
                reasons.append(
                    f"ratio={ratio_value:.2f} < threshold={ratio_threshold:.2f}"
                )
            if not duration_ok:
                reasons.append(
                    f"duration={duration_value:.2f}s < min_duration={duration_threshold:.2f}s"
                )
        else:  # hits ratio mode
            if not ratio_ok:
                reasons.append(
                    f"ratio={ratio_value:.2f} < threshold={ratio_threshold:.2f}"
                )
        return (len(reasons) == 0, reasons)

    def _cluster_qualifies(self, cluster: DetectionCluster) -> bool:
        qualifies, _ = self._cluster_gate_status(cluster)
        return qualifies

    def _emit_detection(self, cluster: DetectionCluster):
        cluster.emitted = True
        best_seg = cluster.best_seg
        confidence = self._compute_confidence(cluster)
        cluster_center_hz = self._cluster_center_hz(cluster)
        window_ratio = self._cluster_window_ratio(cluster)
        duration_seconds = self._cluster_duration_seconds(cluster)
        emit_low, emit_high = self._shape_emit_span(cluster_center_hz, cluster.f_low_hz, cluster.f_high_hz)
        cluster.f_low_hz = emit_low
        cluster.f_high_hz = emit_high
        span_width = max(float(cluster.f_high_hz - cluster.f_low_hz), float(best_seg.bandwidth_hz), self.bin_hz)
        combined_seg = Segment(
            f_low_hz=cluster.f_low_hz,
            f_high_hz=cluster.f_high_hz,
            f_center_hz=cluster_center_hz,
            peak_db=best_seg.peak_db,
            noise_db=best_seg.noise_db,
            snr_db=best_seg.snr_db,
            bandwidth_hz=span_width,
        )
        svc, reg, note = self.bandplan.lookup(combined_seg.f_center_hz)

        is_new_detection = self._persist_detection(cluster, combined_seg, confidence)
        self._pending_emits += 1
        if is_new_detection:
            self._pending_new_signals += 1

        occ_ratio = self._lookup_occ_ratio(combined_seg.f_center_hz)
        is_new_flag = bool(is_new_detection or (occ_ratio is not None and occ_ratio < self.args.new_ema_occ))

        self._log(
            "cluster_emit",
            baseline_id=self.baseline_ctx.id,
            center_hz=combined_seg.f_center_hz,
            width_hz=combined_seg.bandwidth_hz,
            snr_db=combined_seg.snr_db,
            peak_db=combined_seg.peak_db,
            noise_db=combined_seg.noise_db,
            confidence=confidence,
            hits=cluster.hits,
            windows=len(cluster.windows),
            window_ratio=window_ratio,
            duration_s=duration_seconds,
            is_new=is_new_flag,
            occ_ratio=occ_ratio,
            service=svc,
            region=reg,
        )

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
            "service": svc,
            "region": reg,
            "notes": note,
            "is_new": is_new_flag,
            "confidence": confidence,
            "window_ratio": window_ratio,
            "duration_s": duration_seconds,
            "persistence_mode": self.persistence_mode,
        }
        if self.profile_name:
            record["profile"] = self.profile_name
        maybe_emit_jsonl(self.args.jsonl, record)
        if is_new_flag:
            body = f"{combined_seg.f_center_hz/1e6:.6f} MHz; SNR {combined_seg.snr_db:.1f} dB; {svc or 'Unknown'} {reg or ''}"
            maybe_notify("SDRWatch: New signal", body, self.args.notify)

    def _persist_detection(self, cluster: DetectionCluster, seg: Segment, confidence: float) -> bool:
        timestamp = utc_now_str()
        match = self._match_persistent(seg)
        cluster_center_hz = seg.f_center_hz
        self.store.begin()
        try:
            if match:
                # Clamp width growth so persistent detections do not absorb adjacent stations over time.
                blended_center = self._blend_centers(match.f_center_hz, match.total_hits, cluster_center_hz, cluster.hits)
                prev_width = max(float(match.f_high_hz - match.f_low_hz), self.bin_hz)
                cluster_width = max(float(cluster.f_high_hz - cluster.f_low_hz), self.bin_hz)
                alpha = 0.25
                target_width = prev_width + alpha * (cluster_width - prev_width)
                if self.max_detection_width_hz > 0.0 and target_width > self.max_detection_width_hz:
                    target_width = self.max_detection_width_hz
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
        finally:
            self.store.commit()
        return is_new

    def _match_persistent(self, seg: Segment) -> Optional[PersistentDetection]:
        # Width-aware matching ensures narrow persistents are not polluted by much wider clusters.
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

    def _find_persistent_by_id(self, detection_id: int) -> Optional[PersistentDetection]:
        for det in self._persisted:
            if det.id == detection_id:
                return det
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

    def _constrain_revisit_segment(self, det: Optional[PersistentDetection], seg: Segment) -> Segment:
        limit = float(getattr(self, "revisit_span_limit_hz", 0.0) or 0.0)
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
        seg.bandwidth_hz = max(float(seg.f_high_hz - seg.f_low_hz), self.min_emit_bandwidth_hz, self.bin_hz)
        self._log(
            "revisit_trim",
            detection_id=(det.id if det else None),
            original_width_hz=span_width,
            trimmed_width_hz=float(seg.bandwidth_hz),
            anchor_hz=anchor,
        )
        return seg

    def _filter_tags(self, tags: List[RevisitTag]) -> List[RevisitTag]:
        seen: Set[str] = set()
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
            for det in to_mark:
                self.store.mark_detection_missing(det.id, det.baseline_id, missing_ts)
            self.store.commit()
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
        if not tag.detection_id:
            return
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
        self.store.update_baseline_detection(det)
        self.store.commit()
        self._log(
            "revisit_apply",
            action="confirmed",
            tag_id=tag.tag_id,
            detection_id=det.id,
            center_hz=det.f_center_hz,
            width_hz=max(det.f_high_hz - det.f_low_hz, 0),
        )

    def apply_revisit_miss(self, tag: RevisitTag) -> None:
        if tag.detection_id is None:
            return
        det = self._find_persistent_by_id(tag.detection_id)
        if det is None:
            return
        timestamp = utc_now_str()
        if tag.reason == "new":
            self.store.begin()
            self.store.delete_baseline_detection(det.id, det.baseline_id)
            self.store.commit()
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
        self.store.mark_detection_missing(det.id, det.baseline_id, timestamp)
        self.store.commit()
        self._log(
            "revisit_apply",
            action="marked_missing",
            tag_id=tag.tag_id,
            detection_id=det.id,
            reason=tag.reason,
            center_hz=det.f_center_hz,
        )

    def _prune_clusters(self, window_idx: int, force: bool = False):
        to_remove: List[DetectionCluster] = []
        for cluster in self.clusters:
            gap = window_idx - cluster.last_window
            if force or gap > self.max_gap_windows:
                if not cluster.emitted and self._cluster_qualifies(cluster):
                    self._emit_detection(cluster)
                to_remove.append(cluster)
        for cluster in to_remove:
            self.clusters.remove(cluster)

    def _spur_should_ignore(self, seg: Segment) -> bool:
        if getattr(self.args, "spur_calibration", False):
            return False
        spur = self.store.lookup_spur(seg.f_center_hz, int(self.spur_tolerance_hz))
        if not spur:
            return False
        _, mean_power_db, hits = spur
        if hits < self.spur_min_hits:
            return False
        if seg.peak_db >= mean_power_db + self.spur_margin_db:
            return False
        if seg.snr_db >= self.spur_override_snr:
            return False
        return True

    def _compute_confidence(self, cluster: DetectionCluster) -> float:
        best_seg = cluster.best_seg
        snr_component = float(np.clip(best_seg.snr_db / 30.0, 0.0, 1.0))
        hit_component = float(np.clip(cluster.hits / self.conf_hit_normalizer, 0.0, 1.0))
        span_windows = max(cluster.last_window - cluster.first_window + 1, 1)
        persistence_component = float(np.clip(len(cluster.windows) / span_windows, 0.0, 1.0))
        duration_component = float(np.clip(span_windows / self.conf_duration_norm, 0.0, 1.0))
        raw_score = (
            0.45 * snr_component
            + 0.25 * hit_component
            + 0.2 * persistence_component
            + 0.1 * duration_component
        )
        raw_score += self.confidence_bias
        penalty = self._spur_confidence_penalty(cluster)
        return float(np.clip(raw_score - penalty, 0.0, 1.0))

    def _spur_confidence_penalty(self, cluster: DetectionCluster) -> float:
        if getattr(self.args, "spur_calibration", False):
            return 0.0
        seg = cluster.best_seg
        spur = self.store.lookup_spur(seg.f_center_hz, int(self.spur_tolerance_hz))
        if not spur:
            return 0.0
        _, mean_power_db, hits = spur
        if hits < self.spur_min_hits:
            return 0.0
        diff = float(seg.peak_db - mean_power_db)
        if diff >= self.spur_margin_db + 5.0:
            return 0.05
        if diff >= self.spur_margin_db:
            return min(0.15, self.spur_penalty_max)
        return self.spur_penalty_max

    def _drain_pending_emits(self) -> Tuple[int, int]:
        emitted = self._pending_emits
        new_emitted = self._pending_new_signals
        self._pending_emits = 0
        self._pending_new_signals = 0
        return emitted, new_emitted


def maybe_notify(title: str, body: str, enabled: bool):
    if not enabled:
        return
    try:
        subprocess.Popen(["notify-send", title, body])
    except Exception:
        pass


def maybe_emit_jsonl(path: Optional[str], record: dict):
    if not path:
        return
    try:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass
