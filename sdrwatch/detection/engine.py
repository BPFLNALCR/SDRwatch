"""Detection engine module coordinating segments with persistent baselines."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from sdrwatch.baseline.persistence import BaselinePersistence
from sdrwatch.baseline.spur import SpurEvaluator
from sdrwatch.baseline.store import BaselineContext, Store
from sdrwatch.detection.types import DetectionCluster, RevisitTag, Segment
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
        self.revisit_margin_hz = float(
            getattr(args, "revisit_margin_hz", max(self.freq_merge_hz, 25_000.0)) or max(self.freq_merge_hz, 25_000.0)
        )
        raw_span_limit = getattr(args, "revisit_span_limit_hz", None)
        try:
            span_limit = float(raw_span_limit) if raw_span_limit not in (None, "") else 0.0
        except Exception:
            span_limit = 0.0
        self.revisit_span_limit_hz = max(0.0, span_limit)
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
        self._calibration_mode = bool(getattr(args, "spur_calibration", False))
        self.spur_evaluator = SpurEvaluator(
            store,
            tolerance_hz=self.spur_tolerance_hz,
            margin_db=self.spur_margin_db,
            min_hits=self.spur_min_hits,
            override_snr=self.spur_override_snr,
            penalty_max=self.spur_penalty_max,
        )
        self.persistence = BaselinePersistence(
            store=store,
            baseline_ctx=baseline_ctx,
            args=args,
            bin_hz=self.bin_hz,
            freq_merge_hz=self.freq_merge_hz,
            center_match_hz=self.center_match_hz,
            max_detection_width_ratio=self.max_detection_width_ratio,
            max_detection_width_hz=self.max_detection_width_hz,
            logger=logger,
            revisit_margin_hz=self.revisit_margin_hz,
            revisit_span_limit_hz=self.revisit_span_limit_hz,
        )

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
        # Ensure the emitted center is within the active baseline span.
        # When centroiding spans beyond the scan edges, the raw center can land
        # slightly outside the configured sweep range; clamping avoids emitting
        # out-of-band centers with nonsensical low/high bounds.
        center_hz = int(
            min(
                max(int(center_hz), int(self.baseline_ctx.freq_start_hz)),
                int(self.baseline_ctx.freq_stop_hz),
            )
        )
        width = max(float(raw_high - raw_low), self.bin_hz)
        if self.bandwidth_pad_hz > 0.0:
            width += self.bandwidth_pad_hz * 2.0
        min_emit = self.min_emit_bandwidth_hz
        if min_emit > 0.0 and width < min_emit:
            width = min_emit

        # Apply a hard cap to the emitted span if configured. This prevents
        # runaway widths when clusters drift/chain across adjacent segments.
        if self.max_detection_width_hz > 0.0 and width > self.max_detection_width_hz:
            width = self.max_detection_width_hz
        half = width / 2.0
        low = int(round(center_hz - half))
        high = int(round(center_hz + half))
        low = max(low, self.baseline_ctx.freq_start_hz)
        high = min(high, self.baseline_ctx.freq_stop_hz)
        if high <= low:
            high = low + int(max(1.0, self.bin_hz))
        return low, high

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

        persist_result = self.persistence.persist_detection(
            cluster=cluster,
            combined_seg=combined_seg,
            confidence=confidence,
            window_ratio=window_ratio,
            duration_seconds=duration_seconds,
            persistence_mode=self.persistence_mode,
            service=svc,
            region=reg,
            notes=note,
        )
        self._pending_emits += 1
        if persist_result.is_new:
            self._pending_new_signals += 1

        occ_ratio = persist_result.occ_ratio
        is_new_flag = persist_result.is_new

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


    def finalize_coarse_pass(self) -> List[RevisitTag]:
        return self.persistence.finalize_coarse_pass()

    def apply_revisit_confirmation(self, tag: RevisitTag, seg: Segment) -> None:
        self.persistence.apply_revisit_confirmation(tag, seg)

    def apply_revisit_miss(self, tag: RevisitTag) -> None:
        self.persistence.apply_revisit_miss(tag)

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
        return self.spur_evaluator.should_mask(seg, calibration_mode=self._calibration_mode)

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
        return self.spur_evaluator.confidence_penalty(cluster.best_seg, calibration_mode=self._calibration_mode)

    def _drain_pending_emits(self) -> Tuple[int, int]:
        emitted = self._pending_emits
        new_emitted = self._pending_new_signals
        self._pending_emits = 0
        self._pending_new_signals = 0
        return emitted, new_emitted
