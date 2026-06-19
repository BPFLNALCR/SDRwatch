"""Detection engine module coordinating segments with persistent baselines."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

import numpy as np

from sdrwatch.baseline.persistence import BaselinePersistence, CharacterizationSnapshot
from sdrwatch.baseline.spur import SpurEvaluator
from sdrwatch.baseline.store import BaselineContext, Store
from sdrwatch.detection.span_policy import SignalSpanPolicy, resolve_signal_span_policy
from sdrwatch.detection.types import (
    CharacterizationEvidence,
    CharacterizationSpan,
    CrossSweepCandidateState,
    CrossSweepObservation,
    DetectionCluster,
    RevisitTag,
    Segment,
)
from sdrwatch.util.detection_diagnostics import build_characterization_record
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
        try:
            raw_min_sweep_loops = int(getattr(args, "persistence_min_sweep_loops", 1) or 1)
        except Exception:
            raw_min_sweep_loops = 1
        self.persistence_min_sweep_loops = max(1, raw_min_sweep_loops)
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
        raw_center_match = getattr(args, "center_match_hz", None)
        try:
            center_match_override = float(raw_center_match) if raw_center_match not in (None, "") else None
        except Exception:
            center_match_override = None
        if center_match_override is not None and center_match_override > 0.0:
            self.center_match_hz = float(center_match_override)
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
        self._cross_sweep_candidates: List[CrossSweepCandidateState] = []
        self._cross_sweep_max_age_loops = max(5, self.persistence_min_sweep_loops * 3)
        self._cross_sweep_max_candidates = max(32, int(getattr(args, "cross_sweep_max_candidates", 256) or 256))
        self._current_sweep_loop_id = 1
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
        self.signal_span_policy: SignalSpanPolicy = resolve_signal_span_policy(args)
        # Span shaping is split into:
        # - match span: what is persisted/matched in baseline_detections
        # - display span: what is emitted/logged for humans
        legacy_pad = float(getattr(args, "bandwidth_pad_hz", 0.0) or 0.0)
        legacy_min = float(getattr(args, "min_emit_bandwidth_hz", 0.0) or 0.0)
        self.match_bandwidth_pad_hz = max(
            0.0,
            float(getattr(args, "match_bandwidth_pad_hz", None) or legacy_pad or 0.0),
        )
        self.min_match_bandwidth_hz = max(
            0.0,
            float(self.signal_span_policy.min_identity_bandwidth_hz or legacy_min or 0.0),
        )
        self.display_bandwidth_pad_hz = max(
            0.0,
            float(getattr(args, "display_bandwidth_pad_hz", None) or legacy_pad or 0.0),
        )
        self.min_display_bandwidth_hz = max(
            0.0,
            float(self.signal_span_policy.min_display_bandwidth_hz or legacy_min or 0.0),
        )
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
            signal_span_policy=self.signal_span_policy,
        )

    def _log(self, event: str, **fields: Any) -> None:
        if not self.logger:
            return
        payload = dict(fields)
        if self.profile_name:
            payload.setdefault("profile", self.profile_name)
        self.logger.log(event, **payload)

    def ingest(
        self,
        window_idx: int,
        segments: List[Segment],
        *,
        sweep_loop_id: Optional[int] = None,
    ) -> Tuple[int, int, int, int]:
        self._last_window_idx = max(self._last_window_idx, window_idx)
        if sweep_loop_id is not None:
            try:
                self._current_sweep_loop_id = max(1, int(sweep_loop_id))
            except Exception:
                self._current_sweep_loop_id = 1
        accepted = 0
        spur_ignored = 0
        if not segments:
            self._prune_clusters(window_idx)
            self._prune_cross_sweep_candidates(self._current_sweep_loop_id)
            emitted, new_emitted = self._drain_pending_emits()
            return accepted, spur_ignored, emitted, new_emitted
        timestamp = utc_now_str()
        for seg in segments:
            if self._spur_should_ignore(seg):
                spur_ignored += 1
                continue
            self._record_hit(window_idx, seg, timestamp, self._current_sweep_loop_id)
            accepted += 1
        self._prune_clusters(window_idx)
        self._prune_cross_sweep_candidates(self._current_sweep_loop_id)
        emitted, new_emitted = self._drain_pending_emits()
        return accepted, spur_ignored, emitted, new_emitted

    def flush(self) -> Tuple[int, int]:
        self._prune_clusters(self._last_window_idx if self._last_window_idx >= 0 else 0, force=True)
        return self._drain_pending_emits()

    def _record_hit(self, window_idx: int, seg: Segment, timestamp: str, sweep_loop_id: int):
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
                sweep_loop_ids={int(sweep_loop_id)},
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
            cluster.sweep_loop_ids.add(int(sweep_loop_id))
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

    def _shape_span(
        self,
        center_hz: int,
        raw_low: int,
        raw_high: int,
        *,
        stage: str,
        pad_hz: float,
        min_bw_hz: float,
    ) -> Tuple[int, int]:
        # Ensure the emitted center is within the active baseline span.
        # When centroiding spans beyond the scan edges, the raw center can land
        # slightly outside the configured sweep range; clamping avoids emitting
        # out-of-band centers with nonsensical low/high bounds.
        original_center_hz = int(center_hz)
        center_hz = int(
            min(
                max(int(center_hz), int(self.baseline_ctx.freq_start_hz)),
                int(self.baseline_ctx.freq_stop_hz),
            )
        )
        input_width = max(float(raw_high - raw_low), self.bin_hz)
        width = input_width
        if pad_hz > 0.0:
            width += float(pad_hz) * 2.0
        padded_width = width
        was_floored = False
        if min_bw_hz > 0.0 and width < float(min_bw_hz):
            width = float(min_bw_hz)
            was_floored = True

        # Apply a hard cap to the emitted span if configured. This prevents
        # runaway widths when clusters drift/chain across adjacent segments.
        was_clamped = False
        if self.max_detection_width_hz > 0.0 and width > self.max_detection_width_hz:
            width = self.max_detection_width_hz
            was_clamped = True
        half = width / 2.0
        low = int(round(center_hz - half))
        high = int(round(center_hz + half))
        unclipped_low = low
        unclipped_high = high
        low = max(low, self.baseline_ctx.freq_start_hz)
        high = min(high, self.baseline_ctx.freq_stop_hz)
        if high <= low:
            high = low + int(max(1.0, self.bin_hz))
        output_width = max(float(high - low), self.bin_hz)
        self._log(
            "width_decision",
            stage=stage,
            baseline_id=self.baseline_ctx.id,
            center_hz=center_hz,
            requested_center_hz=original_center_hz,
            raw_low_hz=int(raw_low),
            raw_high_hz=int(raw_high),
            input_width_hz=input_width,
            padded_width_hz=padded_width,
            output_width_hz=output_width,
            pad_hz=float(pad_hz),
            min_width_hz=float(min_bw_hz),
            min_identity_bandwidth_hz=(
                float(self.signal_span_policy.min_identity_bandwidth_hz)
                if stage == "shape_match" and self.signal_span_policy.min_identity_bandwidth_hz is not None
                else None
            ),
            width_floor_applied_hz=(float(min_bw_hz) if was_floored else 0.0),
            max_width_hz=float(self.max_detection_width_hz),
            was_floored=was_floored,
            was_clamped=was_clamped,
            baseline_clipped=bool(low != unclipped_low or high != unclipped_high),
        )
        return low, high

    def _shape_match_span(self, center_hz: int, raw_low: int, raw_high: int) -> Tuple[int, int]:
        return self._shape_span(
            center_hz,
            raw_low,
            raw_high,
            stage="shape_match",
            pad_hz=self.match_bandwidth_pad_hz,
            min_bw_hz=self.min_match_bandwidth_hz,
        )

    def _shape_display_span(self, center_hz: int, raw_low: int, raw_high: int) -> Tuple[int, int]:
        return self._shape_span(
            center_hz,
            raw_low,
            raw_high,
            stage="shape_display",
            pad_hz=self.display_bandwidth_pad_hz,
            min_bw_hz=self.min_display_bandwidth_hz,
        )

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
            if self.persistence_min_sweep_loops > 1 and self._cluster_cross_sweep_eligible(cluster):
                self._record_cross_sweep_observation(cluster)
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
        if self.persistence_min_sweep_loops > 1:
            self._record_cross_sweep_observation(cluster)
            cluster.emitted = True
            return
        self._emit_detection(cluster)

    def _cluster_cross_sweep_eligible(self, cluster: DetectionCluster) -> bool:
        width_hz = float(cluster.f_high_hz - cluster.f_low_hz)
        if width_hz < self.min_width_hz:
            return False
        return cluster.hits >= 1

    def _record_cross_sweep_observation(self, cluster: DetectionCluster) -> None:
        if self.persistence_min_sweep_loops <= 1:
            return
        sweep_loop_id = max(cluster.sweep_loop_ids) if cluster.sweep_loop_ids else self._current_sweep_loop_id
        best_seg = cluster.best_seg
        center_hz = self._cluster_center_hz(cluster)
        raw_low = int(cluster.f_low_hz)
        raw_high = int(cluster.f_high_hz)
        match_low, match_high = self._shape_match_span(center_hz, raw_low, raw_high)
        match_width = max(float(match_high - match_low), float(best_seg.bandwidth_hz), self.bin_hz)
        observed_at = utc_now_str()
        observation = CrossSweepObservation(
            sweep_loop_id=int(sweep_loop_id),
            window_idx=int(cluster.last_window),
            center_hz=int(center_hz),
            raw_low_hz=raw_low,
            raw_high_hz=raw_high,
            raw_bandwidth_hz=max(float(raw_high - raw_low), float(best_seg.bandwidth_hz), self.bin_hz),
            match_low_hz=match_low,
            match_high_hz=match_high,
            match_bandwidth_hz=match_width,
            measured_bandwidth_hz=max(float(raw_high - raw_low), float(best_seg.bandwidth_hz), self.bin_hz),
            source_pass="coarse",
            snr_db=float(best_seg.snr_db),
            peak_db=float(best_seg.peak_db),
            noise_db=float(best_seg.noise_db),
            observed_at_utc=observed_at,
        )
        candidate = self._find_cross_sweep_candidate(observation)
        if candidate is None:
            candidate = CrossSweepCandidateState(
                candidate_id=f"cs-{self.baseline_ctx.id}-{int(center_hz)}",
                baseline_id=self.baseline_ctx.id,
                first_observed_sweep_id=int(sweep_loop_id),
                last_observed_sweep_id=int(sweep_loop_id),
                observation_count=0,
                observed_sweep_ids=set(),
                stable_center_hz=int(center_hz),
                match_low_hz=match_low,
                match_high_hz=match_high,
                match_bandwidth_hz=match_width,
                measured_bandwidth_hz=observation.measured_bandwidth_hz,
                last_raw_segment=best_seg,
                last_observation=observation,
            )
            self._cross_sweep_candidates.append(candidate)
            self._log(
                "persistence_decision",
                action="cross_sweep_no_match",
                baseline_id=self.baseline_ctx.id,
                candidate_id=candidate.candidate_id,
                center_hz=int(center_hz),
                reason="created candidate",
            )

        previous_loop_count = candidate.observation_loop_count
        candidate.observation_count += 1
        if int(sweep_loop_id) not in candidate.observed_sweep_ids:
            candidate.observed_sweep_ids.add(int(sweep_loop_id))
            candidate.last_observed_sweep_id = int(sweep_loop_id)
            loop_count = candidate.observation_loop_count
            if loop_count > 0:
                candidate.stable_center_hz = int(
                    round(
                        (
                            float(candidate.stable_center_hz * max(loop_count - 1, 0))
                            + float(center_hz)
                        )
                        / float(loop_count)
                    )
                )
        candidate.match_low_hz = match_low
        candidate.match_high_hz = match_high
        candidate.match_bandwidth_hz = match_width
        candidate.measured_bandwidth_hz = observation.measured_bandwidth_hz
        candidate.last_raw_segment = best_seg
        candidate.last_observation = observation

        action = "cross_sweep_match" if previous_loop_count > 0 else "cross_sweep_observe"
        self._log(
            "cross_sweep_observation",
            baseline_id=self.baseline_ctx.id,
            candidate_id=candidate.candidate_id,
            action=action,
            sweep_loop_id=int(sweep_loop_id),
            window_idx=int(cluster.last_window),
            center_hz=int(center_hz),
            match_low_hz=match_low,
            match_high_hz=match_high,
            measured_bandwidth_hz=observation.measured_bandwidth_hz,
            observation_count=candidate.observation_count,
            observation_loop_count=candidate.observation_loop_count,
            required_loop_count=self.persistence_min_sweep_loops,
            source_pass="coarse",
        )
        self._log(
            "persistence_decision",
            action=("cross_sweep_match" if previous_loop_count > 0 else "cross_sweep_observe"),
            baseline_id=self.baseline_ctx.id,
            candidate_id=candidate.candidate_id,
            center_hz=int(center_hz),
            observation_count=candidate.observation_count,
            observation_loop_count=candidate.observation_loop_count,
            required_loop_count=self.persistence_min_sweep_loops,
        )
        if candidate.observation_loop_count >= self.persistence_min_sweep_loops:
            self._promote_cross_sweep_candidate(candidate)

    def _find_cross_sweep_candidate(
        self,
        observation: CrossSweepObservation,
    ) -> Optional[CrossSweepCandidateState]:
        for candidate in self._cross_sweep_candidates:
            if candidate.promotion_ready:
                continue
            if self._cross_sweep_candidate_matches(candidate, observation):
                return candidate
        return None

    def _cross_sweep_candidate_matches(
        self,
        candidate: CrossSweepCandidateState,
        observation: CrossSweepObservation,
    ) -> bool:
        center_close = abs(int(observation.center_hz) - int(candidate.stable_center_hz)) <= self.center_match_hz
        spans_overlap = not (
            observation.match_high_hz < (candidate.match_low_hz - self.freq_merge_hz)
            or observation.match_low_hz > (candidate.match_high_hz + self.freq_merge_hz)
        )
        if not (center_close or spans_overlap):
            return False
        previous_width = max(float(candidate.match_bandwidth_hz), self.bin_hz)
        observed_width = max(float(observation.match_bandwidth_hz), self.bin_hz)
        max_ratio = float(self.max_detection_width_ratio)
        if previous_width > 0.0 and observed_width > previous_width * max_ratio:
            return False
        if observed_width > 0.0 and previous_width > observed_width * max_ratio:
            return False
        return True

    def _promote_cross_sweep_candidate(self, candidate: CrossSweepCandidateState) -> None:
        if candidate.promotion_ready:
            return
        candidate.promotion_ready = True
        obs = candidate.last_observation
        raw = candidate.last_raw_segment
        timestamp = obs.observed_at_utc
        cluster = DetectionCluster(
            f_low_hz=int(raw.f_low_hz),
            f_high_hz=int(raw.f_high_hz),
            first_seen_ts=timestamp,
            last_seen_ts=timestamp,
            first_window=int(obs.window_idx),
            last_window=int(obs.window_idx),
            hits=max(candidate.observation_loop_count, candidate.observation_count),
            windows=set(range(candidate.observation_loop_count)),
            sweep_loop_ids=set(candidate.observed_sweep_ids),
            best_seg=raw,
            center_weight_sum=float(candidate.stable_center_hz),
            center_weight_total=1.0,
        )
        self._log(
            "persistence_decision",
            action="cross_sweep_promote",
            baseline_id=self.baseline_ctx.id,
            candidate_id=candidate.candidate_id,
            observation_count=candidate.observation_count,
            observation_loop_count=candidate.observation_loop_count,
            required_loop_count=self.persistence_min_sweep_loops,
            center_hz=int(candidate.stable_center_hz),
            match_width_hz=float(candidate.match_bandwidth_hz),
        )
        self._emit_detection(cluster)
        self._cross_sweep_candidates = [
            item for item in self._cross_sweep_candidates if item.candidate_id != candidate.candidate_id
        ]

    def _prune_cross_sweep_candidates(self, current_sweep_loop_id: int) -> None:
        if self.persistence_min_sweep_loops <= 1:
            return
        kept: List[CrossSweepCandidateState] = []
        for candidate in self._cross_sweep_candidates:
            age = int(current_sweep_loop_id) - int(candidate.last_observed_sweep_id)
            if age > self._cross_sweep_max_age_loops:
                candidate.rejection_reason = "expired before required loop count"
                self._log(
                    "persistence_decision",
                    action="cross_sweep_reject",
                    baseline_id=self.baseline_ctx.id,
                    candidate_id=candidate.candidate_id,
                    reason=candidate.rejection_reason,
                    observation_count=candidate.observation_count,
                    observation_loop_count=candidate.observation_loop_count,
                    required_loop_count=self.persistence_min_sweep_loops,
                )
                continue
            kept.append(candidate)
        kept.sort(key=lambda item: item.last_observed_sweep_id, reverse=True)
        overflow = kept[self._cross_sweep_max_candidates :]
        for candidate in overflow:
            self._log(
                "persistence_decision",
                action="cross_sweep_reject",
                baseline_id=self.baseline_ctx.id,
                candidate_id=candidate.candidate_id,
                reason="candidate limit exceeded",
                observation_count=candidate.observation_count,
                observation_loop_count=candidate.observation_loop_count,
                required_loop_count=self.persistence_min_sweep_loops,
            )
        self._cross_sweep_candidates = kept[: self._cross_sweep_max_candidates]

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
        raw_low = int(cluster.f_low_hz)
        raw_high = int(cluster.f_high_hz)

        match_low, match_high = self._shape_match_span(cluster_center_hz, raw_low, raw_high)
        match_width = max(float(match_high - match_low), float(best_seg.bandwidth_hz), self.bin_hz)
        match_seg = Segment(
            f_low_hz=match_low,
            f_high_hz=match_high,
            f_center_hz=cluster_center_hz,
            peak_db=best_seg.peak_db,
            noise_db=best_seg.noise_db,
            snr_db=best_seg.snr_db,
            bandwidth_hz=match_width,
        )
        prepersist_display_low, prepersist_display_high = self._shape_display_span(cluster_center_hz, raw_low, raw_high)
        prepersist_display_width = max(
            float(prepersist_display_high - prepersist_display_low),
            float(best_seg.bandwidth_hz),
            self.bin_hz,
        )
        prepersist_display_seg = Segment(
            f_low_hz=prepersist_display_low,
            f_high_hz=prepersist_display_high,
            f_center_hz=cluster_center_hz,
            peak_db=best_seg.peak_db,
            noise_db=best_seg.noise_db,
            snr_db=best_seg.snr_db,
            bandwidth_hz=prepersist_display_width,
        )
        svc, reg, note = self.bandplan.lookup(cluster_center_hz)

        # Persist using the match span, but emit/log using the display span.
        # Important: keep the live cluster extent tight so emitted clusters do
        # not absorb neighbors in subsequent windows.
        try:
            cluster.f_low_hz = match_low
            cluster.f_high_hz = match_high
            persist_result = self.persistence.persist_detection(
                cluster=cluster,
                combined_seg=match_seg,
                emit_seg=prepersist_display_seg,
                confidence=confidence,
                window_ratio=window_ratio,
                duration_seconds=duration_seconds,
                persistence_mode=self.persistence_mode,
                service=svc,
                region=reg,
                notes=note,
            )
        finally:
            cluster.f_low_hz = raw_low
            cluster.f_high_hz = raw_high

        persisted_detection = persist_result.detection
        characterization = persist_result.characterization
        stable_center_hz = (
            characterization.stable_center_hz
            if characterization is not None
            else (persisted_detection.f_center_hz if persisted_detection else cluster_center_hz)
        )
        match_center_hz = persisted_detection.f_center_hz if persisted_detection else cluster_center_hz
        match_span = CharacterizationSpan.from_bounds(
            low_hz=(persisted_detection.f_low_hz if persisted_detection else match_low),
            high_hz=(persisted_detection.f_high_hz if persisted_detection else match_high),
            center_hz=match_center_hz,
            bandwidth_hz=float(
                (persisted_detection.f_high_hz - persisted_detection.f_low_hz)
                if persisted_detection
                else match_width
            ),
            min_bandwidth_hz=self.bin_hz,
        )
        display_low, display_high = self._shape_display_span(stable_center_hz, raw_low, raw_high)
        display_width = max(float(display_high - display_low), float(best_seg.bandwidth_hz), self.bin_hz)
        display_seg = Segment(
            f_low_hz=display_low,
            f_high_hz=display_high,
            f_center_hz=stable_center_hz,
            peak_db=best_seg.peak_db,
            noise_db=best_seg.noise_db,
            snr_db=best_seg.snr_db,
            bandwidth_hz=display_width,
        )
        self._emit_characterization_record(
            source_pass="coarse",
            persisted_detection=persisted_detection,
            raw_segment=best_seg,
            measured_center_hz=cluster_center_hz,
            measured_low_hz=raw_low,
            measured_high_hz=raw_high,
            measured_bandwidth_hz=max(float(raw_high - raw_low), float(best_seg.bandwidth_hz), self.bin_hz),
            match_span=match_span,
            display_span=CharacterizationSpan.from_bounds(
                low_hz=display_low,
                high_hz=display_high,
                center_hz=display_seg.f_center_hz,
                bandwidth_hz=float(display_width),
                min_bandwidth_hz=self.bin_hz,
            ),
            peak_db=float(best_seg.peak_db),
            noise_db=float(best_seg.noise_db),
            snr_db=float(best_seg.snr_db),
            base_confidence=float(confidence),
            characterization_method="coarse_cluster_span",
            characterization=characterization,
            evidence_sources=["coarse_cluster"],
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
            center_hz=display_seg.f_center_hz,
            width_hz=display_seg.bandwidth_hz,
            snr_db=display_seg.snr_db,
            peak_db=display_seg.peak_db,
            noise_db=display_seg.noise_db,
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

    def _characterization_confidence(
        self,
        *,
        base_confidence: float,
        characterization: Optional[CharacterizationSnapshot],
    ) -> float:
        revisit_count = characterization.revisit_measurement_count if characterization is not None else 0
        revisit_boost = min(0.15, 0.05 * revisit_count)
        return float(np.clip(base_confidence + revisit_boost, 0.0, 1.0))

    def _emit_characterization_record(
        self,
        *,
        source_pass: str,
        persisted_detection,
        raw_segment: Segment,
        measured_center_hz: int,
        measured_low_hz: int,
        measured_high_hz: int,
        measured_bandwidth_hz: float,
        match_span: CharacterizationSpan,
        display_span: CharacterizationSpan,
        peak_db: float,
        noise_db: float,
        snr_db: float,
        base_confidence: float,
        characterization_method: str,
        characterization: Optional[CharacterizationSnapshot],
        evidence_sources: List[str],
        service: Optional[str],
        region: Optional[str],
        notes: Optional[str],
    ) -> None:
        stable_center_hz = (
            characterization.stable_center_hz if characterization is not None else int(display_span.center_hz)
        )
        confidence = self._characterization_confidence(
            base_confidence=base_confidence,
            characterization=characterization,
        )
        characterization_record = build_characterization_record(
            evidence=CharacterizationEvidence(
                detection_id=(persisted_detection.id if persisted_detection else None),
                baseline_id=self.baseline_ctx.id,
                source_pass=source_pass,
                raw_segment=CharacterizationSpan.from_bounds(
                    low_hz=raw_segment.f_low_hz,
                    high_hz=raw_segment.f_high_hz,
                    center_hz=raw_segment.f_center_hz,
                    bandwidth_hz=float(raw_segment.bandwidth_hz),
                    min_bandwidth_hz=self.bin_hz,
                ),
                measured_span=CharacterizationSpan.from_bounds(
                    low_hz=measured_low_hz,
                    high_hz=measured_high_hz,
                    center_hz=measured_center_hz,
                    bandwidth_hz=float(measured_bandwidth_hz),
                    min_bandwidth_hz=self.bin_hz,
                ),
                match_span=match_span,
                display_span=display_span,
                stable_center_hz=int(stable_center_hz),
                center_delta_hz=(
                    characterization.center_delta_hz
                    if characterization is not None
                    else int(measured_center_hz - stable_center_hz)
                ),
                peak_db=float(peak_db),
                noise_db=float(noise_db),
                snr_db=float(snr_db),
                measured_bandwidth_confidence=confidence,
                characterization_confidence=confidence,
                characterization_method=characterization_method,
                center_stability_hz=(
                    characterization.center_stability_hz if characterization is not None else 0.0
                ),
                bandwidth_stability_hz=(
                    characterization.bandwidth_stability_hz if characterization is not None else 0.0
                ),
                revisit_measurement_count=(
                    characterization.revisit_measurement_count if characterization is not None else 0
                ),
                coarse_measurement_count=(
                    characterization.coarse_measurement_count if characterization is not None else 1
                ),
                classification_candidate="unknown",
                classification_evidence=[],
                evidence_sources=evidence_sources,
                bandplan_service=service or None,
                bandplan_region=region or None,
                bandplan_notes=notes or None,
                profile_context=self.profile_name,
                context_only=False,
                bandwidth_interpretation=self.signal_span_policy.raw_fragment_interpretation,
                width_floor_applied_hz=self._identity_floor_applied_hz(measured_bandwidth_hz, match_span),
                persist_width_floor_applied_hz=self._persist_floor_applied_hz(
                    measured_bandwidth_hz,
                    persisted_detection,
                ),
                persisted_card_bandwidth_hz=(
                    float(persisted_detection.f_high_hz - persisted_detection.f_low_hz)
                    if persisted_detection
                    else match_span.bandwidth_hz
                ),
                baseline_clipped=self._persisted_span_is_clipped(persisted_detection),
                clip_reason=(
                    "scan_edge" if self._persisted_span_is_clipped(persisted_detection) else None
                ),
            )
        )
        char_event = str(characterization_record.pop("event", "characterization_record"))
        self._log(char_event, **characterization_record)

    def _identity_floor_applied_hz(self, measured_bandwidth_hz: float, match_span: CharacterizationSpan) -> float:
        floor = float(self.signal_span_policy.min_identity_bandwidth_hz or 0.0)
        if floor <= 0.0:
            return 0.0
        measured = max(float(measured_bandwidth_hz), self.bin_hz)
        if measured >= floor:
            return 0.0
        if match_span.bandwidth_hz < floor:
            return 0.0
        return float(floor - measured)

    def _persist_floor_applied_hz(self, measured_bandwidth_hz: float, persisted_detection) -> float:
        floor = float(self.signal_span_policy.min_persist_bandwidth_hz or 0.0)
        if floor <= 0.0:
            return 0.0
        measured = max(float(measured_bandwidth_hz), self.bin_hz)
        if measured >= floor:
            return 0.0
        if persisted_detection and (persisted_detection.f_high_hz - persisted_detection.f_low_hz) < floor:
            return 0.0
        return float(floor - measured)

    def _persisted_span_is_clipped(self, persisted_detection) -> bool:
        if not persisted_detection:
            return False
        floor = float(self.signal_span_policy.min_persist_bandwidth_hz or 0.0)
        if floor <= 0.0:
            return False
        width = float(persisted_detection.f_high_hz - persisted_detection.f_low_hz)
        if width >= floor:
            return False
        return (
            int(persisted_detection.f_low_hz) <= int(self.baseline_ctx.freq_start_hz)
            or int(persisted_detection.f_high_hz) >= int(self.baseline_ctx.freq_stop_hz)
        )


    def finalize_coarse_pass(self) -> List[RevisitTag]:
        return self.persistence.finalize_coarse_pass()

    def apply_revisit_confirmation(self, tag: RevisitTag, seg: Segment) -> None:
        result = self.persistence.apply_revisit_confirmation(tag, seg)
        if result is None:
            return
        det = result.detection
        characterization = result.characterization
        display_low, display_high = self._shape_display_span(det.f_center_hz, seg.f_low_hz, seg.f_high_hz)
        display_width = max(float(display_high - display_low), float(seg.bandwidth_hz), self.bin_hz)
        match_span = CharacterizationSpan.from_bounds(
            low_hz=det.f_low_hz,
            high_hz=det.f_high_hz,
            center_hz=det.f_center_hz,
            bandwidth_hz=float(det.f_high_hz - det.f_low_hz),
            min_bandwidth_hz=self.bin_hz,
        )
        self._emit_characterization_record(
            source_pass="revisit",
            persisted_detection=det,
            raw_segment=seg,
            measured_center_hz=int(seg.f_center_hz),
            measured_low_hz=int(seg.f_low_hz),
            measured_high_hz=int(seg.f_high_hz),
            measured_bandwidth_hz=max(float(seg.f_high_hz - seg.f_low_hz), float(seg.bandwidth_hz), self.bin_hz),
            match_span=match_span,
            display_span=CharacterizationSpan.from_bounds(
                low_hz=display_low,
                high_hz=display_high,
                center_hz=det.f_center_hz,
                bandwidth_hz=float(display_width),
                min_bandwidth_hz=self.bin_hz,
            ),
            peak_db=float(seg.peak_db),
            noise_db=float(seg.noise_db),
            snr_db=float(seg.snr_db),
            base_confidence=float(det.confidence),
            characterization_method="revisit_refinement",
            characterization=characterization,
            evidence_sources=["coarse_cluster", "revisit_confirmation"],
            service=det.service,
            region=det.region,
            notes=det.bandplan_notes,
        )

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
