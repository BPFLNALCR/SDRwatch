"""Sweep orchestration helpers tying drivers, DSP, and detection together."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional

import numpy as np  # type: ignore

from sdrwatch.baseline.events import BaselineEventWriter
from sdrwatch.baseline.spur import SpurCalibrationTracker
from sdrwatch.baseline.stats import BaselineStatsUpdater
from sdrwatch.baseline.store import BaselineContext, Store
from sdrwatch.detection.engine import DetectionEngine
from sdrwatch.detection.types import RevisitTag, Segment
from sdrwatch.dsp.detection import detect_segments
from sdrwatch.dsp.fft import compute_psd_db
from sdrwatch.dsp.noise_estimation import robust_noise_floor_db
from sdrwatch.dsp.power_monitor import WindowPowerMonitor
from sdrwatch.io.bandplan import Bandplan
from sdrwatch.sweep.scheduler import WindowScheduler
from sdrwatch.util.scan_logger import ScanLogger


def _select_revisit_segment(tag: RevisitTag, segments: List[Segment]) -> Optional[Segment]:
    for seg in segments:
        if seg.f_low_hz <= tag.f_center_hz <= seg.f_high_hz:
            return seg
        if tag.f_low_hz <= seg.f_center_hz <= tag.f_high_hz:
            return seg
    return None


def _run_revisit_pass(
    args,
    src,
    detection_engine: DetectionEngine,
    tags: List[RevisitTag],
    logger: Optional[ScanLogger] = None,
) -> Dict[str, int]:
    stats = {"total": len(tags), "confirmed": 0, "false_positive": 0}
    if not tags:
        return stats

    revisit_fft = int(getattr(args, "revisit_fft", 0) or max(int(args.fft), int(args.fft * 2)))
    revisit_avg = int(getattr(args, "revisit_avg", 0) or max(int(args.avg), 4))
    revisit_threshold = float(getattr(args, "revisit_floor_threshold_db", args.threshold_db))
    revisit_guard = int(getattr(args, "guard_bins", 1))
    revisit_min_width_bins = int(max(1, getattr(args, "min_width_bins", 2)))
    raw_margin = getattr(args, "revisit_margin_hz", None)
    if raw_margin is None or float(raw_margin) <= 0.0:
        revisit_margin_hz = float(getattr(detection_engine, "revisit_margin_hz", 0.0))
    else:
        revisit_margin_hz = float(raw_margin)
    revisit_params = {
        "fft": revisit_fft,
        "avg": revisit_avg,
        "threshold_db": revisit_threshold,
        "guard_bins": revisit_guard,
        "min_width_bins": revisit_min_width_bins,
        "margin_hz": revisit_margin_hz,
        "samp_rate_hz": args.samp_rate,
        "two_pass": bool(getattr(args, "two_pass", False)),
        "max_bands": int(getattr(args, "revisit_max_bands", 0) or 0),
    }
    if logger:
        logger.log(
            "revisit_start",
            baseline_id=detection_engine.baseline_ctx.id,
            tag_count=len(tags),
            revisit_params=revisit_params,
        )

    for tag in tags:
        print(f"[revisit] tag={tag.tag_id} reason={tag.reason} center={tag.f_center_hz/1e6:.6f}MHz", flush=True)
        try:
            src.tune(tag.f_center_hz)
            _ = src.read(int(revisit_fft))
            samples = src.read(int(revisit_fft * revisit_avg))
        except Exception as exc:
            print(f"[revisit] tag={tag.tag_id} tune_error={exc}", file=sys.stderr)
            detection_engine.apply_revisit_miss(tag)
            if tag.reason == "new":
                stats["false_positive"] += 1
            if logger:
                logger.log(
                    "revisit_result",
                    tag_id=tag.tag_id,
                    reason=tag.reason,
                    center_hz=tag.f_center_hz,
                    coarse_width_hz=max(tag.f_high_hz - tag.f_low_hz, 0),
                    error=str(exc),
                    matched=False,
                )
            continue

        baseband_f, psd_db = compute_psd_db(samples, args.samp_rate, revisit_fft, revisit_avg)
        rf_freqs = baseband_f + tag.f_center_hz
        segs, _, _ = detect_segments(
            rf_freqs,
            psd_db,
            thresh_db=revisit_threshold,
            guard_bins=revisit_guard,
            min_width_bins=revisit_min_width_bins,
            cfar_mode=args.cfar,
            cfar_train=args.cfar_train,
            cfar_guard=args.cfar_guard,
            cfar_quantile=args.cfar_quantile,
            cfar_alpha_db=args.cfar_alpha_db,
            abs_power_floor_db=getattr(args, "abs_power_floor_db", None),
        )
        match = _select_revisit_segment(tag, segs)
        if match:
            detection_engine.apply_revisit_confirmation(tag, match)
            stats["confirmed"] += 1
        else:
            detection_engine.apply_revisit_miss(tag)
            if tag.reason == "new":
                stats["false_positive"] += 1
        if logger:
            measured_width = float(match.bandwidth_hz) if match else None
            logger.log(
                "revisit_result",
                tag_id=tag.tag_id,
                reason=tag.reason,
                center_hz=tag.f_center_hz,
                coarse_width_hz=max(tag.f_high_hz - tag.f_low_hz, 0),
                matched=bool(match),
                measured_width_hz=measured_width,
                segment_count=len(segs),
                strongest_snr_db=(max(seg.snr_db for seg in segs) if segs else None),
            )
    if logger:
        logger.log(
            "revisit_summary",
            total=stats.get("total", 0),
            confirmed=stats.get("confirmed", 0),
            false_positive=stats.get("false_positive", 0),
        )
    return stats


class Sweeper:
    """Orchestrate a single sweep across scheduled windows."""

    def __init__(
        self,
        args,
        store: Store,
        bandplan: Bandplan,
        baseline_ctx: BaselineContext,
        logger: Optional[ScanLogger] = None,
    ) -> None:
        self.args = args
        self.store = store
        self.bandplan = bandplan
        self.baseline_ctx = baseline_ctx
        self.logger = logger

    def _sweep_params(self) -> Dict[str, Any]:
        args = self.args
        return {
            "start_hz": args.start,
            "stop_hz": args.stop,
            "step_hz": args.step,
            "samp_rate_hz": args.samp_rate,
            "fft": args.fft,
            "avg": args.avg,
            "threshold_db": args.threshold_db,
            "guard_bins": args.guard_bins,
            "min_width_bins": args.min_width_bins,
            "cfar_mode": args.cfar,
            "cfar_train": args.cfar_train,
            "cfar_guard": args.cfar_guard,
            "cfar_quantile": args.cfar_quantile,
            "cfar_alpha_db": args.cfar_alpha_db,
            "gain": args.gain,
            "driver": args.driver,
            "profile": getattr(args, "profile", None),
            "spur_calibration": bool(args.spur_calibration),
            "two_pass": bool(getattr(args, "two_pass", False)),
            "persistence_mode": getattr(args, "persistence_mode", None),
            "persistence_hit_ratio": getattr(args, "persistence_hit_ratio", None),
            "persistence_min_seconds": getattr(args, "persistence_min_seconds", None),
            "persistence_min_hits": getattr(args, "persistence_min_hits", None),
            "persistence_min_windows": getattr(args, "persistence_min_windows", None),
            "bandwidth_pad_hz": getattr(args, "bandwidth_pad_hz", None),
            "min_emit_bandwidth_hz": getattr(args, "min_emit_bandwidth_hz", None),
            "confidence_hit_normalizer": getattr(args, "confidence_hit_normalizer", None),
            "confidence_duration_norm": getattr(args, "confidence_duration_norm", None),
            "confidence_bias": getattr(args, "confidence_bias", None),
            "revisit_span_limit_hz": getattr(args, "revisit_span_limit_hz", None),
            "cluster_merge_hz": getattr(args, "cluster_merge_hz", None),
            "max_detection_width_ratio": getattr(args, "max_detection_width_ratio", None),
            "max_detection_width_hz": getattr(args, "max_detection_width_hz", None),
        }

    def run(self, src, sweep_seq: int) -> None:
        """Perform a single wideband sweep and update baseline state."""

        args = self.args
        store = self.store
        bandplan = self.bandplan
        baseline_ctx = self.baseline_ctx
        logger = self.logger

        scheduler = WindowScheduler(args.start, args.stop, args.step)
        power_monitor = WindowPowerMonitor()
        spur_tracker = SpurCalibrationTracker() if args.spur_calibration else None
        bin_hz = float(args.samp_rate) / float(args.fft) if args.fft else float(args.samp_rate)
        stats_updater = BaselineStatsUpdater(store, baseline_ctx, sweep_bin_hz=bin_hz)
        event_writer = BaselineEventWriter(store, baseline_ctx, logger)
        if args.spur_calibration:
            detection_engine: Optional[DetectionEngine] = None
        else:
            detection_engine = DetectionEngine(
                store,
                bandplan,
                args,
                bin_hz=bin_hz,
                baseline_ctx=baseline_ctx,
                min_hits=int(getattr(args, "persistence_min_hits", 2)),
                min_windows=int(getattr(args, "persistence_min_windows", 2)),
                logger=logger,
            )

        if logger:
            logger.start_sweep(
                sweep_seq,
                baseline_id=baseline_ctx.id,
                baseline_span_hz=[baseline_ctx.freq_start_hz, baseline_ctx.freq_stop_hz],
                bin_hz=bin_hz,
                params=self._sweep_params(),
            )

        total_segments = 0
        total_hits = 0
        total_new_signals = 0
        total_promoted = 0
        total_revisits = 0
        total_revisit_confirmed = 0
        total_revisit_false = 0
        window_count = 0

        try:
            print(
                f"[scan] begin sweep baseline={baseline_ctx.id} range={args.start/1e6:.3f}-{args.stop/1e6:.3f} MHz step={args.step/1e6:.3f} samp_rate={args.samp_rate/1e6:.3f} fft={args.fft} avg={args.avg}",
                flush=True,
            )
            for window in scheduler:
                center = window.center_hz
                window_idx = window.index
                window_count = window_idx + 1
                src.tune(center)
                nsamps = int(args.fft * args.avg)
                _ = src.read(int(args.fft))
                samples = src.read(nsamps)
                baseband_f, psd_db = compute_psd_db(samples, args.samp_rate, args.fft, args.avg)
                rf_freqs = baseband_f + center

                segs, occ_mask_cfar, noise_per_bin_db = detect_segments(
                    rf_freqs,
                    psd_db,
                    thresh_db=args.threshold_db,
                    guard_bins=args.guard_bins,
                    min_width_bins=args.min_width_bins,
                    cfar_mode=args.cfar,
                    cfar_train=args.cfar_train,
                    cfar_guard=args.cfar_guard,
                    cfar_quantile=args.cfar_quantile,
                    cfar_alpha_db=args.cfar_alpha_db,
                    abs_power_floor_db=getattr(args, "abs_power_floor_db", None),
                )

                if logger:
                    widths = np.array([max(float(seg.bandwidth_hz), 0.0) for seg in segs], dtype=float)
                    avg_bw = float(np.mean(widths)) if widths.size else None
                    min_bw = float(np.min(widths)) if widths.size else None
                    median_bw = float(np.median(widths)) if widths.size else None
                    max_bw = float(np.max(widths)) if widths.size else None
                    strongest_snr = max((seg.snr_db for seg in segs), default=None)
                    logger.log(
                        "segment_inventory",
                        window_idx=window_idx,
                        center_hz=float(center),
                        num_segments=len(segs),
                        min_bandwidth_hz=min_bw,
                        median_bandwidth_hz=median_bw,
                        max_bandwidth_hz=max_bw,
                        strongest_snr_db=strongest_snr,
                    )

                mean_psd_db = float(np.mean(psd_db))
                p90_psd_db = float(np.percentile(psd_db, 90.0))
                is_anom, _ema_power_db, _delta_db = power_monitor.update(mean_psd_db)
                accepted_hits = 0
                spur_ignored = 0
                promoted = 0
                new_signals = 0

                if is_anom:
                    if detection_engine:
                        accepted_hits, spur_ignored, promoted, new_signals = detection_engine.ingest(window_idx, [])
                else:
                    noise_db = robust_noise_floor_db(psd_db)
                    dynamic = noise_db + args.threshold_db
                    occupied_mask = np.asarray(
                        occ_mask_cfar if (args.cfar and args.cfar != "off") else (psd_db > dynamic),
                        dtype=bool,
                    )
                    stats_updater.update_window(rf_freqs, psd_db, noise_per_bin_db, occupied_mask)

                    if spur_tracker is not None:
                        spur_tracker.observe(segs)
                    if detection_engine:
                        accepted_hits, spur_ignored, promoted, new_signals = detection_engine.ingest(window_idx, segs)
                total_segments += len(segs)
                total_hits += accepted_hits
                total_promoted += promoted
                total_new_signals += new_signals

                fields = [
                    f"center_hz={center:.1f}",
                    f"det_count={len(segs)}",
                    f"mean_db={mean_psd_db:.1f}",
                    f"p90_db={p90_psd_db:.1f}",
                    f"anomalous={1 if is_anom else 0}",
                ]
                if detection_engine:
                    fields.extend(
                        [
                            f"accepted={accepted_hits}",
                            f"promoted={promoted}",
                            f"new_sig={new_signals}",
                            f"spur_masked={spur_ignored}",
                        ]
                    )
                print(f"[scan] window {' '.join(fields)}", flush=True)

            revisit_tags: List[RevisitTag] = []
            if detection_engine:
                revisit_tags = detection_engine.finalize_coarse_pass()
            if detection_engine and getattr(args, "two_pass", False) and revisit_tags:
                max_bands = int(getattr(args, "revisit_max_bands", 0) or 0)
                if max_bands > 0:
                    revisit_tags = revisit_tags[:max_bands]
                revisit_stats = _run_revisit_pass(
                    args,
                    src,
                    detection_engine,
                    revisit_tags,
                    logger,
                )
                total_revisits += revisit_stats.get("total", 0)
                total_revisit_confirmed += revisit_stats.get("confirmed", 0)
                total_revisit_false += revisit_stats.get("false_positive", 0)

        finally:
            if detection_engine:
                flushed, new_flush = detection_engine.flush()
                if flushed:
                    total_promoted += flushed
                    total_new_signals += new_flush
                    print(
                        f"[scan] sweep baseline={baseline_ctx.id} flushed pending detections={flushed}",
                        flush=True,
                    )
            if args.spur_calibration and spur_tracker is not None:
                spur_tracker.persist(store, window_count)
            stats_updater.update_span((min(args.start, args.stop), max(args.start, args.stop)))
            event_writer.record_scan_summary(
                hits=total_hits,
                segments=total_segments,
                promoted=total_promoted,
                new_signals=total_new_signals,
                revisits_total=total_revisits,
                revisits_confirmed=total_revisit_confirmed,
                revisits_false_positive=total_revisit_false,
            )
            print(
                f"[scan] end sweep baseline={baseline_ctx.id} hits={total_hits} promoted={total_promoted} new={total_new_signals}",
                flush=True,
            )


def run_sweep(
    args,
    store: Store,
    bandplan: Bandplan,
    src,
    baseline_ctx: BaselineContext,
    sweep_seq: int,
    logger: Optional[ScanLogger] = None,
) -> None:
    """Backwards-compatible helper that instantiates a Sweeper and runs it."""

    Sweeper(args, store, bandplan, baseline_ctx, logger).run(src, sweep_seq)
