"""Scan profile dataclasses and helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ScanProfile:
    name: str
    f_low_hz: int
    f_high_hz: int
    samp_rate: float
    fft: int
    avg: int
    gain_db: float
    threshold_db: float
    min_width_bins: int
    guard_bins: int
    abs_power_floor_db: Optional[float] = None
    step_hz: Optional[float] = None
    cfar_train: Optional[int] = None
    cfar_guard: Optional[int] = None
    cfar_quantile: Optional[float] = None
    persistence_hit_ratio: Optional[float] = None
    persistence_min_seconds: Optional[float] = None
    persistence_min_hits: Optional[int] = None
    persistence_min_windows: Optional[int] = None
    revisit_fft: Optional[int] = None
    revisit_avg: Optional[int] = None
    revisit_margin_hz: Optional[float] = None
    revisit_max_bands: Optional[int] = None
    revisit_floor_threshold_db: Optional[float] = None
    two_pass: Optional[bool] = None
    bandwidth_pad_hz: Optional[float] = None
    min_emit_bandwidth_hz: Optional[float] = None
    confidence_hit_normalizer: Optional[float] = None
    confidence_duration_norm: Optional[float] = None
    confidence_bias: Optional[float] = None
    revisit_span_limit_hz: Optional[float] = None
    cluster_merge_hz: Optional[float] = None
    max_detection_width_ratio: Optional[float] = None
    max_detection_width_hz: Optional[float] = None


def default_scan_profiles() -> Dict[str, ScanProfile]:
    profiles = [
        ScanProfile(
            name="vhf_uhf_general",
            f_low_hz=400_000_000,
            f_high_hz=470_000_000,
            samp_rate=2.4e6,
            fft=4096,
            avg=8,
            gain_db=20.0,
            threshold_db=10.0,
            min_width_bins=3,
            guard_bins=1,
            abs_power_floor_db=-95.0,
        ),
        ScanProfile(
            name="fm_broadcast",
            f_low_hz=88_000_000,
            f_high_hz=108_000_000,
            samp_rate=2.4e6,
            fft=8192,
            avg=10,
            gain_db=20.0,
            threshold_db=6.0,
            min_width_bins=12,
            guard_bins=3,
            abs_power_floor_db=-92.0,
            step_hz=1.2e6,
            cfar_train=32,
            cfar_guard=6,
            cfar_quantile=0.6,
            persistence_hit_ratio=0.25,
            persistence_min_seconds=2.0,
            persistence_min_hits=1,
            persistence_min_windows=1,
            revisit_fft=32768,
            revisit_avg=4,
            revisit_margin_hz=200_000.0,
            revisit_max_bands=40,
            revisit_floor_threshold_db=6.0,
            two_pass=True,
            bandwidth_pad_hz=60_000.0,
            min_emit_bandwidth_hz=180_000.0,
            confidence_hit_normalizer=2.0,
            confidence_duration_norm=2.0,
            confidence_bias=0.05,
            revisit_span_limit_hz=420_000.0,
            cluster_merge_hz=12_000.0,
            max_detection_width_ratio=2.5,
            max_detection_width_hz=270_000.0,
        ),
        ScanProfile(
            name="ism_902",
            f_low_hz=902_000_000,
            f_high_hz=928_000_000,
            samp_rate=2.4e6,
            fft=4096,
            avg=10,
            gain_db=25.0,
            threshold_db=12.0,
            min_width_bins=3,
            guard_bins=1,
            abs_power_floor_db=-92.0,
        ),
    ]
    return {p.name.lower(): p for p in profiles}


def serialize_profiles() -> Dict[str, Any]:
    """Return ordered JSON-serializable description of built-in profiles."""

    profiles = default_scan_profiles()
    ordered = sorted(profiles.values(), key=lambda p: p.name.lower())
    payload = {
        "profiles": [
            {
                "name": prof.name,
                "f_low_hz": prof.f_low_hz,
                "f_high_hz": prof.f_high_hz,
                "samp_rate": prof.samp_rate,
                "fft": prof.fft,
                "avg": prof.avg,
                "gain_db": prof.gain_db,
                "threshold_db": prof.threshold_db,
                "min_width_bins": prof.min_width_bins,
                "guard_bins": prof.guard_bins,
                "abs_power_floor_db": prof.abs_power_floor_db,
                "step_hz": prof.step_hz,
                "cfar_train": prof.cfar_train,
                "cfar_guard": prof.cfar_guard,
                "cfar_quantile": prof.cfar_quantile,
                "persistence_hit_ratio": prof.persistence_hit_ratio,
                "persistence_min_seconds": prof.persistence_min_seconds,
                "persistence_min_hits": prof.persistence_min_hits,
                "persistence_min_windows": prof.persistence_min_windows,
                "revisit_fft": prof.revisit_fft,
                "revisit_avg": prof.revisit_avg,
                "revisit_margin_hz": prof.revisit_margin_hz,
                "revisit_max_bands": prof.revisit_max_bands,
                "revisit_floor_threshold_db": prof.revisit_floor_threshold_db,
                "two_pass": prof.two_pass,
                "bandwidth_pad_hz": prof.bandwidth_pad_hz,
                "min_emit_bandwidth_hz": prof.min_emit_bandwidth_hz,
                "confidence_hit_normalizer": prof.confidence_hit_normalizer,
                "confidence_duration_norm": prof.confidence_duration_norm,
                "confidence_bias": prof.confidence_bias,
                "revisit_span_limit_hz": prof.revisit_span_limit_hz,
                "cluster_merge_hz": prof.cluster_merge_hz,
                "max_detection_width_ratio": prof.max_detection_width_ratio,
                "max_detection_width_hz": prof.max_detection_width_hz,
            }
            for prof in ordered
        ]
    }
    return payload
