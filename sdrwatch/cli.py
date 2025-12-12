#!/usr/bin/env python3
"""SDRwatch scanner CLI entrypoint (package module)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Optional, Set

from sdrwatch.drivers.rtlsdr import HAVE_RTLSDR
from sdrwatch.drivers.soapy import HAVE_SOAPY
from sdrwatch.io.profiles import default_scan_profiles, serialize_profiles
from sdrwatch.sweep.runner import run_scan
from sdrwatch.util.duration import parse_duration_to_seconds


def run(args: argparse.Namespace) -> None:
    """Top-level CLI dispatcher that delegates execution to sweep.runner."""
    if getattr(args, "list_profiles", False):
        _emit_profiles_json()
        return
    run_scan(args)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]

    p = argparse.ArgumentParser(
        description="Wideband scanner & baseline builder using SoapySDR or native RTL-SDR",
        argument_default=argparse.SUPPRESS,
    )
    p.add_argument("--start", type=float, help="Start frequency in Hz (e.g., 88e6)")
    p.add_argument("--stop", type=float, help="Stop frequency in Hz (e.g., 108e6)")
    p.add_argument("--step", type=float, help="Center frequency step per window [Hz] (default 2.4e6)")

    p.add_argument("--samp-rate", dest="samp_rate", type=float, help="Sample rate [Hz] (default 2.4e6)")
    p.add_argument("--fft", type=int, help="FFT size (per Welch segment) (default 4096)")
    p.add_argument("--avg", type=int, help="Averaging factor (segments per PSD) (default 8)")

    p.add_argument("--driver", type=str, help="Soapy driver key (e.g., rtlsdr, hackrf, airspy, etc.) or 'rtlsdr_native' for direct librtlsdr (default rtlsdr)")
    p.add_argument("--soapy-args", type=str, help="Comma-separated Soapy device args (e.g., 'serial=00000001,index=0')")
    p.add_argument("--gain", type=str, help='Gain in dB or "auto" (default auto)')

    p.add_argument("--threshold-db", dest="threshold_db", type=float, help="Detection threshold above noise floor [dB] (default 8.0)")
    p.add_argument("--guard-bins", dest="guard_bins", type=int, help="Allow this many below-threshold bins inside a detection (default 1)")
    p.add_argument("--min-width-bins", dest="min_width_bins", type=int, help="Minimum contiguous bins for a detection (default 2)")
    p.add_argument(
        "--persistence-mode",
        choices=["hits", "duration", "both"],
        help="Persistence gate: hit/window ratio, wall-clock duration, or both (default hits)",
    )
    p.add_argument(
        "--persistence-hit-ratio",
        dest="persistence_hit_ratio",
        type=float,
        help="Minimum occupied-window ratio (0-1) within a cluster span to mark persistent (default 0.6)",
    )
    p.add_argument(
        "--persistence-min-seconds",
        dest="persistence_min_seconds",
        type=float,
        help="Minimum wall-clock duration in seconds for duration-based persistence (default 10)",
    )
    p.add_argument(
        "--persistence-min-hits",
        dest="persistence_min_hits",
        type=int,
        help="Minimum hits required before persistence evaluation (default 2)",
    )
    p.add_argument(
        "--persistence-min-windows",
        dest="persistence_min_windows",
        type=int,
        help="Minimum distinct windows required before persistence evaluation (default 2)",
    )
    p.add_argument(
        "--cluster-merge-hz",
        dest="cluster_merge_hz",
        type=float,
        help="Override Hz span when merging per-window segments into clusters/persistent detections",
    )
    p.add_argument(
        "--max-detection-width-ratio",
        dest="max_detection_width_ratio",
        type=float,
        help="Reject cluster matches when the segment width exceeds this ratio of the persisted width (default 3.0)",
    )
    p.add_argument(
        "--max-detection-width-hz",
        dest="max_detection_width_hz",
        type=float,
        help="Clamp persistent detection widths to this maximum Hz span (0 disables)",
    )
    p.add_argument("--cfar", choices=["off", "os", "ca"], help="CFAR mode (default: os)")
    p.add_argument("--cfar-train", dest="cfar_train", type=int, help="Training cells per side for CFAR (default 24)")
    p.add_argument("--cfar-guard", dest="cfar_guard", type=int, help="Guard cells per side (excluded around CUT) for CFAR (default 4)")
    p.add_argument("--cfar-quantile", dest="cfar_quantile", type=float, help="Quantile (0..1) for OS-CFAR order statistic (default 0.75)")
    p.add_argument("--cfar-alpha-db", dest="cfar_alpha_db", type=float, help="Override threshold scaling for CFAR in dB; defaults to --threshold-db")

    p.add_argument("--bandplan", type=str, help="Optional bandplan CSV to map detections")
    p.add_argument("--db", type=str, help="SQLite DB path (default sdrwatch.db)")
    p.add_argument("--baseline-id", dest="baseline_id", type=str, help="Baseline id to attach scans to (or 'latest')")
    p.add_argument("--jsonl", type=str, help="Emit detections as line-delimited JSON to this path")
    p.add_argument("--notify", action="store_true", help="Desktop notifications for new signals")
    p.add_argument("--new-ema-occ", dest="new_ema_occ", type=float, help="EMA occupancy threshold to flag a bin as NEW (default 0.02)")
    p.add_argument("--latitude", type=float, help="Optional latitude in decimal degrees for this scan")
    p.add_argument("--longitude", type=float, help="Optional longitude in decimal degrees for this scan")
    p.add_argument("--profile", type=str, help="Scan profile name to pre-load sane defaults (see documentation)")
    p.add_argument("--spur-calibration", dest="spur_calibration", action="store_true", help="Learn persistent internal spurs instead of emitting detections")
    p.add_argument("--list-profiles", dest="list_profiles", action="store_true", help="Print built-in scan profiles as JSON and exit")
    p.add_argument("--two-pass", dest="two_pass", action="store_true", help="Enable coarse + targeted revisit confirmation sweep")
    p.add_argument("--revisit-fft", dest="revisit_fft", type=int, help="FFT size for revisit windows (defaults to 2x --fft)")
    p.add_argument("--revisit-avg", dest="revisit_avg", type=int, help="Averaging factor for revisit windows (defaults to max(--avg,4))")
    p.add_argument(
        "--revisit-margin-hz",
        dest="revisit_margin_hz",
        type=float,
        help="Additional Hz margin added to revisit windows around each tagged center",
    )
    p.add_argument(
        "--revisit-span-limit-hz",
        dest="revisit_span_limit_hz",
        type=float,
        help="Maximum Hz span allowed when confirming detections during revisit passes (0 disables clamping)",
    )
    p.add_argument(
        "--revisit-max-bands",
        dest="revisit_max_bands",
        type=int,
        help="Maximum revisit targets per sweep (0 = unlimited)",
    )
    p.add_argument(
        "--revisit-floor-threshold-db",
        dest="revisit_floor_threshold_db",
        type=float,
        help="Detection threshold (dB) used during revisit windows (defaults to --threshold-db)",
    )

    group = p.add_mutually_exclusive_group()
    group.add_argument("--loop", action="store_true", help="Run continuous sweep cycles until cancelled")
    group.add_argument("--repeat", type=int, help="Run exactly N full sweep cycles, then exit")
    group.add_argument("--duration", type=str, help="Run sweeps for a duration (e.g., '300', '10m', '2h'). Overrides --repeat count while time remains")

    p.add_argument("--sleep-between-sweeps", dest="sleep_between_sweeps", type=float, help="Seconds to sleep between sweep cycles (default 0)")
    p.add_argument("--tmpdir", type=str, help="Scratch directory for temp files (defaults to $TMPDIR)")

    args = p.parse_args(argv)
    args._cli_overrides = set()

    _set_default(args, args._cli_overrides, "step", 2.4e6)
    _set_default(args, args._cli_overrides, "samp_rate", 2.4e6)
    _set_default(args, args._cli_overrides, "fft", 4096)
    _set_default(args, args._cli_overrides, "avg", 8)
    _set_default(args, args._cli_overrides, "driver", "rtlsdr")
    _set_default(args, args._cli_overrides, "soapy_args", None)
    _set_default(args, args._cli_overrides, "gain", "auto")
    _set_default(args, args._cli_overrides, "threshold_db", 8.0)
    _set_default(args, args._cli_overrides, "guard_bins", 1)
    _set_default(args, args._cli_overrides, "min_width_bins", 2)
    _set_default(args, args._cli_overrides, "persistence_mode", "hits")
    _set_default(args, args._cli_overrides, "persistence_hit_ratio", 0.6)
    _set_default(args, args._cli_overrides, "persistence_min_seconds", 10.0)
    _set_default(args, args._cli_overrides, "persistence_min_hits", 2)
    _set_default(args, args._cli_overrides, "persistence_min_windows", 2)
    _set_default(args, args._cli_overrides, "cluster_merge_hz", None)
    _set_default(args, args._cli_overrides, "max_detection_width_ratio", 3.0)
    _set_default(args, args._cli_overrides, "max_detection_width_hz", 0.0)
    _set_default(args, args._cli_overrides, "cfar", "os")
    _set_default(args, args._cli_overrides, "cfar_train", 24)
    _set_default(args, args._cli_overrides, "cfar_guard", 4)
    _set_default(args, args._cli_overrides, "cfar_quantile", 0.75)
    _set_default(args, args._cli_overrides, "cfar_alpha_db", None)
    _set_default(args, args._cli_overrides, "bandplan", None)
    _set_default(args, args._cli_overrides, "db", "sdrwatch.db")
    _set_default(args, args._cli_overrides, "jsonl", None)
    _set_default(args, args._cli_overrides, "notify", False)
    _set_default(args, args._cli_overrides, "new_ema_occ", 0.02)
    _set_default(args, args._cli_overrides, "latitude", None)
    _set_default(args, args._cli_overrides, "longitude", None)
    _set_default(args, args._cli_overrides, "profile", None)
    _set_default(args, args._cli_overrides, "spur_calibration", False)
    _set_default(args, args._cli_overrides, "list_profiles", False)
    _set_default(args, args._cli_overrides, "two_pass", False)
    _set_default(args, args._cli_overrides, "revisit_fft", None)
    _set_default(args, args._cli_overrides, "revisit_avg", None)
    _set_default(args, args._cli_overrides, "revisit_margin_hz", None)
    _set_default(args, args._cli_overrides, "revisit_span_limit_hz", None)
    _set_default(args, args._cli_overrides, "revisit_max_bands", 0)
    _set_default(args, args._cli_overrides, "revisit_floor_threshold_db", None)
    _set_default(args, args._cli_overrides, "loop", False)
    _set_default(args, args._cli_overrides, "repeat", None)
    _set_default(args, args._cli_overrides, "duration", None)
    _set_default(args, args._cli_overrides, "sleep_between_sweeps", 0.0)
    _set_default(args, args._cli_overrides, "tmpdir", os.environ.get("TMPDIR"))
    setattr(args, "abs_power_floor_db", None)

    has_span = hasattr(args, "start") and hasattr(args, "stop")
    if not args.list_profiles and not has_span:
        p.error("--start and --stop are required unless --list-profiles is used")

    if has_span:
        _apply_scan_profile(args, p)

    if not args.list_profiles:
        baseline_raw = getattr(args, "baseline_id", None)
        if baseline_raw is None:
            p.error("--baseline-id is required for scanning runs")
        baseline_text = str(baseline_raw).strip()
        if not baseline_text:
            p.error("--baseline-id is required for scanning runs")
        if baseline_text.lower() == "latest":
            setattr(args, "baseline_id", "latest")
        else:
            try:
                baseline_val = int(baseline_text)
            except ValueError:
                p.error("--baseline-id must be an integer or 'latest'")
            setattr(args, "baseline_id", baseline_val)

    if hasattr(args, "_cli_overrides"):
        delattr(args, "_cli_overrides")

    if not args.list_profiles:
        if args.driver != "rtlsdr_native" and not HAVE_SOAPY:
            p.error("python3-soapysdr not installed. Install it (or use --driver rtlsdr_native).")
        if args.driver == "rtlsdr_native" and not HAVE_RTLSDR:
            p.error("pyrtlsdr not installed. Install with: pip3 install pyrtlsdr")
        if args.stop < args.start:
            p.error("--stop must be >= --start")
        if args.step <= 0:
            p.error("--step must be > 0")

    if args.duration:
        _ = parse_duration_to_seconds(args.duration)

    return args


def _set_default(args: argparse.Namespace, overrides: Set[str], attr: str, value: Any) -> None:
    if hasattr(args, attr):
        overrides.add(attr)
    else:
        setattr(args, attr, value)


def _apply_scan_profile(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    profile_name = getattr(args, "profile", None)
    if not profile_name:
        return
    profiles = default_scan_profiles()
    profile = profiles.get(str(profile_name).lower())
    if not profile:
        parser.error(f"Unknown scan profile '{profile_name}'. Use --list-profiles to inspect options.")

    requested_low = min(args.start, args.stop)
    requested_high = max(args.start, args.stop)
    if requested_low < profile.f_low_hz or requested_high > profile.f_high_hz:
        print(
            f"[profile] Requested span {requested_low/1e6:.3f}-{requested_high/1e6:.3f} MHz outside profile '{profile.name}' band, skipping profile defaults.",
            file=sys.stderr,
        )
        return

    overrides: Set[str] = getattr(args, "_cli_overrides", set())

    def maybe_set(attr: str, value: Any) -> None:
        if value is None:
            return
        if attr in overrides:
            return
        setattr(args, attr, value)

    if profile.step_hz is not None:
        maybe_set("step", profile.step_hz)
    maybe_set("samp_rate", profile.samp_rate)
    maybe_set("fft", profile.fft)
    maybe_set("avg", profile.avg)
    maybe_set("threshold_db", profile.threshold_db)
    maybe_set("guard_bins", profile.guard_bins)
    maybe_set("min_width_bins", profile.min_width_bins)
    maybe_set("cfar_train", profile.cfar_train)
    maybe_set("cfar_guard", profile.cfar_guard)
    maybe_set("cfar_quantile", profile.cfar_quantile)
    maybe_set("persistence_hit_ratio", profile.persistence_hit_ratio)
    maybe_set("persistence_min_seconds", profile.persistence_min_seconds)
    maybe_set("persistence_min_hits", profile.persistence_min_hits)
    maybe_set("persistence_min_windows", profile.persistence_min_windows)
    maybe_set("revisit_fft", profile.revisit_fft)
    maybe_set("revisit_avg", profile.revisit_avg)
    maybe_set("revisit_margin_hz", profile.revisit_margin_hz)
    maybe_set("revisit_max_bands", profile.revisit_max_bands)
    maybe_set("revisit_floor_threshold_db", profile.revisit_floor_threshold_db)
    maybe_set("revisit_span_limit_hz", profile.revisit_span_limit_hz)
    maybe_set("two_pass", profile.two_pass)
    maybe_set("cluster_merge_hz", profile.cluster_merge_hz)
    maybe_set("max_detection_width_ratio", profile.max_detection_width_ratio)
    maybe_set("max_detection_width_hz", profile.max_detection_width_hz)
    maybe_set("segment_center_mode", profile.segment_center_mode)
    maybe_set("segment_centroid_span_hz", profile.segment_centroid_span_hz)
    maybe_set("segment_centroid_drop_db", profile.segment_centroid_drop_db)
    maybe_set("segment_centroid_floor_margin_db", profile.segment_centroid_floor_margin_db)

    if profile.bandwidth_pad_hz is not None:
        setattr(args, "bandwidth_pad_hz", profile.bandwidth_pad_hz)
    if profile.min_emit_bandwidth_hz is not None:
        setattr(args, "min_emit_bandwidth_hz", profile.min_emit_bandwidth_hz)
    if profile.confidence_hit_normalizer is not None:
        setattr(args, "confidence_hit_normalizer", profile.confidence_hit_normalizer)
    if profile.confidence_duration_norm is not None:
        setattr(args, "confidence_duration_norm", profile.confidence_duration_norm)
    if profile.confidence_bias is not None:
        setattr(args, "confidence_bias", profile.confidence_bias)
    if profile.abs_power_floor_db is not None:
        setattr(args, "abs_power_floor_db", profile.abs_power_floor_db)

    gain_override = "gain" in overrides and not (isinstance(getattr(args, "gain"), str) and getattr(args, "gain").lower() == "auto")
    if not gain_override:
        if isinstance(getattr(args, "gain"), str) and getattr(args, "gain").lower() == "auto":
            print(
                f"[profile] Overriding auto gain with fixed {profile.gain_db:.1f} dB from profile '{profile.name}'.",
                file=sys.stderr,
            )
        setattr(args, "gain", float(profile.gain_db))

    print(f"[profile] Applied profile '{profile.name}'", flush=True)


def _emit_profiles_json() -> None:
    payload = serialize_profiles()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    run(parse_args())
