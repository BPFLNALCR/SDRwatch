#!/usr/bin/env python3
"""SDRwatch scanner CLI entrypoint (package module)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from typing import Any, List, Optional, Set

from sdrwatch.drivers.rtlsdr import HAVE_RTLSDR, RTLSDR_IMPORT_ERROR
from sdrwatch.io.profiles import default_scan_profiles, serialize_profiles
from sdrwatch.sweep.runner import run_scan
from sdrwatch.util.duration import parse_duration_to_seconds
from sdrwatch.util.exit_codes import ExitCode
from sdrwatch.util.logging import configure_logging, get_logger

_log = get_logger(__name__)


def run(args: argparse.Namespace) -> int:
    """Top-level CLI dispatcher that delegates execution to sweep.runner.

    Returns an exit code from ExitCode.
    """
    if getattr(args, "list_profiles", False):
        _emit_profiles_json()
        return ExitCode.SUCCESS

    # Configure logging based on verbosity or environment
    log_level = "DEBUG" if os.environ.get("SDRWATCH_DEBUG", "").strip() in ("1", "true", "yes") else "INFO"
    configure_logging(level=log_level)

    try:
        run_scan(args)
        return ExitCode.SUCCESS
    except KeyboardInterrupt:
        _log.info("interrupted by user")
        return ExitCode.SUCCESS
    except RuntimeError as exc:
        msg = str(exc).lower()
        if "baseline" in msg and ("not found" in msg or "does not exist" in msg):
            _log.error("baseline not found: %s", exc)
            return ExitCode.BASELINE_NOT_FOUND
        if "device" in msg or "sdr" in msg or "rtlsdr" in msg:
            _log.error("device unavailable: %s", exc)
            return ExitCode.DEVICE_UNAVAILABLE
        _log.exception("runtime error")
        return ExitCode.GENERAL_ERROR
    except Exception:
        _log.exception("unexpected error")
        return ExitCode.GENERAL_ERROR


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    if argv is None:
        argv = sys.argv[1:]

    p = argparse.ArgumentParser(
        description="Wideband scanner & baseline builder (native RTL-SDR by default)",
        argument_default=argparse.SUPPRESS,
    )
    p.add_argument("--start", type=float, help="Start frequency in Hz (e.g., 88e6)")
    p.add_argument("--stop", type=float, help="Stop frequency in Hz (e.g., 108e6)")
    p.add_argument("--step", type=float, help="Center frequency step per window [Hz] (default 2.4e6)")

    p.add_argument("--samp-rate", dest="samp_rate", type=float, help="Sample rate [Hz] (default 2.4e6)")
    p.add_argument("--fft", type=int, help="FFT size (per Welch segment) (default 4096)")
    p.add_argument("--avg", type=int, help="Averaging factor (segments per PSD) (default 8)")

    p.add_argument("--driver", type=str, help="Driver key (default rtlsdr_native).")
    p.add_argument("--device-key", dest="device_key", type=str, help="Controller device key for diagnostics")
    p.add_argument("--job-id", dest="job_id", type=str, help="Controller job ID for diagnostics")
    p.add_argument("--role-run-id", dest="role_run_id", type=str, help="Role-run ID for diagnostics")
    p.add_argument("--receiver-role", dest="receiver_role", type=str, help="Receiver role for diagnostics")
    p.add_argument("--role-lane", dest="role_lane", type=str, help="Receiver role lane for diagnostics")
    p.add_argument("--source-task", dest="source_task", type=str, help="Role source task for diagnostics")
    p.add_argument("--device-identity", dest="device_identity", type=str, help="Stable receiver identity for diagnostics")
    p.add_argument("--device-serial", dest="device_serial", type=str, help="Receiver serial for diagnostics")
    p.add_argument("--device-index", dest="device_index", type=int, help="Runtime receiver index for diagnostics")
    p.add_argument("--identity-confidence", dest="identity_confidence", type=str, help="Receiver identity confidence")
    p.add_argument("--active-device-count", dest="active_device_count", type=int, help="Active receiver count for diagnostics")
    p.add_argument("--active-role-count", dest="active_role_count", type=int, help="Active role count for diagnostics")
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
        "--persistence-min-sweep-loops",
        dest="persistence_min_sweep_loops",
        type=int,
        help="Minimum distinct complete sweep loops required before persistence promotion (default 1)",
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
    p.add_argument(
        "--max-persist-width-hz",
        dest="max_persist_width_hz",
        type=float,
        help="Alias for --max-detection-width-hz for persisted baseline spans",
    )
    p.add_argument(
        "--max-card-width-hz",
        dest="max_card_width_hz",
        type=float,
        help="Alias for --max-detection-width-hz for operator card spans",
    )
    p.add_argument("--center-match-hz", dest="center_match_hz", type=float, help="Center-frequency match tolerance [Hz]")
    p.add_argument("--segment-center-mode", dest="segment_center_mode", choices=["midpoint", "peak", "centroid"], help="Segment center calculation mode")
    p.add_argument("--segment-centroid-span-hz", dest="segment_centroid_span_hz", type=float, help="Centroid search span around detected segment center [Hz]")
    p.add_argument("--segment-centroid-drop-db", dest="segment_centroid_drop_db", type=float, help="Centroid mask drop below segment peak [dB]")
    p.add_argument("--segment-centroid-floor-margin-db", dest="segment_centroid_floor_margin_db", type=float, help="Centroid mask floor margin above noise [dB]")
    p.add_argument("--match-bandwidth-pad-hz", dest="match_bandwidth_pad_hz", type=float, help="Hz padding for persistence match span")
    p.add_argument("--min-match-bandwidth-hz", dest="min_match_bandwidth_hz", type=float, help="Minimum persistence match span width [Hz]")
    p.add_argument("--min-identity-bandwidth-hz", dest="min_identity_bandwidth_hz", type=float, help="Minimum signal identity/match span width [Hz]")
    p.add_argument("--min-persist-bandwidth-hz", dest="min_persist_bandwidth_hz", type=float, help="Minimum persisted/card span width [Hz]")
    p.add_argument("--max-persist-bandwidth-hz", dest="max_persist_bandwidth_hz", type=float, help="Maximum persisted/card span width [Hz]")
    p.add_argument("--display-bandwidth-pad-hz", dest="display_bandwidth_pad_hz", type=float, help="Hz padding for operator display span")
    p.add_argument("--min-display-bandwidth-hz", dest="min_display_bandwidth_hz", type=float, help="Minimum operator display span width [Hz]")
    p.add_argument(
        "--allow-revisit-to-shrink-identity",
        dest="allow_revisit_to_shrink_identity",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Allow revisit evidence to reduce signal identity span",
    )
    p.add_argument(
        "--allow-revisit-to-move-center",
        dest="allow_revisit_to_move_center",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Allow revisit evidence to move stable signal center within policy gates",
    )
    p.add_argument(
        "--min-revisit-bandwidth-for-identity-update-hz",
        dest="min_revisit_bandwidth_for_identity_update_hz",
        type=float,
        help="Minimum revisit bandwidth required before identity update authority [Hz]",
    )
    p.add_argument(
        "--max-revisit-center-delta-for-identity-update-hz",
        dest="max_revisit_center_delta_for_identity_update_hz",
        type=float,
        help="Maximum revisit center delta allowed before identity update authority [Hz]",
    )
    p.add_argument("--fragmented-revisit-policy", dest="fragmented_revisit_policy", type=str, help="Policy for ambiguous or fragmented revisit evidence")
    p.add_argument("--raw-fragment-interpretation", dest="raw_fragment_interpretation", type=str, help="Diagnostic interpretation label for raw detector fragments")
    p.add_argument(
        "--center-smoothing-enabled",
        dest="center_smoothing_enabled",
        action=argparse.BooleanOptionalAction,
        default=argparse.SUPPRESS,
        help="Enable profile-governed center smoothing",
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
    p.add_argument(
        "--diagnostic-jsonl",
        dest="diagnostic_jsonl",
        type=str,
        help="Emit per-window detection tuning diagnostics as JSONL to this path",
    )
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
    _set_default(args, args._cli_overrides, "driver", "rtlsdr_native")
    _set_default(args, args._cli_overrides, "device_key", None)
    _set_default(args, args._cli_overrides, "job_id", None)
    _set_default(args, args._cli_overrides, "role_run_id", None)
    _set_default(args, args._cli_overrides, "receiver_role", None)
    _set_default(args, args._cli_overrides, "role_lane", None)
    _set_default(args, args._cli_overrides, "source_task", None)
    _set_default(args, args._cli_overrides, "device_identity", None)
    _set_default(args, args._cli_overrides, "device_serial", None)
    _set_default(args, args._cli_overrides, "device_index", None)
    _set_default(args, args._cli_overrides, "identity_confidence", None)
    _set_default(args, args._cli_overrides, "active_device_count", None)
    _set_default(args, args._cli_overrides, "active_role_count", None)
    _set_default(args, args._cli_overrides, "gain", "auto")
    _set_default(args, args._cli_overrides, "threshold_db", 8.0)
    _set_default(args, args._cli_overrides, "guard_bins", 1)
    _set_default(args, args._cli_overrides, "min_width_bins", 2)
    _set_default(args, args._cli_overrides, "persistence_mode", "hits")
    _set_default(args, args._cli_overrides, "persistence_hit_ratio", 0.6)
    _set_default(args, args._cli_overrides, "persistence_min_seconds", 10.0)
    _set_default(args, args._cli_overrides, "persistence_min_hits", 2)
    _set_default(args, args._cli_overrides, "persistence_min_windows", 2)
    _set_default(args, args._cli_overrides, "persistence_min_sweep_loops", 1)
    _set_default(args, args._cli_overrides, "cluster_merge_hz", None)
    _set_default(args, args._cli_overrides, "max_detection_width_ratio", 3.0)
    _set_default(args, args._cli_overrides, "max_detection_width_hz", 0.0)
    _set_default(args, args._cli_overrides, "max_persist_width_hz", None)
    _set_default(args, args._cli_overrides, "max_card_width_hz", None)
    _set_default(args, args._cli_overrides, "center_match_hz", None)
    _set_default(args, args._cli_overrides, "segment_center_mode", None)
    _set_default(args, args._cli_overrides, "segment_centroid_span_hz", None)
    _set_default(args, args._cli_overrides, "segment_centroid_drop_db", None)
    _set_default(args, args._cli_overrides, "segment_centroid_floor_margin_db", None)
    _set_default(args, args._cli_overrides, "match_bandwidth_pad_hz", None)
    _set_default(args, args._cli_overrides, "min_match_bandwidth_hz", None)
    _set_default(args, args._cli_overrides, "min_identity_bandwidth_hz", None)
    _set_default(args, args._cli_overrides, "min_persist_bandwidth_hz", None)
    _set_default(args, args._cli_overrides, "max_persist_bandwidth_hz", None)
    _set_default(args, args._cli_overrides, "display_bandwidth_pad_hz", None)
    _set_default(args, args._cli_overrides, "min_display_bandwidth_hz", None)
    _set_default(args, args._cli_overrides, "allow_revisit_to_shrink_identity", None)
    _set_default(args, args._cli_overrides, "allow_revisit_to_move_center", None)
    _set_default(args, args._cli_overrides, "min_revisit_bandwidth_for_identity_update_hz", None)
    _set_default(args, args._cli_overrides, "max_revisit_center_delta_for_identity_update_hz", None)
    _set_default(args, args._cli_overrides, "fragmented_revisit_policy", None)
    _set_default(args, args._cli_overrides, "raw_fragment_interpretation", None)
    _set_default(args, args._cli_overrides, "center_smoothing_enabled", None)
    _set_default(args, args._cli_overrides, "cfar", "os")
    _set_default(args, args._cli_overrides, "cfar_train", 24)
    _set_default(args, args._cli_overrides, "cfar_guard", 4)
    _set_default(args, args._cli_overrides, "cfar_quantile", 0.75)
    _set_default(args, args._cli_overrides, "cfar_alpha_db", None)
    _set_default(args, args._cli_overrides, "bandplan", None)
    _set_default(args, args._cli_overrides, "db", "sdrwatch.db")
    _set_default(args, args._cli_overrides, "jsonl", None)
    _set_default(args, args._cli_overrides, "diagnostic_jsonl", None)
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
    _initialize_profile_application_metadata(args)

    has_span = hasattr(args, "start") and hasattr(args, "stop")
    if not args.list_profiles and not has_span:
        p.error("--start and --stop are required unless --list-profiles is used")

    if has_span:
        _apply_scan_profile(args, p)
        _apply_width_cap_aliases(args, getattr(args, "_cli_overrides", set()))

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
        if args.driver != "rtlsdr_native":
            p.error("unsupported driver. This build supports only --driver rtlsdr_native")
        if args.driver == "rtlsdr_native" and not HAVE_RTLSDR:
            detail = f" ({RTLSDR_IMPORT_ERROR})" if RTLSDR_IMPORT_ERROR else ""
            p.error(f"rtlsdr Python backend unavailable{detail}. Ensure pyrtlsdr and setuptools are installed.")
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


def _apply_width_cap_aliases(args: argparse.Namespace, overrides: Set[str]) -> None:
    if "max_detection_width_hz" in overrides:
        return
    for alias in ("max_persist_width_hz", "max_card_width_hz"):
        value = getattr(args, alias, None)
        if value not in (None, ""):
            setattr(args, "max_detection_width_hz", value)
            return


def _fallback_defaults_from_args(args: argparse.Namespace) -> dict[str, Any]:
    keys = (
        "step",
        "samp_rate",
        "fft",
        "avg",
        "threshold_db",
        "guard_bins",
        "min_width_bins",
        "persistence_mode",
        "persistence_hit_ratio",
        "persistence_min_seconds",
        "persistence_min_hits",
        "persistence_min_windows",
        "persistence_min_sweep_loops",
        "max_detection_width_hz",
        "gain",
    )
    return {key: getattr(args, key) for key in keys if hasattr(args, key)}


def _initialize_profile_application_metadata(args: argparse.Namespace) -> None:
    overrides: Set[str] = getattr(args, "_cli_overrides", set())
    setattr(args, "_requested_profile", getattr(args, "profile", None))
    setattr(args, "_applied_profile", None)
    setattr(args, "_profile_applied", False)
    setattr(args, "_profile_skip_reason", None)
    setattr(args, "_profile_defaults", {})
    setattr(args, "_fallback_defaults", _fallback_defaults_from_args(args))
    setattr(
        args,
        "_operator_overrides",
        {
            key: getattr(args, key)
            for key in sorted(overrides)
            if key != "profile" and hasattr(args, key)
        },
    )


def _scan_profile_defaults(profile) -> dict[str, Any]:
    defaults = asdict(profile)
    return {key: value for key, value in defaults.items() if value is not None}


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
        reason = (
            f"requested span {requested_low/1e6:.3f}-{requested_high/1e6:.3f}MHz "
            f"outside {profile.name} {profile.f_low_hz/1e6:.3f}-{profile.f_high_hz/1e6:.3f}MHz"
        )
        setattr(args, "_requested_profile", profile.name)
        setattr(args, "_applied_profile", None)
        setattr(args, "_profile_applied", False)
        setattr(args, "_profile_skip_reason", reason)
        setattr(args, "_profile_defaults", {})
        setattr(args, "_fallback_defaults", _fallback_defaults_from_args(args))
        _log.warning(
            "requested span %.3f-%.3fMHz outside profile '%s' band, skipping profile defaults",
            requested_low / 1e6,
            requested_high / 1e6,
            profile.name,
        )
        return

    overrides: Set[str] = getattr(args, "_cli_overrides", set())
    setattr(args, "_requested_profile", profile.name)
    setattr(args, "_applied_profile", profile.name)
    setattr(args, "_profile_applied", True)
    setattr(args, "_profile_skip_reason", None)
    setattr(args, "_profile_defaults", _scan_profile_defaults(profile))
    setattr(args, "_fallback_defaults", {})

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
    maybe_set("persistence_min_sweep_loops", getattr(profile, "persistence_min_sweep_loops", None))
    maybe_set("revisit_fft", profile.revisit_fft)
    maybe_set("revisit_avg", profile.revisit_avg)
    maybe_set("revisit_margin_hz", profile.revisit_margin_hz)
    maybe_set("revisit_max_bands", profile.revisit_max_bands)
    maybe_set("revisit_floor_threshold_db", profile.revisit_floor_threshold_db)
    maybe_set("revisit_span_limit_hz", profile.revisit_span_limit_hz)
    maybe_set("two_pass", profile.two_pass)
    maybe_set("cluster_merge_hz", profile.cluster_merge_hz)
    maybe_set("center_match_hz", getattr(profile, "center_match_hz", None))
    maybe_set("max_detection_width_ratio", profile.max_detection_width_ratio)
    maybe_set("max_detection_width_hz", profile.max_detection_width_hz)
    maybe_set("segment_center_mode", profile.segment_center_mode)
    maybe_set("segment_centroid_span_hz", profile.segment_centroid_span_hz)
    maybe_set("segment_centroid_drop_db", profile.segment_centroid_drop_db)
    maybe_set("segment_centroid_floor_margin_db", profile.segment_centroid_floor_margin_db)

    maybe_set("match_bandwidth_pad_hz", getattr(profile, "match_bandwidth_pad_hz", None))
    maybe_set("min_match_bandwidth_hz", getattr(profile, "min_match_bandwidth_hz", None))
    maybe_set("min_identity_bandwidth_hz", getattr(profile, "min_identity_bandwidth_hz", None))
    maybe_set("min_persist_bandwidth_hz", getattr(profile, "min_persist_bandwidth_hz", None))
    maybe_set("max_persist_bandwidth_hz", getattr(profile, "max_persist_bandwidth_hz", None))
    maybe_set("display_bandwidth_pad_hz", getattr(profile, "display_bandwidth_pad_hz", None))
    maybe_set("min_display_bandwidth_hz", getattr(profile, "min_display_bandwidth_hz", None))
    maybe_set("allow_revisit_to_shrink_identity", getattr(profile, "allow_revisit_to_shrink_identity", None))
    maybe_set("allow_revisit_to_move_center", getattr(profile, "allow_revisit_to_move_center", None))
    maybe_set(
        "min_revisit_bandwidth_for_identity_update_hz",
        getattr(profile, "min_revisit_bandwidth_for_identity_update_hz", None),
    )
    maybe_set(
        "max_revisit_center_delta_for_identity_update_hz",
        getattr(profile, "max_revisit_center_delta_for_identity_update_hz", None),
    )
    maybe_set("fragmented_revisit_policy", getattr(profile, "fragmented_revisit_policy", None))
    maybe_set("raw_fragment_interpretation", getattr(profile, "raw_fragment_interpretation", None))
    maybe_set("center_smoothing_enabled", getattr(profile, "center_smoothing_enabled", None))

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
            _log.info(
                "overriding auto gain with fixed %.1fdB from profile '%s'",
                profile.gain_db,
                profile.name,
            )
        setattr(args, "gain", float(profile.gain_db))

    _log.info("applied profile '%s'", profile.name)


def _emit_profiles_json() -> None:
    payload = serialize_profiles()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    sys.exit(run(parse_args()))
