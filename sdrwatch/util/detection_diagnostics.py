"""JSONL diagnostics for scan detection tuning."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from sdrwatch.detection.span_policy import resolve_signal_span_policy
from sdrwatch.detection.types import CharacterizationEvidence, CharacterizationSpan, DeviceTelemetrySnapshot, Segment
from sdrwatch.util.time import utc_now_str


def segment_to_dict(seg: Segment) -> Dict[str, Any]:
    return {
        "f_low_hz": seg.f_low_hz,
        "f_high_hz": seg.f_high_hz,
        "f_center_hz": seg.f_center_hz,
        "bandwidth_hz": seg.bandwidth_hz,
        "peak_db": seg.peak_db,
        "noise_db": seg.noise_db,
        "snr_db": seg.snr_db,
    }


def build_characterization_record(*, evidence: CharacterizationEvidence) -> Dict[str, Any]:
    record = evidence.to_record()
    record.setdefault("time_utc", utc_now_str())
    return record


def _get_attr(source: Any, attr: str, default: Any = None) -> Any:
    if source is None:
        return default
    if isinstance(source, dict):
        return source.get(attr, default)
    return getattr(source, attr, default)


_PROVENANCE_KEYS = (
    "job_id",
    "role_run_id",
    "receiver_role",
    "role_lane",
    "source_task",
    "device_identity",
    "device_key",
    "device_serial",
    "device_index",
    "identity_confidence",
    "active_device_count",
    "active_role_count",
)


def _provenance_fields(source: Any) -> Dict[str, Any]:
    fields: Dict[str, Any] = {}
    for key in _PROVENANCE_KEYS:
        value = _get_attr(source, key)
        if value not in (None, ""):
            fields[key] = value
    backend = _get_attr(source, "backend", _get_attr(source, "driver"))
    if backend not in (None, ""):
        fields["backend"] = backend
    source_profile = _get_attr(source, "source_profile", _get_attr(source, "profile"))
    if source_profile not in (None, ""):
        fields["source_profile"] = source_profile
    return fields


def _clean_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(str(value)))
        except (TypeError, ValueError):
            return None


def _bin_width_hz(args: Any) -> Optional[float]:
    sample_rate = _clean_float(_get_attr(args, "samp_rate", _get_attr(args, "sample_rate_hz")))
    fft = _clean_int(_get_attr(args, "fft"))
    if sample_rate is None or fft in (None, 0):
        return None
    return float(sample_rate) / float(fft)


def _call_or_value(value: Any) -> Any:
    if callable(value):
        try:
            return value()
        except TypeError:
            return None
        except Exception:
            return None
    return value


def _supported_gains(dev: Any) -> Optional[list[float]]:
    value = _call_or_value(_get_attr(dev, "valid_gains_db"))
    if value is None:
        value = _call_or_value(_get_attr(dev, "get_gains"))
    if value is None:
        return None
    try:
        return [float(item) for item in value]
    except TypeError:
        return None


def build_device_telemetry_snapshot(
    args: Any,
    source: Any = None,
    *,
    device_key: Optional[str] = None,
    selected_profile: Optional[str] = None,
) -> Dict[str, Any]:
    dev = _get_attr(source, "dev")
    requested_gain_raw = _get_attr(args, "gain")
    requested_gain = None if requested_gain_raw in (None, "") else str(requested_gain_raw)
    gain_mode_raw = _get_attr(args, "gain_mode")
    gain_mode = str(gain_mode_raw or ("auto" if str(requested_gain or "").lower() == "auto" else "manual"))
    sample_rate = _clean_float(_get_attr(args, "samp_rate", _get_attr(args, "sample_rate_hz")))
    fft = _clean_int(_get_attr(args, "fft"))
    actual_gain = _clean_float(_get_attr(dev, "gain", _get_attr(source, "gain")))
    supported_gains = _supported_gains(dev)
    actual_sample_rate = _clean_float(_get_attr(dev, "sample_rate", _get_attr(source, "sample_rate")))
    device_serial = _get_attr(dev, "serial_number", _get_attr(source, "serial_number"))
    device_tuner = _get_attr(dev, "tuner_type", _get_attr(dev, "tuner", _get_attr(source, "tuner")))
    device_index = _clean_int(_get_attr(source, "device_index", _get_attr(args, "device_index")))
    label = _get_attr(source, "device", _get_attr(source, "device_label"))
    driver = _get_attr(args, "driver", "rtlsdr_native")
    key = device_key or _get_attr(args, "device_key")
    unavailable_fields: list[str] = []
    required_nullable = {
        "actual_gain": actual_gain,
        "supported_gains": supported_gains,
        "device_index": device_index,
        "device_serial": device_serial,
        "device_tuner": device_tuner,
        "actual_sample_rate_hz": actual_sample_rate,
    }
    for field, value in required_nullable.items():
        if value is None:
            unavailable_fields.append(field)
    if label is None:
        unavailable_fields.append("device_label")
    snapshot = DeviceTelemetrySnapshot(
        event="device_telemetry",
        device_key=str(key) if key not in (None, "") else None,
        device_kind=(str(key).split(":", 1)[0] if key not in (None, "") and ":" in str(key) else None),
        device_index=device_index,
        device_serial=(str(device_serial) if device_serial not in (None, "") else None),
        device_label=(str(label) if label not in (None, "") else None),
        device_tuner=(str(device_tuner) if device_tuner not in (None, "") else None),
        driver=str(driver) if driver not in (None, "") else None,
        requested_gain=requested_gain,
        actual_gain=actual_gain,
        gain_mode=gain_mode,
        supported_gains=supported_gains,
        sample_rate_hz=sample_rate,
        actual_sample_rate_hz=actual_sample_rate,
        fft=fft,
        bin_width_hz=_bin_width_hz(args),
        selected_profile=selected_profile if selected_profile is not None else _get_attr(args, "profile"),
        unavailable_fields=unavailable_fields,
    )
    record = snapshot.to_record()
    record.update(_provenance_fields(args))
    if device_key not in (None, ""):
        record["device_key"] = str(device_key)
    if record.get("device_serial") is None and _get_attr(args, "device_serial") not in (None, ""):
        record["device_serial"] = str(_get_attr(args, "device_serial"))
    if record.get("device_index") is None and _get_attr(args, "device_index") not in (None, ""):
        record["device_index"] = _clean_int(_get_attr(args, "device_index"))
    return record


def build_resource_telemetry_snapshot(args: Any, *, pid: Optional[int] = None) -> Dict[str, Any]:
    unavailable_fields: list[str] = []
    rss_memory_bytes = None
    try:
        import resource  # type: ignore

        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss_memory_bytes = int(getattr(usage, "ru_maxrss", 0) or 0)
        if rss_memory_bytes and rss_memory_bytes < 10_000_000:
            rss_memory_bytes *= 1024
    except Exception:
        unavailable_fields.append("rss_memory_bytes")
    cpu_load = None
    try:
        import os

        cpu_load = float(os.getloadavg()[0])  # type: ignore[attr-defined]
    except Exception:
        unavailable_fields.append("cpu_load")
    record = {
        "event": "resource_telemetry",
        "pid": int(pid) if pid is not None else None,
        "sample_rate": _clean_int(_get_attr(args, "samp_rate", _get_attr(args, "sample_rate_hz"))),
        "cpu_load": cpu_load,
        "rss_memory_bytes": rss_memory_bytes,
        "unavailable_fields": unavailable_fields,
    }
    record.update(_provenance_fields(args))
    return record


def _dict_attr(source: Any, attr: str) -> Dict[str, Any]:
    value = _get_attr(source, attr, {})
    return dict(value) if isinstance(value, dict) else {}


def build_effective_parameter_manifest(
    args: Any,
    *,
    job_id: Optional[str] = None,
    device_telemetry: Optional[Dict[str, Any]] = None,
    profile_application_source: str = "scanner_effective_parameters",
    profile_audit_complete: Optional[bool] = None,
    profile_applied_unknown_if_unreported: bool = False,
) -> Dict[str, Any]:
    requested_profile = _get_attr(args, "_requested_profile", _get_attr(args, "profile"))
    explicit_profile_applied = _get_attr(args, "_profile_applied", None)
    explicit_applied_profile = _get_attr(args, "_applied_profile", None)
    profile_skip_reason = _get_attr(args, "_profile_skip_reason")
    has_scanner_profile_audit = (
        explicit_profile_applied is not None
        or explicit_applied_profile not in (None, "")
        or profile_skip_reason not in (None, "")
    )
    if profile_applied_unknown_if_unreported and not has_scanner_profile_audit:
        profile_applied = None
        applied_profile = None
    else:
        if explicit_profile_applied is None:
            profile_applied = bool(explicit_applied_profile)
        else:
            profile_applied = bool(explicit_profile_applied)
        applied_profile = explicit_applied_profile
        if applied_profile in (None, "") and profile_applied:
            applied_profile = requested_profile
    if profile_audit_complete is None:
        profile_audit_complete = bool(
            profile_application_source == "scanner_effective_parameters"
            and (requested_profile in (None, "") or has_scanner_profile_audit)
        )
    device = dict(device_telemetry or _dict_attr(args, "_device_telemetry"))
    requested_gain = device.get("requested_gain")
    if requested_gain is None:
        gain_value = _get_attr(args, "gain")
        requested_gain = None if gain_value in (None, "") else str(gain_value)
    max_width = _clean_float(_get_attr(args, "max_detection_width_hz"))
    max_persist_width = _clean_float(_get_attr(args, "max_persist_width_hz")) or max_width
    max_card_width = _clean_float(_get_attr(args, "max_card_width_hz")) or max_width
    signal_span_policy = resolve_signal_span_policy(args).to_effective_parameters()
    manifest = {
        **_provenance_fields(args),
        "job_id": job_id or _get_attr(args, "job_id"),
        "runnable_backend": _get_attr(args, "driver"),
        "source_profile": _get_attr(args, "profile"),
        "requested_profile": requested_profile,
        "applied_profile": applied_profile,
        "profile_applied": profile_applied,
        "profile_skip_reason": profile_skip_reason,
        "profile_application_source": profile_application_source,
        "profile_audit_complete": bool(profile_audit_complete),
        "operator_overrides": _dict_attr(args, "_operator_overrides"),
        "profile_defaults": _dict_attr(args, "_profile_defaults"),
        "fallback_defaults": _dict_attr(args, "_fallback_defaults"),
        "final_effective_params": {
            "start_hz": _clean_int(_get_attr(args, "start")),
            "stop_hz": _clean_int(_get_attr(args, "stop")),
            "step_hz": _clean_float(_get_attr(args, "step")),
            "sample_rate_hz": _clean_float(_get_attr(args, "samp_rate")),
            "fft": _clean_int(_get_attr(args, "fft")),
            "avg": _clean_int(_get_attr(args, "avg")),
            "bin_width_hz": _bin_width_hz(args),
            "driver": _get_attr(args, "driver"),
        },
        "frequency_range_hz": {
            "start_hz": _clean_int(_get_attr(args, "start")),
            "stop_hz": _clean_int(_get_attr(args, "stop")),
        },
        "persistence": {
            "mode": _get_attr(args, "persistence_mode"),
            "hit_ratio": _clean_float(_get_attr(args, "persistence_hit_ratio")),
            "min_seconds": _clean_float(_get_attr(args, "persistence_min_seconds")),
            "min_hits": _clean_int(_get_attr(args, "persistence_min_hits")),
            "min_windows": _clean_int(_get_attr(args, "persistence_min_windows")),
            "min_sweep_loops": _clean_int(_get_attr(args, "persistence_min_sweep_loops")) or 1,
        },
        "revisit": {
            "two_pass": bool(_get_attr(args, "two_pass", False)),
            "fft": _clean_int(_get_attr(args, "revisit_fft")),
            "avg": _clean_int(_get_attr(args, "revisit_avg")),
            "margin_hz": _clean_float(_get_attr(args, "revisit_margin_hz")),
            "span_limit_hz": _clean_float(_get_attr(args, "revisit_span_limit_hz")),
            "max_bands": _clean_int(_get_attr(args, "revisit_max_bands")),
            "floor_threshold_db": _clean_float(_get_attr(args, "revisit_floor_threshold_db")),
        },
        "span_controls": {
            "segment_center_mode": _get_attr(args, "segment_center_mode"),
            "segment_centroid_span_hz": _clean_float(_get_attr(args, "segment_centroid_span_hz")),
            "segment_centroid_drop_db": _clean_float(_get_attr(args, "segment_centroid_drop_db")),
            "segment_centroid_floor_margin_db": _clean_float(_get_attr(args, "segment_centroid_floor_margin_db")),
            "match_bandwidth_pad_hz": _clean_float(_get_attr(args, "match_bandwidth_pad_hz")),
            "min_match_bandwidth_hz": _clean_float(_get_attr(args, "min_match_bandwidth_hz")),
            "min_identity_bandwidth_hz": signal_span_policy["min_identity_bandwidth_hz"],
            "min_persist_bandwidth_hz": signal_span_policy["min_persist_bandwidth_hz"],
            "max_persist_bandwidth_hz": signal_span_policy["max_persist_bandwidth_hz"],
            "display_bandwidth_pad_hz": _clean_float(_get_attr(args, "display_bandwidth_pad_hz")),
            "min_display_bandwidth_hz": _clean_float(_get_attr(args, "min_display_bandwidth_hz")),
            "min_revisit_bandwidth_for_identity_update_hz": signal_span_policy[
                "min_revisit_bandwidth_for_identity_update_hz"
            ],
            "max_revisit_center_delta_for_identity_update_hz": signal_span_policy[
                "max_revisit_center_delta_for_identity_update_hz"
            ],
            "allow_revisit_to_shrink_identity": signal_span_policy["allow_revisit_to_shrink_identity"],
            "allow_revisit_to_move_center": signal_span_policy["allow_revisit_to_move_center"],
            "fragmented_revisit_policy": signal_span_policy["fragmented_revisit_policy"],
            "raw_fragment_interpretation": signal_span_policy["raw_fragment_interpretation"],
            "center_smoothing_enabled": signal_span_policy["center_smoothing_enabled"],
            "invalid_policy_fields": signal_span_policy["invalid_fields"],
            "center_match_hz": _clean_float(_get_attr(args, "center_match_hz")),
            "max_persist_width_hz": max_persist_width,
            "max_card_width_hz": max_card_width,
            "max_detection_width_hz": max_width,
        },
        "signal_span_policy": signal_span_policy,
        "gain": {
            "requested_gain": requested_gain,
            "gain_mode": device.get("gain_mode") or ("auto" if str(requested_gain or "").lower() == "auto" else "manual"),
            "actual_gain": device.get("actual_gain"),
            "supported_gains": device.get("supported_gains"),
        },
        "device": {
            "device_key": device.get("device_key") or _get_attr(args, "device_key"),
            "device_kind": device.get("device_kind"),
            "device_index": device.get("device_index"),
            "device_serial": device.get("device_serial"),
            "device_label": device.get("device_label"),
            "driver": device.get("driver") or _get_attr(args, "driver"),
            "tuner": device.get("device_tuner"),
            "unavailable_fields": list(device.get("unavailable_fields") or []),
        },
    }
    return manifest


def _count_key(counter: Dict[str, int], key: Any) -> None:
    key_text = str(key or "unknown")
    counter[key_text] = counter.get(key_text, 0) + 1


def _span_from_record(record: Dict[str, Any], *, prefix: str, nested_key: str) -> CharacterizationSpan:
    nested = record.get(nested_key)
    if isinstance(nested, dict):
        low = nested.get("low_hz", nested.get("f_low_hz", record.get(f"{prefix}_low_hz", 0)))
        high = nested.get("high_hz", nested.get("f_high_hz", record.get(f"{prefix}_high_hz", 0)))
        center = nested.get("center_hz", nested.get("f_center_hz", record.get(f"{prefix}_center_hz", 0)))
        bandwidth = nested.get("bandwidth_hz", record.get(f"{prefix}_bandwidth_hz", 0.0))
    else:
        low = record.get(f"{prefix}_low_hz", 0)
        high = record.get(f"{prefix}_high_hz", 0)
        center = record.get(f"{prefix}_center_hz", 0)
        bandwidth = record.get(f"{prefix}_bandwidth_hz", 0.0)
    return CharacterizationSpan.from_bounds(
        low_hz=int(low),
        high_hz=int(high),
        center_hz=int(center),
        bandwidth_hz=float(bandwidth),
    )


def summarize_characterization_record(record: Dict[str, Any]) -> Dict[str, Any]:
    evidence = CharacterizationEvidence(
        detection_id=record.get("detection_id"),
        baseline_id=int(record.get("baseline_id", 0) or 0),
        source_pass=str(record.get("source_pass") or "unknown"),
        raw_segment=_span_from_record(record, prefix="raw", nested_key="raw_segment"),
        measured_span=_span_from_record(record, prefix="measured", nested_key="measured_span"),
        match_span=_span_from_record(record, prefix="match", nested_key="match_span"),
        display_span=_span_from_record(record, prefix="display", nested_key="display_span"),
        stable_center_hz=int(
            record.get(
                "stable_center_hz",
                record.get("display_center_hz", record.get("match_center_hz", record.get("measured_center_hz", 0))),
            )
            or 0
        ),
        center_delta_hz=int(
            record.get(
                "center_delta_hz",
                int(record.get("measured_center_hz", 0) or 0)
                - int(
                    record.get(
                        "stable_center_hz",
                        record.get(
                            "display_center_hz",
                            record.get("match_center_hz", record.get("measured_center_hz", 0)),
                        ),
                    )
                    or 0
                ),
            )
            or 0
        ),
        peak_db=float(record.get("peak_db", 0.0) or 0.0),
        noise_db=float(record.get("noise_db", 0.0) or 0.0),
        snr_db=float(record.get("snr_db", 0.0) or 0.0),
        measured_bandwidth_confidence=float(record.get("measured_bandwidth_confidence", 0.0) or 0.0),
        characterization_confidence=float(record.get("characterization_confidence", 0.0) or 0.0),
        characterization_method=str(record.get("characterization_method") or "unknown"),
        center_stability_hz=float(record.get("center_stability_hz", 0.0) or 0.0),
        bandwidth_stability_hz=float(record.get("bandwidth_stability_hz", 0.0) or 0.0),
        revisit_measurement_count=int(record.get("revisit_measurement_count", 0) or 0),
        coarse_measurement_count=int(record.get("coarse_measurement_count", 0) or 0),
        classification_candidate=str(record.get("classification_candidate") or "unknown"),
        classification_evidence=list(record.get("classification_evidence") or []),
        evidence_sources=list(record.get("evidence_sources") or []),
        bandplan_service=record.get("bandplan_service"),
        bandplan_region=record.get("bandplan_region"),
        bandplan_notes=record.get("bandplan_notes"),
        profile_context=record.get("profile_context"),
        context_only=bool(record.get("context_only", False)),
        bandwidth_interpretation=str(record.get("bandwidth_interpretation") or "threshold_fragment"),
        width_floor_applied_hz=float(record.get("width_floor_applied_hz", 0.0) or 0.0),
        persist_width_floor_applied_hz=float(record.get("persist_width_floor_applied_hz", 0.0) or 0.0),
        persisted_card_bandwidth_hz=(
            float(record.get("persisted_card_bandwidth_hz"))
            if record.get("persisted_card_bandwidth_hz") not in (None, "")
            else None
        ),
        baseline_clipped=bool(record.get("baseline_clipped", False)),
        clip_reason=record.get("clip_reason"),
    )
    return evidence.to_summary()


def summarize_characterization_records(
    records: Iterable[Dict[str, Any]],
    *,
    sample_limit: int = 10,
) -> Dict[str, Any]:
    limit = max(1, int(sample_limit))
    summary: Dict[str, Any] = {
        "record_count": 0,
        "source_pass_counts": {},
        "characterization_methods": {},
        "classification_candidates": {},
        "truncated": False,
        "records": [],
    }
    for record in records:
        if not isinstance(record, dict):
            continue
        if record.get("event") != "characterization_record":
            continue
        summary["record_count"] += 1
        _count_key(summary["source_pass_counts"], record.get("source_pass"))
        _count_key(summary["characterization_methods"], record.get("characterization_method"))
        _count_key(summary["classification_candidates"], record.get("classification_candidate"))
        if len(summary["records"]) < limit:
            summary["records"].append(summarize_characterization_record(record))
    summary["truncated"] = summary["record_count"] > len(summary["records"])
    return summary


def build_window_record(
    *,
    sweep_id: Optional[int],
    window_idx: int,
    center_hz: float,
    window_low_hz: float,
    window_high_hz: float,
    profile: Optional[str],
    baseline_id: Optional[int],
    tuning_params: Dict[str, Any],
    detection_diagnostics: Dict[str, Any],
    accepted_hits: int,
    spur_ignored: int,
    promoted: int,
    new_signals: int,
    anomalous_power: bool,
    emitted_segments: list[Segment],
) -> Dict[str, Any]:
    timing = dict(tuning_params.get("timing") or {})
    for key in (
        "tune_ms",
        "flush_ms",
        "read_ms",
        "fft_ms",
        "detect_ms",
        "db_update_ms",
        "jsonl_ms",
        "total_window_ms",
    ):
        timing.setdefault(key, None)
    unavailable_fields = list(tuning_params.get("unavailable_fields") or [])
    if tuning_params.get("dropped_reads") is None and "dropped_reads" not in unavailable_fields:
        unavailable_fields.append("dropped_reads")
    record = {
        "time_utc": utc_now_str(),
        "event": "detection_window",
        "sweep_id": sweep_id,
        "window_idx": window_idx,
        "baseline_id": baseline_id,
        "center_hz": float(center_hz),
        "window_low_hz": float(window_low_hz),
        "window_high_hz": float(window_high_hz),
        "profile": profile,
        "source_profile": tuning_params.get("source_profile") or profile,
        "tuning_params": tuning_params,
        "sample_rate": tuning_params.get("sample_rate")
        or tuning_params.get("samp_rate")
        or tuning_params.get("samp_rate_hz"),
        "fft": tuning_params.get("fft"),
        "avg": tuning_params.get("avg"),
        "num_segments": int(len(emitted_segments)),
        "samples_requested": tuning_params.get("samples_requested"),
        "samples_read": tuning_params.get("samples_read"),
        "short_read": tuning_params.get("short_read"),
        "dropped_reads": tuning_params.get("dropped_reads"),
        "timing": timing,
        "unavailable_fields": unavailable_fields,
        "threshold_info": detection_diagnostics.get("threshold_info", {}),
        "bins_above_threshold": int(detection_diagnostics.get("bins_above_threshold", 0) or 0),
        "raw_candidate_segment_count": int(
            detection_diagnostics.get("raw_candidate_segment_count", 0) or 0
        ),
        "final_emitted_segment_count": int(
            detection_diagnostics.get("final_emitted_segment_count", len(emitted_segments)) or 0
        ),
        "accepted_hits": int(accepted_hits),
        "spur_ignored": int(spur_ignored),
        "promoted": int(promoted),
        "new_signals": int(new_signals),
        "anomalous_power": bool(anomalous_power),
        "segments": detection_diagnostics.get("segments", [segment_to_dict(seg) for seg in emitted_segments]),
    }
    record.update(_provenance_fields(tuning_params))
    return record


class DetectionDiagnosticWriter:
    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        if not self.path.is_absolute():
            self.path = (Path.cwd() / self.path).absolute()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: Dict[str, Any]) -> None:
        try:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, sort_keys=True) + "\n")
        except Exception as exc:
            print(f"[detection_diagnostics] write failed to {self.path}: {exc}", file=sys.stderr)
