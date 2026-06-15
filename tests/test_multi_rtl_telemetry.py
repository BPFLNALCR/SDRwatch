"""Role-aware telemetry/provenance diagnostic tests."""

from __future__ import annotations

from types import SimpleNamespace

from sdrwatch.detection.types import Segment
from sdrwatch.util.detection_diagnostics import (
    build_device_telemetry_snapshot,
    build_effective_parameter_manifest,
    build_resource_telemetry_snapshot,
    build_window_record,
)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        driver="rtlsdr_native",
        device_key="rtl:0",
        job_id="job-1",
        role_run_id="rr-1",
        receiver_role="GUARD",
        role_lane="guard_primary",
        source_task="guard_window",
        device_identity="rtl:serial:S1",
        device_serial="S1",
        device_index=0,
        identity_confidence="stable",
        active_device_count=1,
        active_role_count=1,
        profile="fm_broadcast",
        samp_rate=2_400_000,
        fft=4096,
        avg=8,
        gain="20",
    )


def test_device_and_effective_parameter_records_include_role_provenance() -> None:
    args = _args()

    telemetry = build_device_telemetry_snapshot(args, None, device_key=args.device_key)
    manifest = build_effective_parameter_manifest(args, job_id=args.job_id, device_telemetry=telemetry)

    for record in (telemetry, manifest):
        assert record["job_id"] == "job-1"
        assert record["role_run_id"] == "rr-1"
        assert record["receiver_role"] == "GUARD"
        assert record["role_lane"] == "guard_primary"
        assert record["source_task"] == "guard_window"
        assert record["device_identity"] == "rtl:serial:S1"
        assert record["device_serial"] == "S1"
        assert record["device_index"] == 0
        assert record["active_device_count"] == 1
        assert record["active_role_count"] == 1
    assert manifest["source_profile"] == "fm_broadcast"
    assert manifest["runnable_backend"] == "rtlsdr_native"


def test_window_record_includes_timing_sample_accounting_and_unavailable_fields() -> None:
    args = _args()
    tuning_params = vars(args).copy()
    tuning_params.update(
        {
            "sample_rate": 2_400_000,
            "samples_requested": 32768,
            "samples_read": 32000,
            "short_read": True,
            "dropped_reads": None,
            "timing": {
                "tune_ms": 1.0,
                "flush_ms": 2.0,
                "read_ms": 3.0,
                "fft_ms": 4.0,
                "detect_ms": 5.0,
                "db_update_ms": 6.0,
                "jsonl_ms": 7.0,
                "total_window_ms": 28.0,
            },
            "unavailable_fields": ["dropped_reads"],
        }
    )

    record = build_window_record(
        sweep_id=1,
        window_idx=0,
        center_hz=102_300_000,
        window_low_hz=101_100_000,
        window_high_hz=103_500_000,
        profile="fm_broadcast",
        baseline_id=1,
        tuning_params=tuning_params,
        detection_diagnostics={},
        accepted_hits=1,
        spur_ignored=0,
        promoted=0,
        new_signals=0,
        anomalous_power=False,
        emitted_segments=[
            Segment(
                f_low_hz=102_000_000,
                f_high_hz=102_200_000,
                f_center_hz=102_100_000,
                peak_db=-20.0,
                noise_db=-40.0,
                snr_db=20.0,
                bandwidth_hz=200_000.0,
            )
        ],
    )

    assert record["device_identity"] == "rtl:serial:S1"
    assert record["samples_requested"] == 32768
    assert record["samples_read"] == 32000
    assert record["short_read"] is True
    assert record["dropped_reads"] is None
    assert record["num_segments"] == 1
    assert record["timing"]["db_update_ms"] == 6.0
    assert "dropped_reads" in record["unavailable_fields"]


def test_resource_telemetry_marks_unavailable_platform_fields() -> None:
    record = build_resource_telemetry_snapshot(_args(), pid=1234)

    assert record["event"] == "resource_telemetry"
    assert record["pid"] == 1234
    assert record["sample_rate"] == 2_400_000
    assert record["active_device_count"] == 1
    assert record["active_role_count"] == 1
    assert "cpu_load" in record
    assert "rss_memory_bytes" in record
    assert isinstance(record["unavailable_fields"], list)
