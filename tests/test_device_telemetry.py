"""Device and gain telemetry contract tests."""

from __future__ import annotations

from types import SimpleNamespace

from sdrwatch.util.detection_diagnostics import build_device_telemetry_snapshot


class FakeRTLDevice:
    gain = 29.7
    valid_gains_db = [0.0, 9.9, 19.7, 29.7]
    serial_number = "00000001"
    tuner_type = "R820T2"
    sample_rate = 2_400_000


class FakeRTLSource:
    device = "RTL-SDR #0"
    device_index = 0
    dev = FakeRTLDevice()


def _args(**overrides):
    values = {
        "gain": "30",
        "samp_rate": 2_400_000,
        "fft": 8192,
        "profile": "fm_broadcast",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_device_telemetry_records_available_rtl_sdr_fields() -> None:
    telemetry = build_device_telemetry_snapshot(_args(), FakeRTLSource(), device_key="rtl:0")

    assert telemetry["event"] == "device_telemetry"
    assert telemetry["device_key"] == "rtl:0"
    assert telemetry["device_index"] == 0
    assert telemetry["device_serial"] == "00000001"
    assert telemetry["device_tuner"] == "R820T2"
    assert telemetry["device_label"] == "RTL-SDR #0"
    assert telemetry["driver"] == "rtlsdr_native"
    assert telemetry["requested_gain"] == "30"
    assert telemetry["gain_mode"] == "manual"
    assert telemetry["actual_gain"] == 29.7
    assert telemetry["supported_gains"] == [0.0, 9.9, 19.7, 29.7]
    assert telemetry["sample_rate_hz"] == 2_400_000
    assert telemetry["actual_sample_rate_hz"] == 2_400_000
    assert telemetry["fft"] == 8192
    assert telemetry["bin_width_hz"] == 2_400_000 / 8192
    assert telemetry["selected_profile"] == "fm_broadcast"
    assert telemetry["unavailable_fields"] == []


def test_device_telemetry_marks_missing_driver_fields_unavailable_without_failure() -> None:
    telemetry = build_device_telemetry_snapshot(_args(gain="auto", profile=None), object(), device_key="rtl:0")

    assert telemetry["requested_gain"] == "auto"
    assert telemetry["gain_mode"] == "auto"
    assert telemetry["actual_gain"] is None
    assert telemetry["supported_gains"] is None
    assert telemetry["device_serial"] is None
    assert telemetry["device_tuner"] is None
    assert telemetry["actual_sample_rate_hz"] is None
    assert set(telemetry["unavailable_fields"]) >= {
        "actual_gain",
        "supported_gains",
        "device_serial",
        "device_tuner",
        "actual_sample_rate_hz",
    }
