"""FM Validation profile and scanner-application tests."""

from __future__ import annotations

from sdrwatch import cli
from sdrwatch.io.profiles import serialize_profiles


def _fm_profile_payload() -> dict:
    profiles = serialize_profiles()["profiles"]
    return next(profile for profile in profiles if profile["name"] == "fm_broadcast")


def test_fm_broadcast_profile_serializes_stability_fields() -> None:
    profile = _fm_profile_payload()

    assert profile["f_low_hz"] == 88_000_000
    assert profile["f_high_hz"] == 108_000_000
    assert profile["samp_rate"] == 2.4e6
    assert profile["step_hz"] == 1.2e6
    assert profile["fft"] == 8192
    assert profile["avg"] == 10
    assert profile["gain_db"] == 20.0
    assert profile["threshold_db"] == 6.0
    assert profile["two_pass"] is True
    assert profile["cluster_merge_hz"] == 12_000.0
    assert profile["center_match_hz"] == 60_000.0
    assert profile["min_match_bandwidth_hz"] == 80_000.0
    assert profile["min_display_bandwidth_hz"] == 200_000.0
    assert profile["max_detection_width_hz"] == 270_000.0
    assert profile["segment_center_mode"] == "centroid"


def test_cli_applies_fm_profile_hidden_stability_fields(monkeypatch) -> None:
    monkeypatch.setattr(cli, "HAVE_RTLSDR", True)

    args = cli.parse_args(
        [
            "--start",
            "88000000",
            "--stop",
            "108000000",
            "--baseline-id",
            "1",
            "--profile",
            "fm_broadcast",
        ]
    )

    assert args.profile == "fm_broadcast"
    assert args.step == 1.2e6
    assert args.avg == 10
    assert args.gain == 20.0
    assert args.two_pass is True
    assert args.revisit_fft == 32768
    assert args.revisit_span_limit_hz == 420_000.0
    assert args.center_match_hz == 60_000.0
    assert args.match_bandwidth_pad_hz == 10_000.0
    assert args.min_match_bandwidth_hz == 80_000.0
    assert args.display_bandwidth_pad_hz == 30_000.0
    assert args.min_display_bandwidth_hz == 200_000.0
    assert args.segment_center_mode == "centroid"
    assert args.segment_centroid_span_hz == 240_000.0


def test_cli_profile_preserves_explicit_operator_overrides(monkeypatch) -> None:
    monkeypatch.setattr(cli, "HAVE_RTLSDR", True)

    args = cli.parse_args(
        [
            "--start",
            "88000000",
            "--stop",
            "108000000",
            "--baseline-id",
            "1",
            "--profile",
            "fm_broadcast",
            "--avg",
            "8",
            "--gain",
            "30",
        ]
    )

    assert args.avg == 8
    assert args.gain == "30"
    assert args.two_pass is True
    assert args.min_match_bandwidth_hz == 80_000.0
