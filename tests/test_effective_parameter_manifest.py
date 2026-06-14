"""Effective-parameter manifest contract tests."""

from __future__ import annotations

from types import SimpleNamespace

from sdrwatch.util.detection_diagnostics import build_effective_parameter_manifest


def _args(**overrides):
    values = {
        "job_id": "job-007",
        "start": 88_000_000,
        "stop": 108_000_000,
        "step": 1_200_000,
        "samp_rate": 2_400_000,
        "fft": 8192,
        "avg": 10,
        "gain": "30",
        "driver": "rtlsdr_native",
        "profile": "fm_broadcast",
        "persistence_mode": "hits",
        "persistence_hit_ratio": 0.25,
        "persistence_min_seconds": 2.0,
        "persistence_min_hits": 1,
        "persistence_min_windows": 1,
        "persistence_min_sweep_loops": 2,
        "two_pass": True,
        "revisit_fft": 32768,
        "revisit_avg": 4,
        "revisit_margin_hz": 200_000,
        "revisit_span_limit_hz": 420_000,
        "revisit_max_bands": 40,
        "revisit_floor_threshold_db": 6,
        "segment_center_mode": "centroid",
        "segment_centroid_span_hz": 240_000,
        "segment_centroid_drop_db": 20,
        "segment_centroid_floor_margin_db": 2,
        "match_bandwidth_pad_hz": 10_000,
        "min_match_bandwidth_hz": 80_000,
        "display_bandwidth_pad_hz": 30_000,
        "min_display_bandwidth_hz": 200_000,
        "max_detection_width_hz": 270_000,
        "center_match_hz": 60_000,
        "_requested_profile": "fm_broadcast",
        "_applied_profile": "fm_broadcast",
        "_profile_applied": True,
        "_profile_skip_reason": None,
        "_operator_overrides": {"gain": "30"},
        "_profile_defaults": {"step_hz": 1_200_000, "samp_rate": 2_400_000, "fft": 8192},
        "_fallback_defaults": {},
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_effective_parameter_manifest_records_in_band_fm_profile_application() -> None:
    manifest = build_effective_parameter_manifest(_args(), job_id="abc123")

    assert manifest["job_id"] == "abc123"
    assert manifest["requested_profile"] == "fm_broadcast"
    assert manifest["applied_profile"] == "fm_broadcast"
    assert manifest["profile_applied"] is True
    assert manifest["profile_skip_reason"] is None
    assert manifest["operator_overrides"] == {"gain": "30"}
    assert manifest["profile_defaults"]["fft"] == 8192
    assert manifest["fallback_defaults"] == {}
    assert manifest["final_effective_params"]["start_hz"] == 88_000_000
    assert manifest["final_effective_params"]["stop_hz"] == 108_000_000
    assert manifest["final_effective_params"]["bin_width_hz"] == 2_400_000 / 8192
    assert manifest["persistence"]["min_sweep_loops"] == 2
    assert manifest["revisit"]["two_pass"] is True
    assert manifest["span_controls"]["segment_center_mode"] == "centroid"
    assert manifest["span_controls"]["max_persist_width_hz"] == 270_000
    assert manifest["span_controls"]["max_card_width_hz"] == 270_000
    assert manifest["gain"]["requested_gain"] == "30"


def test_effective_parameter_manifest_records_out_of_band_profile_skip_and_fallbacks() -> None:
    manifest = build_effective_parameter_manifest(
        _args(
            start=120_000_000,
            stop=130_000_000,
            step=2_400_000,
            profile="fm_broadcast",
            _applied_profile=None,
            _profile_applied=False,
            _profile_skip_reason="requested span 120.000-130.000MHz outside fm_broadcast 88.000-108.000MHz",
            _profile_defaults={},
            _fallback_defaults={"step_hz": 2_400_000, "fft": 4096},
        ),
        job_id="outband",
    )

    assert manifest["requested_profile"] == "fm_broadcast"
    assert manifest["applied_profile"] is None
    assert manifest["profile_applied"] is False
    assert "outside fm_broadcast" in manifest["profile_skip_reason"]
    assert manifest["fallback_defaults"]["step_hz"] == 2_400_000
    assert manifest["final_effective_params"]["start_hz"] == 120_000_000
    assert manifest["final_effective_params"]["stop_hz"] == 130_000_000
