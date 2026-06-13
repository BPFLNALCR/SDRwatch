"""Controller command coverage for GUI FM Validation params."""

from __future__ import annotations

from tests.helpers_control import build_scanner_cmd


def test_controller_command_passes_fm_validation_profile_two_pass_and_bounds() -> None:
    cmd = build_scanner_cmd(
        {
            "start": 88_000_000,
            "stop": 108_000_000,
            "samp_rate": 2_400_000,
            "step": 1_200_000,
            "fft": 8192,
            "avg": 10,
            "threshold_db": 6,
            "guard_bins": 3,
            "min_width_bins": 5,
            "cfar": "os",
            "cfar_train": 32,
            "cfar_guard": 6,
            "cfar_quantile": 0.6,
            "persistence_hit_ratio": 0.25,
            "persistence_min_seconds": 2,
            "persistence_min_hits": 1,
            "persistence_min_windows": 1,
            "cluster_merge_hz": 12_000,
            "max_detection_width_ratio": 2.5,
            "max_detection_width_hz": 270_000,
            "profile": "fm_broadcast",
            "two_pass": True,
            "revisit_fft": 32768,
            "revisit_avg": 4,
            "revisit_margin_hz": 200_000,
            "revisit_span_limit_hz": 420_000,
            "revisit_max_bands": 40,
            "revisit_floor_threshold_db": 6,
        }
    )

    assert cmd[cmd.index("--profile") + 1] == "fm_broadcast"
    assert "--two-pass" in cmd
    assert cmd[cmd.index("--revisit-fft") + 1] == "32768"
    assert cmd[cmd.index("--revisit-span-limit-hz") + 1] == "420000"
    assert cmd[cmd.index("--cluster-merge-hz") + 1] == "12000"
    assert cmd[cmd.index("--max-detection-width-hz") + 1] == "270000"


def test_controller_command_keeps_discovery_without_profile_or_two_pass() -> None:
    cmd = build_scanner_cmd(
        {
            "start": 88_000_000,
            "stop": 108_000_000,
            "samp_rate": 2_400_000,
            "step": 2_400_000,
            "fft": 8192,
            "avg": 8,
            "threshold_db": 8,
            "persistence_min_hits": 1,
            "persistence_min_windows": 1,
        }
    )

    assert "--profile" not in cmd
    assert "--two-pass" not in cmd
    assert cmd[cmd.index("--step") + 1] == "2400000"


def test_controller_command_ignores_characterization_only_fields() -> None:
    cmd = build_scanner_cmd(
        {
            "start": 88_000_000,
            "stop": 108_000_000,
            "samp_rate": 2_400_000,
            "step": 1_200_000,
            "fft": 8192,
            "avg": 10,
            "threshold_db": 6,
            "guard_bins": 3,
            "min_width_bins": 5,
            "profile": "fm_broadcast",
            "two_pass": True,
            "measured_center_hz": 100_100_000,
            "measured_bandwidth_hz": 180_000,
            "characterization_confidence": 0.9,
            "classification_candidate": "fm_broadcast_candidate",
            "profile_context": "fm_broadcast",
        }
    )

    assert "--profile" in cmd
    assert "--two-pass" in cmd
    assert "--measured-center-hz" not in cmd
    assert "--measured-bandwidth-hz" not in cmd
    assert "--characterization-confidence" not in cmd
    assert "--classification-candidate" not in cmd
    assert "--profile-context" not in cmd
