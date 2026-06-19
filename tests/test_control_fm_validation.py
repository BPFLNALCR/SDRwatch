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


def test_controller_command_passes_supported_characterization_revisit_and_persistence_params() -> None:
    params = {
        "start": 88_000_000,
        "stop": 108_000_000,
        "profile": "fm_broadcast",
        "segment_center_mode": "centroid",
        "segment_centroid_span_hz": 240_000,
        "segment_centroid_drop_db": 20,
        "segment_centroid_floor_margin_db": 2,
        "match_bandwidth_pad_hz": 10_000,
        "min_match_bandwidth_hz": 80_000,
        "min_identity_bandwidth_hz": 80_000,
        "min_persist_bandwidth_hz": 80_000,
        "max_persist_bandwidth_hz": 270_000,
        "display_bandwidth_pad_hz": 30_000,
        "min_display_bandwidth_hz": 200_000,
        "min_revisit_bandwidth_for_identity_update_hz": 80_000,
        "max_revisit_center_delta_for_identity_update_hz": 60_000,
        "fragmented_revisit_policy": "confirmation_only",
        "raw_fragment_interpretation": "threshold_fragment",
        "allow_revisit_to_shrink_identity": False,
        "allow_revisit_to_move_center": True,
        "center_smoothing_enabled": True,
        "max_persist_width_hz": 270_000,
        "max_card_width_hz": 270_000,
        "center_match_hz": 60_000,
        "persistence_mode": "hits",
        "persistence_hit_ratio": 0.25,
        "persistence_min_seconds": 2,
        "persistence_min_hits": 1,
        "persistence_min_windows": 1,
        "persistence_min_sweep_loops": 3,
        "two_pass": True,
        "revisit_fft": 32768,
        "revisit_avg": 4,
        "revisit_margin_hz": 200_000,
        "revisit_span_limit_hz": 420_000,
        "revisit_max_bands": 40,
        "revisit_floor_threshold_db": 6,
    }

    cmd = build_scanner_cmd(params)

    expected_flags = {
        "--segment-center-mode": "centroid",
        "--segment-centroid-span-hz": "240000",
        "--segment-centroid-drop-db": "20",
        "--segment-centroid-floor-margin-db": "2",
        "--match-bandwidth-pad-hz": "10000",
        "--min-match-bandwidth-hz": "80000",
        "--min-identity-bandwidth-hz": "80000",
        "--min-persist-bandwidth-hz": "80000",
        "--max-persist-bandwidth-hz": "270000",
        "--display-bandwidth-pad-hz": "30000",
        "--min-display-bandwidth-hz": "200000",
        "--min-revisit-bandwidth-for-identity-update-hz": "80000",
        "--max-revisit-center-delta-for-identity-update-hz": "60000",
        "--max-detection-width-hz": "270000",
        "--center-match-hz": "60000",
        "--persistence-min-sweep-loops": "3",
        "--revisit-span-limit-hz": "420000",
    }
    for flag, value in expected_flags.items():
        assert cmd[cmd.index(flag) + 1] == value
    assert cmd[cmd.index("--fragmented-revisit-policy") + 1] == "confirmation_only"
    assert cmd[cmd.index("--raw-fragment-interpretation") + 1] == "threshold_fragment"
    assert "--no-allow-revisit-to-shrink-identity" in cmd
    assert "--allow-revisit-to-move-center" in cmd
    assert "--center-smoothing-enabled" in cmd
    assert "--two-pass" in cmd


def test_controller_command_passes_role_metadata_flags_without_backend_change() -> None:
    cmd = build_scanner_cmd(
        {
            "start": 101_100_000,
            "stop": 103_500_000,
            "step": 2_400_000,
            "receiver_role": "GUARD",
            "role_lane": "guard_primary",
            "role_run_id": "rr-1",
            "job_id": "job-1",
            "source_task": "guard_window",
            "device_identity": "rtl:serial:S1",
            "device_serial": "S1",
            "device_index": 0,
            "identity_confidence": "stable",
            "active_device_count": 1,
            "active_role_count": 1,
        }
    )

    expected_flags = {
        "--receiver-role": "GUARD",
        "--role-lane": "guard_primary",
        "--role-run-id": "rr-1",
        "--job-id": "job-1",
        "--source-task": "guard_window",
        "--device-identity": "rtl:serial:S1",
        "--device-serial": "S1",
        "--device-index": "0",
        "--identity-confidence": "stable",
        "--active-device-count": "1",
        "--active-role-count": "1",
    }
    for flag, value in expected_flags.items():
        assert cmd[cmd.index(flag) + 1] == value
    assert cmd[cmd.index("--driver") + 1] == "rtlsdr_native"
