import numpy as np

from sdrwatch.dsp.detection import detect_segments


def _freq_axis(length: int) -> np.ndarray:
    return np.linspace(100_000_000, 101_000_000, num=length, dtype=np.float64)


def _run_with_diagnostics(psd: np.ndarray) -> dict:
    diagnostics: dict = {}
    detect_segments(
        _freq_axis(psd.size),
        psd,
        thresh_db=6.0,
        guard_bins=1,
        min_width_bins=2,
        cfar_mode="off",
        diagnostics=diagnostics,
    )
    return diagnostics


def test_diagnostics_shape_for_strong_narrow_signal() -> None:
    psd = np.full(64, -90.0)
    psd[10:14] = np.array([-55.0, -33.0, -30.0, -46.0])

    diagnostics = _run_with_diagnostics(psd)

    assert diagnostics["threshold_info"]["mode"] == "off"
    assert diagnostics["bins_above_threshold"] == 4
    assert diagnostics["raw_candidate_segment_count"] == 1
    assert diagnostics["final_emitted_segment_count"] == 1
    assert len(diagnostics["segments"]) == 1
    segment = diagnostics["segments"][0]
    assert {
        "f_low_hz",
        "f_high_hz",
        "f_center_hz",
        "bandwidth_hz",
        "peak_db",
        "noise_db",
        "snr_db",
    }.issubset(segment)
    assert segment["snr_db"] > 50.0


def test_diagnostics_shape_for_two_nearby_signals_split_by_valley() -> None:
    psd = np.full(64, -90.0)
    psd[8:12] = np.array([-55.0, -32.0, -30.0, -48.0])
    psd[12:16] = np.array([-58.0, -70.0, -65.0, -55.0])
    psd[16:20] = np.array([-48.0, -31.0, -29.0, -46.0])

    diagnostics = _run_with_diagnostics(psd)

    assert diagnostics["bins_above_threshold"] == 12
    assert diagnostics["raw_candidate_segment_count"] == 2
    assert diagnostics["final_emitted_segment_count"] == 2
    centers = sorted(segment["f_center_hz"] for segment in diagnostics["segments"])
    assert centers[1] > centers[0]


def test_diagnostics_shape_for_noise_only_input() -> None:
    psd = np.full(64, -90.0)

    diagnostics = _run_with_diagnostics(psd)

    assert diagnostics["threshold_info"]["mode"] == "off"
    assert diagnostics["bins_above_threshold"] == 0
    assert diagnostics["raw_candidate_segment_count"] == 0
    assert diagnostics["final_emitted_segment_count"] == 0
    assert diagnostics["segments"] == []
