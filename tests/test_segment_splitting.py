import numpy as np

from sdrwatch.dsp.detection import detect_segments, split_segment_by_valleys


def _build_freq_axis(length: int, start: float = 100e6, stop: float = 101e6) -> np.ndarray:
    return np.linspace(start, stop, num=length, dtype=np.float64)


def _make_noise(shape, level: float = -90.0) -> np.ndarray:
    return np.full(shape, level, dtype=np.float64)


def test_split_segment_by_valleys_returns_two_ranges_for_deep_valley() -> None:
    psd = np.full(64, -90.0)
    psd[8:12] = np.array([-55.0, -32.0, -30.0, -48.0])
    psd[12:16] = np.array([-58.0, -70.0, -65.0, -55.0])
    psd[16:20] = np.array([-48.0, -31.0, -29.0, -46.0])
    noise = _make_noise(psd.shape)
    ranges = split_segment_by_valleys(
        psd,
        noise,
        6,
        22,
        drop_db=4.0,
        noise_margin_db=1.5,
        min_valley_bins=2,
        min_segment_bins=2,
        min_peak_prominence_db=2.0,
    )
    assert len(ranges) == 2
    first, second = ranges
    assert first[0] == 6
    assert second[0] >= first[1]


def test_detect_segments_emits_single_segment_for_one_peak() -> None:
    psd = np.full(64, -90.0)
    psd[10:14] = np.array([-55.0, -33.0, -30.0, -46.0])
    freqs = _build_freq_axis(psd.size)
    segs, _, _ = detect_segments(
        freqs,
        psd,
        thresh_db=6.0,
        guard_bins=1,
        min_width_bins=2,
        cfar_mode="off",
    )
    assert len(segs) == 1


def test_detect_segments_splits_multi_peak_lobe_with_deep_valley() -> None:
    psd = np.full(64, -90.0)
    psd[8:12] = np.array([-55.0, -32.0, -30.0, -48.0])
    psd[12:16] = np.array([-58.0, -70.0, -65.0, -55.0])
    psd[16:20] = np.array([-48.0, -31.0, -29.0, -46.0])
    freqs = _build_freq_axis(psd.size)
    segs, _, _ = detect_segments(
        freqs,
        psd,
        thresh_db=6.0,
        guard_bins=1,
        min_width_bins=2,
        cfar_mode="off",
    )
    assert len(segs) == 2
    centers = sorted(seg.f_center_hz for seg in segs)
    assert centers[1] - centers[0] > 5_000


def test_detect_segments_keeps_single_segment_for_shallow_valley() -> None:
    psd = np.full(64, -90.0)
    psd[8:12] = np.array([-35.0, -31.5, -29.0, -31.0])
    psd[12:16] = np.array([-31.5, -31.2, -31.0, -31.4])
    psd[16:20] = np.array([-31.0, -29.5, -29.0, -30.5])
    freqs = _build_freq_axis(psd.size)
    segs, _, _ = detect_segments(
        freqs,
        psd,
        thresh_db=6.0,
        guard_bins=1,
        min_width_bins=2,
        cfar_mode="off",
    )
    assert len(segs) == 1