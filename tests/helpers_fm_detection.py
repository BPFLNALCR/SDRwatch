"""Shared no-hardware FM detection fixtures."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable
import json

from sdrwatch.baseline.store import Store
from sdrwatch.detection.engine import DetectionEngine
from sdrwatch.detection.types import Segment
from sdrwatch.io.bandplan import Bandplan


FM_START_HZ = 88_000_000
FM_STOP_HZ = 108_000_000
FM_BIN_HZ = 2_400_000.0 / 8192.0


class ListLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []

    def log(self, event: str, **fields: Any) -> None:
        self.records.append({"event": event, **fields})

    def events(self, event: str) -> list[dict[str, Any]]:
        return [record for record in self.records if record.get("event") == event]


def fm_args(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "profile": "fm_broadcast",
        "min_width_bins": 5,
        "persistence_mode": "hits",
        "persistence_hit_ratio": 0.25,
        "persistence_min_seconds": 2.0,
        "persistence_min_hits": 1,
        "persistence_min_windows": 1,
        "new_ema_occ": 0.02,
        "notify": False,
        "jsonl": None,
        "diagnostic_jsonl": None,
        "two_pass": True,
        "revisit_margin_hz": 200_000.0,
        "revisit_span_limit_hz": 420_000.0,
        "cluster_merge_hz": 12_000.0,
        "center_match_hz": 60_000.0,
        "max_detection_width_ratio": 2.5,
        "max_detection_width_hz": 270_000.0,
        "match_bandwidth_pad_hz": 10_000.0,
        "min_match_bandwidth_hz": 80_000.0,
        "display_bandwidth_pad_hz": 30_000.0,
        "min_display_bandwidth_hz": 200_000.0,
        "confidence_hit_normalizer": 2.0,
        "confidence_duration_norm": 2.0,
        "confidence_bias": 0.05,
        "spur_calibration": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def narrow_args(**overrides: Any) -> SimpleNamespace:
    values: dict[str, Any] = {
        "profile": None,
        "min_width_bins": 2,
        "persistence_mode": "hits",
        "persistence_hit_ratio": 0.0,
        "persistence_min_seconds": 0.0,
        "persistence_min_hits": 1,
        "persistence_min_windows": 1,
        "new_ema_occ": 0.02,
        "notify": False,
        "jsonl": None,
        "diagnostic_jsonl": None,
        "two_pass": False,
        "cluster_merge_hz": None,
        "center_match_hz": None,
        "max_detection_width_ratio": 3.0,
        "max_detection_width_hz": 0.0,
        "match_bandwidth_pad_hz": None,
        "min_match_bandwidth_hz": None,
        "display_bandwidth_pad_hz": None,
        "min_display_bandwidth_hz": None,
        "confidence_hit_normalizer": 6.0,
        "confidence_duration_norm": 8.0,
        "confidence_bias": 0.0,
        "spur_calibration": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def make_segment(center_hz: int, width_hz: int = 2_000, snr_db: float = 24.0) -> Segment:
    half = int(round(width_hz / 2.0))
    return Segment(
        f_low_hz=int(center_hz - half),
        f_high_hz=int(center_hz + half),
        f_center_hz=int(center_hz),
        peak_db=-35.0,
        noise_db=-80.0,
        snr_db=float(snr_db),
        bandwidth_hz=float(width_hz),
    )


def make_store(tmp_path: Path, *, start_hz: int = FM_START_HZ, stop_hz: int = FM_STOP_HZ):
    store = Store(str(tmp_path / "sdrwatch.db"))
    ctx = store.create_baseline(
        name="fm-test",
        freq_start_hz=start_hz,
        freq_stop_hz=stop_hz,
        bin_hz=FM_BIN_HZ,
    )
    return store, ctx


def enable_isolated_jsonl_paths(tmp_path: Path, args: SimpleNamespace) -> SimpleNamespace:
    if getattr(args, "jsonl", None) in (None, ""):
        args.jsonl = str(tmp_path / "signals.jsonl")
    if getattr(args, "diagnostic_jsonl", None) in (None, ""):
        args.diagnostic_jsonl = str(tmp_path / "diagnostic.jsonl")
    return args


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(json.loads(line))
    return records


def make_engine(
    tmp_path: Path,
    *,
    args: SimpleNamespace | None = None,
    logger: ListLogger | None = None,
    isolated_jsonl: bool = False,
):
    store, ctx = make_store(tmp_path)
    args = args or fm_args()
    if isolated_jsonl:
        args = enable_isolated_jsonl_paths(tmp_path, args)
    logger = logger or ListLogger()
    engine = DetectionEngine(
        store=store,
        bandplan=Bandplan(None),
        args=args,
        bin_hz=FM_BIN_HZ,
        baseline_ctx=ctx,
        min_hits=int(getattr(args, "persistence_min_hits", 1)),
        min_windows=int(getattr(args, "persistence_min_windows", 1)),
        logger=logger,
    )
    return engine, store, ctx, logger


def ingest_segments(engine: DetectionEngine, centers_hz: Iterable[int], *, width_hz: int = 2_000) -> None:
    for window_idx, center_hz in enumerate(centers_hz):
        engine.ingest(window_idx, [make_segment(center_hz, width_hz=width_hz)])
