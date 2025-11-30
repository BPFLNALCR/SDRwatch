"""Baseline event + scan update helpers."""

from __future__ import annotations

from typing import Optional

from sdrwatch.baseline.model import BaselineContext
from sdrwatch.baseline.store import Store
from sdrwatch.util.scan_logger import ScanLogger
from sdrwatch.util.time import utc_now_str


class BaselineEventWriter:
    """Persist scan update counters and emit structured logs."""

    def __init__(self, store: Store, baseline_ctx: BaselineContext, logger: Optional[ScanLogger] = None) -> None:
        self.store = store
        self.baseline_ctx = baseline_ctx
        self.logger = logger

    def record_scan_summary(
        self,
        *,
        hits: int,
        segments: int,
        promoted: int,
        new_signals: int,
        revisits_total: int,
        revisits_confirmed: int,
        revisits_false_positive: int,
    ) -> None:
        timestamp = utc_now_str()
        self.store.begin()
        self.store.insert_scan_update(
            self.baseline_ctx.id,
            timestamp,
            num_hits=hits,
            num_segments=segments,
            num_new_signals=new_signals,
            num_revisits=revisits_total,
            num_confirmed=revisits_confirmed,
            num_false_positive=revisits_false_positive,
        )
        self.store.commit()
        try:
            self.store.refresh_baseline_snapshot(self.baseline_ctx, last_update_utc=timestamp)
        except Exception:
            if self.logger:
                self.logger.log(
                    "baseline_snapshot_error",
                    baseline_id=self.baseline_ctx.id,
                    error="snapshot_refresh_failed",
                )
        if self.logger:
            self.logger.log(
                "sweep_summary",
                baseline_id=self.baseline_ctx.id,
                hits=hits,
                segments=segments,
                promoted=promoted,
                new_signals=new_signals,
                revisits_total=revisits_total,
                revisits_confirmed=revisits_confirmed,
                revisits_false_positive=revisits_false_positive,
            )
