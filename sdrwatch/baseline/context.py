"""Baseline context resolution helpers."""

from __future__ import annotations

from typing import Any, Optional, Tuple

from sdrwatch.baseline.store import BaselineContext, Store


def resolve_baseline_context(
    store: Store,
    baseline_id_raw: Optional[Any],
    *,
    span_hint: Optional[Tuple[float, float]] = None,
    bin_hz_hint: Optional[float] = None,
) -> BaselineContext:
    """Return an existing baseline or create a new one using the provided hints."""

    if baseline_id_raw is None:
        return _create_from_hints(store, span_hint=span_hint, bin_hz_hint=bin_hz_hint)

    if isinstance(baseline_id_raw, str) and baseline_id_raw.lower() == "latest":
        latest_id = store.get_latest_baseline_id()
        if latest_id is None:
            return _create_from_hints(store, span_hint=span_hint, bin_hz_hint=bin_hz_hint)
        ctx = store.get_baseline(latest_id)
        if ctx is None:
            return _create_from_hints(store, span_hint=span_hint, bin_hz_hint=bin_hz_hint)
        return ctx

    try:
        baseline_id = int(baseline_id_raw)
    except (TypeError, ValueError):  # pragma: no cover - defensive CLI parsing
        raise SystemExit("--baseline-id must be an integer or 'latest'")

    ctx = store.get_baseline(baseline_id)
    if ctx is None:
        raise SystemExit(f"Baseline id {baseline_id} not found. Create it via the controller/web UI first.")
    return ctx


def _create_from_hints(
    store: Store,
    *,
    span_hint: Optional[Tuple[float, float]] = None,
    bin_hz_hint: Optional[float] = None,
) -> BaselineContext:
    freq_start = int(span_hint[0]) if span_hint else 0
    freq_stop = int(span_hint[1]) if span_hint else 0
    bin_hz = float(bin_hz_hint or 0.0)
    ctx = store.create_baseline(freq_start_hz=freq_start, freq_stop_hz=freq_stop, bin_hz=bin_hz)
    print(
        f"[baseline] created id={ctx.id} name='{ctx.name}' span_hint={freq_start}-{freq_stop} bin={bin_hz:.1f} Hz",
        flush=True,
    )
    return ctx
