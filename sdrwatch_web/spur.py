"""
Spur map helpers for SDRwatch Web.

Provides functions for loading spur calibration data and annotating
detections that fall near known spur frequencies.
"""
from __future__ import annotations

from bisect import bisect_left
from typing import Any, Dict, List, Tuple

from sdrwatch_web.db import get_con_optional, qa


def load_spur_bins() -> List[int]:
    """
    Load known spur frequencies from the spur_map table.

    Returns:
        Sorted list of spur frequencies in Hz.
    """
    connection = get_con_optional()
    if connection is None:
        return []

    try:
        rows = qa(connection, "SELECT bin_hz FROM spur_map")
    except Exception:
        return []

    bins: List[int] = []
    for row in rows:
        try:
            val = row.get("bin_hz") if isinstance(row, dict) else row[0]
            if val is not None:
                bins.append(int(val))
        except Exception:
            continue

    bins.sort()
    return bins


def annotate_near_spur(
    records: List[Dict[str, Any]],
    bins: List[int],
    *,
    tolerance_hz: int = 5_000,
) -> None:
    """
    Annotate records with a "near_spur" flag based on proximity to spur frequencies.

    Modifies records in place to add a "near_spur" boolean key.

    Args:
        records: List of detection dicts with f_center_hz.
        bins: Sorted list of spur frequencies in Hz.
        tolerance_hz: Maximum distance from a spur to be considered "near".
    """
    if not bins:
        for rec in records:
            rec["near_spur"] = False
        return

    for rec in records:
        fc = rec.get("f_center_hz")
        near = False
        if fc is not None:
            try:
                fc_int = int(fc)
                idx = bisect_left(bins, fc_int)
                for pos in (idx, idx - 1):
                    if 0 <= pos < len(bins) and abs(fc_int - bins[pos]) <= tolerance_hz:
                        near = True
                        break
            except Exception:
                near = False
        rec["near_spur"] = near


def load_baseline_bins(
    f_min: int | None,
    f_max: int | None,
) -> List[Tuple[int, float]]:
    """
    Load baseline occupancy data for annotation (legacy baseline table).

    Args:
        f_min: Minimum frequency filter (Hz), or None.
        f_max: Maximum frequency filter (Hz), or None.

    Returns:
        List of (bin_hz, ema_occ) tuples.
    """
    connection = get_con_optional()
    if connection is None:
        return []

    query = "SELECT bin_hz, ema_occ FROM baseline"
    params: List[Any] = []
    clauses: List[str] = []

    if f_min is not None:
        clauses.append("bin_hz >= ?")
        params.append(int(f_min))
    if f_max is not None:
        clauses.append("bin_hz <= ?")
        params.append(int(f_max))
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += " ORDER BY bin_hz"

    try:
        rows = qa(connection, query, tuple(params))
    except Exception:
        return []

    bins: List[Tuple[int, float]] = []
    for row in rows:
        try:
            bin_hz = row.get("bin_hz") if isinstance(row, dict) else row[0]
            occ = row.get("ema_occ") if isinstance(row, dict) else row[1]
            if bin_hz is None or occ is None:
                continue
            bins.append((int(bin_hz), float(occ)))
        except Exception:
            continue

    return bins


def annotate_baseline_status(
    records: List[Dict[str, Any]],
    bins: List[Tuple[int, float]],
    threshold: float,
) -> None:
    """
    Annotate records with baseline status (new vs known) based on occupancy.

    Modifies records in place to add "baseline_status" and "is_new" keys.

    Args:
        records: List of detection dicts with f_center_hz.
        bins: List of (bin_hz, ema_occ) tuples from load_baseline_bins().
        threshold: Occupancy threshold below which a signal is considered new.
    """
    if not bins:
        for rec in records:
            rec["baseline_status"] = "unknown"
            rec["is_new"] = None
        return

    freqs = [b[0] for b in bins]

    for rec in records:
        fc = rec.get("f_center_hz")
        status = "unknown"
        is_new = None

        if fc is not None:
            try:
                fc_int = int(fc)
                idx = bisect_left(freqs, fc_int)
                closest_occ = None
                best_diff = None

                for pos in (idx, idx - 1):
                    if 0 <= pos < len(freqs):
                        diff = abs(fc_int - freqs[pos])
                        if best_diff is None or diff < best_diff:
                            closest_occ = bins[pos]
                            best_diff = diff

                if closest_occ is not None:
                    occ_val = closest_occ[1]
                    is_new = bool(occ_val < threshold)
                    status = "new" if is_new else "known"
            except Exception:
                status = "unknown"

        rec["baseline_status"] = status
        rec["is_new"] = is_new
