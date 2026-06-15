"""Additive persistence provenance tests for multi-RTL mode."""

from __future__ import annotations

import sqlite3

from sdrwatch.baseline.store import Store


def test_scan_updates_have_nullable_role_device_provenance_columns(tmp_path) -> None:
    db_path = tmp_path / "sdrwatch.db"
    store = Store(str(db_path))
    try:
        rows = store.con.execute("PRAGMA table_info(scan_updates)").fetchall()
        columns = {row[1] for row in rows}
    finally:
        store.con.close()

    assert {
        "receiver_role",
        "device_key",
        "device_serial",
        "device_index",
        "job_id",
        "role_run_id",
        "source_profile",
        "source_task",
    }.issubset(columns)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            """
            INSERT INTO scan_updates(
                baseline_id, timestamp_utc, num_hits, num_segments, num_new_signals,
                receiver_role, device_key, device_serial, device_index, job_id,
                role_run_id, source_profile, source_task
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                1,
                "2026-06-14T00:00:00Z",
                1,
                2,
                3,
                "GUARD",
                "rtl:0",
                "S1",
                0,
                "job-1",
                "rr-1",
                "fm_broadcast",
                "guard_window",
            ),
        )
        row = conn.execute("SELECT receiver_role, role_run_id, source_task FROM scan_updates").fetchone()
    finally:
        conn.close()

    assert row == ("GUARD", "rr-1", "guard_window")
