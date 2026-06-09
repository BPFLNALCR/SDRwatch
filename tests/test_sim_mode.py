from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from sdrwatch.baseline.store import Store
from sdrwatch.cli import parse_args, run
from sdrwatch.util.exit_codes import ExitCode
from sdrwatch_web import create_app
from sdrwatch_web.baseline_helpers import baseline_summary_map, tactical_snapshot_payload


def _create_baseline(db_path: Path) -> int:
    store = Store(str(db_path))
    baseline = store.create_baseline(
        name="sim-test",
        freq_start_hz=88_000_000,
        freq_stop_hz=91_600_000,
        bin_hz=0.0,
    )
    return baseline.id


def _sim_cli_args(db_path: Path, baseline_id: int) -> list[str]:
    return [
        "--driver",
        "sim",
        "--baseline-id",
        str(baseline_id),
        "--db",
        str(db_path),
        "--start",
        "88e6",
        "--stop",
        "91.6e6",
        "--step",
        "1.2e6",
        "--samp-rate",
        "2.4e6",
        "--fft",
        "2048",
        "--avg",
        "4",
        "--threshold-db",
        "6",
        "--guard-bins",
        "1",
        "--min-width-bins",
        "2",
        "--cfar",
        "off",
        "--persistence-hit-ratio",
        "0.1",
        "--persistence-min-hits",
        "1",
        "--persistence-min-windows",
        "1",
        "--repeat",
        "2",
    ]


def _table_count(db_path: Path, table: str, baseline_id: int) -> int:
    con = sqlite3.connect(str(db_path))
    try:
        row = con.execute(
            f"SELECT COUNT(*) FROM {table} WHERE baseline_id = ?",
            (baseline_id,),
        ).fetchone()
        return int(row[0]) if row else 0
    finally:
        con.close()


def _detection_summary(db_path: Path, baseline_id: int) -> list[tuple[int, int, int]]:
    con = sqlite3.connect(str(db_path))
    try:
        rows = con.execute(
            """
            SELECT f_center_hz, total_hits, total_windows
            FROM baseline_detections
            WHERE baseline_id = ?
            ORDER BY f_center_hz
            """,
            (baseline_id,),
        ).fetchall()
        return [(int(freq), int(total_hits), int(total_windows)) for freq, total_hits, total_windows in rows]
    finally:
        con.close()


def _run_simulated_scan(db_path: Path) -> tuple[int, int]:
    baseline_id = _create_baseline(db_path)
    args = parse_args(_sim_cli_args(db_path, baseline_id))
    exit_code = run(args)
    return baseline_id, exit_code


def test_parse_args_accepts_sim_driver(tmp_path: Path) -> None:
    db_path = tmp_path / "sim.db"
    baseline_id = _create_baseline(db_path)

    args = parse_args(_sim_cli_args(db_path, baseline_id))

    assert args.driver == "sim"
    assert args.baseline_id == baseline_id


def test_simulated_source_is_deterministic() -> None:
    from sdrwatch.drivers.simulate import SimulatedSource

    src_a = SimulatedSource(samp_rate=2.4e6, gain="auto")
    src_b = SimulatedSource(samp_rate=2.4e6, gain="auto")
    try:
        src_a.tune(89.1e6)
        src_b.tune(89.1e6)
        warm_a = src_a.read(2048)
        warm_b = src_b.read(2048)
        samples_a = src_a.read(8192)
        samples_b = src_b.read(8192)
    finally:
        src_a.close()
        src_b.close()

    np.testing.assert_allclose(warm_a, warm_b)
    np.testing.assert_allclose(samples_a, samples_b)
    assert samples_a.dtype == np.complex64


def test_simulated_scan_writes_standard_tables(tmp_path: Path) -> None:
    db_path = tmp_path / "sim.db"

    baseline_id, exit_code = _run_simulated_scan(db_path)

    assert exit_code == ExitCode.SUCCESS
    assert _table_count(db_path, "baseline_noise", baseline_id) > 0
    assert _table_count(db_path, "baseline_occupancy", baseline_id) > 0
    assert _table_count(db_path, "baseline_detections", baseline_id) > 0
    assert _table_count(db_path, "scan_updates", baseline_id) == 2


def test_simulated_scan_is_repeatable_across_fresh_databases(tmp_path: Path) -> None:
    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"

    baseline_a, exit_a = _run_simulated_scan(db_a)
    baseline_b, exit_b = _run_simulated_scan(db_b)

    assert exit_a == ExitCode.SUCCESS
    assert exit_b == ExitCode.SUCCESS
    assert _detection_summary(db_a, baseline_a) == _detection_summary(db_b, baseline_b)


def test_simulated_scan_dashboard_helpers_can_read_results(tmp_path: Path) -> None:
    db_path = tmp_path / "sim.db"

    baseline_id, exit_code = _run_simulated_scan(db_path)

    assert exit_code == ExitCode.SUCCESS

    app = create_app(str(db_path))
    with app.app_context():
        summary_map = baseline_summary_map()
        assert baseline_id in summary_map
        assert int(summary_map[baseline_id]["persistent_detections"]) > 0
        assert summary_map[baseline_id]["last_update_utc"] is not None

        payload = tactical_snapshot_payload(baseline_id)
        assert payload is not None
        assert int(payload["snapshot"]["persistent_signals"]) > 0
        assert payload["snapshot"]["latest_update"] is not None
        assert payload["active_signals"]


def test_dashboard_route_renders_simulated_results(tmp_path: Path) -> None:
    db_path = tmp_path / "sim.db"

    baseline_id, exit_code = _run_simulated_scan(db_path)

    assert exit_code == ExitCode.SUCCESS

    app = create_app(str(db_path))
    client = app.test_client()
    response = client.get(f"/?baseline_id={baseline_id}")

    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "Baseline overview" in body
    assert "sim-test" in body
    assert "Signal cards" in body
    assert "SIG-" in body


def test_real_driver_does_not_silently_fall_back_to_sim(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sdrwatch.cli.HAVE_RTLSDR", False)
    monkeypatch.setattr("sdrwatch.cli.RTLSDR_IMPORT_ERROR", "simulated missing backend")

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--driver",
                "rtlsdr_native",
                "--baseline-id",
                "1",
                "--start",
                "88e6",
                "--stop",
                "89e6",
            ]
        )