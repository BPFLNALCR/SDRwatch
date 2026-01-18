"""
Database helpers for SDRwatch Web.

Provides read-only SQLite connection management, query helpers (q1/qa),
schema probing (table_exists, table_columns), and state checking (db_state).
"""
from __future__ import annotations

import os
import sqlite3
from typing import Any, Dict, Optional, Set, Tuple

from flask import Flask, current_app, g


def open_db_ro(path: str) -> sqlite3.Connection:
    """
    Open a read-only SQLite connection with dict row factory.

    Args:
        path: Path to the SQLite database file.

    Returns:
        sqlite3.Connection configured for read-only access with dict rows.
    """
    abspath = os.path.abspath(path)
    con = sqlite3.connect(f"file:{abspath}?mode=ro", uri=True, check_same_thread=False)
    con.execute("PRAGMA busy_timeout=2000;")
    con.row_factory = lambda cur, row: {d[0]: row[i] for i, d in enumerate(cur.description)}
    return con


def q1(con: sqlite3.Connection, sql: str, params: Any = ()) -> Optional[Dict[str, Any]]:
    """Execute SQL and return the first row as a dict, or None."""
    cur = con.execute(sql, params)
    return cur.fetchone()


def qa(con: sqlite3.Connection, sql: str, params: Any = ()) -> list:
    """Execute SQL and return all rows as a list of dicts."""
    cur = con.execute(sql, params)
    return cur.fetchall()


# ---------------------------------------------------------------------------
# App-level connection management (stored on Flask app instance)
# ---------------------------------------------------------------------------


def init_db(app: Flask, db_path: str) -> None:
    """
    Initialize database state on the Flask app instance.

    Args:
        app: Flask application instance.
        db_path: Path to the SQLite database file.
    """
    app.config['SDRWATCH_DB_PATH'] = db_path
    app.config['SDRWATCH_DB_ERROR'] = None
    app.config['SDRWATCH_DB_CON'] = None
    app.config['SDRWATCH_DB_HAS_CONFIDENCE'] = None
    app.config['SDRWATCH_DB_COLUMNS_CACHE'] = {}
    app.config['SDRWATCH_DB_EXISTS_CACHE'] = {}

    # Attempt initial connection (tolerates failure if DB is missing)
    _ensure_con(app)


def _ensure_con(app: Flask) -> Optional[sqlite3.Connection]:
    """Lazy-open the read-only connection, caching on success."""
    if app.config.get('SDRWATCH_DB_CON') is not None:
        return app.config['SDRWATCH_DB_CON']
    try:
        app.config['SDRWATCH_DB_CON'] = open_db_ro(app.config['SDRWATCH_DB_PATH'])
        app.config['SDRWATCH_DB_ERROR'] = None
        app.config['SDRWATCH_DB_HAS_CONFIDENCE'] = None
    except Exception as exc:
        app.config['SDRWATCH_DB_CON'] = None
        app.config['SDRWATCH_DB_ERROR'] = str(exc)
        app.config['SDRWATCH_DB_HAS_CONFIDENCE'] = None
    return app.config['SDRWATCH_DB_CON']


def reset_ro_connection(app: Flask) -> None:
    """Close and reset the cached read-only connection."""
    if app.config.get('SDRWATCH_DB_CON') is not None:
        try:
            app.config['SDRWATCH_DB_CON'].close()
        except Exception:
            pass
    app.config['SDRWATCH_DB_CON'] = None
    app.config['SDRWATCH_DB_COLUMNS_CACHE'] = {}
    app.config['SDRWATCH_DB_EXISTS_CACHE'] = {}


def get_con() -> sqlite3.Connection:
    """
    Get the current read-only database connection.

    Raises:
        RuntimeError: If the database connection is unavailable.

    Returns:
        Active sqlite3.Connection instance.
    """
    connection = _ensure_con(current_app)
    if connection is None:
        raise RuntimeError("database connection unavailable")
    return connection


def get_con_optional() -> Optional[sqlite3.Connection]:
    """Get the current connection, or None if unavailable."""
    return _ensure_con(current_app)


def table_columns(table_name: str) -> Set[str]:
    """
    Get the set of column names for a table (cached).

    Args:
        table_name: Name of the table.

    Returns:
        Set of lowercase column names.
    """
    cache = current_app.config['SDRWATCH_DB_COLUMNS_CACHE']
    key = table_name.lower()
    if key in cache:
        return cache[key]

    connection = _ensure_con(current_app)
    columns: Set[str] = set()
    if connection is None:
        cache[key] = columns
        return columns

    try:
        rows = qa(connection, f"PRAGMA table_info({table_name})")
    except sqlite3.OperationalError:
        cache[key] = columns
        return columns

    for row in rows:
        name = row.get("name") if isinstance(row, dict) else row[1]
        if name:
            columns.add(str(name).lower())
    cache[key] = columns
    return columns


def table_exists(table_name: str) -> bool:
    """
    Check if a table exists in the database (cached).

    Args:
        table_name: Name of the table.

    Returns:
        True if the table exists.
    """
    cache = current_app.config['SDRWATCH_DB_EXISTS_CACHE']
    key = table_name.lower()
    if key in cache:
        return cache[key]

    connection = _ensure_con(app)
    if connection is None:
        cache[key] = False
        return False

    try:
        row = q1(
            connection,
            "SELECT name FROM sqlite_master WHERE type='table' AND lower(name)=?",
            (key,),
        )
        exists = bool(row and row.get('name'))
    except sqlite3.OperationalError:
        exists = False

    cache[key] = exists
    return exists


def db_state() -> Tuple[str, str]:
    """
    Check database readiness state.

    Returns:
        Tuple of (state, message) where state is one of:
        - "ready": Database is connected and has required tables
        - "waiting": Database exists but is not yet initialized
        - "unavailable": Database could not be opened
    """
    connection = _ensure_con(current_app)
    if connection is None:
        return (
            "unavailable",
            current_app.config.get('SDRWATCH_DB_ERROR') or "Database file could not be opened in read-only mode.",
        )
    try:
        cur = connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        names: Set[str] = set()
        for row in cur.fetchall():
            if isinstance(row, dict):
                value = row.get('name', '')
            else:
                value = row[0]
            if value:
                names.add(str(value).lower())

        modern_ready = "baselines" in names
        legacy_required = {"scans", "detections", "baseline"}
        if modern_ready or legacy_required.issubset(names):
            return ("ready", "")
        return ("waiting", "")
    except sqlite3.OperationalError as exc:
        current_app.config['SDRWATCH_DB_CON'] = None
        current_app.config['SDRWATCH_DB_ERROR'] = str(exc)
        return (
            "waiting",
            f"Database not initialized yet ({exc}). Start a scan to populate it.",
        )


def db_waiting_context(state: str, message: str) -> Dict[str, Any]:
    """
    Build template context for database waiting/unavailable states.

    Args:
        state: Database state string.
        message: Associated error/status message.

    Returns:
        Dict suitable for passing to templates.
    """
    return {
        "db_status": state,
        "db_status_message": message,
        "db_path": current_app.config.get('SDRWATCH_DB_PATH'),
    }


def detections_have_confidence() -> bool:
    """Check if the detections table has a confidence column (cached)."""
    cached = current_app.config.get('SDRWATCH_DB_HAS_CONFIDENCE')
    if cached is not None:
        return bool(cached)

    connection = _ensure_con(current_app)
    if connection is None:
        app.config['SDRWATCH_DB_HAS_CONFIDENCE'] = False
        return False

    try:
        cur = connection.execute("PRAGMA table_info(detections)")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        current_app.config['SDRWATCH_DB_HAS_CONFIDENCE'] = False
        return False

    has_col = False
    for row in rows:
        name = row.get("name") if isinstance(row, dict) else row[1]
        if name == "confidence":
            has_col = True
            break

    current_app.config['SDRWATCH_DB_HAS_CONFIDENCE'] = has_col
    return has_col
