"""Structured logging configuration for SDRwatch.

Provides a centralized logging setup with:
- Console handler (stderr) with configurable level
- Optional file handler (JSON lines for machine parsing)
- Contextual formatters (human-readable or JSON)
- Environment-based configuration

Usage:
    from sdrwatch.util.logging import get_logger, configure_logging

    configure_logging(level="DEBUG", json_file="/var/log/sdrwatch.log")
    logger = get_logger(__name__)
    logger.info("Starting scan", extra={"baseline_id": 3})
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, Optional


# Module-level state
_configured = False
_root_logger_name = "sdrwatch"


class JSONFormatter(logging.Formatter):
    """Emit log records as single-line JSON for machine parsing."""

    def format(self, record: logging.LogRecord) -> str:
        output: Dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Include extra fields passed via extra={}
        for key in ("baseline_id", "device", "sweep_id", "error_type", "duration_ms"):
            if hasattr(record, key):
                output[key] = getattr(record, key)
        # Include exception info if present
        if record.exc_info:
            output["traceback"] = "".join(traceback.format_exception(*record.exc_info))
        return json.dumps(output, default=str)


class ConsoleFormatter(logging.Formatter):
    """Human-readable console format with optional color."""

    LEVEL_COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(self, use_color: bool = True):
        super().__init__()
        self.use_color = use_color and sys.stderr.isatty()

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        level = record.levelname
        if self.use_color:
            color = self.LEVEL_COLORS.get(level, "")
            level_str = f"{color}{level:8}{self.RESET}"
        else:
            level_str = f"{level:8}"
        name = record.name.replace("sdrwatch.", "")
        msg = record.getMessage()
        base = f"[{ts}] {level_str} [{name}] {msg}"
        if record.exc_info:
            base += "\n" + "".join(traceback.format_exception(*record.exc_info))
        return base


def configure_logging(
    *,
    level: Optional[str] = None,
    json_file: Optional[str] = None,
    use_color: bool = True,
) -> None:
    """Configure the sdrwatch logging subsystem.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR). Defaults to INFO,
               or DEBUG if SDRWATCH_DEBUG=1 is set.
        json_file: Optional path to write JSON-formatted logs.
        use_color: Whether to colorize console output (auto-disabled if not a TTY).

    This function is idempotent; calling it multiple times reconfigures handlers.
    """
    global _configured

    # Determine log level
    if level is None:
        if os.environ.get("SDRWATCH_DEBUG", "").strip() in ("1", "true", "yes"):
            level = "DEBUG"
        else:
            level = os.environ.get("SDRWATCH_LOG_LEVEL", "INFO").upper()

    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Get or create the root sdrwatch logger
    logger = logging.getLogger(_root_logger_name)
    logger.setLevel(numeric_level)

    # Remove existing handlers to allow reconfiguration
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Console handler (stderr)
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(ConsoleFormatter(use_color=use_color))
    logger.addHandler(console_handler)

    # Optional JSON file handler
    if json_file:
        try:
            file_handler = logging.FileHandler(json_file, mode="a", encoding="utf-8")
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(JSONFormatter())
            logger.addHandler(file_handler)
        except OSError as exc:
            logger.warning("Failed to open JSON log file %s: %s", json_file, exc)

    # Prevent propagation to root logger
    logger.propagate = False

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger under the sdrwatch namespace.

    Args:
        name: Usually __name__ of the calling module.

    Returns:
        A logging.Logger instance.

    If configure_logging() has not been called, a default configuration
    is applied automatically.
    """
    global _configured
    if not _configured:
        configure_logging()

    # Ensure the name is under the sdrwatch namespace
    if not name.startswith(_root_logger_name):
        if name == "__main__":
            name = f"{_root_logger_name}.main"
        else:
            name = f"{_root_logger_name}.{name}"

    return logging.getLogger(name)


def log_exception(
    logger: logging.Logger,
    message: str,
    *,
    error_type: Optional[str] = None,
    **extra: Any,
) -> None:
    """Log an exception with structured context.

    Call this inside an except block to capture the current exception.

    Args:
        logger: The logger to use.
        message: Human-readable error description.
        error_type: Category of error (e.g., "driver_init", "db_write").
        **extra: Additional context fields.
    """
    extra_dict = dict(extra)
    if error_type:
        extra_dict["error_type"] = error_type
    logger.exception(message, extra=extra_dict)
