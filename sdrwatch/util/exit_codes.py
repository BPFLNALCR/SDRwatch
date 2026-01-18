"""Documented exit codes for SDRwatch CLI and controller.

Exit codes follow UNIX conventions:
- 0: Success
- 1: General/unspecified error
- 2: Invalid command-line arguments or usage
- 3-5: Application-specific errors

These codes enable shell scripts and monitoring tools to distinguish
failure modes without parsing stderr output.

Usage:
    from sdrwatch.util.exit_codes import ExitCode
    sys.exit(ExitCode.BASELINE_NOT_FOUND)
"""

from __future__ import annotations


class ExitCode:
    """Exit code constants for SDRwatch processes.

    Attributes:
        SUCCESS: Normal termination, no errors.
        GENERAL_ERROR: Unspecified runtime error.
        INVALID_ARGS: Command-line argument validation failed.
        BASELINE_NOT_FOUND: Requested baseline ID does not exist.
        DEVICE_UNAVAILABLE: SDR device could not be opened or is busy.
        DB_ERROR: Database connection or write failure.
    """

    SUCCESS: int = 0
    GENERAL_ERROR: int = 1
    INVALID_ARGS: int = 2
    BASELINE_NOT_FOUND: int = 3
    DEVICE_UNAVAILABLE: int = 4
    DB_ERROR: int = 5

    @classmethod
    def message(cls, code: int) -> str:
        """Return a human-readable message for an exit code."""
        messages = {
            cls.SUCCESS: "Success",
            cls.GENERAL_ERROR: "General error",
            cls.INVALID_ARGS: "Invalid arguments",
            cls.BASELINE_NOT_FOUND: "Baseline not found",
            cls.DEVICE_UNAVAILABLE: "SDR device unavailable",
            cls.DB_ERROR: "Database error",
        }
        return messages.get(code, f"Unknown exit code {code}")
