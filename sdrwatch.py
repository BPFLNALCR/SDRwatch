#!/usr/bin/env python3
"""Legacy SDRwatch CLI shim; prefer `python -m sdrwatch.cli`."""

from __future__ import annotations

import sys
import warnings

from sdrwatch.cli import parse_args, run


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    warnings.warn(
        "sdrwatch.py is deprecated; invoke the scanner via `python -m sdrwatch.cli` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    run(parse_args(argv))


if __name__ == "__main__":
    main()
