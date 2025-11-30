"""Structured scan logging helpers."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, List, Optional, Set

from sdrwatch.util.time import utc_now_str


class ScanLogger:
    def __init__(self, log_path: Path, mirror_paths: Optional[List[Path]] = None):
        self.log_path = log_path
        self.mirror_paths: List[Path] = []
        self._ensure_parent(self.log_path)
        seen: Set[str] = {str(self.log_path)}
        for mirror in mirror_paths or []:
            try:
                resolved = mirror
                if not resolved.is_absolute():
                    resolved = (Path.cwd() / resolved).absolute()
                if str(resolved) in seen:
                    continue
                self._ensure_parent(resolved)
                self.mirror_paths.append(resolved)
                seen.add(str(resolved))
            except Exception:
                continue
        self.run_id = f"run-{int(time.time() * 1000)}-pid{os.getpid()}"
        self.current_sweep: Optional[int] = None

    @staticmethod
    def _ensure_parent(path: Path) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    @classmethod
    def from_db_path(cls, db_path: str, extra_targets: Optional[List[str]] = None) -> "ScanLogger":
        if not db_path or db_path == ":memory:":
            base_dir = Path.cwd()
        else:
            expanded = Path(db_path).expanduser()
            if not expanded.is_absolute():
                expanded = (Path.cwd() / expanded).absolute()
            base_dir = expanded.parent if expanded.parent != Path("") else Path.cwd()
        log_path = base_dir / "sdrwatch-scan.log"
        extra_paths: List[Path] = []
        for target in extra_targets or []:
            if not target:
                continue
            extra_paths.append(Path(target).expanduser())
        return cls(log_path, extra_paths)

    def start_sweep(self, sweep_id: int, **metadata: Any) -> None:
        self.current_sweep = sweep_id
        self.log("sweep_start", **metadata)

    def log(self, event: str, **fields: Any) -> None:
        record = {
            "ts": utc_now_str(),
            "run_id": self.run_id,
            "sweep_id": self.current_sweep,
            "event": event,
            **fields,
        }
        targets = [self.log_path] + self.mirror_paths
        for target in targets:
            try:
                with target.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record) + "\n")
            except Exception:
                continue
