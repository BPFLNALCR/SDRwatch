"""Bandplan CSV loading and lookup helpers."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class Band:
    low_hz: int
    high_hz: int
    service: str
    region: str
    notes: str


class Bandplan:
    def __init__(self, csv_path: Optional[str] = None):
        self.bands: List[Band] = []
        if csv_path and os.path.exists(csv_path):
            self._load_csv(csv_path)
        else:
            self.bands = [
                Band(433_050_000, 434_790_000, "ISM/SRD", "ITU-R1 (EU)", "Short-range devices"),
                Band(902_000_000, 928_000_000, "ISM", "US (FCC)", "902-928 MHz ISM"),
                Band(2_400_000_000, 2_483_500_000, "ISM", "Global", "2.4 GHz ISM"),
                Band(1_420_000_000, 1_427_000_000, "Radio Astronomy", "Global", "Hydrogen line"),
                Band(88_000_000, 108_000_000, "FM Broadcast", "Global", "88-108 MHz Radio"),
            ]

    def _load_csv(self, path: str) -> None:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    low = row.get("low_hz") or row.get("f_low_hz")
                    high = row.get("high_hz") or row.get("f_high_hz")
                    if low is None or high is None:
                        continue
                    self.bands.append(
                        Band(
                            int(float(low)),
                            int(float(high)),
                            (row.get("service") or "").strip(),
                            (row.get("region") or "").strip(),
                            (row.get("notes") or "").strip(),
                        )
                    )
                except Exception:
                    continue

    def lookup(self, f_hz: int) -> Tuple[str, str, str]:
        for band in self.bands:
            if band.low_hz <= f_hz <= band.high_hz:
                return band.service, band.region, band.notes
        return "", "", ""
