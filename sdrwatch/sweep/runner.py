"""High-level scanner runner that orchestrates sweeps across devices and baselines."""

from __future__ import annotations

import os
import sys
import time
from typing import Dict, Optional

from sdrwatch.baseline.context import resolve_baseline_context
from sdrwatch.baseline.store import Store
from sdrwatch.drivers.rtlsdr import RTLSDRSource
from sdrwatch.drivers.soapy import SDRSource
from sdrwatch.io.bandplan import Bandplan
from sdrwatch.sweep.sweeper import Sweeper
from sdrwatch.util.duration import parse_duration_to_seconds
from sdrwatch.util.scan_logger import ScanLogger


class ScannerRunner:
    """Bind CLI args to device, baseline, and sweeper execution."""

    def __init__(self, args):
        self.args = args
        self.store = Store(args.db)
        self.bandplan = Bandplan(args.bandplan)
        self.logger = ScanLogger.from_db_path(args.db, extra_targets=self._extra_targets())
        self.src = None
        self.sweeper: Optional[Sweeper] = None

    def _extra_targets(self):
        jsonl_path = getattr(self.args, "jsonl", None)
        return [jsonl_path] if jsonl_path else None

    def _resolve_baseline(self):
        planned_start = min(self.args.start, self.args.stop)
        planned_stop = max(self.args.start, self.args.stop)
        bin_hint = float(self.args.samp_rate) / float(self.args.fft) if self.args.fft else float(self.args.samp_rate)
        baseline_ctx = resolve_baseline_context(
            self.store,
            getattr(self.args, "baseline_id", None),
            span_hint=(planned_start, planned_stop),
            bin_hz_hint=bin_hint,
        )
        self.logger.log(
            "baseline_context",
            baseline_id=baseline_ctx.id,
            baseline_span_hz=[baseline_ctx.freq_start_hz, baseline_ctx.freq_stop_hz],
            bin_hz=baseline_ctx.bin_hz,
            db_path=os.path.abspath(self.args.db),
        )
        self.args.baseline_id = baseline_ctx.id
        if baseline_ctx.bin_hz <= 0.0 and bin_hint > 0.0:
            self.store.set_baseline_bin(baseline_ctx.id, bin_hint)
            baseline_ctx.bin_hz = bin_hint
            print(
                f"[baseline] repaired bin_hz to {bin_hint:.1f} Hz based on current sweep",
                flush=True,
            )
        if baseline_ctx.freq_start_hz <= 0:
            baseline_ctx.freq_start_hz = planned_start
        if planned_stop > baseline_ctx.freq_stop_hz:
            baseline_ctx.freq_stop_hz = planned_stop
        if baseline_ctx.freq_start_hz > 0 and baseline_ctx.freq_stop_hz > 0:
            span_text = f"{baseline_ctx.freq_start_hz/1e6:.3f}-{baseline_ctx.freq_stop_hz/1e6:.3f} MHz"
        else:
            span_text = "auto (pending scans)"
        print(
            f"[baseline] using id={baseline_ctx.id} name='{baseline_ctx.name}' span={span_text} bin={baseline_ctx.bin_hz:.1f} Hz",
            flush=True,
        )
        return baseline_ctx

    def _select_source(self):
        args = self.args
        soapy_args_dict: Optional[Dict[str, str]] = None
        if getattr(args, "soapy_args", None):
            soapy_args_dict = {}
            for kv in str(args.soapy_args).split(","):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    soapy_args_dict[k.strip()] = v.strip()

        if args.driver == "rtlsdr_native":
            src = RTLSDRSource(samp_rate=args.samp_rate, gain=args.gain)
            setattr(src, "device", "RTL-SDR (native)")
            return src

        try:
            src = SDRSource(driver=args.driver, samp_rate=args.samp_rate, gain=args.gain, soapy_args=soapy_args_dict)
            return src
        except Exception as exc:
            msg = str(exc)
            if args.driver == "rtlsdr" and (
                "no match" in msg.lower() or "device::make" in msg or "rtlsdr" in msg.lower()
            ):
                idx_hint = None
                serial_hint = None
                if soapy_args_dict:
                    if "serial" in soapy_args_dict:
                        serial_hint = soapy_args_dict.get("serial")
                    if "index" in soapy_args_dict:
                        try:
                            val = soapy_args_dict.get("index")
                            if val is not None:
                                idx_hint = int(val)
                        except Exception:
                            idx_hint = None
                last_err = None
                for _ in range(3):
                    try:
                        src = RTLSDRSource(
                            samp_rate=args.samp_rate,
                            gain=args.gain,
                            device_index=idx_hint,
                            serial_number=serial_hint,
                        )
                        setattr(src, "device", "RTL-SDR (native fallback)")
                        args.driver = "rtlsdr_native"
                        return src
                    except Exception as retry_exc:  # pragma: no cover - hardware specific
                        last_err = retry_exc
                        time.sleep(0.2)
                raise last_err if last_err else exc
            raise

    def _termination_policy(self):
        duration_s = parse_duration_to_seconds(self.args.duration)
        start_time = time.time()
        if self.args.loop:
            sweeps_remaining: Optional[int] = None
        elif self.args.repeat is not None:
            sweeps_remaining = int(self.args.repeat)
        elif duration_s is not None:
            sweeps_remaining = None
        else:
            sweeps_remaining = 1
        return duration_s, start_time, sweeps_remaining

    def run(self):
        if self.args.tmpdir:
            os.environ["TMPDIR"] = self.args.tmpdir

        baseline_ctx = self._resolve_baseline()
        src = self._select_source()
        self.src = src
        self.sweeper = Sweeper(self.args, self.store, self.bandplan, baseline_ctx, self.logger)

        duration_s, start_time, sweeps_remaining = self._termination_policy()
        sweep_seq = 1
        try:
            while True:
                if duration_s is not None and (time.time() - start_time) >= duration_s:
                    break
                assert self.sweeper is not None
                self.sweeper.run(src, sweep_seq)
                sweep_seq += 1
                if duration_s is not None and (time.time() - start_time) >= duration_s:
                    break
                if sweeps_remaining is not None:
                    sweeps_remaining -= 1
                    if sweeps_remaining <= 0:
                        break
                if self.args.sleep_between_sweeps > 0:
                    time.sleep(self.args.sleep_between_sweeps)
        except KeyboardInterrupt:
            pass
        finally:
            try:
                if getattr(src, "close", None):
                    src.close()
            except Exception:
                pass


def run_scan(args) -> None:
    runner = ScannerRunner(args)
    runner.run()
