#!/usr/bin/env python3
"""
SDRwatch Control Daemon / CLI (patched)

Changes in this patch
---------------------
1) **Automatic lock cleanup (reaper):**
   - When a scan process exits, a background thread now marks the job finished,
     stores the exit code, releases the device lock, and persists state.

2) **Stale-lock handling on start:**
   - If a lock file exists but its owning job isn't actually running anymore,
     the manager will auto-clear the stale lock and proceed.

3) **Startup reconciliation improved:**
   - On initialization, any jobs previously marked as running but with dead PIDs
     are immediately marked finished and their device locks are released.

These changes fix the issue where a device stayed "busy (lock present)" after a
completed scan.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import sqlite3
import subprocess
import sys
import time
import uuid
import threading
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

# ---------- Configuration ----------
BASE_DIR = Path(os.environ.get("SDRWATCH_CONTROL_BASE", "/tmp/sdrwatch-control"))
STATE_PATH = BASE_DIR / "state.json"
LOGS_DIR = BASE_DIR / "logs"
LOCKS_DIR = BASE_DIR / "locks"

# If your project locates scripts elsewhere, tweak these defaults:
PYTHON_EXE = sys.executable or "python3"


_SCRIPT_CACHE: Optional[Tuple[Path, Path]] = None  # (project_dir, script_path)
_PROFILE_CACHE: Optional[Dict[str, Any]] = None
_PROFILE_CACHE_TS: float = 0.0
_PROFILE_CACHE_TTL = 30.0


def resolve_scanner_paths(force_refresh: bool = False) -> Tuple[Path, Path]:
    """Return (project_dir, sdrwatch_script) with automatic fallback discovery."""
    global _SCRIPT_CACHE
    if not force_refresh and _SCRIPT_CACHE:
        project_dir, script_path = _SCRIPT_CACHE
        if script_path.exists():
            return project_dir, script_path

    candidates: List[Path] = []

    def _add_candidate(path_like: Optional[Path] | str | None) -> None:
        if not path_like:
            return
        p = Path(path_like).expanduser()
        candidates.append(p)

    # Ordered preference list
    _add_candidate(os.environ.get("SDRWATCH_SCRIPT"))
    project_env = os.environ.get("SDRWATCH_PROJECT_DIR")
    if project_env:
        _add_candidate(Path(project_env).expanduser() / "sdrwatch.py")
    controller_dir = Path(__file__).resolve().parent
    _add_candidate(controller_dir / "sdrwatch.py")
    _add_candidate(controller_dir.parent / "sdrwatch.py")
    _add_candidate(Path.cwd() / "sdrwatch.py")
    # Common deployment roots
    _add_candidate(Path("/opt/sdrwatch") / "sdrwatch.py")
    _add_candidate(Path("/opt/sdrwatch/current") / "sdrwatch.py")

    seen: set[str] = set()
    for cand in candidates:
        key = str(cand)
        if key in seen:
            continue
        seen.add(key)
        try:
            if cand.is_file():
                script_path = cand.resolve()
                project_dir = script_path.parent
                _SCRIPT_CACHE = (project_dir, script_path)
                return project_dir, script_path
        except OSError:
            continue

    raise FileNotFoundError(
        "Unable to locate sdrwatch.py. Set SDRWATCH_SCRIPT or keep it alongside sdrwatch-control.py."
    )


def _fetch_profiles_payload(force_refresh: bool = False) -> Dict[str, Any]:
    global _PROFILE_CACHE, _PROFILE_CACHE_TS
    now = time.time()
    if not force_refresh and _PROFILE_CACHE and (now - _PROFILE_CACHE_TS) < _PROFILE_CACHE_TTL:
        return _PROFILE_CACHE

    project_dir, script_path = resolve_scanner_paths()
    cmd = [PYTHON_EXE, str(script_path), "--list-profiles"]
    popen_kwargs: Dict[str, Any] = {
        "capture_output": True,
        "text": True,
    }
    if project_dir.exists():
        popen_kwargs["cwd"] = str(project_dir)

    result = subprocess.run(cmd, **popen_kwargs)
    if result.returncode != 0:
        stderr = (result.stderr or "").strip()
        stdout = (result.stdout or "").strip()
        detail = stderr or stdout or "sdrwatch.py returned non-zero"
        raise RuntimeError(detail)
    try:
        payload = json.loads(result.stdout)
    except Exception as exc:
        raise RuntimeError(f"invalid JSON from sdrwatch.py: {exc}")

    _PROFILE_CACHE = payload
    _PROFILE_CACHE_TS = now
    return payload


def _default_db_path() -> Path:
    explicit = os.environ.get("SDRWATCH_DB")
    if explicit:
        return Path(explicit).expanduser()
    try:
        project_dir, _ = resolve_scanner_paths()
        return project_dir / "sdrwatch.db"
    except FileNotFoundError:
        return Path.cwd() / "sdrwatch.db"

# ---------- Utilities ----------

def ensure_dirs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    LOCKS_DIR.mkdir(parents=True, exist_ok=True)
    BASE_DIR.mkdir(parents=True, exist_ok=True)


def short_uuid() -> str:
    return uuid.uuid4().hex[:12]


def now_ts() -> float:
    return time.time()


def read_state() -> Dict[str, Any]:
    if STATE_PATH.exists():
        try:
            with STATE_PATH.open("r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            # Corrupted state; archive and start fresh
            bak = STATE_PATH.with_suffix(".corrupt.json")
            try:
                STATE_PATH.replace(bak)
            except Exception:
                pass
            return {"jobs": {}}
    return {"jobs": {}}


def write_state(state: Dict[str, Any]) -> None:
    tmp = STATE_PATH.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(STATE_PATH)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


# ---------- Device discovery ----------
@dataclass
class Device:
    key: str                 # e.g., "rtl:0" or "hackrf:serial"
    kind: str                # e.g., "rtlsdr", "hackrf", "unknown"
    label: str               # human-friendly
    extra: Dict[str, Any] = field(default_factory=dict)


def discover_rtlsdr() -> List[Device]:
    devices: List[Device] = []
    # Try Soapy first for robust enumeration (if available)
    try:
        import SoapySDR  # type: ignore
        devs = SoapySDR.Device.enumerate(dict(driver="rtlsdr"))
        for i, d in enumerate(devs):
            serial = d.get("serial", None)
            label = d.get("label", f"RTL-SDR #{i}")
            devices.append(Device(key=f"rtl:{i}", kind="rtlsdr", label=label + (f" (SN {serial})" if serial else ""), extra={"index": i, "serial": serial, "soapy_args": d}))
        if devices:
            return devices
    except Exception:
        pass
    # Fallback: pyrtlsdr quick-probe by index
    try:
        from rtlsdr import RtlSdr  # type: ignore
    except Exception:
        return devices

    for i in range(8):
        try:
            sdr = RtlSdr(i)
            serial = getattr(sdr, "serial_number", None)
            devices.append(Device(key=f"rtl:{i}", kind="rtlsdr",
                                  label=f"RTL-SDR #{i}" + (f" (SN {serial})" if serial else ""),
                                  extra={"index": i, "serial": serial}))
            sdr.close()
        except Exception:
            break
    return devices


def discover_hackrf() -> List[Device]:
    devices: List[Device] = []
    # Try Soapy first
    try:
        import SoapySDR  # type: ignore
        devs = SoapySDR.Device.enumerate(dict(driver="hackrf"))
        for i, d in enumerate(devs):
            serial = d.get("serial", None)
            label = d.get("label", f"HackRF One #{i}")
            devices.append(Device(key=f"hackrf:{i}", kind="hackrf", label=label + (f" (SN {serial})" if serial else ""), extra={"index": i, "serial": serial, "soapy_args": d}))
        if devices:
            return devices
    except Exception:
        pass
    # Fallback: hackrf_info parsing
    try:
        cp = subprocess.run(["hackrf_info"], capture_output=True, text=True, timeout=2)
        out = cp.stdout
        blocks = out.split("Found HackRF")
        idx = 0
        for b in blocks:
            if "Board ID Number" in b or "Serial number" in b:
                serial = None
                for line in b.splitlines():
                    line = line.strip()
                    if line.lower().startswith("serial number"):
                        serial = line.split(":", 1)[-1].strip()
                        break
                key = f"hackrf:{idx}"
                label = f"HackRF One #{idx}" + (f" (SN {serial})" if serial else "")
                devices.append(Device(key=key, kind="hackrf", label=label, extra={"serial": serial, "index": idx}))
                idx += 1
    except Exception:
        pass
    return devices


def discover_devices() -> List[Device]:
    devs = []
    devs.extend(discover_rtlsdr())
    devs.extend(discover_hackrf())
    # TODO: add KrakenSDR/SoapySDR/etc. as needed
    # Deduplicate by key
    uniq: Dict[str, Device] = {d.key: d for d in devs}
    return list(uniq.values())


# ---------- Job model ----------
@dataclass
class Job:
    id: str
    created_ts: float
    label: str
    device_key: str
    baseline_id: Optional[int]
    status: str              # "running", "stopped", "finished", "error"
    pid: Optional[int]
    cmd: List[str]
    log_path: str
    params: Dict[str, Any]
    exit_code: Optional[int] = None
    finished_ts: Optional[float] = None


class JobManager:
    def __init__(self) -> None:
        ensure_dirs()
        self.state = read_state()
        self.default_db_path = _default_db_path()
        self.jobs: Dict[str, Job] = {}
        for jid, j in self.state.get("jobs", {}).items():
            if "baseline_id" not in j:
                j["baseline_id"] = None
            job = Job(**j)
            # Reconcile process liveness and cleanup locks at startup
            if job.pid and job.status == "running" and not pid_alive(job.pid):
                job.status = "finished"
                job.finished_ts = now_ts()
                job.exit_code = job.exit_code if job.exit_code is not None else -1
                self._release_device(job.device_key)
            self.jobs[jid] = job
        self._persist()

    # ---- persistence ----
    def _persist(self) -> None:
        data = {"jobs": {jid: asdict(j) for jid, j in self.jobs.items()}}
        write_state(data)

    # ---- device locking ----
    def _lock_path(self, device_key: str) -> Path:
        safe = device_key.replace(":", "_")
        return LOCKS_DIR / f"{safe}.lock"

    def _is_job_running(self, job: Job) -> bool:
        return bool(job.pid and pid_alive(job.pid) and job.status == "running")

    def _acquire_device(self, device_key: str, owner: str) -> None:
        lp = self._lock_path(device_key)
        if lp.exists():
            try:
                existing_owner = lp.read_text(encoding="utf-8").strip()
            except Exception:
                existing_owner = ""
            # If the lock is owned by a known job that isn't actually running, clear it.
            if existing_owner and existing_owner in self.jobs:
                job = self.jobs[existing_owner]
                if not self._is_job_running(job):
                    try:
                        lp.unlink()
                    except Exception:
                        pass
            # If still present and clearly stale (no job knows about it), clear
            if lp.exists():
                # Optional: if file is older than N minutes treat as stale
                try:
                    mtime = lp.stat().st_mtime
                    if (now_ts() - mtime) > 3600:  # 1 hour
                        lp.unlink()
                except Exception:
                    pass
        if lp.exists():
            raise RuntimeError(f"Device {device_key} is busy (lock present)")
        with lp.open("w", encoding="utf-8") as f:
            f.write(owner)

    def _release_device(self, device_key: str) -> None:
        lp = self._lock_path(device_key)
        if lp.exists():
            try:
                lp.unlink()
            except Exception:
                pass

    # ---- job lifecycle ----
    def list_jobs(self) -> List[Job]:
        return sorted(self.jobs.values(), key=lambda j: j.created_ts, reverse=True)

    def get_job(self, job_id: str) -> Job:
        if job_id not in self.jobs:
            raise KeyError(f"Job {job_id} not found")
        # update liveness lazily
        job = self.jobs[job_id]
        if job.pid and job.status == "running" and not pid_alive(job.pid):
            job.status = "finished"
            job.finished_ts = now_ts()
            self._release_device(job.device_key)
            self._persist()
        return job

    def _spawn_reaper(self, job_id: str, proc: subprocess.Popen, device_key: str) -> None:
        def _watch():
            try:
                rc = proc.wait()
            except Exception:
                rc = -1
            # Update job status and free device
            job = self.jobs.get(job_id)
            if job:
                job.exit_code = rc
                job.pid = None
                job.finished_ts = now_ts()
                job.status = "finished" if rc == 0 else "error"
            self._release_device(device_key)
            self._persist()
        t = threading.Thread(target=_watch, name=f"reaper-{job_id}", daemon=True)
        t.start()

    def start_job(self, *, device_key: str, label: str, baseline_id: int, sdrwatch_args: Dict[str, Any]) -> Job:
        try:
            baseline_id_int = int(baseline_id)
        except (TypeError, ValueError):
            raise ValueError("baseline_id must be an integer")
        project_dir, script_path = resolve_scanner_paths()
        # Refuse to start if the device is already locked (but clear stale locks first)
        self._acquire_device(device_key, owner="pending")
        try:
            # If we have discovery metadata for this device, attach it for downstream
            discover_map = {d.key: d for d in discover_devices()}
            if device_key in discover_map:
                meta = discover_map[device_key].extra
                sdrwatch_args = dict(sdrwatch_args)  # copy
                sdrwatch_args["__discover_meta"] = meta

            jid = short_uuid()
            log_path = str(LOGS_DIR / f"{jid}.log")
            cmd = self._build_cmd(
                script_path=script_path,
                device_key=device_key,
                baseline_id=baseline_id_int,
                args=sdrwatch_args,
            )
            # Update lock with owner id
            with self._lock_path(device_key).open("w", encoding="utf-8") as f:
                f.write(jid)

            with open(log_path, "w", encoding="utf-8") as logf:
                popen_kwargs: Dict[str, Any] = {
                    "stdout": logf,
                    "stderr": subprocess.STDOUT,
                    "text": True,
                }
                if project_dir.exists():
                    popen_kwargs["cwd"] = str(project_dir)
                proc = subprocess.Popen(cmd, **popen_kwargs)

            job = Job(
                id=jid,
                created_ts=now_ts(),
                label=label,
                device_key=device_key,
                baseline_id=baseline_id_int,
                status="running",
                pid=proc.pid,
                cmd=cmd,
                log_path=log_path,
                params={k: v for k, v in sdrwatch_args.items() if k != "__discover_meta"},
            )
            self.jobs[jid] = job
            self._persist()
            # Launch reaper to free the device when the process exits
            self._spawn_reaper(jid, proc, device_key)
            return job
        except Exception:
            # on failure, release device
            self._release_device(device_key)
            raise

    def stop_job(self, job_id: str, *, wait: float = 6.0) -> Job:
        job = self.get_job(job_id)
        if not job.pid or job.status != "running":
            return job
        try:
            os.kill(job.pid, signal.SIGTERM)
        except ProcessLookupError:
            job.status = "finished"
            job.finished_ts = now_ts()
            self._release_device(job.device_key)
            self._persist()
            return job

        t0 = time.time()
        while time.time() - t0 < wait:
            if not pid_alive(job.pid):
                job.status = "finished"
                job.finished_ts = now_ts()
                self._release_device(job.device_key)
                self._persist()
                return job
            time.sleep(0.2)
        # force kill
        try:
            os.kill(job.pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except ProcessLookupError:
            pass
        job.status = "finished"
        job.finished_ts = now_ts()
        self._release_device(job.device_key)
        self._persist()
        return job

    # Pass optional profile/spur_calibration/loop/repeat/duration/jsonl params through to sdrwatch.py.
    def _build_cmd(
        self,
        *,
        script_path: Path,
        device_key: str,
        baseline_id: Optional[int],
        args: Dict[str, Any],
    ) -> List[str]:
        """Translate a stable API dict into the concrete sdrwatch.py CLI."""
        if baseline_id is None:
            raise ValueError("baseline_id is required for scanner command")
        cmd = [PYTHON_EXE, str(script_path), "--baseline-id", str(int(baseline_id))]
        soapy_args_kv: Dict[str, Any] = {}
        # Respect explicit driver override from caller (e.g., 'rtlsdr_native')
        explicit_driver = str(args.get("driver", "")).strip() if args.get("driver") is not None else ""

        # Device mapping with override support
        if device_key.startswith("rtl:"):
            if explicit_driver == "rtlsdr_native":
                cmd += ["--driver", "rtlsdr_native"]
                # native path uses librtlsdr; no --soapy-args
                soapy_args_kv = {}
            else:
                cmd += ["--driver", explicit_driver or "rtlsdr"]  # default to Soapy rtlsdr
                # Prefer serial; if not available, use index
                idx = device_key.split(":", 1)[-1]
                if args.get("__discover_meta") and args["__discover_meta"].get("serial"):
                    soapy_args_kv["serial"] = args["__discover_meta"]["serial"]
                else:
                    soapy_args_kv["index"] = idx
        elif device_key.startswith("hackrf:"):
            cmd += ["--driver", explicit_driver or "hackrf"]
            if args.get("__discover_meta") and args["__discover_meta"].get("serial"):
                soapy_args_kv["serial"] = args["__discover_meta"]["serial"]
        else:
            # Unknown key: fall back to caller-provided driver (if any)
            if explicit_driver:
                cmd += ["--driver", explicit_driver]

        # Numeric params
        mapping_num = {
            "start": "--start",
            "stop": "--stop",
            "step": "--step",
            "samp_rate": "--samp-rate",
            "fft": "--fft",
            "avg": "--avg",
            "threshold_db": "--threshold-db",
            "sleep_between_sweeps": "--sleep-between-sweeps",
            "latitude": "--latitude",
            "longitude": "--longitude",
        }
        for k, flag in mapping_num.items():
            v = args.get(k)
            if v is not None:
                cmd += [flag, str(v)]

        # Simple string/file params
        if args.get("db"):
            cmd += ["--db", str(args["db"])]
        if args.get("bandplan"):
            cmd += ["--bandplan", str(args["bandplan"])]
        if args.get("gain"):
            cmd += ["--gain", str(args["gain"])]
        if args.get("profile"):
            cmd += ["--profile", str(args["profile"])]
        if args.get("duration"):
            cmd += ["--duration", str(args["duration"])]
        if args.get("jsonl"):
            cmd += ["--jsonl", str(args["jsonl"])]

        # Loop/repeat flags
        if args.get("loop"):
            cmd.append("--loop")
        repeat_val = args.get("repeat")
        if repeat_val is not None:
            try:
                repeat_int = int(repeat_val)
            except (TypeError, ValueError):
                repeat_int = None
            if repeat_int is not None:
                cmd += ["--repeat", str(repeat_int)]

        # Spur calibration toggle
        if args.get("spur_calibration"):
            cmd.append("--spur-calibration")

        # Booleans
        if args.get("use_baseline"):
            cmd += ["--use-baseline"]

        # Soapy device selection hints
        if args.get("soapy_args"):
            cmd += ["--soapy-args", str(args["soapy_args"])]
        elif soapy_args_kv:
            kv = ",".join(f"{k}={v}" for k, v in soapy_args_kv.items())
            cmd += ["--soapy-args", kv]

        # Passthrough for any additional raw args
        extra: List[str] = args.get("extra_args", [])
        if extra:
            cmd += [str(x) for x in extra]

        return cmd

    # ---- logs ----
    def read_logs(self, job_id: str, tail: Optional[int] = None) -> str:
        job = self.get_job(job_id)
        try:
            with open(job.log_path, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
        except FileNotFoundError:
            return "<no logs yet>"
        if tail is not None and tail > 0:
            lines = data.splitlines()
            return "\n".join(lines[-tail:])
        return data

    # ---- baseline helpers ----
    def list_baselines(self) -> List[Dict[str, Any]]:
        return self._query_baselines()

    def get_baseline(self, baseline_id: int) -> Dict[str, Any]:
        rows = self._query_baselines(baseline_id=baseline_id)
        if not rows:
            raise KeyError(f"Baseline {baseline_id} not found")
        return rows[0]

    def _query_baselines(self, baseline_id: Optional[int] = None) -> List[Dict[str, Any]]:
        db_path = self.default_db_path
        if not db_path.exists():
            return []
        columns = "id, name, created_at, freq_start_hz, freq_stop_hz, notes"
        sql = f"SELECT {columns} FROM baselines"
        params: Tuple[Any, ...] = ()
        if baseline_id is not None:
            sql += " WHERE id = ?"
            params = (baseline_id,)
        else:
            sql += " ORDER BY created_at DESC, id DESC"
        try:
            conn = sqlite3.connect(str(db_path))
            conn.row_factory = sqlite3.Row
        except sqlite3.Error as exc:
            raise RuntimeError(f"failed to open database {db_path}: {exc}")
        try:
            cur = conn.execute(sql, params)
            rows = [dict(row) for row in cur.fetchall()]
            return rows
        except sqlite3.Error as exc:
            raise RuntimeError(f"failed to query baselines: {exc}")
        finally:
            conn.close()


# ---------- HTTP server (optional) ----------

def make_app(manager: JobManager, token: Optional[str] = None):
    try:
        from flask import Flask, request, jsonify # type: ignore
    except Exception as e:
        raise SystemExit("Flask is required for --serve mode. pip install flask")

    app = Flask(__name__)

    def _auth_ok() -> bool:
        if not token:
            return True
        hdr = request.headers.get("Authorization", "")
        parts = hdr.split()
        return len(parts) == 2 and parts[0].lower() == "bearer" and parts[1] == token

    @app.before_request
    def check_auth():
        if not _auth_ok():
            return (jsonify({"error": "unauthorized"}), 401)

    @app.get("/devices")
    def devices():
        devs = [asdict(d) for d in discover_devices()]
        return jsonify(devs)

    @app.get("/profiles")
    def profiles():
        try:
            payload = _fetch_profiles_payload()
        except Exception as exc:
            detail = str(exc)
            return (jsonify({"error": "failed_to_list_profiles", "detail": detail}), 500)
        return app.response_class(json.dumps(payload), mimetype="application/json")

    @app.get("/baselines")
    def baselines():
        try:
            rows = manager.list_baselines()
            return jsonify(rows)
        except Exception as exc:
            detail = str(exc)
            return (jsonify({"error": "failed_to_list_baselines", "detail": detail}), 500)

    @app.get("/baselines/<int:baseline_id>")
    def baseline_detail(baseline_id: int):
        try:
            row = manager.get_baseline(baseline_id)
            return jsonify(row)
        except KeyError as exc:
            return (jsonify({"error": str(exc)}), 404)
        except Exception as exc:
            detail = str(exc)
            return (jsonify({"error": "failed_to_fetch_baseline", "detail": detail}), 500)

    @app.get("/jobs")
    def jobs():
        return jsonify([asdict(j) for j in manager.list_jobs()])

    @app.post("/jobs")
    def start():
        payload = request.get_json(force=True) or {}
        device_key = payload.get("device_key")
        if not device_key:
            return (jsonify({"error": "device_key is required"}), 400)
        if "baseline_id" not in payload:
            return (
                jsonify({"error": "baseline_id is required; create or select a baseline first."}),
                400,
            )
        baseline_id = payload.get("baseline_id")
        try:
            baseline_id_int = int(baseline_id)
        except (TypeError, ValueError):
            return (jsonify({"error": "baseline_id must be an integer"}), 400)
        label = payload.get("label") or f"job-{short_uuid()}"
        params = payload.get("params") or {}
        try:
            job = manager.start_job(
                device_key=device_key,
                label=label,
                baseline_id=baseline_id_int,
                sdrwatch_args=params,
            )
            return jsonify(asdict(job))
        except Exception as e:
            return (jsonify({"error": str(e)}), 400)

    @app.get("/jobs/<job_id>")
    def job_detail(job_id: str):
        try:
            job = manager.get_job(job_id)
            return jsonify(asdict(job))
        except KeyError as e:
            return (jsonify({"error": str(e)}), 404)

    @app.delete("/jobs/<job_id>")
    def job_delete(job_id: str):
        try:
            job = manager.stop_job(job_id)
            return jsonify(asdict(job))
        except KeyError as e:
            return (jsonify({"error": str(e)}), 404)

    @app.get("/jobs/<job_id>/logs")
    def job_logs(job_id: str):
        tail = request.args.get("tail", type=int)
        try:
            data = manager.read_logs(job_id, tail=tail)
            return app.response_class(data, mimetype="text/plain")
        except KeyError as e:
            return (jsonify({"error": str(e)}), 404)

    return app


# ---------- CLI ----------

def cmd_discover(_args: argparse.Namespace) -> int:
    devs = discover_devices()
    if not devs:
        print("No devices found.")
        return 1
    for d in devs:
        print(f"{d.key}\t{d.kind}\t{d.label}")
    return 0


def parse_kv_pairs(pairs: List[str]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for p in pairs:
        if "=" not in p:
            continue
        k, v = p.split("=", 1)
        # Try to coerce numbers where sensible
        try:
            if v.lower().endswith("e6") or v.lower().endswith("e3"):
                out[k] = float(v)
            else:
                # int or float
                if "." in v:
                    out[k] = float(v)
                else:
                    out[k] = int(v)
        except Exception:
            if v.lower() in ("true", "false"):
                out[k] = (v.lower() == "true")
            else:
                out[k] = v
    return out


def cmd_start(args: argparse.Namespace) -> int:
    jm = JobManager()
    params = {
        "start": args.start,
        "stop": args.stop,
        "step": args.step,
        "samp_rate": args.samp_rate,
        "fft": args.fft,
        "avg": args.avg,
        "gain": args.gain,
        "threshold_db": args.threshold_db,
        "sleep_between_sweeps": args.sleep_between_sweeps,
        "bandplan": args.bandplan,
        "db": args.db,
        "duration": args.duration,
        "use_baseline": args.use_baseline,
        "latitude": args.latitude,
        "longitude": args.longitude,
    }
    params.update(parse_kv_pairs(args.param))
    if args.extra:
        params["extra_args"] = args.extra

    job = jm.start_job(
        device_key=args.device,
        label=args.label or f"job-{short_uuid()}",
        baseline_id=args.baseline_id,
        sdrwatch_args=params,
    )
    print(job.id)
    return 0


def cmd_list(_args: argparse.Namespace) -> int:
    jm = JobManager()
    rows = jm.list_jobs()
    if not rows:
        print("No jobs.")
        return 0
    for j in rows:
        created = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(j.created_ts))
        print(f"{j.id}\t{j.status}\t{j.device_key}\t{created}\t{j.label}")
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    jm = JobManager()
    try:
        j = jm.get_job(args.job_id)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(json.dumps(asdict(j), indent=2))
    return 0


def cmd_logs(args: argparse.Namespace) -> int:
    jm = JobManager()
    try:
        data = jm.read_logs(args.job_id, tail=args.tail)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 1
    sys.stdout.write(data)
    if not data.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def cmd_stop(args: argparse.Namespace) -> int:
    jm = JobManager()
    try:
        j = jm.stop_job(args.job_id)
    except KeyError as e:
        print(str(e), file=sys.stderr)
        return 1
    print(json.dumps(asdict(j), indent=2))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    token = args.token or os.environ.get("SDRWATCH_CONTROL_TOKEN")
    jm = JobManager()
    app = make_app(jm, token=token)
    app.run(host=args.host, port=args.port, debug=False)
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sdrwatch-control", description="Manage sdrwatch.py jobs across SDR devices")
    sub = p.add_subparsers(dest="cmd", required=True)

    # discover
    s = sub.add_parser("discover", help="List available SDR devices")
    s.set_defaults(func=cmd_discover)

    # start
    s = sub.add_parser("start", help="Start a new sdrwatch.py job")
    s.add_argument("--device", required=True, help="Device key (e.g., rtl:0, hackrf:<serial>)")
    s.add_argument("--label", default=None, help="Human-friendly label for this job")
    s.add_argument("--baseline-id", type=int, required=True, help="Existing baseline id to associate with the scan")

    # Common sdrwatch args
    s.add_argument("--start", type=float, help="Start frequency (Hz)")
    s.add_argument("--stop", type=float, help="Stop frequency (Hz)")
    s.add_argument("--step", type=float, help="Step size (Hz)")
    s.add_argument("--samp-rate", dest="samp_rate", type=float, help="Sample rate (Hz)")
    s.add_argument("--fft", type=int, help="FFT bins")
    s.add_argument("--avg", type=int, help="Averaging")
    s.add_argument("--gain", type=str, help="Gain (e.g., auto or dB)")
    s.add_argument("--threshold-db", dest="threshold_db", type=float, help="Detection threshold (dB)")
    s.add_argument("--sleep-between-sweeps", type=float, help="Seconds to sleep between sweeps")
    s.add_argument("--duration", type=str, help="Total runtime (e.g., 30m, 10s)")
    s.add_argument("--bandplan", type=str, help="Path to bandplan CSV")
    s.add_argument("--db", type=str, help="Path to SQLite DB")
    s.add_argument("--use-baseline", action="store_true", help="Enable baseline subtraction if supported")
    s.add_argument("--latitude", type=float, help="Latitude in decimal degrees to tag scans")
    s.add_argument("--longitude", type=float, help="Longitude in decimal degrees to tag scans")

    s.add_argument("--param", action="append", default=[], help="Extra k=v args to pass through (coerces numbers/bools)")
    s.add_argument("--extra", nargs=argparse.REMAINDER, help="Raw extra args appended after '--' to sdrwatch.py")
    s.set_defaults(func=cmd_start)

    # list
    s = sub.add_parser("list", help="List all jobs")
    s.set_defaults(func=cmd_list)

    # status
    s = sub.add_parser("status", help="Show job details as JSON")
    s.add_argument("job_id")
    s.set_defaults(func=cmd_status)

    # logs
    s = sub.add_parser("logs", help="Print job logs")
    s.add_argument("job_id")
    s.add_argument("--tail", type=int, default=None)
    s.set_defaults(func=cmd_logs)

    # stop
    s = sub.add_parser("stop", help="Stop a running job")
    s.add_argument("job_id")
    s.set_defaults(func=cmd_stop)

    # serve
    s = sub.add_parser("serve", help="Run localhost JSON API (requires Flask)")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    s.add_argument("--token", default=None, help="Bearer token for Authorization header")
    s.set_defaults(func=cmd_serve)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    ensure_dirs()
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    try:
        return args.func(args)
    except RuntimeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
