# Quickstart: Validate No-Hardware Simulation Mode

## Purpose

Use this guide to validate the no-hardware simulation mode introduced for SDRwatch.

## Prerequisites

- Repository checked out locally.
- Python 3.11 or newer available on the target machine.
- No SDR hardware is required.
- A virtual environment with SDRwatch installed, for example `python -m pip install -e ".[dev]"`.

## Validation References

- Contract: [contracts/simulation-mode-contract.md](./contracts/simulation-mode-contract.md)
- Data model: [data-model.md](./data-model.md)
- Plan: [plan.md](./plan.md)

## Scenario 1: Create a baseline for simulation

1. Start the web app against a fresh database file:

```bash
python sdrwatch-web.py --db sim-sdrwatch.db --host 127.0.0.1 --port 8080
```

2. In another shell, start the controller if you want the browser workflow available:

```bash
python sdrwatch-control.py serve --host 127.0.0.1 --port 8765
```

3. Open the web UI, create a baseline, and note the baseline ID.

**Expected outcome**: A baseline exists in `sim-sdrwatch.db` and can be selected for subsequent scans.

**CLI-only shortcut**: If you only need the no-hardware scanner path, you can skip the browser setup and use `--baseline-id latest` in Scenario 2. SDRwatch will reuse the newest baseline in `sim-sdrwatch.db` or create one on first run.

## Scenario 2: Run a simulated FM-band scan without hardware

1. From the repository root, run a deterministic simulated scan against the FM broadcast band:

```bash
python -m sdrwatch.cli --driver sim --profile fm_broadcast --baseline-id latest --start 88e6 --stop 108e6 --db sim-sdrwatch.db --repeat 3
```

2. Optionally emit JSONL alongside the DB writes:

```bash
python -m sdrwatch.cli --driver sim --profile fm_broadcast --baseline-id latest --start 88e6 --stop 108e6 --db sim-sdrwatch.db --repeat 3 --jsonl sim-events.jsonl
```

**Expected outcome**: The scan completes without trying to access SDR hardware, `scan_updates` records one row per sweep, `baseline_noise` and `baseline_occupancy` contain per-bin updates, and `baseline_detections` contains persistent detections produced by the simulated carriers.

## Scenario 3: Inspect simulated detections in the dashboard

1. Point the web UI at the database used in the simulated run if it is not already running there:

```bash
python sdrwatch-web.py --db sim-sdrwatch.db --host 127.0.0.1 --port 8080
```

2. Open the dashboard and select the simulated baseline.

3. Confirm that signal cards, recent activity, and baseline summaries reflect the simulated run.

**Expected outcome**: The existing dashboard surfaces detections and recent scan activity from `sim-sdrwatch.db` without any hardware attached, including a populated Baseline overview section and signal cards for the simulated detections.

## Scenario 4: Run hardware-free automated validation

1. Execute the focused no-hardware test slice covering simulation determinism, dashboard compatibility, and DB writes:

```bash
python -m pytest -q tests/test_sim_mode.py
```

2. Execute the broader no-hardware regression suite used by CI:

```bash
python -m pytest -q tests
```

**Expected outcome**: The focused simulation tests pass on a machine with no SDR hardware and validate deterministic simulated output plus required writes to the standard baseline tables, and the broader no-hardware regression suite still passes.

## Completion Criteria

- A baseline can be created and reused with a simulated scan.
- A developer can run the documented FM-band simulation without SDR hardware.
- The generated database shows simulated detections and recent activity in the current dashboard.
- Automated no-hardware validation covers deterministic output and required DB writes.