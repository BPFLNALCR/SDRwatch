# Quickstart: Validate the Python Development Setup

## Purpose

Use this guide to validate the reproducible Python development setup introduced for SDRwatch.

## Prerequisites

- Repository checked out locally.
- Python 3.11 or newer available on the target machine.
- For Linux/Pi no-hardware validation: a shell that can create virtual environments.
- For Linux/Pi hardware validation only: OS packages and device permissions installed separately.

## Validation References

- Contract: [contracts/development-setup-contract.md](./contracts/development-setup-contract.md)
- Data model: [data-model.md](./data-model.md)
- Plan: [plan.md](./plan.md)

## Scenario 1: Linux/Pi no-hardware contributor setup

1. Create and populate a local virtual environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

2. Run the no-hardware smoke checks:

```bash
python -c "import sdrwatch.cli; from sdrwatch_web import create_app; print('imports ok')"
python -m sdrwatch.cli --help
python sdrwatch-control.py --help
python sdrwatch-web.py --help
python -m pytest tests/test_detection_diagnostics.py tests/test_extent_hysteresis.py tests/test_segment_splitting.py
```

**Expected outcome**: The editable install succeeds, the CLI and web import surfaces load without touching SDR hardware, the root scripts report help output successfully, and the existing no-hardware pytest suite passes.

## Scenario 2: Raspberry Pi/Linux hardware-oriented setup

1. Install the documented OS prerequisites first:

```bash
sudo apt update
sudo apt install -y python3-venv python3-numpy python3-scipy librtlsdr0 librtlsdr-dev rtl-sdr libusb-1.0-0 libusb-1.0-0-dev
```

2. Create a virtual environment that can reuse the OS-provided numeric stack and install the project:

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

3. Run hardware-boundary checks separately from no-hardware smoke tests:

```bash
rtl_test -t
python -m sdrwatch.cli --list-profiles
```

4. If a prepared baseline and SDR device are available, run an optional live scan validation:

```bash
python -m sdrwatch.cli --baseline-id 1 --start 88e6 --stop 108e6 --step 1.8e6 --duration 10
```

**Expected outcome**: OS prerequisites satisfy the numeric and SDR stack, `rtl_test` confirms device visibility, no-hardware smoke commands still work, and an optional live scan remains a Linux/Pi-only step.

## Scenario 3: Windows no-hardware contributor setup

1. Create a virtual environment without relying on shell activation:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -U pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

2. Run the supported no-hardware validation commands:

```powershell
.\.venv\Scripts\python -c "import sdrwatch.cli; from sdrwatch_web import create_app; print('imports ok')"
.\.venv\Scripts\python -m sdrwatch.cli --help
.\.venv\Scripts\python sdrwatch-control.py --help
.\.venv\Scripts\python sdrwatch-web.py --help
.\.venv\Scripts\python -m pytest tests\test_detection_diagnostics.py tests\test_extent_hysteresis.py tests\test_segment_splitting.py
```

**Expected outcome**: The editable install and no-hardware smoke checks succeed on Windows, and the documentation makes clear that SDR hardware validation is not part of the supported Windows workflow.

## Scenario 4: CI parity check

1. From a clean environment matching the CI runner, install and validate using the documented no-hardware path:

```bash
python -m pip install -U pip
python -m pip install -e ".[dev]"
python -c "import sdrwatch.cli; from sdrwatch_web import create_app; print('imports ok')"
python -m pytest tests/test_detection_diagnostics.py tests/test_extent_hysteresis.py tests/test_segment_splitting.py
```

2. Confirm the workflow file uses the same install source and no-hardware checks.

**Expected outcome**: The local validation commands match the updated CI workflow's install source and smoke/test contract.

## Completion Criteria

- Editable installation works from a clean checkout without `install-sdrwatch.sh`.
- CLI import smoke and web import smoke both succeed without SDR hardware.
- The repository's current no-hardware pytest suite passes in the contributor environment.
- Linux/Pi hardware checks are documented separately from no-hardware validation.
- CI uses the same canonical install source and no-hardware commands.