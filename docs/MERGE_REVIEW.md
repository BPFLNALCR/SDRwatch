# Spec Kit / Copilot Merge Review

Reviewed on: 2026-06-09
Baseline: `devControl` (`f02ee88`)
Consolidation branch: `consolidate-speckit-workflow`

## Branches Reviewed

| Branch | Head | Relationship to `devControl` | Review result |
| --- | --- | --- | --- |
| `001-document-repo-inventory` | `467113f` | One commit ahead of `devControl` | Mostly documentation and Spec Kit template updates, plus unwanted tracked bytecode churn. Safe pieces should be cherry-picked manually. |
| `002-add-python-dev-setup` | `467113f` | Same commit as `001-document-repo-inventory` | Duplicate branch name pointing at the same inventory/stabilization commit. No separate diff to merge. |
| `003-add-sim-mode` | `9f6dac5` | Builds on `467113f` with additional dev setup and simulation-mode feature work | Do not merge wholesale. It includes product feature changes and tests that need isolated review. |

Although three numbered branches exist, only two distinct heads were created by the
recent Spec Kit/Copilot work: `467113f` for `001`/`002`, and `9f6dac5` for `003`.

## Files Changed by Reviewed Branches

### `001-document-repo-inventory` and `002-add-python-dev-setup`

Changed relative to `devControl`:

- `.github/copilot-instructions.md`
- `.gitignore`
- `.specify/feature.json`
- `.specify/memory/constitution.md`
- `.specify/templates/plan-template.md`
- `.specify/templates/spec-template.md`
- `.specify/templates/tasks-template.md`
- `AGENTS.md`
- `README.md`
- `docs/PROJECT_INVENTORY.md`
- `specs/001-document-repo-inventory/**`
- `tests/__pycache__/*.pyc`

### `003-add-sim-mode`

Additional changed files relative to `devControl`:

- `.github/workflows/ci.yml`
- `install-sdrwatch.sh`
- `pyproject.toml`
- `sdrwatch/cli.py`
- `sdrwatch/drivers/simulate.py`
- `sdrwatch/sweep/runner.py`
- `specs/002-python-dev-setup/**`
- `specs/003-add-sim-mode/**`
- `tests/test_import_smoke.py`
- `tests/test_pyproject_contract.py`
- `tests/test_sim_mode.py`

It also includes all changes from the `001`/`002` head.

## Safe Changes to Keep

- Replace the placeholder `.specify/memory/constitution.md` with a real SDRwatch
  constitution.
- Keep the existing Spec Kit scaffolding already present on `devControl`.
- Preserve `.github/copilot-instructions.md` and clarify that Codex is the default
  Spec Kit integration while Copilot prompts/agents remain usable.
- Add `docs/PROJECT_INVENTORY.md` as a current-state map of runtime entry points,
  package ownership, database ownership, generated artifacts, and test surface.
- Preserve detection diagnostics and `tests/test_detection_diagnostics.py`, which
  already exist on `devControl`.

## Risky Changes to Delay

- Do not merge `sdrwatch/drivers/simulate.py`, `sdrwatch/cli.py`, or
  `sdrwatch/sweep/runner.py` from `003-add-sim-mode` until simulation mode is
  reviewed as a separate product feature with passing tests.
- Do not merge `pyproject.toml` from `003-add-sim-mode` until packaging policy is
  decided. It changes developer workflow expectations.
- Do not merge `.github/workflows/ci.yml` from `003-add-sim-mode` until CI behavior
  is reviewed independently.
- Do not merge installer changes from `003-add-sim-mode` without Raspberry Pi
  deployment validation.
- Do not carry over tracked `tests/__pycache__/*.pyc` modifications from any branch.
- Do not merge broad README edits as part of this consolidation unless they are
  reduced to workflow-state corrections.
- Do not use this consolidation pass to merge large UI rewrites or additional
  monitoring-zone/friendly-signal product work. Any such changes should remain
  isolated with their own tests and operator-facing review.

## Tests and Checks Performed

Commands requested for this consolidation:

```bash
python -m pytest -q
python -m sdrwatch.cli --list-profiles
python -c "from sdrwatch_web import create_app; app = create_app(); print(app.name)"
```

Results from this branch:

| Check | Result |
| --- | --- |
| `python -m pytest -q` | Not runnable in this shell because `python` is not on PATH. Retried with bundled Codex Python; failed before collection because `pytest` is not installed in that runtime. |
| `python -m sdrwatch.cli --list-profiles` | Passed with bundled Codex Python. It listed `fm_broadcast`, `ism_902`, and `vhf_uhf_general`. |
| `from sdrwatch_web import create_app` | Not runnable with system `python` because `python` is not on PATH. Retried with bundled Codex Python; failed because `flask` is not installed in that runtime. |

No scanner, controller, web UI, or database-layer feature code was changed on this
branch, so hardware validation was not performed here.

## Recommended Merge Order

1. Merge `consolidate-speckit-workflow` into `devControl` first. This fixes the
   constitution placeholder, documents the reviewed branch state, and clarifies the
   Spec Kit/Copilot workflow without adding product features.
2. Close or delete `002-add-python-dev-setup` if it remains identical to
   `001-document-repo-inventory`; it does not represent separate work.
3. Review `001-document-repo-inventory` only for any remaining docs or template
   changes not already captured here. Do not merge tracked bytecode.
4. Review `003-add-sim-mode` as a standalone feature branch after deciding whether
   no-hardware simulation mode is desired. Require focused review of scanner-driver
   integration, CLI behavior, database writes, CI changes, installer changes, and the
   new tests before merge.

## Current Consolidation Scope

This branch intentionally keeps the diff small:

- Fixed `.specify/memory/constitution.md`.
- Added workflow-state notes for Codex/Copilot usage.
- Added project inventory and merge-review documentation.
- Avoided product feature code, UI rewrites, database rewrites, and installer changes.
