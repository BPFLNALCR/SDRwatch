# Research: Repository Inventory and Stabilization

## Decision 1: Center the inventory on authoritative entry points and explicitly label compatibility shims

- **Decision**: Treat `python -m sdrwatch.cli`, `sdrwatch-control.py`, and `sdrwatch-web.py` as the canonical runtime entry points, while documenting `sdrwatch.py` and installer-created web shims as compatibility paths.
- **Rationale**: The repository currently contains both preferred entry points and legacy wrappers. Contributors need one place that says which executable path should be treated as authoritative for future work.
- **Alternatives considered**:
  - Treat every top-level script as equally current. Rejected because it would hide the deprecation boundary and undercut the stabilization goal.
  - Document only the newest paths and omit wrappers. Rejected because maintainers still need to understand compatibility behavior that exists in the repo and installer.

## Decision 2: Document schema ownership as intentionally split across scanner and web layers

- **Decision**: Describe core schema creation as owned by `sdrwatch/baseline/store.py`, with `sdrwatch_web/schema.py` and `sdrwatch_web/db.py` responsible for startup-time additive schema work such as classification columns, monitoring zones, and friendly signals.
- **Rationale**: Repository inspection shows no single migrations directory or formal schema versioning layer. The most accurate documentation is to name the current split ownership and the startup migration path rather than imply there is a centralized migration system.
- **Alternatives considered**:
  - Describe the schema as owned solely by the scanner. Rejected because the web package performs real ALTER TABLE and CREATE TABLE work at startup.
  - Describe the schema as owned solely by the web app. Rejected because the scanner package initializes the core baseline tables and persists scan-time data.

## Decision 3: Classify operational files by lifecycle instead of mixing committed and generated artifacts

- **Decision**: Organize runtime and deployment files into three buckets in the future inventory document: committed repository artifacts, installer-generated assets, and runtime-generated state.
- **Rationale**: The main source of contributor confusion is not just where files live, but whether they are expected to exist before install, after install, or only while services are running. A lifecycle-based presentation makes troubleshooting faster.
- **Alternatives considered**:
  - Present files only by absolute path. Rejected because identical-looking paths can represent very different ownership and creation moments.
  - Focus only on committed files. Rejected because the feature specifically needs to explain generated service units, env files, locks, logs, and state files.

## Decision 4: Keep local validation hardware-optional and source-backed

- **Decision**: Treat the documentation feature's main validation path as no-hardware verification using source inspection, CLI help output, profile listing, existing pytest coverage, and optional web startup against an existing database; treat Raspberry Pi and RTL-SDR validation as a secondary check for deployment claims.
- **Rationale**: This feature does not change runtime behavior, so requiring live hardware to validate the documentation would add cost without improving confidence in the implementation itself.
- **Alternatives considered**:
  - Require end-to-end hardware validation before considering the feature complete. Rejected because the change is documentation-only.
  - Skip all runnable validation and rely purely on prose review. Rejected because the constitution requires a reproducible verification path.

## Decision 5: Record missing standard project files as current repository gaps, not remediation work

- **Decision**: The future inventory document will call out absent packaging metadata, dependency manifests, committed service files, and formal migrations directories as observed gaps without attempting to add or normalize them in this feature.
- **Rationale**: The specification explicitly limits the work to discovery and documentation. The planning artifacts should preserve that boundary and avoid turning the inventory feature into a packaging or deployment refactor.
- **Alternatives considered**:
  - Expand scope to add `pyproject.toml`, requirements files, or committed unit files now. Rejected because it violates the discovery-only requirement.
  - Omit missing assets from the inventory to keep the document positive-only. Rejected because the missing files are part of the contributor-facing reality the feature is meant to explain.