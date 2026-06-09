# Data Model: Repository Inventory and Stabilization

## Overview

This feature does not introduce application persistence changes. Its data model is a conceptual model for the developer-facing inventory document and its validation artifacts.

## Entities

### 1. Inventory Document

- **Purpose**: The single developer-facing document to be created at `docs/PROJECT_INVENTORY.md`.
- **Fields**:
  - `path`: expected output path.
  - `audience`: contributors and maintainers.
  - `scope`: repository-wide runtime and structure inventory.
  - `sections`: ordered list of inventory sections.
  - `evidence_sources`: repository files used to support claims.
  - `last_verified_date`: date the document was checked against the repo.
- **Validation rules**:
  - Must remain documentation-only.
  - Must distinguish canonical, compatibility, generated, and missing artifacts.
  - Must cover all categories required by the specification.

### 2. Inventory Section

- **Purpose**: A required section inside the inventory document.
- **Fields**:
  - `title`: section heading.
  - `purpose`: why the section exists.
  - `required_topics`: list of topics that must appear.
  - `evidence_files`: files that substantiate the section.
  - `status`: draft, verified, or needs-update.
- **Validation rules**:
  - Each section must map to at least one requirement from the spec.
  - Each section must cite current repository evidence before implementation is considered complete.

### 3. Runtime Component

- **Purpose**: A scanner, controller, web, query, installer, or support component that contributors need to place correctly.
- **Fields**:
  - `name`: component name.
  - `category`: scanner, controller, web, helper, deployment, or support.
  - `entrypoint`: module path, script path, or directory path.
  - `status`: canonical, compatibility, generated-only, or support-only.
  - `responsibility_summary`: contributor-facing explanation of its role.
  - `related_sections`: inventory sections where it appears.
- **Validation rules**:
  - Canonical and compatibility components must be distinguishable.
  - Responsibility summaries must not overstate ownership beyond observed code behavior.

### 4. Operational Artifact

- **Purpose**: A file or directory that is committed, installer-generated, or runtime-generated.
- **Fields**:
  - `name`: artifact name.
  - `lifecycle`: committed, installer-generated, or runtime-generated.
  - `default_path`: expected default location if known.
  - `producer`: script, service, or module that creates or manages it.
  - `purpose`: why the artifact exists.
  - `notes`: caveats such as optional creation or environment overrides.
- **Validation rules**:
  - Generated artifacts must state who creates them and under what conditions.
  - Missing artifacts must not be listed as committed.

### 5. Repository Gap

- **Purpose**: A notable absent contributor-facing artifact that should be acknowledged in the inventory.
- **Fields**:
  - `name`: missing file, directory, or standard asset.
  - `reason_for_note`: why contributors would expect it.
  - `current_state`: absent, generated-only, or partial.
  - `impact`: contributor or maintenance consequence.
- **Validation rules**:
  - Must be directly observable from the current repository state.
  - Must be phrased as an observation, not as implementation work for this feature.

### 6. Validation Scenario

- **Purpose**: A reproducible way to check the completed inventory against the repo.
- **Fields**:
  - `name`: scenario title.
  - `environment`: workstation/no-hardware or Raspberry Pi/hardware.
  - `prerequisites`: what must exist before running it.
  - `commands_or_checks`: executable commands or inspection steps.
  - `expected_outcome`: observable result.
- **Validation rules**:
  - At least one scenario must be runnable without SDR hardware.
  - Hardware-dependent scenarios must be optional for this feature.

## Relationships

- One **Inventory Document** contains many **Inventory Sections**.
- Each **Inventory Section** references one or more **Runtime Components**, **Operational Artifacts**, or **Repository Gaps**.
- **Validation Scenarios** verify that the **Inventory Document** accurately represents those components and artifacts.
- **Operational Artifacts** and **Repository Gaps** provide evidence for deployment and stabilization sections of the document.

## State Transitions

### Inventory Section Status

- `draft` -> `verified`: after the section is checked against current repository files.
- `verified` -> `needs-update`: when repository structure changes or a claim becomes stale.
- `needs-update` -> `verified`: after the section is refreshed and re-validated.

### Inventory Document Status

- `draft` -> `verified`: once all required sections are present and validation scenarios pass.
- `verified` -> `stale`: when a runtime component, artifact, or gap note no longer matches the repository.