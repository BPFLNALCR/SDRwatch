# Contract: Characterization And Persistence Compatibility

This contract defines how the new characterization layer must coexist with the existing persistent detection model without breaking FM card stability or current invariants.

## Existing Persistent Detection Contract To Preserve

The existing persistent signal model remains compatible:

- `baseline_detections` remains the stable signal record used by current FM Validation behavior.
- Existing compatibility fields such as `f_low_hz`, `f_high_hz`, `f_center_hz`, `confidence`, `service`, `region`, and `bandplan_notes` remain valid.
- Existing UI behavior that depends on stable station-scale cards must continue to work.

## Separation Rules

- Persistence or match span must not be redefined to mean measured occupied bandwidth.
- Display or card span must not be redefined to mean measured occupied bandwidth.
- Measured occupied bandwidth must not become the only basis for persistence matching if doing so would merge nearby stations.
- Contextual service or profile labels must not be treated as measured classification evidence.

## Required Invariants

- `f_low_hz <= f_center_hz <= f_high_hz` must remain true after coarse-pass updates.
- `f_low_hz <= f_center_hz <= f_high_hz` must remain true after revisit confirmation or trimming.
- Width and center updates must preserve nearby-station separation in the accepted FM validation scenarios.
- Narrow non-FM signals outside FM Validation must not inherit FM-scale widening or FM candidate labels by default.

## Storage Mode A: Diagnostics-First Characterization

If the first implementation keeps characterization outside the persistent schema:

- persistent detection rows continue to store the existing stable match or display behavior as they do today
- characterization records must still be exported with a stable enough key to relate them to the corresponding persistent signal or diagnostic event
- center and bandwidth stability must still be observable in diagnostics or summaries

## Storage Mode B: Additive Persistent Characterization

If later planning proves durable characterization fields are necessary:

- new fields must be additive
- existing fields must keep their current meaning
- additive fields should use explicit names such as `measured_center_hz`, `measured_bandwidth_hz`, `center_stability_hz`, and `bandwidth_stability_hz`
- migration and compatibility validation must be planned before implementation

## Acceptance Checks

- The current FM Validation stable-card behavior is preserved.
- Nearby FM stations remain separate.
- Revisit refinement improves characterization without creating extra cards for the same station.
- The persistence invariant test passes after characterization-driven updates.
- The chosen storage mode is explicit and justified in implementation planning.
