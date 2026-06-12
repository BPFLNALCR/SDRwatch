# Phase 0 Research: FM Detection Card Stability

## Decision: Treat the FM failure as card overproduction, not candidate starvation

**Rationale**: The diagnostic bundle contains 1408 emitted segments, 1331 accepted hits, 828 promoted detections, and 234 persisted baseline rows. This is the opposite of the previous zero-card failure. The issue is that most rows are tiny and unstable: 193 persisted detections are under 1 kHz, and 209 of 234 were missing at export.

**Alternatives considered**:

- Raise thresholds until fewer cards appear: rejected because it hides activity without stabilizing segmentation, matching, or width behavior.
- Reopen candidate-starvation tuning: rejected because the bundle proves candidate, hit, promotion, and persistence paths are active.
- Treat this as a dashboard rendering problem: rejected because `baseline_detections.json` itself contains the explosion.

## Decision: Keep Discovery and FM Validation as separate GUI paths

**Rationale**: Discovery intentionally uses relaxed promotion so new users can see first-light cards. FM Validation needs more band-specific stability for 88-108 MHz. Combining both into one preset would either regress first-light or keep the FM explosion.

**Alternatives considered**:

- Make Discovery stricter globally: rejected because it risks returning to no-card onboarding.
- Replace Discovery with FM Validation: rejected because Discovery is broader than FM and is already a completed operator-facing workflow.
- Add CLI-only FM recipes: rejected because SDRwatch is GUI-operated.

## Decision: Reuse the existing `fm_broadcast` profile before adding new algorithms

**Rationale**: `sdrwatch/io/profiles.py` already defines an FM-focused profile with overlap, two-pass, center matching, cluster merge width, min match/display bandwidths, width cap, centroid mode, and revisit limits. The failing bundle did not use `--profile fm_broadcast`, so those values were bypassed.

**Alternatives considered**:

- Write a new FM detector: rejected as too broad before testing existing profile behavior.
- Duplicate all FM profile values only in browser JavaScript: possible but risks drift unless tests enforce sync with `serialize_profiles()`.
- Use only threshold/gain changes: rejected because the observed rows are promoted fragments with width/matching issues.

## Decision: Make FM profile application testable from the GUI/controller path

**Rationale**: Some important FM profile values are currently only reachable when the scanner applies `--profile fm_broadcast`; they are not standalone CLI/controller flags. FM Validation must either submit `profile=fm_broadcast` intentionally or add narrow passthrough for the existing fields. Tests should prove which behavior is chosen.

**Alternatives considered**:

- Rely on hidden profile defaults without tests: rejected because the current failure is partly a wiring gap.
- Expose every profile field as a visible Expert control: rejected because it would clutter the GUI and make operator workflow worse.
- Move profile expansion into the web layer: rejected because scanner/profile ownership belongs in `sdrwatch.io.profiles` and `sdrwatch.cli`.

## Decision: Stabilize FM cards with span shaping and matching, not only thresholds

**Rationale**: FM stations can present narrow pilots, edges, or spiky lobes in coarse PSD windows. For useful cards, the persisted/displayed span should reflect FM-scale signal behavior while the match span remains tight enough to keep adjacent stations separate. Existing profile values already suggest this split: tighter match width, wider display width, center tolerance, and width cap.

**Alternatives considered**:

- Force every FM row to 200 kHz for matching: rejected because adjacent stations could merge.
- Keep all rows at raw sub-kHz segment widths: rejected because it creates unusable tiny cards and unstable missing behavior.
- Increase `min_width_bins` alone: rejected because it can drop fragments without solving update/merge behavior.

## Decision: Cover baseline upsert/matching explicitly

**Rationale**: The bundle contains 234 rows in a 20 MHz FM band, with many active and missing rows under 1 kHz. Repeated nearby fragments should update existing rows under FM Validation tolerances. `BaselinePersistence._match_persistent()` is the stage that decides whether a promoted detection updates an existing row or inserts a new one.

**Alternatives considered**:

- Only adjust the detector's emitted segments: rejected because persistence can still over-insert if matching remains too fine.
- Only clean old rows after the fact: rejected because it treats symptoms and could hide create/update regressions.
- Add a new table for FM stations: rejected because no schema migration is needed for the current problem.

## Decision: Keep non-FM narrow-signal behavior scoped and tested

**Rationale**: SDRwatch must still support narrow carriers outside FM Validation. FM-specific display and match widths must not become global defaults that widen unrelated signals.

**Alternatives considered**:

- Apply FM minimum widths globally: rejected because it would damage non-FM detection fidelity.
- Add band-specific behavior based only on frequency range: deferred; explicit FM Validation/profile selection is safer and testable.

## Decision: Make width clamp behavior observable

**Rationale**: The failing run did not pass `--max-detection-width-hz`, so the effective hard cap was disabled. The existing code can apply minimum width, padding, EMA/hysteresis, outlier rejection, and max-width caps, but the bundle does not make those decisions obvious.

**Alternatives considered**:

- Leave width behavior implicit: rejected because future tuning would still require inference.
- Add verbose unbounded logs for every bin: rejected because diagnostic bundles must stay bounded.
- Add a compact per-sweep or per-decision summary: accepted as the target direction.

## Decision: Enable or expose two-pass behavior deliberately for FM Validation

**Rationale**: The existing `fm_broadcast` profile sets `two_pass=True`, but the failing command lacked `--two-pass` and diagnostics showed `num_revisits=0`. FM Validation should make revisit behavior visible and bounded if it is part of stabilization.

**Alternatives considered**:

- Leave two-pass off for FM Validation: possible only if tests prove profile span/matching alone stabilizes cards.
- Enable unbounded revisits: rejected because Raspberry Pi scan cadence must remain predictable.
- Make two-pass a CLI-only instruction: rejected because operator validation is GUI-first.

## Decision: Improve diagnostics at persistence/revisit boundaries

**Rationale**: Current diagnostics are good for window-level evidence, but the FM bundle still requires inference about create/update/no-match, width clamp, and missing decisions. Future bundles should show those decisions directly.

**Alternatives considered**:

- Rely only on scanner log tail text: rejected because log tails are bounded and not structured enough for reports.
- Export the entire database or unbounded logs: rejected because bundle size must stay bounded.
- Emit compact structured summaries and include them in the bundle: accepted.

## Decision: Validate through no-hardware tests first, then GUI hardware acceptance

**Rationale**: Deterministic fixtures can prove grouping, separation, width, and upsert behavior before touching hardware. Final acceptance still requires Raspberry Pi 5 and RTL-SDR Blog v4 through the web GUI because RF behavior is hardware/environment dependent.

**Alternatives considered**:

- Depend only on hardware iteration: rejected because it is slower and less precise for regression coverage.
- Treat scanner CLI runs as acceptance: rejected because scanner CLI is internal backend tooling.
