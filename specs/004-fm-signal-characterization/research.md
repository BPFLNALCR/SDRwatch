# Phase 0 Research: FM Signal Characterization

## Decision: Add characterization as a separate evidence layer, not as a replacement for stable FM cards

**Rationale**: The previous feature already solved the operator-facing card explosion by stabilizing FM cards. The next feature should preserve that win and add a technically honest characterization layer beside it, rather than changing the card span to mean something new.

**Alternatives considered**:

- Treat the displayed card width as the measured occupied bandwidth: rejected because the display span is already shaped for stable operator presentation.
- Replace stable FM cards with raw fragment-level evidence: rejected because it would regress the operator-facing outcome that was just restored.
- Delay characterization entirely until later classification work: rejected because FM broadcast is the current best control target for validating the next layer safely.

## Decision: Keep raw detector span, measured characterization, persistence match span, and display span as four distinct concepts

**Rationale**: The current code already distinguishes raw cluster extent from match and display span. The new feature should add a separate measured occupied-bandwidth concept instead of collapsing any of these layers together.

**Alternatives considered**:

- Reuse raw segment width as the measured occupied bandwidth: rejected because raw fragments can be far narrower than the real station-scale signal.
- Reuse match span as the measured occupied bandwidth: rejected because the match span is tuned to keep nearby stations separate, not to describe true occupancy.
- Reuse display span as the measured occupied bandwidth: rejected because display spans are intentionally shaped for human stability and visibility.

## Decision: Treat bandplan and profile metadata as contextual evidence, not classification proof

**Rationale**: `Bandplan.lookup()` and the selected profile already provide useful context such as "FM Broadcast" and `fm_broadcast`, but those values are externally supplied context. They are helpful hints, not measured proof of modulation or service identity.

**Alternatives considered**:

- Assume every signal inside the FM broadcast band is truly FM broadcast: rejected because this would overclaim based on context alone.
- Remove contextual labels entirely: rejected because operators and maintainers still benefit from contextual band and profile information.
- Merge contextual labels into the future classification result: rejected because it would make later automatic classification less trustworthy.

## Decision: Use revisit-derived evidence as a preferred refinement source for characterization

**Rationale**: Revisit passes are the best existing mechanism for obtaining sharper evidence without redesigning the detector. They are already part of the FM profile and can improve measured center, measured bandwidth, and optional FM-specific evidence.

**Alternatives considered**:

- Ignore revisit and rely only on coarse fragments: rejected because characterization quality would remain limited by coarse sweep resolution.
- Make revisit mandatory for all characterization: rejected because bounded operation on Raspberry Pi still matters and some runs may keep revisit disabled.
- Add a new measurement-only scan mode: rejected as broader than needed for this feature.

## Decision: Confidence must report evidence sources, not only a score

**Rationale**: Later classification work will need to know whether confidence came from strong coarse SNR, repeat stability, revisit confirmation, or optional FM-specific indicators. A score without provenance would be hard to trust and hard to debug.

**Alternatives considered**:

- Emit only a numeric confidence value: rejected because it would hide why the score changed.
- Emit only qualitative labels such as high or low: rejected because maintainers still need a measurable signal of confidence change.
- Delay confidence reporting until a later classification feature: rejected because this feature already depends on evidence-based characterization.

## Decision: Protect nearby stations by keeping measurement and matching goals separate

**Rationale**: Nearby FM stations staying separate is a core regression risk. The measured occupied bandwidth may be station-scale, but the persistence match span still needs to remain tight enough to prevent accidental merges.

**Alternatives considered**:

- Expand match spans to nominal FM bandwidth globally: rejected because adjacent stations could collapse together.
- Use only center distance without overlap checks: rejected because stability and matching would become fragile.
- Accept occasional merges as a tradeoff for simpler code: rejected because it would undermine the stability work.

## Decision: Prefer diagnostics-first emission for new characterization evidence

**Rationale**: The repository already has bounded diagnostic JSONL and bundle exports, and the user explicitly asked to avoid schema migration unless it is necessary. Diagnostics-first evidence lets the team validate the new model before expanding persistent schema surface.

**Alternatives considered**:

- Add all proposed characterization fields to the database immediately: rejected because it increases migration pressure before the final data model is proven.
- Keep characterization entirely in memory with no export path: rejected because operators and maintainers would have no evidence to inspect.
- Delay characterization until a full persistent data model is agreed: rejected because it would slow validation and future classification planning.

## Decision: Keep optional FM-specific indicators explicitly optional

**Rationale**: Indicators such as a 19 kHz pilot or 57 kHz RDS or RBDS evidence could be useful supporting evidence, but their absence should not make normal FM characterization fail. The first implementation should remain useful even when those indicators are unavailable or unreliable.

**Alternatives considered**:

- Require pilot or RDS detection before calling something FM-like: rejected because it would be too strict for the first implementation.
- Exclude FM-specific indicators entirely: rejected because they are valuable when practical and fit the long-term classification goal.
- Present pilot or RDS indicators as proof on their own: rejected because they should remain supporting evidence inside a broader confidence model.

## Decision: Begin with deterministic fixtures that mirror the requested validation cases

**Rationale**: The requested test matrix already captures the main regression boundaries: stable bounded cards, separated nearby stations, revisit refinement, narrow non-FM protection, bandplan separation, diagnostics export, and persistence invariants. Those tests should guide implementation before any runtime changes land.

**Alternatives considered**:

- Depend only on hardware iteration: rejected because it is slower and less precise for regression boundaries.
- Test only diagnostics output: rejected because persistence and span behavior need direct fixture coverage.
- Focus only on FM-like success cases: rejected because the failure modes are just as important as the desired outcome.

## Open Research Follow-Ups

- Determine whether center and bandwidth stability across long runs can be derived cleanly from existing persistent state plus diagnostics summaries, or whether additive persistent fields are required.
- Evaluate whether optional FM-specific evidence is practical at current revisit FFT and averaging settings without harming scan cadence.
- Decide the minimal operator-facing UI exposure for measured characterization in the first implementation versus diagnostics-only evidence.
