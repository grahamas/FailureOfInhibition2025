# Failure of Inhibition Point Model

## Goals

- Maintain a tested two-population point-model implementation for the failure-of-inhibition study.
- Produce deterministic CPU experiments and traceable figure data from documented model assumptions.
- Test whether raising the inhibitory failure threshold preserves distinct active dynamics while suppressing a candidate pathological attractor more selectively than alternative interventions.

## Next Steps

- [ ] Refine coexistence boundaries in the two `e_to_i=12:0.5:28` by `theta_off=6:0.25:12` planes with matched monotone controls; track stable and unstable branches, retaining search failures and incomplete coverage.
- [ ] Establish what retained activity represents at representative coexistence points using E, I, inhibitory input and response slope. Validate any proposed periodic orbit with phase, closure, refinement, and Floquet checks; keep biological roles unresolved until justified.
- [ ] Compare individual changes to failure threshold, E-to-I recruitment, E-to-E excitation, and I-to-E output using continued activity branches and pulse-transition intervals. Distinguish selective suppression from loss of all nonquiescent activity.
- [ ] Refine positive E/I/equal-input and separate negative-E amplitude-duration maps from each attracting equilibrium and validated cycle phases. Extend unresolved post-withdrawal traces beyond 5 seconds as needed; report the tested intervention class and finite horizons.
- [ ] Test promising regions with the targeted slices and deterministic joint samples in `docs/intervention_study.md`, including time-constant ratio and numerical sensitivity; check candidate bifurcation nondegeneracy before stronger claims.
- [ ] Reconcile manuscript claims with the arbitrary-coupling monotone ordering result, equal-slope response symmetry, and experimental evidence. Return revision instructions while keeping manuscript changes manual.

## Context

The current equations and numerical policies are documented in `docs/model.md`.
The scientific sequence, supplied exploration bounds, evidence gates, and pulse
protocols are in `docs/intervention_study.md`. These bounds are exploratory,
not empirically calibrated. The minimal matched-model experiment remains a
separate synthetic workflow check. `docs/intervention_results.md` records the
executed primary map, representative continuation and pulse experiments,
targeted robustness, and finite periodic-candidate screen. The high-E/high-I
branch persists when the low-I branch turns under increased failure threshold;
functional activity and exhaustive attractor discovery remain unresolved.

## Constraints

- Keep `docs/model.md`, the implementation, and tests synchronized when model behavior changes.
- Preserve the approved model; new scientific parameter choices and biological interpretations remain author inputs.
- Keep the excitatory mechanism and every non-inhibitory parameter identical in matched model comparisons.
- Preserve `[E, I]`, `source_to_target` coupling names, and CSV column order `time,E,I`.
- Require deterministic configurations, explicit tolerances, and machine-readable outputs for reported experiments.
- Treat pulsed-drive snapshots as frozen autonomous systems and report numerical searches as incomplete discovery rather than completeness certification.
- Separate discovered equilibria and local spectra, finite-window trajectory observations, validated periodic orbits, and biological interpretations. Unsupported classifications remain unresolved.
- Describe recovery only over its recorded post-intervention horizon; do not infer permanent rescue from finite follow-up.
- Keep implementation work in this repository and manuscript edits manual. Use sub-agents with separate file ownership and independent review for execution.

## Open Questions

- Which discovered nonquiescent object has evidence for the manuscript's intended functional active regime? Three attracting equilibria alone do not answer this.
- Which biological observations can distinguish pathological activity from high excitation with substantial inhibition, including when both lie on the descending response branch?
- Do the numerical candidate bifurcations and periodic orbits survive refinement, and what remains undiscovered?

## Completed

- [x] Implemented and independently reviewed pseudo-arclength continuation, numerical periodic shooting, and explicit pulse experiments with retained unresolved outcomes and replayable evidence.
- [x] Executed both full primary parameter planes, representative continuation and pulse protocols, intervention/robustness searches, and periodic-candidate screening; documented analytical constraints and initial results.
- [x] Added and independently reviewed the sampled diagnostic contract and reproducible matched control/FoI runner, with explicit numerical criteria, full search records, source snapshots, checksums, and retained unresolved outcomes.
- [x] Repaired mixed-precision RHS/Jacobian scaling with independent review and underflow, overflow, and mixed-type regressions.
- [x] Added and independently reviewed local equilibrium solving, deterministic multistart discovery, candidate validation and deduplication, and local linear stability diagnostics.
