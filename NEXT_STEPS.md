# Failure of Inhibition Point Model

## Goals

- Maintain a tested two-population point-model implementation for the failure-of-inhibition study.
- Produce deterministic CPU experiments and traceable figure data from documented model assumptions.
- Connect computational claims to validated mathematical objects and clearly delimited biological interpretations.

## Next Steps

- [ ] Specify author-selected scientific parameter sets, continuation axes/ranges, and operational biological definitions; keep the synthetic workflow example separate from scientific evidence.
- [ ] Continue equilibrium branches in both directions, including unstable branches. Refine folds and Hopf candidates, check crossing/nondegeneracy conditions, and retain residuals, spectra, coverage limits, and unresolved outcomes.
- [ ] Compute periodic orbits with a phase condition; validate nonconstant amplitude, period, closure/equation residuals, refinement convergence, and Floquet stability before reporting cycles or their stability.
- [ ] Specify perturbation and intervention protocols with starting attractor, E/I pulse direction, amplitude, duration, onset/phase, withdrawal, and follow-up. Test coexistence and transitions under common autonomous parameters; produce amplitude-duration maps with controls, post-withdrawal recovery, and unresolved cells.
- [ ] Assess numerical and author-defined model-parameter sensitivity. Generate reproducible figure data and map each manuscript claim to validated results, configuration/code provenance, and limitations.
- [ ] In parallel, reconcile existing theory and manuscript claims with the approved equations and emerging evidence. Return evidence and author revision instructions while keeping historical implementation and manuscript repositories read-only.

## Context

The current equations and numerical policies are documented in `docs/model.md`. Equilibrium discovery, local linear stability, sampled trajectory diagnostics, and the minimal matched-model experiment are implemented. The synthetic experiment exercises baseline and pulsed conditions with standard, refined, and extended-horizon settings; it does not establish biological regimes. Continuation, validated periodic orbits, scientific protocols, and publication claims are subsequent gates.

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

- Which scientific parameter sets, continuation axes/ranges, and biological criteria should the author supply for the first scientific comparison?
- Which perturbation/intervention directions, onset phases, amplitude-duration ranges, and recovery horizons support the intended claims?

## Completed

- [x] Added and independently reviewed the sampled diagnostic contract and reproducible matched control/FoI runner, with explicit numerical criteria, full search records, source snapshots, checksums, and retained unresolved outcomes.
- [x] Repaired mixed-precision RHS/Jacobian scaling with independent review and underflow, overflow, and mixed-type regressions.
- [x] Added and independently reviewed local equilibrium solving, deterministic multistart discovery, candidate validation and deduplication, and local linear stability diagnostics.
