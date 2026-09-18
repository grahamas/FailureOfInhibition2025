# Failure of Inhibition Point Model

## Goals

- Maintain a tested two-population point-model implementation for the failure-of-inhibition study.
- Produce deterministic CPU experiments and machine-readable data from documented model assumptions.

## Next Steps

- [ ] Define the dynamical regimes, classification criteria, and diagnostics required for the study's computational claims.
- [ ] Add continuation and periodic-orbit diagnostics that distinguish autonomous invariant objects from transients and externally driven states.
- [ ] Define deterministic experiment configurations and stable machine-readable figure-data schemas.

## Context

The current equations, drive semantics, numerical equilibrium discovery, and local linear stability policies are documented in `docs/model.md` and covered by the integrated test suite. Equilibrium searches operate on autonomous or explicitly frozen systems and do not certify completeness. Continuation, periodic-orbit and bifurcation analysis, regime diagnostics, experiments, plotting, and manuscript claims are later phases.

## Constraints

- Keep `docs/model.md`, the implementation, and tests synchronized when model behavior changes.
- Keep the excitatory mechanism and every non-inhibitory parameter identical in matched model comparisons.
- Preserve `[E, I]`, `source_to_target` coupling names, and CSV column order `time,E,I`.
- Require deterministic configurations, explicit tolerances, and machine-readable outputs for reported experiments.
- Treat pulsed-drive snapshots as frozen autonomous systems and report numerical searches as incomplete discovery rather than completeness certification.
- Keep manuscript prose and plotting outside this repository.

## Open Questions

- Which parameter regimes, operational thresholds, and diagnostics are required for the study's computational claims?

## Completed

- [x] Added and independently reviewed local equilibrium solving, deterministic multistart discovery, candidate validation and deduplication, and local linear stability diagnostics.
