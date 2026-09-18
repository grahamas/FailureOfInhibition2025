# Failure of Inhibition Point Model

## Goals

- Maintain a tested two-population point-model implementation for the failure-of-inhibition study.
- Produce deterministic CPU experiments and machine-readable data from documented model assumptions.

## Next Steps

- [ ] Define the dynamical regimes, classification criteria, and diagnostics required for the study's computational claims.
- [ ] Implement and verify equilibrium finding, physical-admissibility checks, and stability classification using the analytical Jacobian.
- [ ] Add numerical diagnostics that distinguish equilibria, autonomous limit cycles, transients, and externally driven states.
- [ ] Define deterministic experiment configurations and stable machine-readable figure-data schemas.

## Context

The current equations, assumptions, response functions, drive semantics, and numerical-domain handling are documented in `docs/model.md` and covered by the integrated test suite. Equilibria, bifurcations, regime diagnostics, experiments, plotting, and manuscript claims are later phases.

## Constraints

- Keep `docs/model.md`, the implementation, and tests synchronized when model behavior changes.
- Keep the excitatory mechanism and every non-inhibitory parameter identical in matched model comparisons.
- Preserve `[E, I]`, `source_to_target` coupling names, and CSV column order `time,E,I`.
- Require deterministic configurations, explicit tolerances, and machine-readable outputs for reported experiments.
- Keep manuscript prose and plotting outside this repository.

## Open Questions

- Which parameter regimes, operational thresholds, and diagnostics are required for the study's computational claims?
