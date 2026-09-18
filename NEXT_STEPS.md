# Failure of Inhibition Point Model

## Goals

- Maintain an authoritative, testable two-population point-model implementation for the failure-of-inhibition study.
- Produce deterministic CPU experiments and machine-readable data from an explicitly approved mathematical contract.

## Next Steps

- [ ] Write and approve the complete mathematical contract, including state domain, parameter conventions, external input, and admissible response domains.
- [ ] Choose the inhibitory-response formula, normalization, nonnegativity rule, and limiting behavior from explicit candidate comparisons.
- [ ] Add matched monotone-inhibition and failure-of-inhibition constructors that differ only in the approved inhibitory mechanism.
- [ ] Implement and verify equilibrium finding, analytical Jacobians, stability classification, and relevant dynamical regimes.
- [ ] Add numerical diagnostics that distinguish equilibria, limit cycles, transients, and externally driven states.
- [ ] Define deterministic experiment configurations and stable machine-readable figure-data schemas.

## Context

The repository now contains only a typed, CPU-only two-population point-model core. The surviving right-hand side preserves earlier algebra provisionally; it is not yet the paper's approved mathematical contract. Historical spatial, GPU, optimization, sensitivity, and traveling-wave implementations were removed rather than treated as validated scientific infrastructure.

## Constraints

- Do not silently select or normalize an inhibitory response.
- Keep the excitatory mechanism and every non-inhibitory parameter identical in matched model comparisons.
- Require deterministic configurations, explicit tolerances, and machine-readable outputs for reported experiments.
- Keep manuscript prose and plotting outside this repository.

## Open Questions

- Which inhibitory candidate, domain, normalization, and limiting behavior define the failure-of-inhibition model?
- Which parameter regimes and diagnostics are required for the paper's claims?

## Completed

- [x] Narrowed the package to a typed two-population point-model core and removed unrelated computational surfaces.
