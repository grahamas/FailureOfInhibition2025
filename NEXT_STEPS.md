# Failure of Inhibition 2025

## Goals

- Maintain a trustworthy Julia package for neural field modeling, Wilson-Cowan dynamics, spatial connectivity, stimulation, sensitivity analysis, and bifurcation workflows.
- Keep examples, tests, and documentation aligned with the current public API so research experiments can be rerun reproducibly.

## Next Steps

- [ ] Audit the README and example docs against the current API, especially sections called out as AI-assisted or likely to drift.
- [ ] Clarify and test lattice augmentation and embedding utilities, including coordinate splitting and random embedding behavior.
- [ ] Fix or document spatial metric TODOs so distance calculations use the intended L2 behavior.
- [ ] Resolve periodic lattice sampling TODOs by confirming the expected uniformity behavior and adding regression coverage.
- [ ] Expand integration coverage around nonlinearities and core simulation examples before changing model APIs.

## Context

This repo is a Julia reimplementation of the earlier Failure of Inhibition neural-field modeling work. The README describes a broad modeling surface, including Wilson-Cowan models, spatial connectivity, stimuli, simulation helpers, sensitivity analysis, bifurcation analysis, traveling-wave analysis, and optional GPU support.

## Constraints

- Preserve reproducibility for research examples and benchmark-style workflows.
- Treat large README sections cautiously when they are not directly backed by tests or examples.
- Prefer small API-preserving fixes before broad documentation or model-surface rewrites.
