# Failure of Inhibition Point Model

## Goals

- Maintain a tested two-population model and a reproducible path from new figure measurements to rendered figures.
- Explain the model's measured coexistence, switching, and intervention tradeoffs without extending bounded numerical observations into completeness or biological claims.

## Next Steps

- [ ] Review the five figure claims and candidate captions in `docs/figure_redesign_20261006.md`; choose the final manuscript presentation and biological wording.
- [ ] If the paper needs a cycle-onset claim, extend and independently test the truncated periodic branch before identifying a bifurcation.
- [ ] Revise the manuscript manually using the selected results, keeping its scientific choices and wording under author control.
- [ ] After final result selection, freeze the corresponding data, code, environment, and replay instructions as the active reproducibility packet.

## Context

The five-figure candidate used a 232-case screen, eight detailed circuits, and
independent trajectory checks. It is bounded point-model evidence. The candidate
packet is not shipped; current code and inputs support a new run
using `reproducibility/figure_redesign_20261006/README.md`. The periodic branch's
onset is unresolved. Final results have not been selected.

## Constraints

- Keep `docs/model.md`, implementation, and tests synchronized when model behavior changes.
- Preserve the approved model and the recorded five-figure protocol until an author-approved scientific change is specified.
- Keep discovered equilibria, finite-window destinations, validated periodic orbits, and biological interpretation distinct. Numerical searches do not certify completeness.
- Keep manuscript edits manual.

## Open Questions

- Which bounded model results belong in the final manuscript and supplement?
- Does the paper need the periodic branch's onset identified?
- What biological evidence supports the final interpretation?

## Completed

- [x] CERT-01–07: Implemented fixed-point completeness API, exact artifacts and proof replay; numerical/IO fixtures, full root suite (3,603/3,603), plotting (11/11), and independent review passed. Bounded fixture benchmarks only; no study or batch. See [certification contract](docs/fixed_point_certification.md).

- [x] Built and visually reviewed the five-figure candidate study; retired its generated packet from the active tree.
