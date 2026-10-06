# Failure of Inhibition Point Model

## Goals

- Maintain a tested two-population point-model implementation for the failure-of-inhibition study.
- Produce deterministic CPU experiments and traceable figure data from documented model assumptions.
- Explain coexistence, controllable rest–active switching, and the measured tradeoffs between ordinary activity and control of high-E/low-I activity.
- Preserve the distinction between herald and seizure candidates, and between tonic-input withdrawal and direct activity displacement.

## Next Steps

- [ ] Review the five figure claims and candidate captions in `docs/figure_redesign_20261006.md`, then choose the final manuscript presentation and biological wording.
- [ ] If the manuscript needs a cycle-onset claim, extend and independently test the truncated periodic branch before identifying a bifurcation; retain unresolved input cells as sampling limits.
- [ ] Revise the manuscript manually using the reviewed point-model results, keeping the derivation and deferring physiological fitting and autonomous feedback.
- [ ] Before reusing the historical 74-case response batch, reconcile its affected geometry selections and responses under the current exact-domain handoff policy. Keep it separate from the refreshed 232-case screen and eight-case figure study.

## Context

The approved figure redesign uses one reference circuit plus contrasts within
the existing coupling, threshold, input, and time-scale envelope. The refreshed
screen covers 232 parameterizations and 44,643 input contexts. Detailed work
covers eight circuits at fourteen baselines, including one attracting periodic
source tested at sixteen phases. All 99 independent geometry confirmation
searches agree with their corresponding discovered root sets.

The connected reference simulation visits rest, active, herald, and seizure
without resets. Paired withdrawal measurements distinguish retention of both
high states, joint recovery, and selective herald recovery across contexts.
Eighty-four intervention settings measure the original high-state destination
alongside ordinary switching with original or remeasured pulses. The independent
equation check passed 176 trajectory comparisons. These results support a
bounded point-model account; they do not certify complete attractor discovery
or identify the birth of the truncated periodic branch.

See `docs/figure_redesign_20261006.md` for the scientific questions, measurements,
limits, and captions, and `reproducibility/figure_redesign_20261006/README.md`
for the portable packet and replay commands. Earlier narrative and full-atlas
results remain historical evidence with their original provenance; the new
study does not certify every older response protocol.

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

- Which bounded model results should be emphasized in the manuscript, and which remain supporting evidence?
- Does the manuscript require resolving the periodic branch's onset, or is the measured finite branch sufficient?
- What biological evidence supports the final interpretation of the measured activity and switching properties?

## Completed

- [x] Delivered five rendered and visually reviewed figure drafts from the refreshed 232-case screen, eight targeted circuits, a connected switching sequence, sixteen cycle phases, paired controls, and 84 intervention settings; independently checked 176 trajectories.
- [x] Completed and archived the historical 74-case response batch, audited corrected regional selections, and retained confirmation differences and source provenance.
- [x] Prepared the original paper handoff, contribution brief, figure candidates, and experiment card with independent checks of the dated selective-withdrawal witnesses.
- [x] Implemented two-input equilibrium and response characterization, ordinary switching and paired-control protocols, and archive validation.
- [x] Implemented and independently reviewed pseudo-arclength continuation, numerical periodic shooting, and explicit pulse experiments with retained unresolved outcomes and replayable evidence.
