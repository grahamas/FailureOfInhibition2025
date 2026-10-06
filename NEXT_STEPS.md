# Failure of Inhibition Point Model

## Goals

- Maintain a tested two-population point-model implementation for the failure-of-inhibition study.
- Produce deterministic CPU experiments and traceable figure data from documented model assumptions.
- Explain coexistence, controllable rest–active switching, and the measured tradeoffs between ordinary activity and control of high-E/low-I activity.
- Preserve the distinction between herald and seizure candidates, and between tonic-input withdrawal and direct activity displacement.

## Next Steps

- [ ] Review completion, unresolved outcomes, and any confirmation disagreements from the 74 detailed input-response follow-ups using `docs/input_response_results.md` and the batch status recorded there.
- [ ] Manually revise the manuscript using the figure sequence and paragraph purposes in `docs/narrative_results.md`; keep rest–active switching distinct from demonstrated cortical computation.
- [ ] Review the exploratory time-scale ratio and positive-I seizure-induction witness before giving the selected example physiological meaning.
- [ ] Select the input-response comparisons that best support the manuscript argument, keeping sustained withdrawal, finite pulses, and direct E displacement distinct.
- [ ] Specify a later coupled E–I feedback model and its available control signals before attempting self-terminating herald spikes or failed seizure control.

## Context

The equations remain documented in `docs/model.md`. The completed narrative
study screened 2,103 contexts and qualified seven examples for ordinary
switching plus seizure access. The selected positive-input four-state example
supports a paired herald/seizure displacement comparison and 44 intervention
settings. Raising the failure threshold preserves the measured switching but
sends the former seizure source to herald in that selected example. Subsequent
matched counterexamples show that this destination is parameter-dependent:
positive-input withdrawal can recover herald while seizure persists, and a
higher failure threshold can send seizure to intermediate activity.
See `docs/narrative_study.md` for the protocol and `docs/narrative_results.md`
for the original results, limitations, and manual manuscript revision
instructions. `docs/input_response_study.md` documents the systematic two-input
characterization and its sampling limits.
Its eight anchors and 224-case parameter screen are complete; 74 selected
detailed follow-ups continue as a resumable batch with automatic verification
and report generation. Completed results and batch status are documented in
`docs/input_response_results.md`.

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

- What biological evidence would justify interpreting the demonstrated rest–active switch as cortical function?
- Which parameter neighborhoods preserve selective herald recovery, and which observed differences are robust enough to motivate a coupled feedback model?
- What feedback dynamics could make a stable point-model herald candidate a transient spike in a coupled system?

## Completed

- [x] Implemented two-input equilibrium and response characterization, completed the anchor maps and parameter screen, and independently checked selective withdrawal, neighboring parameterizations, pulse boundaries, and archived-source replay.
- [x] Completed the narrative study, paired reduction measurements, intervention comparisons, local map, neighborhood checks, independent numerical checks, and archived-source trajectory replay; delivered figures and an author-facing outline.
- [x] Implemented and independently reviewed pseudo-arclength continuation, numerical periodic shooting, and explicit pulse experiments with retained unresolved outcomes and replayable evidence.
- [x] Executed both full primary parameter planes, representative continuation and pulse protocols, intervention/robustness searches, and periodic-candidate screening; documented analytical constraints and initial results.
- [x] Added and independently reviewed the sampled diagnostic contract and reproducible matched control/FoI runner, with explicit numerical criteria, full search records, source snapshots, checksums, and retained unresolved outcomes.
