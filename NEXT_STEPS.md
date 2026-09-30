# Failure of Inhibition Point Model

## Goals

- Maintain a tested two-population point-model implementation for the failure-of-inhibition study.
- Produce deterministic CPU experiments and traceable figure data from documented model assumptions.
- Explain coexistence, controllable rest–active switching, and the measured tradeoffs between ordinary activity and control of high-E/low-I activity.
- Preserve the distinction between herald and seizure candidates, and between tonic-input withdrawal and direct activity displacement.

## Next Steps

- [ ] Finish and review the resumed 74-case input-response batch; use the dated partial audit in `docs/paper_evidence.md` and retain confirmation differences.
- [ ] Review the contribution brief, four figure questions, and 24 calendar-sized deliverables in `docs/paper_delivery.md` and `docs/paper_evidence.md`.
- [ ] Choose the proposed coupling and response-parameter domains in `docs/paper_experiment_card.md` before launching the staged expansion.
- [ ] Connect ordinary switching, paired high-state control, and intervention tradeoffs across selected parameter regions, remeasuring effective stimuli where needed.
- [ ] Assemble the confirmed figures and manually revise the manuscript using the new candidate paragraphs; retain the derivation and defer fitting competition and autonomous feedback.

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
Its eight anchors and 224-case parameter screen are complete. A 30 September
UTC audit found nine of 74 detailed follow-ups complete and a tenth partial;
the stopped batch was resumed with its verified frozen source. Independent
replay also confirms 13 selective-withdrawal contexts, including herald-to-rest
contrasts at time-scale ratios about 1.143 and 4.4. The paper now targets staged
parameter-region evidence within the point model. See `docs/paper_evidence.md`
for the dated snapshot and `docs/input_response_results.md` for study details.

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

- Which parameter regions support ordinary switching, high-state access, and selective control together?
- Which proposed expansion domains should be selected for the next bounded batch?
- What biological evidence supports the final interpretation of the measured activity and switching properties?

## Completed

- [x] Prepared the 24-task paper handoff, contribution brief, figure candidates, and proposed experiment card; audited completed response artifacts and independently confirmed 13 selective-withdrawal contexts with 52 trajectories.
- [x] Implemented two-input equilibrium and response characterization, completed the anchor maps and parameter screen, and independently checked selective withdrawal, neighboring parameterizations, pulse boundaries, and archived-source replay.
- [x] Completed the narrative study, paired reduction measurements, intervention comparisons, local map, neighborhood checks, independent numerical checks, and archived-source trajectory replay; delivered figures and an author-facing outline.
- [x] Implemented and independently reviewed pseudo-arclength continuation, numerical periodic shooting, and explicit pulse experiments with retained unresolved outcomes and replayable evidence.
- [x] Executed both full primary parameter planes, representative continuation and pulse protocols, intervention/robustness searches, and periodic-candidate screening; documented analytical constraints and initial results.
