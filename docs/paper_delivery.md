# Point-model paper: deliverables and calendar blocks

The endpoint is a paper explaining how failure of inhibition changes the activity
a circuit can sustain, how its states can be reached and controlled, and which
interventions preserve ordinary switching. The path goes from successful
examples to mapped parameter regions and explanations of their boundaries.

The author selected a point-model paper, staged parameter expansion, retention
of the population-response derivation without a synthetic-fitting competition,
and 30–60 minute calendar blocks. Feedback is a separate follow-on project.

## How to use these tasks

Each numbered row is one TickTick task. Its duration is author decision/review
time, not agent effort or numerical runtime. Agent preparation and unattended
execution are separate subtasks. Dependencies identify when a task can finish;
preliminary drafting can start earlier. No dates or external TickTick writes
are assigned here.

Use [the evidence handoff](paper_evidence.md) for current results and figure
material, and [the experiment card](paper_experiment_card.md) for the next
scientific choices. `paper_tasks.toml` provides the same task IDs and dependency
graph for later translation; it is not a claim of native TickTick import support.

| ID | Task | Minutes | Depends on | Deliverable / done when |
| --- | --- | ---: | --- | --- |
| 01 | Reconcile the historical input-response batch | 45 | — | All 74 frozen follow-ups are accounted for; corrected selections, affected screens and responses, and newly selected detailed cases are checked before figure use. |
| 02 | Define the contribution relative to prior work | 45 | — | Review the sourced comparison and select the paper's contribution and biological framing. |
| 03 | Approve the argument and four figure questions | 45 | 01, 02 | One sentence per main result identifies its evidence and the observation still needed. |
| 04 | Approve the next experiment card | 30 | 03 | Choose concrete parameter domains, measurements, batch budget, and stopping rule. |
| 05 | Map ordinary switching alongside seizure access | 45 | 04 | A sampled region supports actual switching and recorded induction routes, with matched monotone controls. |
| 06 | Map the herald–seizure control contrast | 45 | 04 | Paired same-context sustained/pulsed input maps show amplitude-duration dependence; direct displacement remains separate. |
| 07 | Measure intervention benefits and activity costs | 60 | 05, 06 | Parameter dose-response curves report high-state behavior and ordinary switching together. |
| 08 | Screen beyond the two fixed coupling families | 30 | 04 | Review a bounded coupling screen, its candidate list, and contrasting cases. |
| 09 | Screen response and time-scale sensitivity | 30 | 04 | Review supported response/time-scale variations, including the manuscript ratio. |
| 10 | Retest the strongest regions and counterexamples | 45 | 05, 06, 07, 08, 09 | Review detailed tests at selected and fresh parameter points, with fixed and remeasured stimuli separated. |
| 11 | Explain the important transitions | 60 | 10 | Continuation and phase-plane diagrams explain the selected boundaries and state/control changes. |
| 12 | Independently confirm the headline results | 45 | 11 | Headline trajectories and boundaries survive numerical refinement, independent equations, and matched controls. |
| 13 | Finish Figure 1: mechanism and ordering | 45 | 03 | Review the response illustration, matched models, ordering argument, caption, and Results paragraph. |
| 14 | Finish Figure 2: ordinary switching and additional states | 60 | 12 | Review actual switching/induction trajectories, shared parameter labels, caption, and Results paragraph. |
| 15 | Finish Figure 3: different states, different control | 60 | 12 | Review paired input-control curves, duration dependence, regional support, caption, and Results paragraph. |
| 16 | Finish Figure 4: intervention tradeoffs | 60 | 12 | Review intervention curves alongside switching preservation, caption, and Results paragraph. |
| 17 | Assemble the robustness supplement | 45 | 12 | Parameter coverage, selection reasons, contrasting regimes, confirmation, and additional maps support the figures. |
| 18 | Revise the Introduction | 60 | 02, 03 | Manually revise the biological motivation, unresolved question, and contribution. |
| 19 | Revise the model and theory Methods | 60 | 13 | Manually align equations, derivation, conventions, and theorem assumptions with implementation. |
| 20 | Revise the experimental Methods | 60 | 17 | Manually describe domains, selection, interventions, outcomes, numerical checks, and replay. |
| 21 | Revise the Discussion | 60 | 13, 14, 15, 16, 18 | Manually develop the mechanism, parameter dependence, implications, and testable predictions. |
| 22 | Finalize the title and abstract | 45 | 19, 20, 21 | Every result sentence maps to a final figure or analytical result. |
| 23 | Run an independent reviewer critique | 60 | 17, 22 | Resolve a short issue list covering novelty, alternatives, selection, and claim-to-figure support. |
| 24 | Freeze the submission package | 30 | 23 | Author-approved manuscript, figures, supplement, and reproducibility manifest agree. |

The first pass allocates 19.5 hours of author blocks. This is a scheduling
estimate, not a total project-duration forecast. Split longer writing/review
tasks into additional paragraph or figure-panel blocks as needed.

## Current handoff state

Task 01 has a verified historical snapshot and a completed frozen batch. The
6 October selection audit requires corrected selection, screening, and response
reruns before figure selection; its final review remains pending. Task 02 has
a sourced contribution brief. Task 03 has an argument and figure handoff for
author review. Task 04 has a proposed experiment card awaiting domain choices.
Tasks 05–07 and 13–17 already have reusable evidence/figure material; they are
not marked complete before expanded results and final selection are reviewed.
Manuscript revisions, author approvals, and submission remain pending.

## Agent execution contract

Agents prepare archive summaries, literature evidence, figures, bounded code
changes, and candidate prose. The author chooses scientific domains, biological
interpretation, final narrative, and manuscript wording. The manuscript stays
read-only to agents; candidate text is retained in this repository for manual use.

Implementation chunks belong in a container-level plan, have an enforced
file allowlist and executable verification, and run through bubblewrapped
`chunkrun`. Codex reviews results and the full project suite. A worker's done
marker does not establish scientific acceptance. Do not duplicate a running
study, rewrite archived provenance, or promote an unresolved observation.

This handoff's task-index chunk is retained in container-level `PAPER_PLAN.md`
and selected with `chunkrun --plan`; the historical `PLAN.md` is preserved.
The worker's candidate was independently checked against all 24 table rows,
then its status fields and author-ownership wording were corrected centrally.
The review record is `output/paper_handoff_20260930/agent_review.json`.

## Evidence and stopping criteria

Begin with existing data. Use cheap geometry/stability screens before detailed
pulse maps, and expand fixed parameter choices in stages. Record whether an
apparent restriction comes from state availability, dynamics, available input,
or the particular tested pulse. Remeasure stimuli when testing capability.

Each main claim needs a confirmed example, a characterized neighborhood, and
an explanation of its important boundaries. Keep useful counterexamples visible.
Additional runs answer named questions; finite negative searches leave broader
possibilities open. No model API change or new biological label is implied.
