# Paper evidence and author handoff

Prepared 30 September 2026 UTC from the existing model, retained studies, and
a dated audit. This is supporting material for manual manuscript revision.
It adds candidate text and figure instructions without replacing prior drafts.
The [task list](paper_delivery.md) separates preparation from author acceptance.

## Argument

Failure of inhibition changes the state repertoire of an activity-supporting
circuit and changes how its states respond to control. Ordinary switching,
entry into high activity, and recovery are distinct measurable properties.
The useful intervention question is which parameter changes improve high-state
control while retaining ordinary switching.

The scientific opportunity is to show where these properties occur together
and why their boundaries move. The newer selective-withdrawal and
threshold-to-active examples already show that conclusions drawn from the first
selected parameter set need not hold elsewhere.

| Result question | Existing support | Next discriminating test |
| --- | --- | --- |
| What does the failure response make possible? | Fire-then-fail construction and the general same-input equilibrium-ordering proof in `theory_notes.md`. | Check the exposition against the implemented equal-slope response and pair it with matched phase planes. |
| Can ordinary switching coexist with high-state access? | The narrative study qualifies seven cases; its selected example repeats rest–active switching and reaches both high-state roles. | Map neighborhoods while remeasuring effective pulses, then compare matched monotone controls. |
| Can available input distinguish herald from seizure? | At `(17,13,19,6)`, baseline E input 0.35, finite E withdrawal recovers herald to active while seizure persists in the tested pulse map. | Connect successful contexts into sampled regions; compare baselines and duration dependence. |
| Can a parameter intervention preserve ordinary switching? | At recurrent excitation 19, raising failure threshold redirects the seizure source to herald while retaining measured switching. A separate recurrent-excitation-16 example redirects seizure to active. | Test switching and recovery together at the latter setting and across neighborhoods. |

The last row deliberately retains two parameterizations. Recovery to active
from one example and preserved switching from another cannot be combined into
a single demonstrated selective-treatment result.

## Current audit and useful new leads

The batch's earlier `running` marker was stale: a host process inspection found
no Julia process, and its log last advanced at 04:15 UTC. The existing frozen
study was resumed at 05:11 UTC after its configuration and all 27 source hashes
were verified. This resume retains the original 74 selected cases and domains.

The snapshot in `output/paper_handoff_20260930/` contains:

- Eight complete anchors covering 91 response baselines, all 224 parameter
  screen cases, and nine fully characterized selected follow-ups. A tenth has
  17 of 22 response baselines complete. The other 64 selected cases have yet to
  supply detailed results in this snapshot.
- 243 completed response baselines, 672,537 sustained-input observations,
  and 1,920,111 pulse observations. These are retained observations, not
  independent experimental replicates or estimates of parameter prevalence.
- 248 independent root-search comparisons with no recorded disagreement,
  and no held-input control mismatch among the completed response baselines.
- 37 tighter-confirmation differences: 36 previously unresolved observations
  resolve to active and one resolves to an oscillatory destination. None is
  a disagreement between two previously resolved destination labels. Preserve
  both records and use the confirmed result when explaining that witness.

`audit.json` verifies 38,458 files through 503 checkpoints: all completed response
records, independent geometry confirmations, and the geometry summaries read by
the handoff. Compressed raw root-search contents and unfinished responses are
outside this partial audit; the existing batch's final verifier handles them.
The snapshot summary is hash-bound in `audit.json` and will not be overwritten
by the resumed batch's own final report.

The 224-case screen has 13 distinct parameter/baseline contexts where complete
E withdrawal recovers herald while seizure persists: 11 to active and two to
rest. The two rest destinations occur at time-scale ratios about 1.143 and 4.4.
These are especially useful follow-up targets because the control contrast is
not confined in the screen to the original ratio 0.2. Case lists and parameters
are in `selective_screen_contexts.json`. Independent 41×41 SciPy root searches
and 52 DOP853 trajectories confirm all 13 paired contexts, including held-input
controls, at a 10,000 ms horizon and `1e-12` integration tolerances. The two
higher-ratio examples therefore supply confirmed witnesses beyond the original
fast-inhibition setting. `selective_confirmation.json` records coordinates,
spectra, errors, and source hashes. Induction and neighborhood extent remain
separate measurements.

The [three-context figure and replay report](../output/paper_handoff_20260930/confirmed_withdrawal.html)
show independently integrated trajectories at ratios 0.2, about 1.143, and 4.4.
PNG and SVG versions and all twelve plotted withdrawal/control trajectories
are retained with the report. The PNG was visually inspected for panel, legend,
and label clarity. It is a supporting figure draft, not the final regional map.

Fifteen completed paired direct-displacement measurements have recorded
seizure/herald ratios from about 1.32 to 5.37. This variation motivates plotting
the measurement against context, rather than choosing one ratio as a universal
control requirement. State displacement and input withdrawal keep separate axes.

## Contribution and biological positioning

The proposed contribution is the combination of a general ordering restriction,
ordinary switching alongside additional states, and measured control/intervention
tradeoffs across parameter contexts. This is an argument to assess against prior
work, not a claim that nonmonotone inhibition itself is new.

| Primary source | Relevant result | Positioning for this paper |
| --- | --- | --- |
| [Kim and Nykamp, 2017](https://doi.org/10.1007/s10827-017-0647-7), especially Figures 3–6 | A nonmonotone inhibitory Wilson–Cowan response supports physiological/seizure coexistence, input-driven transitions, hysteresis, and threshold-dependent bifurcation changes. | Cite this as the closest dynamical predecessor. Explain the added questions about ordinary switching and paired control, rather than claiming the first coexistence or threshold intervention. |
| [Tryba et al., 2019](https://doi.org/10.1152/jn.00392.2019), especially Figures 4–6 | Cellular experiments, simulations, and human recordings support a role for inhibitory firing failure near an ictal wave, alongside maintained inhibition in oscillatory territories. | Motivate a local failure response and region-specific interpretation. These observations do not set this model's numerical time-scale ratio or dimensionless couplings. |
| [Liou et al., 2020](https://doi.org/10.7554/eLife.50927) | Exhaustible inhibition, adaptation, spatial coupling, and feedback reproduce seizure evolution and motivate herald/spatial predictions. | Explain how the present point model isolates state accessibility and control. Reserve autonomous termination and propagation for the explicitly deferred extension. |
| [Agopyan-Miu et al., 2023](https://doi.org/10.1093/brain/awad262) | Human limbic recordings show region- and recruitment-dependent cell-type dynamics, including widespread firing reduction. | Specify the phenomenon and tissue context the model is intended to explain; do not use a local high-E/low-I example as a definition of every focal seizure. |

Sources were checked through primary-paper text/PDF or indexed primary-paper
text on 30 September 2026 UTC. This targeted comparison is not an exhaustive
novelty review. A key positive prediction to develop is that changing available
tonic excitation and intervention duration changes the contrast between the two
high-state responses, even before adding an autonomous feedback mechanism.

## Four figure handoffs and candidate Results paragraphs

The paragraphs below are new candidate text for author revision. They do not
replace any existing manuscript text. Final panels and numerical claims must
follow the selected, confirmed evidence.

### Figure 1: mechanism and ordering

Use the ordered onset/failure construction, the implemented response, matched
control/FoI phase planes, and a short statement of the ordering proof. Existing
material: `output/narrative_figures_20260928/01_mechanism.pdf`; analytical source:
`docs/theory_notes.md`. Keep the excitatory mechanism and all non-inhibitory
parameters matched. The existing illustration needs the theorem panel before
it is a final paper figure.

Candidate paragraph: Adding an ordered failure threshold to a firing population
produces a response that rises and then falls with input. This changes the
equilibrium arrangements available to the circuit. With nondecreasing inhibition,
two equilibria under common parameters and input cannot have higher excitatory
activity but lower inhibitory activity at the second state. The failure response
permits this ordering, providing a direct connection between the response
mechanism and the additional high-E/low-I state.

### Figure 2: switching and entry into high activity

Use the selected narrative example and actual concatenated switching trajectories.
Existing material: `output/narrative_figures_20260928/02_transitions.pdf`.
Show pulse targets and amplitudes explicitly, including its I-input seizure
induction. Add a neighborhood panel with fixed and remeasured protocols separated.

Candidate paragraph: The model supports a circuit that can repeatedly switch
between rest and intermediate activity while retaining access to two distinct
high-activity states. In the selected example, input pulses repeat the
rest–active–rest sequence using the actual state reached after each transition.
Other pulses reach the high-E/high-I and high-E/low-I states. Thus the additional
states coexist with an operational switching behavior that can be measured
before and after intervention.

### Figure 3: state-dependent input control

Use the selective-withdrawal anchor at `(17,13,19,6)`, failure threshold 8,
ratio 0.2, and baseline `(0.35,0)`. Existing source:
`output/input_response_20260929_v3/anchors/selective_withdrawal/`;
its reviewed preview is in `output/input_response_anchor_report_20260929/`.
Pair finite withdrawal trajectories and duration-dependent outcome brackets;
place sustained withdrawal and direct displacement in separately labeled panels.
Add the confirmed higher-ratio contexts as contrasts, not as interpolated regions.

Candidate paragraph: Available excitatory-input withdrawal can distinguish the
two high-activity states. At a shared positive baseline, finite withdrawal pulses
return the herald candidate to intermediate activity after the input is restored,
while the tested seizure-source pulses return to seizure. The withdrawal needed
for a successful herald transition decreases with longer pulse duration in the
selected brackets. By comparison, additional inhibitory-population input can
drive herald into seizure. Control therefore depends on the source state, the
input direction, and the duration of the intervention.

### Figure 4: intervention and ordinary activity

Start with `output/narrative_figures_20260928/04_interventions.pdf` and the
separately validated threshold-to-active anchor. Show parameter trajectories,
destinations, and switching measurements together. Final assembly needs the
joint ordinary-switching check for the threshold-to-active case and selected
regional confirmation; do not label the current drafts as final.

Candidate paragraph: Changing the inhibitory failure threshold alters the
destination of high-E/low-I activity. In the narrative example, raising the
threshold redirects the former seizure source to herald while preserving the
measured rest–active switch. At a different recurrent-excitation setting, a
threshold increase instead returns the seizure source to intermediate activity.
These outcomes motivate measuring intervention effects on ordinary switching
alongside their effects on the high-activity states across parameter contexts.

## Methods and review handoff

The model/theory Methods should follow `docs/model.md` and `docs/theory_notes.md`.
The experimental Methods should combine `docs/narrative_study.md` and
`docs/input_response_study.md`, with one table of parameter domains, protocols,
source selection, outcome criteria, and independent checks. Report stimulus
amplitudes and durations separately; no physiological cost conversion is selected.

Before freezing the paper, review three issues across the complete argument:
whether the claimed advance is distinguishable from the closest predecessor;
whether ordinary switching and improved control were measured in the same
context; and whether the observed parameter regions support the wording chosen.
These questions determine further experiments. They are not reasons to treat
an incompletely explored parameter space as a settled model limitation.

## Handoff verification

- Full Julia package suite: 4,809/4,809 assertions passed in this handoff run;
  existing iteration-limit failure-path tests emitted their expected warnings.
- Archive-storage suite: 9/9 tests passed.
- Independent confirmations: all 13 contexts and 52 withdrawal/control
  trajectories passed; maximum endpoint coordinate difference was below
  `1.6e-12`. Twelve figure trajectories were independently reintegrated and
  checked against those endpoints.
- All 24 task titles, durations, dependencies, and completion criteria match
  between the human table and TOML; the graph is acyclic and totals 1,170 author
  minutes. Local document/report links and planning-document structure passed.
- The model and public APIs were not changed. This handoff does not rerun the
  separate certification or plotting-package suites; the new Python figure was
  rendered, numerically checked, and visually inspected directly.

Logs, source identities, the delegation review, and artifact hashes are retained
in `output/paper_handoff_20260930/`. The ongoing study and author-owned paper
tasks are explicitly separate from these completed checks.
