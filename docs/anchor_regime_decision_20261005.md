# Paper anchors and sampled regimes — decision package, 5 October 2026

**6 October reconciliation status:** The selective-anchor switching, induction,
and threshold replay was repeated after the merged phase-handoff correction;
its destination labels and repeated-switching results held. The tonic-E scan
also retained its sampled brackets and destinations on the merged source. The
completed two-input batch and this packet's regime and ratio summaries use the
historical region-selection rule. The [selection audit](input_response_selection_audit_20261005.md)
found changed representative sets under the corrected rule, so these summaries
remain dated evidence until affected selection, responses, and aggregates are
rerun. The narrative switching witnesses also predate the phase-handoff fix.
The author has not selected figure anchors.

## Decision in front of the author

The [focused Julia result](selective_anchor_joint_exploration_20261005.md)
makes the selective-withdrawal coupling and tonic input a measured candidate
for a **shared Figures 2–4 example**: repeated ordinary switching, separate
induction of both high-state roles, historical paired herald/seizure input control, and a
threshold change that redirects the original seizure source while switching
remains demonstrable. The historical narrative setting supports switching,
induction, and threshold redirection, so separate narrative and selective
examples remain an option. The author must choose the figure anchors; neither
option is an approved biological interpretation or final manuscript selection.
The threshold-to-active and higher-time-scale cases remain contrasts.

The completed two-input study ended on 1 October, after the dated partial
paper handoff. Its final controller records 8 anchors, 224 screened parameter
cases, 74 detailed follow-ups, and 1,171 completed response baselines. The
final summary records no independent root-search disagreement and no held-input
control mismatch. All 192 tighter-confirmation differences start from an
unresolved observation: 189 resolve to active and three to oscillatory. None
changes one previously resolved destination into another. The
[completion record](../output/input_response_validation_20260929/full_execution.json),
[study summary](../output/input_response_validation_20260929/study_summary.json),
and [parameter comparison](../output/input_response_validation_20260929/parameter_comparison.json)
are the sources for that historical completed run; the earlier partial counts
retain their September snapshot meaning.

## Anchor cards

| Paper question | Best-supported candidate and observation | Boundary of the claim |
| --- | --- | --- |
| Ordinary switching with access to additional states | **Selective withdrawal:** `(e_to_e,i_to_e,e_to_i,i_to_i)=(17,13,19,6)`, `theta_off=8`, `tau_I/tau_E=0.2`, `(B_E,B_I)=(0.35,0)`. A 100 ms E-input increase of 0.5 sends rest → active; a 100 ms I-input increase of 3 sends active → rest. Julia trajectories repeat the full cycle twice from actual endpoints. Separate 100 ms pulses send rest → herald with E increase 1 and active → seizure with I increase 4.5; each pulse returns to the tonic input. [Focused result](selective_anchor_joint_exploration_20261005.md), [Julia reference summary](../reproducibility/selective_anchor_joint_20261005/reference_summary.toml). **Narrative alternative:** `(19,13,19,6)`, ratio `0.2`, `(0.015625,0)` also has repeated switching and both induction routes. [Narrative results](narrative_results.md#selected-example-and-transitions). | The demonstrated seizure induction uses added **I input** in both settings. These are finite-window, sampled-context results; switching and induction at the manuscript ratio 4.4 remain unshown. Fixed-pulse neighboring failures do not prove loss of switching after pulse remeasurement. |
| Paired input control | **Selective withdrawal at that same `theta_off=8` context:** complete E withdrawal sends the herald source to active while the seizure source stays seizure; both held-input controls stay at their sources. Finite withdrawal pulses restore baseline input and recover herald at selected amplitude-duration pairs, while tested seizure-source pulses persist. [Raw baseline](../output/input_response_20260929_v3/anchors/selective_withdrawal/responses/baseline_1/summary.toml), [herald confirmation](../output/input_response_20260929_v3/anchors/selective_withdrawal/responses/baseline_1/root_5/sustained/confirmations.csv), [seizure confirmation](../output/input_response_20260929_v3/anchors/selective_withdrawal/responses/baseline_1/root_7/sustained/confirmations.csv), [pulse brackets](input_response_results.md#finite-withdrawal-can-control-herald-after-the-input-returns). | The measured destinations are finite-window outcomes. The paired withdrawal comparison has not been repeated after the threshold change to 8.75. Direct E displacement is a separate intervention with different units. |
| Intervention effect while retaining ordinary switching | **Selective threshold change at `(0.35,0)`:** raising `theta_off` from 8 to 8.75 sends the original seizure source to herald while the same fixed switching pulses complete two rest → active → rest cycles from actual endpoints. The sampled original-source destination remains seizure at 8.25 and 8.5 and becomes herald at 8.75, 9, 10, and 12. [Julia reference summary](../reproducibility/selective_anchor_joint_20261005/reference_summary.toml). **Narrative alternative:** `theta_off: 8 → 8.75` also redirects seizure to herald while switching remains demonstrable. [Narrative intervention results](narrative_results.md#intervention-tradeoffs). | The sampled transition lies between 8.5 and 8.75; its exact threshold is unknown. Redirection to herald is not recovery to rest or active. The separate `(16,13,19,6)`, ratio `0.2`, baseline `(0.5,0)`, `theta_off: 8 → 12` example sends seizure to active but lacks a joint switching test. [Threshold-to-active check](input_response_results.md#what-the-controlled-checks-establish). |

The author can consider the selective setting across Figures 2–4, or use the
narrative switching and intervention results alongside the selective paired
control. The threshold-to-active case can illustrate parameter dependence
without supplying a combined selective-treatment claim. No choice here asserts
exact attractor counts or assigns a biological state label.

The selective setting also has a [tonic E induction and cessation
measurement](selective_anchor_joint_exploration_20261005.md#tonic-e-induction-and-cessation).
Starting from either rest or active at `B_E=0.35`, each of the seven sampled
`theta_off` values has the same observed bracket
`1.22890625 < B_E,on <= 1.2328125` for reaching herald under held tonic E
and active after release to `0.17` or `0`. Release to `0.35` did not give that
joint outcome at any sampled on input through `16`; at the bracket's first
qualifying input it returned to herald. At that input, the discovered seizure
source stayed seizure after all three releases
for `theta_off=8`, `8.25`, and `8.5`; no seizure source was identified there
at `8.75`, `9`, `10`, or `12`. The [Julia reference
summary](../reproducibility/selective_tonic_e_release_20261005/reference_summary.toml)
records the seven separate rows. This sustained induction/release result does
not repeat the existing paired withdrawal test from an established herald
source at `B_E=0.35`.

## Regime map at the two measured scales

**Tonic-input plane.** Each row is a representative sampled input, not a
certified continuous region. Source roles are the discovered attracting
coordinate roles at that input; role absence means none was found under the
recorded search. The full [representative-context packet](../output/anchor_regime_decision_20261005/representative_contexts.json)
retains all 1,171 contexts, source roles, controls, unresolved sampling, and
paths back to the raw records.

| Fixed parameterization | Representative `(B_E,B_I)` and observed roles | What changes across the listed samples |
| --- | --- | --- |
| Narrative `(19,13,19,6)`, ratio 0.2 | `(0.015625,0)`: rest, active, herald, seizure; `(0.15625,1.5)`: rest, active, seizure; `(0.15625,9.5)`: seizure; `(0.15625,10.75)`: rest. | The herald and ordinary active roles are absent from some sampled input contexts; a high-I sample has only a discovered seizure role, and a nearby higher-I sample only rest. |
| Selective withdrawal `(17,13,19,6)`, ratio 0.2 | `(0.35,0)` and `(0.325,0.84375)`: all four roles; `(0.1,0)`: rest, active, seizure; `(0.175,9.5)`: seizure; `(0.175,10.75)`: rest. | The paired high-state comparison, repeated switching, and both induction routes are measured together at `(0.35,0)`. The other listed inputs show role availability, not measured switching or induction. |
| Threshold-to-active `(16,13,19,6)`, ratio 0.2 | `(0.5,0)`: active, seizure; `(1.25,0.5)`: active, herald, seizure; `(0.25,10.75)`: rest. | The intervention baseline does not itself display the four-role repertoire measured at the narrative or selective switching baselines. |

**Parameter samples and contrasts.** At the selective-withdrawal baseline
`(0.35,0)`, fixed-protocol screen points at `e_to_e=16.75` and `16.875` have no
discovered attracting herald source; `17` and `17.25` show herald → active and
seizure → seizure under complete E withdrawal. The same contrast is recorded
at the tested one-axis values `e_to_i=18.5–19.5`, `theta_off=7.75–8.25`, and
`tau_I/tau_E=0.2–0.4`, including their sampled midpoints. These are points
along separate axes, not a confirmed six-dimensional box. The
[screen comparison](../output/input_response_validation_20260929/parameter_comparison.json)
also retains 13 paired contexts with selective complete withdrawal: 11
herald-to-active and two herald-to-rest while seizure persists. Counts depend
on the declared screen and baseline selection; they are not prevalence.

A separate [Julia threshold sweep](selective_anchor_joint_exploration_20261005.md)
held the selective coupling, ratio, and `(0.35,0)` input fixed. The original
seizure source remained seizure at `theta_off=8.25` and `8.5`, then reached
herald at `8.75`, `9`, `10`, and `12`. At `8.75`, the same pulses still completed
two switching cycles. This samples one threshold axis; it neither locates the
exact transition nor extends the paired withdrawal screen, which was measured
only through `theta_off=8.25`.

The tonic E scan sampled 147 absolute `B_E,on` values from `0.35` through `16`
at each of those seven failure thresholds, refining observed changes to width
`0.01` and confirming the onset bracket with 41-by-41 root searches. Its shared
bracket is a repeated sampled observation, not proof that the threshold is
invariant with `theta_off` or that no narrower input interval was missed.

The 74 detailed follow-ups also supply alternatives. Among their 142
representative baselines with positive `B_E`, both discovered high-state roles,
and one tighter-confirmed complete E-withdrawal observation per source, the
recorded herald/seizure destination pairs are: 66 herald/seizure unchanged,
44 rest/seizure, 27 active/seizure, four rest/rest, and one
periodic-compatible oscillatory/seizure. These selected baselines are not
independent replicates or a frequency estimate for model parameters. They do
show why the established selective anchor is a paper candidate: it has
finite-pulse duration brackets and independent checks in addition to the
sustained-withdrawal pair. [Detailed baseline index](../output/anchor_regime_decision_20261005/representative_contexts.json).

The two herald-to-rest contrasts include ratio about `1.143` at
`figure4_joint_29`, baseline `(1,1)`, and ratio `4.4` at
`input_dependent_e_to_e_4.25`, baseline `(8,0)`. All 13 paired contexts have
independent root and withdrawal/held-control trajectory checks at a 10,000 ms
horizon in the [confirmation record](../output/paper_handoff_20260930/selective_confirmation.json).
At the nearby `input_dependent` anchor with `e_to_e=4` and ratio 4.4,
complete withdrawal instead sends **both** high-state sources to rest at the
sampled `(8,0)` baseline. This contrast shows parameter dependence; neither
case has a measured ordinary-switching route at those inputs. The separate
`(2,1)` input-dependent context has a validated attracting periodic source and
phase-dependent pulse outcomes, which must not be merged with equilibrium
destinations. [Input-response results](input_response_results.md#numerical-reliability-and-limits).
Induction routes and neighborhood extent at the ratio-4.4 selective screen
point remain unmeasured.

## Remaining decisions and stop rule

The author can consider a unified selective setting or separate narrative and
selective examples for Figures 2–4. The selective setting has corrected-source
switching, induction, and threshold-redirection observations; its paired
E-withdrawal control remains a historical frozen-batch measurement. Reconfirm
that baseline paired response under the current source before manuscript use.
The paired E-withdrawal response
**from an established herald source at `B_E=0.35`** after `theta_off` changes
to 8.75 is still unmeasured, so a claim that this specific selective input
control persists after that intervention needs another test. Ordinary
switching after the separate threshold-to-active intervention also remains
unmeasured.

The tonic E result supports active recovery when the input is reduced to
`0.17` or `0`. Ceasing only the added input and returning to the anchor's
`0.35` baseline did not produce the joint herald-induction/active-recovery
outcome in the sampled scan, so that return path is not a measured
self-termination example.

Any proposed parameter or input **region** requires more than the single
selective context: remeasure switching and induction pulses at the selected
neighboring settings, and use matched held-input controls for paired
withdrawal. A failed inherited pulse establishes only failure of that
protocol. The broader Stage B/C domains in the
[experiment card](paper_experiment_card.md) remain unselected and were not run
for this decision package.

All parameter and input maps here are bounded numerical discoveries with
`CompletenessNotCertified`. Sampled adjacency, independent numerical
agreement, and finite follow-up do not certify all attractors, whole-region
behavior, asymptotic rescue, or a biological interpretation. Existing
manuscript text remains for manual author revision.

## Packet verification

The [packet manifest](../output/anchor_regime_decision_20261005/manifest.json)
records SHA-256 identities of the final controller, summary, comparison,
independent confirmation records, executed source metadata, and final study
checksum manifest. The extractor checks final completion, expected case and
baseline counts, zero root/control disagreement, and the exact 192 refinement
transitions before indexing every completed representative baseline. The final
study verifier previously checked 172,099 artifact files and the report; this
decision review checks selected raw records and does not rehash that 14 GB
archive.

The [tracked Julia replay](../reproducibility/selective_anchor_joint_20261005/README.md)
and [reference summary](../reproducibility/selective_anchor_joint_20261005/reference_summary.toml)
add the selective anchor's joint switching, induction, and threshold evidence.
The reviewed run searched baseline and threshold-8.75 equilibria on 41-by-41
grids and other threshold samples on 21-by-21 grids. It used tenfold tighter
solver tolerances and a 5,000 ms finite follow-up for each reported
destination. The merged-source replay was run locally; its trajectory files
remain in [local output](../output/selective_anchor_joint_postmerge_20261006/summary.toml).
The separate [tonic E replay](../reproducibility/selective_tonic_e_release_20261005/README.md)
keeps actual on-input endpoints for three release inputs and held-on controls;
its source-specific summaries and 41-by-41 boundary confirmations are in the
[local merged-source run](../output/selective_tonic_e_release_postmerge_20261006/summary.toml).

The bubblewrapped [subagent review record](../output/anchor_regime_decision_20261005/agent_review.json)
distinguishes applied work from timeouts. A narrow regime reviewer completed;
its reminder to treat higher-ratio cases as contrast points was incorporated.
Its repeated 248-comparison count came from the older partial handoff and was
corrected to 1,171 final independent root comparisons. Anchor-worker attempts
timed out without accepted files, so the anchor cards were checked centrally
against the retained records named above.

## Independent cross-check after the worker timeouts

Three read-only reviewers subsequently checked the anchor cards, sampled
regime map, and provenance separately against the retained raw records. None
found a conflicting numerical result. The anchor review checked the narrative
selected setting, seven qualified cases, repeated switching and induction,
threshold-8.75 destinations, and both selective-withdrawal source and control
records. The regime and provenance reviews independently recounted 91 anchor
plus 1,080 detailed baselines and the five destination-pair counts among the
142 eligible detailed baselines. They also checked the 13 paired screen contexts,
their four trials each, the sampled one-axis contrasts, and the separate
periodic source. The missing seizure-record link in the paired card was added.

The tracked [independent replay](../reproducibility/paper_handoff_20260930/README.md)
was rerun with its pinned Python dependencies. It confirmed all 13 contexts and 52
withdrawal/control trajectories, then rendered the three-context figure and
12 figure trajectories. The replay writes to a new output directory and can
run from a fresh checkout. It does not reproduce the whole 74-case study.

Links under `../output/` refer to the retained **local** study archive; that
directory is ignored by version control and is unavailable in a fresh
checkout. The tracked replay bundle supports the 13-context independent
confirmation, while the full 1,171-baseline recount and narrative raw-record
checks require that local archive. The hashes in the local packet manifest
matched their seven source files during this cross-check.
