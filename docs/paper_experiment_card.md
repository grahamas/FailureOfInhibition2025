# Next paper experiments: proposed execution card

This card implements the agreed staged expansion within the existing point
model. **New numerical ranges below are proposals, not author-selected domains.**
The frozen 74-case follow-up is complete; Stage A reconciles its historical
output with corrected selections and protocols.
The broader experiments start only after the author chooses their domains.

## Question and measurements

Determine where ordinary rest–active switching coexists with additional
high-activity states, where available inputs control those states differently,
and where parameter interventions improve high-state control while preserving
ordinary switching. Keep successful examples and contrasting regimes.

Reuse the current state roles, input chart, outcome diagnostics, root-search
checks, source/configuration snapshots, and amplitude-duration protocols.
Sources are discovered at their actual inputs; a zero-input counterpart is not
required. Compare both high-state sources at the same parameters and input.
Retain E withdrawal, I stimulation, combined input changes, and direct E
displacement as separate measurements. Record actual induction endpoints.

Report state coordinates, destination, switching success, observed successful
amplitude-duration pairs, and settling time separately. Do not introduce a
weighted functional score or a biological cost conversion. A failed fixed pulse
is followed by the existing stimulus search before declaring switching
undemonstrated. Compare the unchanged pulse and the remeasured pulse explicitly.

## Stage A: reconcile retained evidence

Review the 74 selected detailed cases in `input_response_20260929_v3` as
historical output under its frozen source, including confirmation differences.
The [selection audit](input_response_selection_audit_20261005.md) shows that
the corrected cell and region rules can change representative inputs. Recompute
geometry selection for the 306 archived cases under the corrected source, then
rerun affected screening and response protocols, including newly selected
detailed cases, and independently confirm candidate contexts before selecting
figures. Repeat any permanent-release induction and switching-pulse witnesses
proposed for figures under the corrected exact-domain handoff policy. Preserve
the frozen records and identify which source produced each result. The 224
parameter screens, 91 anchor baselines, original narrative and release studies,
and independently checked selective-withdrawal examples remain historical leads
pending those checks. This stage needs no new
parameter-domain choice and should precede another large atlas.

The existing expanded axes are recurrent excitation 0–24, recruitment 12–28,
failure threshold 6–12, and time-scale ratio 0.2–4.4, with inhibitory output and
self-coupling fixed at either (9,4) or (13,6). These are exploration bounds.

## Stage B: proposed coupling screen

Use 64 deterministic joint samples over the six-dimensional box below, plus
the existing anchors. Start the new six-dimensional sequence at its first
index and archive the exact case list before evaluating outcomes. The earlier
joint samples came from two separate four-dimensional 64-point sequences at
fixed couplings, so they are not a prefix of this design. The purpose is to
explore combinations between the two previously fixed coupling families.

| Parameter | Proposed range | Relation to existing exploration |
| --- | --- | --- |
| `e_to_e` | 0–24 | Unchanged |
| `e_to_i` | 12–28 | Unchanged |
| `i_to_e` | 9–13 | New continuous variation between existing anchors |
| `i_to_i` | 4–6 | New continuous variation between existing anchors |
| `theta_off` | 6–12 | Unchanged |
| `tau_ratio` | 0.2–4.4 | Unchanged |

Keep response slopes 5, E threshold 1.5, I onset 4, and `tau_E=7.8 ms`.
Use the existing nonnegative input probes and response-tail rule. Run the cheap
geometry and fixed-protocol screen first, not a full pulse atlas for every case.
After review, select at most eight detailed cases: successful joint behaviors,
contrasting outcomes, and points near observed changes. Retain selection reasons;
do not equate shared screen signatures with behavioral equivalence.

## Stage C: proposed response and time-scale checks

Start from three established anchors: narrative, selective withdrawal, and
threshold-to-active. Vary one quantity at a time, retaining baseline values:

| Quantity | Proposed values |
| --- | --- |
| E response slope | 4, 5, 6 |
| Shared I onset/failure slope | 4, 5, 6 |
| E threshold | 1.25, 1.5, 1.75 |
| I onset threshold | 3.5, 4, 4.5 |
| `tau_I/tau_E` | 0.2, 0.4, 1, 2, 4.4 |

This gives 13 distinct settings per anchor including the baseline, or 39 before
any cross-anchor deduplication. Failure threshold remains that anchor's value;
its separate intervention sweep uses the existing 6–12 exploration envelope.
Equal inhibitory slopes, response formula, state domain, and normalization stay
unchanged. These perturbations test sensitivity, not physiological calibration.
Joint response-parameter sampling is a later, separately specified batch if the
one-at-a-time results identify an interaction worth testing.

## Selection, confirmation, and stop rule

For each selected case, search for ordinary switching and induction, then compare
paired control and intervention outcomes. Prefer a common parameterization for
multiple figures only when it actually meets their criteria. Use separate,
clearly identified examples when their regimes differ.

Confirm headline trajectories with 41×41 independent root searches, tighter
tolerances, longer recorded follow-up, and independently written equations.
Use held-input controls from the identical actual endpoint. When an equilibrium
loses stability, inspect recurrence with the existing periodic-orbit machinery
before treating the loss as disappearance of sustained activity.

One bounded screen per stage is followed by review. No automatic multiplication
of scan size is planned. End exploration when the main claims have confirmed
examples, characterized neighborhoods, and explanations of important boundaries.
An undecided claim receives an explicit follow-up question or remains open;
failure to find a behavior does not establish its impossibility.

Before launch, the author chooses the Stage B/C domains and the detailed-case
budget. The proposed default is the ranges above and at most eight detailed
Stage B cases. Benchmark one existing representative to estimate runtime before
reserving the full batch. Continue using at most two scientific cases concurrently.
