# Initial intervention-study results

These are numerical results for the supplied exploration parameters and the
approved point model, using Julia 1.10.12. They are not physiological
calibration, exhaustive attractor discovery, or biological regime assignments.
The [study plan](intervention_study.md) gives the protocols and replay commands;
the [theory notes](theory_notes.md) separate analytical results from numerical
observations. Generated artifacts are under the ignored `output/` directory.

## Main finding

At the Figure 3 anchor, increasing `theta_off` from 8 to 10 or 12 leaves a
locally attracting high-E/high-I equilibrium and the quiescent-coordinate
equilibrium. The high-E/low-I branch turns at a numerical fold candidate near
`theta_off=8.723`; its low-I attracting continuation is not found at 10 or 12.
The second anchor gives the same qualitative branch pattern, with a fold
candidate near 8.634. These are approximate continuation observations, not
certified fold locations or proof that no other attractor exists.

This supports testing selective suppression of the low-I state. It does not
yet establish preservation of the manuscript's intended functional active
regime: both high-E attracting equilibria at the Figure 3 baseline lie on
the descending inhibitory response branch, and the intermediate equilibrium
near `(0.311,0.425)` is repelling. Raising the threshold changes inhibitory
activity on the retained high-E branch, rather than leaving that activity
exactly unchanged.

| Figure 3 threshold | Discovered attracting equilibria | Retained high-E/high-I state | High-E/low-I attracting candidate |
| ---: | ---: | --- | --- |
| 8 | 3 | `(0.499999911,0.447721024)` | `(0.500000000,0.000558674)` |
| 10 | 2 | `(0.499999068,0.499999062)` | Not discovered |
| 12 | 2 | `(0.499999068,0.499999994)` | Not discovered |

The quiescent-coordinate equilibrium is approximately
`(0.000580379,2.178e-9)` throughout this threshold slice. At threshold 10,
the retained high-E/high-I state still has negative inhibitory slope; at 12
it has positive slope. Response-branch sign alone cannot assign its role.

## Coexistence map and continuation

`output/coexistence_study` contains 3,300 matched-model cells, 1,716 unique
searches (controls are reused along their inactive failure-threshold axis),
235,092 retained nonlinear attempts, and 14,348 equilibrium rows across
matched cells. All search contexts executed. There were 10,849 rejected and
3,048 boundary-ambiguous attempts; these remain in the artifacts. Solver
return status and independently checked candidate acceptance are separate.

For each anchor there are 825 FoI cells. Figure 3 has 75 cells with three
discovered attracting candidates and 750 with two; the second anchor has 149
with three and 676 with two. Matched controls have two throughout. This
three-candidate region follows a diagonal band. At threshold 8, its sampled
E-to-I couplings are 18–19 for Figure 3 and 18–20.5 for the second anchor.
At threshold 10 they shift to 22–23 and 22–24.5, respectively. These are grid
observations, not continuous boundary estimates.

The standalone count map is
`output/study_figures/coexistence.png`, with its source hash record alongside.
Its colors count discovered locally attracting equilibrium candidates; they
do not count biological states or all possible attractors.

`output/continuation_study` contains 96 branch traversals and 6,710 accepted
points, including attracting, saddle, and repelling portions. Eighty-six
traversals stop near a requested parameter boundary. Ten stop at the minimum
step with numerical ambiguity near activity saturation boundaries; these
are incomplete portions, not additional bifurcation findings. Maximum
accepted balance residual is below `1e-10`.

There are 32 fold-screening brackets, including repeated traversals of the
same branches from different initial roots, and no Hopf brackets in these
representative runs. The high-E/high-I branches remain locally attracting
through threshold 12. No branch switching, exhaustive continuation, or
bifurcation nondegeneracy certification was performed.

## Explicit pulse outcomes at the Figure 3 baseline

`output/pulse_study` contains 5,408 trials: 5,280 initial grid trials plus
128 adaptive refinements. All returned finite-window compatibility with a
locally attracting discovered equilibrium at 5,000 ms after withdrawal.
There were no integration failures or unresolved trials in this baseline
run. Sixty-four adjacent outcome-change brackets were retained, 42 for FoI
and 22 for control.

For the baseline FoI search, local root IDs 1, 5, and 7 denote the
quiescent-coordinate, high-E/high-I, and high-E/low-I equilibria listed above.
These IDs have meaning only in this search context. Positive E pulses can
move root 1 to root 7; positive I pulses can move root 5 to root 7. None of
the sampled positive E, I, or equal pulses moves root 7 to another equilibrium.
Negative E pulses can move root 7 to root 1. The positive-drive result agrees
with the separate invariant-region proof in the theory notes.

The table gives adjacent tested amplitude endpoints with different outcomes;
it does not assert a unique monotone threshold between them or exclude
unobserved success islands. Lower and upper endpoints have the initial and
destination outcomes, respectively, for these listed transitions.

| Transition and target | Duration (ms) | Lower amplitude | Upper amplitude |
| --- | ---: | ---: | ---: |
| Root 1 to root 7, positive E | 1 | 1.125 | 1.1875 |
| Root 1 to root 7, positive E | 20 | 0.5000 | 0.5625 |
| Root 1 to root 7, positive E | 200 | 0.3750 | 0.4375 |
| Root 5 to root 7, positive I | 1 | 0.4375 | 0.5000 |
| Root 5 to root 7, positive I | 5 | 0.0625 | 0.1250 |
| Root 5 to root 7, positive I | 200 | 0.0000 | 0.0625 |
| Root 7 to root 1, negative E magnitude | 20 | 6.3125 | 6.3750 |
| Root 7 to root 1, negative E magnitude | 200 | 6.1250 | 6.1875 |

For negative E pulses from root 7, no transition was observed at durations
1, 2, 5, or 10 ms within the sampled magnitude range through 8. This is a
statement about this protocol and finite follow-up, not unrestricted rescue.
All component input integrals are recorded as defined control costs, not
biological energy.

## Intervention and robustness comparisons

`output/intervention_study` contains 216 matched-condition searches: 90
baseline/intervention, 78 targeted single-axis robustness, and 48
deterministic joint-sample searches. There are 29,592 attempts and 1,004
equilibrium rows. All search contexts executed; 3,352 nonlinear attempts
returned nonsuccess and were retained separately from acceptance checks.

The initial slices give substantive counterpoints to a simple intervention
ranking:

- Reducing E-to-I coupling to 17.1 already leaves two discovered attracting
  equilibria, including high-E/high-I. At the 60% endpoint, 11.4, the high-E
  equilibrium has I approximately 0.3940. Thus recruitment reduction can
  also suppress discovery of the low-I branch while shifting retained I.
- Reducing E-to-E coupling to 12.75 loses discovery of the high-E/high-I
  attracting candidate while retaining the high-E/low-I candidate. The same
  pattern persists at the 60% endpoint, 10.2.
- Increasing I-to-E coupling to 12.9375 similarly loses discovery of the
  high-E/high-I attracting candidate while retaining high-E/low-I, including
  at the 150% endpoint, 13.5.

These comparisons use raw coordinates and discovered branches; they do not
declare an optimal intervention or assign functional activity. A scalar
disturbance metric, empirical observation model, or biological weighting was
not invented for this study.

`output/pulse_comparison` executes the six representative complements to the
separate baseline pulse run. Together they contain 30,166 trials. All are
finite-window equilibrium-compatible, with no unresolved or failed
integrations under the specified settings. Each case retains its full
amplitude-duration boundary table and raw endpoint coordinates.

| Complement case | Pulse trials | Discovered FoI high-activity attracting endpoint(s) |
| --- | ---: | --- |
| `theta_off=10` | 4,312 | High E/high I approximately `(0.499999,0.499999)` |
| `theta_off=12` | 4,312 | High E/high I approximately `(0.499999,0.500000)` |
| `e_to_i=11.4` | 4,308 | Approximately `(0.500000,0.394007)` |
| `e_to_e=10.2` | 3,214 | High E/low I approximately `(0.500000,0.000559)` |
| `i_to_e=13.5` | 3,210 | High E/low I approximately `(0.500000,0.000559)` |
| Second coupling anchor | 5,402 | High E/high I and high E/low I |

At the second anchor, negative E pulses from the low-I equilibrium reach
compatibility with the high-I equilibrium in nine sampled/refined trials and
with the quiescent-coordinate equilibrium in twenty. Positive I/equal pulses
can instead move the high-I equilibrium to the low-I equilibrium. These
transitions illustrate why the intervention class and destination must be
specified; a single rescue/no-rescue label loses relevant behavior.

## Periodic-candidate screening

`output/periodic_candidates` contains 44 requested initial conditions and
45 integration stages. Every final stage is equilibrium-compatible. One
Figure 3 FoI trajectory starting at `(0.5,0.5)` remained unresolved at 5
seconds and became compatible after extension to 10 seconds. No trace met
the recurrence screen for a shooting seed, so no point-model periodic orbit
was validated by this run.

This finite sampling cannot exclude periodic orbits in other basins or
parameter neighborhoods. The shooting machinery was independently tested on
analytical attracting, repelling, and neutral cycles, equilibria, damped
transients, and incomplete integrations. Those tests validate the machinery,
not the existence of a manuscript active cycle.

## Remaining scientific work

Refine and certify the branch-change candidates as appropriate; widen basin
and cycle discovery; resolve the functional-active interpretation; and
compare pulse-transition intervals and branch disturbance under numerical
and parameter refinement. Source snapshots and checksums identify each
executed revision. Subsequent implementation or documentation changes do
not silently replace the evidence archived with a previous run.

## Implementation validation

The final integrated command
`julia --project=. -e 'using Pkg; Pkg.test()'` passed all 2,728 assertions.
Failure-path tests deliberately trigger solver iteration-limit warnings.
Independent numerical review covered continuation, shooting, scientific
runner interpretation, and provenance. Artifact checksums were verified,
and an archived-source pulse smoke replay reproduced 89 numerical/configuration
artifacts byte-for-byte. `git diff --check` and whitespace checks on all 22
new files passed. No dependency changes or manuscript edits were made.
