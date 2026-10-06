# Two-input response results

**Selection audit (5 October 2026):** The
[selection audit](input_response_selection_audit_20261005.md) identifies
historical follow-up inputs affected by a region-graph correction; the frozen
full study output and its response measurements remain unchanged. Six affected
geometries were replayed separately under the corrected source.

## Paper-handoff update, 30 September 2026 UTC

The partial audit in [the paper evidence handoff](paper_evidence.md) covers 243
completed response baselines: all 91 anchor baselines and 152 from ten selected
follow-ups, nine of which are complete. The interrupted batch was resumed after
verifying its frozen source. Its earlier `running` marker alone was not evidence
of a live process.

All 248 completed independent root comparisons agree and held-input controls
have no recorded mismatch. Thirty-seven tighter confirmations resolve previously
unresolved observations: 36 to active and one to an oscillatory destination.
These are retained refinement differences, not contradictory resolved labels.

Independent SciPy checks confirm 13 selective-withdrawal screen contexts with 52
withdrawal/control trajectories. Eleven recover herald to active while seizure
persists; two recover herald to rest while seizure persists, at time-scale
ratios approximately 1.143 and 4.4. These extend the set of confirmed witnesses
beyond the original fast-inhibition example. Their induction routes and full
neighborhoods remain targets for the paper's staged exploration.

The dated audit and checks are in `output/paper_handoff_20260930/`; the broader
study and its final interpretation remain incomplete. Earlier handoff details
below retain their original temporal scope.

**Historical status note (29 September 2026):** The execution and report paths
below describe the frozen batch controller and its retained local output. The
exploratory Python renderer is no longer part of the active repository; the
current method and Julia archive command are in [the study protocol](input_response_study.md).

## Execution status

At implementation handoff on 29 September 2026, all eight anchors (91 input
settings) and the 224-case parameter screen are complete. The 74 selected
detailed follow-ups in `output/input_response_20260929_v3` remain in progress.
The batch controller writes its current state to
`output/input_response_validation_20260929/full_execution.json` and its numerical
log to `full_run.log` in the same directory. It automatically archives completed
geometry records, verifies checkpoints and source identity, summarizes results,
and generates `output/input_response_figures_20260929/report.html` after the
numerical run finishes. Final interpretation of that expanded set remains
pending review.

The completed anchor report is
`output/input_response_anchor_report_20260929/report.html`; it is labeled partial
because the expanded follow-up is incomplete. The earlier
`output/input_response_20260929` run was interrupted after identifying a
numerical restart issue. Version 2 was interrupted to remove redundant
long-horizon integration before periodic classification. Both are retained as
verified `.tar.gz` archives. Version 3 reuses only their unchanged equilibrium
computations, with source comparisons and unit checksums recorded in
`geometry_cache_provenance.toml`; all response measurements are recomputed.
The active batch uses the frozen source archive. The current implementation
adds only two metadata assignments to that runner, correcting its purpose and
emitted replay command; the exact source comparison is retained in
`implementation_source_comparison.json` in the validation directory. After a
stopped batch, its resume command is
`python3 output/input_response_validation_20260929/finish_batch.py`.

See [the method](input_response_study.md) for the equations, parameter ranges,
input bounds, source discovery, protocols, and numerical limits. The manuscript
has not been edited.

## What the controlled checks establish

At `(a,b,c,d)=(17,13,19,6)`, failure threshold 8, and `tau_I/tau_E=0.2`,
the positive baseline `(B_E,B_I)=(0.35,0)` supports distinct herald and seizure
candidate sources. Reducing `B_E` to 0.1 sends herald to intermediate activity
while seizure persists. This was reproduced with independently written SciPy
equations and integration, and in the Julia regression suite. An archived-source
Julia replay also reproduces the contrast for complete E-input withdrawal.
These are recorded finite-window outcomes, not a claim of permanent recovery.

The threshold-to-active check fixes `(a,b,c,d)=(16,13,19,6)`,
`tau_I/tau_E=0.2`, and `(B_E,B_I)=(0.5,0)`. Raising the failure threshold from
8 to 12 sends the former seizure source to intermediate activity in both the
Julia and independent SciPy integrations. Thus seizure-to-herald redirection
is a property of the earlier selected example, not a universal consequence of
raising that threshold. This matched intervention is a separate check from
the two-input maps at fixed parameters.

The systematic maps retain equilibria at their actual input values, including
states without a zero-input counterpart. Existence, local stability, and the
destination reached after changing input are recorded separately. A change in
the number of discovered equilibria does not by itself determine recovery.

For the input-dependent anchor, `(a,b,c,d)=(4,9,19,4)`, failure threshold 8,
and `tau_I/tau_E=4.4`, the searches find three equilibria at `(B_E,B_I)=(8,0)`:
two attracting high-state candidates and a saddle. At `(0,0)`, they find one
rest equilibrium. Complete E-input withdrawal from either high-state source
reaches rest in the confirmed trajectories, while the held-input controls
remain at the corresponding high state. These discoveries illustrate why
source availability must be evaluated at the actual tonic input. Their counts
are not completeness certificates.

The equilibrium input chart contains no time constants. Changing the timescale
ratio therefore preserves its equilibrium coordinates while potentially changing
stability and the trajectories that separate attraction basins. Couplings and
response thresholds also change the chart itself. This separates two mechanisms
for parameter dependence: movement of the equilibrium branches and changes in
the dynamics around those branches. Neither mechanism implies a universal
herald/seizure rescue ratio.

## Finite withdrawal can control herald after the input returns

In the selective example above, E-input withdrawal pulses return to the same
baseline `B_E=0.35` after the pulse. The following pairs are adjacent sampled
outcomes: the lower withdrawal retains herald, while the upper withdrawal reaches
the active equilibrium after the baseline is restored.

| Pulse duration | Lower withdrawal: herald | Upper withdrawal: active |
| --- | ---: | ---: |
| 2 ms | 0.314453 | 0.317188 |
| 10 ms | 0.120313 | 0.123047 |
| 100 ms | 0.046484 | 0.049219 |

Independent SciPy integration reproduced all six endpoint outcomes. These are
sampled brackets, not certified minimal thresholds. No seizure recovery was
observed over the tested E-withdrawal range `0–0.35` and pulse durations
`1, 2, 5, 10, 20, 50, 100, 200 ms`. A 1 ms withdrawal did not recover herald in
the sampled range either. Thus pulse duration matters even when the available
input change is sufficient under longer protocols.

I stimulation gives a different outcome in this same example. A 100 ms pulse
adding `B_I=0.917969` returns to herald, while the adjacent sampled amplitude
`0.921875` reaches seizure after the baseline is restored. Independent SciPy
integration reproduced both outcomes. The sampled pure-I pulse maps contain
herald and seizure destinations, with no observed recovery. This is a concrete
reason to retain separate E-withdrawal and I-stimulation coordinates: increasing
input to the inhibitory population need not improve control in a model with
inhibitory failure.

The direct-displacement comparison gives a different numerical result:

| Example | Starting `B_E` (`B_I=0`) | Herald E reduction | Seizure E reduction | Seizure / herald |
| --- | ---: | ---: | ---: | ---: |
| Earlier narrative, `a=19` | 0.015625 | 0.067732 | 0.131805 | 1.946 |
| Selective withdrawal, `a=17` | 0.35 | 0.016232 | 0.058441 | 3.600 |

Both rows hold `(b,c,d)=(13,19,6)`, failure threshold 8, and `tau_I/tau_E=0.2`.
They differ in both recurrent excitation and starting input; this table does
not isolate the effect of either variable. Each value is the smallest observed
successful direct E displacement at the specified input, with I initially
unchanged. All four selected trajectories reach the active equilibrium and
passed tighter-tolerance confirmation.

## Parameter-neighborhood checks

The initial expansion screen is complete: 224 distinct parameter tuples comprise
53 local perturbations, 128 joint Halton samples, and 43 midpoint refinements.
There are 116 cases in the `(b,d)=(9,4)` family and 108 in `(13,6)`. The fixed
protocol screens produce 98 distinct observed signatures and select 74 cases
for detailed follow-up. That follow-up remains incomplete; a shared screen
signature does not establish equivalence over other inputs or protocols.
The response-tail rule keeps 200 screened cases within `0–16 × 0–16` and expands
24 to `0–16 × 0–32`; all reach the configured tail bounds.

At the selective-withdrawal baseline `(B_E,B_I)=(0.35,0)`, complete withdrawal
preserves the herald-to-active / seizure-to-seizure contrast under each of these
single-parameter changes:

| Changed parameter | Tested value(s) | Other parameters |
| --- | --- | --- |
| Recurrent excitation `a` | 17.25 | Selective example values |
| Excitatory recruitment of inhibition `c` | 18.5, 19.5 | Selective example values |
| Inhibitory failure threshold | 7.75, 8.25 | Selective example values |
| `tau_I/tau_E` | 0.4 | Selective example values |

Independent SciPy searches agree with source-root discovery at all seven local
parameterizations, and independent integration reproduces the twelve withdrawal
trajectories for the six cases in the table. At `a=16.75`, the searches find no
attracting herald candidate at the same starting input; that case has no paired
herald/seizure withdrawal measurement. These finite neighboring checks support
parameter dependence without establishing a whole robust region or its extent.

## Relation to the paper

The useful narrative is parameter-dependent control of coexisting activity
states. The positive-baseline comparison gives a concrete example in which
the same available input withdrawal controls herald but fails to control
seizure. That supplies a selective-control example for developing a later
feedback explanation of a transient herald spike.

Direct E displacement answers a different question: how far the state must
be moved along the E coordinate, with I and tonic inputs held fixed, before
the sampled trajectory recovers. Its herald/seizure ratio should be presented
alongside the parameterization and starting inputs. It is not a ratio of
afferent-input costs, nor a demonstrated biological feedback mechanism.

The input maps and pulse-duration measurements provide requirements for a
later coupled model: which corrections work, from which states, and over which
tested durations. A stable point-model herald candidate is not itself a
self-terminating herald spike. The feedback dynamics that create and terminate
such a transient remain a separate modeling task.

## Numerical reliability and limits

The final package suite passes 4,809 assertions. Nine standalone storage tests
pass, including interrupted-archive recovery and checksum-failure handling.
The completed anchor report passes its link and interaction-handler checks;
selected exported figures, including cycle-phase plots, were visually inspected.

The completed anchors cover 91 representative input settings, with 206,423
sustained-input observations and 532,832 pulse observations. All 91 independent
root checks agree with the corresponding discovery results. The held-input
controls and selected tighter confirmations have no recorded destination
mismatches. There are 218 unresolved sweep observations; these remain separate
from equilibrium-compatible and periodic-compatible outcomes.

The input-dependent anchor at `(B_E,B_I)=(2,1)` also supplies a numerically
validated attracting periodic source, with period approximately 44.63 ms.
Its pulse outcomes vary across the sixteen tested starting phases. Oscillatory
and equilibrium destinations remain separate in the report; this cycle is not
assigned a herald-spike interpretation.

The two-input study uses the existing solver's strict domain rejection and
stops at saved times. This prevents a tiny negative inhibitory endpoint or
interpolated sample from invalidating a subsequent solve. Forty trials affected
in the interrupted run were replayed successfully after the fix. No state is
clipped, and no endpoint is replaced by an equilibrium.

Root searches and sampled critical sets remain numerical discoveries.
Independent agreement, index consistency, and response-tail bounds do not
certify attractor completeness. Cells and intervals left unresolved by sampling
budgets remain in the artifacts. Finite state probes and cycle phases can miss
attractors or phase-dependent responses. A sampled signature shared by two
parameterizations does not establish behavioral equivalence, and the selected
parameter sample does not estimate prevalence in an unspecified population.

Validation artifacts are retained in
`output/input_response_validation_20260929`: independent equations and checks,
archived-source replay, the boundary regression replay, test results, and
machine-readable summaries. Scientific output remains local and on demand;
CI tests the machinery rather than certifying these conclusions.
