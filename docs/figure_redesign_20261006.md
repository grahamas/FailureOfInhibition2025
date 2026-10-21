# Figure redesign: scientific questions and evidence

The author approved a broader point-model repertoire, question-driven new
measurements within the existing exploration envelope, one reference circuit
with contrasting regimes, and detailed work on at most eight parameterizations.
This replaces the previous single-setting figure design. It does not extend
the model to spatial recruitment, autonomous feedback, or physiological fitting.

## What earns a panel

Each main panel must reveal a mechanism, dependence, boundary, or comparison.
One connected trajectory introduces the state transitions. Repeated examples
of the same transition and unchanged endpoint lists belong in text or the
evidence bundle. A simpler sentence can replace a figure when the plotted
variation carries no additional scientific meaning.

The delivered sequence is:

1. **Mechanism and ordering.** The approved fire-then-fail response, matched
   monotone/FoI nullclines, discovered equilibria, and the same-input ordering
   restriction. The physical domain is context in an inset.
2. **Dynamical repertoire.** Sampled two-input equilibrium maps, measured
   periodic-source locations, equilibrium stability versus time-scale ratio,
   and continued cycle amplitude/period. Equilibrium positions and their
   stability are different measurements.
3. **Accessibility.** A connected rest/active/high-state demonstration, marked
   on source-specific amplitude-duration maps, alongside finite-settling input
   sweeps. Every pulse begins at the preceding actual endpoint.
4. **Control.** Matched herald/seizure withdrawal outcomes, contrasts across
   contexts, the analytical positive-increment restriction, and cycle-phase
   dependence. The restriction requires increments above baseline, not merely
   nonnegative total drive.
5. **Intervention tradeoffs.** Destinations after changing failure threshold,
   recurrent excitation, or recruitment, measured alongside two rest-active
   switching cycles with original and remeasured stimuli.

The old repeated tonic-on bracket is reported in text in the earlier dated
packet; it does not receive a main panel here. Direct E displacement remains
in the full numerical evidence as a separate measurement from afferent input.

## Measurement design

The frozen registry contains the historical eight anchors and 224 screen
parameter records. Only their parameters and input probes are reused. Geometry
and response observations are regenerated under the corrected cell-consistency
and exact-domain handoff rules. Completed new screen contexts are archived
losslessly with verified per-file digests. Historical output is untouched.

The declared envelope is recurrent excitation 0–24, E-to-I recruitment 12–28,
failure threshold 6–12, and tau_I/tau_E 0.2–4.4, with inhibitory coupling
families (i_to_e,i_to_i)=(9,4) and (13,6). Existing response parameters and
adaptive nonnegative input bounds are retained. The fixed screen uses the
existing 180-evaluation adaptive-cell budget; presampled baselines and critical
offsets are additional observations. Detailed geometry uses the existing
4,000-evaluation budget. Gray/mixed/unresolved cells are not inferred regions.

Six predeclared contrast leads are selective withdrawal, narrative, weak
input-dependent recurrence, threshold-to-active, weak recurrence at e_to_e=4.25,
and figure4_joint_29. The remaining two slots use the first registry cases with
uncovered measured screen signatures. Selection is deterministic and recorded;
it is not an estimate of prevalence or an assertion of behavioral equivalence.

The selected circuits differ as follows. Contrasts across rows are not isolated
causal effects of a single parameter; only the explicitly matched comparisons
hold other parameters fixed. Values below are rounded for reading; the registry
retains full precision.

| Circuit ID | (E→E, I→E, E→I, I→I) | theta_off | tau_I/tau_E |
| --- | --- | --- | --- |
| selective_withdrawal | (17,13,19,6) | 8 | 0.2 |
| narrative | (19,13,19,6) | 8 | 0.2 |
| input_dependent | (4,9,19,4) | 8 | 4.4 |
| threshold_to_active | (16,13,19,6) | 8 | 0.2 |
| input_dependent_e_to_e_4.25 | (4.25,9,19,4) | 8 | 4.4 |
| figure4_joint_29 | (17.25,13,22.8642,6) | 10.848 | 1.142857 |
| four_sinks_central | (17,9,19,4) | 8 | 0.2 |
| three_sinks_ascending | (19,13,14,6) | 6 | 4.4 |

Detailed responses use the declared anchor baselines and the retained periodic
lead at (B_E,B_I)=(2,1) for the weak-recurrence case. They do not launch every
representative baseline in a new full atlas. Each detailed baseline has an
independent 41-by-41 seed search. Full input geometry retains additional
component representatives and their root-search/continuation checks.

Pulse durations remain 1, 2, 5, 10, 20, 50, 100, and 200 ms; follow-up horizons
remain 5,000, 10,000, and 20,000 ms. Temporary pulses restore baseline, while
sustained withdrawal sets E input to zero. Held-input controls use the same
source. Zero-E baselines are excluded from withdrawal comparisons.

Periodic sources require shooting validation and attracting Floquet status.
Eight initial phases are increased to sixteen when the measured sustained maps
differ. Unresolved recurrence remains unresolved. Time-scale continuation
retains accepted cycle points, corrector attempts, stability, and termination
reasons; trace-zero equilibrium points are labeled numerical candidates.

The connected demonstration uses midpoints of the widest observed successful
100 ms intervals, then tests the complete sequence from actual endpoints.
Successful intervals are sampled observations; choosing their midpoint requires
a fresh solve. Display breaks omit settled portions and are explicitly marked.

Interventions retain the actual original high-state coordinates as their
initial condition. Failure threshold is sampled at 0.25 spacing over 6–12.
Recurrent excitation and recruitment use the existing 5% decrement sequence
through 60% of baseline, restricted to the approved envelope. Original pulses
are E+0.5 and I+3 for 100 ms. If those fail and both switching sources remain
available, the existing duration list and 0.25-spaced amplitudes through 8 are
searched for replacement pulses and two complete cycles are remeasured.

## Evidence and figure acceptance

The data builder requires completed stage checkpoints and successful independent
equation checks. The latter use separately written response and ODE equations
with 1e-12 tolerances; they use the same Tsit5 integrator, not an independent
solver implementation. They check the connected sequence, paired held/released
trajectories, periodic waveforms, confirmed pulse/restoration trajectories at
every plotted cycle phase, and intervention trajectories. Periodic source and
destination checks independently recompute closure and attraction. Mechanism
curves and reported roots, sampled geometry signatures and fills, time-scale
spectra and accepted periodic points, and plotted input-history sweep are also checked
independently before packet building.
The time-scale stop check supports the declared step limit and parameter
boundary; a different stop requires explicit verification before rendering.

The renderer consumes a portable CSV/TOML packet with exact file-set and SHA-256
checks. It writes PDF, SVG, PNG, and rendering provenance into a new directory.
Data selection and omitted full-study measurements are recorded in the packet.

Numerical searches retain `CompletenessNotCertified`. Source roles remain
provisional model-coordinate labels. A missing source, failed pulse, unresolved
trajectory, and an observed alternative destination remain distinct. A loss of
discovered equilibria is not a demonstrated recovery, and a continued branch
termination is not automatically a bifurcation identification.

Final captions and biological interpretation remain for author review. Visual
acceptance requires inspection of all five rendered figures at manuscript size,
followed by revision and rerendering for any material layout or semantic issue.

## Results

The refreshed screen completed all **232 parameterizations**, recording 44,643
input contexts and 5,616 fixed-protocol response observations. These are
purposefully selected parameter records and adaptive samples, not independent
replicates or estimates of prevalence. The deterministic additional selections
were `four_sinks_central` and `three_sinks_ascending`.

The connected reference simulation completed this route without state resets:

**rest → active → rest → herald → active → rest → active → seizure.**

All seven pulses lasted 100 ms and each restoration was followed for 5,000 ms.
Their increments, in order, were E+0.48046875, I+3.2421875, E+8.26640625,
E−0.199609375, I+3.2421875, E+0.48046875, and I+10.0234375. Baseline input
was (0.35,0). These are model input units; the widest-successful-interval rule
selects illustrative interior pulses, not calibrated physiological doses.

The weak-recurrence circuit at input (2,1) has an attracting periodic source
coexisting with a discovered attracting equilibrium that has no assigned role.
The pulse maps distinguish return to oscillation from arrival at that equilibrium;
phase is therefore relevant to destination, not only to transient waveform.
The time-scale study continued 251 distinct, numerically validated attracting
cycle points from tau_I/tau_E = 4.4 down to approximately 2.56667. Periods span
approximately 19.195–44.631 ms over this segment. The decreasing branch stopped
at the 250-step budget; the increasing branch started on the approved upper
parameter boundary. The trace-zero equilibrium candidate is approximately
2.43950, beyond the continued segment. **The run does not identify the cycle's
birth or certify a Hopf bifurcation.** At the reference circuit's ordinary
active equilibrium, the corresponding linear candidate is approximately
0.50177; changing time scales leaves the equilibrium coordinates fixed.

The five positive-E contexts with both high-state sources have these permanent
E-withdrawal outcomes; held-input controls retained each original source:

| Circuit and baseline (B_E,B_I) | Herald source | Seizure source |
| --- | --- | --- |
| Reference, (0.35,0) | active | seizure |
| Narrative contrast, (0.015625,0) | herald | seizure |
| Weak recurrence, (8,0) | rest | rest |
| Nearby recurrence e_to_e=4.25, (8,0) | rest | seizure |
| Joint contrast 29, (1,1) | rest | seizure |

This directly distinguishes selective control, joint recovery, and retention
of both high states. Baselines with B_E=0 are excluded from withdrawal
comparisons. Temporary decrements and permanently zero input remain separate
protocols.

The intervention study completed 42 settings in each of two circuits. Raising
theta_off from 8.5 to 8.75 changes the sampled destination of the original
seizure source from seizure to herald in the reference, and from seizure to
active in the contrast. These are brackets between sampled settings, not exact
transition values. The reference retains its original ordinary switching
protocol throughout the higher-threshold settings. The contrast lacks the
required pair of classified rest/active sources there, so the same switching
claim is unavailable. Two reference settings retain switching only after
remeasuring stimuli; most lower recruitment settings instead reach an
unassigned equilibrium. Neither an unassigned root nor loss of a named source
is promoted to recovery.

## Why the panels remain

| Figure | Additional information beyond a transition example |
| --- | --- |
| 1A–C | Relates the response mechanism to matched equilibrium geometry and the analytical ordering restriction. |
| 2A–B | Shows how the discovered equilibrium repertoire varies with two independent inputs, including unresolved sampling. |
| 2C–D | Separates fixed equilibrium coordinates from changing stability, and measures a finite periodic branch. |
| 3A | Replaces the four repetitive transition demonstrations with one connected simulation and aligned inputs. |
| 3B–C | Measures amplitude-duration dependence and different outcomes from different source states. |
| 3D | Measures history dependence under a continuing input sweep. |
| 4A–B | Compares temporary control of paired sources and permanent withdrawal across contrasting circuits. |
| 4C | Explains a protocol obstruction under the explicit invariant-region assumptions. |
| 4D | Tests starting phase as an additional control variable for a periodic source. |
| 5 | Compares intervention destination with preservation of ordinary switching; one symbol distinguishes original from remeasured successful stimuli. |

## Caption requirements

All five figures are model results for author review. Rest, active, herald,
and seizure are provisional coordinate roles. A plus denotes a discovered
attracting equilibrium without an assigned role; it is distinct from a gray
cross denoting unresolved observation. Input amplitudes are model units and
time is in ms unless an axis explicitly uses seconds. No normalization or
physiological fit has been introduced.

Figure 1 uses the reference parameters (e_to_e,i_to_e,e_to_i,i_to_i)=(17,13,19,6),
theta_off=8, tau_I/tau_E=0.2, and input (0.35,0). Only the inhibitory response
changes in the monotone comparison. The inset shows the full physical state
domain [0,1]²; main portraits enlarge the region containing the equilibria.
The ordering statement concerns equilibria of one common fixed-input system,
not arbitrary points on trajectories.

Figure 2 compares that circuit with couplings (4,9,19,4), theta_off=8,
tau_I/tau_E=4.4. Cell colors summarize matching measured signatures; gray cells
remain unresolved. Neither cell homogeneity nor the finite root searches
certify a continuous region. Black baseline markers distinguish tested
periodic sources by a star. The periodic branch is at input (2,1), while the
reference spectrum is at (0.35,0). Their time-scale scans therefore answer
related questions in different declared contexts.

Figure 3 displays 50 ms before, the 100 ms pulse, and 250 ms after each event.
Marked breaks omit later settled portions; onset labels retain actual elapsed
seconds. Classifications use all 5,000 ms of recorded follow-up. Pulse maps
show measured samples, not fitted boundaries. The input sweep carries actual
endpoints between finite-settling steps; connecting segments guide the eye and
do not constitute equilibrium continuation.

Figure 4's rectangle is a schematic analytical region E≥E*, I≤I* for a source
on the descending inhibitory response. Its drive assumption is ΔB_E≥0 and
ΔB_I≥0 relative to baseline throughout the intervention. Nonnegative total
input alone does not meet that assumption. The phase panel uses the B=(2,1)
baseline and its unique validated orbit; multiple orbits require author
selection. It selects the axis-duration combination with the largest number of
sampled amplitudes that have different destinations across phases, with
deterministic tie-breaking.
It is a measured comparison, not phase-complete control or a probability map.
Phase is elapsed time divided by the validated period, measured from the stored
orbit initial state; zero and one denote the same cycle position.

Figure 5 starts every setting from the original seizure coordinates of its
circuit. Filled black circles retain two switching cycles with original pulses;
open circles require replacement pulses; triangles indicate no demonstrated
switching within the tested protocol; gray crosses indicate missing classified
sources. Successful replacement pulses do not imply equivalence of response
cost, and connecting parameter intervals are not inferred from sampled marks.

## Verification and rendered review

The completed study contains eight detailed circuits at fourteen baselines,
including sixteen starting phases of the validated periodic source. All 99
independent geometry confirmation searches agreed with the corresponding
discovered root sets. This agreement remains bounded numerical evidence.

All 176 independent-equation trajectory comparisons passed. The maximum
coordinate discrepancy was 1.6671e-9. Their source, configuration, selection,
and detailed-stage metadata identities are recorded in the portable packet.
Both the packet and rendered output passed exact file-set and SHA-256 checks.

Validation ran with Julia 1.10.12: the full package suite passed 4,951 assertions;
the final focused study-contract and repair-guard suites passed 28 and 9;
the plotting suite passed 207. The focused suites include checks added after
the full package run; the full suite was not rerun after those additions.

All five complete-data figures were rendered and visually inspected. Revisions
addressed heading widths, symbol meanings, parameter/context labels, the
switching-outcome row, and the state-domain inset. The final inset revision
was inspected again; Figures 2–5 retained identical PNG content. The original
local gallery was `output/figure_redesign_20261006/index.html`. Generated
candidate data and renders are generated into ignored local output and are not
shipped with this checkout. The workflow for a new study, packet, and
PDF/SVG/PNG render is in `reproducibility/figure_redesign_20261006/README.md`.

Remaining scientific limits are explicit in the figures and captions: finite
sampling and follow-up, provisional coordinate roles, unresolved input cells,
and a truncated periodic branch whose onset has not been identified. Final
manuscript selection and biological interpretation remain for author review.
