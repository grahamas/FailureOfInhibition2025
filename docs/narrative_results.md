# Narrative study: results and author handoff

## Result

The bounded study found a circuit with reproducible rest–active switching,
separate herald and seizure candidates, and access to both high-activity
states through input pulses. Raising the inhibitory failure threshold preserves
the measured ordinary switching while changing the destination of the former
seizure state to the herald branch. This is a selective change in the modeled
state repertoire; it is not recovery to rest or intermediate activity.

The herald and seizure candidates also differ in the sampled direct E
displacement required to reach an active recovery target. Neither recovers
under withdrawal of the selected example's small available tonic E input.
Thus the basin comparison distinguishes them, while a differential afferent
withdrawal requirement remains unestablished here.

## Evidence package

The final run is `output/narrative_final_20260928`. It completed **2,103 cells**:
1,024 joint samples per coupling family and 55 explicit anchor/input contexts.
Twelve cells passed the structural screen for rest, intermediate active, and
seizure roles. Seven passed the input-driven switching and induction test.
The predeclared expansion to 4,096 samples per family was therefore not used.
These counts describe this design, not parameter prevalence or completeness.

The run retains 27,024 qualification pulse trials, 81,138 intervention pulse
trials, 340 selected-source reduction trials, 44 intervention settings,
36 local input/coupling map cells, and seven in-bounds parameter neighbors.
Every selected reduction trial was equilibrium-compatible under its recorded
finite-window criterion. Raw files preserve other contexts and statuses.

The figure/report directory is `output/narrative_figures_20260928`:

- `report.html`: standalone illustrated report.
- `01_mechanism`: matched models, response functions, and equilibrium geometry.
- `02_transitions`: actual switching and high-state induction trajectories.
- `03_reduction`: input withdrawal and direct state displacement, kept separate.
- `04_interventions`: switching preservation and seizure-state persistence.
- `05_local_map`: nearby fixed-protocol outcomes and continued branch segments.

Each figure has PNG and PDF versions. The [protocol](narrative_study.md)
documents the runner, sampling rules, interpretation, and replay commands.

## Selected example and transitions

The selected case is `figure4_anchor2_B2`, with couplings
`(e_to_e,i_to_e,e_to_i,i_to_i)=(19,13,19,6)`, failure threshold 8,
`B_E=0.015625`, `B_I=0`, and `tau_I/tau_E=0.2`.
The remaining response parameters retain the approved conventions.
This fast inhibitory time scale is exploratory; it differs from the original
manuscript ratio 4.4 and is not physiological calibration.

| Provisional role | E | I |
| --- | ---: | ---: |
| Rest | 0.000634369 | 2.18919e-9 |
| Intermediate active | 0.331099365 | 0.370040060 |
| Herald | 0.499871742 | 0.499861934 |
| Seizure | 0.500000000 | 0.000561854 |

Seven equilibria were discovered, four locally attracting. The count has not
been certified exhaustive for this nonzero-input context.

Representative successful sampled stimuli, added to the common background:

| Transition | Target | Amplitude | Duration (ms) |
| --- | --- | ---: | ---: |
| Rest → active | E | 0.40625 | 100 |
| Active → rest | I | 2.125 | 100 |
| Rest → herald | E | 0.46875 | 50 |
| Active → herald | E | 0.46875 | 10 |
| Active → seizure | I | 3.875 | 10 |

Rest → active → rest was repeated twice using actual terminal states, restoring
the original tonic drive after every pulse. Both 21×21 and 41×41 confirmations
with tighter tolerances reproduced switching and seizure induction. These
stimuli are witnesses, not exact minimal thresholds. In this selected example,
the demonstrated seizure induction uses **positive I input**; it should not be
described as an E-only seizure trigger or a physiologically modest stimulus.

## Herald versus seizure

At the same parameters and tonic input, direct reduction of E at fixed I and
fixed tonic input produced these adjacent outcome brackets:

| Initial role | Lower ΔE | Upper ΔE | Endpoint destinations |
| --- | ---: | ---: | --- |
| Herald | 0.067701135 | 0.067731645 | Herald / active |
| Seizure | 0.131774902 | 0.131805420 | Herald / active |

The smallest observed successful reductions were approximately 0.06773 and
0.13181, respectively. The latter is about 1.95 times the former, for this
specified displacement direction, recovery target, and sample resolution.
The lower endpoint for the seizure-source bracket already reaches herald:
leaving the seizure basin is easier than reaching the active recovery target.
This is why a generic basin-exit distance would answer a different question.

The entire permitted tonic withdrawal interval, `0 ≤ ΔB_E ≤ 0.015625`, was
sampled with 65 observations per source. All herald-source trials returned
to the herald role and all seizure-source trials to the seizure role.
This bounded negative result does not establish unrestricted unrescuability
or an ordering of the required input reductions. Input withdrawal and direct
activity displacement have different units and change different parts of the
dynamics. The direct-displacement ratio is not an input-current ratio.

Both comparisons were also run from actual induced endpoints, with controls
that retained the original input. No trajectory was snapped to an equilibrium
at the intervention switch. The point model still produces stable herald
candidates, not self-terminating herald spikes.

## Intervention tradeoffs

Increasing the failure threshold from 8 to 8.5 retains the seizure-source
destination. At sampled thresholds 8.75 through 12, the former seizure source
instead reaches the herald branch, while rest–active switching remains
demonstrable. The sampled switching witnesses remain E=0.40625 for 100 ms
and I=2.125 for 100 ms. At threshold 8.75 the intermediate equilibrium changes
by less than `4e-10` in either coordinate; this observed small change is not an
exact independence theorem.

Alternative interventions show why preserving activity must be measured:

| Intervention example | Rest–active switching | Former seizure source |
| --- | --- | --- |
| Failure threshold 8.75 | Retained | Herald |
| Recruitment `e_to_i=17.1` | Not demonstrated; active branch unmatched | Herald |
| Recurrent excitation `e_to_e=17.1` | Retained, with changed activity and stimuli | Seizure |
| Inhibitory output `i_to_e=17.0625` | Retained, with changed activity and stimuli | Seizure |

These are comparisons of particular parameter changes, not equally dosed
drugs, optimized interventions, or a universal ranking. In particular, the
first row does not show seizure-to-rest recovery: high excitation persists
at the herald destination.

Seven neighboring parameter settings were checked with the original selected
stimuli and separate reduction scans. Some retain switching but lose the
specific induction witness; others lose the specific switching witness while
retaining the states. This supports sensitivity of transitions to parameters,
not a broad claim of robust functional control. A failed fixed stimulus does
not exclude another successful stimulus.

## Manuscript narrative outline — for manual author revision

1. **Opening question and scope.** Ask how an activity-supporting circuit can
   also enter persistent high-E/low-I activity, and how intervention changes
   both capabilities. State that rest–active switching is the operational
   property tested. Do not equate it with demonstrated cortical computation.
2. **Mechanism and mathematical restriction (Figure 1).** Give the fire-then-fail
   population construction concisely, then the ordering result for two
   equilibria of one model under common constant input and arbitrary allowed
   coupling. Avoid the manuscript's unit-connectivity reduction and its broad
   heading that WCM cannot model a focal core. Generic seizures, cycles, and
   different-input comparisons are outside this theorem.
3. **Coexistence becomes behavior (Figure 2).** Show the four coordinate roles,
   the repeated ordinary switch, and the actual routes to herald and seizure.
   Explain the fast inhibitory time scale and the positive-I seizure trigger.
   This supplies evidence beyond merely counting equilibria.
4. **Persistence and control are conditional (Figure 3; local map supplement).**
   Retain the earlier input-dependent recovery example as a contrast, while
   keeping its parameter setting separate. Explain why existence, induction,
   sustained-input recovery, and basin displacement require different tests.
   The paired displacement result keeps herald and seizure distinct without
   claiming that feedback-mediated herald termination has been reproduced.
5. **Selective change with a remaining control problem (Figure 4).** Lead with
   preservation of ordinary switching under threshold change. Explicitly show
   that the former seizure trajectory reaches herald, leaving a high-activity
   state that still needs control. Compare the measured alternatives without
   claiming clinical superiority or a scalar functional cost.
6. **Discussion and next mechanism.** Conclude with the supported coexistence,
   switching, and intervention tradeoffs. Treat a coupled E–I feedback model as
   the next test of whether available input withdrawal can terminate a herald
   excursion while failing to control seizure. The present displacement result
   motivates that question but does not supply the feedback mechanism.

The manuscript already cites [Kim and Nykamp (2017)](https://doi.org/10.1007/s10827-017-0647-7).
Their Figures 3–4 already include recovery and failure-threshold manipulations,
and Figures 5–6 analyze oscillatory transitions and bifurcation structure.
Position the present contribution around the ordering restriction, measured
rest–active switching alongside additional states, and explicit intervention
tradeoffs. A nonmonotone response, a recovery example, or a threshold shift
alone is not the new contribution. This comparison is not a comprehensive
literature novelty certification.

Keep synthetic fitting superiority, independently variable sigmoid slopes,
herald-spike identification, spontaneous termination, and spatial/core/EZ
conclusions out of the demonstrated-results argument unless separately
supported. The supplied manuscript remains unchanged.

## Verification

- Full Julia suite: **4,746 / 4,746 assertions passed**. Expected iteration-limit
  warnings came from existing failure-path tests.
- An independent SciPy implementation using a 41×41 root search reproduced all
  seven selected roots within `4.72e-15` maximum coordinate difference and
  independently checked 12 transition endpoints with DOP853 integration.
- Archived Julia source reproduced **12 trajectory CSVs byte-for-byte**,
  covering repeated switching and both selected reduction protocols.
- Source/configuration identity and per-unit manifests guard resumptions.
  SHA-256 verification passed for 131,920 scientific files, 39 renderer files,
  and 25 independent-validation files.

Use the archived source for exact replay. A subsequent docstring correction
clarifies that role correspondence uses nearest-coordinate matching with tie
and collision rejection; it does not change the executed algorithm. Strict
source-identity checks intentionally reject resuming the archive with that
changed working-copy file.

Independent validation scripts and reports are retained in
`output/narrative_validation_20260928`. Numerical discovery, agreement between
solvers, and finite follow-up do not certify global attractor completeness,
biological roles, parameter realism, or manuscript claims.
