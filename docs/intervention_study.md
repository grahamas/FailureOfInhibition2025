# Failure-threshold intervention study

## Question and evidence gates

Test whether raising the inhibitory failure threshold suppresses a candidate
pathological attractor while preserving distinct nonquiescent dynamics, and
whether it perturbs those dynamics less than changes to recruitment,
recurrent excitation, or inhibitory output. The ranges below are exploratory
bounds supplied for this study, not calibrated physiological ranges.

The first deliverable is a matched equilibrium-coexistence map. A collection
of stable equilibria does not by itself identify rest, functional activity,
or seizure. Store state coordinates, inputs, response slopes, residuals, and
local spectra before assigning any biological interpretation. An inhibitory
failure candidate must have evidence of occupying the descending response
branch; low inhibition can also reflect weak recruitment. Even a descending
branch can sustain substantial inhibition and must not automatically be
called pathological.

The scientific sequence is:

1. Discover equilibria across both specified parameter planes, with matched
   monotone controls and explicit search coverage. Continue representative
   branches in both directions, retaining unstable equilibria and numerical
   fold/Hopf candidates. Failed discovery does not establish absence.
2. Examine coexistence examples using trajectories and local spectra.
   Distinguish stationary intermediate activity, high E/high I, high E/low I,
   and periodic motion through continuous measurements. Validate proposed
   cycles with a phase condition, nonzero amplitude, closure, refinement,
   and Floquet diagnostics. Numerical validation is not a rigorous proof.
3. Compare each intervention independently at a convincing coexistence point.
   Assess candidate pathological-branch persistence, pulse-transition
   boundaries, and changes to the retained activity branch or cycle. Keep
   active-regime biological interpretation explicit and unresolved until
   justified; removing every nonquiescent attractor is a different outcome.
4. Test the promising neighborhoods with targeted slices and deterministic
   joint samples; refine numerical settings and ambiguous trajectories.
5. Map supported conclusions and limitations to manuscript revision
   instructions. Manuscript edits remain manual.

## Baseline and primary plane

Use `a_E=a_I=5`, `theta_E=1.5`, `theta_on=4`, `tau_E=7.8 ms`,
`tau_I=34.32 ms`, and zero E/I baseline drive. Coupling fields retain the
source-to-target convention of [the model reference](model.md).

| Anchor | `e_to_e` | `i_to_e` | `e_to_i` | `i_to_i` | `theta_off` |
| --- | ---: | ---: | ---: | ---: | ---: |
| Figure 3 | 17 | 9 | 19 at the example | 4 | 8 at the example |
| Figure 4 exploration | 19 | 13 | swept | 6 | swept |

For both anchors sweep `e_to_i=12:0.5:28` and
`theta_off=6:0.25:12`: 825 coordinates per anchor and 3,300 matched-model
cells in total. Figure 4's variable E-to-I coupling is supplied by this
exploration, not reconstructed as a missing caption value. Preserve matched
controls at each coordinate even though their response is independent of
`theta_off`.

The heuristic line `e_to_i = 2 theta_off` follows from `E approximately 0.5`
and small I, which give `u_I approximately e_to_i/2`. It is an interpretive
reference, not an exact bifurcation boundary. Finite slopes, inhibitory
self-coupling, and the actual equilibrium coordinates modify it.

Numerical seeds, tolerances, failures, root deduplication, source snapshots,
and checksums accompany the maps. Root indices are local to each search;
they are not branch identities across parameter changes. Continuation
supplies branch evidence, including unstable portions. Candidate crossings
require refinement and nondegeneracy checks before bifurcation claims.

## Activity and periodic motion

Record E, I, effective inhibitory input `u_I`, inhibitory response value, and
`F_I_prime` alongside local stability. Report time in milliseconds and
spectral rates in inverse milliseconds. The response midpoint identifies its
ascending/descending sides analytically; a floating-point derivative can
underflow, so retain the input and thresholds too.

For pulse simulations observe at least 5,000 ms after withdrawal, extending
unresolved cases to 10,000 and 20,000 ms. A locally attracting equilibrium's
slowest decay rate can motivate longer follow-up, but does not bound global
transients. Two sampled terminal windows compatible with a discovered root
remain finite-window observations. Persistent oscillations alone remain
unresolved until a nonconstant periodic solution passes the numerical checks.

## Intervention comparison

| Changed quantity | Exploration | Interpretation to test |
| --- | --- | --- |
| `theta_off` | 8 to 12 | Resistance to inhibitory failure |
| `e_to_i` | Baseline to 60% | Protection versus inhibitory recruitment |
| `e_to_e` | Baseline to 60% | General excitation reduction |
| `i_to_e` | Baseline to 150% | Stronger output when I may be negligible |

Change only the named quantity per intervention, preserving matched models.
Report E/I shifts on continued activity branches; for validated cycles report
amplitude and period changes. Do not choose a scalar activity-disturbance
weighting or rank interventions by an invented biological threshold.
Report transition intervals for explicitly identified initial and destination
objects, distinguishing induction from low-activity and other active states.

## Rectangular pulse protocol

Positive targets are E alone, I alone, and equal E/I. The broad exploratory
scan uses component amplitudes `0:0.25:8` and the coarse duration grid
`1,2,5,10,20,50,100,200 ms`.
Start at each discovered locally attracting equilibrium and, when available,
multiple phases of numerically validated cycles. Measure the post-withdrawal
destination under the same autonomous parameters. Zero-amplitude cases
provide a within-state control. Equal E/I pulses deliver the stated amplitude
to each component and therefore have twice the absolute integrated input of
single-target pulses of equal amplitude and duration.

Test negative E-drive pulses separately under `AbstractIntervention`. They
are abstract suppressive inputs and do not have the afferent-excitation
interpretation. Report amplitude, duration, signed component integrals, and
the sum of their absolute values separately. The latter is a defined input
cost, not biological energy. Report duration both in milliseconds and as
`duration/tau_E` and `duration/tau_I`; this is model-timescale normalization,
not biological calibration.

Refine every sampled adjacent amplitude interval with differing resolved
destinations; retain unresolved brackets. Do not binary-search under an
assumption of monotonic success. A coarse scan can miss islands within
equal-outcome intervals, so reported boundaries remain conditional on grid
and refinement coverage. A missing transition means none was observed in
the specified targets, amplitudes, durations, starts, phases, and follow-up;
it does not establish global unreachability or permanent rescue.

For claim-specific transition work, first evaluate the coarse duration grid.
Insert an arithmetic midpoint only where adjacent durations differ in the set
of resolved destinations, the number or order of amplitude-boundary outcomes,
transition presence, unresolved presence, or integration-failure presence.
Repeat only for the configured number of refinement levels. Retain the final
adjacent durations as finite protocol-change brackets; do not report an exact
minimum duration. Broad-grid maps remain exploratory even when every sampled
trajectory resolves.

## Secondary robustness

Use targeted slices and deterministic joint sampling, avoiding a full
Cartesian product: `e_to_e` 14–24, `i_to_e` 6–18, `i_to_i` 0–10,
`tau_I/tau_E` 0.5–8 including 4.4, slopes 2.5/5/10 with equal inhibitory
onset/failure slopes, and onset thresholds within 20% of baseline. Changing
the time-constant ratio leaves balance equations and equilibria unchanged,
but changes stability and possible oscillations. Verify this separation and
revisit pulse boundaries at promising samples.

Scientific conclusions require numerical refinement, branch tracking, and
explicit activity interpretation. The workflow can produce negative or
unresolved results without relaxing its criteria to favor the hypothesis.

## Running the study

Run these commands from the repository root. Each output directory must be
absent or empty (the comparison runner requires an absent root). The full
configuration and executed source are archived with numerical artifacts and
SHA-256 checksums. `--smoke` selects a smaller execution check; it does not
replace the full study.

```bash
julia --project=. scripts/run_coexistence_map.jl --config experiments/coexistence.toml --output output/coexistence_study
julia --project=. scripts/run_continuation_experiment.jl --config experiments/coexistence.toml --output output/continuation_study
julia --project=. scripts/run_periodic_candidates.jl --config experiments/coexistence.toml --output output/periodic_candidates
julia --project=. scripts/run_pulse_experiment.jl --config experiments/pulses.toml --output output/pulse_study
julia --project=. scripts/run_intervention_experiment.jl --config experiments/interventions.toml --robustness --output output/intervention_study
julia --project=. scripts/run_pulse_comparison.jl --config experiments/pulses.toml --output output/pulse_comparison
```

The continuation runner starts from every discovered equilibrium at
`(e_to_i,theta_off)=(19,8)` for both anchors and both models, then traces both
axes in both directions. It retains duplicated branch traversals from
different roots; candidate counts must not be treated as distinct
bifurcation counts. A supplied configuration must contain the representative
point inside its bounds.

The periodic-candidate runner uses nine initial states on a 3-by-3 grid in
`[0,0.5]^2` at each baseline anchor and model, plus one initial state per
model at four neighboring points (44 requested trajectories total). It
records the exact allocation and screening policy in metadata. Sampled E
recurrence in a terminal trace supplies only a period/state guess for
shooting. Screening can miss small-amplitude or long-period cycles and
unvisited basins; no candidate found does not establish cycle absence.

The pulse-comparison runner derives seven explicit cases: the baseline,
failure thresholds 10 and 12, 60% E-to-I recruitment, 60% E-to-E excitation,
150% I-to-E output, and the second coupling anchor with E-to-I 19. Repeat
`--case NAME` to run a subset. Every case archives its derived configuration
and has its own local equilibrium IDs. These IDs cannot be compared across
cases without examining state coordinates and continuation evidence.

Primary-map `map.csv` contains one row per matched cell. `equilibria.csv`
retains state/input/response/stability observations; `attempts.csv` and full
context TOML retain rejected and uncertain candidates. Pulse `trials.csv`,
`followups.csv`, `tails.csv`, and `boundaries.csv` retain every sampled
protocol and finite-window result. `duration_refinements.csv` records inserted
midpoints and their topology-change reasons; `duration_brackets.csv` retains
the final finite intervals. When `retain_trajectories=false`, saved
trajectories contain pulse endpoints and terminal diagnostic samples;
they must not be plotted as if the unsampled transient had been observed.

The claim-report figures use the repository's isolated CairoMakie environment.
The renderer verifies consumed artifact checksums and writes three PNGs with
companion TOML provenance without altering the experiment artifacts:

```bash
julia --project=plotting -e 'using Pkg; Pkg.instantiate()'
julia --project=plotting scripts/plot_manuscript_evidence.jl \
  --coexistence output/coexistence_study \
  --output output/study_figures
```

Follow the checksummed source snapshot's replay command to reproduce a
particular artifact. A later working-tree revision may differ from the
revision that generated an earlier run.
