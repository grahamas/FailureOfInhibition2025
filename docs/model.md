# Point-Model Reference

This document describes the assumptions, equations, and numerical conventions
implemented by the package. It records current behavior rather than the
history of how the model was selected.

## Scope and state

The package implements a deterministic, CPU-only, two-population point model.
The state is ordered `[E, I]`, where `E(t)` and `I(t)` are dimensionless,
temporally coarse-grained mean fractions of the modeled excitatory and
inhibitory populations in the active state:

```math
E(t) \in [0,1], \qquad I(t) \in [0,1].
```

Zero denotes no modeled neurons active and one denotes full occupancy of the
modeled active-state capacity. The states are not dimensional firing rates;
comparison with measured neural signals requires a separate observation or
scaling model.

The square `[0,1]^2` is the physical state domain. Low-level routines may
evaluate the vector field outside it for solver stages and mathematical
diagnostics, but those states are not physically interpreted.

## Governing equations

The implemented normalized occupancy equations are

```math
\tau_E \dot E = -E + (1-E)F_E(u_E),
```

```math
\tau_I \dot I = -I + (1-I)F_I(u_I).
```

The factors `(1-E)` and `(1-I)` are the inactive fractions available for
recruitment. There are no independent decay or saturation parameters, and the
right-hand side never clips or projects the state.

Because the supported responses are bounded by one, an equilibrium satisfies

```math
X^* = \frac{F_X(u_X^*)}{1+F_X(u_X^*)} \leq \frac12.
```

## Parameters and coupling

Time and both time constants are measured in milliseconds. The time constants
must be finite and strictly positive. Inputs, response thresholds, external
drives, and coupling-induced input changes are dimensionless; a response slope
has reciprocal effective-input units.

The total inputs are

```math
u_E(t) = J_{E\leftarrow E}E(t) - J_{E\leftarrow I}I(t) + P_E(t),
```

```math
u_I(t) = J_{I\leftarrow E}E(t) - J_{I\leftarrow I}I(t) + P_I(t).
```

Every coupling is represented as a finite, nonnegative magnitude. Excitatory
source terms enter positively and inhibitory source terms enter negatively.
The code preserves `source_to_target` names:

| Field | Mathematical magnitude | Contribution |
| --- | --- | --- |
| `e_to_e` | `J_{E<-E}` | `+e_to_e * E` in `u_E` |
| `i_to_e` | `J_{E<-I}` | `-i_to_e * I` in `u_E` |
| `e_to_i` | `J_{I<-E}` | `+e_to_i * E` in `u_I` |
| `i_to_i` | `J_{I<-I}` | `-i_to_i * I` in `u_I` |

Coupling magnitudes are meaningful only relative to the thresholds, slopes,
and drives in the same input parameterization.

## Population responses

The excitatory response and monotone inhibitory-control response are standard
logistics on the real input line:

```math
F(u) = \frac{1}{1+\exp[-a(u-\theta)]}, \qquad a>0.
```

The response is analytic, strictly increasing, and evaluated with an
overflow-safe implementation. At the threshold, `F(theta)=1/2` and
`F'(theta)=a/4`. At zero effective input the response is generally positive,
so a quiescent state need not be the exact origin.

The failure-of-inhibition response is the raw equal-slope difference

```math
F_I^{\mathrm{FoI}}(u)
=\sigma\!\left(a_I(u-\theta_{\mathrm{on}})\right)
-\sigma\!\left(a_I(u-\theta_{\mathrm{off}})\right),
```

with finite parameters satisfying

```math
a_I>0, \qquad \theta_{\mathrm{on}}<\theta_{\mathrm{off}}.
```

It is not normalized, rectified, or clamped. With
`Delta_I = theta_off - theta_on`, its range and maximum are

```math
0 < F_I^{\mathrm{FoI}}(u)
\leq \tanh\!\left(\frac{a_I\Delta_I}{4}\right) < 1,
```

with the maximum at the midpoint of the thresholds. Its derivative is

```math
{F_I^{\mathrm{FoI}}}'(u)
=a_I F_I^{\mathrm{FoI}}(u)
\left[1-\sigma_{\mathrm{on}}(u)-\sigma_{\mathrm{off}}(u)\right].
```

The implementation evaluates the response and derivative in overflow- and
cancellation-safe forms. Extreme finite inputs may underflow numerically to
zero, but there is no modeled finite-input cutoff.

`matched_point_models` constructs control and failure-of-inhibition models
that share the excitatory population, time constants, couplings, drive, and
inhibitory onset parameters. They differ only by the inhibitory failure term.

## External drive

For each population, the external drive is a baseline plus additive pulses:

```math
P_X(t) = B_X + \sum_{k=1}^{K} \Delta P_{X,k}
\mathbf{1}_{[t_k^{\mathrm{on}},\,t_k^{\mathrm{off}})}(t).
```

Pulse intervals are half-open, have finite bounds, and require onset before
offset. Overlapping pulses sum componentwise. `NoDrive()` is the zero-baseline,
no-pulse case.

`AfferentExcitation` requires both total drive components to remain
nonnegative on every constant segment. `AbstractIntervention` permits signed
totals. Pulse schedules are stored immutably after validation.

Every in-span onset and offset is supplied to the solver as a discontinuity
and included in the saved output times.

## Analytical Jacobian

At a fixed time, the state Jacobian is

```math
J(E,I,t)=
\begin{pmatrix}
\dfrac{-1-F_E+(1-E)F_E'J_{E\leftarrow E}}{\tau_E}
&
\dfrac{-(1-E)F_E'J_{E\leftarrow I}}{\tau_E}
\\[1.1em]
\dfrac{(1-I)F_I'J_{I\leftarrow E}}{\tau_I}
&
\dfrac{-1-F_I-(1-I)F_I'J_{I\leftarrow I}}{\tau_I}
\end{pmatrix}.
```

Piecewise-constant drive contributes no state derivative, although the vector
field and Jacobian can jump in time at a drive transition.

The balance and ODE kernels share scalar calculations. The ODE kernels divide
each balance or Jacobian entry by its time constant before assigning it to
the caller's output array. In particular, a lower-precision output array is
not used to store an unscaled intermediate that could underflow or overflow
even when the final scaled value is representable.

## Domain handling

The response bounds make `[0,1]^2` forward invariant. On the lower boundaries,

```math
\tau_E\dot E\big|_{E=0}=F_E(u_E)>0,
\qquad
\tau_I\dot I\big|_{I=0}=F_I(u_I)>0,
```

and on the upper boundaries,

```math
\tau_E\dot E\big|_{E=1}=-1,
\qquad
\tau_I\dot I\big|_{I=1}=-1.
```

`solve_point_model` rejects nonfinite initial states and initial coordinates
outside `[0,1]`. Before constructing the ODE, it materializes the state as a
standard one-based vector and converts integral coordinates to floating-point
values while preserving existing floating-point types such as `BigFloat`. It
uses an absolute numerical-domain tolerance, defaulting to `domain_atol=1e-8`,
and rejects accepted steps or returned states outside `[-domain_atol,
1+domain_atol]`. States are never clipped or projected.

The response maxima also imply sharper forward-invariant rectangles. The
control model has `[0,1/2]^2`. For the failure-of-inhibition model, let

```math
M_I=\tanh\!\left(
\frac{a_I(\theta_{\mathrm{off}}-\theta_{\mathrm{on}})}{4}
\right);
```

then the sharper rectangle is

```math
[0,1/2]\times\left[0,\frac{M_I}{1+M_I}\right].
```

These sharper bounds do not replace `[0,1]^2` as the physical state domain.

## Equilibrium discovery

Equilibrium analysis uses the dimensionless population-balance residual

```math
g(E,I)=
\begin{pmatrix}
-E+(1-E)F_E(u_E)\\
-I+(1-I)F_I(u_I)
\end{pmatrix}.
```

Its analytical Jacobian is evaluated directly by
`point_balance_jacobian!`. If `Df` is the Jacobian of the original ODE, then

```math
Dg=\operatorname{diag}(\tau_E,\tau_I)Df.
```

The implementation does not reconstruct `g` or `Dg` by multiplying an ODE
derivative by possibly extreme time constants. `solve_equilibrium` and
`find_equilibria` use `SimpleTrustRegion` with the analytical `Dg` and explicit
iteration limits. Candidate acceptance uses a separately evaluated `g`, not
the solver status or a small dimensional ODE derivative. Solver status and
validation are both retained.

`NoDrive()` and a `PiecewiseConstantDrive` with no pulses are autonomous. A
model with any pulse requires an explicit finite `snapshot_time`. The package
evaluates the original drive at that time, including half-open endpoints and
overlapping increments, and constructs an autonomous copy that preserves the
population and coupling objects. Results record the frozen E/I inputs and the
source time. These are equilibria of the frozen system, not equilibria of the
full driven protocol.

The default multistart coverage is a deterministic 5-by-5 tensor grid over
the sharper equilibrium rectangle above, including its edges and interior.
Callers may supply any finite real trial seeds, including points outside the
physical domain. Seeds are copied into fresh one-based floating vectors and
are never mutated or clipped. End-to-end equilibrium and stability analysis
supports `Float32` and `Float64`; integral inputs promote to `Float64`, while
unsupported arbitrary-precision analysis is rejected rather than narrowed.

The default numerical policies are:

| Option | Default | Scale or units |
| --- | ---: | --- |
| `solver_abstol` | `1e-12` | dimensionless balance residual |
| `solver_reltol` | `1e-10` | dimensionless |
| `residual_atol` | `1e-9` | dimensionless balance residual |
| `domain_atol` | `1e-8` | activity coordinate |
| `dedup_atol` | `1e-7` | activity coordinate, infinity distance |
| `singular_atol` | `1e-10` | dimensionless balance-Jacobian singular value |
| `singular_rtol` | `1e-8` | dimensionless |
| `maxiters` | `100` | nonlinear iterations |

A finite candidate with residual infinity norm at most `residual_atol` is
checked against both `[0,1]^2` and the sharper equilibrium rectangle. A point
inside both closed rectangles is admissible. A point outside a boundary by no
more than `domain_atol` is retained as boundary-ambiguous. A larger excursion,
a large or nonfinite residual, or nonfinite diagnostics is rejected. Raw
coordinates and reasons are retained in every case.

The balance Jacobian is flagged near-singular when

```math
\sigma_{\min}(Dg)
\leq \texttt{singular_atol}
+\texttt{singular_rtol}\,\sigma_{\max}(Dg).
```

A small residual near a singular Jacobian does not certify coordinate
accuracy. Admissible candidates within `dedup_atol` are grouped by coordinate
infinity distance. The representative is an actual candidate selected by
residual norm, coordinates, and seed; roots are never averaged. A connected
tolerance chain whose full diameter exceeds `dedup_atol` is retained as an
unresolved-nearby diagnostic rather than collapsed to one candidate. Its
candidate outputs are partitioned deterministically into complete-linkage
subgroups, within each of which every pair is within `dedup_atol`; this still
collapses repeated attempts while preserving the ambiguous original component
for inspection. Representatives are sorted by `(E,I)`, so their order is
invariant to seed order.

Every search reports `completeness = CompletenessNotCertified` and retains all
seeds, solver outcomes, validation failures, and duplicate memberships. An
empty result is unresolved discovery, not evidence that no equilibrium
exists.

## Local linear stability

For each admissible candidate, local stability is evaluated with
`point_jacobian!` using the original time constants. Results retain the
Jacobian, eigenvalues in `ms^-1`, trace in `ms^-1`, determinant in `ms^-2`,
spectral abscissa in `ms^-1`, and the tolerance applied to each eigenvalue.

With defaults `spectral_atol=1e-10 ms^-1` and
`spectral_rtol=1e-8`, the real part of each eigenvalue is resolved only when

```math
|\operatorname{Re}\lambda|>
\texttt{spectral_atol}+
\texttt{spectral_rtol}|\lambda|.
```

Two resolved negative real parts are `Attracting`, two positive real parts
are `Repelling`, and one of each is a `Saddle`. Any near-zero or poorly
resolved real part produces `StabilityUnresolved`. Spectral geometry is
reported separately as real-distinct, real-repeated, or complex-conjugate;
repeated eigenvalues alone do not make stability unresolved. For geometry,
imaginary parts no larger than the maximum per-eigenvalue tolerance are
treated as numerically real. The resulting real parts are repeated when their
separation is no larger than that same tolerance and distinct otherwise.

These are local linear classifications. A single spectrum does not establish
global stability, a center, a Hopf bifurcation, a limit cycle, or a scientific
regime.

## Sampled trajectory diagnostics

`diagnose_trajectory(solution, model; equilibria, options=DiagnosticOptions())`
compares a trajectory with an `EquilibriumSearchResult` from the same
population and coupling parameters. It reports finite-window observations,
separately from the equilibrium search's local stability classifications.
Neither result assigns a biological regime.

The two consecutive terminal windows each have duration `window_duration`.
Both windows are closed, so their shared endpoint is counted in both. All
three window endpoints must be saved explicitly, and each window must contain
at least `min_samples` samples. Diagnostics retain E/I arithmetic sample means
(not time averages) and ranges,
maximum dimensionless balance-residual norms, and the maximum coordinate
infinity distance across both windows to each discovered equilibrium.

| Option | Default | Meaning |
| --- | ---: | --- |
| `window_duration` | `5.0` | duration of each window in ms |
| `coordinate_atol` | `1e-6` | coordinate-distance and within-window range tolerance |
| `balance_atol` | `1e-8` | dimensionless balance-residual tolerance |
| `min_samples` | `3` | minimum saved samples per window |

`EquilibriumCompatible` means that integration succeeded, the two windows
have valid coverage and physical states up to the equilibrium search's
`domain_atol`, and the drive is constant throughout
both windows and agrees with the equilibrium search's frozen input. Every
sample must be within `coordinate_atol` of the same unique admissible
equilibrium; both coordinate ranges and both maximum balance residuals must
also satisfy their respective tolerances. Equality at a tolerance is accepted.
The matched equilibrium must not be flagged near-singular or belong to an
unresolved-nearby group. Drive changes are checked at pulse transitions,
including transitions between saved samples and at the final endpoint.

This label describes compatibility with an equilibrium over the saved
windows. A stationary trajectory at a saddle can satisfy it; the saddle's
local stability remains a separate result. It does not establish asymptotic
convergence, attraction, global stability, or search completeness. A small
dimensional ODE derivative alone is insufficient because large time constants
can hide a large balance residual.

All unmet criteria produce `TrajectoryUnresolved` with explicit reasons.
Missing metrics are represented by `NaN`, and no unique coordinate match is
represented by `nothing`. A unique match is retained even if other criteria
leave the classification unresolved. Failed or incomplete integration, insufficient
coverage, nonfinite samples, changing drive, ambiguous matches, and numerical
uncertainty are retained rather than converted into a regime label. Invalid
API arguments and mismatched population/coupling parameters raise
`ArgumentError`. Oscillatory appearance does not validate a periodic orbit;
periodic-orbit computation and biological classification remain separate work.
When the solution includes its requested integration interval, the last saved
time must reach that interval's endpoint. For a supplied sample record without
that interval, completion is assessed only over its saved span.
When the solution records `PointModelParameters` in its ODE problem, those
parameters and the drive totals throughout the integration interval must agree
with the supplied model. Sample records without a recorded model rely on the
caller's explicit model context.

## Synthetic experiment records

`experiments/minimal.toml` defines a synthetic matched-model workflow check.
It uses the README example's population parameters, coupling, initial state,
and pulses, plus an otherwise identical zero-drive baseline. The control and
failure-of-inhibition conditions differ only in the inhibitory failure term.

The standard variant integrates from 0 to 20 ms with a 0.1-ms saved interval
and ODE absolute/relative tolerances of `1e-10`. The refined variant retains
the horizon, halves the saved interval, and tightens both tolerances to
`1e-12`. The extended variant retains the standard numerical settings and
integrates to 40 ms. The domain tolerance remains `1e-8`. These 12 cases are
workflow checks, not biological regime tests. The refined variant changes
sampling and integration tolerances together; it does not isolate their
individual effects.

Equilibrium searches use the explicit configured equilibrium and stability
options and record their deterministic seeds. The pulsed protocol has segment
source times 0, 2, 4, 5, and 7 ms. Repeated baseline inputs at different segment
times retain separate context records. Searches are shared across trajectory
variants with identical model and drive contexts. Terminal diagnostics use
the corresponding final constant-drive equilibrium context.

| Artifact | Contents |
| --- | --- |
| `config.toml` | complete experiment configuration |
| `cases.csv` | per-case execution and trajectory classification |
| `comparisons.csv` | classification changes under refinement and horizon extension |
| `trajectories/` | one `time,E,I` CSV per returned trajectory |
| `attempts.csv`, `equilibria.csv` | equilibrium discovery summaries |
| `contexts/` | full search records, including all attempts, raw residuals, Jacobians, spectra, memberships, ambiguity, tolerances, and completeness |
| `diagnostics/` | raw per-case window metrics, criteria, and unresolved reasons |
| `metadata.toml`, `source/`, `checksums.toml` | execution environment, revision/local state, working source snapshot, and artifact hashes |

Numerical execution failures are retained and make the command fail; an
unresolved scientific or finite-window classification is an ordinary recorded
outcome. Successful execution does not require a desired difference between
the matched models. Repeatability is assessed with repeated runs from the
same configuration and environment, and conclusions remain limited to the
reported numerical checks and finite observation horizons.

## Equilibrium continuation

`continue_equilibria(factory, state, parameter; parameter_bounds, options)`
traces an equilibrium branch in both initial parameter directions using
`ContinuationOptions`. The factory returns an autonomous point model with a
consistent concrete type. Correctors solve the two balance equations plus a
pseudo-arclength phase plane, allowing traversal through parameter folds and
unstable segments. Every accepted state is independently checked with the
equilibrium acceptance policy, and stability uses the original-time Jacobian.

Arclength uses scaled coordinates `(E/s_E,I/s_I,p/s_p)` with default scales
one. These are numerical solver scales, not new model normalizations. The
state Jacobian is analytical; the parameter derivative uses a finite
difference with default increment `1e-5 * max(1,abs(p))`, reduced to stay
inside the supplied bounds. Differences are centered in the interior and
one-sided at a bound. Default arclength steps start at `0.03`, range from
`1e-5` to `0.1`, and permit 500 accepted steps per direction. Correctors have
12 iterations and dimensionless/scaled residual tolerance `1e-10`.

Results retain the initial solve, actual model at every accepted point,
tangents, failed correctors and retries, termination statuses, and
`CompletenessNotCertified`. Correction failures halve the step. Parameter
boundary termination does not guarantee an exact endpoint. Step limits and
minimum-step failures leave partial branches. A tangent-parameter sign
change is a fold candidate; a trace sign change with positive determinants
and resolved complex spectra at both endpoints is a Hopf candidate. These
are adjacent-point screening brackets, not located or certified
bifurcations. Branch switching and nondegeneracy verification are not
performed automatically.

## Finite tetrastability search

`scripts/run_tetrastability_search.jl` executes the protocol in
`experiments/tetrastability.toml`. It searches for at least four distinct
discovered locally attracting equilibria in one autonomous failure-of-
inhibition model. The matched monotone control is evaluated and retained at
every parameter cell, but it is not required to have fewer attractors.

The protocol fixes zero drive, `tau_E = 7.8` ms, `tau_I/tau_E = 4.4`, response
slopes `a_E = a_I = 5`, `theta_E = 1.5`, and `theta_on = 4`. It first repeats
the supplied Figure 3 and Figure 4 exploration planes over
`e_to_i = 12:0.5:28` and `theta_off = 6:0.25:12`. It then evaluates an
unscrambled, one-indexed Halton sequence over `(e_to_e, i_to_e, e_to_i,
i_to_i, theta_off)` with bases `(2,3,5,7,11)` and bounds `[14,24]`, `[6,18]`,
`[12,28]`, `[0,10]`, and `[6,12]`. Indices 1--1024 form the initial batch. If
the planes and initial batch contain no screen-positive FoI cell, indices
1025--4096 are evaluated as one extension batch.

Discovery uses the union of the default 5-by-5 equilibrium seeds and an
11-by-11 grid over the same sharper rectangle. A screen-positive cell has at
least four discovered FoI equilibria classified locally attracting. Only that
FoI attracting count contributes to the threshold; total roots,
matched-control counts, and unresolved stability do not, and a failed FoI
search cannot qualify. Each screen-positive cell is rerun for both conditions
with 21-by-21 and 41-by-41 grids and tenfold tighter solver, residual,
near-singularity, and spectral tolerances. The matched-control confirmation
searches must complete. FoI confirmation requires at least four one-to-one
coordinate tracks within `1e-6`, pairwise separation greater than
`100 * dedup_atol = 1e-5` at all three grid densities, recomputed residuals and
Jacobians within the declared tolerances, spectral abscissa at most `-1e-8`,
and no near-singular or unresolved-nearby flags. The recomputation uses the
same package balance and Jacobian kernels, so it can reveal inconsistent
artifacts but not a shared formula defect.

Every confirmed FoI root is continued separately along each of the four
couplings and `theta_off`, with the other parameters frozen at the candidate
cell. Continuation starts at step `0.01`, permits steps from `1e-5` through
`0.03`, and retains both directions, partial branches, corrector failures, and
fold or Hopf screening brackets. The search streams every declared cell,
matched-condition result, nonlinear attempt, equilibrium, failure,
confirmation match, and continuation record, and archives its configuration,
source snapshot, environment, and checksums. A completed run with no execution
failures and no sampled cell meeting the confirmation rule is a successful
finite negative result. Partial artifacts are diagnostic only and are not
resumable; an interrupted run must restart in a new empty output directory.
Neither discovery nor confirmation certifies completeness, an exact attractor
count, general prevalence, biological states, or a publication claim.

## Numerical periodic-orbit shooting

`solve_periodic_orbit(model, state_guess, period_guess; options)` uses a
phase-conditioned shooting solve for an autonomous point model, with the
variational equations for the monodromy matrix. `PeriodicOrbitOptions`
controls positive period bounds, ODE/shooting tolerances, sample count,
nonconstant amplitude, phase degeneracy, and Floquet checks. Shooting uses
Float64 explicitly; unsupported arbitrary precision is rejected rather than
silently narrowed. A callback overload supports autonomous two-dimensional
systems for mathematical verification; the point-model wrapper enforces its
physical domain and model context.

The phase condition is orthogonality to the vector field at the supplied
reference state. A near-zero reference speed is unresolved. Integration uses
normalized time from zero to one and is repeated with tenfold tighter ODE
tolerances by default. Numerical validation requires nonconstant sampled
amplitude, closure and phase residuals, agreement of the waveform, period,
and monodromy under refinement, and a trivial Floquet multiplier near one.
An ODE defect compares the dense interpolant derivative with `period * rhs`
at interior sample points. The default ODE tolerances are `1e-10`, shooting
tolerance `1e-9`, validation tolerance `1e-6`, amplitude threshold `1e-5`,
Floquet tolerance `1e-4`, and sample count 257. These are explicit numerical
policies, not physiological criteria or rigorous error bounds.

`NumericallyValidatedPeriodicOrbit` reports this evidence, not a rigorous
existence certificate, primitive-period proof, or exhaustive cycle search.
Orbital stability is separate: the transverse multiplier gives
`PeriodicOrbitAttracting` or `PeriodicOrbitRepelling` only when separated from
the unit circle by its tolerance. Neutral or insufficient evidence stays
unresolved. `periodic_orbit_phases(result, phases)` samples validated results
at fractional phases in `[0,1)`; validation alone does not imply attraction.

`continue_periodic_orbit(model_at_parameter, state, period, parameter;
parameter_bounds, options, periodic_options)` follows one validated periodic
branch in both initial parameter directions. A callback overload accepts
autonomous planar `rhs!`, `jacobian!`, and `parameters_at_parameter` functions.
The pseudo-arclength corrector uses scaled
`[state[1], state[2], log(period), parameter]` coordinates, so it can traverse
parameter reversals. Every accepted point is independently re-shot with the
periodic-orbit validation policy; failed correctors, retries, tangent
reversals, boundary stops, and unresolved termination states are retained.
Attempts retain separate corrector and post-shoot residuals. Acceptance is
gated again after independent shooting against the original phase and
arclength equations, so a stored orbit cannot inherit stale residuals from
its pre-shoot candidate.

Generic helpers report coordinate half-ranges, Euclidean distances from a
caller-supplied center, signed phase-plane area, winding about a supplied
center, a finite integer-divisor primitive-period screen, phase-invariant
sample equivalence, and a sampled planar divergence/Floquet cross-check.
Phase equivalence refines the best discrete offset continuously. Winding uses
per-segment angular-rate resolution and requires agreement after doubling its
sampling grid; it stays unresolved if an adaptive finite cap is exhausted.
These observables do not classify branch endpoints or certify a primitive period.
In particular, parameter reversal alone is not a certified fold of cycles,
and runner-level Hopf, fold, or homoclinic interpretation remains separate.

## Explicit pulse experiments

`run_pulse_trial` and `run_pulse_experiments` apply rectangular pulses to a
supplied autonomous model and matching equilibrium-search context. Targets
are `:E`, `:I`, `:equal`, and separate `:negative_E` suppression. Pulse onset
is zero, withdrawal occurs at the specified duration, and baseline drive
resumes afterward. Positive targets use the afferent interpretation when the
baseline allows it; negative E-drive uses `AbstractIntervention`.

The API defaults are amplitudes `0:0.25:8`, discovery-grid durations
`[1,2,5,10,20,50,100,200]` ms, no duration refinement, and post-withdrawal
follow-ups of `[5000,10000,20000]` ms. The tracked study configuration requests
one duration-refinement level. These values define numerical coverage; they do
not assert that every duration is scientifically informative. An unresolved
observation is rerun from the same
initial state and pulse with the longer horizon. Each terminal diagnostic
window defaults to 100 ms with 21 samples and exact window endpoints;
coordinate and balance tolerances remain `1e-6` and `1e-8`. Successful
integration and compatibility with a uniquely matched locally attracting
equilibrium are required for a resolved equilibrium destination. This remains
a finite-window observation and is not permanent rescue or global reachability.
Stationary saddles, near-singular or ambiguous matches, failed integrations,
and unvalidated oscillations cannot become attracting destinations.

Default starts use discovered locally attracting equilibria. Explicit
`(id,state,provenance)` records also allow caller-verified cycle phases;
the caller must match their autonomous model context and check orbital
stability. The current pulse destination classifier recognizes equilibria;
other outcomes remain unresolved even if a separate periodic orbit has been
validated. Each attempted follow-up retains samples, diagnostic metrics,
continuous E/I/input/slope summaries, and execution errors. Optional denser
trajectory retention preserves the `time,E,I` CSV convention.

Refinement bisects every adjacent sampled amplitude interval with different
resolved equilibrium destinations, with two refinement levels by default.
Unresolved brackets are retained, and no monotonic-success assumption is
made. Optional duration refinement compares adjacent durations by resolved
destination set, ordered amplitude-boundary topology, unresolved presence, and
integration-failure presence;
it ignores the numerical locations of amplitude boundaries. Each requested
level inserts arithmetic duration midpoints only where that topology differs.
The remaining intervals are finite duration brackets, not exact minimum
durations. Islands between equal-outcome sampled endpoints can be missed.
Artifacts report duration in milliseconds and normalized by both population
timescales. The component integrals are signed `duration * increment`; the sum
of their absolute values is a defined input cost, not biological energy. A
resolved implementation run does not by itself change a manuscript-claim
disposition; see the claim-specific protocol in the intervention study.

## Figure-5b topology and local Hopf candidates

`classify_figure5b_topology` compares the independently generated 11×11,
21×21, and 41×41 equilibrium-search refinements. The schedule is fixed and
each search must contain the exact deterministic union of the default seeds
and the declared sharper grid. It requires seven uniquely coordinate-matched
roots at every refinement, with three locally attracting equilibria, three
saddles, and one repeller. Every root is checked against a recomputed balance
residual, balance Jacobian, and original-time spectrum with a fixed safety
margin. Matched roots cannot change classification across refinements. The
repeller must lie where both implicit nullclines rise:
`-g_E,E/g_E,I > 0` and `-g_I,E/g_I,I > 0`. Passing this classifier describes
a finite discovered topology; it does not certify that all equilibria or
attractors have been found.

The internal Figure-5b root-lineage component is the numerical identity
contract for later targeted continuation. At each parameter point it checks
the deterministic 11×11, 21×21, and 41×41 searches for a common model and
seed schedule, consistent discovered root count, unique cross-grid matches,
root separation, no unresolved nearby roots, and independently recomputed
balance residuals, Jacobians, and local spectra. A locally unresolved
stability label passes only when every eigenvalue real part is within the
lineage spectral margin; the label alone is insufficient. The component
reconstructs each frozen model and drive from its source model and snapshot
time, and rejects inconsistent stored search records. Search policies must
retain finite, nonnegative tolerances and a positive iteration limit. Each root
must be tied to admissible solver candidates within discovery's maximum-
coordinate deduplication tolerance; duplicate members may have an unsuccessful
solver return code or meet only the search residual tolerance. The selected
representative still must meet the stricter lineage residual gate, and its
stored residuals and local stability fields must agree with independent
recomputation. Every admissible attempt must belong to exactly one discovered
root's member set; dropping equilibria while retaining their attempts leaves
the search unresolved. Numerical options are validated through both keyword and
positional construction. Intermediate points may have a different root count
from the initial seven-root topology; the full seven-root gate still applies
separately to an initial Figure-5b seed and a
proposed Hopf point. A transition carries the prior three-grid root
constellation and accepts the followed root only when forward and reciprocal
matches, an optional predictor, the corrected root, and a displacement bound
agree uniquely. The bound must be smaller than half the local root separation
after coordinate uncertainty is included. When roots disappear, each
destination root must map to a distinct prior root. When roots appear, every
prior root must have a distinct destination; unmatched destinations are
retained as new discoveries. Equal-count transitions require a bijection,
so a simultaneous disappearance and appearance remains unresolved. Missing,
ambiguous, poorly resolved, or skipped required checks remain unresolved.
These are finite numerical tracking rules, not proof that a branch cannot
change between sampled parameter points; later continuation must adapt its
step size and retain unresolved intervals.

The internal Figure-5b axis search starts from a genuine seven-root central
seed and searches both parameter directions. A local solve advances the path
only after an independent three-grid search and root-lineage transition.
Its initial, minimum, and maximum step settings are fractions of the
authorized parameter interval width (defaults 0.01, 0.00001, and 0.05),
not absolute coupling increments. Each trial retains an immutable snapshot
of its local solver candidate, validation, status, residual, near-singularity
flag, and reasons even when the solve is rejected; exceptions are separate.
Bracket corrections retain the raw augmented-solver status, iteration
count, and any internal exception text separately from the final validation
status. Singleton zero-trace checks explicitly record that no augmented
corrector ran.
Trace sign changes retain both accepted endpoint anchors; a balance-plus-trace
correction evaluates the model only within that bracket, then independently
checks residuals, augmented Jacobian rank, and lineage from both endpoints.
A trace-zero sampled point is validated as a singleton, with rank probes
inside the authorized parameter bounds. Both rank gates compare three
distinct, representable finite-difference scales, reducing the base step
for narrow intervals and leaving unresolvable intervals unresolved. Full
Jacobian changes must be small relative to the finest minimum singular value;
this prevents a stationary trace zero from qualifying on a spurious derivative
or three probes clipped to the same width. Failed solves, bracket corrections,
and step retries remain typed unresolved evidence. `finite_exhausted` requires
both exact parameter bounds and every detected sign-change bracket or sampled
zero endpoint resolved. This describes the finite sampled path, not model-wide
absence.

The internal two-parameter Figure-5b seed component is a separate finite
discovery layer; it does not continue a curve. It accepts an autonomous FoI
base model with all five coordinates inside the authorized search domain,
two distinct parameter names from those coordinates, a bounded rectangle
inside their authorized ranges, and a claimed central state. Box endpoints
must already be finite Float64 values; no silent conversion can widen the
supplied rectangle. The box must be strictly ordered, authorized, and contain
the original base parameters and their Float64 representation.
Regenerating the origin at Float64 precision must preserve the supplied model identity.
It independently confirms the origin's seven-root repeller topology and
central-root lineage. From that origin it always tries one bounded normalized
interior stationary-distance KKT solve and all four bounded
edge solves, retaining every numerical attempt and failed backtrack. Each
converged candidate is assessed independently before only qualified duplicate
seeds are collapsed. KKT stationarity does not establish a local or global
minimum-distance or search-completeness certificate.

A qualified seed needs fresh 11×11, 21×21, and 41×41 searches at its own
model. These must contain seven well-resolved matched roots, three robust
outer attractors, three saddles, and the same near-neutral central root on
both rising nullcline arms, using the same robust slope tolerance as the
origin topology gate. The center must be classified neutral on every grid,
with trace bounded by an independent 1e-7 ceiling even if axis options are
looser. Independently recomputed balance, original-time
trace and determinant must resolve a simple imaginary pair. A regular
three-equation/four-variable Jacobian must agree across three distinct,
box-confined finite-difference scales and an independent derivative. Adaptive
straight-line parameter homotopy from the origin uses local solves and the
reciprocal three-grid root-lineage gate at every accepted step; any gap leaves
the candidate unresolved. Numerical solver success or a locally
Hopf-compatible trace zero alone never qualifies a seed. No cycles,
biological interpretation, or manuscript claim follows from these records.
The result distinguishes `qualified_seeds`, `no_qualified_seed` (all five
methods completed but had resolved finite rejections),
`search_unresolved` (one or more methods failed or were numerically
unresolved with no qualified seed), and `origin_unresolved`. Raw attempts,
qualifications, and unresolved method names remain available for audit. None
of these statuses certifies global absence or completeness.

The internal Figure-5b curve-traversal component starts only from an indexed,
qualified seed record. It rechecks the frozen model, exact parameter box,
selected method and candidate, then freshly verifies that seed's origin,
central-root lineage, and numerical qualification. Its traversal core uses a
source-agnostic verified-seed token, which `traverse_verified_seed` accepts so a
later search runner can add a separate adapter that freshly verifies accepted
one-parameter axis-zero locations and their endpoint lineages before using them.
That entry performs no qualification of its own, so a caller must obtain its
token from a gate that does. This component alone cannot establish a
ratio-4.4 branch when its seed search found no qualified seed.

Two signed normalized pseudo-arclength directions use an independently
recomputed balance/trace/phase residual, regular three-by-four curve rank,
regular four-by-four corrector rank, positive oriented progress, and bounded
local predictor correction. Every accepted point gets new 11×11, 21×21, and
41×41 searches and reciprocal predictor/corrected central-root lineage.
Seven-root neutral-center Hopf qualification is recorded separately for each
point; a tracked five-root trace-zero point is not a Figure-5b Hopf candidate.
That per-point record is not advisory: a directional exit is reported as a
qualified boundary only when its own neutrality evidence qualifies it, and a
segment containing any unqualified point is reported as
`finite_two_unqualified_boundary_segment` rather than as a two-boundary segment.
`finite_two_boundary_segment` is therefore a statement about geometry and
neutrality together, and a consumer must still read `point.hopf.qualified` at
every point it treats as a Figure-5b Hopf candidate.
Failed correctors, retries, rank loss, loops, and iteration-cap endpoints
remain explicit unresolved outcomes. A revisit is reported only when it is the
sole failure of its step; `revisit_atol` is required to stay below the smallest
forward progress a legal step can make, so a perfectly forward step can never
read as a loop. Projection folds may be traversed.

A parameter-box endpoint is qualified only by a local fixed-edge three-variable
balance/trace solve from the first tangent-ray intersection. The remaining
parameter must stay in range, the fixed-edge Jacobian must have stable full
rank, the corrected tangent must cross outward, and independent residual,
phase, locality, progress, and reciprocal lineage checks must pass. Near-corner
hits retain both edge attempts and remain unresolved for that step, because the
first crossed edge is not geometrically unique at that tolerance; a later
smaller step may resolve the geometry to a single edge, which is then judged on
its own merits. A wrong or stale edge ray is rejected before numerical correction.
An out-of-box corrector is never itself boundary evidence. A validated exit is
carried as a typed endpoint, so it never has to be rediscovered inside a
boundary-attempt history that also holds rejected and ambiguous attempts. Two
qualified directional endpoints describe only a finite sampled segment, not a
complete curve, parameter domain, or attractor regime.

For a fixed balance field and `r = tau_I/tau_E`, `hopf_diagnostics` evaluates
the local trace-zero candidate `r_H = -Dg[2,2] / Dg[1,1]`. It requires a
positive determinant and nonzero eigenvalue transversality. The calculation
is explicitly `Float64`: state coordinates and `tau_E` may be integers,
rationals, `Float32`, or `Float64`, with exact `Float64` representability
required for integer and rational values. Higher-precision inputs such as
`BigFloat` are rejected rather than silently narrowed. The first
Lyapunov coefficient uses the original-time Taylor convention
`f = A*y + B(y,y)/2 + C(y,y,y)/6`, a unit Euclidean right eigenvector, and
the Kuznetsov normalization `l1 = real(G21)/(2omega)`. A ForwardDiff tensor
calculation must agree with a separately finite-differenced planar
Guckenheimer-Holmes formula over several steps. Near-zero coefficients,
missing finite-difference plateaus, and sign or magnitude disagreement remain
unresolved. A negative coefficient is reported as a
`supercritical_candidate`; the signed nonzero transversality separately says
which parameter side contains the branch. Local diagnostics do not prove
that a periodic orbit exists.

The modal normal-form amplitude obeys
`rho^2 approximately -beta*(r-r_H)/real(c1)`, where `c1=G21/2` and
`l1=real(c1)/omega`. The dimensionless `l1` is not the physical-time cubic
coefficient. Orbit continuation and independent shooting evidence are
required before comparing this local scaling with a finite-amplitude cycle.

The tracked `experiments/figure5b_hopf.toml` and
`scripts/run_figure5b_protocol.jl` freeze the Figure-5b anchor
`(e_to_e,i_to_e,e_to_i,i_to_i)=(19,13,19,6)` with `theta_off=8` and zero
drive. The candidate-specific runner independently repeats the 11×11, 21×21,
and 41×41 root searches at timescale ratios 4.4, 0.5, and 0.4; bisects trace
zero using finite-difference ODE Jacobians; shoots deterministic Hopf-eigenspace
seeds; and validates any selected orbit with tighter integration, doubled
sampling, primitive-period, winding, phase-equivalence, Jacobian, divergence,
and finite-difference Poincare checks. A validated ratio-0.5 orbit can then be
continued over the configured finite interval. All shooting failures and
continuation corrector failures are retained.

A non-smoke run requires `--accepted-revision` and is evidence-eligible only
when that exact 40-character revision is checked out at detached `HEAD` with a
clean worktree. `--replay-parent` verifies the parent artifact's complete
checksum manifest plus its archived config and source hashes before repeating
the parent mode from the snapshot into a sibling output directory, leaving the
source artifact checksum-valid. Replays never become scientific evidence and
do not require a Git checkout. Runner endpoint labels are conservative
compatibility observations: Hopf requires a successful near-Hopf parameter
boundary with shrinking radius and matching period; fold requires both a
parameter reversal and near-unit transverse multipliers; global saddle and
state-boundary labels require monotone validated terminal segments with the
configured period, distance, and boundary checks. Unresolved solver
termination remains recorded separately, and none of these labels is a
bifurcation proof.

The output is an ignored numerical artifact with archived source/configuration
hashes and checksums, not tracked evidence. Smoke mode exercises the exact
candidate with fewer orbit seeds and continuation steps but disables all
scientific acceptance. Passing the full finite protocol can support only the
reported coexistence of three discovered locally attracting equilibria and
one numerically validated attracting orbit, or four discovered locally
attracting equilibria below the Hopf point. It does not establish an exact
attractor count, search completeness, prevalence, biological states, or a
global bifurcation diagram.

## Scientific experiment configurations

[The intervention study](intervention_study.md) describes the supplied
exploration bounds, execution commands, and evidence gates. Separate TOML
configurations drive the coexistence map, pulse experiments, and individual
intervention/targeted robustness sweeps. All preserve matched non-inhibitory
parameters and the approved response family, retain unstable equilibria and
failed attempts, and archive source/configuration provenance with checksums.
The manuscript-facing analytical restrictions are in
[the theory notes](theory_notes.md). Numerical candidates do not inherit
biological labels from these workflows.

## Comparison responses and analysis limitations

`PointModelParameters` supports `LogisticResponse` for the excitatory
population and either `LogisticResponse` or `FailureOfInhibitionResponse` for
the inhibitory population. The exported rectified and unequal-slope response
types remain available only for standalone comparisons.

Equilibrium discovery, local stability, trajectory diagnostics, continuation,
and periodic-orbit shooting are numerical procedures with the explicit limits
above. The minimal matched-model experiment remains a synthetic workflow
check. Search completeness, rigorous existence and bifurcation certificates,
biological regime definitions, and publication claims are not supplied by
these numerical labels.
