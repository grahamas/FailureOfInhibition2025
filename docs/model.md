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

## Comparison responses and analysis limitations

`PointModelParameters` supports `LogisticResponse` for the excitatory
population and either `LogisticResponse` or `FailureOfInhibitionResponse` for
the inhibitory population. The exported rectified and unequal-slope response
types remain available only for standalone comparisons.

Equilibrium discovery, physical-admissibility classification of equilibrium
candidates, and local linear stability are implemented only as the numerical,
non-certifying procedures described above. Completeness certification,
continuation, periodic-orbit and bifurcation analysis, regime diagnostics,
deterministic experiment schemas, plotting, and manuscript claims remain
outside the current implementation.
