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
outside `[0,1]`. It uses an absolute numerical-domain tolerance, defaulting to
`domain_atol=1e-8`, and rejects accepted steps or returned states outside
`[-domain_atol, 1+domain_atol]`. States are never clipped or projected.

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

## Comparison responses and deferred work

`PointModelParameters` supports `LogisticResponse` for the excitatory
population and either `LogisticResponse` or `FailureOfInhibitionResponse` for
the inhibitory population. The exported rectified and unequal-slope response
types remain available only for standalone comparisons.

Equilibrium finding, physical-admissibility classification of invariant
objects, stability and bifurcation analysis, regime diagnostics, deterministic
experiment schemas, plotting, and manuscript claims are outside the current
implementation. Numerical solver selection, adaptive-step settings,
floating-point precision, and diagnostic tolerances belong to experiment
configuration.
