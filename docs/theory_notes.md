# Analytical constraints on the intervention experiments

These results concern the approved occupancy equations in [model.md](model.md).
They do not assign biological meaning to an equilibrium or certify numerical
discovery. Coupling magnitudes remain arbitrary finite nonnegative numbers;
both time constants remain finite and strictly positive.

## Monotone inhibition excludes oppositely ordered equilibria

Consider two physical equilibria of the **same autonomous model**, with the
same constant drives and parameters, and let the inhibitory response be
nonnegative and nondecreasing. At either equilibrium,

```math
I = R(F_I(u_I)),\qquad R(z)=\frac{z}{1+z},\qquad
u_I=J_{I\leftarrow E}E-J_{I\leftarrow I}I+B_I.
```

The map `R` is strictly increasing for nonnegative arguments. Suppose that
`E_2 >= E_1` and `I_2 < I_1`. Nonnegative coupling magnitudes give

```math
u_{I,2}-u_{I,1}
=J_{I\leftarrow E}(E_2-E_1)
-J_{I\leftarrow I}(I_2-I_1)\geq0.
```

Monotonicity then gives `I_2 = R(F_I(u_I,2)) >= R(F_I(u_I,1)) = I_1`, a
contradiction. Thus higher excitatory activity cannot coexist with lower
inhibitory activity at another equilibrium of this model. Zero couplings are
allowed. The proof uses neither the excitatory equilibrium equation nor the
time constants, and constant drives need not be zero. Unit connectivity is
unnecessary and is not a justified general change of variables.

For the logistic control and `J_{I<-E} > 0`, the inhibitory nullcline is in
fact strictly increasing. Implicit differentiation gives

```math
\frac{dI}{dE}=
\frac{(1-I)F_I'(u_I)J_{I\leftarrow E}}
{1+F_I(u_I)+(1-I)F_I'(u_I)J_{I\leftarrow I}}>0.
```

When `J_{I<-E}=0`, the nullcline is constant in `E`. For each fixed `E`,
`I/(1-I)-F_I(J_{I<-E}E-J_{I<-I}I+B_I)` is strictly increasing on `[0,1)`
and crosses zero exactly once for the logistic response.

This is an equilibrium-ordering result. It does not exclude a lone high-E,
low-I equilibrium, coexistence of equilibria ordered in the same direction,
periodic attractors, or every state one might call a seizure. It does not
compare different parameter settings or different frozen drives.

## Equal-slope failure response is symmetric

Write `m=(theta_on+theta_off)/2`, `d=a_I*(theta_off-theta_on)/2 > 0`, and
`x=u-m`. The approved response has the exact form

```math
F_I(m+x)
=\sigma(a_Ix+d)-\sigma(a_Ix-d)
=\frac{\sinh(d)}{\cosh(a_Ix)+\cosh(d)}.
```

Consequently `F_I(m+x)=F_I(m-x)`. Its derivative is

```math
F_I'(m+x)
=-\frac{a_I\sinh(d)\sinh(a_Ix)}
{[\cosh(a_Ix)+\cosh(d)]^2}.
```

The response increases below `m`, has its unique maximum
`tanh(a_I*(theta_off-theta_on)/4)` at `m`, and decreases above `m`.
`theta_off` is the center of the failing logistic; the response begins
descending at the **midpoint**, not at `theta_off`. The hyperbolic expression
is an analytical identity, not a replacement for the overflow-safe code.

Independently adjustable rising and falling slopes are unavailable in the
approved model. Such fitting claims require either a change in interpretation
or a separately justified response extension. Simply giving the two raw
logistics unequal positive slopes does not preserve nonnegativity on the
whole real input line: their argument difference is affine with nonzero
slope, changes sign at a finite input, and strict logistic monotonicity then
forces the response difference to be negative on one tail. A proposed
asymmetric extension must justify its response form and supported domain.
Raising `theta_off` at fixed `u` increases `F_I(u)` by

```math
\frac{\partial F_I(u)}{\partial\theta_{\mathrm{off}}}
=a_I\sigma_{\mathrm{off}}(u)[1-\sigma_{\mathrm{off}}(u)]>0.
```

It also changes the midpoint and maximum. A smaller disturbance to active
dynamics is therefore a hypothesis to measure, not an exact independence
property of this intervention.

## A specified positive-drive class has a global rescue obstruction

Let `(E_*,I_*)` be an equilibrium of the FoI model with baseline drives
`(B_E,B_I)`, and suppose its inhibitory input satisfies `u_I,* >= m`.
Allow finite piecewise-constant additive drives `Delta P_E(t) >= 0` and
`Delta P_I(t) >= 0`, followed by return to baseline. Then

```math
\mathcal R_*=[E_*,1]\times[0,I_*]
```

is forward invariant. To see this, inspect its four faces:

- At `E=E_*`, `I <= I_*` and nonnegative E drive imply
  `u_E >= u_E,*`. Monotone `F_E` and the equilibrium balance imply `E_dot >= 0`.
- At `I=I_*`, `E >= E_*` and nonnegative I drive imply
  `u_I >= u_I,* >= m`. The decreasing FoI response and equilibrium balance
  imply `I_dot <= 0`.
- At `E=1`, `tau_E*E_dot=-1`; at `I=0`, `tau_I*I_dot=F_I(u_I) >= 0`.

Each constant-drive segment points into this closed rectangle at its faces.
The state is continuous at drive switches, so invariance persists through
the whole protocol and after removal. An initial state at the equilibrium
cannot reach, or converge to, an attractor outside this rectangle. In
particular, a lower-E rest equilibrium cannot be reached by these positive
pulses. This establishes a global restriction for a defined intervention
class; a local response arrow alone would not establish it.

The result is specific to the approved response, nonnegative coupling
magnitudes, a starting equilibrium on its descending branch, and additive
drives that never fall below baseline. It does not establish unrestricted
unrescuability or identify a biological seizure. A negative E-drive pulse or
a parameter intervention can break its assumptions. Numerical pulse maps
should retain their finite-horizon uncertainty even when this separate
analytical obstruction applies.

## Figure 3 anchor check

For the supplied anchor `(e_to_e,i_to_e,e_to_i,i_to_i)=(17,9,19,4)`,
`theta_off=8`, slopes `5`, onset thresholds `(1.5,4)`, time constants
`(7.8,34.32)` ms, and zero drives, a 31-by-31 uniform multistart search on
`[0,0.5]^2` with Julia 1.10.12 found the following seven FoI equilibria.
The search used default equilibrium/stability tolerances with `maxiters=200`.
Every listed candidate was admissible, had balance residual below `3e-16`,
and had neither a near-singular nor an unresolved-nearby flag.

| E | I | u_I | F_I'(u_I) | Eigenvalues (ms^-1) | Local stability |
| ---: | ---: | ---: | ---: | --- | --- |
| 0.000580378942 | 2.17798859e-9 | 0.0110271912 | 1.08899430e-8 | -0.121958609, -0.0291375307 | Attracting |
| 0.0556359623 | 4.06943072e-7 | 1.05708166 | 2.03471536e-6 | -0.0291370447, 0.434811698 | Saddle |
| 0.311014590 | 0.425138163 | 4.20872456 | 0.963082727 | 0.0733669308, 1.48465857 | Repelling |
| 0.350837394 | 0.492423107 | 4.69621805 | 0.144818463 | -0.0423416502, 1.53585559 | Saddle |
| 0.499999911374 | 0.447721024 | 7.70911422 | -0.767392408 | -0.256409228, -0.00336211504 | Attracting |
| 0.499999939139 | 0.439369305 | 7.74252162 | -0.847556165 | -0.256409610, 0.00340859579 | Saddle |
| 0.500000000000 | 0.000558673969 | 9.49776530 | -0.00279336897 | -0.256410256, -0.0288284310 | Attracting |

A separate SciPy 1.18.1 calculation using independently written balance and
Jacobian formulas, `scipy.optimize.root`, and a 51-by-51 seed grid found the
same seven roots and spectra. Both calculations found five matched-control
equilibria: two attracting, two saddles, and one repelling. These are
multistart cross-checks, not completeness certificates, and do not exclude
periodic attractors.

The high-E/high-I attracting equilibrium is also on the descending response
branch. Its slow linear relaxation time is about `297` ms, and the nearby
saddle has `I=0.439369305`. Thus a negative inhibitory response derivative
does not by itself distinguish a failed-inhibition seizure candidate from
every other nonquiescent attractor. Both high-E attractors satisfy the
positive-drive obstruction above. The intermediate equilibrium near
`(0.311,0.425)` is repelling at this anchor, not a stable intermediate active
equilibrium.

## Boundaries on numerical interpretation

- Track equilibrium branches with their continuous coordinates and spectra;
  sorted root indices can change and are not attractor identities. Failed
  discovery of a branch is not evidence that the branch was removed.
- The estimate `e_to_i ~= 2*theta_off` uses `E ~= 0.5` and negligible `I`.
  It concerns access to strong failure near the falling-logistic center;
  it is not an exact bifurcation or response-branch boundary.
- Continuation can follow unstable equilibria, but sampled branch changes
  alone do not certify folds or Hopf bifurcations. A periodic solution needs
  a return/phase condition, a nonzero amplitude check, and stability evidence
  such as its nontrivial Floquet multiplier; a long oscillatory trace is
  insufficient.
- Changing the positive time constants leaves the balance equations and
  equilibrium positions unchanged. With balance Jacobian `Dg`, the ODE
  determinant is `det(Dg)/(tau_E*tau_I)` and the trace is
  `Dg[1,1]/tau_E + Dg[2,2]/tau_I`. The ratio can therefore change stability
  while preserving equilibria; a trace crossing alone is not a proven Hopf
  bifurcation.
- Pulse outcomes require a declared source, destination, target, signed
  amplitude, duration, post-removal horizon, and uncertainty. Search all
  sampled transition intervals; a single monotone amplitude threshold need
  not exist. Integrated input has effective-input-times-time units, and a
  scalar combination across targets requires an explicit cost convention.

## Manuscript locations for author review

The manuscript was read through the GitHub API at revision
`ed18f283f2c947d6a4710989664b2a05dcc74447`; no manuscript files were changed.
The [Figure 3 caption](https://github.com/grahamas/FailureOfInhibitionWCM/blob/ed18f283f2c947d6a4710989664b2a05dcc74447/sn-article.tex#L360)
specifies the first coupling anchor. The
[Figure 4 caption](https://github.com/grahamas/FailureOfInhibitionWCM/blob/ed18f283f2c947d6a4710989664b2a05dcc74447/sn-article.tex#L371)
specifies the second anchor's three fixed couplings but omits the varying
E-to-I values. The
[parameter table](https://github.com/grahamas/FailureOfInhibitionWCM/blob/ed18f283f2c947d6a4710989664b2a05dcc74447/sn-article.tex#L390)
gives the stated time constants, slopes, onset/failure thresholds, and zero
stimulus.

The author should review the
[unit-connectivity reduction](https://github.com/grahamas/FailureOfInhibitionWCM/blob/ed18f283f2c947d6a4710989664b2a05dcc74447/sn-article.tex#L285)
against the general ordering proof above, the
[independent-slope claim](https://github.com/grahamas/FailureOfInhibitionWCM/blob/ed18f283f2c947d6a4710989664b2a05dcc74447/sn-article.tex#L279)
against the approved symmetric response, and the
[rescue claim](https://github.com/grahamas/FailureOfInhibitionWCM/blob/ed18f283f2c947d6a4710989664b2a05dcc74447/sn-article.tex#L352)
against an explicit intervention class and reachable-target definition.
