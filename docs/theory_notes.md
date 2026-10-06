# Analytical constraints on the intervention experiments

These results concern the approved occupancy equations in [model.md](model.md).
They do not assign biological meaning to an equilibrium or certify numerical
discovery. Coupling magnitudes remain arbitrary finite nonnegative numbers;
both time constants remain finite and strictly positive.

For reference, the equations are

```math
\begin{aligned}
\tau_E\dot E&=-E+(1-E)F_E(u_E),\\
\tau_I\dot I&=-I+(1-I)F_I(u_I),\\
u_E&=J_{E\leftarrow E}E-J_{E\leftarrow I}I+P_E(t),\\
u_I&=J_{I\leftarrow E}E-J_{I\leftarrow I}I+P_I(t).
\end{aligned}
```

Here `E` and `I` are active population fractions in `[0,1]`, and
`E_dot = dE/dt` and `I_dot = dI/dt` are their rates of change. The effective
inputs `u_E` and `u_I` include recurrent excitation, recurrent inhibition,
and the external drives `P_E` and `P_I`. Each `J` is a nonnegative coupling
magnitude; the minus signs above supply the inhibitory effect.
The functions `F_E` and `F_I` give the population responses to effective
input, and a prime denotes differentiation with respect to that input.

A model is **autonomous** when its equations have no explicit dependence on
time. Here this means fixed parameters and constant external drives
`P_E(t)=B_E`, `P_I(t)=B_I`. The activities can still change with time, and the
constant drives need not be zero. A scheduled pulse makes the full protocol
time dependent; each interval with a fixed drive has its own autonomous
equations.

## Monotone inhibition excludes oppositely ordered equilibria

Consider two physical equilibria of the **same autonomous model**, with the
same constant drives and parameters, and let the inhibitory response be
nonnegative and nondecreasing. An equilibrium is a state at which both rates
of change vanish. The inhibitory balance gives

```math
\begin{aligned}
0&=-I+(1-I)F_I(u_I),\\
I[1+F_I(u_I)]&=F_I(u_I),\\
I&=\frac{F_I(u_I)}{1+F_I(u_I)}.
\end{aligned}
```

The same rearrangement applies to the excitatory balance. For the ordering
argument, only the inhibitory relation is needed. Write it as

```math
I = R(F_I(u_I)),\qquad R(z)=\frac{z}{1+z},\qquad
u_I=J_{I\leftarrow E}E-J_{I\leftarrow I}I+B_I.
```

The map `R` is strictly increasing for nonnegative arguments because
`R'(z)=1/(1+z)^2 > 0`. Suppose, for a contradiction, that two equilibria
satisfy `E_2 >= E_1` and `I_2 < I_1`. Their parameters and constant drive are
identical, so `B_I` cancels when we subtract their inhibitory inputs:

```math
u_{I,2}-u_{I,1}
=J_{I\leftarrow E}(E_2-E_1)
+J_{I\leftarrow I}(I_1-I_2)\geq0.
```

Both terms are nonnegative: the second state has at least as much excitation
and no more inhibitory self input. No comparison between the two coupling
magnitudes is required.

Monotonicity then gives

```math
I_2=R(F_I(u_{I,2}))\geq R(F_I(u_{I,1}))=I_1,
```

contradicting `I_2<I_1`. Thus higher excitatory activity cannot coexist with lower
inhibitory activity at another equilibrium of this model. Zero couplings are
allowed. The proof uses neither the excitatory equilibrium equation nor the
time constants, and constant drives need not be zero. Unit connectivity is
unnecessary and is not a justified general change of variables.

This is an equilibrium-ordering result. It does not exclude a lone high-E,
low-I equilibrium, coexistence of equilibria ordered in the same direction,
periodic attractors, or every state one might call a seizure. It does not
compare different parameter settings or different frozen drives. The FoI
response is not globally nondecreasing, so this proof does not apply to it.

### Separate geometric proof for the logistic control

The inhibitory **nullcline** is the set of states where `I_dot=0`, whether or
not `E_dot` also vanishes. Every equilibrium lies on this set. For each fixed
`E`, define

```math
H_E(I)=\frac{I}{1-I}
-F_I(J_{I\leftarrow E}E-J_{I\leftarrow I}I+B_I),\qquad 0\leq I<1.
```

For the logistic control, `H_E(0)<0`, while `H_E(I)` tends to infinity as
`I` tends to one. Moreover,

```math
\frac{dH_E}{dI}=\frac{1}{(1-I)^2}
+J_{I\leftarrow I}F_I'(u_I)>0.
```

Thus each `E` has exactly one inhibitory-nullcline value `I(E)`. The endpoint
`I=1` cannot belong to the nullcline because there `tau_I*I_dot=-1`.
For the logistic control and `J_{I<-E} > 0`, the inhibitory nullcline is in
fact strictly increasing. Differentiating its balance equation with respect
to `E` gives

```math
0=-\frac{dI}{dE}-F_I(u_I)\frac{dI}{dE}
+(1-I)F_I'(u_I)
\left(J_{I\leftarrow E}-J_{I\leftarrow I}\frac{dI}{dE}\right).
```

Collecting the terms containing `dI/dE` yields

```math
\frac{dI}{dE}=
\frac{(1-I)F_I'(u_I)J_{I\leftarrow E}}
{1+F_I(u_I)+(1-I)F_I'(u_I)J_{I\leftarrow I}}>0.
```

The denominator is positive, and the numerator is positive when
`J_{I<-E}>0`, since a logistic has positive derivative at every finite input
and `I<1`. When `J_{I<-E}=0`, the nullcline is constant in `E`. Consequently,
two equilibria on this curve cannot have higher `E` but lower `I`. This is a
separate proof for the differentiable logistic control; the preceding
ordering proof requires only a nonnegative, nondecreasing response.

## Equal-slope failure response is symmetric

This section supplies supporting algebra for the approved response. The
rescue argument below uses its nonnegativity and the fact that it is
nonincreasing for inputs at or above the threshold midpoint. It does not
otherwise require symmetry.

The sigmoid is the standard logistic,

```math
\sigma(z)=\frac{1}{1+e^{-z}}
=\frac{1+\tanh(z/2)}{2}.
```

The approved response is

```math
F_I(u)=\sigma(a_I(u-\theta_{\mathrm{on}}))
-\sigma(a_I(u-\theta_{\mathrm{off}})),\qquad
a_I>0,\quad\theta_{\mathrm{on}}<\theta_{\mathrm{off}}.
```

It is positive at every finite input because the first logistic argument
exceeds the second. To expose its symmetry, write
`m=(theta_on+theta_off)/2`, `d=a_I*(theta_off-theta_on)/2 > 0`, and
`x=u-m`. The two logistic arguments become `a_I*x+d` and `a_I*x-d`.
With `y=a_I*x`, the identities

```math
\tanh A-\tanh B=\frac{\sinh(A-B)}{\cosh A\cosh B},\qquad
2\cosh A\cosh B=\cosh(A+B)+\cosh(A-B)
```

give the intermediate steps

```math
\begin{aligned}
F_I(m+x)
&=\frac12\left[\tanh\!\left(\frac{y+d}{2}\right)
-\tanh\!\left(\frac{y-d}{2}\right)\right]
\\
&=\frac{\sinh(d)}{2\cosh((y+d)/2)\cosh((y-d)/2)}\\
&=\frac{\sinh(d)}{\cosh(a_Ix)+\cosh(d)}.
\end{aligned}
```

Since `cosh` is even, `F_I(m+x)=F_I(m-x)`. Its derivative is

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
At `x=0`, the maximum follows from
`sinh(d)/(1+cosh(d))=tanh(d/2)`.

Independently adjustable rising and falling slopes are unavailable in the
approved model. Such fitting claims require either a change in interpretation
or a separately justified response extension. Simply giving the two raw
logistics unequal positive slopes does not preserve nonnegativity on the
whole real input line. To see why, let `a_on` and `a_off` be distinct positive
slopes. The first logistic argument minus the second is

```math
a_{\mathrm{on}}(u-\theta_{\mathrm{on}})
-a_{\mathrm{off}}(u-\theta_{\mathrm{off}})
=(a_{\mathrm{on}}-a_{\mathrm{off}})u
-a_{\mathrm{on}}\theta_{\mathrm{on}}
+a_{\mathrm{off}}\theta_{\mathrm{off}}.
```

This is affine in `u` with nonzero slope, so it changes sign at a finite
input. Strict logistic monotonicity then forces the response difference to
be negative on one tail. Thus equal slopes are necessary for global
nonnegativity within this raw two-logistic form with positive slopes. This
constraint does not establish that the response form itself is biologically
appropriate. A proposed asymmetric extension must justify its response form
and supported domain.

Raising `theta_off` at fixed `u` increases `F_I(u)` by

```math
\frac{\partial F_I(u)}{\partial\theta_{\mathrm{off}}}
=a_I\sigma_{\mathrm{off}}(u)[1-\sigma_{\mathrm{off}}(u)]>0.
```

Here `sigma_off(u)=sigma(a_I*(u-theta_off))`.
Raising `theta_off` also changes the midpoint and maximum. A smaller
disturbance to active dynamics is therefore a hypothesis to measure, not an
exact independence property of this intervention.

## A specified positive-drive class has a global rescue obstruction

Let `(E_*,I_*)` be an equilibrium of the FoI model with baseline drives
`(B_E,B_I)`. A star denotes the value at this baseline equilibrium; for
example, `u_E,*=J_{E<-E}*E_*-J_{E<-I}*I_*+B_E`. Suppose its inhibitory input
satisfies `u_I,* >= m`, so it is at the maximum or on the descending branch
of the FoI response.

During the intervention, write

```math
P_E(t)=B_E+\Delta P_E(t),\qquad
P_I(t)=B_I+\Delta P_I(t),\qquad
\Delta P_E(t)\geq0,\quad\Delta P_I(t)\geq0.
```

Allow a finite sequence of constant-drive intervals with finite amplitudes,
followed by return to baseline. The restriction is that neither drive ever
falls **below its baseline**, not merely that the total drive is nonnegative.

Define the following region of the `(E,I)` state plane:

```math
\mathcal R_*=[E_*,1]\times[0,I_*]
=\{(E,I):E_*\leq E\leq1,\ 0\leq I\leq I_*\}.
```

The symbol `R_*` names a rectangle; it is unrelated to the scalar map `R(z)`
used in the ordering proof. With `E` on the horizontal axis and `I` on the
vertical axis, this rectangle extends rightward and downward from the
starting equilibrium, which is its upper-left corner. It contains states
with at least the starting E activity and at most the starting I activity.

This region is **forward invariant**: any trajectory starting inside it,
including on its boundary, stays inside it at all later times under the
allowed protocol. To prove this, consider the velocity vector
`(E_dot,I_dot)`, the arrow specifying how the state moves at each point.
At every boundary, its component perpendicular to that boundary must point
inward or be zero. A zero component allows motion along the boundary.

At the **left boundary**, `E=E_*` and `I<=I_*`. Subtracting the baseline
equilibrium input gives

```math
u_E-u_{E,*}
=J_{E\leftarrow I}(I_*-I)+\Delta P_E(t)\geq0.
```

There is no increase in recurrent inhibition and no reduction in external
drive, so the effective E input is at least its equilibrium value. Using the
baseline balance `-E_*+(1-E_*)F_E(u_E,*)=0` gives

```math
\tau_E\dot E
=(1-E_*)[F_E(u_E)-F_E(u_{E,*})]\geq0.
```

Monotonicity of `F_E` supplies the final inequality. The velocity therefore
points rightward or along the left boundary; it cannot point leftward out
of the rectangle.

At the **top boundary**, `I=I_*` and `E>=E_*`. Similarly,

```math
u_I-u_{I,*}
=J_{I\leftarrow E}(E-E_*)+\Delta P_I(t)\geq0.
```

Both inhibitory inputs are at or above `m`, where the FoI response is
nonincreasing. Subtracting the baseline inhibitory balance gives

```math
\tau_I\dot I
=(1-I_*)[F_I(u_I)-F_I(u_{I,*})]\leq0.
```

The velocity points downward or along the top boundary, so it cannot point
upward out of the rectangle.

At the **right boundary**, `E=1`, the occupancy equation gives
`tau_E*E_dot=-1<0`, so the velocity points leftward. At the **bottom
boundary**, `I=0`, it gives `tau_I*I_dot=F_I(u_I)>=0`, so the velocity points
upward or along the boundary. Positive time constants preserve all these
signs. The four conditions are

| Boundary | Velocity component | Direction allowed by the equations |
| --- | --- | --- |
| Left: `E=E_*` | `E_dot>=0` | Rightward or tangent |
| Right: `E=1` | `E_dot<0` | Leftward |
| Top: `I=I_*` | `I_dot<=0` | Downward or tangent |
| Bottom: `I=0` | `I_dot>=0` | Upward or tangent |

On each constant-drive interval, the smooth response functions give unique
solutions, and these boundary conditions prevent trajectories from crossing
outward. At a drive switch, the velocity may change abruptly but the state
is continuous: a finite pulse does not instantaneously move `(E,I)`. The
same inequalities hold on the next interval and after removal, when both
drive increments are zero. Thus invariance persists through the whole
protocol and after return to baseline.

All future states and their limiting states lie in this closed rectangle.
Therefore a trajectory cannot reach or converge to an equilibrium outside
it, or approach a compact attractor disjoint from it. In particular, a
lower-E rest equilibrium cannot be reached by these positive pulses. This
establishes a global restriction for a defined intervention class; a local
response arrow alone would not establish it.

The result is specific to the approved response, nonnegative coupling
magnitudes, a starting equilibrium at the response maximum or on its
descending branch, and additive drives that never fall below baseline. It
does not establish unrestricted unrescuability or identify a biological
seizure. A negative E-drive pulse or a parameter intervention can break its
assumptions. Numerical pulse maps should retain their finite-horizon
uncertainty even when this separate analytical obstruction applies.

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
