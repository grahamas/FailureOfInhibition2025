# Zero-drive parameter atlas

The atlas records sampled structural regions, examined connections, local transitions, periodic-orbit evidence, and unresolved work. Its domain and seeds come from the equations. The historical candidate collections neither bound nor initialize this workflow.

A completed bounded run means that the requested protocol reached its budget or job limit with attempted and deferred work accounted for. It does **not** establish that all parameter regions, bifurcations, or attractors were discovered. The parameter atlas remains `CompletenessNotCertified` even when some individual equilibrium inventories receive fixed-parameter certificates.

## Domain and exact reductions

Both external drives are zero. The excitatory response is logistic; the inhibitory response is the unnormalized difference of equal-slope onset and failure logistics. Use the eight dimensionless coordinates

```
A = sE*wEE       B = sE*wIE       C = sI*wEI       D = sI*wII
 a = sE*thetaE    b = sI*thetaOn   h = sI*(thetaOff-thetaOn)
r = tauI/tauE
```

Here `A,B,C,D >= 0`, `a,b` are arbitrary finite real numbers, and `h,r > 0`. The geometry has seven coordinates: equilibria do not depend on `r`. Slopes and the common time scale have been absorbed into coordinate units. Physical reconstruction requires supplying positive slopes and `tauE`; divide the corresponding dimensionless coordinates by those slopes and set `tauI=r*tauE`. Quantitative responses to physical drives must retain these conversion factors.

For dimensionless equilibrium inputs `u,v`, let `sigma` denote logistic and set

```
E = sigma(u)/(1+sigma(u))
F = sigma(v)-sigma(v-h)
I = F/(1+F)
a = A*E-B*I-u
b = C*E-D*I-v
```

Choosing `u,v,A,B,C,D,h` therefore constructs an equilibrium and its parameter tuple exactly in real arithmetic. The chart covers equilibrium tuples; projection into parameter space can have multiple sheets. It does not establish that a separately chosen parameter tuple has no additional equilibria.

Let `K` be the balance-equation Jacobian, before time-scale division. In units where `tauE=1`, `J=diag(1,1/r)*K`. Thus its determinant, trace, and discriminant determine equilibrium stability over all finite positive `r` without an eight-dimensional grid. A trace-zero point requires positive determinant and further nondegeneracy checks before being called a Hopf bifurcation. A discriminant-zero node–focus boundary alone is not a bifurcation.

## Expanding exploration envelope

There is no finite outer parameter boundary implied by this model. The search uses nested windows

```
Dk: A,B,C,D in [0,2^k], a,b in [-2^k,2^k], h in [2^-k,2^k], k >= 1.
```

Their union covers the finite admissible geometry domain. Independent global jobs sample disjoint shells `D1`, `D2\D1`, and so on. The shell schedule `k=1+nu2(n)` repeatedly revisits each shell while introducing larger ones; `nu2(n)` is the number of factors of two in positive integer `n`. Persistent Halton counters supply deterministic samples, with logarithmic width coordinates. This mathematical scheduling property is asymptotic, not a finite coverage guarantee. The implementation also has explicit finite arithmetic and sampling limits; reaching one leaves the remaining direction unresolved rather than closing the domain.

The fifteen admissible zero-coupling boundary masks receive explicit sampling. The limits `h=0`, `r=0`, and infinite parameters are excluded limiting faces, not sampled finite models. Analytically seeded transition neighborhoods outside sampled shells are recorded separately; they do not fill the intervening parameter volume. Sampling density is a computational convention, not a probability distribution over models.

The default time allocations are 40% independent discovery (half parameter probes, half equilibrium-chart probes), 40% event continuation and local investigation, 10% periodic work, and 10% fixed-parameter certification. Every fourth discovery job targets a zero-coupling stratum. Unused allocations return to discovery. Every job and run has hard limits, and unfinished tasks remain in the checkpoint.

## Transitions, organizing candidates, and representative cases

Continuation follows structural changes rather than stopping when a signature changes. Recorded transitions retain both sides, localization evidence, and correction failures. A failed corrector does not establish that a branch terminates.

The first version targets numerical cusp and Bogdanov–Takens investigations. Event equations alone establish candidates. Classification additionally requires nondegenerate normal-form coefficients and a rank-two unfolding in explicitly named admissible parameter directions. Definitions and coefficient conventions should be read with [Kuznetsov's normal-form derivation](https://ir.cwi.nl/pub/4709). Two-parameter sections are local views of a larger domain; crossings of projected curves can belong to different equilibria and do not themselves identify a nexus. Generalized-Hopf classification and homoclinic continuation remain outside this release.

Periodic work starts from numerically nondegenerate Hopf candidates and retains stable as well as unstable branches when initialized successfully. Period, amplitude, residual, and Floquet evidence have distinct meanings. Failed shooting or a finite search without a cycle does not establish absence of periodic behavior.

The report builds provisional region components only from explicit completed paths whose declared structural signature was preserved. Two samples with equal signatures remain separate without an examined connection. A component's representative is its lexicographically first saved point ID. Transition and periodic witnesses supply additional base cases. `base_cases.csv` links representatives to their component members and the path ledger preserves the observed connections. No claim is made that every region touches a single shell or nexus.

Point certificates cover the equilibrium rectangle `[0,1/2]^2` at their exact saved parameters. They do not certify continuous parameter neighborhoods, stability, basin structure, or all attractors.

## Running and reading the evidence

The entry point is `scripts/zero_drive_atlas.jl`, run with the repository's Julia 1.10 project. It provides `benchmark`, `run`, `resume`, and `report` modes; the versioned default configuration is `scripts/atlas/defaults.toml`. The commands are:

```sh
julia --project=. scripts/zero_drive_atlas.jl benchmark OUTPUT [--config FILE]
julia --project=. scripts/zero_drive_atlas.jl run OUTPUT --budget-seconds N [--config FILE]
julia --project=. scripts/zero_drive_atlas.jl resume OUTPUT --budget-seconds ADDITIONAL_N
julia --project=. scripts/zero_drive_atlas.jl report OUTPUT
```

`resume` retains the saved configuration. The report mode reads saved results and performs no scientific computations.

Execution currently requires Linux: a kernel lock excludes concurrent coordinators,
and parent-death signaling stops workers if their coordinator exits. Workers run
from the saved source snapshot with one Julia and BLAS thread. Report generation
uses the same exclusive run lock.

The bounded benchmark permits up to ten minutes of setup/warm-up followed by twenty minutes of measured work, with a default 120-second job timeout. It exercises each job class and records startup separately. The larger pilot requires an explicit user-selected budget after reviewing measured costs. The implementation/benchmark milestone does not authorize an automatic open-ended campaign.

Setup loads the worker and its modules. A handler's first measured invocation can
still compile specialized methods; raw worker timing records retain compilation
separately. Wall-time summaries therefore include cold-handler costs and must not
be interpreted as uniformly warmed execution times.

Each output directory retains `state.toml`, exact `config.toml`, referenced raw result files, source/configuration provenance, and pending tasks. Resume checks the recorded inputs before continuing. The generated report provides:

- `index.html`: browsable points, branches, transitions, cycles, certificate evidence, deferred work, and costs.
- `coverage.toml`: counts and explicit unresolved frontier, without a percentage of unbounded parameter space covered.
- `base_cases.csv`: provisional representatives and additional witnesses.
- `timings.csv`: per-kind completed-job timing distributions and failure counts.

Timing medians and tails require their sample counts. Timeouts are censored observations rather than successful cost samples; a few easy jobs cannot predict difficult neighborhoods reliably. Numerical failures, unexecuted queued work, unsampled finite parameters, unbounded shells, and unsearched attractors remain visible even when a run ends normally at its budget.

The report renders sparse evidence as tables and examined connections, not filled high-dimensional regions. Saved numerical classifications and proof statuses remain separate. Re-rendering is deterministic from the same saved evidence and does not silently run additional discovery.
