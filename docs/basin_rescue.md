# Basin geometry and tonic-drive rescue

These methods concern the approved two-population point model. The words
`high_e_low_i` and `high_e_high_i` identify coordinate-tracked numerical
candidates, not biological states. The first two configured zero-drive cases
are the Figure 3 anchor and a Figure 4 exploration cell with an attracting
high-E/high-I candidate on the ascending inhibitory response branch.
The proposed [mathematical exemplar catalogue](exemplar_models.md) gives
stable parameter identifiers, including a four-sink case; the study runner
has not yet been extended to measure that fourth basin.

## Five state-space distances and basin area

The area fraction is a finite-grid estimate of **basin stability** for a
uniform initial-state distribution on the physical square. The shortest
distance to the basin boundary is commonly called a **stability threshold**;
the four signed-axis distances show its direction dependence. These terms
describe state perturbations. The minimum amplitude or integrated input of
a finite drive is a separate, protocol-dependent control threshold.
See the original [basin stability](https://www.nature.com/articles/nphys2516)
and [stability threshold](https://arxiv.org/abs/1504.04476) papers.

`measure_basin(model, equilibria, source_index; options)` samples initial
states in the physical `[0,1]^2` square. It reports the fraction of sampled
cell centers compatible with the source equilibrium and an upper sampled
fraction that assigns every unresolved cell center to that basin. These
fractions are **not mathematical bounds on basin area**. The grid width,
counts of other and unresolved samples, source coordinates, and numerical
policy accompany them. The current destination classifier recognizes
discovered attracting equilibria only. A periodic or undiscovered attractor
can cause samples to remain unresolved or be misclassified by finite-window
compatibility, so numerical basin estimates remain conditional on discovery.

From the source equilibrium, the same routine samples rays in the positive
and negative E and I directions. For each ray, it brackets the first
**observed** departure from the source's basin among sampled initial states
and refines that bracket. It samples additional angles to estimate the
nearest Euclidean departure. A ray without an observed exit before the
physical-domain edge is censored; an unresolved trajectory is retained as
uncertainty. No ray scan excludes narrower unsampled basin intersections.
These are state perturbations, not applied drive amplitudes.

`basin_destination` independently integrates each configured follow-up
horizon and applies the package's two-window diagnostic and local-attraction
check. Its result is finite-window equilibrium compatibility, not proof of
asymptotic convergence. `BasinMeasurementOptions` selects grid and ray
resolution and reuses the existing pulse diagnostic and solver policy.

## Nonnegative total drive with tonic E withdrawal

`run_tonic_rescue_trial` starts from an equilibrium of an autonomous model
with a nonnegative tonic drive `(B_E,B_I)`. For one rectangular interval of
length `T`, it applies totals `(B_E-A_E,B_I+A_I)` with `0 <= A_E <= B_E` and
`A_I >= 0`, then restores the original baseline. E withdrawal and I
excitation start and stop together. The integrated **increments** are
`(-A_E*T,A_I*T)`; they are abstract effective-input costs, not energy.
The destination is classified after withdrawal against equilibria of the
original baseline system. Pure E withdrawal, pure I excitation, their
combination, and the zero-input case fit this interface.

The distinction between total drive and increments matters. The positive
drive obstruction in [the theory notes](theory_notes.md) requires increments
that never fall below baseline. It therefore applies to the Figure 3
positive-I-only control, while an E withdrawal from a positive baseline
does not satisfy its assumptions even if total drive stays nonnegative.

`scripts/run_basin_rescue_study.jl` runs one configured case in `basin` or
`tonic` mode. Full tonic mode tests baselines `0:0.25:8`, E reductions from
zero through each baseline, I increments `0:0.25:8`, and the configured
duration grid. It refines adjacent differing outcomes along both amplitude
axes, with no monotonicity assumption. A missing or ambiguous coordinate
match makes the source or lower-activity target unavailable. It does not
become a failed rescue, and a lost branch is not silently reacquired at a
later baseline. `onset_brackets.csv` records every adjacent baseline
status change; any bracket involving an unresolved or unavailable status is
censored. Such brackets are tested-protocol observations, not exact minimum
tonic baselines or proofs of unreachability.

Run from the repository root, using a new output directory:

```bash
julia --project=. scripts/run_basin_rescue_study.jl --mode basin --case figure3 --output output/figure3_basin
julia --project=. scripts/run_basin_rescue_study.jl --mode basin --case figure4_rising --output output/figure4_basin
julia --project=. scripts/run_basin_rescue_study.jl --mode tonic --case figure3 --output output/figure3_tonic
julia --project=. scripts/run_basin_rescue_study.jl --mode tonic --case figure4_rising --output output/figure4_tonic
```

`--smoke` performs a small execution check and is not a scientific run.
`--e-to-i`, `--theta-off`, and `--tau-ratio` override one parameter at a time
or in combination within the configured exploratory bounds, supporting
local sensitivity studies without silently changing other parameters.
Outputs contain source snapshots, configuration, search contexts, numerical
policy, status rows, and SHA-256 checksums. Full tonic scans are substantial
local, on-demand experiments.

## Exact counts

The existing numerical equilibrium search and all basin and pulse outputs
retain `CompletenessNotCertified`. The separate
[interval certification driver](../certification/README.md) proves complete
equilibrium counts for the two selected zero-drive cases but does **not**
certify exact counts of all attractors. Its global cycle-exclusion gate fails
in both cases. Numerical root matching, a filled basin grid, and finite
pulse follow-up cannot close that gap. No manuscript claim is promoted by
these measurements.
