# Basin geometry and tonic-drive rescue

These methods concern the approved two-population point model. The words
`high_e_low_i` and `high_e_high_i` identify coordinate-tracked numerical
candidates, not biological states. The first two configured zero-drive cases
are the Figure 3 anchor and a Figure 4 exploration cell with an attracting
high-E/high-I candidate on the ascending inhibitory response branch.
The proposed [mathematical exemplar catalogue](exemplar_models.md) gives
stable parameter identifiers, including a four-sink case. The legacy study
runner below retains its original two-case protocol. The adaptive exemplar
runner described later records all attracting sinks, including the fourth.

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

## Adaptive exemplar drive study

`scripts/run_adaptive_rescue_study.jl` reads
[`adaptive_rescue.toml`](../experiments/adaptive_rescue.toml) and the four
provisional exemplars. It tracks named zero-drive attracting branches by
coordinate through one-at-a-time parameter changes and tonic E baselines.
Lost or ambiguous matches stay unavailable. Every discovered attracting sink
is still sampled as a source, with unmatched sinks recorded by local root
index. These indices are local to one search and are not branch identities.

The runner visits I-increment ranges `0–2`, `2–4`, and `4–8`, including later
ranges if an earlier range has no transition. At each baseline and duration,
it starts with amplitude spacing 1. For positive I alone it samples every
interval midpoint and bisects intervals whose endpoints or midpoint differ
or are unresolved. For E withdrawal plus I increment it samples each coarse
cell's corners and center, then recursively divides cells with differing
outcomes or unresolved observations. Observed amplitude transitions are
resolved to cell width `0.0625`; uniform sampled cells remain coarse. This
policy can miss a narrower island inside a uniform sampled cell. It does not
assume that destination varies monotonically with drive.

Tonic E baselines are probed every `0.5` from 0 through 8. An adjacent
baseline interval with a changed tracked-source summary is also sampled at
its midpoint, giving `0.25` resolution there. Equilibrium branches are
searched at every `0.25` baseline for tracking even where no pulse trials
are requested. All eight configured pulse durations are retained.

Rows with `E_reduction = 0` belong to the **positive I** protocol. Rows with
`E_reduction > 0` belong to the separate **tonic E withdrawal** protocol;
both totals remain nonnegative. A transition is any finite-window compatible
arrival at a different attracting equilibrium. The initial *rescue* summary
is narrower: its source must be the tracked high-E/low-I `seizure` branch and
its destination the tracked `quiescent` or `active_mid` branch. Arrival at
`herald` is recorded but excluded from rescue until the explicit target set
is changed. Transitions from other sources are recorded without a rescue
label. These names express the author's study roles, not certified biological
regimes.

The nominal model and every distinct in-bounds neighbor at `e_to_i ±0.5`,
`theta_off ±0.25`, and `tau_I/tau_E ±0.2` receive the same adaptive search.
Existing rescue drives are useful first probes but cannot establish the
absence of a different successful drive at a neighboring parameter value.
Output units checkpoint by case, parameter cell, baseline, and source;
rerunning the same command verifies and skips completed units. The runner
reconstructs context TOML and branch CSV files on every resume, replacing
files left incomplete by an interruption before it aggregates results. It
requires at least 2 GB available RAM and 5 GB free disk before and during
execution, and should run with one Julia worker. For example:

```bash
JULIA_NUM_THREADS=1 julia --project=. scripts/run_adaptive_rescue_study.jl \
  --config experiments/adaptive_rescue.toml \
  --output output/adaptive_rescue_exemplars
```

`--case` and `--cell` select a partial run; `--smoke` runs a reduced execution
check. Artifacts retain each sampled trial, observed outcome-change bracket,
per-protocol rescue presence, search contexts, source snapshots, and checksums.
An unobserved rescue means only that no sampled drive in the stated finite
domain and resolution reached a configured rescue target.

## Exact counts

The existing numerical equilibrium search and all basin and pulse outputs
retain `CompletenessNotCertified`. The separate
[interval certification driver](../certification/README.md) proves complete
equilibrium counts for the two selected zero-drive cases but does **not**
certify exact counts of all attractors. Its global cycle-exclusion gate fails
in both cases. Numerical root matching, a filled basin grid, and finite
pulse follow-up cannot close that gap. No manuscript claim is promoted by
these measurements.
