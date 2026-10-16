# Equilibrium nomenclature sensitivity — 2026-10-16

This packet evaluates candidate activity-based descriptions without changing
the existing classifier. See `docs/nomenclature_sensitivity_20261016.md` for
the author decisions, exact predicates, results, and remaining choices.

## Replay

Use the repository's recorded Julia 1.10 environments. No simulations or new
packages are needed. Run from the working-copy root:

```sh
julia --project=. scripts/build_nomenclature_sensitivity.jl reproducibility/equilibrium_audit_20261016/data output/nomenclature_new
julia --project=. scripts/verify_nomenclature_sensitivity.jl output/nomenclature_new reproducibility/equilibrium_audit_20261016/data
julia --project=plotting scripts/render_nomenclature_sensitivity.jl output/nomenclature_new output/nomenclature_figures_new
```

Render the retained packet directly with:

```sh
julia --project=plotting scripts/render_nomenclature_sensitivity.jl reproducibility/nomenclature_sensitivity_20261016/data output/nomenclature_replay
```

Builders and renderers refuse existing destinations. `checksums.toml` binds
every packet file; `study.toml` binds the source audit manifest and analysis
sources. The audit manifest in turn binds the original model and registry.
Rendering records the renderer and input-manifest hashes separately. Historical
packets remain unchanged. Diagnostic output is not a replacement manuscript figure.

## Contents

- `inventory_summary.csv` preserves counts for every source scope, stability,
  and near-singular status. Analysis cohorts are screen and detailed geometry,
  separately, restricted to attracting non-near-singular roots.
- `cdf_*` gzip chunks retain cumulative counts at every distinct observed E,
  I, and max(E,I). `quantiles_*` save 101 observed-value quantiles per coordinate.
- `surfaces_*` retain 101×101 independent two-cutoff counts for each of rest,
  seizure, and herald. Equality is inclusive for low rest bounds and high
  herald bounds; seizure requires strictly greater E and inclusive low I.
- `scenarios_*` enumerate all 729 independent six-boundary combinations per
  cohort, saving quantile ranks and exact cutoffs. `scenario_counts_*` retain
  each membership count, overlaps, and gaps, without priority rules.
- `membership_support_*` retains all 135,233 included root records and the
  number of scenarios admitting each membership. Root context/index identifies
  the original inventory record; memberships are not probabilities.
- `activity_probes_*` provides 34 readable occupancy-scale probes per cohort:
  rest ceiling targets {0.001,0.01,0.02,0.05} independently for E/I; seizure
  E targets {0.01,0.02,0.05}, I targets {0.001,0.01,0.05}; herald E targets
  {0.25,0.4,0.45}, I targets {0.1,0.3,0.45}. Each target snaps independently to
  the nearest observed coordinate, with the lower value breaking an exact tie.
  Both target and actual boundary are saved. No target is adopted scientifically.
- `old_roles_*` compares memberships to original local roles.
  `intervention_destinations.csv` retains all 84 actual destinations with
  original local/reported names and inheritance provenance; `interventions_*`
  evaluates them under both cohort grids, for 122,472 destination/scenario rows.
- `dynamics_*` contains reconstructed Jacobians and local decay times for all
  48,381 attracting lower/middle-arm records. `representatives_*` selects the
  smallest max(E,I) per circuit and arm, with context/root tie breaking.
- Copied seed, root, continuation-connection, and fold-check tables preserve
  the existing bounded branch evidence, including its unresolved comparisons.

Chunks are lossless CSV compressed with deterministic gzip. Concatenate parsed
rows of a given prefix; ordering of chunk filenames is not scientific ordering.
Display limits and any omitted numerical zeros are disclosed on the figures.
No activity normalization or claim of population prevalence is introduced.

## Checks

The independent verifier compares root identities and retained fields against
the source inventory; directly enumerates every six-boundary scenario and every
root's membership support; verifies all intervention/scenario memberships and
exact CDFs; and checks a 10×10 boundary subgrid for each two-boundary surface
plus every readable occupancy probe. It does not invoke the analysis predicate
or prefix-counting implementation.

Focused tests cover boundary equality, overlapping definitions, multiple members,
absence of contextual/stability/arm prerequisites, no-turning-point-compatible
inputs, low-E/high-I non-rest, low-I ascending non-seizure, response peaks,
invalid inputs, and rectangular counting against direct enumeration.

The retained run passed 2,126 independent aggregate comparisons, all 135,233
root membership-support checks, and all 122,472 intervention/scenario checks.
The full package suite passed 5,026 assertions (including 31 new assertions
also run separately); the plotting suite passed 207.
All four rendered figures were visually reviewed, with a reduced-size PDF
inspection of the low-activity figure; all SVGs parsed and render provenance
verified. Original audit and manuscript-figure manifests verified unchanged.
The packet contains 221 files, totaling approximately 13.2 MB.
