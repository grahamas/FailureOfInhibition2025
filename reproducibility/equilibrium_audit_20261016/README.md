# Unassigned-equilibrium investigation

This packet supports the [scientific report](../../docs/equilibrium_audit_20261016.md).
It characterizes the October 6 screen and detailed study without changing
their role assignments or main figures.

The checked data packet contains 433 files (about 33 MB). Its lossless root and
context chunks were compared row-for-row against the complete audit outputs.

## Inspect and render

The local diagnostic gallery is `output/equilibrium_audit_figures_20261016/index.html`.
From a working copy with the recorded Julia 1.10 environments:

```sh
julia --project=plotting scripts/render_equilibrium_audit.jl reproducibility/equilibrium_audit_20261016/data output/equilibrium_audit_replay
```

The renderer verifies the exact file set and SHA-256 hashes, then writes five
PDF/SVG/PNG figures, a gallery, and rendering provenance. It refuses an existing
output directory. No numerical simulation is needed to render the packet.

All five final diagnostic figures were visually reviewed, including a
reduced-size PDF rasterization of Figure 5. The final SVGs parsed successfully,
and the rendered-output manifest and renderer/packet provenance were verified.
The original October 6 figure packet and figure manifests were reverified
unchanged.

## Contents and identities

- `inventory_chunks.csv` indexes lossless gzip CSV chunks containing all
  215,657 root records. Concatenate their parsed rows to reconstruct the full
  inventory. Counts include repeated contexts across recorded study stages.
- `contexts_*.csv.gz` retain the 77,455 context records and source locations;
  `audit.toml` records the verified source digests and inventory design.
- `seeds.csv` and `seed_parameters.toml` retain the eleven selected states,
  selection reasons, parameters, and original source identities.
- `seed_*_points.csv`, attempts, candidate brackets, and connections retain
  both default and half-step continuation variants. Scalar records avoid
  serialization of solver internals.
- `continuation_checks.csv` retains the first 173 independent-search and
  refinement comparisons. `fold_checks.csv` and `fold_attempts.csv` supplement
  them without overwriting the original results. Twenty-nine refinement
  comparisons remain unconfirmed; consult the report before using a branch.
- `intervention_labels.csv` distinguishes local names, reference-coordinate
  matches, and numbered destinations. `phase_labels.csv` maps recorded pulse
  destinations at all sixteen phases. `timescale_labels.csv` contains 860
  spectrum observations, including the original unresolved-stability labels.
- Nullclines, selected roots, the existing periodic orbit, and intervention
  coordinates are saved plot arrays. Map roots are sampled observations;
  no region interpolation is added.

`packet.toml` binds scientific source files and upstream manifests.
`checksums.toml` verifies the portable packet itself. Original full study
archives and detailed follow-up context searches remain in local `output/`;
their digests and locations are retained. The portable packet supports the
figures and audit tables, not a claim to include every original solver attempt.

## Recompute from the retained current studies

```sh
julia --project=. scripts/run_equilibrium_audit.jl output/figure_screen_20261006 output/figure_details_20261006 output/equilibrium_audit_new
julia --project=. scripts/continue_equilibrium_audit.jl output/equilibrium_audit_new output/equilibrium_continuation_new
julia --project=. scripts/check_equilibrium_fold_matches.jl output/equilibrium_audit_new output/equilibrium_continuation_new output/equilibrium_folds_new
julia --project=. scripts/build_equilibrium_audit_packet.jl output/equilibrium_audit_new output/equilibrium_continuation_new output/equilibrium_folds_new output/equilibrium_packet_new
julia --project=plotting scripts/render_equilibrium_audit.jl output/equilibrium_packet_new output/equilibrium_figures_new
```

This study targets the named October 6 source studies; continuation uses their
retained selected-case registry. Earlier studies are not interchangeable inputs.
The scripts refuse existing destinations. Preserve any failed run separately
before restarting; do not modify source archives or previously completed outputs.

The two initial development failures were a test fixture placed on the wrong
side of the response peak and a continuation-check access to the solve wrapper
instead of its `attempt` field. Both were corrected and covered by tests.
They did not change an equation, existing classifier, or original figure data.
