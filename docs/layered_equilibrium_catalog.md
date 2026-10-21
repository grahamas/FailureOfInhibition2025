# Three-layer equilibrium descriptions

The catalog now provides an explicit separation between continuous activity,
structure/local dynamics, and deferred biological interpretation. No activity
bands or biological candidate names are assigned by this view.

## Interface and storage

`EquilibriumCatalog.layers(d)` returns four named groups:

- `identity`: original catalog-local `context_id` and `root_id`.
- `activity`: exact original `E` and `I` occupancies, each in [0, 0.5].
- `structure_dynamics`: original `E_arm`, `I_branch`, `stability`,
  `near_singular`, `lambda_max_real`, `lambda_min_real`, and `omega`.
- `interpretation`: `status = :deferred`, `label = nothing`.

Real parts have units ms⁻¹; omega is angular frequency in rad/ms. Original
numerical stability and unresolved codes are preserved, not recomputed from
rounded eigenvalues. The inhibitory branch describes the response function,
not a numbering of inhibitory-nullcline folds. IDs must be joined with the
catalog manifest hash; they are not continuation branch identities.

`EquilibriumCatalog.layered_text(d; digits=4)` gives a human-readable description.
Rounding affects only that string. All descriptors apply independently of
coexisting roots. Structure never supplies an implicit biological label.

The catalog schema and root files are unchanged. Grouped views are constructed
in memory. The interpretation policy is stored once in gallery metadata;
no redundant root annotation table is generated. The summary packet reuses the
existing structural-distribution format. Earlier candidate-label experiments
remain separate and are not invoked by the new interface or renderer.

## Scientific views

The gallery retains the six structural-distribution figures: quantile intervals,
exact empirical CDFs, inhibitory occupancy histograms, stability composition,
and separate screen/detailed joint E–I histograms. It adds:

1. **Source stability across activity space.** All roots, including unresolved
   and near-singular records, shown at exact E/I coordinates. Color and marker
   encode source stability. This locates unstable equilibria omitted from the
   attracting-only occupancy plots.
2. **Local dynamics across activity space.** Attracting, nonsingular records,
   colored by largest eigenvalue real part on a shared linear scale. Circles
   denote a real spectrum; triangles denote a complex pair. This asks how local
   decay and oscillatory modes vary across the activity plane.

All scatter records are drawn in context/root ID order with no subsampling.
Overlapping points can obscure other stability or spectrum types; the scatter
plots cannot establish absence or estimate frequencies. Counts by cohort,
stability, spectrum type, and near-singularity are available in
`dynamics_counts.csv`, including zero-count combinations.

The existing occupancy plots show concentrations near the E/I endpoints and
an interior band associated with middle-E-arm attracting roots. These remain
descriptive patterns under adaptive sampling. Screen and detailed cohorts
remain separate, overlap, and do not represent independent replication or
physiological prevalence. A complex attracting spectrum describes local damped
modes, not an established periodic orbit. Biological interpretation is deferred;
continuation and finite-perturbation validation are future investigations.

## Reproduce

Run from the working copy with fresh output directories:

```sh
julia --project=. scripts/catalog_structural_distributions.jl output/equilibrium_catalog_20261016 output/catalog_layered_summaries_20261021
julia --project=plotting scripts/render_layered_catalog.jl output/equilibrium_catalog_20261016 output/catalog_layered_summaries_20261021 output/catalog_layered_figures_20261021
python3 scripts/verify_layered_catalog.py output/equilibrium_catalog_20261016 output/catalog_layered_summaries_20261021 output/catalog_layered_figures_20261021
```

The gallery is `output/catalog_layered_figures_20261021/index.html`, with PNG,
PDF, and SVG exports. Source hashes and exact file manifests bind outputs to
the catalog and implementation. The independent verifier reconstructs every
structural summary and histogram bin, checks dynamics counts, parses all SVGs,
and verifies current analysis/renderer/reader hashes. Numerical discovery remains
`CompletenessNotCertified`.

## Verification of the 2026-10-21 run

- Focused catalog tests passed 633 assertions (48 existing and 585 layered-view
  checks), covering every structural/stability code, endpoints, signed zero,
  original identity and spectrum, near-singularity, and deferred interpretation.
- The independent verifier reproduced 108 distributions, 5,400 one-dimensional
  bins, 80,194 nonempty joint bins, 27 structural summaries, and 32 dynamics
  groups covering 214,901 records across the two plotted cohorts.
- All eight PNGs were visually inspected; the new local-dynamics PDF was also
  rasterized and inspected. All eight SVGs parsed successfully. Shared scales,
  endpoint visibility, filtering, legends, and overlap limitations were checked.
- Summary data occupy 301,962 bytes; the additional dynamics counts occupy
  921 bytes. The complete eight-figure gallery, including all export formats,
  occupies 9,479,256 bytes. The original catalog has no added per-root bytes.
- The first renderer attempt failed on a tuple-assignment syntax error before
  creating output. After correction the complete renderer succeeded. The
  separate plotting test suite was not run; rendering was tested end to end.
- The full Julia package suite passed **5,745/5,745 assertions** in 9m07.0s.
  `git diff --check` also passed. No dependencies or production classifier rules
  changed.
