# Selective-anchor paper figure data

`data/` is the portable input for the six Julia-rendered figures described in
[the figure review note](../../docs/paper_figures_20261005.md). It contains the
four main-figure traces, the 2,058 sampled tonic-input/source rows, the
established-state E-withdrawal rows, discovered equilibrium coordinates, and
SHA-256 checksums. The local full study archives are not required to render it.
Trace tables retain the first 500 ms and the full-follow-up endpoint; the
builder checks the untrimmed source traces before compacting them.
The main-figure traces and tonic rows were rebuilt on 6 October from replays
after the merged phase-handoff correction. The established-state withdrawal
rows in S2 come from the frozen historical two-input batch and have not been
rerun under the corrected region-selection rule.

From a Julia 1.10 checkout with the recorded plotting environment, run:

```sh
julia --project=plotting scripts/render_paper_figures.jl \
  reproducibility/paper_figures_20261005/data output/paper_figures_20261005
```

The renderer rejects an existing output directory and checks every bundled
file, selected model/protocol sources, and both reviewed reference summaries
before writing PDF, SVG, and PNG files. To rebuild the bundle from the retained
local archives instead, use the Julia builder:

```sh
julia --project=. scripts/build_paper_figure_data.jl \
  output/selective_anchor_joint_postmerge_20261006 \
  output/selective_tonic_e_release_postmerge_20261006 \
  output/input_response_20260929_v3/anchors/selective_withdrawal/responses/baseline_1 \
  output/paper_figure_data_rebuilt
```

The builder checks the first two local summaries byte for byte against their
tracked reviewed references, checks the selected trajectory outcomes and
endpoints against the tonic records, and refuses an existing destination.
The supplement shows sampled points without interpolation. The figures do not
locate a bifurcation, certify all attractors, or establish biological state
identities. The selected time-scale ratio is exploratory (`0.2`), distinct from
the older manuscript ratio (`4.4`).
