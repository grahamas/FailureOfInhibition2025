# Selective-anchor paper figure data

`data/` is the portable input for the six Julia-rendered figures described in
[the figure review note](../../docs/paper_figures_20261005.md). It contains the
four main-figure traces, the 2,058 sampled tonic-input/source rows, the
established-state E-withdrawal rows, discovered equilibrium coordinates, and
SHA-256 checksums. The local full study archives are not required to render it.
Reviewed trace tables retain the first 500 ms and the full-follow-up endpoint;
the builder checks the untrimmed source traces before compacting them. Three
additional `preon_*.csv` traces simulate 50 ms at the unchanged input before
the depicted pulse or tonic-input change. Their time-zero states must match
the reviewed trajectory sources within `1e-7`; they do not replace the
reviewed post-onset trajectories or endpoint classifications.
The main-figure traces and tonic rows were rebuilt on 6 October from replays
after the merged phase-handoff correction. The established-state withdrawal
rows in S2 come from the frozen historical two-input batch and have not been
rerun under the corrected region-selection rule.

From a Julia 1.10.12 checkout with the recorded plotting environment, run:

```sh
julia --project=plotting scripts/render_paper_figures.jl \
  reproducibility/paper_figures_20261005/data output/paper_figures_new
```

The renderer rejects an existing output directory and checks every bundled
file against a complete expected file set, selected model/protocol sources, the
recorded Julia version and root/plotting project and manifest hashes, every
numerical source recorded in the joint and tonic references, and all four
reviewed reference files before writing PDF, SVG, and PNG files. Bundle
checksum paths use `/` on every platform. The reviewed source hashes predate
the runner consolidation. Rebuild from a separate working copy at change
`wkkotuwyqlsr` (the consolidation parent); the builder intentionally rejects
the newer runner sources. Point it at the retained archives in the main working
copy:

```sh
jj workspace add ../workspaces/paper-figure-replay -r wkkotuwyqlsr
cd ../workspaces/paper-figure-replay
julia --project=. scripts/build_paper_figure_data.jl \
  ../../main/output/selective_anchor_joint_postmerge_20261006 \
  ../../main/output/selective_tonic_e_release_postmerge_20261006 \
  ../../main/output/input_response_20260929_v3/anchors/selective_withdrawal/responses/baseline_1 \
  ../../main/output/paper_figure_data_rebuilt
```

The builder checks the first two local summaries byte for byte against their
tracked reviewed references and verifies every consumed joint trajectory and
tonic point/theta file against the
[reviewed source artifact digest](source_artifacts.toml) before and after
building the packet. It requires the same Julia 1.10.12 version and root and
plotting project/manifest files as the reviewed runs. It verifies the
complete shared source-hash table from both reviewed run summaries before
regenerating traces and records those hashes in `data.toml`. It verifies the
historical response archive's manifest and the exact frozen metadata,
parameters, input, roles, summary, and four S2 source tables against
[the response reference](response_reference.toml) before copying any S2 rows.
It also checks selected trajectory outcomes and endpoints against the tonic
records, and refuses an existing destination.
The supplement shows sampled points without interpolation. The figures do not
locate a bifurcation, certify all attractors, or establish biological state
identities. The selected time-scale ratio is exploratory (`0.2`), distinct from
the older manuscript ratio (`4.4`).
