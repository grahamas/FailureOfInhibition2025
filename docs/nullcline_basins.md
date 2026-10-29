# Nullcline basin overlays

The mechanism panels draw initial conditions in `[0,0.5]^2`. Basin colouring
uses one persistent dense recurrence cache per autonomous model, with a
consistent 3×3 neighbourhood required before reusing a transient cell. Trajectories
may leave the displayed window while staying inside the physical `[0,1]^2`
domain. A neutral band marks sampled changes in basin identity; it is not a
certified boundary enclosure. Unknown destinations remain neutral.

Generate the mechanism alone, without a full screen or detailed study:

```sh
julia --project=. scripts/run_figure_details.jl mechanism-only experiments/figure_redesign.toml output/new_mechanism
julia --project=. -e 'include("scripts/verify_figure_redesign.jl"); VerifyFigureRedesign.verify_mechanism_evidence("output/new_mechanism")'
```

The configuration's `[nullcline_overlays]` table independently controls `basins`
and `separatrices`. `[nullcline_overlays.basin]` accepts `BasinGridOptions`
keywords (including `grid_points`); `[nullcline_overlays.separatrix]` accepts
`SeparatrixOptions` keywords. Figure evidence currently requires display bounds
`[[0.0,0.5],[0.0,0.5]]`; the package basin API supports other bounds. Library
usage can override generation without changing configuration:

```julia
include("scripts/run_figure_details.jl")
FigureDetails.run_mechanism("experiments/figure_redesign.toml", "output/new_mechanism";
    basins=true, separatrices=true)
```

Completed evidence is reusable only with matching source, configuration, model,
root identities, and unchanged complete file sets. Use a new output directory
when any of these change. The evidence contains explicit grid indices, root
labels, unresolved statuses, numerical options, integration statistics, and
both manifold-tracing attempts. Its checksums and independent numerical checks
remain separate evidence.

Render just the mechanism from that checked evidence, using the established
plotting environment:

```julia
include("scripts/render_figure_redesign.jl")
FigureRedesignRenderer.FigurePacketIO.verify_manifest("output/new_mechanism")
mkpath("output/new_mechanism_render")
FigureRedesignRenderer.mechanism("output/new_mechanism", "output/new_mechanism_render";
    show_basins=true, show_separatrices=true)
```

For a complete figure packet, normal generation and packet building include the
new evidence. `FigureRedesignData.with_mechanism_evidence(old_packet,
mechanism_evidence, new_packet)` can enrich a checked older packet when its
mechanism roots agree. Old packets without overlays remain renderable. Rendering
never launches simulations. Basin and manifold visibility can be changed
independently through renderer keywords; CLI switches are documented by its
usage error.

Reproduce warmed scaling measurements and independent coarse-grid/boundary
checks with:

```sh
julia --project=. scripts/benchmark_nullcline_basins.jl output/new_basin_benchmark
```

`timings.csv` records computation time, cached starts, basin hits, integration
work, and early/late trajectory times for 50², 100², and 200² grids. These are
times to a reusable basin region, not equilibrium settling times.
`validation.csv` distinguishes agreement, disagreement, and unassessed points;
unresolved labels do not count as agreement. `decision.toml` compares median
manifold/grid costs across three warmed runs per condition. The installed
separatrix default is fixed from this benchmark, not chosen on each render.

Attractors.jl is pinned to 2.0.5 because the path-retention adapter depends on
that release's recurrence state machine. Revalidate the adapter and its cache
and interruption tests before upgrading. The overlay maps only discovered
attracting equilibria; it does not identify periodic basins, prove completeness,
or promote provisional coordinate roles to biological conclusions.

The 2026-10-29 warmed benchmark measured median grid/manifold generation times
of 0.613/0.0061 seconds for the matched monotone model and 0.600/0.0109 seconds
for the failure model. Separatrices therefore default on, with
`--no-separatrices` available to disable drawing. Both models retained one
unconverged branch rather than displaying it. Across a full 17×17 grid per
model and targeted 100×100 transition samples, all 534 comparable destinations
agreed with independent integration; 244 neutral samples were unassessed.
These are bounded checks of the two mechanism models, not whole-grid guarantees.
