# Question-led figure redesign

`cases.toml` freezes 232 historical parameter records (eight anchors and 224
screen cases). Hashes identify those parameter records, not a reuse of their
historical numerical results. The scientific design and limits are in
[the figure study note](../../docs/figure_redesign_20261006.md).

## Render the delivered packet

The `data/` directory contains 174 checked CSV/TOML/source-history files
(about 6.7 MB), including the selected plot arrays and independent verification.
From the working copy:

```sh
julia --project=plotting scripts/render_figure_redesign.jl reproducibility/figure_redesign_20261006/data output/figure_redesign_replay
```

This writes five PDFs, SVGs, PNGs, and a browsable `index.html`. The dated local
render is `output/figure_redesign_20261006/`. Its provenance records the packet
manifest, renderer, plotting environment, Julia, and CairoMakie versions.

All five figures were visually reviewed after rendering. The packet records
176 passing independent-equation trajectory comparisons (maximum coordinate
discrepancy 1.6671e-9) and 99 agreeing independent geometry searches. See the
study note for test results, panel rationales, captions, and scientific limits.

## Recompute the measurements

Use Julia 1.10.12 and the recorded root/plotting environments. From the working
copy, the pipeline is:

```sh
julia --project=. scripts/run_figure_screen.jl experiments/figure_redesign.toml output/figure_screen_new
julia --project=. scripts/run_figure_details.jl experiments/figure_redesign.toml output/figure_screen_new output/figure_details_new cases
julia --project=. scripts/run_figure_details.jl experiments/figure_redesign.toml output/figure_screen_new output/figure_details_new sequence
julia --project=. scripts/run_figure_details.jl experiments/figure_redesign.toml output/figure_screen_new output/figure_details_new timescales
julia --project=. scripts/run_figure_details.jl experiments/figure_redesign.toml output/figure_screen_new output/figure_details_new interventions
julia --project=. scripts/run_figure_details.jl experiments/figure_redesign.toml output/figure_screen_new output/figure_details_new paired
julia --project=. scripts/run_figure_details.jl experiments/figure_redesign.toml output/figure_screen_new output/figure_details_new finalize
julia --project=. scripts/verify_figure_redesign.jl output/figure_details_new output/figure_details_new/verification
julia --project=. scripts/build_figure_redesign.jl output/figure_screen_new output/figure_details_new output/figure_packet_new
julia --project=plotting scripts/render_figure_redesign.jl output/figure_packet_new output/figure_redesign_new
```

The screen and detailed stages resume only against the same recorded sources
and configuration. Completed units are checked before reuse. Limit scientific
case concurrency to two; two Julia threads allow the existing case scheduler
to run both. A third argument to the screen runner limits case count for a
development check and explicitly leaves its completion flag false.

The full details can be restricted to one selected case by a fifth argument
after `cases`. This does not substitute for the required complete set at
`finalize`. The renderer and packet builder refuse existing destinations.

Raw source snapshots, geometry attempts, phase-specific maps, pulse outcomes,
controls, and continuation stops remain in the full local study. The portable
packet includes the plotted arrays, selection reasons, verification results,
and checksums. Historical and newly generated evidence remain separate.

The plotting packet stores axial pulse observations at every measured cycle
phase and the exact displayed windows of the connected trajectory. Joint-input
pulse maps and omitted settled intervals remain in the full study. Each filtered
CSV records its original digest, input/output row counts, and exact selection
rule in `packet.toml`; no observations are interpolated or relabeled.

An interrupted local run encountered recursive serialization of a periodic
continuation attempt's solver object. `scripts/repair_figure_serialization.jl`
accepts only the exact storage-only source replacement, preserves the previous
driver and metadata, and records both hashes. It rejects any additional source
change. This repair does not change an equation, solver option, candidate,
trajectory, or validation criterion. Fresh runs use the corrected serialization
directly. The local packet retains the repair record.
A second exact, metadata-only repair corrects the inherited screen replay command
in the detailed driver. Its command uses `SCREEN_OUTPUT` and `STAGE` as explicit
placeholders; the complete invocation sequence above supplies them. Both repair
histories preserve the preceding source and metadata and remain in the packet.
