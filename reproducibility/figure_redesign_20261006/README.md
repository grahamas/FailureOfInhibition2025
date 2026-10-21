# Five-figure candidate workflow

`cases.toml` freezes 232 parameter records (eight anchors and 224 screen
cases) as inputs for a new run. The scientific design and limits are in
[the figure study note](../../docs/figure_redesign_20261006.md).

Generated packets and rendered figures go to ignored local output. The current
tree contains the inputs and tools to make a new packet. Final result selection
will determine which data and provenance to preserve as the active
reproducibility record.

## Run the measurements and render

Use Julia 1.10.12 and the tracked root/plotting environments. From the working
copy, the pipeline is:

```sh
julia --project=. scripts/run_figure_screen.jl experiments/figure_redesign.toml output/figure_screen_new
julia --project=. scripts/run_figure_details.jl experiments/figure_redesign.toml output/figure_screen_new output/figure_details_new cases
julia --project=. scripts/run_figure_details.jl experiments/figure_redesign.toml output/figure_screen_new output/figure_details_new mechanism
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
and configuration; detail selection and packet building also require matching
screen and detail configuration hashes. Completed units are checked before
reuse. Limit scientific case concurrency to two; two Julia threads allow the
existing case scheduler to run both. A third argument to the screen runner
limits case count for a development check and leaves its completion flag false.

Selection records the screen's complete checksum manifest. A changed screen
requires a new detail study and selection. Verification records the exact
completed detail-stage manifests. If a detail stage is regenerated afterward,
move the old `verification/` directory aside and rerun the verifier before
building a packet. The builder rejects stale screen or detail evidence even
when regenerated stages have valid markers.

Figure 5 switching attempts retain each driven pulse and available baseline
continuation, including the leg that ends an unsuccessful cycle. Independent
verification replays both phases, checks their state handoffs and destination
roots, and checks the original E+0.5 and I+3 pulses against their declared
100 ms protocol. Every attempted leg requires a continuation before packet
building. The resulting report is bound to the completed detail stages.
Remeasured switching legs must match the deterministic interior witnesses from
the recorded finite search; their pulse and return trajectories are checked
independently.

Cycle-phase confirmations and paired temporary-pulse controls also retain both
phases locally. Verification replays their trajectories, checks the declared
source and handoff, and compares the outcome with the saved equilibrium or
validated orbit. For a reported unsuccessful replacement search, verification
replays the complete sampled pulse grid and its stopping rule before accepting
the bounded `not_demonstrated_within_protocol` label.

The connected sequence is replayed from its declared rest root through every
reported source and destination. Every plotted axial pulse in Figure 3B/C is
replayed from its declared equilibrium. Paired sustained outcomes are checked
against both the original and released contexts. Every axial cycle-phase pulse
row used to choose or draw Figure 4D is independently replayed from its declared
orbit phase; the joint-input maps remain in the local study but are not used by
that panel.

Figure 1 mechanism data is a completed detail stage. Its responses, nullclines,
reported equilibria, stability labels, and reference roles are checked against
independently written equations before packet building. Figure 2's sampled
geometry signatures are checked against the recorded roots and independent
equations; filled cells require agreeing corner and center samples.
Verification checks periodic closure and attraction for phase sources,
independently recomputes the Figure 2 time-scale spectra and continued orbit
summaries, and replays every connected step plotted in Figure 3D's input-history
sweep.

The full details can be restricted to one selected case by a fifth argument
after `cases`. This does not substitute for the required complete set at
`finalize`. The renderer and packet builder refuse existing destinations.

Raw source snapshots, geometry attempts, phase-specific maps, pulse outcomes,
controls, and continuation stops remain in the full local study. A generated
packet includes the plotted arrays, selection reasons, verification results,
and checksums. Rendered figures and their provenance are written to the
specified output directory. All of these outputs are ignored by Git.

The plotting packet stores axial pulse observations at every measured cycle
phase and the exact displayed windows of the connected trajectory. Joint-input
pulse maps and omitted settled intervals remain in the full study. Each filtered
CSV records its original digest, input/output row counts, and exact selection
rule in `packet.toml`; no observations are interpolated or relabeled.
The phase panel uses only the B=(2,1) baseline and its sole validated orbit;
multiple validated orbits require an explicit author selection before rendering.
