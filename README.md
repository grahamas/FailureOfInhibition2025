# FailureOfInhibition2025

A Julia 1.10 package for a deterministic, CPU-only, two-population point model.
States are dimensionless active-population fractions ordered `[E, I]` in
`[0,1]^2`. [The model reference](docs/model.md) defines the equations, response
families, coupling convention, numerical methods, and output contracts.
[The analytical notes](docs/theory_notes.md) state the assumptions behind the
equilibrium-ordering and positive-drive restrictions.

The package provides matched monotone-control and failure-of-inhibition models,
equilibrium discovery and local stability, finite-window trajectory diagnostics,
equilibrium and periodic-orbit continuation, pulse experiments, and basin
measurements. Numerical searches report discovered results and unresolved
outcomes; they do not certify a complete attractor set or a biological state.
The trajectory CSV column order is `time,E,I`.

## Current figure study

The five-figure candidate's questions, historical measurements, limits, and
caption requirements are in [the figure study note](docs/figure_redesign_20261006.md).
The parameter registry and protocol are fixed by
`reproducibility/figure_redesign_20261006/cases.toml` and
`experiments/figure_redesign.toml`. The generated candidate packet is outside
the repository. Use Julia
1.10.12 and [the study workflow](reproducibility/figure_redesign_20261006/README.md)
to make new measurements, build a checked packet, and render figures. Final
result selection will determine which data and provenance need an active packet.

## Development

The root and plotting environments have tracked manifests. From the working
copy, instantiate and run the test suites with:

```sh
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=plotting -e 'using Pkg; Pkg.instantiate()'
julia --project=plotting plotting/test/runtests.jl
```

[Current next steps](NEXT_STEPS.md) concern author review of the figure claims
and any further numerical work needed for the final paper. Scientific parameter
choices, biological wording, and manuscript edits remain with the author.
