# FailureOfInhibition2025

This Julia package contains a deliberately narrow, CPU-only implementation of
a two-population Wilson-Cowan-type point model. It is being prepared as the
authoritative computational implementation for a failure-of-inhibition study.

The mathematical contract is not yet scientifically approved. In particular,
the inhibitory response, normalization, physical input domain, nonnegativity,
and limiting behavior remain author decisions. The current right-hand side is
preserved only as provisional behavior so that those decisions can be made and
tested explicitly.

## Point-model interface

States are two-element vectors ordered `[E, I]`. Coupling fields are named as
`source_to_target`, avoiding matrix-index ambiguity.

```julia
using FailureOfInhibition2025

excitatory = PopulationParameters(
    decay=0.7,
    saturation=1.0,
    timescale=2.0,
    response=LogisticResponse(slope=2.0, threshold=0.1),
)
inhibitory = PopulationParameters(
    decay=0.9,
    saturation=1.0,
    timescale=1.5,
    response=LogisticResponse(slope=1.5, threshold=0.2),
)
coupling = PointCoupling(
    e_to_e=1.2,
    i_to_e=-0.6,
    e_to_i=0.7,
    i_to_i=-0.4,
)
model = PointModelParameters(
    excitatory=excitatory,
    inhibitory=inhibitory,
    coupling=coupling,
)

solution = solve_point_model(
    [0.1, 0.15],
    (0.0, 20.0),
    model;
    saveat=0.1,
    abstol=1e-10,
    reltol=1e-10,
)
write_trajectory_csv("trajectory.csv", solution)
```

`write_trajectory_csv` writes the stable column order `time,E,I`.

## Noncanonical response candidates

The package exposes the formulations that must be evaluated during the next
scientific stage without selecting one as authoritative:

- `LogisticResponse`
- `RectifiedZeroedLogisticResponse`
- `DifferenceOfLogisticsCandidate`
- `DifferenceOfRectifiedZeroedLogisticsCandidate`

The difference candidates are not normalized or clamped. Depending on their
parameters, they may be negative. Their presence is for explicit comparison
and characterization, not a scientific endorsement.

## External drive

`NoDrive` contributes zero to both populations. `PiecewiseConstantDrive`
accepts separate E/I baseline and amplitude values and applies the amplitude
during inclusive time windows.

## Development

The repository tracks `Manifest.toml` because reproducibility takes priority
over library-style environment flexibility.

```julia
using Pkg
Pkg.instantiate()
Pkg.test()
```

Spatial fields, GPU execution, sensitivity analysis, parameter optimization,
plotting, and manuscript writing are outside this repository's current scope.
See `NEXT_STEPS.md` for the unresolved scientific work.
