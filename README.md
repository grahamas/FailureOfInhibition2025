# FailureOfInhibition2025

This Julia 1.10 package implements a deterministic, CPU-only,
two-population point model for a failure-of-inhibition study. The implemented
model assumptions and numerical conventions are documented in
[`docs/model.md`](docs/model.md).

States are dimensionless active-population fractions ordered `[E, I]` in the
physical domain `[0,1]^2`. Physiological rest is **quiescent, not silent**:
the logistic response can produce small nonzero recruitment at zero effective
input, so rest need not be the exact origin.

Version 0.3 is an intentional pre-1.0 API break: `PopulationParameters` no
longer has `decay` or `saturation` fields, and inhibitory couplings are supplied
as nonnegative magnitudes rather than signed coefficients.

## Point-model interface

The example below constructs matched control and failure-of-inhibition models.
The four coupling values are nonnegative `source_to_target` magnitudes;
inhibitory-source terms are subtracted internally.

```julia
using FailureOfInhibition2025

excitatory = PopulationParameters(
    timescale=2.0,
    response=LogisticResponse(slope=2.0, threshold=0.1),
)
inhibitory_control = PopulationParameters(
    timescale=1.5,
    response=LogisticResponse(slope=1.5, threshold=0.2),
)

coupling = PointCoupling(
    e_to_e=1.2,
    i_to_e=0.6,
    e_to_i=0.7,
    i_to_i=0.4,
)

pulses = [
    DrivePulse(onset=2.0, offset=5.0, increment=(0.20, 0.10)),
    DrivePulse(onset=4.0, offset=7.0, increment=(0.05, 0.00)),
]
drive = PiecewiseConstantDrive(
    baseline=(0.0, 0.0),
    pulses=pulses,
    interpretation=AfferentExcitation,
)

models = matched_point_models(
    excitatory=excitatory,
    inhibitory_control=inhibitory_control,
    failure_threshold=1.1,
    coupling=coupling,
    drive=drive,
)

control_solution = solve_point_model(
    [0.1, 0.15],
    (0.0, 20.0),
    models.control;
    saveat=0.1,
    abstol=1e-10,
    reltol=1e-10,
    domain_atol=1e-8,
)
failure_solution = solve_point_model(
    [0.1, 0.15],
    (0.0, 20.0),
    models.failure_of_inhibition;
    saveat=0.1,
    abstol=1e-10,
    reltol=1e-10,
    domain_atol=1e-8,
)

write_trajectory_csv("control.csv", control_solution)
write_trajectory_csv("failure_of_inhibition.csv", failure_solution)
```

`matched_point_models` returns a named tuple with `control` and
`failure_of_inhibition` models. They share all supplied parameters and the
inhibitory onset response; only the latter subtracts the equal-slope failure
component.

## Responses and external drive

`LogisticResponse` is the supported excitatory response and monotone
inhibitory control. `FailureOfInhibitionResponse` is the raw,
unnormalized, unclamped equal-slope difference of ordered logistic responses.
It can also be constructed directly:

```julia
onset = LogisticResponse(slope=1.5, threshold=0.2)
foi = FailureOfInhibitionResponse(onset; failure_threshold=1.1)
```

`DrivePulse` uses the half-open interval `[onset, offset)`. Pulses are additive,
so the two example pulses overlap and sum between 4 and 5 ms.
`AfferentExcitation` requires nonnegative total E/I drive on every constant
segment; use `AbstractIntervention` when a protocol has signed totals. Solver
stops and saved times automatically include every in-span pulse transition.

The exported rectified and unequal-slope difference response candidates remain
available for standalone comparisons, but `PointModelParameters` rejects them
as supported population responses.

## Physical-domain and output guarantees

`solve_point_model` rejects nonfinite initial states and initial states outside
`[0,1]^2`. It checks accepted steps and returned states against
`[-domain_atol, 1+domain_atol]`, with a default `domain_atol=1e-8`, and throws
`DomainError` for a larger excursion. It never clips or projects states. The
low-level `point_rhs!` and `point_jacobian!` functions remain evaluable outside
the physical domain for numerical methods and mathematical diagnostics.

`write_trajectory_csv` preserves the stable column order:

```text
time,E,I
```

## Development

The repository tracks `Manifest.toml` because reproducibility takes priority
over library-style environment flexibility.

```julia
using Pkg
Pkg.instantiate()
Pkg.test()
```

Equilibria, stability and bifurcation analysis, regime diagnostics,
deterministic experiments, plotting, and manuscript writing remain future
work. See `NEXT_STEPS.md` for the current sequence.
