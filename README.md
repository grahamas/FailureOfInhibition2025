# FailureOfInhibition2025

A Julia package for neural field modeling with failure of inhibition mechanisms.

[![CI Tests](https://github.com/grahamas/FailureOfInhibition2025/actions/workflows/ci.yml/badge.svg)](https://github.com/grahamas/FailureOfInhibition2025/actions/workflows/ci.yml)

## Features

- **Sigmoid nonlinearities**: Standard and rectified zeroed sigmoid functions for neural modeling
- **Wilson-Cowan model**: Classic neural population dynamics model (Wilson & Cowan 1973)
- **Neural field models**: Implementation with customizable parameters  
- **Spatial connectivity**: Gaussian connectivity patterns with FFT-based convolution
- **Per-population-pair connectivity**: Each population pair can have its own connectivity kernel via ConnectivityMatrix
- **Stimulus handling**: Flexible stimulation interfaces
- **Multi-population support**: Support for multiple neural populations with flexible coupling
- **Simulation utilities**: Solve models over time using DifferentialEquations.jl and save results to CSV
- **Parameter sensitivity analysis**: Integrated support for computing parameter sensitivities using SciMLSensitivity.jl

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/grahamas/FailureOfInhibition2025.git")
```

## Quick Start

### Per-Population-Pair Connectivity

The Wilson-Cowan model requires a `ConnectivityMatrix{P}` that specifies connectivity for each population pair:

```julia
using FailureOfInhibition2025

# Create a spatial lattice
lattice = CompactLattice(extent=(10.0,), n_points=(21,))

# Define connectivity for each population pair
# connectivity[i,j] maps population j → population i
conn_ee = GaussianConnectivityParameter(1.0, (2.0,))    # E → E (excitatory)
conn_ei = GaussianConnectivityParameter(-0.5, (1.5,))   # I → E (inhibitory)
conn_ie = GaussianConnectivityParameter(0.8, (2.5,))    # E → I (excitatory)
conn_ii = GaussianConnectivityParameter(-0.3, (1.0,))   # I → I (inhibitory)

# Create connectivity matrix
connectivity = ConnectivityMatrix{2}([
    conn_ee conn_ei;   # Row 1: inputs to E from [E, I]
    conn_ie conn_ii    # Row 2: inputs to I from [E, I]
])

# Create stimulus and nonlinearity
stimulus = CircleStimulus(
    radius=2.0, strength=0.5, 
    time_windows=[(0.0, 10.0)], 
    lattice=lattice
)
nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5)

# Create Wilson-Cowan model parameters (2 populations: E and I)
params = WilsonCowanParameters{2}(
    α = (1.0, 1.5),          # Decay rates
    β = (1.0, 1.0),          # Saturation coefficients
    τ = (1.0, 0.8),          # Time constants
    connectivity = connectivity,
    nonlinearity = nonlinearity,
    stimulus = stimulus,
    lattice = lattice,
    pop_names = ("E", "I")   # Population names
)

# Set up initial state (21 spatial points × 2 populations)
A = 0.1 .+ 0.05 .* rand(21, 2)
dA = zeros(size(A))

# Compute derivatives using Wilson-Cowan equations
wcm1973!(dA, A, params, 0.0)
```

The indexing convention follows matrix multiplication: `connectivity[i,j]` describes how population `j` affects population `i`. This means each row represents inputs to a target population, and each column represents outputs from a source population.

See `examples/example_connectivity_matrix.jl` for detailed demonstration.

## Simulation

The package provides simulation utilities to solve Wilson-Cowan models over time using `DifferentialEquations.jl`:

```julia
using FailureOfInhibition2025

# Create model parameters
lattice = PointLattice()
connectivity = ConnectivityMatrix{2}([
    ScalarConnectivity(0.5) ScalarConnectivity(-0.3);
    ScalarConnectivity(0.4) ScalarConnectivity(-0.2)
])

params = WilsonCowanParameters{2}(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (10.0, 8.0),
    connectivity = connectivity,
    nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E", "I")
)

# Solve the model
A₀ = reshape([0.1, 0.1], 1, 2)  # Initial condition
tspan = (0.0, 100.0)             # Time span
sol = solve_model(A₀, tspan, params, saveat=0.1)

# Save results to CSV
save_simulation_results(sol, "results.csv", params=params)
save_simulation_summary(sol, "summary.csv", params=params)
```

See `examples/example_simulation.jl` for comprehensive simulation examples including point models, spatial models, and WCM 1973 modes.

## Parameter Sensitivity Analysis

The package integrates with SciMLSensitivity.jl to enable parameter sensitivity analysis for Wilson-Cowan models. This allows you to understand how changes in model parameters affect the dynamics.

```julia
using FailureOfInhibition2025
using SciMLSensitivity

# Create model parameters
lattice = PointLattice()
connectivity = ConnectivityMatrix{2}([
    ScalarConnectivity(0.5) ScalarConnectivity(-0.3);
    ScalarConnectivity(0.4) ScalarConnectivity(-0.2)
])

params = WilsonCowanParameters{2}(
    α = (1.0, 1.5),    # Decay rates
    β = (1.0, 1.0),    # Saturation coefficients
    τ = (10.0, 8.0),   # Time constants
    connectivity = connectivity,
    nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E", "I")
)

# Solve model with parameter structure for sensitivity analysis
A₀ = reshape([0.1, 0.1], 1, 2)
tspan = (0.0, 50.0)
result = compute_local_sensitivities(A₀, tspan, params, saveat=1.0)

# Use SciMLSensitivity.jl to compute sensitivities
loss(sol) = sum(abs2, sol[end])  # Define a loss function
sens = adjoint_sensitivities(result.solution, Tsit5(), loss, InterpolatingAdjoint())

println("Parameter sensitivities: ", sens)
```

The `compute_local_sensitivities` function:
- Extracts specified parameters (α, β, τ) into a vector for sensitivity analysis
- Solves the ODE system with the parametrized structure
- Returns a solution that can be used with SciMLSensitivity.jl functions

Supported sensitivity methods include:
- **InterpolatingAdjoint**: Efficient continuous adjoint method (recommended for most cases)
- **QuadratureAdjoint**: Alternative adjoint method using quadrature
- **BacksolveAdjoint**: Backsolve adjoint approach

See `examples/example_sensitivity_analysis.jl` for comprehensive sensitivity analysis examples.

## Examples

See the `examples/` directory for detailed usage examples:
- `examples/example_sigmoid.jl`: Demonstrates sigmoid nonlinearity usage
- `examples/example_wilson_cowan.jl`: Demonstrates Wilson-Cowan model usage
- `examples/example_connectivity_matrix.jl`: Demonstrates per-population-pair connectivity with ConnectivityMatrix
- `examples/example_point_model.jl`: Demonstrates non-spatial (point) models using PointLattice
- `examples/example_wcm1973_modes.jl`: Demonstrates the three dynamical modes from Wilson & Cowan 1973
- `examples/example_simulation.jl`: Demonstrates solving models over time and saving results
- `examples/example_sensitivity_analysis.jl`: Demonstrates parameter sensitivity analysis using SciMLSensitivity.jl

## Wilson-Cowan 1973 Validation

This package includes validated implementations of the three dynamical modes described in the seminal 1973 paper:

1. **Active Transient Mode** - Sensory neo-cortex behavior with self-generated transient responses
2. **Oscillatory Mode** - Thalamic behavior with sustained oscillations
3. **Steady-State Mode** - Archi-/prefrontal cortex with stable spatial patterns

For detailed parameter mappings, mathematical formulation, and usage instructions, see:
- `docs/wcm1973_validation.md` - Complete parameter mapping and validation documentation
- `test/test_wcm1973_validation.jl` - Comprehensive validation test suite
- `examples/example_wcm1973_modes.jl` - Usage examples for all three modes

**Reference:** Wilson, H. R., & Cowan, J. D. (1973). A mathematical theory of the functional dynamics of cortical and thalamic nervous tissue. *Kybernetik*, 13(2), 55-80.

## Point Models (Non-Spatial)

The package supports both spatial and non-spatial models using the same `wcm1973!` function. For non-spatial models (simple ODEs without spatial structure), use a `PointLattice` with `ScalarConnectivity` for population interactions:

```julia
using FailureOfInhibition2025

# Create a point lattice (zero-dimensional space)
lattice = PointLattice()

# Define scalar connectivity between populations
# For a 2-population (E, I) model: connectivity[i,j] maps j → i
conn_ee = ScalarConnectivity(1.0)    # E → E (excitatory self-connection)
conn_ei = ScalarConnectivity(-0.5)   # I → E (inhibitory to excitatory)
conn_ie = ScalarConnectivity(0.8)    # E → I (excitatory to inhibitory)
conn_ii = ScalarConnectivity(-0.3)   # I → I (inhibitory self-connection)

connectivity = ConnectivityMatrix{2}([
    conn_ee conn_ei;
    conn_ie conn_ii
])

# Create parameters for a non-spatial 2-population model
params = WilsonCowanParameters{2}(
    α = (1.0, 1.5),          # Decay rates
    β = (1.0, 1.0),          # Saturation coefficients
    τ = (1.0, 0.8),          # Time constants
    connectivity = connectivity,  # Population-to-population connectivity
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
    stimulus = nothing,
    lattice = lattice,       # PointLattice for non-spatial model
    pop_names = ("E", "I")
)

# Activity state: use (1, P) shape for point models with connectivity
# This maintains consistency with spatial models (N_spatial, P)
A = reshape([0.3, 0.5], 1, 2)  # Shape: (1, 2) for 1 point, 2 populations
dA = zeros(1, 2)

# Use the same wcm1973! function
wcm1973!(dA, A, params, 0.0)
```

**Note**: For point models *without* connectivity (connectivity=nothing), you can use a simpler 1D array: `A = [0.3, 0.5]`. For point models *with* connectivity, use the (1, P) shape to maintain consistency with how populations are indexed in spatial models.

This allows you to:
- Model population interactions in non-spatial systems
- Test model dynamics without spatial complications
- Perform parameter exploration for non-spatial models
- Implement mean-field approximations
- Use the classical Wilson-Cowan equations (original 1972/1973 formulation)
- Run faster simulations when spatial structure is not needed

## Implementation Notes

This package reimplements the Wilson-Cowan model from [WilsonCowanModel.jl](https://github.com/grahamas/WilsonCowanModel) with key architectural differences:

1. **No callable objects**: Uses plain structs with separate functions instead of functors
2. **Direct function dispatch**: Parameters are passed directly to `wcm1973!()` instead of creating intermediate "Action" objects
3. **Functional programming style**: Follows a more functional approach without object-oriented patterns

These differences make the code simpler and more composable while maintaining the same mathematical behavior.

## Testing

Run the test suite:

```julia
using Pkg
Pkg.test("FailureOfInhibition2025")
```

Or run tests directly:
```bash
julia --project=. test/runtests.jl
```

## Benchmarking

Performance benchmarks are available to track the runtime of components and simulations:

```bash
julia --project=. test/benchmark/run_benchmarks.jl
```

Benchmarks include:
- **Component benchmarks**: Nonlinearity, connectivity, and stimulation functions
- **0D simulations**: Point models (simple ODEs) with 1-3 populations
- **1D simulations**: Spatial models with 51-201 points and 1-2 populations

Results are saved to CSV files with timestamp and commit ID for tracking performance changes over time. See `test/benchmark/README.md` for details.

## Continuous Integration

This package uses GitHub Actions for automated testing across multiple Julia versions (1.9, 1.11, nightly) and operating systems (Ubuntu, Windows, macOS). The CI workflow includes:

- **Comprehensive testing**: Unit tests, integration tests, and example validation
- **Multi-platform support**: Testing on Linux, Windows, and macOS
- **Code quality checks**: Format checking and strict mode testing
- **Coverage reporting**: Automated coverage analysis with Codecov integration
- **Documentation building**: Automatic documentation generation (when available)
- **Performance benchmarking**: Automated benchmarks run as part of integration tests

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass locally
6. Submit a pull request

The CI system will automatically test your changes across all supported platforms and Julia versions.

## License

[Add license information here]