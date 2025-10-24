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

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/grahamas/FailureOfInhibition2025.git")
```

## Quick Start

### Basic Usage Pattern

Here's the basic structure for using the Wilson-Cowan model (see examples/ for complete working code):

```julia
using FailureOfInhibition2025

# Create a spatial lattice
lattice = CompactLattice(extent=(10.0,), n_points=(21,))

# Create connectivity, stimulus, and nonlinearity objects
connectivity = GaussianConnectivityParameter(1.0, (2.0,))  # amplitude, spread
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
# (Use with ODE solvers like DifferentialEquations.jl)
# See examples/example_wilson_cowan.jl for complete usage
```

### Per-Population-Pair Connectivity

For more realistic neural network models, you can specify different connectivity kernels for each population pair:

```julia
using FailureOfInhibition2025

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

nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5)

# Use in Wilson-Cowan model
params = WilsonCowanParameters{2}(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (1.0, 0.8),
    connectivity = connectivity,
    nonlinearity = nonlinearity,
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E", "I")
)

# Compute derivatives
A = 0.1 .+ 0.05 .* rand(21, 2)
dA = zeros(size(A))
wcm1973!(dA, A, params, 0.0)  # Works!
```

The indexing convention follows matrix multiplication: `connectivity[i,j]` describes how population `j` affects population `i`. This means each row represents inputs to a target population, and each column represents outputs from a source population.

See `examples/example_connectivity_matrix.jl` for detailed demonstration.


## Examples

See the `examples/` directory for detailed usage examples:
- `examples/example_sigmoid.jl`: Demonstrates sigmoid nonlinearity usage
- `examples/example_wilson_cowan.jl`: Demonstrates Wilson-Cowan model usage
- `examples/example_connectivity_matrix.jl`: Demonstrates per-population-pair connectivity with ConnectivityMatrix

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

## Continuous Integration

This package uses GitHub Actions for automated testing across multiple Julia versions (1.9, 1.11, nightly) and operating systems (Ubuntu, Windows, macOS). The CI workflow includes:

- **Comprehensive testing**: Unit tests, integration tests, and example validation
- **Multi-platform support**: Testing on Linux, Windows, and macOS
- **Code quality checks**: Format checking and strict mode testing
- **Coverage reporting**: Automated coverage analysis with Codecov integration
- **Documentation building**: Automatic documentation generation (when available)

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