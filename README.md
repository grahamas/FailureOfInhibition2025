# FailureOfInhibition2025

A Julia package for neural field modeling with failure of inhibition mechanisms.

[![CI Tests](https://github.com/grahamas/FailureOfInhibition2025/actions/workflows/ci.yml/badge.svg)](https://github.com/grahamas/FailureOfInhibition2025/actions/workflows/ci.yml)

## Features

- **Sigmoid nonlinearities**: Standard and rectified zeroed sigmoid functions for neural modeling
- **Wilson-Cowan model**: Classic neural population dynamics model (Wilson & Cowan 1973)
- **Neural field models**: Implementation with customizable parameters  
- **Spatial connectivity**: Gaussian connectivity patterns with FFT-based convolution
- **Stimulus handling**: Flexible stimulation interfaces
- **Multi-population support**: Support for multiple neural populations

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/grahamas/FailureOfInhibition2025.git")
```

## Quick Start

```julia
using FailureOfInhibition2025

# Create a sigmoid nonlinearity
sigmoid = SigmoidNonlinearity(a=2.0, θ=0.5)

# Create Wilson-Cowan model parameters (2 populations: E and I)
params = WilsonCowanParameters{2}(
    α = (1.0, 1.5),          # Decay rates
    β = (1.0, 1.0),          # Saturation coefficients
    τ = (1.0, 0.8),          # Time constants
    connectivity = nothing,   # Connectivity (to be configured)
    nonlinearity = sigmoid,
    stimulus = nothing,       # Stimulus (to be configured)
    pop_names = ("E", "I")   # Population names
)

# Set up initial state (3 spatial points × 2 populations)
A = [0.3 0.2; 0.5 0.4; 0.7 0.6]
dA = zeros(size(A))

# Compute derivatives using Wilson-Cowan equations
wcm1973!(dA, A, params, 0.0)

# See examples/ directory for detailed usage
```

## Examples

See the `examples/` directory for detailed usage examples:
- `examples/example_sigmoid.jl`: Demonstrates sigmoid nonlinearity usage
- `examples/example_wilson_cowan.jl`: Demonstrates Wilson-Cowan model usage

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