# FailureOfInhibition2025

A Julia package for neural field modeling with failure of inhibition mechanisms.

[![CI Tests](https://github.com/grahamas/FailureOfInhibition2025/actions/workflows/ci.yml/badge.svg)](https://github.com/grahamas/FailureOfInhibition2025/actions/workflows/ci.yml)

## Features

- **Sigmoid nonlinearities**: Standard and rectified zeroed sigmoid functions for neural modeling
- **Neural field models**: WCM 1973 implementation with customizable parameters  
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

# Test basic sigmoid function
result = simple_sigmoid(0.5, 2.0, 0.5)  # Returns 0.5

# Use with neural field models
# See examples/example_sigmoid.jl for more detailed usage
```

## Examples

See the `examples/` directory for detailed usage examples:
- `examples/example_sigmoid.jl`: Demonstrates sigmoid nonlinearity usage

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