# Benchmarking

This directory contains benchmarking code for tracking the performance of FailureOfInhibition2025 components and simulations.

## Structure

- **benchmark_utils.jl**: Utility functions for timing, CSV writing, and git commit tracking
- **benchmark_components.jl**: Benchmarks for individual components (nonlinearity, connectivity, stimulation)
- **benchmark_simulations.jl**: Benchmarks for 0D (point) and 1D simulations
- **run_benchmarks.jl**: Main runner that executes all benchmarks

## Usage

### Running All Benchmarks

```bash
julia --project=. test/benchmark/run_benchmarks.jl
```

### Running Specific Benchmark Suites

```julia
# Component benchmarks only
include("test/benchmark/benchmark_components.jl")
run_component_benchmarks()

# Simulation benchmarks only
include("test/benchmark/benchmark_simulations.jl")
run_simulation_benchmarks()
```

## Results

Benchmark results are saved to CSV files in the `benchmark_results/` directory:

- `component_benchmarks.csv`: Performance of individual components
- `simulation_benchmarks.csv`: Performance of 0D and 1D simulations

Each CSV file includes:
- **timestamp**: When the benchmark was run
- **commit_id**: Git commit SHA (for tracking changes over time)
- **benchmark_name**: Name of the benchmark
- **mean_time_s**: Mean execution time in seconds
- **min_time_s**: Minimum execution time
- **max_time_s**: Maximum execution time
- **std_time_s**: Standard deviation of execution time
- **n_runs**: Number of runs averaged

## CI Integration

Benchmarks are automatically run as part of the integration tests in CI. Results are uploaded as artifacts and retained for 90 days.

## What's Benchmarked

### Components

- **Nonlinearity**: SigmoidNonlinearity, RectifiedZeroedSigmoidNonlinearity, DifferenceOfSigmoidsNonlinearity
- **Connectivity**: GaussianConnectivity (1D), ScalarConnectivity, ConnectivityMatrix
- **Stimulation**: CircleStimulus (active/inactive), no stimulus

### Simulations

#### 0D (Point Models)
- 1 population without connectivity
- 2 populations with ScalarConnectivity
- 2 populations without connectivity
- 3 populations with full ConnectivityMatrix

#### 1D (Spatial Models)
- 51, 101, and 201 points with 1 population and Gaussian connectivity
- 101 points with 2 populations (E, I) and full ConnectivityMatrix

## Tracking Performance Over Time

The CSV format allows you to track how performance changes across commits:

```bash
# View all results for a specific benchmark
grep '1D: 101 points, 1 population' benchmark_results/simulation_benchmarks.csv

# Compare performance between commits
# (requires analysis tools or spreadsheet software)
```

## Notes

- Benchmarks use multiple runs (50-10000 depending on the operation) to get stable measurements
- A warmup run is performed before timing to avoid JIT compilation overhead
- Results may vary based on system load and hardware
