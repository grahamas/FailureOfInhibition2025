"""
Utilities for benchmarking components and simulations.
"""

using FailureOfInhibition2025
using Printf
using Dates
using BenchmarkTools
using Statistics

"""
    get_git_commit_id()

Get the current git commit ID, or "unknown" if not in a git repository.
"""
function get_git_commit_id()
    try
        commit = strip(read(`git rev-parse --short HEAD`, String))
        return commit
    catch
        return "unknown"
    end
end

"""
    benchmark_function(f::Function, name::String; samples::Int=100, evals::Int=1, seconds::Real=5)

Benchmark a function using BenchmarkTools and return statistics.

# Arguments
- `f::Function`: The function to benchmark
- `name::String`: Name of the benchmark
- `samples::Int=100`: Minimum number of samples to collect
- `evals::Int=1`: Number of evaluations per sample
- `seconds::Real=5`: Maximum time budget in seconds

Returns a NamedTuple with:
- name: benchmark name
- mean_time: mean execution time in seconds
- min_time: minimum execution time in seconds
- max_time: maximum execution time in seconds
- std_time: standard deviation of execution time in seconds
- n_runs: number of samples collected
"""
function benchmark_function(f::Function, name::String; samples::Int=100, evals::Int=1, seconds::Real=5)
    # Create benchmarkable and run it
    b = @benchmarkable ($f)()
    trial = run(b, samples=samples, evals=evals, seconds=seconds)
    
    # Extract statistics (times are in nanoseconds, convert to seconds)
    mean_time = mean(trial.times) / 1e9
    min_time = minimum(trial.times) / 1e9
    max_time = maximum(trial.times) / 1e9
    std_time = std(trial.times) / 1e9
    n_runs = length(trial.times)
    
    return (
        name = name,
        mean_time = mean_time,
        min_time = min_time,
        max_time = max_time,
        std_time = std_time,
        n_runs = n_runs
    )
end

"""
    write_benchmark_results(results::Vector, filename::String)

Write benchmark results to a CSV file with metadata (date, commit ID).

If the file exists, append to it. Otherwise, create it with headers.
"""
function write_benchmark_results(results::Vector, filename::String)
    # Get metadata
    timestamp = string(now())
    commit_id = get_git_commit_id()
    
    # Check if file exists to determine if we need headers
    file_exists = isfile(filename)
    
    # Open file in append mode
    open(filename, "a") do io
        # Write headers if file is new
        if !file_exists
            println(io, "timestamp,commit_id,benchmark_name,mean_time_s,min_time_s,max_time_s,std_time_s,n_runs")
        end
        
        # Write results
        for result in results
            @printf(io, "%s,%s,%s,%.9f,%.9f,%.9f,%.9f,%d\n",
                timestamp,
                commit_id,
                result.name,
                result.mean_time,
                result.min_time,
                result.max_time,
                result.std_time,
                result.n_runs
            )
        end
    end
    
    println("Benchmark results written to: $filename")
end

"""
    print_benchmark_results(results::Vector)

Pretty print benchmark results to stdout.
"""
function print_benchmark_results(results::Vector)
    println("\n" * "="^80)
    println("BENCHMARK RESULTS")
    println("="^80)
    println(@sprintf("%-50s %12s %12s", "Benchmark", "Mean (s)", "Std (s)"))
    println("-"^80)
    
    for result in results
        println(@sprintf("%-50s %12.6f %12.6f", 
            result.name, 
            result.mean_time,
            result.std_time
        ))
    end
    
    println("="^80)
    println()
end
