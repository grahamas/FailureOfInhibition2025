"""
Main benchmark runner for all benchmarks.
This script runs both component and simulation benchmarks.
"""

include("benchmark_components.jl")
include("benchmark_simulations.jl")

"""
Run all benchmarks (components and simulations).
"""
function run_all_benchmarks()
    println("\n" * "="^80)
    println("RUNNING ALL BENCHMARKS")
    println("Commit: $(get_git_commit_id())")
    println("="^80)
    
    # Run component benchmarks
    component_results = run_component_benchmarks()
    
    # Run simulation benchmarks
    simulation_results = run_simulation_benchmarks()
    
    # Summary
    total_benchmarks = length(component_results) + length(simulation_results)
    println("\n" * "="^80)
    println("BENCHMARK SUMMARY")
    println("="^80)
    println("Total benchmarks run: $total_benchmarks")
    println("  - Component benchmarks: $(length(component_results))")
    println("  - Simulation benchmarks: $(length(simulation_results))")
    println("\nResults saved to:")
    println("  - benchmark_results/component_benchmarks.csv")
    println("  - benchmark_results/simulation_benchmarks.csv")
    println("="^80)
    
    return (component_results, simulation_results)
end

# Run if executed directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_benchmarks()
end
