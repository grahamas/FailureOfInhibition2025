#!/usr/bin/env julia

"""
Example demonstrating parameter optimization for traveling waves.

This shows how to use the optimization functions to find Wilson-Cowan model
parameters that produce desired traveling wave behaviors in 1D spatial models.
"""

using FailureOfInhibition2025
using Optim  # For accessing result properties
using Printf

println("="^70)
println("Parameter Optimization for Traveling Waves Example")
println("="^70)

#=============================================================================
Example 1: Optimize for maximum traveling distance
=============================================================================#

println("\n### Example 1: Maximize Traveling Distance ###\n")

# Set up base parameters for 1D model
lattice = CompactLattice(extent=(20.0,), n_points=(101,))
conn = GaussianConnectivityParameter(0.8, (2.0,))

base_params = WilsonCowanParameters{1}(
    α = (1.0,),
    β = (1.0,),
    τ = (8.0,),
    connectivity = ConnectivityMatrix{1}(reshape([conn], 1, 1)),
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E",)
)

println("Base parameters:")
println("  - Connectivity spread: $(base_params.connectivity[1,1].spread[1])")
println("  - Sigmoid steepness (a): $(base_params.nonlinearity.a)")
println("  - Sigmoid threshold (θ): $(base_params.nonlinearity.θ)")

# Define what to optimize
param_ranges = (
    connectivity_width = (1.5, 4.0),  # Range for connectivity spread
    sigmoid_a = (1.5, 3.5)             # Range for sigmoid steepness
)

# Objective: maximize distance traveled
objective = TravelingWaveObjective(
    target_distance = nothing,  # No specific target, maximize
    minimize_decay = true,       # Prefer waves that don't decay too quickly
    require_traveling = true,    # Must have traveling peak
    threshold = 0.15
)

# Initial condition - localized bump
A₀ = zeros(101, 1)
A₀[15:20, 1] .= 0.6

# Run optimization
println("\nOptimizing parameters...")
println("  Parameter ranges:")
println("    - connectivity_width: $(param_ranges.connectivity_width)")
println("    - sigmoid_a: $(param_ranges.sigmoid_a)")

result, optimized_params = optimize_for_traveling_wave(
    base_params,
    param_ranges,
    objective,
    A₀,
    (0.0, 40.0),  # Time span
    saveat = 0.2,
    maxiter = 50
)

println("\nOptimization completed!")
println("  Iterations: $(result.iterations)")
println("  Converged: $(Optim.converged(result))")
println("\nOptimized parameters:")
println("  - Connectivity spread: $(round(optimized_params.connectivity[1,1].spread[1], digits=3))")
println("  - Sigmoid steepness (a): $(round(optimized_params.nonlinearity.a, digits=3))")

# Compare results
println("\n### Comparison: Base vs Optimized ###\n")

# Simulate with base parameters
sol_base = solve_model(A₀, (0.0, 40.0), base_params, saveat=0.2)
has_peak_base, _, _ = detect_traveling_peak(sol_base, 1, threshold=0.15)
dist_base, _ = compute_distance_traveled(sol_base, 1, lattice, threshold=0.15)
decay_base, _ = compute_decay_rate(sol_base, 1)

println("Base parameters:")
println("  - Traveling peak detected: $has_peak_base")
println("  - Distance traveled: $(round(dist_base, digits=2)) units")
if decay_base !== nothing
    println("  - Decay rate: $(round(decay_base, digits=4))")
end

# Simulate with optimized parameters
sol_opt = solve_model(A₀, (0.0, 40.0), optimized_params, saveat=0.2)
has_peak_opt, _, _ = detect_traveling_peak(sol_opt, 1, threshold=0.15)
dist_opt, _ = compute_distance_traveled(sol_opt, 1, lattice, threshold=0.15)
decay_opt, _ = compute_decay_rate(sol_opt, 1)

println("\nOptimized parameters:")
println("  - Traveling peak detected: $has_peak_opt")
println("  - Distance traveled: $(round(dist_opt, digits=2)) units")
if decay_opt !== nothing
    println("  - Decay rate: $(round(decay_opt, digits=4))")
end

if dist_opt > dist_base
    improvement = ((dist_opt - dist_base) / dist_base) * 100
    println("\n✓ Improvement: $(round(improvement, digits=1))% increase in distance!")
end

#=============================================================================
Example 2: Optimize for target distance
=============================================================================#

println("\n\n### Example 2: Target Specific Distance ###\n")

# Use same base parameters but different objective
target_dist = 8.0  # Target 8 units of travel

objective2 = TravelingWaveObjective(
    target_distance = target_dist,
    minimize_decay = true,
    require_traveling = true,
    threshold = 0.15
)

# Optimize only connectivity spread for simplicity
param_ranges2 = (connectivity_width = (1.5, 4.5),)

println("Targeting distance: $(target_dist) units")
println("Optimizing connectivity spread only...")

result2, optimized_params2 = optimize_for_traveling_wave(
    base_params,
    param_ranges2,
    objective2,
    A₀,
    (0.0, 40.0),
    saveat = 0.2,
    maxiter = 30
)

# Simulate and measure
sol_target = solve_model(A₀, (0.0, 40.0), optimized_params2, saveat=0.2)
dist_target, _ = compute_distance_traveled(sol_target, 1, lattice, threshold=0.15)

println("\nResults:")
println("  - Target distance: $(target_dist) units")
println("  - Achieved distance: $(round(dist_target, digits=2)) units")
println("  - Optimized connectivity spread: $(round(optimized_params2.connectivity[1,1].spread[1], digits=3))")
println("  - Error: $(round(abs(dist_target - target_dist), digits=2)) units")

#=============================================================================
Example 3: Optimize multiple objectives
=============================================================================#

println("\n\n### Example 3: Multiple Objectives ###\n")

# Target specific amplitude and width in addition to traveling
objective3 = TravelingWaveObjective(
    target_distance = 10.0,
    target_amplitude = 0.7,
    target_width = 3.0,
    minimize_decay = true,
    require_traveling = true,
    threshold = 0.15
)

println("Optimizing for:")
println("  - Target distance: 10.0 units")
println("  - Target amplitude: 0.7")
println("  - Target width: 3.0 units")

param_ranges3 = (
    connectivity_width = (1.5, 4.0),
    sigmoid_a = (1.5, 3.5),
    sigmoid_θ = (0.15, 0.35)
)

println("\nOptimizing 3 parameters...")

result3, optimized_params3 = optimize_for_traveling_wave(
    base_params,
    param_ranges3,
    objective3,
    A₀,
    (0.0, 40.0),
    saveat = 0.2,
    maxiter = 50
)

# Measure all metrics
sol_multi = solve_model(A₀, (0.0, 40.0), optimized_params3, saveat=0.2)
dist_multi, _ = compute_distance_traveled(sol_multi, 1, lattice, threshold=0.15)
amp_multi = compute_amplitude(sol_multi, 1, method=:max)
width_multi, _, _ = compute_half_max_width(sol_multi, 1, nothing, lattice)

println("\nOptimized parameters:")
println("  - Connectivity spread: $(round(optimized_params3.connectivity[1,1].spread[1], digits=3))")
println("  - Sigmoid a: $(round(optimized_params3.nonlinearity.a, digits=3))")
println("  - Sigmoid θ: $(round(optimized_params3.nonlinearity.θ, digits=3))")

println("\nAchieved metrics:")
println("  - Distance: $(round(dist_multi, digits=2)) (target: 10.0)")
println("  - Amplitude: $(round(amp_multi, digits=3)) (target: 0.7)")
println("  - Width: $(round(width_multi, digits=2)) (target: 3.0)")

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary")
println("="^70)
println()
println("This example demonstrated:")
println("  ✓ Optimizing to maximize traveling distance")
println("  ✓ Optimizing to achieve a target distance")
println("  ✓ Multi-objective optimization (distance + amplitude + width)")
println()
println("Key features:")
println("  • Flexible objective specification via TravelingWaveObjective")
println("  • Ability to optimize any subset of parameters")
println("  • Support for target values or maximization/minimization")
println("  • Integration with traveling wave analysis metrics")
println()
println("Applications:")
println("  • Finding parameter regimes that support traveling waves")
println("  • Tuning models to match experimental observations")
println("  • Exploring parameter space for specific dynamics")
println("  • Investigating failure of inhibition mechanisms")
println()
println("="^70)
