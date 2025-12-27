#!/usr/bin/env julia

"""
Example demonstrating fixed point analysis and parameterization search.

This example shows how to:
1. Find fixed points (equilibria) of Wilson-Cowan models
2. Compute stability of fixed points
3. Count stable fixed points
4. Search for parameters that produce a target number of stable fixed points

The tool is useful for bifurcation analysis and understanding how
parameter changes affect system dynamics.
"""

using FailureOfInhibition2025

println("\n" * "="^70)
println("Fixed Point Analysis and Parameterization Search")
println("="^70)

#=============================================================================
Example 1: Finding Fixed Points
=============================================================================#

println("\n### Example 1: Finding Fixed Points ###\n")

# Create a point model (steady state mode)
params_ss = create_point_model_wcm1973(:steady_state)

# Find fixed points
println("Finding fixed points for steady state mode...")
fixed_points, converged = find_fixed_points(params_ss, n_trials=15)

println("Found $(length(fixed_points)) fixed point(s):")
for (i, fp) in enumerate(fixed_points)
    println("  FP $i: E=$(round(fp[1], digits=4)), I=$(round(fp[2], digits=4))")
end

#=============================================================================
Example 2: Computing Stability of Fixed Points
=============================================================================#

println("\n### Example 2: Computing Stability ###\n")

# Analyze stability of each fixed point
for (i, fp) in enumerate(fixed_points)
    eigenvalues, is_stable = compute_stability(fp, params_ss)
    
    status = is_stable ? "Stable" : "Unstable"
    println("FP $i: $status")
    println("  Eigenvalues:")
    for λ in eigenvalues
        real_part = round(real(λ), digits=4)
        imag_part = round(imag(λ), digits=4)
        if abs(imag_part) < 1e-6
            println("    λ = $real_part")
        else
            sign_str = imag_part >= 0 ? "+" : "-"
            println("    λ = $real_part $sign_str $(abs(imag_part))i")
        end
    end
end

#=============================================================================
Example 3: Counting Stable Fixed Points Across Different Modes
=============================================================================#

println("\n### Example 3: Counting Stable Fixed Points ###\n")

modes = [:active_transient, :oscillatory, :steady_state]

for mode in modes
    params = create_point_model_wcm1973(mode)
    n_stable, fps, stabilities = count_stable_fixed_points(params, n_trials=15)
    
    println("Mode: $mode")
    println("  Total fixed points: $(length(fps))")
    println("  Stable fixed points: $n_stable")
    
    # Show stability of each
    for (i, is_stable) in enumerate(stabilities)
        status = is_stable ? "stable" : "unstable"
        fp = fps[i]
        println("    FP $i: $status (E=$(round(fp[1], digits=3)), I=$(round(fp[2], digits=3)))")
    end
    println()
end

#=============================================================================
Example 4: Searching for Parameters with Target Number of Stable Fixed Points
=============================================================================#

println("\n### Example 4: Parameter Search for Target Stability ###\n")

# Start with active transient mode
base_params = create_point_model_wcm1973(:active_transient)

# Count current stable fixed points
n_current, _, _ = count_stable_fixed_points(base_params, n_trials=15)
println("Current parameters have $n_current stable fixed point(s)")

# Define parameter ranges to search
param_ranges = (
    connectivity_ee = (1.0, 3.0),     # Excitatory-to-excitatory
    connectivity_ei = (0.8, 2.5),     # Inhibitory-to-excitatory
    sigmoid_theta_e = (5.0, 15.0)     # Excitatory threshold
)

# Target: Find parameters with exactly 1 stable fixed point
target_n_stable = 1
println("\nSearching for parameters with $target_n_stable stable fixed point(s)...")
println("Parameter ranges:")
println("  E→E connectivity: $(param_ranges.connectivity_ee)")
println("  I→E connectivity: $(param_ranges.connectivity_ei)")
println("  E threshold: $(param_ranges.sigmoid_theta_e)")

# Run optimization
result, best_params = optimize_for_stable_fixed_points(
    base_params,
    param_ranges,
    target_n_stable,
    n_trials_per_eval=12,
    maxiter=20,  # Increase for better results
    population_size=10
)

# Verify the result
n_optimized, fps_opt, stab_opt = count_stable_fixed_points(best_params, n_trials=15)
println("\nOptimization complete!")
println("  Target: $target_n_stable stable fixed point(s)")
println("  Found: $n_optimized stable fixed point(s)")

# Display optimized parameters
println("\nOptimized parameters:")
conn_ee = best_params.connectivity.matrix[1,1].weight
conn_ei = abs(best_params.connectivity.matrix[1,2].weight)
theta_e = best_params.nonlinearity[1].θ
println("  E→E connectivity: $(round(conn_ee, digits=3))")
println("  I→E connectivity: $(round(conn_ei, digits=3))")
println("  E threshold: $(round(theta_e, digits=3))")

# Show the fixed points
println("\nFixed points with optimized parameters:")
for (i, (fp, is_stable)) in enumerate(zip(fps_opt, stab_opt))
    status = is_stable ? "stable" : "unstable"
    println("  FP $i: $status (E=$(round(fp[1], digits=4)), I=$(round(fp[2], digits=4)))")
end

#=============================================================================
Example 5: Comparing Different Target Numbers
=============================================================================#

println("\n### Example 5: Searching for Different Target Numbers ###\n")

# Try to find parameters with 2 stable fixed points
target_2 = 2
println("Searching for parameters with $target_2 stable fixed points...")

param_ranges_2 = (
    connectivity_ee = (1.0, 3.5),
    connectivity_ei = (0.5, 2.0)
)

result_2, best_params_2 = optimize_for_stable_fixed_points(
    base_params,
    param_ranges_2,
    target_2,
    n_trials_per_eval=10,
    maxiter=15,
    population_size=8
)

n_2, _, _ = count_stable_fixed_points(best_params_2, n_trials=12)
println("  Target: $target_2 stable fixed points")
println("  Found: $n_2 stable fixed points")

#=============================================================================
Summary and Applications
=============================================================================#

println("\n" * "="^70)
println("Summary")
println("="^70)
println()
println("These tools enable systematic exploration of Wilson-Cowan dynamics:")
println()
println("1. find_fixed_points:")
println("   - Locates all equilibrium points of the system")
println("   - Uses optimization to find zeros of the dynamics")
println("   - Multiple trials help find all fixed points")
println()
println("2. compute_stability:")
println("   - Analyzes eigenvalues of Jacobian at fixed points")
println("   - Determines if fixed points are stable or unstable")
println("   - Complex eigenvalues indicate oscillatory dynamics")
println()
println("3. count_stable_fixed_points:")
println("   - Convenience function combining finding and stability")
println("   - Quick way to characterize system dynamics")
println()
println("4. optimize_for_stable_fixed_points:")
println("   - Searches parameter space for target dynamics")
println("   - Uses global optimization (particle swarm)")
println("   - Useful for bifurcation analysis and system design")
println()
println("Applications:")
println("  • Bifurcation analysis: understand parameter transitions")
println("  • System design: find parameters for desired behavior")
println("  • Multistability: identify parameters with multiple attractors")
println("  • Pattern formation: study spatial steady states")
println()
println("="^70)
