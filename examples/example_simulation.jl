#!/usr/bin/env julia

"""
Example demonstrating simulation of Wilson-Cowan models.

This shows how to use the simulation functions to:
1. Solve the model over time using DifferentialEquations.jl
2. Save complete simulation results to CSV
3. Save summary statistics to CSV
"""

using FailureOfInhibition2025
using Statistics

println("="^70)
println("Wilson-Cowan Model Simulation Examples")
println("="^70)

#=============================================================================
Example 1: Point Model Simulation (Non-Spatial)
=============================================================================#

println("\n### Example 1: Point Model (Non-Spatial) ###\n")

# Create a point model with 2 populations (E and I)
lattice = PointLattice()

# Define connectivity matrix for E-I interactions
conn_ee = ScalarConnectivity(0.5)    # E → E (reduced for stability)
conn_ei = ScalarConnectivity(-0.3)   # I → E  
conn_ie = ScalarConnectivity(0.4)    # E → I
conn_ii = ScalarConnectivity(-0.2)   # I → I

connectivity = ConnectivityMatrix{2}([
    conn_ee conn_ei;
    conn_ie conn_ii
])

# Create parameters
params_point = WilsonCowanParameters{2}(
    α = (1.0, 1.5),          # Decay rates
    β = (1.0, 1.0),          # Saturation coefficients
    τ = (10.0, 8.0),         # Time constants (slower dynamics for stability)
    connectivity = connectivity,
    nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E", "I")
)

# Initial condition: (1, 2) array for point model with connectivity
A₀_point = reshape([0.1, 0.1], 1, 2)

# Time span (0 to 100 time units)
tspan = (0.0, 100.0)

println("Solving point model...")
println("  Initial condition: E=$(A₀_point[1,1]), I=$(A₀_point[1,2])")
println("  Time span: $(tspan)")
println("  Populations: E (excitatory), I (inhibitory)")

# Solve the model
sol_point = solve_model(A₀_point, tspan, params_point, saveat=0.1)

println("  ✓ Simulation complete")
println("  Number of time points: $(length(sol_point.t))")
println("  Final state: E=$(round(sol_point.u[end][1,1], digits=4)), I=$(round(sol_point.u[end][1,2], digits=4))")

# Save results to CSV files
output_dir = joinpath(dirname(@__FILE__), "..", "examples", "output")
mkpath(output_dir)

point_results_file = joinpath(output_dir, "point_model_results.csv")
point_summary_file = joinpath(output_dir, "point_model_summary.csv")

println("\nSaving results...")
save_simulation_results(sol_point, point_results_file, params=params_point)
println("  ✓ Full results saved to: $point_results_file")

save_simulation_summary(sol_point, point_summary_file, params=params_point)
println("  ✓ Summary saved to: $point_summary_file")

#=============================================================================
Example 2: Spatial Model Simulation (1D) - Simple Case
=============================================================================#

println("\n\n### Example 2: Spatial Model (1D) - Simple Case ###\n")

# Create a 1D spatial lattice
lattice_spatial = CompactLattice(extent=(10.0,), n_points=(21,))

# Define Gaussian connectivity for spatial coupling (single population for simplicity)
conn_spatial = GaussianConnectivityParameter(0.3, (2.0,))

connectivity_spatial = ConnectivityMatrix{1}(reshape([conn_spatial], 1, 1))

# Create parameters for spatial model (single population)
params_spatial = WilsonCowanParameters{1}(
    α = (1.0,),
    β = (1.0,),
    τ = (10.0,),
    connectivity = connectivity_spatial,
    nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
    stimulus = nothing,
    lattice = lattice_spatial,
    pop_names = ("E",)
)

# Initial condition: small random activity across space
A₀_spatial = 0.1 .+ 0.05 .* rand(21, 1)

println("Solving spatial model...")
println("  Spatial lattice: 21 points over extent 10.0")
println("  Time span: $(tspan)")
println("  Single population with lateral Gaussian connectivity")

# Solve the spatial model
sol_spatial = solve_model(A₀_spatial, tspan, params_spatial, saveat=0.5)

println("  ✓ Simulation complete")
println("  Number of time points: $(length(sol_spatial.t))")

# Calculate spatial statistics at final time
final_activity = sol_spatial.u[end][:, 1]
println("  Final activity: mean=$(round(mean(final_activity), digits=4)), std=$(round(std(final_activity), digits=4))")

# Save results
spatial_results_file = joinpath(output_dir, "spatial_model_results.csv")
spatial_summary_file = joinpath(output_dir, "spatial_model_summary.csv")

println("\nSaving results...")
save_simulation_results(sol_spatial, spatial_results_file, params=params_spatial)
println("  ✓ Full results saved to: $spatial_results_file")

save_simulation_summary(sol_spatial, spatial_summary_file, params=params_spatial)
println("  ✓ Summary saved to: $spatial_summary_file")

#=============================================================================
Example 3: WCM 1973 Mode Simulation
=============================================================================#

println("\n\n### Example 3: WCM 1973 Active Transient Mode ###\n")

# Create parameters for active transient mode
params_active = create_point_model_wcm1973(:active_transient)

# Initial condition
A₀_active = reshape([0.1, 0.1], 1, 2)

# Shorter time span for demonstration
tspan_active = (0.0, 50.0)

println("Solving WCM 1973 Active Transient mode...")
println("  Mode: Active Transient (sensory neo-cortex)")
println("  Time span: $(tspan_active)")
println("  This mode exhibits self-generated transient responses")

# Solve
sol_active = solve_model(A₀_active, tspan_active, params_active, saveat=0.1)

println("  ✓ Simulation complete")
println("  Number of time points: $(length(sol_active.t))")
println("  Final state: E=$(round(sol_active.u[end][1,1], digits=4)), I=$(round(sol_active.u[end][1,2], digits=4))")

# Save results
wcm1973_results_file = joinpath(output_dir, "wcm1973_active_results.csv")
wcm1973_summary_file = joinpath(output_dir, "wcm1973_active_summary.csv")

println("\nSaving results...")
save_simulation_results(sol_active, wcm1973_results_file, params=params_active)
println("  ✓ Full results saved to: $wcm1973_results_file")

save_simulation_summary(sol_active, wcm1973_summary_file, params=params_active)
println("  ✓ Summary saved to: $wcm1973_summary_file")

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary")
println("="^70)
println()
println("This example demonstrated:")
println("  ✓ Solving point models (non-spatial ODEs)")
println("  ✓ Solving spatial models (1D PDEs with lateral connectivity)")
println("  ✓ Using WCM 1973 validated parameters")
println("  ✓ Saving complete simulation results to CSV")
println("  ✓ Saving summary statistics to CSV")
println()
println("Output files saved to: $output_dir")
println("  - point_model_results.csv & point_model_summary.csv")
println("  - spatial_model_results.csv & spatial_model_summary.csv")
println("  - wcm1973_active_results.csv & wcm1973_active_summary.csv")
println()
println("You can now:")
println("  - Load CSV files in your favorite analysis tool")
println("  - Plot time series data")
println("  - Analyze spatial patterns")
println("  - Compare different parameter regimes")
println()
println("="^70)
