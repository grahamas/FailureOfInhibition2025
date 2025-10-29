#!/usr/bin/env julia

"""
Example demonstrating parameter sensitivity analysis for Wilson-Cowan models.

This shows how to use SciMLSensitivity.jl to:
1. Compute local sensitivity (derivatives ∂u/∂p)
2. Analyze which parameters most affect model behavior
3. Save and summarize sensitivity results
"""

using FailureOfInhibition2025
using Statistics
using Printf
using SciMLSensitivity

println("="^70)
println("Wilson-Cowan Model Parameter Sensitivity Analysis")
println("="^70)

#=============================================================================
Example 1: Local Sensitivity Analysis for Point Model
=============================================================================#

println("\n### Example 1: Local Sensitivity - Point Model ###\n")

# Create a point model with 2 populations (E and I)
lattice = PointLattice()

# Define connectivity matrix for E-I interactions
conn_ee = ScalarConnectivity(0.5)
conn_ei = ScalarConnectivity(-0.3)
conn_ie = ScalarConnectivity(0.4)
conn_ii = ScalarConnectivity(-0.2)

connectivity = ConnectivityMatrix{2}([
    conn_ee conn_ei;
    conn_ie conn_ii
])

# Create parameters
params_point = WilsonCowanParameters{2}(
    α = (1.0, 1.5),          # Decay rates
    β = (1.0, 1.0),          # Saturation coefficients
    τ = (10.0, 8.0),         # Time constants
    connectivity = connectivity,
    nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E", "I")
)

# Initial condition
A₀_point = reshape([0.1, 0.1], 1, 2)

# Time span
tspan = (0.0, 50.0)

println("Computing local sensitivities for point model...")
println("  Parameters analyzed: α (decay), β (saturation), τ (time constant)")
println("  Populations: E (excitatory), I (inhibitory)")
println("  Time span: $(tspan)")

# Compute sensitivities using forward mode (good for few parameters)
result = compute_local_sensitivities(
    A₀_point, tspan, params_point,
    include_params=[:α, :β, :τ],
    method=ForwardDiffSensitivity(),
    saveat=1.0  # Save every 1 time unit
)

println("  ✓ Sensitivity analysis complete")
println("  Number of parameters: $(length(result.param_names))")
println("  Parameters: $(result.param_names)")

# Summarize sensitivities
println("\nComputing sensitivity summary statistics...")
summary = summarize_sensitivities(result, params=params_point)

println("\n--- Sensitivity Summary (sorted by mean absolute sensitivity) ---")
sorted_summary = sort(summary, :mean_abs_sensitivity, rev=true)
for row in eachrow(sorted_summary[1:min(10, nrow(sorted_summary)), :])
    @printf("  %s → %s: mean_abs=%.4f, max_abs=%.4f, final=%.4f\n",
            row.param_name, row.state_name,
            row.mean_abs_sensitivity, row.max_abs_sensitivity, row.final_sensitivity)
end

# Save results to CSV
output_dir = joinpath(dirname(@__FILE__), "output")
mkpath(output_dir)

sens_file = joinpath(output_dir, "point_model_sensitivities.csv")
summary_file = joinpath(output_dir, "point_model_sensitivity_summary.csv")

println("\nSaving results...")
save_local_sensitivities(result, sens_file, params=params_point)
println("  ✓ Full sensitivities saved to: $sens_file")

using CSV, DataFrames
CSV.write(summary_file, summary)
println("  ✓ Summary saved to: $summary_file")

#=============================================================================
Example 2: Analyzing Sensitivity Time Evolution
=============================================================================#

println("\n\n### Example 2: Time Evolution of Sensitivities ###\n")

# Look at how sensitivities change over time for specific parameters
println("Sensitivity evolution for E population w.r.t. key parameters:")
println()

# Get indices
e_idx = 1  # Excitatory population (first population)
param_indices = Dict(
    :α_1 => findfirst(==(Symbol("α_1")), result.param_names),
    :τ_1 => findfirst(==(Symbol("τ_1")), result.param_names),
    :β_1 => findfirst(==(Symbol("β_1")), result.param_names)
)

println("Time points: [0, 10, 20, 30, 40, 50]")
println()

for (param_name, p_idx) in param_indices
    if p_idx !== nothing
        println("Parameter: $param_name")
        for t in [1, 11, 21, 31, 41, 51]  # indices for times 0, 10, 20, ...
            if t <= length(result.times)
                sens = result.sensitivities[t, e_idx, p_idx]
                @printf("  t=%-3d: ∂E/∂%s = %+.6f\n", result.times[t], param_name, sens)
            end
        end
        println()
    end
end

#=============================================================================
Example 3: Spatial Model Sensitivity (if time permits)
=============================================================================#

println("\n### Example 3: Local Sensitivity - Spatial Model ###\n")

# Create a small spatial model
lattice_spatial = CompactLattice(extent=(10.0,), n_points=(11,))

# Single population with lateral connectivity
conn_spatial = GaussianConnectivityParameter(0.3, (2.0,))
connectivity_spatial = ConnectivityMatrix{1}(reshape([conn_spatial], 1, 1))

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

# Initial condition with spatial variation
A₀_spatial = 0.1 .+ 0.05 .* rand(11, 1)

# Shorter time span for efficiency
tspan_spatial = (0.0, 20.0)

println("Computing local sensitivities for spatial model...")
println("  Spatial lattice: 11 points over extent 10.0")
println("  Time span: $(tspan_spatial)")
println("  Single population with Gaussian connectivity")

# Compute sensitivities (only analyze α, β, τ for single population)
result_spatial = compute_local_sensitivities(
    A₀_spatial, tspan_spatial, params_spatial,
    include_params=[:α, :β, :τ],
    method=ForwardDiffSensitivity(),
    saveat=2.0
)

println("  ✓ Sensitivity analysis complete")
println("  Number of parameters: $(length(result_spatial.param_names))")
println("  Number of state variables: $(size(A₀_spatial, 1) * size(A₀_spatial, 2))")

# Analyze spatial average sensitivities
summary_spatial = summarize_sensitivities(result_spatial, params=params_spatial)

println("\n--- Spatial Model Sensitivity Summary ---")
for row in eachrow(summary_spatial)
    @printf("  %s (state %d): mean_abs=%.4f, max_abs=%.4f\n",
            row.param_name, row.state_idx,
            row.mean_abs_sensitivity, row.max_abs_sensitivity)
end

# Save spatial results
spatial_sens_file = joinpath(output_dir, "spatial_model_sensitivities.csv")
spatial_summary_file = joinpath(output_dir, "spatial_model_sensitivity_summary.csv")

println("\nSaving spatial model results...")
save_local_sensitivities(result_spatial, spatial_sens_file, params=params_spatial)
println("  ✓ Sensitivities saved to: $spatial_sens_file")

CSV.write(spatial_summary_file, summary_spatial)
println("  ✓ Summary saved to: $spatial_summary_file")

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary")
println("="^70)
println()
println("This example demonstrated:")
println("  ✓ Local sensitivity analysis for point models")
println("  ✓ Computing ∂u/∂p (derivative of state w.r.t. parameters)")
println("  ✓ Time evolution of parameter sensitivities")
println("  ✓ Spatial model sensitivity analysis")
println("  ✓ Summarizing and ranking parameter importance")
println()
println("Key insights from sensitivity analysis:")
println("  - Local sensitivities show how small parameter changes affect dynamics")
println("  - Forward mode (ForwardDiffSensitivity) is efficient for few parameters")
println("  - Sensitivities evolve over time, revealing transient vs. steady-state effects")
println("  - Spatial models can have position-dependent sensitivities")
println()
println("Output files saved to: $output_dir")
println("  - point_model_sensitivities.csv")
println("  - point_model_sensitivity_summary.csv")
println("  - spatial_model_sensitivities.csv")
println("  - spatial_model_sensitivity_summary.csv")
println()
println("Next steps:")
println("  - Use adjoint methods for models with many parameters")
println("  - Explore GlobalSensitivity.jl for global variance-based indices")
println("  - Use sensitivity information to inform parameter estimation")
println()
println("="^70)
