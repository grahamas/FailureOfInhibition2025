#!/usr/bin/env julia

"""
Example demonstrating global sensitivity analysis of Wilson-Cowan model parameters.

This script shows how to:
1. Set up a Wilson-Cowan model
2. Define parameter ranges for sensitivity analysis
3. Run Sobol analysis to quantify parameter importance
4. Run Morris screening for efficient parameter ranking
5. Interpret and visualize results
"""

using FailureOfInhibition2025
using Statistics

println("="^70)
println("Wilson-Cowan Model Global Sensitivity Analysis")
println("="^70)

#=============================================================================
Example 1: Point Model Sensitivity Analysis
=============================================================================#

println("\n### Example 1: Point Model (Non-Spatial) Sensitivity Analysis ###\n")

# Create a point model with 2 populations (E and I)
lattice = PointLattice()

# Define base connectivity matrix for E-I interactions
conn_ee = ScalarConnectivity(0.5)
conn_ei = ScalarConnectivity(-0.3)
conn_ie = ScalarConnectivity(0.4)
conn_ii = ScalarConnectivity(-0.2)

connectivity = ConnectivityMatrix{2}([
    conn_ee conn_ei;
    conn_ie conn_ii
])

# Create base parameters
base_params = WilsonCowanParameters{2}(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (10.0, 8.0),
    connectivity = connectivity,
    nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E", "I")
)

println("Base Parameters:")
println("  α (decay): E=$(base_params.α[1]), I=$(base_params.α[2])")
println("  τ (time const): E=$(base_params.τ[1]), I=$(base_params.τ[2])")
println("  Connectivity: EE=0.5, EI=-0.3, IE=0.4, II=-0.2")
println()

# Define parameter ranges for sensitivity analysis
# Format: (parameter_name, lower_bound, upper_bound)
# Parameter names follow convention: 
#   - "α_<pop>", "τ_<pop>" for population parameters
#   - "conn_<src><dst>" for connectivity weights
param_ranges = [
    ("α_E", 0.5, 2.0),       # Decay rate for E population
    ("α_I", 0.5, 2.0),       # Decay rate for I population
    ("τ_E", 5.0, 15.0),      # Time constant for E population
    ("τ_I", 4.0, 12.0),      # Time constant for I population
    ("conn_EE", 0.2, 0.8),   # E → E connectivity
    ("conn_EI", -0.6, -0.1), # I → E connectivity (inhibitory)
    ("conn_IE", 0.2, 0.6),   # E → I connectivity
    ("conn_II", -0.4, -0.1)  # I → I connectivity (inhibitory)
]

println("Parameter Ranges:")
for (name, lb, ub) in param_ranges
    println("  $name: [$lb, $ub]")
end
println()

#=============================================================================
Morris Screening (Fast, qualitative ranking)
=============================================================================#

println("\n--- Morris Screening (Elementary Effects) ---\n")
println("Morris screening is efficient for identifying most important parameters.")
println("Uses fewer model evaluations than Sobol analysis.")
println()

# Run Morris screening with 50 trajectories
# For a quick test, use fewer trajectories; for production, use 100-500
morris_result = morris_sensitivity_analysis(
    base_params,
    param_ranges,
    50,  # Number of trajectories
    tspan=(0.0, 100.0),
    output_metric=:final_mean
)

println("\nMorris Results:")
println("Parameter rankings by importance (mean of absolute effects):")
println()

# Sort parameters by importance (μ*)
param_importance = [(morris_result[:param_names][i], morris_result[:means_star][i], morris_result[:variances][i]) 
                    for i in 1:length(morris_result[:param_names])]
sort!(param_importance, by=x->x[2], rev=true)

println(rpad("Parameter", 12), " ", rpad("μ* (importance)", 18), " σ² (interactions)")
println("-"^50)
for (name, mu_star, variance) in param_importance
    println(rpad(name, 12), " ", rpad(round(mu_star, digits=6), 18), " ", round(variance, digits=6))
end

println()
println("Interpretation:")
println("  - μ* (mean of absolute effects): Higher values indicate more important parameters")
println("  - σ² (variance): High values suggest parameter has interactions or nonlinear effects")
println()

#=============================================================================
Sobol Analysis (Quantitative variance decomposition)
=============================================================================#

println("\n--- Sobol Sensitivity Analysis ---\n")
println("Sobol analysis quantifies the fraction of output variance due to each parameter.")
println("This is more computationally expensive but provides quantitative indices.")
println()

# Run Sobol analysis with 500 samples
# Note: Sobol requires N*(2k+2) model evaluations, where k = number of parameters
# For 8 parameters and N=500: 500*(2*8+2) = 9000 evaluations
# For faster testing, use N=100-200; for production, use N=500-1000

println("Running Sobol analysis (this may take a few minutes)...")
sobol_result = sobol_sensitivity_analysis(
    base_params,
    param_ranges,
    200,  # Number of samples (reduced for faster execution)
    tspan=(0.0, 100.0),
    output_metric=:final_mean
)

println("\nSobol Results:")
println()
println(rpad("Parameter", 12), " ", rpad("S1 (first-order)", 18), " ST (total-order)")
println("-"^50)

for i in 1:length(sobol_result[:param_names])
    name = sobol_result[:param_names][i]
    s1 = sobol_result[:S1][i]
    st = sobol_result[:ST][i]
    println(rpad(name, 12), " ", rpad(round(s1, digits=6), 18), " ", round(st, digits=6))
end

println()
println("Interpretation:")
println("  - S1 (first-order): Fraction of variance due to parameter alone")
println("  - ST (total-order): Fraction of variance due to parameter and its interactions")
println("  - ST - S1: Fraction of variance due to interactions with other parameters")
println("  - Sum of S1 ≈ 1 if no interactions; < 1 if strong interactions exist")
println()

# Calculate interaction effects
println("\nInteraction Effects (ST - S1):")
println(rpad("Parameter", 12), " Interaction")
println("-"^25)
for i in 1:length(sobol_result[:param_names])
    name = sobol_result[:param_names][i]
    interaction = sobol_result[:ST][i] - sobol_result[:S1][i]
    println(rpad(name, 12), " ", round(interaction, digits=6))
end

println()

#=============================================================================
Different Output Metrics
=============================================================================#

println("\n### Example 2: Analyzing Different Output Metrics ###\n")

println("The same parameter ranges can be analyzed with different output metrics:")
println("  - :final_mean  - Mean activity at final time (default)")
println("  - :final_E     - Final E population activity")
println("  - :final_I     - Final I population activity")
println("  - :max_mean    - Maximum mean activity over time")
println("  - :variance    - Variance of activity over time")
println("  - :oscillation_amplitude - Amplitude of oscillations")
println()

# Analyze oscillation amplitude
println("Analyzing oscillation amplitude...")
morris_osc = morris_sensitivity_analysis(
    base_params,
    param_ranges,
    30,
    tspan=(0.0, 200.0),  # Longer time for oscillations to develop
    output_metric=:oscillation_amplitude
)

println("\nParameters most affecting oscillation amplitude:")
osc_importance = [(morris_osc[:param_names][i], morris_osc[:means_star][i]) 
                  for i in 1:length(morris_osc[:param_names])]
sort!(osc_importance, by=x->x[2], rev=true)

for (i, (name, importance)) in enumerate(osc_importance[1:min(4, length(osc_importance))])
    println("  $i. $name: $(round(importance, digits=4))")
end

println()

#=============================================================================
Summary
=============================================================================#

println("="^70)
println("Summary")
println("="^70)
println()
println("This example demonstrated:")
println("  ✓ Setting up global sensitivity analysis for Wilson-Cowan models")
println("  ✓ Using Morris screening for efficient parameter ranking")
println("  ✓ Using Sobol analysis for quantitative variance decomposition")
println("  ✓ Analyzing different output metrics")
println()
println("Key findings from this analysis:")
println("  - Morris screening provides quick parameter importance ranking")
println("  - Sobol indices quantify exact contribution of each parameter")
println("  - Different output metrics reveal different parameter sensitivities")
println()
println("Applications:")
println("  - Parameter importance ranking for model simplification")
println("  - Identifying which parameters to measure more accurately")
println("  - Understanding parameter interactions and nonlinear effects")
println("  - Guiding parameter calibration efforts")
println()
println("="^70)
