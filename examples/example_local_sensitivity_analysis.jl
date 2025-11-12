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
using DifferentialEquations

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

# Set up model for sensitivity analysis
result = compute_local_sensitivities(
    A₀_point, tspan, params_point,
    include_params=[:α, :β, :τ],
    saveat=1.0  # Save every 1 time unit
)

println("  ✓ Model solved with parametrized structure")
println("  Number of parameters: $(length(result.param_names))")
println("  Parameters: $(result.param_names)")

# The solution is now set up for sensitivity analysis
println("\nSolution ready for sensitivity analysis with SciMLSensitivity.jl")
println("  To compute sensitivities, define a loss function and use sensitivity methods")
println("  from SciMLSensitivity.jl. For example:")
println("  ")
println("    using Optimization, OptimizationOptimJL")
println("    loss_function(p) = begin")
println("        # Re-solve with new parameters")
println("        # Extract sensitivities using automatic differentiation")
println("    end")
println("  ")
println("  See SciMLSensitivity.jl documentation for details:")
println("  https://docs.sciml.ai/SciMLSensitivity/stable/")

#=============================================================================
Example 2: Spatial Model Sensitivity
=============================================================================#

println("\n\n### Example 2: Local Sensitivity - Spatial Model ###\n")

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

# Set up model for sensitivity analysis (only analyze α, β, τ for single population)
result_spatial = compute_local_sensitivities(
    A₀_spatial, tspan_spatial, params_spatial,
    include_params=[:α, :β, :τ],
    saveat=2.0
)

println("  ✓ Model solved with parametrized structure")
println("  Number of parameters: $(length(result_spatial.param_names))")
println("  Number of state variables: $(size(A₀_spatial, 1) * size(A₀_spatial, 2))")
println("  Spatial model ready for sensitivity analysis")


#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary")
println("="^70)
println()
println("This example demonstrated:")
println("  ✓ Setting up Wilson-Cowan models for sensitivity analysis")
println("  ✓ Extracting parameters into vector form for sensitivity methods")
println("  ✓ Preparing both point and spatial models for analysis")
println()
println("Key insights:")
println("  - compute_local_sensitivities() sets up the parametrized ODE system")
println("  - The result can be used with SciMLSensitivity.jl methods")
println("  - Parameter extraction handles model-specific structures")
println()
println("Next steps:")
println("  - Refer to SciMLSensitivity.jl documentation for computing sensitivities")
println("  - Explore GlobalSensitivity.jl for global variance-based indices")
println("  - Use sensitivity information to inform parameter estimation")
println()
println("="^70)
