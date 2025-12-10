#!/usr/bin/env julia

"""
Generate bifurcation diagrams for Failure of Inhibition (FoI) models.

This script performs continuation analysis on FoI models to explore how
system dynamics change as parameters vary. It generates bifurcation diagrams
showing the emergence of oscillations, traveling waves, and other dynamical
behaviors characteristic of FoI.

Key features:
- Analyzes FoI models with characteristic inhibitory nonlinearity
- Varies E-E connectivity strength to show transitions
- Varies inhibitory nonlinearity parameters (failing threshold)
- Generates publication-quality bifurcation diagrams
- Saves results as PNG files

Usage:
    julia --project=. scripts/generate_foi_bifurcation_diagrams.jl
"""

using FailureOfInhibition2025
using BifurcationKit
using Plots

println("\n" * "="^70)
println("Bifurcation Analysis: Failure of Inhibition Model")
println("="^70)

#=============================================================================
Setup: Create Base FoI Parameters
=============================================================================#

println("\n### Setting up FoI parameters ###\n")

# Use point lattice for faster computation in continuation
lattice = PointLattice()

# Define connectivity for FoI model
# We'll use moderate connectivity strengths
conn_ee = ScalarConnectivity(1.2)     # E → E (will be varied)
conn_ei = ScalarConnectivity(-0.8)    # I → E (inhibitory)
conn_ie = ScalarConnectivity(1.0)     # E → I (excitatory)
conn_ii = ScalarConnectivity(-0.3)    # I → I (weak self-inhibition)

connectivity = ConnectivityMatrix{2}([
    conn_ee conn_ei;
    conn_ie conn_ii
])

# Create FoI parameters with characteristic nonlinearities
# E population: Standard rectified sigmoid
# I population: Difference of sigmoids (characteristic FoI feature)
base_params = FailureOfInhibitionParameters(
    α = (1.0, 1.2),           # Decay rates
    β = (1.0, 1.0),           # Saturation coefficients
    τ = (10.0, 8.0),          # Time constants
    connectivity = connectivity,
    nonlinearity_E = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5),
    nonlinearity_I = DifferenceOfSigmoidsNonlinearity(
        a_activating=5.0, θ_activating=0.3,
        a_failing=3.0, θ_failing=0.7    # This parameter creates FoI dynamics
    ),
    stimulus = nothing,
    lattice = lattice
)

println("✓ Created FoI parameters")
println("  E nonlinearity: RectifiedZeroedSigmoidNonlinearity")
println("  I nonlinearity: DifferenceOfSigmoidsNonlinearity (non-monotonic)")
println()

#=============================================================================
Bifurcation Analysis 1: Varying E-E Connectivity
=============================================================================#

println("\n### Bifurcation 1: Varying E-E Connectivity ###\n")

# Initial condition near rest state
u0_1 = reshape([0.05, 0.05], 1, 2)

# Create lens for E→E connectivity
lens_ee = create_connectivity_lens(1, 1)

# Create bifurcation problem
prob_ee = create_bifurcation_problem(base_params, lens_ee, u0=u0_1)

# Set up continuation parameters
# We'll explore E-E connectivity from 0.5 to 2.5
opts_ee = create_default_continuation_opts(
    p_min=0.5,
    p_max=2.5,
    max_steps=200,
    dsmax=0.05,
    ds=0.01
)

println("Running continuation analysis...")
println("  Parameter: E→E connectivity")
println("  Range: [$(opts_ee.p_min), $(opts_ee.p_max)]")
println()

# Run continuation
try
    br_ee = continuation(prob_ee, PALC(), opts_ee; verbosity=0)
    
    println("✓ Continuation completed successfully")
    println("  Steps: $(length(br_ee.branch))")
    println("  Bifurcations detected: $(length(br_ee.specialpoint))")
    
    # Plot bifurcation diagram
    println("\nGenerating bifurcation diagram...")
    
    # Extract data for plotting
    param_values = [pt.param for pt in br_ee.branch]
    E_values = [pt.u[1] for pt in br_ee.branch]
    I_values = [pt.u[2] for pt in br_ee.branch]
    stability = [pt.stable for pt in br_ee.branch]
    
    # Create plot
    p1 = plot(param_values, E_values, 
              linecolor=ifelse.(stability, :blue, :red),
              xlabel="E→E Connectivity Strength",
              ylabel="Excitatory Activity (E)",
              title="FoI Bifurcation: E→E Connectivity",
              legend=false,
              linewidth=2)
    
    # Mark bifurcation points
    if !isempty(br_ee.specialpoint)
        bif_params = [pt.param for pt in br_ee.specialpoint]
        bif_E = [pt.u[1] for pt in br_ee.specialpoint]
        scatter!(p1, bif_params, bif_E, 
                marker=:star, 
                markersize=8, 
                markercolor=:gold,
                label="Bifurcation Points")
    end
    
    # Add stability legend
    plot!(p1, [], [], linecolor=:blue, linewidth=2, label="Stable")
    plot!(p1, [], [], linecolor=:red, linewidth=2, label="Unstable")
    
    savefig(p1, "foi_bifurcation_ee_connectivity.png")
    println("✓ Saved: foi_bifurcation_ee_connectivity.png")
    
    # Also plot inhibitory population
    p2 = plot(param_values, I_values,
              linecolor=ifelse.(stability, :blue, :red),
              xlabel="E→E Connectivity Strength",
              ylabel="Inhibitory Activity (I)",
              title="FoI Bifurcation: E→E Connectivity (I population)",
              legend=false,
              linewidth=2)
    
    if !isempty(br_ee.specialpoint)
        bif_I = [pt.u[2] for pt in br_ee.specialpoint]
        scatter!(p2, bif_params, bif_I, 
                marker=:star, 
                markersize=8, 
                markercolor=:gold,
                label="Bifurcation Points")
    end
    
    plot!(p2, [], [], linecolor=:blue, linewidth=2, label="Stable")
    plot!(p2, [], [], linecolor=:red, linewidth=2, label="Unstable")
    
    savefig(p2, "foi_bifurcation_ee_connectivity_inhibitory.png")
    println("✓ Saved: foi_bifurcation_ee_connectivity_inhibitory.png")
    
    # Combined plot
    p_combined = plot(p1, p2, layout=(2,1), size=(800, 800))
    savefig(p_combined, "foi_bifurcation_ee_connectivity_combined.png")
    println("✓ Saved: foi_bifurcation_ee_connectivity_combined.png")
    
catch e
    println("⚠ Continuation failed: $(typeof(e))")
    println("  This can happen if:")
    println("  - Initial condition is not close to a steady state")
    println("  - Parameter range is too extreme")
    println("  - Step sizes are too large")
    println("  Error message: $e")
end

println()

#=============================================================================
Bifurcation Analysis 2: Varying Inhibitory Failing Threshold
=============================================================================#

println("\n### Bifurcation 2: Varying Inhibitory Failing Threshold ###\n")

# This is the key parameter that creates FoI dynamics
# Create a lens for the failing threshold of the inhibitory nonlinearity
# We need to access params.nonlinearity[2].θ_failing
using Accessors
lens_failing = @optic _.nonlinearity[2].θ_failing

# Initial condition
u0_2 = reshape([0.05, 0.05], 1, 2)

# Create bifurcation problem
prob_failing = create_bifurcation_problem(base_params, lens_failing, u0=u0_2)

# Set up continuation parameters
# Explore failing threshold from 0.4 to 1.2
opts_failing = create_default_continuation_opts(
    p_min=0.4,
    p_max=1.2,
    max_steps=150,
    dsmax=0.08,
    ds=0.02
)

println("Running continuation analysis...")
println("  Parameter: Inhibitory failing threshold (θ_failing)")
println("  Range: [$(opts_failing.p_min), $(opts_failing.p_max)]")
println()

try
    br_failing = continuation(prob_failing, PALC(), opts_failing; verbosity=0)
    
    println("✓ Continuation completed successfully")
    println("  Steps: $(length(br_failing.branch))")
    println("  Bifurcations detected: $(length(br_failing.specialpoint))")
    
    # Plot bifurcation diagram
    println("\nGenerating bifurcation diagram...")
    
    # Extract data for plotting
    param_values = [pt.param for pt in br_failing.branch]
    E_values = [pt.u[1] for pt in br_failing.branch]
    I_values = [pt.u[2] for pt in br_failing.branch]
    stability = [pt.stable for pt in br_failing.branch]
    
    # Create plot
    p3 = plot(param_values, E_values,
              linecolor=ifelse.(stability, :green, :orange),
              xlabel="Inhibitory Failing Threshold (θ_failing)",
              ylabel="Excitatory Activity (E)",
              title="FoI Bifurcation: Failing Threshold",
              legend=false,
              linewidth=2)
    
    # Mark bifurcation points
    if !isempty(br_failing.specialpoint)
        bif_params = [pt.param for pt in br_failing.specialpoint]
        bif_E = [pt.u[1] for pt in br_failing.specialpoint]
        scatter!(p3, bif_params, bif_E,
                marker=:star,
                markersize=8,
                markercolor=:gold,
                label="Bifurcation Points")
    end
    
    plot!(p3, [], [], linecolor=:green, linewidth=2, label="Stable")
    plot!(p3, [], [], linecolor=:orange, linewidth=2, label="Unstable")
    
    savefig(p3, "foi_bifurcation_failing_threshold.png")
    println("✓ Saved: foi_bifurcation_failing_threshold.png")
    
    # Plot inhibitory population
    p4 = plot(param_values, I_values,
              linecolor=ifelse.(stability, :green, :orange),
              xlabel="Inhibitory Failing Threshold (θ_failing)",
              ylabel="Inhibitory Activity (I)",
              title="FoI Bifurcation: Failing Threshold (I population)",
              legend=false,
              linewidth=2)
    
    if !isempty(br_failing.specialpoint)
        bif_I = [pt.u[2] for pt in br_failing.specialpoint]
        scatter!(p4, bif_params, bif_I,
                marker=:star,
                markersize=8,
                markercolor=:gold,
                label="Bifurcation Points")
    end
    
    plot!(p4, [], [], linecolor=:green, linewidth=2, label="Stable")
    plot!(p4, [], [], linecolor=:orange, linewidth=2, label="Unstable")
    
    savefig(p4, "foi_bifurcation_failing_threshold_inhibitory.png")
    println("✓ Saved: foi_bifurcation_failing_threshold_inhibitory.png")
    
    # Combined plot
    p_combined2 = plot(p3, p4, layout=(2,1), size=(800, 800))
    savefig(p_combined2, "foi_bifurcation_failing_threshold_combined.png")
    println("✓ Saved: foi_bifurcation_failing_threshold_combined.png")
    
catch e
    println("⚠ Continuation failed: $(typeof(e))")
    println("  Error message: $e")
end

println()

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary: FoI Bifurcation Analysis Complete")
println("="^70)
println()
println("Generated bifurcation diagrams:")
println("  1. E→E Connectivity:")
println("     - foi_bifurcation_ee_connectivity.png")
println("     - foi_bifurcation_ee_connectivity_inhibitory.png")
println("     - foi_bifurcation_ee_connectivity_combined.png")
println()
println("  2. Inhibitory Failing Threshold:")
println("     - foi_bifurcation_failing_threshold.png")
println("     - foi_bifurcation_failing_threshold_inhibitory.png")
println("     - foi_bifurcation_failing_threshold_combined.png")
println()
println("These diagrams show how FoI dynamics emerge as parameters vary,")
println("revealing the transitions between different dynamical regimes.")
println("="^70)
