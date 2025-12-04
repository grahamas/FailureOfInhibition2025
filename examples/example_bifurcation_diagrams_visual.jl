#!/usr/bin/env julia

"""
Example demonstrating bifurcation analysis of Wilson-Cowan models using BifurcationKit.

This example shows how to use continuation methods to trace bifurcation curves,
detect bifurcation points (Hopf, fold, etc.), and generate bifurcation diagrams
for the Wilson-Cowan model.

This builds upon the basic example_bifurcation_diagrams.jl by actually running
continuation and generating plots.
"""

using FailureOfInhibition2025
using BifurcationKit
using Plots

println("\n" * "="^70)
println("Bifurcation Diagrams: Wilson-Cowan Model")
println("Using BifurcationKit Continuation Methods")
println("="^70)

#=============================================================================
Example 1: Simple Point Model - Varying E-E Connectivity Strength
=============================================================================#

println("\n### Example 1: Point Model - Varying E-E Connectivity ###\n")

# Create point model parameters for faster computation
base_params = create_point_model_wcm1973(:active_transient)

println("Base parameters:")
println("  - Mode: Active Transient")
println("  - Populations: $(base_params.pop_names)")
println("  - E-E connectivity: $(base_params.connectivity.matrix[1,1].weight)")
println("  - I-E connectivity: $(base_params.connectivity.matrix[1,2].weight)")
println()

# Initial guess for steady state
u0 = reshape([0.1, 0.1], 1, 2)

# Create a parameter lens to vary the E-E connectivity strength
# We need to navigate to: params.connectivity.matrix[1,1].weight
param_lens = @optic _.connectivity.matrix[1,1].weight

println("Creating bifurcation problem...")
prob = create_bifurcation_problem(base_params, param_lens, u0=u0)
println("  ✓ Bifurcation problem created")
println()

# Set up continuation parameters
println("Setting up continuation parameters...")
opts = ContinuationPar(
    dsmax = 0.05,      # Maximum continuation step
    dsmin = 1e-4,      # Minimum continuation step  
    ds = 0.01,         # Initial step (positive = increase parameter)
    max_steps = 200,   # Maximum number of steps
    p_min = 0.5,       # Minimum parameter value (E-E connectivity)
    p_max = 3.0,       # Maximum parameter value
    detect_bifurcation = 3,  # Detect bifurcations
    n_inversion = 6,   # Number of eigenvalues to track
)
println("  ✓ Continuation parameters set")
println("  - Parameter range: [$(opts.p_min), $(opts.p_max)]")
println("  - Max steps: $(opts.max_steps)")
println()

# Run continuation
println("Running continuation analysis...")
println("  (This may take a moment...)")
br = continuation(prob, PALC(), opts)
println("  ✓ Continuation completed")
println("  - Number of points: $(length(br.sol))")
println()

# Display bifurcation information
if length(br.specialpoint) > 0
    println("Bifurcation points detected:")
    for (i, pt) in enumerate(br.specialpoint)
        println("  $i. Type: $(pt.type), Parameter: $(round(pt.param, digits=4))")
    end
else
    println("No bifurcation points detected")
end
println()

# Create bifurcation diagram plot
println("Creating bifurcation diagram...")
p1 = plot(br, 
    xlabel="E-E Connectivity (b_EE)", 
    ylabel="E Activity",
    title="Bifurcation Diagram: Active Transient Mode",
    legend=:topright,
    linewidth=2,
    markersize=3
)

# Save plot
savefig(p1, "bifurcation_diagram_ee_connectivity.png")
println("  ✓ Plot saved to bifurcation_diagram_ee_connectivity.png")
println()

#=============================================================================
Example 2: Varying Nonlinearity Threshold
=============================================================================#

println("\n### Example 2: Point Model - Varying E Threshold (θ_E) ###\n")

# For this we need to vary the nonlinearity parameter
# The nonlinearity is a tuple (nonlinearity_e, nonlinearity_i)
# We want to vary nonlinearity_e.θ

base_params2 = create_point_model_wcm1973(:oscillatory)
println("Base parameters:")
println("  - Mode: Oscillatory")
println("  - E threshold: $(base_params2.nonlinearity[1].θ)")
println("  - I threshold: $(base_params2.nonlinearity[2].θ)")
println()

u0_2 = reshape([0.1, 0.1], 1, 2)

# Create lens for E threshold
param_lens2 = @optic _.nonlinearity[1].θ

println("Creating bifurcation problem...")
prob2 = create_bifurcation_problem(base_params2, param_lens2, u0=u0_2)
println("  ✓ Bifurcation problem created")
println()

# Set up continuation parameters for threshold
opts2 = ContinuationPar(
    dsmax = 0.1,
    dsmin = 1e-4,
    ds = 0.02,
    max_steps = 150,
    p_min = 5.0,        # Lower threshold
    p_max = 15.0,       # Higher threshold
    detect_bifurcation = 3,
    n_inversion = 6,
)

println("Running continuation analysis...")
br2 = continuation(prob2, PALC(), opts2)
println("  ✓ Continuation completed")
println("  - Number of points: $(length(br2.sol))")
println()

# Display bifurcation information
if length(br2.specialpoint) > 0
    println("Bifurcation points detected:")
    for (i, pt) in enumerate(br2.specialpoint)
        println("  $i. Type: $(pt.type), Parameter: $(round(pt.param, digits=4))")
    end
else
    println("No bifurcation points detected")
end
println()

# Create bifurcation diagram plot
println("Creating bifurcation diagram...")
p2 = plot(br2,
    xlabel="E Threshold (θ_E)", 
    ylabel="Activity",
    title="Bifurcation Diagram: Oscillatory Mode",
    legend=:topright,
    linewidth=2,
    markersize=3
)

savefig(p2, "bifurcation_diagram_e_threshold.png")
println("  ✓ Plot saved to bifurcation_diagram_e_threshold.png")
println()

#=============================================================================
Example 3: Two-Parameter Bifurcation Diagram
=============================================================================#

println("\n### Example 3: Two-Parameter Continuation ###\n")

# For a complete bifurcation study, we can trace how bifurcation points
# move as we vary two parameters. This is more advanced but BifurcationKit
# supports it.

println("Note: Two-parameter continuation is an advanced topic.")
println("For a thorough two-parameter analysis, see BifurcationKit documentation:")
println("  https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/")
println()

#=============================================================================
Example 4: Comparing All Three WCM Modes
=============================================================================#

println("\n### Example 4: Comparing WCM Modes ###\n")

# Create a figure comparing bifurcation diagrams for all three modes
println("Analyzing all three WCM modes...")

modes = [:active_transient, :oscillatory, :steady_state]
mode_names = ["Active Transient", "Oscillatory", "Steady-State"]
plots_array = []

for (mode, mode_name) in zip(modes, mode_names)
    println("\nAnalyzing $mode_name mode...")
    
    params_mode = create_point_model_wcm1973(mode)
    u0_mode = reshape([0.1, 0.1], 1, 2)
    
    # Vary E-E connectivity
    lens_mode = @optic _.connectivity.matrix[1,1].weight
    prob_mode = create_bifurcation_problem(params_mode, lens_mode, u0=u0_mode)
    
    opts_mode = ContinuationPar(
        dsmax = 0.05,
        dsmin = 1e-4,
        ds = 0.01,
        max_steps = 200,
        p_min = 0.5,
        p_max = 3.0,
        detect_bifurcation = 3,
        n_inversion = 6,
    )
    
    br_mode = continuation(prob_mode, PALC(), opts_mode)
    println("  ✓ Continuation completed ($(length(br_mode.sol)) points)")
    
    p_mode = plot(br_mode,
        xlabel="E-E Connectivity",
        ylabel="E Activity",
        title=mode_name,
        legend=false,
        linewidth=2,
        markersize=2
    )
    
    push!(plots_array, p_mode)
end

# Combine all plots
p_combined = plot(plots_array..., layout=(1,3), size=(1200, 400))
savefig(p_combined, "bifurcation_diagrams_all_modes.png")
println("\n  ✓ Combined plot saved to bifurcation_diagrams_all_modes.png")
println()

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary")
println("="^70)
println()
println("This example demonstrated:")
println("  ✓ Running continuation analysis on Wilson-Cowan models")
println("  ✓ Detecting bifurcation points (Hopf, fold, etc.)")
println("  ✓ Generating bifurcation diagrams")
println("  ✓ Varying different parameters (connectivity, threshold)")
println("  ✓ Comparing all three WCM modes from Wilson & Cowan 1973")
println()
println("Generated plots:")
println("  - bifurcation_diagram_ee_connectivity.png")
println("  - bifurcation_diagram_e_threshold.png")
println("  - bifurcation_diagrams_all_modes.png")
println()
println("Key concepts:")
println("  - Continuation methods trace solution branches")
println("  - Bifurcation points mark qualitative changes in dynamics")
println("  - Different parameters reveal different bifurcation structures")
println("  - Each WCM mode has distinct bifurcation behavior")
println()
println("For more advanced analysis:")
println("  - Branch switching at bifurcation points")
println("  - Periodic orbit continuation (limit cycles)")
println("  - Two-parameter bifurcation diagrams")
println("  - Stability analysis")
println()
println("See BifurcationKit documentation:")
println("  https://bifurcationkit.github.io/BifurcationKitDocs.jl/stable/")
println()
println("="^70)
