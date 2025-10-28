#!/usr/bin/env julia

"""
Example demonstrating bifurcation analysis of Wilson-Cowan models.

This example shows how to generate bifurcation diagrams for informative parameter pairs,
revealing how system dynamics change across parameter space. Based on the three dynamical
modes from Wilson & Cowan 1973, we explore parameter regions that transition between
different behaviors (steady states, oscillations, active transients).
"""

using FailureOfInhibition2025

# Load the WCM 1973 parameter creation functions
include("../test/test_wcm1973_validation.jl")

println("\n" * "="^70)
println("Bifurcation Analysis: Wilson-Cowan Model")
println("="^70)

#=============================================================================
Example 1: E-E vs I-E Coupling Strength
=============================================================================#

println("\n### Example 1: E-E vs I-E Coupling Strength ###\n")
println("This parameter pair is critical for determining stability.")
println("Higher E-E coupling (bₑₑ) increases excitatory feedback.")
println("Higher I-E coupling (bᵢₑ) increases inhibitory feedback to E population.")
println()

# Create base parameters (point model for faster computation)
base_params = create_point_model_wcm1973(:active_transient)

println("Performing 2D parameter sweep...")
println("  - E-E coupling (bₑₑ): 0.5 to 3.0")
println("  - I-E coupling (bᵢₑ): 0.5 to 3.0")
println("  - Grid: 11x11 points")
println("  - Simulation time: 500 time units")
println()

# Perform parameter sweep
# Note: Using coarser grid for demonstration; increase resolution for publication quality
diagram_ee_ie = parameter_sweep_2d(
    base_params,
    :bₑₑ, 0.5:0.25:3.0,  # E-E coupling
    :bᵢₑ, 0.5:0.25:3.0,  # I-E coupling
    tspan=(0.0, 500.0),
    saveat=0.5
)

println("✓ Sweep complete!")
println("\nResults summary:")
println("  Parameter combinations tested: $(length(diagram_ee_ie.param1_values) * length(diagram_ee_ie.param2_values))")

# Analyze results
n_oscillatory = sum([p.is_oscillatory for p in diagram_ee_ie.points])
n_steady = sum([!isnothing(p.steady_state) for p in diagram_ee_ie.points])
total_points = length(diagram_ee_ie.points)

println("  Oscillatory behavior: $n_oscillatory / $total_points points")
println("  Steady state behavior: $n_steady / $total_points points")
println()

# Show some example points
println("Example parameter combinations:")
println()

# Find a point with oscillations
for (i, p1) in enumerate(diagram_ee_ie.param1_values)
    for (j, p2) in enumerate(diagram_ee_ie.param2_values)
        point = diagram_ee_ie.points[i, j]
        if point.is_oscillatory && !isnothing(point.oscillation_period)
            println("  Oscillatory: bₑₑ=$(p1), bᵢₑ=$(p2)")
            println("    Period: $(round(point.oscillation_period, digits=2))")
            println("    Amplitude: $(round(point.oscillation_amplitude, digits=4))")
            println("    Mean E activity: $(round(point.mean_activity[1], digits=4))")
            break
        end
    end
end

println()

# Find a point with steady state
for (i, p1) in enumerate(diagram_ee_ie.param1_values)
    for (j, p2) in enumerate(diagram_ee_ie.param2_values)
        point = diagram_ee_ie.points[i, j]
        if !isnothing(point.steady_state) && !point.is_oscillatory
            println("  Steady state: bₑₑ=$(p1), bᵢₑ=$(p2)")
            println("    E activity: $(round(point.mean_activity[1], digits=4))")
            println("    I activity: $(round(point.mean_activity[2], digits=4))")
            break
        end
    end
end

#=============================================================================
Example 2: E-I vs I-I Coupling Strength
=============================================================================#

println("\n\n### Example 2: E-I vs I-I Coupling Strength ###\n")
println("This parameter pair controls the inhibitory network dynamics.")
println("E-I coupling (bₑᵢ) determines how strongly E drives I population.")
println("I-I coupling (bᵢᵢ) controls inhibitory self-regulation.")
println()

println("Performing 2D parameter sweep...")
println("  - E-I coupling (bₑᵢ): 0.5 to 3.0")
println("  - I-I coupling (bᵢᵢ): 0.1 to 2.5")
println("  - Grid: 11x11 points")
println()

diagram_ei_ii = parameter_sweep_2d(
    base_params,
    :bₑᵢ, 0.5:0.25:3.0,  # E-I coupling
    :bᵢᵢ, 0.1:0.24:2.5,  # I-I coupling
    tspan=(0.0, 500.0),
    saveat=0.5
)

println("✓ Sweep complete!")
println()

# Analyze results
n_oscillatory = sum([p.is_oscillatory for p in diagram_ei_ii.points])
println("  Oscillatory behavior: $n_oscillatory / $(length(diagram_ei_ii.points)) points")

#=============================================================================
Example 3: Sigmoid Steepness Parameters (vₑ vs vᵢ)
=============================================================================#

println("\n\n### Example 3: Sigmoid Steepness (vₑ vs vᵢ) ###\n")
println("Sigmoid steepness controls neuronal response sharpness.")
println("Higher values → more switch-like behavior")
println("Lower values → more graded responses")
println()

println("Performing 2D parameter sweep...")
println("  - E steepness (vₑ): 0.1 to 2.0")
println("  - I steepness (vᵢ): 0.1 to 2.0")
println("  - Grid: 10x10 points")
println()

diagram_ve_vi = parameter_sweep_2d(
    base_params,
    :vₑ, 0.1:0.21:2.0,  # E sigmoid steepness
    :vᵢ, 0.1:0.21:2.0,  # I sigmoid steepness
    tspan=(0.0, 500.0),
    saveat=0.5
)

println("✓ Sweep complete!")
println()

# Find oscillatory regime
n_oscillatory = sum([p.is_oscillatory for p in diagram_ve_vi.points])
println("  Oscillatory behavior: $n_oscillatory / $(length(diagram_ve_vi.points)) points")
println("  (Note: Steeper sigmoids often promote oscillations)")

#=============================================================================
Example 4: Comparing Different Base Modes
=============================================================================#

println("\n\n### Example 4: Mode-Dependent Bifurcation Structure ###\n")
println("Different WCM 1973 modes have different bifurcation structures.")
println("Testing how oscillatory mode differs from active transient mode.")
println()

# Use oscillatory mode as base
base_params_osc = create_point_model_wcm1973(:oscillatory)

println("Sweeping oscillatory mode parameters...")
diagram_osc = parameter_sweep_2d(
    base_params_osc,
    :bₑₑ, 1.0:0.25:3.0,
    :bᵢₑ, 0.5:0.25:2.5,
    tspan=(0.0, 500.0),
    saveat=0.5
)

n_osc_mode = sum([p.is_oscillatory for p in diagram_osc.points])
println("✓ Complete!")
println("  Oscillations in oscillatory mode: $n_osc_mode / $(length(diagram_osc.points)) points")
println("  (Compare to active transient mode which has fewer oscillatory regions)")

#=============================================================================
Summary and Usage Notes
=============================================================================#

println("\n" * "="^70)
println("Summary")
println("="^70)
println()
println("Bifurcation diagrams reveal:")
println("  ✓ Parameter regions with different dynamics (steady, oscillatory, transient)")
println("  ✓ Transitions between behavioral regimes")
println("  ✓ Critical parameter combinations for specific behaviors")
println()
println("Informative parameter pairs demonstrated:")
println("  1. E-E vs I-E coupling (bₑₑ vs bᵢₑ): Stability and excitation-inhibition balance")
println("  2. E-I vs I-I coupling (bₑᵢ vs bᵢᵢ): Inhibitory network dynamics")
println("  3. Sigmoid steepness (vₑ vs vᵢ): Nonlinearity effects")
println()
println("The BifurcationDiagram objects can be used for:")
println("  - Creating heatmaps of activity levels")
println("  - Plotting stability boundaries")
println("  - Identifying oscillatory vs non-oscillatory regimes")
println("  - Analyzing bifurcation curves")
println()
println("Access results via:")
println("  - diagram.points[i,j]: BifurcationPoint at (param1[i], param2[j])")
println("  - point.mean_activity: Mean activity of each population")
println("  - point.is_oscillatory: Boolean indicating oscillations")
println("  - point.oscillation_period: Period of oscillations (if present)")
println()
println("For publication-quality diagrams:")
println("  - Increase grid resolution (more parameter values)")
println("  - Increase simulation time for better convergence")
println("  - Use appropriate visualization tools (e.g., Plots.jl, Makie.jl)")
println()
println("Example visualization code (requires Plots.jl):")
println("```julia")
println("using Plots")
println()
println("# Extract mean E activity across parameter space")
println("mean_E = [diagram.points[i,j].mean_activity[1] for i in 1:length(diagram.param1_values), j in 1:length(diagram.param2_values)]")
println()
println("# Create heatmap")
println("heatmap(diagram.param2_values, diagram.param1_values, mean_E,")
println("        xlabel=\"I-E coupling (bᵢₑ)\", ylabel=\"E-E coupling (bₑₑ)\",")
println("        title=\"Mean E Activity\", colorbar_title=\"Activity\")")
println()
println("# Extract oscillatory regions")
println("is_osc = [diagram.points[i,j].is_oscillatory for i in 1:length(diagram.param1_values), j in 1:length(diagram.param2_values)]")
println("heatmap(diagram.param2_values, diagram.param1_values, is_osc,")
println("        xlabel=\"I-E coupling (bᵢₑ)\", ylabel=\"E-E coupling (bₑₑ)\",")
println("        title=\"Oscillatory Regime\", color=:viridis)")
println("```")
println()
println("="^70)
