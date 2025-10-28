#!/usr/bin/env julia

"""
Example demonstrating the three dynamical modes from Wilson & Cowan 1973.

This example shows how to set up and simulate the three dynamical modes described in:
Wilson, H. R., & Cowan, J. D. (1973). A mathematical theory of the functional 
dynamics of cortical and thalamic nervous tissue. Kybernetik, 13(2), 55-80.

The three modes are:
1. Active Transient Mode - characteristic of sensory neo-cortex
2. Oscillatory Mode - characteristic of thalamus
3. Steady-State Mode - characteristic of archi- or prefrontal cortex

Each mode exhibits distinct dynamical behavior that corresponds to different
functional properties of neural tissue.
"""

using FailureOfInhibition2025

# Load the WCM 1973 parameter creation functions
include("../test/test_wcm1973_validation.jl")

println("\n" * "="^70)
println("Wilson-Cowan Model: Three Dynamical Modes (1973)")
println("="^70)

#=============================================================================
Example 1: Point Model (Non-Spatial) Simulations
=============================================================================#

println("\n### Example 1: Point Model (Non-Spatial) Simulations ###\n")

println("Point models represent spatially localized aggregates of neurons.")
println("These are useful for understanding basic temporal dynamics without")
println("spatial complications.\n")

# Create point model parameters for each mode
params_active = create_point_model_wcm1973(:active_transient)
params_osc = create_point_model_wcm1973(:oscillatory)
params_ss = create_point_model_wcm1973(:steady_state)

println("1. Active Transient Mode (Sensory Neo-Cortex)")
println("   - Brief stimuli elicit self-generated transient responses")
println("   - Activity peaks after stimulus cessation, then decays")
println("   - Parameters: τ = $(params_active.τ[1]) msec")
println("   - Populations: $(params_active.pop_names)")

println("\n2. Oscillatory Mode (Thalamus)")
println("   - Sustained oscillations in response to adequate stimulation")
println("   - Oscillation frequency encodes stimulus properties")
println("   - Parameters: τ = $(params_osc.τ[1]) msec")
println("   - Key difference: Weaker inhibitory-inhibitory coupling")

println("\n3. Steady-State Mode (Archi- or Prefrontal Cortex)")
println("   - Spatially inhomogeneous stable states")
println("   - Can retain contour information about prior stimuli")
println("   - Parameters: τ = $(params_ss.τ[1]) msec")
println("   - Key difference: Stronger excitatory-excitatory coupling")

# Demonstrate basic dynamics for each mode
println("\n--- Computing sample dynamics ---")

# Start with low activity
A = reshape([0.1, 0.1], 1, 2)

println("\nActive Transient Mode:")
dA_active = zeros(1, 2)
wcm1973!(dA_active, A, params_active, 0.0)
println("  At A = [$(A[1,1]), $(A[1,2])]")
println("  dA/dt = [$(round(dA_active[1,1], digits=6)), $(round(dA_active[1,2], digits=6))]")

println("\nOscillatory Mode:")
dA_osc = zeros(1, 2)
wcm1973!(dA_osc, A, params_osc, 0.0)
println("  At A = [$(A[1,1]), $(A[1,2])]")
println("  dA/dt = [$(round(dA_osc[1,1], digits=6)), $(round(dA_osc[1,2], digits=6))]")

println("\nSteady-State Mode:")
dA_ss = zeros(1, 2)
wcm1973!(dA_ss, A, params_ss, 0.0)
println("  At A = [$(A[1,1]), $(A[1,2])]")
println("  dA/dt = [$(round(dA_ss[1,1], digits=6)), $(round(dA_ss[1,2], digits=6))]")

#=============================================================================
Example 2: Spatial Model Simulations
=============================================================================#

println("\n\n### Example 2: Spatial Model Simulations ###\n")

println("Spatial models include lateral connectivity, allowing for")
println("spatially structured activity patterns.\n")

# Create spatial model parameters
params_spatial_active = create_wcm1973_parameters(:active_transient)
params_spatial_osc = create_wcm1973_parameters(:oscillatory)
params_spatial_ss = create_wcm1973_parameters(:steady_state)

println("Spatial lattice configuration:")
println("  - 1D lattice with $(size(params_spatial_active.lattice)[1]) points")
println("  - Extent: 1000 μm (1 mm)")
println("  - Spacing: ~10 μm between points")
println("  - Connectivity length scales: 20-60 μm")

# Demonstrate spatial dynamics
n_points = size(params_spatial_active.lattice)[1]
A_spatial = 0.1 .* ones(n_points, 2)

println("\nActive Transient Mode (Spatial):")
dA_spatial_active = zeros(n_points, 2)
wcm1973!(dA_spatial_active, A_spatial, params_spatial_active, 0.0)
println("  Center point dynamics:")
center_idx = div(n_points, 2) + 1
println("    At A[center] = [$(A_spatial[center_idx,1]), $(A_spatial[center_idx,2])]")
println("    dA/dt[center] = [$(round(dA_spatial_active[center_idx,1], digits=6)), $(round(dA_spatial_active[center_idx,2], digits=6))]")

#=============================================================================
Example 3: Parameter Comparisons
=============================================================================#

println("\n\n### Example 3: Key Parameter Differences Between Modes ###\n")

println("Understanding how parameters differ helps understand the distinct behaviors:\n")

println("Active Transient vs. Steady-State:")
println("  - Both have same inhibitory parameters")
println("  - Steady-state has STRONGER E→E coupling (bₑₑ = 2.0 vs 1.5)")
println("  - This allows stable activity patterns to persist")
println("  - Active transient returns to rest after brief stimulus")

println("\nActive Transient vs. Oscillatory:")
println("  - Oscillatory has STEEPER inhibitory sigmoid (vᵢ = 1.0 vs 0.3)")
println("  - Oscillatory has WEAKER I→I coupling (bᵢᵢ = 0.1 vs 1.8)")
println("  - These create stronger negative feedback loops")
println("  - Result: sustained oscillations rather than transients")

println("\nOscillatory vs. Steady-State:")
println("  - Fundamentally different dynamical regimes")
println("  - Oscillatory: temporal encoding via oscillation frequency")
println("  - Steady-state: spatial encoding via stable patterns")

#=============================================================================
Example 4: Usage with ODE Solvers
=============================================================================#

println("\n\n### Example 4: Integration with ODE Solvers ###\n")

println("These parameters can be used directly with Julia's DifferentialEquations.jl:")
println()
println("```julia")
println("using DifferentialEquations")
println()
println("# Create parameters for desired mode")
println("params = create_point_model_wcm1973(:active_transient)")
println()
println("# Initial condition: (1, 2) array for point model with 2 populations")
println("A₀ = reshape([0.1, 0.1], 1, 2)")
println()
println("# Time span (in milliseconds)")
println("tspan = (0.0, 100.0)")
println()
println("# Create ODE problem")
println("prob = ODEProblem(wcm1973!, A₀, tspan, params)")
println()
println("# Solve")
println("sol = solve(prob)")
println()
println("# Extract results")
println("times = sol.t")
println("E_activity = [u[1,1] for u in sol.u]  # Excitatory population")
println("I_activity = [u[1,2] for u in sol.u]  # Inhibitory population")
println("```")

println("\nFor spatial models, use a (N_points, 2) initial condition:")
println()
println("```julia")
println("# Spatial model")
println("params_spatial = create_wcm1973_parameters(:active_transient)")
println("n_points = size(params_spatial.lattice)[1]")
println("A₀_spatial = 0.1 .* ones(n_points, 2)")
println("prob_spatial = ODEProblem(wcm1973!, A₀_spatial, tspan, params_spatial)")
println("```")

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary")
println("="^70)
println()
println("This example demonstrated how to:")
println("  ✓ Create parameters for the three WCM 1973 dynamical modes")
println("  ✓ Use point models for non-spatial dynamics")
println("  ✓ Use spatial models with lateral connectivity")
println("  ✓ Understand parameter differences between modes")
println("  ✓ Integrate with ODE solvers for time-evolution")
println()
println("For detailed validation against the 1973 paper, see:")
println("  - test/test_wcm1973_validation.jl")
println()
println("For the original paper, see:")
println("  Wilson, H. R., & Cowan, J. D. (1973). A mathematical theory")
println("  of the functional dynamics of cortical and thalamic nervous")
println("  tissue. Kybernetik, 13(2), 55-80.")
println()
println("="^70)
