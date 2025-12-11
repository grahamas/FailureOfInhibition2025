#!/usr/bin/env julia

"""
Illustrate nullclines for Wilson-Cowan Model.

This script demonstrates phase space analysis with nullclines for an oscillatory
Wilson-Cowan model. Nullclines show where each population's rate of change is zero,
revealing fixed points and the structure of phase space dynamics.

The nullclines are curves in (E, I) phase space where:
- E-nullcline: dE/dt = 0 (vertical flow along nullcline)
- I-nullcline: dI/dt = 0 (horizontal flow along nullcline)

Fixed points occur where nullclines intersect. This example uses parameters
tuned to show interesting nullcline shapes with a fixed point at moderate
activity levels.
"""

using FailureOfInhibition2025

# Check for and install Plots if needed
try
    using Plots
catch
    using Pkg
    println("Installing Plots...")
    Pkg.add("Plots")
    using Plots
end

println("\n" * "="^70)
println("Wilson-Cowan Model: Nullcline Analysis")
println("="^70)

#=============================================================================
Nullcline Computation Functions
=============================================================================#

"""
    compute_derivative_fields(E_grid, I_grid, params)

Compute both dE/dt and dI/dt fields on a grid of (E, I) values.
The E-nullcline is the zero-level contour of dE/dt field.
The I-nullcline is the zero-level contour of dI/dt field.
Uses wcm1973! to compute derivatives properly.

Returns (dE_dt_field, dI_dt_field)
"""
function compute_derivative_fields(E_grid, I_grid, params)
    # Compute both dE/dt and dI/dt at each grid point using wcm1973!
    dE_dt = similar(E_grid)
    dI_dt = similar(E_grid)
    
    for i in eachindex(E_grid)
        E = E_grid[i]
        I = I_grid[i]
        
        # Create state array for this point
        A = reshape([E, I], 1, 2)
        dA = zeros(1, 2)
        
        # Use wcm1973! to compute derivatives
        wcm1973!(dA, A, params, 0.0)
        
        # Extract both dE/dt and dI/dt
        dE_dt[i] = dA[1, 1]
        dI_dt[i] = dA[1, 2]
    end
    
    return dE_dt, dI_dt
end



#=============================================================================
Main Analysis: Oscillatory Mode
=============================================================================#

println("\n### Wilson-Cowan Model with Interesting Nullclines ###\n")

# Create custom parameters that produce interesting nullcline shapes
# with a fixed point at moderate activity levels
lattice = PointLattice()

# Adjusted parameters for nullcline visualization with fixed point away from origin
# Using RectifiedZeroedSigmoidNonlinearity with parameters that produce:
# - N-shaped E-nullcline to enable multiple intersections
# - Fixed point at moderate activity (not near origin)
# - Activity that stays positive (important for rectified sigmoid)
# Strategy: Moderate excitation, balanced inhibition, negative thresholds, moderate decay
vₑ, θₑ = 5.0, -0.5   # Sigmoid steepness and threshold for E (steep and negative)
vᵢ, θᵢ = 3.5, -0.3   # Sigmoid steepness and threshold for I (less steep)
bₑₑ = 2.5            # E → E (moderate excitatory self-connection)
bᵢₑ = 1.8            # I → E (moderate inhibitory to excitatory)
bₑᵢ = 3.5            # E → I (strong excitatory to inhibitory)
bᵢᵢ = 0.2            # I → I (weak inhibitory self-connection)
α_E = 1.0            # Moderate decay rate for E to prevent negative activity
α_I = 1.0            # Moderate decay rate for I to prevent negative activity

# Create nonlinearity - use RectifiedZeroedSigmoidNonlinearity
nonlinearity_e = RectifiedZeroedSigmoidNonlinearity(a=vₑ, θ=θₑ)
nonlinearity_i = RectifiedZeroedSigmoidNonlinearity(a=vᵢ, θ=θᵢ)
nonlinearity = (nonlinearity_e, nonlinearity_i)

# Create connectivity
conn_ee = ScalarConnectivity(bₑₑ)
conn_ei = ScalarConnectivity(-bᵢₑ)
conn_ie = ScalarConnectivity(bₑᵢ)
conn_ii = ScalarConnectivity(-bᵢᵢ)

connectivity = ConnectivityMatrix{2}([
    conn_ee conn_ei;
    conn_ie conn_ii
])

# Create parameters
params_osc = WilsonCowanParameters{2}(
    α = (α_E, α_I),
    β = (1.0, 1.0),
    τ = (10.0, 10.0),
    connectivity = connectivity,
    nonlinearity = nonlinearity,
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E", "I")
)

println("Model parameters:")
println("  α_E = $(params_osc.α[1]), α_I = $(params_osc.α[2])")
println("  β_E = $(params_osc.β[1]), β_I = $(params_osc.β[2])")
println("  τ_E = $(params_osc.τ[1]), τ_I = $(params_osc.τ[2])")
println("  b_EE = $bₑₑ, b_EI = -$bᵢₑ")
println("  b_IE = $bₑᵢ, b_II = -$bᵢᵢ")
println("  θ_E = $θₑ, θ_I = $θᵢ")

# Compute nullclines using grid-based approach for contour plotting
println("\nComputing nullcline fields...")
E_range = range(0.0, 0.5, length=200)
I_range = range(0.0, 0.5, length=200)

# Create meshgrid for contour plotting
E_grid = [E for E in E_range, I in I_range]
I_grid = [I for E in E_range, I in I_range]

# Compute dE/dt and dI/dt fields using wcm1973! (computed together for efficiency)
dE_dt_field, dI_dt_field = compute_derivative_fields(E_grid, I_grid, params_osc)

println("✓ Nullcline fields computed")

# Simulate trajectory - use custom solve with positive domain constraint
# This ensures activity never goes below zero, which is required for RectifiedZeroedSigmoidNonlinearity
println("\nSimulating trajectory...")
using DifferentialEquations
A₀_osc = reshape([0.3, 0.2], 1, 2)  # Initial condition that produces oscillations
tspan = (0.0, 300.0)

# Create ODE problem
prob = ODEProblem(wcm1973!, A₀_osc, tspan, params_osc)

# Create a callback to enforce non-negative activity
# This is necessary because RectifiedZeroedSigmoidNonlinearity requires non-negative inputs
# The decay term in WCM can push activity negative, so we clamp at each time step
function affect_clamp!(integrator)
    # Clamp all values to be non-negative
    for i in eachindex(integrator.u)
        if integrator.u[i] < 0.0
            integrator.u[i] = 0.0
        end
    end
    u_modified!(integrator, true)
end

# Use PeriodicCallback to clamp frequently (every 0.001 time units)
cb = PeriodicCallback(affect_clamp!, 0.001, save_positions=(false,false))

# Solve with callback
sol = solve(prob, Tsit5(), callback=cb, saveat=0.5)

times_osc = sol.t
E_activity = [sol.u[i][1, 1] for i in 1:length(times_osc)]
I_activity = [sol.u[i][1, 2] for i in 1:length(times_osc)]

println("✓ Trajectory simulated")

# Extract just the oscillatory portion (skip initial transient)
osc_start_idx = findfirst(t -> t > 50.0, times_osc)
osc_end_idx = length(times_osc)
E_activity_osc = E_activity[osc_start_idx:osc_end_idx]
I_activity_osc = I_activity[osc_start_idx:osc_end_idx]

#=============================================================================
Plotting
=============================================================================#

println("\nGenerating plots...")

# Create phase portrait with nullclines
p = plot(size=(800, 600), dpi=150)

# Plot nullclines using contour with level 0
contour!(p, E_range, I_range, dE_dt_field',
    levels=[0.0],
    linewidth=2.5,
    color=:blue,
    linestyle=:solid,
    colorbar=false,
    label="")  # Disable automatic label from contour

contour!(p, E_range, I_range, dI_dt_field',
    levels=[0.0],
    linewidth=2.5,
    color=:red,
    linestyle=:solid,
    colorbar=false,
    label="")  # Disable automatic label from contour

# Add dummy lines for legend entries (they won't be visible but will appear in legend)
plot!(p, [NaN], [NaN], 
    label="E-nullcline (dE/dt=0)",
    linewidth=2.5,
    color=:blue,
    linestyle=:solid)

plot!(p, [NaN], [NaN],
    label="I-nullcline (dI/dt=0)",
    linewidth=2.5,
    color=:red,
    linestyle=:solid)

# Plot full trajectory (lighter)
plot!(p, E_activity, I_activity,
    label="Full trajectory",
    linewidth=1,
    linestyle=:dot,
    color=:gray,
    alpha=0.4)

# Plot oscillatory portion (emphasized)
plot!(p, E_activity_osc, I_activity_osc,
    label="Limit cycle",
    linewidth=2,
    linestyle=:solid,
    color=:purple,
    alpha=0.8)

# Mark starting point of oscillations
scatter!(p, [E_activity_osc[1]], [I_activity_osc[1]],
    label="Cycle start",
    marker=:circle,
    markersize=6,
    color=:purple)

# Find and mark approximate fixed point (where both dE/dt ≈ 0 and dI/dt ≈ 0)
function find_fixed_point(E_grid, I_grid, dE_dt_field, dI_dt_field)
    # Find point where both derivatives are closest to zero
    min_sum = Inf
    fixed_E, fixed_I = 0.0, 0.0
    for i in eachindex(E_grid)
        sum_sq = dE_dt_field[i]^2 + dI_dt_field[i]^2
        if sum_sq < min_sum
            min_sum = sum_sq
            fixed_E = E_grid[i]
            fixed_I = I_grid[i]
        end
    end
    return fixed_E, fixed_I
end

fixed_E, fixed_I = find_fixed_point(E_grid, I_grid, dE_dt_field, dI_dt_field)

scatter!(p, [fixed_E], [fixed_I],
    label="Fixed point",
    marker=:star,
    markersize=10,
    color=:gold)

# Format plot
xlabel!(p, "E Activity (Excitatory)")
ylabel!(p, "I Activity (Inhibitory)")
title!(p, "Phase Portrait with Nullclines\nWilson-Cowan Model")
xlims!(p, 0, 0.5)
ylims!(p, 0, 0.5)
plot!(p, legend=:topright)
plot!(p, grid=true, gridalpha=0.3)

# Save plot
output_file = "wcm_nullclines_oscillatory.png"
savefig(p, output_file)
println("✓ Plot saved to: $output_file")

# Also create time series plot
p2 = plot(size=(800, 400), dpi=150)
plot!(p2, times_osc, E_activity,
    label="E (Excitatory)",
    linewidth=2,
    color=:blue)
plot!(p2, times_osc, I_activity,
    label="I (Inhibitory)",
    linewidth=2,
    color=:red)
xlabel!(p2, "Time (msec)")
ylabel!(p2, "Activity")
title!(p2, "Oscillatory Dynamics\nSustained oscillations after stimulus")
plot!(p2, legend=:topright)
plot!(p2, grid=true, gridalpha=0.3)

output_file2 = "wcm_timeseries_oscillatory.png"
savefig(p2, output_file2)
println("✓ Time series saved to: $output_file2")

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary")
println("="^70)
println()
println("This example demonstrates phase space analysis with nullclines for")
println("an oscillatory Wilson-Cowan model:")
println()
println("Key observations:")
println("  • E-nullcline (blue): curve where dE/dt = 0")
println("  • I-nullcline (red): curve where dI/dt = 0")
println("  • Fixed point (gold star): intersection of nullclines (unstable)")
println("  • Limit cycle (purple): stable oscillation in phase space")
println()
println("The trajectory is attracted to a stable limit cycle around the unstable")
println("fixed point, which is characteristic of the oscillatory mode. The nullclines")
println("reveal the structure of the phase space that produces these dynamics.")
println("When E is high, the system crosses the E-nullcline and E decreases.")
println("When I is high, the system crosses the I-nullcline and I decreases.")
println()
println("Files generated:")
println("  • $output_file")
println("  • $output_file2")
println()
println("="^70)
