#!/usr/bin/env julia

"""
Illustrate nullclines for Wilson-Cowan Model.

This script demonstrates phase space analysis with nullclines for the oscillatory mode
from Wilson & Cowan (1973). Nullclines show where each population's rate of change is zero,
revealing fixed points and the structure of phase space dynamics.

The nullclines are curves in (E, I) phase space where:
- E-nullcline: dE/dt = 0 (vertical flow along nullcline)
- I-nullcline: dI/dt = 0 (horizontal flow along nullcline)

Fixed points occur where nullclines intersect.
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
println("Wilson-Cowan Model: Nullclines for Oscillatory Mode")
println("="^70)

#=============================================================================
Nullcline Computation Functions
=============================================================================#

"""
    compute_dE_dt_field(E_grid, I_grid, params; stimulus_value=0.0)

Compute dE/dt on a grid of (E, I) values for E-nullcline plotting.
The E-nullcline is the zero-level contour of this field.
"""
function compute_dE_dt_field(E_grid, I_grid, params; stimulus_value=0.0)
    # Extract parameters
    α_E = params.α[1]
    β_E = params.β[1]
    τ_E = params.τ[1]
    
    # Get connectivity weights
    conn = params.connectivity
    b_EE = conn[1,1].weight  # E → E
    b_EI = conn[1,2].weight  # I → E
    
    # Get nonlinearity for E population
    nonlin_E = params.nonlinearity isa Tuple ? params.nonlinearity[1] : params.nonlinearity
    
    # Compute dE/dt at each grid point
    dE_dt = similar(E_grid)
    for i in eachindex(E_grid)
        E = E_grid[i]
        I = I_grid[i]
        input = stimulus_value + b_EE * E + b_EI * I
        f_val = 1.0 / (1.0 + exp(-nonlin_E.a * (input - nonlin_E.θ)))
        dE_dt[i] = (-α_E * E + β_E * (1 - E) * f_val) / τ_E
    end
    
    return dE_dt
end

"""
    compute_dI_dt_field(E_grid, I_grid, params; stimulus_value=0.0)

Compute dI/dt on a grid of (E, I) values for I-nullcline plotting.
The I-nullcline is the zero-level contour of this field.
"""
function compute_dI_dt_field(E_grid, I_grid, params; stimulus_value=0.0)
    # Extract parameters
    α_I = params.α[2]
    β_I = params.β[2]
    τ_I = params.τ[2]
    
    # Get connectivity weights
    conn = params.connectivity
    b_IE = conn[2,1].weight  # E → I
    b_II = conn[2,2].weight  # I → I
    
    # Get nonlinearity for I population
    nonlin_I = params.nonlinearity isa Tuple ? params.nonlinearity[2] : params.nonlinearity
    
    # Compute dI/dt at each grid point
    dI_dt = similar(E_grid)
    for i in eachindex(E_grid)
        E = E_grid[i]
        I = I_grid[i]
        input = stimulus_value + b_IE * E + b_II * I
        f_val = 1.0 / (1.0 + exp(-nonlin_I.a * (input - nonlin_I.θ)))
        dI_dt[i] = (-α_I * I + β_I * (1 - I) * f_val) / τ_I
    end
    
    return dI_dt
end



#=============================================================================
Main Analysis: Oscillatory Mode
=============================================================================#

println("\n### Oscillatory Mode (Thalamus) ###\n")

# Create oscillatory mode parameters
params_osc = create_point_model_wcm1973(:oscillatory)

println("Model parameters:")
println("  α_E = $(params_osc.α[1]), α_I = $(params_osc.α[2])")
println("  β_E = $(params_osc.β[1]), β_I = $(params_osc.β[2])")
println("  τ_E = $(params_osc.τ[1]), τ_I = $(params_osc.τ[2])")
println("  b_EE = $(params_osc.connectivity[1,1].weight)")
println("  b_EI = $(params_osc.connectivity[1,2].weight)")
println("  b_IE = $(params_osc.connectivity[2,1].weight)")
println("  b_II = $(params_osc.connectivity[2,2].weight)")

# Compute nullclines using grid-based approach for contour plotting
println("\nComputing nullcline fields...")
E_range = range(0.0, 0.5, length=200)
I_range = range(0.0, 0.5, length=200)

# Create meshgrid for contour plotting
E_grid = [E for E in E_range, I in I_range]
I_grid = [I for E in E_range, I in I_range]

# Compute dE/dt and dI/dt fields
dE_dt_field = compute_dE_dt_field(E_grid, I_grid, params_osc, stimulus_value=0.0)
dI_dt_field = compute_dI_dt_field(E_grid, I_grid, params_osc, stimulus_value=0.0)

println("✓ Nullcline fields computed")

# Simulate trajectory - use solve_model with initial condition that shows oscillations
println("\nSimulating trajectory...")
A₀_osc = reshape([0.3, 0.2], 1, 2)  # Initial condition that produces oscillations
tspan = (0.0, 300.0)
sol = solve_model(A₀_osc, tspan, params_osc, saveat=0.5)

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
    label="E-nullcline (dE/dt=0)",
    linewidth=2.5,
    color=:blue,
    linestyle=:solid)

contour!(p, E_range, I_range, dI_dt_field',
    levels=[0.0],
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
title!(p, "Phase Portrait with Nullclines\nOscillatory Mode (WCM 1973)")
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
println("the oscillatory mode from Wilson & Cowan (1973):")
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
