#!/usr/bin/env julia

"""
Illustrate nullclines for Wilson-Cowan Model with Blocking Inhibition.

This script demonstrates phase space analysis with nullclines using the full dynamics
model with blocking (non-monotonic) inhibition. This parameterization uses a difference
of sigmoids for the inhibitory population, creating the characteristic "failure of inhibition"
dynamics where inhibition fails at high activity levels.

The nullclines are curves in (E, I) phase space where:
- E-nullcline: dE/dt = 0 (vertical flow along nullcline)
- I-nullcline: dI/dt = 0 (horizontal flow along nullcline)

Fixed points occur where nullclines intersect. The non-monotonic inhibition creates
interesting nullcline shapes that can support multiple fixed points and complex dynamics.
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
println("Full Dynamics with Blocking Inhibition")
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

println("\n### Full Dynamics with Blocking Inhibition ###\n")

# Create parameters using the canonical constructor for full dynamics with
# blocking (difference-of-sigmoids) inhibition. Use a point model lattice so
# this example performs phase-space analysis (non-spatial).
println("Using `create_full_dynamics_blocking_parameters` for model parameters")

# Use PointLattice for a point-model analysis and prefer canonical defaults
lattice = PointLattice()

# Only specify the lattice here — other parameters use the defaults from
# `create_full_dynamics_blocking_parameters` in `src/canonical.jl`.
params_osc = create_full_dynamics_blocking_parameters(lattice = lattice)

println("Model parameters loaded from canonical constructor (defaults used)")
println("  α = $(params_osc.α), β = $(params_osc.β), τ = $(params_osc.τ)")
println("  Lattice: $(typeof(params_osc.lattice))")

# Compute nullclines using grid-based approach for contour plotting
println("\nComputing nullcline fields...")
E_range = range(0.0, 1.0, length=200)
I_range = range(0.0, 1.0, length=200)

# Create meshgrid for contour plotting
E_grid = [E for E in E_range, I in I_range]
I_grid = [I for E in E_range, I in I_range]

# Compute dE/dt and dI/dt fields using wcm1973! (computed together for efficiency)
dE_dt_field, dI_dt_field = compute_derivative_fields(E_grid, I_grid, params_osc)

println("✓ Nullcline fields computed")

# Simulate trajectory
println("\nSimulating trajectory...")
A₀_osc = reshape([0.3, 0.2], 1, 2)  # Initial condition
tspan = (0.0, 300.0)

# Solve with the standard ODE solver
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
title!(p, "Phase Portrait with Nullclines\nFull Dynamics with Blocking Inhibition")
xlims!(p, 0, 1.0)
ylims!(p, 0, 1.0)
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
title!(p2, "Activity Dynamics\nFull Dynamics with Blocking Inhibition")
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
println("the full dynamics model with blocking (non-monotonic) inhibition:")
println()
println("Key observations:")
println("  • E-nullcline (blue): curve where dE/dt = 0")
println("  • I-nullcline (red): curve where dI/dt = 0")
println("  • Fixed point (gold star): intersection of nullclines")
println("  • Trajectory (gray/purple): system evolution in phase space")
println()
println("The blocking inhibition creates a non-monotonic response in the I population,")
println("characteristic of 'failure of inhibition' dynamics where inhibition fails at")
println("high activity levels. This can produce interesting nullcline shapes with")
println("multiple intersections and complex phase space structure.")
println()
println("Files generated:")
println("  • $output_file")
println("  • $output_file2")
println()
println("="^70)
