#!/usr/bin/env julia

"""
Illustrate nullclines for Wilson-Cowan Model with Blocking Inhibition.

This example demonstrates phase space analysis using library functions from
FailureOfInhibition2025.jl. It uses the full dynamics model with blocking 
(non-monotonic) inhibition which creates the characteristic "failure of inhibition"
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
Main Analysis: Full Dynamics with Blocking Inhibition
=============================================================================#

println("\n### Full Dynamics with Blocking Inhibition ###\n")

# Create parameters for point model using full_dynamics_blocking
lattice = PointLattice()
params_blocking = create_full_dynamics_blocking_parameters(lattice=lattice)

println("Using full dynamics with blocking inhibition parameters")
println("Model parameters:")
println("  α_E = $(params_blocking.α[1]), α_I = $(params_blocking.α[2])")
println("  β_E = $(params_blocking.β[1]), β_I = $(params_blocking.β[2])")
println("  τ_E = $(params_blocking.τ[1]), τ_I = $(params_blocking.τ[2])")
println("  E nonlinearity: $(typeof(params_blocking.nonlinearity[1]))")
println("  I nonlinearity: $(typeof(params_blocking.nonlinearity[2]))")

# Simulate trajectory
println("\nSimulating trajectory...")
A₀ = reshape([0.3, 0.2], 1, 2)  # Initial condition
tspan = (0.0, 300.0)

# Solve with the standard ODE solver
sol = solve_model(A₀, tspan, params_blocking, saveat=0.5)

times = sol.t
E_activity = [sol.u[i][1, 1] for i in 1:length(times)]
I_activity = [sol.u[i][1, 2] for i in 1:length(times)]

println("✓ Trajectory simulated")

# Generate phase portrait using library functions
println("\nGenerating phase portrait with nullclines...")

# Use library functions to compute phase space
E_range = 0.0:0.005:1.0
I_range = 0.0:0.005:1.0
E_grid, I_grid, dE_dt_field, dI_dt_field = compute_phase_space_derivatives(E_range, I_range, params_blocking)

# Find all fixed points
fixed_points = find_fixed_points(E_range, I_range, params_blocking)

# Create phase portrait plot
p = plot(size=(900, 900), dpi=150)

# Plot E-nullcline (dE/dt = 0)
contour!(p, collect(E_range), collect(I_range), dE_dt_field',
    levels=[0.0],
    linewidth=2.5,
    color=:blue,
    colorbar=false,
    label="")

# Plot I-nullcline (dI/dt = 0)
contour!(p, collect(E_range), collect(I_range), dI_dt_field',
    levels=[0.0],
    linewidth=2.5,
    color=:red,
    colorbar=false,
    label="")

# Add legend entries
plot!(p, [NaN], [NaN], 
    label="E-nullcline (dE/dt=0)",
    linewidth=2.5,
    color=:blue)

plot!(p, [NaN], [NaN],
    label="I-nullcline (dI/dt=0)",
    linewidth=2.5,
    color=:red)

# Plot trajectory
plot!(p, E_activity, I_activity,
    label="Trajectory",
    linewidth=1,
    linestyle=:dot,
    color=:gray,
    alpha=0.6)

# Mark start
scatter!(p, [E_activity[1]], [I_activity[1]],
    label="Start",
    marker=:circle,
    markersize=6,
    color=:green)

# Plot all fixed points
if !isempty(fixed_points)
    E_fps = [fp[1] for fp in fixed_points]
    I_fps = [fp[2] for fp in fixed_points]
    scatter!(p, E_fps, I_fps,
        label="Fixed points ($(length(fixed_points)))",
        marker=:star,
        markersize=10,
        color=:gold)
end

# Format plot
xlabel!(p, "E Activity (Excitatory)")
ylabel!(p, "I Activity (Inhibitory)")
title!(p, "Phase Portrait with Nullclines\nFull Dynamics with Blocking Inhibition")
xlims!(p, 0, 1.0)
ylims!(p, 0, 1.0)
plot!(p, legend=:topright)
plot!(p, grid=true, gridalpha=0.3)

# Save plot
output_file = "wcm_nullclines_blocking.png"
savefig(p, output_file)
println("✓ Plot saved to: $output_file")

# Also create time series plot
p2 = plot(size=(800, 400), dpi=150)
plot!(p2, times, E_activity,
    label="E (Excitatory)",
    linewidth=2,
    color=:blue)
plot!(p2, times, I_activity,
    label="I (Inhibitory)",
    linewidth=2,
    color=:red)
xlabel!(p2, "Time (msec)")
ylabel!(p2, "Activity")
title!(p2, "Activity Dynamics\nFull Dynamics with Blocking Inhibition")
plot!(p2, legend=:topright)
plot!(p2, grid=true, gridalpha=0.3)

output_file2 = "wcm_timeseries_blocking.png"
savefig(p2, output_file2)
println("✓ Time series saved to: $output_file2")

# Report fixed points found
println("\nFixed points found:")
for (i, (E, I)) in enumerate(fixed_points)
    println("  FP $i: E = $(round(E, digits=4)), I = $(round(I, digits=4))")
end

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary")
println("="^70)
println()
println("This example demonstrates phase space analysis using library functions")
println("for the full dynamics model with blocking (non-monotonic) inhibition:")
println()
println("Library functions used:")
println("  • compute_phase_space_derivatives() - Compute dE/dt and dI/dt fields")
println("  • find_fixed_points() - Find all fixed points in phase space")
println()
println("Key observations:")
println("  • E-nullcline (blue): curve where dE/dt = 0")
println("  • I-nullcline (red): curve where dI/dt = 0")
println("  • Fixed points (gold stars): intersections of nullclines")
println("  • Trajectory (gray): system evolution in phase space")
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
