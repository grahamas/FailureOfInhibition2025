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
    compute_nullcline_e(I_values, params; stimulus_value=0.0)

Compute E-nullcline: values of E where dE/dt = 0 for given I values.

For each I value, solve for E such that:
    -α_E * E + β_E * (1 - E) * f(S + C_EE*E + C_EI*I) = 0

where C_EE and C_EI are connectivity weights.
"""
function compute_nullcline_e(I_values, params; stimulus_value=0.0)
    # Extract parameters
    α_E = params.α[1]
    β_E = params.β[1]
    τ_E = params.τ[1]
    
    # Get connectivity weights (assuming ScalarConnectivity for point models)
    conn = params.connectivity
    b_EE = conn[1,1].weight  # E → E
    b_EI = conn[1,2].weight  # I → E
    
    # Get nonlinearity for E population
    nonlin_E = params.nonlinearity isa Tuple ? params.nonlinearity[1] : params.nonlinearity
    
    E_values = Float64[]
    
    for I in I_values
        # We need to solve for E such that dE/dt = 0
        # This means: -α_E * E + β_E * (1 - E) * σ(S + b_EE*E + b_EI*I) = 0
        # Rearranging: α_E * E = β_E * (1 - E) * σ(S + b_EE*E + b_EI*I)
        
        # Use a simple bisection/search to find E where dE/dt = 0
        function dE_dt(E)
            input = stimulus_value + b_EE * E + b_EI * I
            f_val = apply_nonlinearity_scalar(nonlin_E, input)
            return (-α_E * E + β_E * (1 - E) * f_val) / τ_E
        end
        
        # Search for zero crossing in [0, 1] range
        E_range = range(0.0, 1.0, length=1000)
        dE_values = [dE_dt(E) for E in E_range]
        
        # Find multiple zero crossings (can have multiple intersections)
        E_nullcline_points = Float64[]
        for i in 1:(length(E_range)-1)
            if sign(dE_values[i]) != sign(dE_values[i+1])
                # Linear interpolation for more accurate zero crossing
                E_zero = E_range[i] - dE_values[i] * (E_range[i+1] - E_range[i]) / (dE_values[i+1] - dE_values[i])
                push!(E_nullcline_points, E_zero)
            end
        end
        
        # Use the first crossing (typically the stable branch)
        if !isempty(E_nullcline_points)
            push!(E_values, E_nullcline_points[1])
        else
            # If no crossing found, check endpoints
            if abs(dE_values[1]) < abs(dE_values[end])
                push!(E_values, E_range[1])
            else
                push!(E_values, E_range[end])
            end
        end
    end
    
    return E_values
end

"""
    compute_nullcline_i(E_values, params; stimulus_value=0.0)

Compute I-nullcline: values of I where dI/dt = 0 for given E values.

For each E value, solve for I such that:
    -α_I * I + β_I * (1 - I) * f(S + C_IE*E + C_II*I) = 0

where C_IE and C_II are connectivity weights.
"""
function compute_nullcline_i(E_values, params; stimulus_value=0.0)
    # Extract parameters for I population
    α_I = params.α[2]
    β_I = params.β[2]
    τ_I = params.τ[2]
    
    # Get connectivity weights
    conn = params.connectivity
    b_IE = conn[2,1].weight  # E → I
    b_II = conn[2,2].weight  # I → I
    
    # Get nonlinearity for I population
    nonlin_I = params.nonlinearity isa Tuple ? params.nonlinearity[2] : params.nonlinearity
    
    I_values = Float64[]
    
    for E in E_values
        # Solve for I such that dI/dt = 0
        # This means: -α_I * I + β_I * (1 - I) * σ(S + b_IE*E + b_II*I) = 0
        
        function dI_dt(I)
            input = stimulus_value + b_IE * E + b_II * I
            f_val = apply_nonlinearity_scalar(nonlin_I, input)
            return (-α_I * I + β_I * (1 - I) * f_val) / τ_I
        end
        
        # Search for zero crossing
        I_range = range(0.0, 1.0, length=1000)
        dI_values = [dI_dt(I) for I in I_range]
        
        # Find zero crossings
        I_nullcline_points = Float64[]
        for i in 1:(length(I_range)-1)
            if sign(dI_values[i]) != sign(dI_values[i+1])
                I_zero = I_range[i] - dI_values[i] * (I_range[i+1] - I_range[i]) / (dI_values[i+1] - dI_values[i])
                push!(I_nullcline_points, I_zero)
            end
        end
        
        if !isempty(I_nullcline_points)
            push!(I_values, I_nullcline_points[1])
        else
            if abs(dI_values[1]) < abs(dI_values[end])
                push!(I_values, I_range[1])
            else
                push!(I_values, I_range[end])
            end
        end
    end
    
    return I_values
end

"""
    apply_nonlinearity_scalar(nonlin, x)

Apply nonlinearity to a scalar value.
"""
function apply_nonlinearity_scalar(nonlin::SigmoidNonlinearity, x)
    return 1.0 / (1.0 + exp(-nonlin.a * (x - nonlin.θ)))
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

# Compute nullclines
println("\nComputing nullclines...")
I_range = range(0.0, 0.5, length=200)
E_range = range(0.0, 0.5, length=200)

E_nullcline = compute_nullcline_e(I_range, params_osc, stimulus_value=0.0)
I_nullcline = compute_nullcline_i(E_range, params_osc, stimulus_value=0.0)

println("✓ E-nullcline computed")
println("✓ I-nullcline computed")

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

# Plot nullclines
plot!(p, E_nullcline, I_range, 
    label="E-nullcline (dE/dt=0)",
    linewidth=2.5,
    linestyle=:solid,
    color=:blue)

plot!(p, E_range, I_nullcline,
    label="I-nullcline (dI/dt=0)",
    linewidth=2.5,
    linestyle=:solid,
    color=:red)

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

# Find and mark approximate fixed point (intersection of nullclines)
# Simple approach: find closest points
function find_fixed_point(E_nullcline, I_range, I_nullcline, E_range)
    min_dist = Inf
    fixed_E, fixed_I = 0.0, 0.0
    for (i, E) in enumerate(E_nullcline)
        I_e = I_range[i]
        for (j, I_i) in enumerate(I_nullcline)
            E_i = E_range[j]
            dist = sqrt((E - E_i)^2 + (I_e - I_i)^2)
            if dist < min_dist
                min_dist = dist
                fixed_E, fixed_I = (E + E_i) / 2, (I_e + I_i) / 2
            end
        end
    end
    return fixed_E, fixed_I
end

fixed_E, fixed_I = find_fixed_point(E_nullcline, I_range, I_nullcline, E_range)

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
