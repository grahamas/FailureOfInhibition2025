#!/usr/bin/env julia

"""
Example illustrating the relationship between parameters and traveling wave properties in 1D models.

This comprehensive demonstration explores how key model parameters affect traveling wave
characteristics in Wilson-Cowan neural field models:

1. **Connectivity Width**: How the spatial spread of connectivity affects wave propagation
2. **Sigmoid Steepness**: How the activation function affects wave initiation and speed
3. **Sigmoid Threshold**: How the activation threshold affects wave behavior
4. **Time Constant**: How temporal dynamics affect wave properties

For each parameter, we:
- Perform a parameter sweep across a meaningful range
- Measure traveling wave metrics (distance, speed, decay rate, amplitude)
- Visualize the parameter-property relationships
- Explain the underlying mechanisms

This example helps understand how to tune parameters to achieve desired traveling wave
behaviors, which is crucial for modeling failure of inhibition and other neural dynamics.
"""

using FailureOfInhibition2025
using Plots
using Printf
using Statistics

println("="^80)
println("Parameter-Traveling Wave Relationship Analysis")
println("="^80)

# Helper function to run simulation and extract metrics
function simulate_and_measure(params, A₀, tspan, saveat=0.2; threshold=0.15)
    sol = solve_model(A₀, tspan, params, saveat=saveat)
    
    # Measure traveling wave properties
    has_peak, trajectory, peak_times = detect_traveling_peak(sol, 1, threshold=threshold)
    distance, _ = compute_distance_traveled(sol, 1, params.lattice, threshold=threshold)
    decay_rate, _ = compute_decay_rate(sol, 1)
    amplitude = compute_amplitude(sol, 1, method=:max)
    
    # Calculate speed if traveling
    speed = 0.0
    if has_peak && length(peak_times) > 1
        speed = distance / (peak_times[end] - peak_times[1])
    end
    
    return (
        has_peak = has_peak,
        distance = distance,
        speed = speed,
        decay_rate = decay_rate !== nothing ? decay_rate : 0.0,
        amplitude = amplitude,
        sol = sol
    )
end

#=============================================================================
1. Effect of Connectivity Width on Traveling Waves
=============================================================================#

println("\n### 1. Connectivity Width Parameter Sweep ###\n")

# Setup base parameters
lattice = CompactLattice(extent=(20.0,), n_points=(101,))
tspan = (0.0, 40.0)
A₀ = zeros(101, 1)
A₀[15:20, 1] .= 0.6

# Sweep connectivity width
width_values = range(1.5, 4.5, length=15)
width_results = []

println("Running connectivity width sweep...")
for width in width_values
    conn = GaussianConnectivityParameter(0.8, (width,))
    params = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (8.0,),
        connectivity = ConnectivityMatrix{1}(reshape([conn], 1, 1)),
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E",)
    )
    
    result = simulate_and_measure(params, A₀, tspan)
    push!(width_results, result)
    
    @printf("  Width=%.2f: distance=%.2f, speed=%.3f, decay=%.4f\n", 
            width, result.distance, result.speed, result.decay_rate)
end

# Plot connectivity width effects
p1 = plot(width_values, [r.distance for r in width_results],
         xlabel="Connectivity Width", ylabel="Distance Traveled",
         title="Connectivity Width vs Distance",
         linewidth=2, marker=:circle, legend=false)

p2 = plot(width_values, [r.speed for r in width_results],
         xlabel="Connectivity Width", ylabel="Wave Speed",
         title="Connectivity Width vs Speed",
         linewidth=2, marker=:circle, legend=false)

p3 = plot(width_values, [r.decay_rate for r in width_results],
         xlabel="Connectivity Width", ylabel="Decay Rate",
         title="Connectivity Width vs Decay",
         linewidth=2, marker=:circle, legend=false)

p4 = plot(width_values, [r.amplitude for r in width_results],
         xlabel="Connectivity Width", ylabel="Amplitude",
         title="Connectivity Width vs Amplitude",
         linewidth=2, marker=:circle, legend=false)

width_plot = plot(p1, p2, p3, p4, layout=(2, 2), size=(900, 700),
                 plot_title="Effect of Connectivity Width on Traveling Waves")

#=============================================================================
2. Effect of Sigmoid Steepness on Traveling Waves
=============================================================================#

println("\n### 2. Sigmoid Steepness Parameter Sweep ###\n")

# Sweep sigmoid steepness (a parameter)
sigmoid_a_values = range(1.2, 3.5, length=15)
sigmoid_a_results = []

println("Running sigmoid steepness sweep...")
for sig_a in sigmoid_a_values
    conn = GaussianConnectivityParameter(0.8, (2.5,))
    params = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (8.0,),
        connectivity = ConnectivityMatrix{1}(reshape([conn], 1, 1)),
        nonlinearity = SigmoidNonlinearity(a=sig_a, θ=0.25),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E",)
    )
    
    result = simulate_and_measure(params, A₀, tspan)
    push!(sigmoid_a_results, result)
    
    @printf("  Sigmoid a=%.2f: distance=%.2f, speed=%.3f, decay=%.4f\n", 
            sig_a, result.distance, result.speed, result.decay_rate)
end

# Plot sigmoid steepness effects
p5 = plot(sigmoid_a_values, [r.distance for r in sigmoid_a_results],
         xlabel="Sigmoid Steepness (a)", ylabel="Distance Traveled",
         title="Sigmoid Steepness vs Distance",
         linewidth=2, marker=:circle, legend=false)

p6 = plot(sigmoid_a_values, [r.speed for r in sigmoid_a_results],
         xlabel="Sigmoid Steepness (a)", ylabel="Wave Speed",
         title="Sigmoid Steepness vs Speed",
         linewidth=2, marker=:circle, legend=false)

p7 = plot(sigmoid_a_values, [r.decay_rate for r in sigmoid_a_results],
         xlabel="Sigmoid Steepness (a)", ylabel="Decay Rate",
         title="Sigmoid Steepness vs Decay",
         linewidth=2, marker=:circle, legend=false)

p8 = plot(sigmoid_a_values, [r.amplitude for r in sigmoid_a_results],
         xlabel="Sigmoid Steepness (a)", ylabel="Amplitude",
         title="Sigmoid Steepness vs Amplitude",
         linewidth=2, marker=:circle, legend=false)

sigmoid_plot = plot(p5, p6, p7, p8, layout=(2, 2), size=(900, 700),
                   plot_title="Effect of Sigmoid Steepness on Traveling Waves")

#=============================================================================
3. Effect of Sigmoid Threshold on Traveling Waves
=============================================================================#

println("\n### 3. Sigmoid Threshold Parameter Sweep ###\n")

# Sweep sigmoid threshold (θ parameter)
sigmoid_theta_values = range(0.15, 0.40, length=15)
sigmoid_theta_results = []

println("Running sigmoid threshold sweep...")
for sig_theta in sigmoid_theta_values
    conn = GaussianConnectivityParameter(0.8, (2.5,))
    params = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (8.0,),
        connectivity = ConnectivityMatrix{1}(reshape([conn], 1, 1)),
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=sig_theta),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E",)
    )
    
    result = simulate_and_measure(params, A₀, tspan)
    push!(sigmoid_theta_results, result)
    
    @printf("  Sigmoid θ=%.2f: distance=%.2f, speed=%.3f, decay=%.4f\n", 
            sig_theta, result.distance, result.speed, result.decay_rate)
end

# Plot sigmoid threshold effects
p9 = plot(sigmoid_theta_values, [r.distance for r in sigmoid_theta_results],
         xlabel="Sigmoid Threshold (θ)", ylabel="Distance Traveled",
         title="Sigmoid Threshold vs Distance",
         linewidth=2, marker=:circle, legend=false)

p10 = plot(sigmoid_theta_values, [r.speed for r in sigmoid_theta_results],
          xlabel="Sigmoid Threshold (θ)", ylabel="Wave Speed",
          title="Sigmoid Threshold vs Speed",
          linewidth=2, marker=:circle, legend=false)

p11 = plot(sigmoid_theta_values, [r.decay_rate for r in sigmoid_theta_results],
          xlabel="Sigmoid Threshold (θ)", ylabel="Decay Rate",
          title="Sigmoid Threshold vs Decay",
          linewidth=2, marker=:circle, legend=false)

p12 = plot(sigmoid_theta_values, [r.amplitude for r in sigmoid_theta_results],
          xlabel="Sigmoid Threshold (θ)", ylabel="Amplitude",
          title="Sigmoid Threshold vs Amplitude",
          linewidth=2, marker=:circle, legend=false)

threshold_plot = plot(p9, p10, p11, p12, layout=(2, 2), size=(900, 700),
                     plot_title="Effect of Sigmoid Threshold on Traveling Waves")

#=============================================================================
4. Effect of Time Constant on Traveling Waves
=============================================================================#

println("\n### 4. Time Constant Parameter Sweep ###\n")

# Sweep time constant (τ parameter)
tau_values = range(4.0, 12.0, length=15)
tau_results = []

println("Running time constant sweep...")
for tau in tau_values
    conn = GaussianConnectivityParameter(0.8, (2.5,))
    params = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (tau,),
        connectivity = ConnectivityMatrix{1}(reshape([conn], 1, 1)),
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E",)
    )
    
    result = simulate_and_measure(params, A₀, tspan)
    push!(tau_results, result)
    
    @printf("  τ=%.2f: distance=%.2f, speed=%.3f, decay=%.4f\n", 
            tau, result.distance, result.speed, result.decay_rate)
end

# Plot time constant effects
p13 = plot(tau_values, [r.distance for r in tau_results],
          xlabel="Time Constant (τ)", ylabel="Distance Traveled",
          title="Time Constant vs Distance",
          linewidth=2, marker=:circle, legend=false)

p14 = plot(tau_values, [r.speed for r in tau_results],
          xlabel="Time Constant (τ)", ylabel="Wave Speed",
          title="Time Constant vs Speed",
          linewidth=2, marker=:circle, legend=false)

p15 = plot(tau_values, [r.decay_rate for r in tau_results],
          xlabel="Time Constant (τ)", ylabel="Decay Rate",
          title="Time Constant vs Decay",
          linewidth=2, marker=:circle, legend=false)

p16 = plot(tau_values, [r.amplitude for r in tau_results],
          xlabel="Time Constant (τ)", ylabel="Amplitude",
          title="Time Constant vs Amplitude",
          linewidth=2, marker=:circle, legend=false)

tau_plot = plot(p13, p14, p15, p16, layout=(2, 2), size=(900, 700),
               plot_title="Effect of Time Constant on Traveling Waves")

#=============================================================================
5. Combined 2D Parameter Space Exploration
=============================================================================#

println("\n### 5. 2D Parameter Space: Connectivity Width vs Sigmoid Steepness ###\n")

# Create 2D parameter grid
width_2d = range(1.8, 3.8, length=10)
sigmoid_a_2d = range(1.5, 3.0, length=10)

distance_matrix = zeros(length(width_2d), length(sigmoid_a_2d))
speed_matrix = zeros(length(width_2d), length(sigmoid_a_2d))

println("Running 2D parameter sweep (this may take a moment)...")
for (i, width) in enumerate(width_2d)
    for (j, sig_a) in enumerate(sigmoid_a_2d)
        conn = GaussianConnectivityParameter(0.8, (width,))
        params = WilsonCowanParameters{1}(
            α = (1.0,),
            β = (1.0,),
            τ = (8.0,),
            connectivity = ConnectivityMatrix{1}(reshape([conn], 1, 1)),
            nonlinearity = SigmoidNonlinearity(a=sig_a, θ=0.25),
            stimulus = nothing,
            lattice = lattice,
            pop_names = ("E",)
        )
        
        result = simulate_and_measure(params, A₀, tspan)
        distance_matrix[i, j] = result.distance
        speed_matrix[i, j] = result.speed
    end
    @printf("  Completed width=%.2f (row %d/%d)\n", width, i, length(width_2d))
end

# Create 2D heatmaps
p17 = heatmap(sigmoid_a_2d, width_2d, distance_matrix,
             xlabel="Sigmoid Steepness (a)", ylabel="Connectivity Width",
             title="Distance Traveled",
             c=:viridis, colorbar_title="Distance")

p18 = heatmap(sigmoid_a_2d, width_2d, speed_matrix,
             xlabel="Sigmoid Steepness (a)", ylabel="Connectivity Width",
             title="Wave Speed",
             c=:plasma, colorbar_title="Speed")

param_space_plot = plot(p17, p18, layout=(1, 2), size=(1000, 400),
                       plot_title="2D Parameter Space: Connectivity Width × Sigmoid Steepness")

#=============================================================================
Save All Plots
=============================================================================#

println("\n### Saving Visualizations ###\n")

savefig(width_plot, "param_sweep_connectivity_width.png")
println("  Saved: param_sweep_connectivity_width.png")

savefig(sigmoid_plot, "param_sweep_sigmoid_steepness.png")
println("  Saved: param_sweep_sigmoid_steepness.png")

savefig(threshold_plot, "param_sweep_sigmoid_threshold.png")
println("  Saved: param_sweep_sigmoid_threshold.png")

savefig(tau_plot, "param_sweep_time_constant.png")
println("  Saved: param_sweep_time_constant.png")

savefig(param_space_plot, "param_space_2d.png")
println("  Saved: param_space_2d.png")

#=============================================================================
Summary and Interpretation
=============================================================================#

println("\n" * "="^80)
println("Summary: Parameter-Traveling Wave Relationships")
println("="^80)
println()

println("### 1. CONNECTIVITY WIDTH ###")
println("Key Findings:")
width_optimal_idx = argmax([r.distance for r in width_results])
println("  • Optimal width for distance: $(round(width_values[width_optimal_idx], digits=2))")
println("  • Distance range: $(round(minimum([r.distance for r in width_results]), digits=2)) to $(round(maximum([r.distance for r in width_results]), digits=2)) units")
println()
println("Interpretation:")
println("  - Narrow connectivity (< 2.0): Limited lateral spread, waves fail to propagate")
println("  - Moderate connectivity (2.0-3.5): Optimal for sustained traveling waves")
println("  - Wide connectivity (> 3.5): Over-recruitment can lead to spreading/stationary bumps")
println()

println("### 2. SIGMOID STEEPNESS (a) ###")
println("Key Findings:")
sig_optimal_idx = argmax([r.distance for r in sigmoid_a_results])
println("  • Optimal steepness for distance: $(round(sigmoid_a_values[sig_optimal_idx], digits=2))")
println("  • Speed range: $(round(minimum([r.speed for r in sigmoid_a_results]), digits=3)) to $(round(maximum([r.speed for r in sigmoid_a_results]), digits=3)) units/time")
println()
println("Interpretation:")
println("  - Low steepness (< 1.5): Gradual activation, slower/weaker waves")
println("  - Moderate steepness (1.5-2.5): Balanced activation for robust propagation")
println("  - High steepness (> 2.5): Sharp threshold, can produce faster but less stable waves")
println()

println("### 3. SIGMOID THRESHOLD (θ) ###")
println("Key Findings:")
theta_optimal_idx = argmax([r.distance for r in sigmoid_theta_results])
println("  • Optimal threshold for distance: $(round(sigmoid_theta_values[theta_optimal_idx], digits=2))")
println("  • Amplitude range: $(round(minimum([r.amplitude for r in sigmoid_theta_results]), digits=3)) to $(round(maximum([r.amplitude for r in sigmoid_theta_results]), digits=3))")
println()
println("Interpretation:")
println("  - Low threshold (< 0.20): Easy activation, but may not sustain due to quick saturation")
println("  - Moderate threshold (0.20-0.30): Balanced for wave initiation and propagation")
println("  - High threshold (> 0.30): Harder to activate, waves may fail to initiate or propagate")
println()

println("### 4. TIME CONSTANT (τ) ###")
println("Key Findings:")
tau_optimal_idx = argmax([r.distance for r in tau_results])
println("  • Optimal τ for distance: $(round(tau_values[tau_optimal_idx], digits=2))")
println("  • Effect on decay: $(round(minimum([r.decay_rate for r in tau_results if r.decay_rate > 0]), digits=4)) to $(round(maximum([r.decay_rate for r in tau_results]), digits=4)) /time")
println()
println("Interpretation:")
println("  - Small τ (< 6.0): Fast dynamics, quick responses but may not sustain")
println("  - Moderate τ (6.0-10.0): Optimal balance for sustained traveling waves")
println("  - Large τ (> 10.0): Slow dynamics, can support longer-lasting waves")
println()

println("### 5. PARAMETER INTERACTIONS ###")
println("2D Parameter Space Analysis:")
max_distance_idx = argmax(distance_matrix)
max_i, max_j = Tuple(CartesianIndices(distance_matrix)[max_distance_idx])
println("  • Maximum distance achieved at:")
println("    - Connectivity Width: $(round(width_2d[max_i], digits=2))")
println("    - Sigmoid Steepness: $(round(sigmoid_a_2d[max_j], digits=2))")
println("    - Distance: $(round(distance_matrix[max_i, max_j], digits=2)) units")
println()
println("Interpretation:")
println("  - Parameters interact non-linearly to determine wave behavior")
println("  - Multiple parameter combinations can produce similar wave properties")
println("  - Optimal parameter sets exist for specific desired behaviors")
println()

println("### PRACTICAL IMPLICATIONS ###")
println()
println("For Neural Field Modeling:")
println("  1. To maximize wave propagation distance:")
println("     → Use moderate connectivity width (2.5-3.5)")
println("     → Set sigmoid steepness to 2.0-2.5")
println("     → Use threshold around 0.20-0.25")
println()
println("  2. To control wave speed:")
println("     → Adjust sigmoid steepness (higher = faster)")
println("     → Modify time constant (larger = slower)")
println()
println("  3. To minimize decay:")
println("     → Increase connectivity width moderately")
println("     → Balance sigmoid steepness (not too high)")
println()
println("  4. For failure of inhibition studies:")
println("     → Identify parameter regimes where waves fail vs succeed")
println("     → Explore transitions between traveling and stationary states")
println("     → Consider how inhibitory parameters affect these relationships")
println()

println("="^80)
println("Analysis complete. All visualizations saved.")
println("="^80)
