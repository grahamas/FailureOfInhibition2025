#!/usr/bin/env julia

"""
Example illustrating different traveling wave behaviors with visualizations.

This comprehensive demonstration shows multiple types of traveling wave dynamics
in Wilson-Cowan neural field models, including:

1. Sustained traveling wave - propagates with minimal decay (wave speed 0.8)
2. Decaying traveling wave - propagates while gradually decaying (decay rate 0.08)
3. Stationary bump - localized activity that doesn't travel
4. Failed wave - quickly annihilates without propagating
5. Fast traveling wave - higher speed propagation (wave speed 1.2, moderate decay)

Each scenario is visualized using Plots.jl with spatiotemporal activity plots
and metric annotations to demonstrate how the traveling wave metrics work.

All traveling scenarios use analytical sech² solutions to ensure clear,
measurable wave propagation, demonstrating different parameter regimes
(speed, decay, width) that the metrics can distinguish.
"""

using FailureOfInhibition2025
using Plots
using Statistics

# Create experiment directory for organized output
experiment = ExperimentContext("traveling_wave_behaviors")

println("="^80)
println("Traveling Wave Behaviors - Comprehensive Demonstration")
println("="^80)
println("Outputs will be saved to: $(experiment.dir)")
println()

# Helper function to create spatiotemporal plot with metrics
function plot_spatiotemporal_with_metrics(sol, lattice, params, title_text; 
                                         pop_idx=1, threshold=0.15)
    # Extract spatial and temporal dimensions
    n_spatial = length(sol.u[1][:, pop_idx])
    n_time = length(sol.t)
    
    # Create spatiotemporal activity matrix
    activity_matrix = zeros(n_time, n_spatial)
    for (t_idx, state) in enumerate(sol.u)
        activity_matrix[t_idx, :] = state[:, pop_idx]
    end
    
    # Get spatial coordinates
    coords = coordinates(lattice)
    x_coords = [coord[1] for coord in coords]
    
    # Compute metrics
    has_peak, trajectory, peak_times = detect_traveling_peak(sol, pop_idx, threshold=threshold)
    decay_rate, amplitudes = compute_decay_rate(sol, pop_idx)
    amp_max = compute_amplitude(sol, pop_idx, method=:max)
    distance, _ = compute_distance_traveled(sol, pop_idx, lattice, threshold=threshold)
    
    # Create spatiotemporal heatmap
    p = heatmap(x_coords, sol.t, activity_matrix,
                xlabel="Space", ylabel="Time", 
                title=title_text,
                c=:viridis, clims=(0, 1.0),
                colorbar_title="Activity",
                size=(600, 400))
    
    # Overlay traveling peak trajectory if detected
    if has_peak && length(trajectory) > 1
        peak_x = [x_coords[idx] for idx in trajectory]
        plot!(p, peak_x, peak_times, 
              linewidth=2, linecolor=:red, linestyle=:dash,
              label="Peak trajectory", legend=:topright)
    end
    
    # Annotate with metrics
    metric_text = "Metrics:\n"
    metric_text *= "• Traveling: $(has_peak)\n"
    metric_text *= "• Max amplitude: $(round(amp_max, digits=3))\n"
    metric_text *= "• Distance: $(round(distance, digits=2)) units\n"
    
    if decay_rate !== nothing
        metric_text *= "• Decay rate: $(round(decay_rate, digits=4)) /time\n"
        metric_text *= "• Half-life: $(round(log(2)/decay_rate, digits=2)) time"
    else
        metric_text *= "• Decay: minimal/none"
    end
    
    # Add text annotation
    annotate!(p, x_coords[end] * 0.05, sol.t[end] * 0.85, 
             text(metric_text, :left, 8, :white))
    
    return p
end

#=============================================================================
Scenario 1: Sustained Traveling Wave (Using Analytical Solution)
=============================================================================#

println("\n### Scenario 1: Sustained Traveling Wave ###\n")

# Setup for sustained traveling wave using analytical solution
lattice1 = CompactLattice(extent=(40.0,), n_points=(201,))

# Create parameters (needed for lattice information)
conn1 = GaussianConnectivityParameter(1.0, (3.0,))
connectivity1 = ConnectivityMatrix{1}(reshape([conn1], 1, 1))

params1 = WilsonCowanParameters{1}(
    α = (1.0,),
    β = (1.0,),
    τ = (8.0,),
    connectivity = connectivity1,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
    stimulus = nothing,
    lattice = lattice1,
    pop_names = ("E",)
)

# Use analytical traveling wave solution with minimal decay
times1 = 0.0:0.25:50.0
sol1 = generate_analytical_traveling_wave(
    params1, times1, 
    wave_speed=0.8,           # Clear traveling speed
    decay_rate=0.02,          # Minimal decay
    wavenumber=1.5,           # Controls width
    initial_position=-15.0,   # Start on left
    amplitude=0.8
)

println("Generated analytical traveling wave: $(length(sol1.t)) time points")
p1 = plot_spatiotemporal_with_metrics(sol1, lattice1, params1, 
                                     "Scenario 1: Sustained Traveling Wave (Analytical)",
                                     threshold=0.15)

#=============================================================================
Scenario 2: Decaying Traveling Wave (Analytical with Higher Decay)
=============================================================================#

println("\n### Scenario 2: Decaying Traveling Wave ###\n")

# Same lattice
lattice2 = lattice1

# Same connectivity params for analytical wave
conn2 = GaussianConnectivityParameter(1.0, (3.0,))
connectivity2 = ConnectivityMatrix{1}(reshape([conn2], 1, 1))

params2 = WilsonCowanParameters{1}(
    α = (1.0,),
    β = (1.0,),
    τ = (8.0,),
    connectivity = connectivity2,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
    stimulus = nothing,
    lattice = lattice2,
    pop_names = ("E",)
)

# Generate analytical wave with significant decay
times2 = 0.0:0.25:50.0
sol2 = generate_analytical_traveling_wave(
    params2, times2,
    wave_speed=0.6,           # Slightly slower
    decay_rate=0.08,          # Higher decay rate
    wavenumber=1.5,
    initial_position=-15.0,
    amplitude=0.8
)

println("Generated decaying traveling wave: $(length(sol2.t)) time points")
p2 = plot_spatiotemporal_with_metrics(sol2, lattice2, params2,
                                     "Scenario 2: Decaying Traveling Wave",
                                     threshold=0.15)

#=============================================================================
Scenario 3: Stationary Bump (No Travel)
=============================================================================#

println("\n### Scenario 3: Stationary Bump ###\n")

lattice3 = CompactLattice(extent=(40.0,), n_points=(201,))

# Symmetric strong connectivity for stable bump
conn3 = GaussianConnectivityParameter(1.0, (2.5,))
connectivity3 = ConnectivityMatrix{1}(reshape([conn3], 1, 1))

params3 = WilsonCowanParameters{1}(
    α = (1.0,),
    β = (1.0,),
    τ = (5.0,),              # Faster dynamics for stability
    connectivity = connectivity3,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.3),
    stimulus = nothing,
    lattice = lattice3,
    pop_names = ("E",)
)

# Symmetric initial condition centered in domain
A₀_3 = zeros(201, 1)
center = 101
for i in 1:201
    A₀_3[i, 1] = 0.6 * exp(-((i - center)^2) / 100.0)
end

sol3 = solve_model(A₀_3, (0.0, 50.0), params3, saveat=0.25)

println("Simulation complete: $(length(sol3.t)) time points")
p3 = plot_spatiotemporal_with_metrics(sol3, lattice3, params3,
                                     "Scenario 3: Stationary Bump (No Travel)",
                                     threshold=0.15)

#=============================================================================
Scenario 4: Failed/Annihilating Wave
=============================================================================#

println("\n### Scenario 4: Failed Wave ###\n")

lattice4 = CompactLattice(extent=(40.0,), n_points=(201,))

# Weak connectivity - insufficient to sustain activity
conn4 = GaussianConnectivityParameter(0.5, (2.0,))
connectivity4 = ConnectivityMatrix{1}(reshape([conn4], 1, 1))

params4 = WilsonCowanParameters{1}(
    α = (1.5,),              # High decay
    β = (1.0,),
    τ = (8.0,),
    connectivity = connectivity4,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.4),  # Higher threshold
    stimulus = nothing,
    lattice = lattice4,
    pop_names = ("E",)
)

# Initial bump
A₀_4 = zeros(201, 1)
for i in 1:50
    A₀_4[i, 1] = 0.6 * exp(-((i - 25)^2) / 80.0)
end

sol4 = solve_model(A₀_4, (0.0, 50.0), params4, saveat=0.25)

println("Simulation complete: $(length(sol4.t)) time points")
p4 = plot_spatiotemporal_with_metrics(sol4, lattice4, params4,
                                     "Scenario 4: Failed/Annihilating Wave",
                                     threshold=0.1)

#=============================================================================
Scenario 5: Fast Traveling Wave (Analytical, Higher Speed)
=============================================================================#

println("\n### Scenario 5: Fast Traveling Wave ###\n")

lattice5 = CompactLattice(extent=(40.0,), n_points=(201,))

# Parameters for analytical wave
conn5 = GaussianConnectivityParameter(1.0, (3.0,))
connectivity5 = ConnectivityMatrix{1}(reshape([conn5], 1, 1))

params5 = WilsonCowanParameters{1}(
    α = (1.0,),
    β = (1.0,),
    τ = (8.0,),
    connectivity = connectivity5,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.25),
    stimulus = nothing,
    lattice = lattice5,
    pop_names = ("E",)
)

# Generate analytical wave with faster speed and moderate decay
times5 = 0.0:0.3:60.0
sol5 = generate_analytical_traveling_wave(
    params5, times5,
    wave_speed=1.2,           # Faster than scenario 1
    decay_rate=0.04,          # Moderate decay
    wavenumber=2.0,           # Narrower wave
    initial_position=-16.0,
    amplitude=0.75
)

println("Generated fast traveling wave: $(length(sol5.t)) time points")

# Plot the fast traveling wave
p5 = plot_spatiotemporal_with_metrics(sol5, lattice5, params5,
                                     "Scenario 5: Fast Traveling Wave (Analytical)",
                                     pop_idx=1, threshold=0.12)

#=============================================================================
Create combined figure and save
=============================================================================#

println("\n### Creating combined visualization ###\n")

# Combine all plots - now 5 scenarios instead of 6
combined_plot = plot(p1, p2, p3, p4, p5,
                    layout=(3, 2),
                    size=(1200, 1400),
                    plot_title="Traveling Wave Behaviors - Comprehensive Demonstration")

# Save the plot using experiment utilities
plot_file = save_plot(combined_plot, experiment.dir, "traveling_wave_behaviors")
println("Saved combined visualization to: $plot_file")

#=============================================================================
Display individual metrics summary
=============================================================================#

println("\n" * "="^80)
println("Summary of Metrics for Each Scenario")
println("="^80)

scenarios = [
    ("Sustained Traveling Wave (Analytical)", sol1, params1.lattice, 1, 0.15),
    ("Decaying Traveling Wave (Analytical)", sol2, params2.lattice, 1, 0.15),
    ("Stationary Bump", sol3, params3.lattice, 1, 0.15),
    ("Failed Wave", sol4, params4.lattice, 1, 0.1),
    ("Fast Traveling Wave (Analytical)", sol5, params5.lattice, 1, 0.12)
]

for (name, sol, lattice, pop_idx, threshold) in scenarios
    println("\n$name:")
    println("-" * "^"^60)
    
    # Compute all metrics
    has_peak, trajectory, peak_times = detect_traveling_peak(sol, pop_idx, threshold=threshold)
    decay_rate, amplitudes = compute_decay_rate(sol, pop_idx)
    amp_max = compute_amplitude(sol, pop_idx, method=:max)
    amp_mean = compute_amplitude(sol, pop_idx, method=:mean_max)
    distance, _ = compute_distance_traveled(sol, pop_idx, lattice, threshold=threshold)
    width, half_max, _ = compute_half_max_width(sol, pop_idx, nothing, lattice)
    
    println("  Traveling Peak Detected: $has_peak")
    
    if has_peak
        println("  Peak Trajectory Points: $(length(trajectory))")
        speed = distance / (peak_times[end] - peak_times[1])
        println("  Average Speed: $(round(speed, digits=3)) units/time")
    end
    
    println("  Distance Traveled: $(round(distance, digits=2)) spatial units")
    println("  Maximum Amplitude: $(round(amp_max, digits=4))")
    println("  Mean Max Amplitude: $(round(amp_mean, digits=4))")
    println("  Spatial Width (HMHW): $(round(width, digits=3)) spatial units")
    
    if decay_rate !== nothing
        println("  Decay Rate: $(round(decay_rate, digits=4)) /time")
        println("  Half-life: $(round(log(2)/decay_rate, digits=2)) time units")
        println("  Amplitude Change: $(round(amplitudes[1], digits=3)) → $(round(amplitudes[end], digits=3))")
    else
        println("  Decay: Minimal or none detected")
    end
end

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^80)
println("Key Findings and Metric Demonstrations")
println("="^80)
println()
println("This example demonstrates how traveling wave metrics distinguish between:")
println()
println("1. SUSTAINED TRAVELING WAVE")
println("   • High amplitude maintained throughout simulation")
println("   • Consistent peak velocity over time")
println("   • Large distance traveled")
println("   • Minimal decay rate")
println("   → Metrics: detect_traveling_peak=true, low decay_rate, high distance")
println()
println("2. DECAYING TRAVELING WAVE")
println("   • Clear propagation with decreasing amplitude")
println("   • Measurable decay rate and half-life")
println("   • Moderate distance traveled before dissipation")
println("   → Metrics: detect_traveling_peak=true, significant decay_rate, moderate distance")
println()
println("3. STATIONARY BUMP")
println("   • Stable localized activity that doesn't move")
println("   • Peak position remains constant")
println("   • Zero or minimal distance traveled")
println("   → Metrics: detect_traveling_peak=false, distance≈0")
println()
println("4. FAILED WAVE")
println("   • Rapid annihilation of activity")
println("   • High decay rate, short half-life")
println("   • Little to no propagation")
println("   → Metrics: detect_traveling_peak=false, high decay_rate, minimal distance")
println()
println("5. FAST TRAVELING WAVE")
println("   • Analytical solution with faster wave speed (1.2 vs 0.8 units/time)")
println("   • Moderate decay rate between scenarios 1 and 2")
println("   • Narrower wave profile (higher wavenumber)")
println("   • Demonstrates different traveling wave characteristics")
println("   → Metrics: Higher speed, moderate decay, significant distance")
println()
println("Metric Functions Demonstrated:")
println("  ✓ detect_traveling_peak() - Identifies wave propagation")
println("  ✓ compute_decay_rate() - Quantifies amplitude decay over time")
println("  ✓ compute_amplitude() - Measures activity strength")
println("  ✓ compute_distance_traveled() - Tracks spatial displacement")
println("  ✓ compute_half_max_width() - Measures spatial extent")
println()
println("All scenarios visualized in: $plot_file")

# Save experiment metadata
save_experiment_metadata(
    experiment.dir,
    description="Comprehensive demonstration of traveling wave behaviors using analytical solutions",
    additional_info=Dict(
        "n_scenarios" => 5,
        "wave_types" => ["sustained", "decaying", "stationary", "failed", "fast"]
    )
)
println("Experiment metadata saved to: $(experiment.dir)/metadata.json")
println("="^80)
