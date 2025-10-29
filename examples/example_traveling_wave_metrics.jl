#!/usr/bin/env julia

"""
Example demonstrating traveling wave analysis metrics.

This shows how to use the traveling wave analysis functions to:
1. Detect traveling peaks in spatiotemporal activity patterns
2. Measure decay rates
3. Compute amplitude
4. Calculate distance traveled by activity peaks
5. Measure spatial width (half-max half-width)
"""

using FailureOfInhibition2025
using Statistics

println("="^70)
println("Traveling Wave Analysis Example")
println("="^70)

#=============================================================================
Setup: Create a 1D spatial model
=============================================================================#

println("\n### Setting up spatial model ###\n")

# Create a 1D spatial lattice
lattice = CompactLattice(extent=(20.0,), n_points=(101,))

# Define connectivity that can support traveling waves
# Use stronger excitatory connectivity with local inhibition
conn = GaussianConnectivityParameter(0.8, (3.0,))  # Broader excitatory coupling

connectivity = ConnectivityMatrix{1}(reshape([conn], 1, 1))

# Create parameters for spatial model
params = WilsonCowanParameters{1}(
    α = (1.0,),              # Decay rate
    β = (1.0,),              # Saturation coefficient
    τ = (8.0,),              # Time constant
    connectivity = connectivity,
    nonlinearity = SigmoidNonlinearity(a=2.5, θ=0.25),  # Steeper nonlinearity
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E",)
)

println("Created 1D spatial model:")
println("  - Spatial extent: 20.0 units")
println("  - Number of points: 101")
println("  - Gaussian connectivity width: 3.0")

#=============================================================================
Simulation 1: Localized initial condition
=============================================================================#

println("\n### Simulation 1: Localized bump ###\n")

# Create initial condition with strong localized activity
A₀_localized = zeros(101, 1)
A₀_localized[15:20, 1] .= 0.6  # Localized bump near left side

# Solve the model
tspan = (0.0, 40.0)
sol_localized = solve_model(A₀_localized, tspan, params, saveat=0.2)

println("Simulation complete: $(length(sol_localized.t)) time points")

# Analyze traveling wave metrics
println("\nAnalyzing traveling wave metrics...")

# 1. Detect traveling peak
has_peak, trajectory, peak_times = detect_traveling_peak(sol_localized, 1, threshold=0.15)

println("\n1. Traveling Peak Detection:")
println("   - Peak detected: $has_peak")
if has_peak
    println("   - Number of tracked time points: $(length(trajectory))")
    println("   - Starting position (index): $(trajectory[1])")
    println("   - Ending position (index): $(trajectory[end])")
    println("   - Index displacement: $(trajectory[end] - trajectory[1])")
end

# 2. Compute decay rate
decay_rate, amplitudes = compute_decay_rate(sol_localized, 1)

println("\n2. Decay Rate:")
if decay_rate !== nothing
    println("   - Decay rate λ: $(round(decay_rate, digits=4)) (1/time)")
    println("   - Half-life: $(round(log(2)/decay_rate, digits=2)) time units")
else
    println("   - No significant decay detected")
end
println("   - Initial amplitude: $(round(amplitudes[1], digits=4))")
println("   - Final amplitude: $(round(amplitudes[end], digits=4))")

# 3. Compute amplitude (different methods)
amp_max = compute_amplitude(sol_localized, 1, method=:max)
amp_mean = compute_amplitude(sol_localized, 1, method=:mean_max)

println("\n3. Amplitude Metrics:")
println("   - Maximum amplitude: $(round(amp_max, digits=4))")
println("   - Mean maximum amplitude: $(round(amp_mean, digits=4))")

# 4. Compute distance traveled
distance_idx, traj = compute_distance_traveled(sol_localized, 1, nothing, threshold=0.15)
distance_phys, _ = compute_distance_traveled(sol_localized, 1, lattice, threshold=0.15)

println("\n4. Distance Traveled:")
println("   - Distance (index units): $(round(distance_idx, digits=2))")
println("   - Distance (physical units): $(round(distance_phys, digits=2))")
if distance_phys > 0
    avg_speed = distance_phys / (peak_times[end] - peak_times[1])
    println("   - Average speed: $(round(avg_speed, digits=4)) units/time")
end

# 5. Compute width at different times
# At peak activity time
peak_time_idx = argmax([maximum(u[:, 1]) for u in sol_localized.u])
width_peak, half_max, profile_peak = compute_half_max_width(sol_localized, 1, peak_time_idx, lattice)

# At final time
width_final, _, profile_final = compute_half_max_width(sol_localized, 1, length(sol_localized.u), lattice)

println("\n5. Spatial Width (Half-Max Half-Width):")
println("   - Width at peak activity (t=$(round(sol_localized.t[peak_time_idx], digits=1))): $(round(width_peak, digits=3)) units")
println("   - Half-maximum threshold: $(round(half_max, digits=4))")
println("   - Width at final time: $(round(width_final, digits=3)) units")

#=============================================================================
Simulation 2: Asymmetric initial condition (more likely to travel)
=============================================================================#

println("\n\n### Simulation 2: Asymmetric initialization ###\n")

# Create initial condition with asymmetric profile to encourage traveling
A₀_asymmetric = zeros(101, 1)
# Gradual rise and sharp fall
for i in 1:25
    A₀_asymmetric[i, 1] = 0.5 * exp(-((i - 15)^2) / 50.0)
end

sol_asymmetric = solve_model(A₀_asymmetric, tspan, params, saveat=0.2)

println("Simulation complete: $(length(sol_asymmetric.t)) time points")

# Analyze this simulation
println("\nAnalyzing metrics for asymmetric case...")

has_peak2, traj2, times2 = detect_traveling_peak(sol_asymmetric, 1, threshold=0.1)
decay_rate2, amps2 = compute_decay_rate(sol_asymmetric, 1)
dist2, _ = compute_distance_traveled(sol_asymmetric, 1, lattice, threshold=0.1)

println("\nComparison with Simulation 1:")
println("  Traveling peak detected:")
println("    - Localized: $has_peak")
println("    - Asymmetric: $has_peak2")
println("  Distance traveled:")
println("    - Localized: $(round(distance_phys, digits=2)) units")
println("    - Asymmetric: $(round(dist2, digits=2)) units")
if decay_rate !== nothing && decay_rate2 !== nothing
    println("  Decay rate:")
    println("    - Localized: $(round(decay_rate, digits=4))")
    println("    - Asymmetric: $(round(decay_rate2, digits=4))")
end

#=============================================================================
Simulation 3: Multi-population model
=============================================================================#

println("\n\n### Simulation 3: Two-population model (E-I) ###\n")

# Create connectivity for E-I system
conn_ee = GaussianConnectivityParameter(1.0, (2.5,))   # E → E
conn_ei = GaussianConnectivityParameter(-0.6, (2.0,))  # I → E
conn_ie = GaussianConnectivityParameter(0.7, (2.5,))   # E → I
conn_ii = GaussianConnectivityParameter(-0.3, (1.5,))  # I → I

connectivity_ei = ConnectivityMatrix{2}([
    conn_ee conn_ei;
    conn_ie conn_ii
])

params_ei = WilsonCowanParameters{2}(
    α = (1.0, 1.5),
    β = (1.0, 1.0),
    τ = (8.0, 6.0),
    connectivity = connectivity_ei,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.3),
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E", "I")
)

# Initial condition for both populations
A₀_ei = zeros(101, 2)
A₀_ei[15:20, 1] .= 0.5  # Excitatory bump
A₀_ei[15:20, 2] .= 0.2  # Smaller inhibitory bump

sol_ei = solve_model(A₀_ei, (0.0, 50.0), params_ei, saveat=0.2)

println("Simulation complete: $(length(sol_ei.t)) time points")

# Analyze both populations
println("\nAnalyzing both populations...")

# Excitatory population
has_peak_E, traj_E, _ = detect_traveling_peak(sol_ei, 1, threshold=0.15)
dist_E, _ = compute_distance_traveled(sol_ei, 1, lattice, threshold=0.15)
width_E, _, _ = compute_half_max_width(sol_ei, 1, nothing, lattice)

# Inhibitory population
has_peak_I, traj_I, _ = detect_traveling_peak(sol_ei, 2, threshold=0.1)
dist_I, _ = compute_distance_traveled(sol_ei, 2, lattice, threshold=0.1)
width_I, _, _ = compute_half_max_width(sol_ei, 2, nothing, lattice)

println("\nExcitatory (E) Population:")
println("  - Traveling peak detected: $has_peak_E")
println("  - Distance traveled: $(round(dist_E, digits=2)) units")
println("  - Spatial width: $(round(width_E, digits=3)) units")

println("\nInhibitory (I) Population:")
println("  - Traveling peak detected: $has_peak_I")
println("  - Distance traveled: $(round(dist_I, digits=2)) units")
println("  - Spatial width: $(round(width_I, digits=3)) units")

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary")
println("="^70)
println()
println("This example demonstrated:")
println("  ✓ detect_traveling_peak() - Detecting traveling activity peaks")
println("  ✓ compute_decay_rate() - Measuring exponential decay of activity")
println("  ✓ compute_amplitude() - Computing maximum and mean amplitudes")
println("  ✓ compute_distance_traveled() - Tracking peak movement in space")
println("  ✓ compute_half_max_width() - Measuring spatial extent of activity")
println()
println("Key findings:")
println("  • Localized initial conditions can produce traveling or decaying waves")
println("  • Asymmetric profiles may enhance traveling behavior")
println("  • Multi-population models show distinct wave properties per population")
println("  • All metrics support both index and physical distance units")
println()
println("These metrics are useful for:")
println("  • Characterizing traveling wave dynamics in neural fields")
println("  • Comparing different parameter regimes")
println("  • Identifying parameter regions supporting wave propagation")
println("  • Analyzing failure of inhibition and related phenomena")
println()
println("="^70)
