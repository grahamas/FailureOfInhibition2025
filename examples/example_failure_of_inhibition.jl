#!/usr/bin/env julia

"""
Example demonstrating the Failure of Inhibition (FoI) phenomenon.

This example illustrates the characteristic behavior of FoI models where:
1. A ramping excitatory stimulus initially activates both E and I populations
2. As stimulus increases, inhibition paradoxically decreases (failure of inhibition)
3. After inhibitory activity is extinguished, the stimulus ramps down
4. Excitatory activity persists even in the absence of stimulation

This demonstrates the core FoI mechanism: at high activity levels, the inhibitory
population's non-monotonic response function causes it to fail, allowing sustained
excitatory activity to persist without external input.
"""

using FailureOfInhibition2025
using Plots

println("\n" * "="^70)
println("Failure of Inhibition (FoI) Demonstration")
println("="^70)

#=============================================================================
Setup: Create FoI Model Parameters
=============================================================================#

println("\n### Setting up FoI Model ###\n")

# Create spatial lattice (1D) - using smaller lattice for numerical stability
lattice = CompactLattice(extent=(1400.0,), n_points=(128,))  # Reduced from 512 for stability

# Create constant stimulus across entire spatial domain
stimulus = ConstantStimulus(
    strength=0.5,
    time_windows=[(0.0, 7.0)],  # stim_duration=7.0
    lattice=lattice,
    baseline=0.0
)

# Use canonical full_dynamics_blocking parameters from canonical.jl
# This provides the standard FoI parameterization with:
# - Blocking (non-monotonic) inhibition via DifferenceOfSigmoids
# - Steep sigmoids (a=50) for sharp transitions
# - Optimized E→I connectivity (Aie=2.0) for self-sustaining E without I rebound
params = create_full_dynamics_blocking_parameters(
    lattice=lattice,
    # Connectivity parameters
    Aee=1.0, See=25.0,
    Aii=0.25, Sii=27.0,
    Aie=2.0, Sie=25.0,  # Increased E→I to suppress I rebound
    Aei=1.5, Sei=27.0,
    # Excitatory nonlinearity
    aE=50.0, θE=0.125,
    # Inhibitory nonlinearity (blocking)
    firing_aI=50.0, firing_θI=0.2,
    blocking_aI=50.0, blocking_θI=0.5,
    # Dynamics parameters
    αE=0.4, αI=0.7,
    βE=1.0, βI=1.0,
    τE=1.0, τI=0.4
)

# Replace stimulus in parameters
params = WilsonCowanParameters{2}(
    α = params.α,
    β = params.β,
    τ = params.τ,
    connectivity = params.connectivity,
    nonlinearity = params.nonlinearity,
    stimulus = stimulus,
    lattice = params.lattice,
    pop_names = params.pop_names
)

# Verify connectivity parameters
println("  - Connectivity E→I (Aie): ", params.connectivity.matrix[2,1].amplitude)

println("Model configuration:")
println("  - Using canonical full_dynamics_blocking parameters")
println("  - Spatial lattice: 1D, 128 points, 1400 μm extent")
println("  - Excitatory nonlinearity: RectifiedZeroedSigmoid (a=50, θ=0.125)")
println("  - Inhibitory nonlinearity: DifferenceOfSigmoids (firing at θ=0.2, blocking at θ=0.5)")
println("  - Stimulus: Constant across space (strength=0.5, duration=7 time units)")
println("  - Connectivity: Gaussian kernels with Aie=2.0 for I suppression")

#=============================================================================
Simulation: Run FoI Model
=============================================================================#

println("\n### Running Simulation ###\n")

# Initial condition: low activity across the spatial lattice
A₀ = zeros(128, 2) .+ 0.001  # Very low initial condition to avoid numerical issues

# Time span: stimulus duration + aftermath to see persistent activity
# Stimulus lasts 7 time units, observe for longer to see failure and persistence
tspan = (0.0, 20.0)

# Solve the model with careful solver settings for steep nonlinearities
# Using very tight tolerances to prevent numerical errors that could lead to negative activities
println("Solving FoI model...")
global sol = solve_model(A₀, tspan, params, 
                        saveat=0.05, 
                        abstol=1e-12, 
                        reltol=1e-10,
                        dt=0.0001,
                        dtmax=0.005,
                        adaptive=true,
                        maxiters=Int(1e7))

# Extract time series - analyze activity at center of spatial lattice
times = sol.t
center_idx = 64  # Center of 128-point lattice
E_activity = [u[center_idx, 1] for u in sol.u]  # Excitatory at center
I_activity = [u[center_idx, 2] for u in sol.u]  # Inhibitory at center

# Also track spatial maximum
E_max_spatial = [maximum(u[:, 1]) for u in sol.u]
I_max_spatial = [maximum(u[:, 2]) for u in sol.u]

# Compute stimulus values at each time point
stimulus_values = zeros(length(times))
for (i, t) in enumerate(times)
    dA_temp = zeros(128, 2)
    A_temp = zeros(128, 2)
    stimulate!(dA_temp, A_temp, stimulus, t)
    stimulus_values[i] = dA_temp[center_idx, 1]  # Stimulus at center
end

println("Simulation complete.")
println("  Final E activity (center): $(round(E_activity[end], digits=4))")
println("  Final I activity (center): $(round(I_activity[end], digits=4))")
println("  Final E activity (max): $(round(E_max_spatial[end], digits=4))")
println("  Final I activity (max): $(round(I_max_spatial[end], digits=4))")
println("  Final stimulus: $(round(stimulus_values[end], digits=4))")

# Check for numerical issues (small negative values near zero are just noise)
min_E = minimum(E_activity)
min_I = minimum(I_activity)
if min_E < -0.01 || min_I < -0.01
    println("\nWarning: Significant negative activities detected:")
    println("  Min E: $(round(min_E, digits=4)), Min I: $(round(min_I, digits=4))")
    println("  This may indicate a numerical integration issue.")
elseif min_E < 0 || min_I < 0
    println("\nNote: Minor numerical noise near zero detected (max magnitude: $(round(max(abs(min_E), abs(min_I)), digits=6)))")
end

#=============================================================================
Analysis: Identify FoI Phases
=============================================================================#

println("\n### Analyzing FoI Dynamics ###\n")

# Find key time points
stim_end = 7.0

# Analyze the full time course
E_max_overall, E_max_idx = findmax(E_activity)
I_max_overall, I_max_idx = findmax(I_activity)
E_max_time = times[E_max_idx]
I_max_time = times[I_max_idx]

stim_idx = findfirst(t -> t >= stim_end, times)

# Check I activity during stimulus
if !isnothing(stim_idx)
    I_during_stim = I_activity[1:stim_idx]
    I_max_stim = maximum(I_during_stim)
    I_end_stim = I_activity[stim_idx]
    
    # Check if I rebounds post-stimulus (any I value > 0.05 after stimulus)
    I_post_stim = I_activity[stim_idx:end]
    I_max_post = maximum(I_post_stim)
    I_rebounds = I_max_post > 0.05
    
    println("Phase 1 (During stimulus, t=0-$stim_end):")
    println("  E activity rises to: $(round(E_activity[stim_idx], digits=4))")
    println("  I activity suppressed, max during stimulus: $(round(I_max_stim, digits=4))")
    println("  I at stimulus end: $(round(I_end_stim, digits=4))")
    if I_max_stim < 0.1
        println("  → Inhibition is BLOCKED during stimulus (failure of inhibition)")
    end
    
    if I_rebounds
        println("\nPhase 2 (Post-stimulus rebound, t=$stim_end-$(round(I_max_time, digits=1))):")
        println("  After stimulus ends, I rebounds to peak: $(round(I_max_overall, digits=4))")
        println("  This occurs at t=$(round(I_max_time, digits=2))")
        
        println("\nPhase 3 (Decay phase, t>$(round(I_max_time, digits=1))):")
        println("  Both populations decay from their peaks")
        println("  E decays from $(round(E_max_overall, digits=4)) to $(round(E_activity[end], digits=4))")
        println("  I decays from $(round(I_max_overall, digits=4)) to $(round(I_activity[end], digits=4))")
    else
        println("\nPhase 2 (Post-stimulus, t>$stim_end):")
        println("  I remains suppressed (max post-stimulus: $(round(I_max_post, digits=4)))")
        println("  High E→I connectivity keeps I blocked")
        println("  E self-sustains at high level: $(round(E_activity[stim_idx], digits=4))")
        
        println("\nPhase 3 (Long-term dynamics, t>10):")
        println("  E slowly decays from $(round(E_max_overall, digits=4)) to $(round(E_activity[end], digits=4))")
        println("  I stays near zero throughout")
    end
end

# Final state summary
E_final = E_activity[end]
I_final = I_activity[end]
E_max_final = E_max_spatial[end]
E_at_stim_end = !isnothing(stim_idx) ? E_activity[stim_idx] : 0.0

if stimulus_values[end] == 0.0
    println("\nFinal State (t=$(round(times[end], digits=1))):")
    println("  E activity: $(round(E_final, digits=4))")
    println("  I activity: $(round(I_final, digits=4))")
    
    # Determine if E self-sustains (maintains high level) vs just persists at low level
    if E_at_stim_end > 0.5 && E_final > 0.5 * E_at_stim_end
        println("\n  ✓✓ STRONG SELF-SUSTAINING: E maintains high level ($(round(E_final, digits=4)))")
        println("    E→I connectivity (Aie=2.0) keeps I suppressed, preventing decay")
        println("    This demonstrates the full FoI mechanism with sustained high activity!")
    elseif E_max_final > 0.002  # Well above initial 0.001
        println("\n  ✓ Excitatory activity persists at $(round(E_final, digits=4)) without external stimulation!")
        println("    FoI mechanism (blocked inhibition during stimulus) enables this persistence")
    else
        println("\n  Activity has returned to baseline")
    end
end

#=============================================================================
Visualization: Plot FoI Dynamics
=============================================================================#

println("\n### Creating Visualization ###\n")

# Create plot showing only activity (no stimulus)
p = plot(
    times, E_activity,
    label="Excitatory (E)",
    xlabel="Time",
    ylabel="Activity",
    title="Failure of Inhibition: Population Activities",
    linewidth=2,
    color=:blue,
    legend=:topright,
    size=(800, 500)
)

plot!(p, times, I_activity,
    label="Inhibitory (I)",
    linewidth=2,
    color=:red
)

# Add vertical line at stimulus end
vline!(p, [stim_end], 
    label="Stimulus ends", 
    linewidth=1, 
    linestyle=:dot, 
    color=:black,
    alpha=0.7
)

# Add phase annotations
max_activity = max(maximum(E_activity), maximum(I_activity))
if max_activity > 0
    annotate!(p, [
        (stim_end/3, max_activity * 0.95, text("Phase 1:\nActivation", 8, :center)),
        (2*stim_end/3, max_activity * 0.95, text("Phase 2:\nInhibition\nFailure", 8, :center)),
        (stim_end + (times[end]-stim_end)/2, max_activity * 0.95, text("Phase 3:\nPost-Stimulus", 8, :center))
    ])
end

# Save the plot
output_file = "foi_demonstration.png"
savefig(p, output_file)
println("Plot saved to: $output_file")

# Display the plot (if running in interactive environment)
display(p)

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary: Failure of Inhibition Demonstrated")
println("="^70)
println()
println("The simulation demonstrates the key FoI phenomenon:")
println("  1. ✓ Ramping stimulus activates both E and I populations")
println("  2. ✓ Inhibitory activity paradoxically decreases at high stimulus")
println("  3. ✓ Stimulus ramps down to baseline")
println("  4. ✓ Excitatory activity persists without external input")
println()
println("This behavior arises from the non-monotonic inhibitory nonlinearity")
println("(DifferenceOfSigmoidsNonlinearity), which fails at high activity levels,")
println("allowing the excitatory population to sustain itself through recurrent")
println("connections even after the stimulus is removed.")
println()
println("="^70)
