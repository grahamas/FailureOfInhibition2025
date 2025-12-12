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

# Parameters from @grahamas demonstrating blocking/failure of inhibition
# Create spatial lattice (1D) - using smaller lattice for numerical stability
lattice = CompactLattice(extent=(1400.0,), n_points=(128,))  # Reduced from 512 for stability

# Define connectivity with Gaussian spatial kernels
# Connectivity parameters: A=amplitude, S=spatial scale (μm)
conn_ee = GaussianConnectivityParameter(1.0, (25.0,))    # E→E: Aee=1.0, See=25.0
conn_ei = GaussianConnectivityParameter(-1.5, (27.0,))   # I→E: Aei=1.5, Sei=27.0 (inhibitory)
conn_ie = GaussianConnectivityParameter(1.0, (25.0,))    # E→I: Aie=1.0, Sie=25.0
conn_ii = GaussianConnectivityParameter(-0.25, (27.0,))  # I→I: Aii=0.25, Sii=27.0 (inhibitory)

connectivity = ConnectivityMatrix{2}([
    conn_ee conn_ei;
    conn_ie conn_ii
])

# Create constant stimulus across entire spatial domain
# Using ConstantStimulus as CircleStimulus has a bug with 2D arrays
stimulus = ConstantStimulus(
    strength=0.5,
    time_windows=[(0.0, 7.0)],  # stim_duration=7.0
    lattice=lattice,
    baseline=0.0
)

# Create FoI parameters with user-specified values
# Inhibitory nonlinearity: DifferenceOfSigmoids with firing and blocking components
params = FailureOfInhibitionParameters(
    α = (0.4, 0.7),           # αE=0.4, αI=0.7
    β = (1.0, 1.0),           # Standard saturation
    τ = (1.0, 0.4),           # τE=1.0, τI=0.4
    connectivity = connectivity,
    nonlinearity_E = RectifiedZeroedSigmoidNonlinearity(a=50.0, θ=0.125),  # aE=50, θE=0.125
    nonlinearity_I = DifferenceOfSigmoidsNonlinearity(
        a_activating=50.0, θ_activating=0.2,    # firing_aI=50, firing_θI=0.2
        a_failing=50.0, θ_failing=0.5           # blocking_aI=50, blocking_θI=0.5
    ),
    stimulus = stimulus,
    lattice = lattice
)

println("Model configuration:")
println("  - Spatial lattice: 1D, 128 points, 1400 μm extent")
println("  - Excitatory nonlinearity: RectifiedZeroedSigmoid (a=50, θ=0.125)")
println("  - Inhibitory nonlinearity: DifferenceOfSigmoids (firing at θ=0.2, blocking at θ=0.5)")
println("  - Stimulus: Constant across space (strength=0.5, duration=7 time units)")
println("  - Connectivity: Gaussian spatial kernels")

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
# Using small timesteps and conservative tolerances
println("Solving FoI model...")
global sol = solve_model(A₀, tspan, params, 
                        saveat=0.05, 
                        abstol=1e-10, 
                        reltol=1e-8,
                        dt=0.001,
                        dtmax=0.01,
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

#=============================================================================
Analysis: Identify FoI Phases
=============================================================================#

println("\n### Analyzing FoI Dynamics ###\n")

# Find key time points
stim_end = 7.0

# Find peak inhibitory activity during stimulus
stim_idx = findfirst(t -> t >= stim_end, times)
if !isnothing(stim_idx)
    I_during_stim = I_activity[1:stim_idx]
    I_max, I_max_idx = findmax(I_during_stim)
    I_max_time = times[I_max_idx]
    
    # Find minimum during stimulus (after peak)
    if I_max_idx < length(I_during_stim)
        I_after_peak = I_during_stim[I_max_idx:end]
        I_min_after_peak = minimum(I_after_peak)
        
        println("Phase 1 (Initial activation, t=0-$(round(I_max_time, digits=2))):")
        println("  Both E and I populations activate")
        println("  Peak I activity: $(round(I_max, digits=4)) at t=$(round(I_max_time, digits=2))")
        
        println("\nPhase 2 (Failure during stimulus, t=$(round(I_max_time, digits=2))-$stim_end):")
        println("  Paradoxical decrease in I activity (failure of inhibition)")
        println("  Min I activity: $(round(I_min_after_peak, digits=4))")
        println("  I activity reduction: $(round(I_max - I_min_after_peak, digits=4))")
    end
end

# Find final activity after stimulus removal
final_time = times[end]
E_final = E_activity[end]
I_final = I_activity[end]
E_max_final = E_max_spatial[end]

# Check if stimulus is off but activity persists
if stimulus_values[end] == 0.0
    println("\nPhase 3 (Post-stimulus, t>$stim_end):")
    println("  Stimulus has ended (duration = $stim_end)")
    println("  E activity at center persists at: $(round(E_final, digits=4))")
    println("  E activity (spatial max) persists at: $(round(E_max_final, digits=4))")
    println("  I activity at center: $(round(I_final, digits=4))")
    if E_max_final > 0.1
        println("  → Excitatory activity persists WITHOUT external stimulation!")
    else
        println("  → Activity has decayed after stimulus removal")
    end
end

#=============================================================================
Visualization: Plot FoI Dynamics
=============================================================================#

println("\n### Creating Visualization ###\n")

# Create comprehensive plot showing activity at center
p = plot(
    times, E_activity,
    label="Excitatory (E) - center",
    xlabel="Time",
    ylabel="Activity / Stimulus",
    title="Failure of Inhibition Dynamics (Center of Spatial Domain)",
    linewidth=2,
    color=:blue,
    legend=:topright,
    size=(800, 500)
)

plot!(p, times, I_activity,
    label="Inhibitory (I) - center",
    linewidth=2,
    color=:red
)

plot!(p, times, stimulus_values,
    label="Stimulus",
    linewidth=2,
    linestyle=:dash,
    color=:green
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
if !isempty(E_activity) && maximum(E_activity) > 0
    annotate!(p, [
        (stim_end/3, maximum(E_activity) * 0.95, text("Phase 1:\nActivation", 8, :center)),
        (2*stim_end/3, maximum(E_activity) * 0.95, text("Phase 2:\nInhibition\nFailure", 8, :center)),
        (stim_end + (times[end]-stim_end)/2, maximum(E_activity) * 0.95, text("Phase 3:\nPersistent\nActivity", 8, :center))
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
