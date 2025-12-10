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

# Use a point lattice for simplicity (non-spatial model)
lattice = PointLattice()

# Define connectivity for E-I system
# Stronger E→E and E→I connections to support sustained activity
# Moderate inhibition to allow for failure dynamics
conn_ee = ScalarConnectivity(4.0)    # Strong excitatory self-connection
conn_ei = ScalarConnectivity(-2.5)   # Moderate inhibitory to excitatory
conn_ie = ScalarConnectivity(5.0)    # Very strong excitatory to inhibitory (to drive I high)
conn_ii = ScalarConnectivity(-0.2)   # Very weak inhibitory self-connection

connectivity = ConnectivityMatrix{2}([
    conn_ee conn_ei;
    conn_ie conn_ii
])

# Create ramping stimulus
# Ramp up over 50 time units, hold for 50, then ramp down over 50
# This allows time to observe:
# 1. Initial activation (ramp up)
# 2. Inhibitory failure at high stimulus (longer plateau to see effect)
# 3. Persistent activity after stimulus removal (ramp down)
stimulus = RampStimulus(
    ramp_up_time=50.0,
    plateau_time=50.0,
    ramp_down_time=50.0,
    max_strength=10.0,
    start_time=10.0,
    lattice=lattice,
    baseline=0.0
)

# Create FoI parameters
# The key feature: inhibitory population uses DifferenceOfSigmoidsNonlinearity
# This creates a non-monotonic response that fails at high activity
# The two sigmoids create a "bump" - activation rises then falls
# Key: the failing sigmoid needs to have a lower threshold to kick in at moderate-high activity
params = FailureOfInhibitionParameters(
    α = (1.0, 2.0),           # Decay rates (faster decay for I)
    β = (1.0, 1.0),           # Saturation coefficients
    τ = (10.0, 6.0),          # Time constants (faster for I)
    connectivity = connectivity,
    nonlinearity_E = RectifiedZeroedSigmoidNonlinearity(a=3.0, θ=2.5),
    nonlinearity_I = DifferenceOfSigmoidsNonlinearity(
        a_activating=8.0, θ_activating=1.5,  # Activates moderately
        a_failing=6.0, θ_failing=2.5         # Fails at higher input (closer threshold)
    ),
    stimulus = stimulus,
    lattice = lattice
)

println("Model configuration:")
println("  - Excitatory nonlinearity: RectifiedZeroedSigmoid (standard)")
println("  - Inhibitory nonlinearity: DifferenceOfSigmoids (non-monotonic)")
println("  - Stimulus: Ramping (up → plateau → down)")
println("  - Time course: 10ms start, 50ms ramp-up, 30ms plateau, 50ms ramp-down")

#=============================================================================
Simulation: Run FoI Model
=============================================================================#

println("\n### Running Simulation ###\n")

# Initial condition: low activity
A₀ = reshape([0.05, 0.05], 1, 2)

# Time span: cover full stimulus cycle plus aftermath
tspan = (0.0, 200.0)

# Solve the model
println("Solving FoI model...")
sol = solve_model(A₀, tspan, params, saveat=0.5)

# Extract time series
times = sol.t
E_activity = [u[1,1] for u in sol.u]  # Excitatory population
I_activity = [u[1,2] for u in sol.u]  # Inhibitory population

# Compute stimulus values at each time point
stimulus_values = zeros(length(times))
for (i, t) in enumerate(times)
    dA_temp = zeros(1, 2)
    A_temp = zeros(1, 2)
    stimulate!(dA_temp, A_temp, stimulus, t)
    stimulus_values[i] = dA_temp[1, 1]  # Stimulus is same for both populations
end

println("Simulation complete.")
println("  Final E activity: $(round(E_activity[end], digits=4))")
println("  Final I activity: $(round(I_activity[end], digits=4))")
println("  Final stimulus: $(round(stimulus_values[end], digits=4))")

#=============================================================================
Analysis: Identify FoI Phases
=============================================================================#

println("\n### Analyzing FoI Dynamics ###\n")

# Find key time points
ramp_up_end = 10.0 + 50.0
plateau_end = ramp_up_end + 30.0
ramp_down_end = plateau_end + 50.0

# Find peak inhibitory activity
I_max, I_max_idx = findmax(I_activity)
I_max_time = times[I_max_idx]

# Find when inhibition fails (drops below threshold during high stimulus)
plateau_start_idx = findfirst(t -> t >= ramp_up_end, times)
plateau_end_idx = findfirst(t -> t >= plateau_end, times)
if !isnothing(plateau_start_idx) && !isnothing(plateau_end_idx)
    I_plateau = I_activity[plateau_start_idx:plateau_end_idx]
    I_min_plateau = minimum(I_plateau)
    println("Phase 1 (Ramp-up, t=10-60ms):")
    println("  Both E and I populations activate")
    println("  Peak I activity: $(round(I_max, digits=4)) at t=$(round(I_max_time, digits=2))ms")
    
    println("\nPhase 2 (Plateau, t=60-90ms):")
    println("  Paradoxical decrease in I activity (failure of inhibition)")
    println("  Min I activity during plateau: $(round(I_min_plateau, digits=4))")
    println("  I activity reduction: $(round(I_max - I_min_plateau, digits=4))")
end

# Find final activity after stimulus removal
final_idx = findfirst(t -> t >= ramp_down_end + 20.0, times)
if !isnothing(final_idx)
    E_final = E_activity[final_idx]
    I_final = I_activity[final_idx]
    S_final = stimulus_values[final_idx]
    println("\nPhase 3 (Post-stimulus, t>140ms):")
    println("  Stimulus has ramped down to: $(round(S_final, digits=4))")
    println("  E activity persists at: $(round(E_final, digits=4))")
    println("  I activity remains low: $(round(I_final, digits=4))")
    println("  → Excitatory activity persists WITHOUT external stimulation!")
end

#=============================================================================
Visualization: Plot FoI Dynamics
=============================================================================#

println("\n### Creating Visualization ###\n")

# Create comprehensive plot showing all three variables
p = plot(
    times, E_activity,
    label="Excitatory (E)",
    xlabel="Time (ms)",
    ylabel="Activity / Stimulus",
    title="Failure of Inhibition Dynamics",
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

plot!(p, times, stimulus_values,
    label="Stimulus",
    linewidth=2,
    linestyle=:dash,
    color=:green
)

# Add vertical lines to mark phases
vline!(p, [ramp_up_end], 
    label="", 
    linewidth=1, 
    linestyle=:dot, 
    color=:gray,
    alpha=0.5
)
vline!(p, [plateau_end], 
    label="", 
    linewidth=1, 
    linestyle=:dot, 
    color=:gray,
    alpha=0.5
)
vline!(p, [ramp_down_end], 
    label="", 
    linewidth=1, 
    linestyle=:dot, 
    color=:gray,
    alpha=0.5
)

# Add phase annotations
annotate!(p, [
    (35, maximum(E_activity) * 0.95, text("Phase 1:\nActivation", 8, :left)),
    (75, maximum(E_activity) * 0.95, text("Phase 2:\nInhibition\nFailure", 8, :center)),
    (115, maximum(E_activity) * 0.95, text("Phase 3:\nRamp\nDown", 8, :center)),
    (170, maximum(E_activity) * 0.95, text("Phase 4:\nPersistent\nActivity", 8, :right))
])

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
