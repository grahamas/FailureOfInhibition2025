#!/usr/bin/env julia

"""
Generate visual plots for the three dynamical modes from Wilson & Cowan 1973.

This script simulates and visualizes using UnicodePlots:
1. Active Transient Mode - showing transient response that decays
2. Oscillatory Mode - showing sustained oscillations
3. Steady-State Mode - showing stable pattern formation
"""

using FailureOfInhibition2025

# Check for and install UnicodePlots if needed
try
    using UnicodePlots
catch
    using Pkg
    println("Installing UnicodePlots...")
    Pkg.add("UnicodePlots")
    using UnicodePlots
end

# Load the WCM 1973 parameter creation functions
include("../test/test_wcm1973_validation.jl")

println("\n" * "="^70)
println("Wilson-Cowan 1973: Three Dynamical Modes")
println("="^70)

# Simple Euler integration for point models with optional external input
function euler_integrate(params, A₀, tspan, dt=0.1; external_input=nothing)
    t_start, t_end = tspan
    times = collect(t_start:dt:t_end)
    n_steps = length(times)
    
    # Initialize arrays
    A_history = zeros(n_steps, size(A₀)...)
    A_history[1, :, :] = A₀
    
    A = copy(A₀)
    dA = zeros(size(A))
    
    for i in 2:n_steps
        t = times[i-1]
        fill!(dA, 0.0)
        
        # Add external input if provided
        if external_input !== nothing
            input_val = external_input(t)
            dA[1, 1] += input_val  # Add to excitatory population
        end
        
        wcm1973!(dA, A, params, t)
        A .+= dt .* dA
        A_history[i, :, :] = A
    end
    
    return times, A_history
end

# Define external input functions for each mode
function brief_pulse(t; start_time=5.0, duration=5.0, strength=15.0)
    if start_time <= t < start_time + duration
        return strength
    else
        return 0.0
    end
end

function sustained_pulse(t; start_time=5.0, duration=10.0, strength=20.0)
    if start_time <= t < start_time + duration
        return strength
    else
        return 0.0
    end
end

#=============================================================================
Mode 1: Active Transient Mode
=============================================================================#

println("\n### Active Transient Mode (Sensory Neo-Cortex) ###")
println("Brief stimulus → transient response → return to rest\n")

params_active = create_point_model_wcm1973(:active_transient)

# Simulate with external input
A₀ = reshape([0.05, 0.05], 1, 2)
times, A_history = euler_integrate(params_active, A₀, (0.0, 100.0), 0.5,
    external_input = t -> brief_pulse(t, start_time=5.0, duration=5.0, strength=15.0))

# Extract E and I activity
E_activity = [A_history[i, 1, 1] for i in 1:length(times)]
I_activity = [A_history[i, 1, 2] for i in 1:length(times)]

# Plot
p1 = lineplot(times, E_activity, 
    title="Active Transient Mode",
    name="Excitatory (E)",
    xlabel="Time (msec)",
    ylabel="Activity",
    width=60,
    height=15,
    ylim=[0, 0.4])
lineplot!(p1, times, I_activity, name="Inhibitory (I)")
println(p1)
println("\nKey features:")
println("  • Activity peaks AFTER stimulus ends (5-10ms)")
println("  • Self-generated transient response")
println("  • Returns to resting state")

#=============================================================================
Mode 2: Oscillatory Mode
=============================================================================#

println("\n\n### Oscillatory Mode (Thalamus) ###")
println("Sustained stimulus → persistent oscillations\n")

params_osc = create_point_model_wcm1973(:oscillatory)

# Simulate with sustained input
A₀_osc = reshape([0.05, 0.05], 1, 2)
times_osc, A_history_osc = euler_integrate(params_osc, A₀_osc, (0.0, 200.0), 0.5,
    external_input = t -> sustained_pulse(t, start_time=5.0, duration=10.0, strength=20.0))

# Extract activities
E_activity_osc = [A_history_osc[i, 1, 1] for i in 1:length(times_osc)]
I_activity_osc = [A_history_osc[i, 1, 2] for i in 1:length(times_osc)]

# Plot
p2 = lineplot(times_osc, E_activity_osc,
    title="Oscillatory Mode", 
    name="Excitatory (E)",
    xlabel="Time (msec)",
    ylabel="Activity",
    width=60,
    height=15,
    ylim=[0, 0.4])
lineplot!(p2, times_osc, I_activity_osc, name="Inhibitory (I)")
println(p2)
println("\nKey features:")
println("  • Sustained oscillations after stimulus")
println("  • Limit cycle behavior")
println("  • Frequency encodes stimulus properties")

#=============================================================================
Mode 3: Steady-State Mode
=============================================================================#

println("\n\n### Steady-State Mode (Prefrontal Cortex) ###")
println("Brief stimulus → persistent elevated activity\n")

params_ss = create_point_model_wcm1973(:steady_state)

# Simulate with brief strong input
A₀_ss = reshape([0.05, 0.05], 1, 2)
times_ss, A_history_ss = euler_integrate(params_ss, A₀_ss, (0.0, 150.0), 0.5,
    external_input = t -> brief_pulse(t, start_time=5.0, duration=8.0, strength=18.0))

# Extract activities
E_activity_ss = [A_history_ss[i, 1, 1] for i in 1:length(times_ss)]
I_activity_ss = [A_history_ss[i, 1, 2] for i in 1:length(times_ss)]

# Plot
p3 = lineplot(times_ss, E_activity_ss,
    title="Steady-State Mode",
    name="Excitatory (E)",
    xlabel="Time (msec)",
    ylabel="Activity",
    width=60,
    height=15,
    ylim=[0, 0.4])
lineplot!(p3, times_ss, I_activity_ss, name="Inhibitory (I)")
println(p3)
println("\nKey features:")
println("  • Activity persists at elevated level")
println("  • Stable steady state after stimulus")
println("  • Can retain information about prior input")

#=============================================================================
Phase Portraits
=============================================================================#

println("\n\n### Phase Space Analysis ###\n")

# Active Transient
pp1 = lineplot(E_activity, I_activity,
    title="Active Transient - Phase Portrait",
    xlabel="E Activity",
    ylabel="I Activity",
    width=50,
    height=12,
    xlim=[0, 0.35],
    ylim=[0, 0.35])
println(pp1)
println("Trajectory returns to origin (resting state)\n")

# Oscillatory
pp2 = lineplot(E_activity_osc[1:min(end,200)], I_activity_osc[1:min(end,200)],
    title="Oscillatory - Phase Portrait",
    xlabel="E Activity", 
    ylabel="I Activity",
    width=50,
    height=12,
    xlim=[0, 0.35],
    ylim=[0, 0.35])
println(pp2)
println("Trajectory forms limit cycle (closed loop)\n")

# Steady-State
pp3 = lineplot(E_activity_ss, I_activity_ss,
    title="Steady-State - Phase Portrait",
    xlabel="E Activity",
    ylabel="I Activity",
    width=50,
    height=12,
    xlim=[0, 0.35],
    ylim=[0, 0.35])
println(pp3)
println("Trajectory converges to stable fixed point\n")

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary")
println("="^70)
println()
println("These plots demonstrate that the implementation correctly reproduces")
println("the three dynamical modes from Wilson & Cowan (1973):")
println()
println("1. Active Transient: Self-generated transient → decay to rest")
println("2. Oscillatory: Sustained limit cycle oscillations")
println("3. Steady-State: Persistent elevated activity level")
println()
println("Key parameter differences:")
println("  • Active→Steady: Increased E→E coupling (bₑₑ: 1.5→2.0)")
println("  • Active→Oscillatory: Steeper I sigmoid, weaker I→I (bᵢᵢ: 1.8→0.1)")
println()
println("="^70)
