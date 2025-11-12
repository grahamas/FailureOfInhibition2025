#!/usr/bin/env julia

"""
Script to optimize parameters for oscillations in Wilson-Cowan point model.

This script uses Optim.jl to find optimal parameters that produce
sustained or more robust oscillations, as requested in the issue.

The goal is to find parameters that:
1. Produce oscillations (not just damped transients)
2. Have longer half-life (more sustained)
3. Work with sustained (non-oscillatory) stimulus
"""

using FailureOfInhibition2025
using Optim
using LinearAlgebra

# Include the WCM 1973 parameter creation functions
include("../test/test_wcm1973_validation.jl")

"""
Create parameters from optimization vector.
Vector contains: [bₑₑ, bᵢᵢ, bₑᵢ, τᵢ]
τₑ is kept constant at 10.0
"""
function create_params_from_vector(x::Vector)
    bₑₑ, bᵢᵢ, bₑᵢ, τᵢ = x
    
    lattice = PointLattice()
    
    # Create connectivity
    conn_ee = ScalarConnectivity(bₑₑ)
    conn_ei = ScalarConnectivity(-bₑᵢ)  # Now optimized (E→I connectivity, inhibitory)
    conn_ie = ScalarConnectivity(1.5)    # Keep baseline
    conn_ii = ScalarConnectivity(-bᵢᵢ)
    
    connectivity = ConnectivityMatrix{2}([
        conn_ee conn_ei;
        conn_ie conn_ii
    ])
    
    # Use baseline nonlinearity
    nonlinearity_e = SigmoidNonlinearity(a=0.5, θ=9.0)
    nonlinearity_i = SigmoidNonlinearity(a=1.0, θ=15.0)
    nonlinearity = (nonlinearity_e, nonlinearity_i)
    
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.0),
        β = (1.0, 1.0),
        τ = (10.0, τᵢ),  # τₑ kept constant at 10.0
        connectivity = connectivity,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    return params
end

"""
Objective function for optimization (to be minimized).
Returns negative score so that maximizing score becomes minimizing objective.
"""
function objective(x::Vector; verbose=false)
    # Extract parameters - already bounded by ParticleSwarm
    bₑₑ, bᵢᵢ, bₑᵢ, τᵢ = x
    
    try
        params = create_params_from_vector(x)
        
        # Solve model
        A₀ = reshape([0.3, 0.2], 1, 2)
        tspan = (0.0, 300.0)
        sol = solve_model(A₀, tspan, params, saveat=0.5)
        
        # Detect oscillations
        has_osc, peak_times, peak_values = detect_oscillations(sol, 1, min_peaks=3)
        
        if !has_osc
            if verbose
                println("  x=$x → No oscillations (returning penalty)")
            end
            return 100.0  # Large penalty for no oscillations
        end
        
        # Compute metrics
        freq, period = compute_oscillation_frequency(sol, 1, method=:peaks)
        amp, _ = compute_oscillation_amplitude(sol, 1, method=:envelope)
        decay_rate, half_life, _ = compute_oscillation_decay(sol, 1, method=:exponential)
        
        # Score based on:
        # 1. Number of peaks (more is better)
        # 2. Low decay rate (more sustained)
        # 3. Reasonable amplitude (not too small, not saturating)
        n_peaks = length(peak_times)
        
        peak_score = min(n_peaks / 10.0, 1.0)  # Normalize to [0, 1]
        
        # Decay rate: lower is better
        if decay_rate === nothing
            decay_score = 1.0  # No decay detected = best
        else
            decay_score = exp(-decay_rate * 100.0)  # Exponential penalty
        end
        
        # Amplitude: prefer 0.01 to 0.3 range
        if amp === nothing
            amp_score = 0.0
        elseif amp < 0.005
            amp_score = 0.1  # Too small
        elseif amp > 0.4
            amp_score = 0.5  # Saturating
        else
            amp_score = 1.0
        end
        
        # Combined score (higher is better)
        score = peak_score * 0.3 + decay_score * 0.5 + amp_score * 0.2
        
        if verbose
            println("  x=$x → score=$score (peaks=$n_peaks, amp=$(round(amp, digits=4)))")
        end
        
        # Return negative score for minimization
        return -score
        
    catch e
        if verbose
            println("  x=$x → Error: $e")
        end
        return 100.0  # Large penalty for errors
    end
end

println("="^70)
println("Parameter Optimization for Oscillations in Point Model")
println("Using Optim.jl for optimization")
println("="^70)

# Start with the base oscillatory mode parameters
println("\n### Baseline: WCM 1973 Oscillatory Mode ###\n")
params_baseline = create_point_model_wcm1973(:oscillatory)

# Test baseline
println("Evaluating baseline...")
A₀ = reshape([0.3, 0.2], 1, 2)
tspan = (0.0, 300.0)
sol_baseline = solve_model(A₀, tspan, params_baseline, saveat=0.5)

has_osc, peak_times, _ = detect_oscillations(sol_baseline, 1)
freq, period = compute_oscillation_frequency(sol_baseline, 1, method=:peaks)
amp, _ = compute_oscillation_amplitude(sol_baseline, 1, method=:envelope)
decay_rate, half_life, _ = compute_oscillation_decay(sol_baseline, 1, method=:exponential)

println("Baseline results:")
println("  Oscillations: $has_osc")
println("  Peaks: $(length(peak_times))")
if amp !== nothing
    println("  Amplitude: $(round(amp, digits=4))")
end
if decay_rate !== nothing
    println("  Decay rate: $(round(decay_rate, digits=6))")
    if half_life !== nothing
        println("  Half-life: $(round(half_life, digits=2)) msec")
    end
end
if freq !== nothing
    println("  Frequency: $(round(freq, digits=6)) Hz")
    println("  Period: $(round(period, digits=2)) msec")
end

println("\n### Running Optimization ###\n")

# Initial guess based on baseline oscillatory mode
# [bₑₑ, bᵢᵢ, bₑᵢ, τᵢ]
# Note: τₑ is kept constant at 10.0
x0 = [2.0, 0.1, 1.5, 10.0]

println("Initial parameters:")
println("  bₑₑ = $(x0[1])")
println("  bᵢᵢ = $(x0[2])")
println("  bₑᵢ = $(x0[3])")
println("  τᵢ = $(x0[4])")
println("  τₑ = 10.0 (kept constant)")
println("  Initial objective value: $(objective(x0))")

# Set bounds for box-constrained optimization
lower = [1.5, 0.01, 0.5, 5.0]
upper = [3.0, 0.3, 3.0, 15.0]

println("\nParameter bounds:")
println("  bₑₑ: [$(lower[1]), $(upper[1])]")
println("  bᵢᵢ: [$(lower[2]), $(upper[2])]")
println("  bₑᵢ: [$(lower[3]), $(upper[3])]")
println("  τᵢ: [$(lower[4]), $(upper[4])]")
println("  τₑ: 10.0 (constant)")

println("\nRunning Particle Swarm optimization...")
println("(This may take a few minutes...)")

# Use Particle Swarm which is good for box-constrained non-smooth objectives
result = optimize(
    objective,
    lower,
    upper,
    x0,
    ParticleSwarm(
        lower=lower,
        upper=upper,
        n_particles=30
    ),
    Optim.Options(
        iterations = 100,
        show_trace = true,
        show_every = 10
    )
)

println("\n### Optimization Results ###\n")

x_opt = Optim.minimizer(result)
f_opt = Optim.minimum(result)

println("Converged: $(Optim.converged(result))")
println("Iterations: $(Optim.iterations(result))")
println("Final objective value: $(round(f_opt, digits=6))")
println("Final score: $(round(-f_opt, digits=6))")

println("\nOptimized parameters:")
println("  bₑₑ = $(round(x_opt[1], digits=3))")
println("  bᵢᵢ = $(round(x_opt[2], digits=3))")
println("  bₑᵢ = $(round(x_opt[3], digits=3))")
println("  τᵢ = $(round(x_opt[4], digits=3))")

# Evaluate optimized parameters
println("\n### Evaluating Optimized Parameters ###\n")
params_opt = create_params_from_vector(x_opt)
sol_opt = solve_model(A₀, tspan, params_opt, saveat=0.5)

has_osc_opt, peak_times_opt, _ = detect_oscillations(sol_opt, 1)
freq_opt, period_opt = compute_oscillation_frequency(sol_opt, 1, method=:peaks)
amp_opt, _ = compute_oscillation_amplitude(sol_opt, 1, method=:envelope)
decay_opt, half_life_opt, _ = compute_oscillation_decay(sol_opt, 1, method=:exponential)

println("Optimized results:")
println("  Oscillations: $has_osc_opt")
println("  Peaks: $(length(peak_times_opt))")
if amp_opt !== nothing
    println("  Amplitude: $(round(amp_opt, digits=4))")
    if amp !== nothing
        improvement = (amp_opt - amp) / amp * 100
        println("  Amplitude improvement: $(round(improvement, digits=1))%")
    end
end
if decay_opt !== nothing
    println("  Decay rate: $(round(decay_opt, digits=6))")
    if half_life_opt !== nothing
        println("  Half-life: $(round(half_life_opt, digits=2)) msec")
    end
end
if freq_opt !== nothing
    println("  Frequency: $(round(freq_opt, digits=6)) Hz")
    println("  Period: $(round(period_opt, digits=2)) msec")
end

# Test with sustained stimulus
println("\n### Testing with Sustained Stimulus ###\n")

lattice = PointLattice()
stim = ConstantStimulus(
    strength=5.0,
    time_windows=[(10.0, 150.0)],
    lattice=lattice
)

params_stim = WilsonCowanParameters{2}(
    α = params_opt.α,
    β = params_opt.β,
    τ = params_opt.τ,
    connectivity = params_opt.connectivity,
    nonlinearity = params_opt.nonlinearity,
    stimulus = stim,
    lattice = params_opt.lattice,
    pop_names = params_opt.pop_names
)

sol_stim = solve_model(A₀, (0.0, 300.0), params_stim, saveat=0.5)

has_osc_stim, peak_times_stim, _ = detect_oscillations(sol_stim, 1)
amp_stim, _ = compute_oscillation_amplitude(sol_stim, 1, method=:envelope)

println("Results with sustained stimulus (strength=5.0, t=10-150ms):")
println("  Oscillations: $has_osc_stim")
println("  Peaks: $(length(peak_times_stim))")
if amp_stim !== nothing
    println("  Amplitude: $(round(amp_stim, digits=4))")
end

println("\n" * "="^70)
println("Optimization Complete")
println("="^70)
println("\nSummary:")
println("  Method: Particle Swarm optimization (Optim.jl)")
println("  Optimized 4 parameters: bₑₑ, bᵢᵢ, bₑᵢ, τᵢ")
println("  Constant parameter: τₑ = 10.0")
println("  Objective: Maximize oscillation quality score")
println("    - 30% weight on number of peaks")
println("    - 50% weight on sustained oscillations (low decay)")
println("    - 20% weight on amplitude")
