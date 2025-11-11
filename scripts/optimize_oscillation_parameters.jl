#!/usr/bin/env julia

"""
Script to optimize parameters for oscillations in Wilson-Cowan point model.

This script explores parameter space to find configurations that produce
sustained or more robust oscillations, as requested in the issue.

The goal is to find parameters that:
1. Produce oscillations (not just damped transients)
2. Have longer half-life (more sustained)
3. Work with sustained (non-oscillatory) stimulus
"""

using FailureOfInhibition2025
using JSON
using Dates

# Include the WCM 1973 parameter creation functions
include("../test/test_wcm1973_validation.jl")

"""
Evaluate oscillation quality for a given parameter set.
Returns a score based on:
- Number of oscillation cycles
- Decay rate (lower is better)
- Amplitude (should be measurable but not too large)
"""
function evaluate_oscillation_quality(params, tspan=(0.0, 300.0); 
                                      stimulus=nothing, 
                                      A₀=reshape([0.3, 0.2], 1, 2))
    # Solve model
    if stimulus !== nothing
        # Create new params with stimulus
        params_with_stim = WilsonCowanParameters{2}(
            α = params.α,
            β = params.β,
            τ = params.τ,
            connectivity = params.connectivity,
            nonlinearity = params.nonlinearity,
            stimulus = stimulus,
            lattice = params.lattice,
            pop_names = params.pop_names
        )
        sol = solve_model(A₀, tspan, params_with_stim, saveat=0.5)
    else
        sol = solve_model(A₀, tspan, params, saveat=0.5)
    end
    
    # Detect oscillations
    has_osc, peak_times, peak_values = detect_oscillations(sol, 1, min_peaks=3)
    
    if !has_osc
        return (score=0.0, has_osc=false, n_peaks=0, decay_rate=Inf, 
                amplitude=0.0, frequency=0.0, half_life=0.0)
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
    
    # Decay rate: lower is better. Convert to score where 0 decay = 1.0
    # Decay rate of 0.001 or less gets full points
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
    
    # Combined score
    score = peak_score * 0.3 + decay_score * 0.5 + amp_score * 0.2
    
    return (score=score, has_osc=true, n_peaks=n_peaks, 
            decay_rate=decay_rate, amplitude=amp, 
            frequency=freq, half_life=half_life, period=period)
end

"""
Create a sustained constant stimulus.
"""
function create_constant_stimulus(strength, start_time, end_time, lattice)
    # Create a simple constant stimulus that doesn't oscillate
    return ConstantStimulus(
        strength=strength,
        time_windows=[(start_time, end_time)],
        lattice=lattice
    )
end

println("="^70)
println("Parameter Optimization for Oscillations in Point Model")
println("="^70)

# Start with the base oscillatory mode parameters
println("\n### Baseline: WCM 1973 Oscillatory Mode ###\n")
params_baseline = create_point_model_wcm1973(:oscillatory)

# Test without stimulus
println("Testing baseline without external stimulus...")
result_baseline = evaluate_oscillation_quality(params_baseline)

println("  Score: $(round(result_baseline.score, digits=3))")
println("  Oscillations: $(result_baseline.has_osc)")
println("  Peaks: $(result_baseline.n_peaks)")
if result_baseline.decay_rate !== nothing
    println("  Decay rate: $(round(result_baseline.decay_rate, digits=6))")
    if result_baseline.half_life !== nothing
        println("  Half-life: $(round(result_baseline.half_life, digits=2)) msec")
    end
end
if result_baseline.amplitude !== nothing
    println("  Amplitude: $(round(result_baseline.amplitude, digits=4))")
end
if result_baseline.frequency !== nothing
    println("  Frequency: $(round(result_baseline.frequency, digits=6)) Hz")
    println("  Period: $(round(result_baseline.period, digits=2)) msec")
end

# Test with sustained stimulus
println("\nTesting baseline with sustained constant stimulus...")
lattice = PointLattice()
stim = create_constant_stimulus(5.0, 5.0, 100.0, lattice)
result_baseline_stim = evaluate_oscillation_quality(params_baseline, stimulus=stim)

println("  Score: $(round(result_baseline_stim.score, digits=3))")
println("  Oscillations: $(result_baseline_stim.has_osc)")
println("  Peaks: $(result_baseline_stim.n_peaks)")
if result_baseline_stim.decay_rate !== nothing
    println("  Decay rate: $(round(result_baseline_stim.decay_rate, digits=6))")
    if result_baseline_stim.half_life !== nothing
        println("  Half-life: $(round(result_baseline_stim.half_life, digits=2)) msec")
    end
end

println("\n### Parameter Exploration ###\n")

# Based on Wilson & Cowan 1973 and neural dynamics theory:
# - Stronger E→E connectivity (bₑₑ) promotes oscillations
# - Weaker I→I connectivity (bᵢᵢ) reduces damping
# - Time constant ratio τₑ/τᵢ affects oscillation frequency
# - Nonlinearity slope (v) affects excitability

global best_score = result_baseline.score
global best_params = nothing
global best_result = result_baseline
global best_config = "baseline"

# Exploration 1: Vary E→E and I→I connectivity
println("Exploring connectivity strengths...")
for bₑₑ in [1.8, 2.0, 2.2, 2.5]
    for bᵢᵢ in [0.05, 0.1, 0.15, 0.2]
        lattice_test = PointLattice()
        
        # Create connectivity
        conn_ee = ScalarConnectivity(bₑₑ)
        conn_ei = ScalarConnectivity(-1.5)
        conn_ie = ScalarConnectivity(1.5)
        conn_ii = ScalarConnectivity(-bᵢᵢ)
        
        connectivity = ConnectivityMatrix{2}([
            conn_ee conn_ei;
            conn_ie conn_ii
        ])
        
        # Use baseline nonlinearity
        nonlinearity_e = SigmoidNonlinearity(a=0.5, θ=9.0)
        nonlinearity_i = SigmoidNonlinearity(a=1.0, θ=15.0)
        nonlinearity = (nonlinearity_e, nonlinearity_i)
        
        params_test = WilsonCowanParameters{2}(
            α = (1.0, 1.0),
            β = (1.0, 1.0),
            τ = (10.0, 10.0),
            connectivity = connectivity,
            nonlinearity = nonlinearity,
            stimulus = nothing,
            lattice = lattice_test,
            pop_names = ("E", "I")
        )
        
        result = evaluate_oscillation_quality(params_test)
        
        if result.score > best_score
            global best_score = result.score
            global best_params = params_test
            global best_result = result
            global best_config = "bₑₑ=$bₑₑ, bᵢᵢ=$bᵢᵢ"
            println("  → New best: bₑₑ=$bₑₑ, bᵢᵢ=$bᵢᵢ, score=$(round(result.score, digits=3))")
        end
    end
end

# Exploration 2: Vary time constants
println("\nExploring time constant ratios...")
for τₑ in [8.0, 10.0, 12.0]
    for τᵢ in [6.0, 8.0, 10.0]
        lattice_test = PointLattice()
        
        # Use best connectivity from previous exploration or baseline
        conn_ee = ScalarConnectivity(2.2)
        conn_ei = ScalarConnectivity(-1.5)
        conn_ie = ScalarConnectivity(1.5)
        conn_ii = ScalarConnectivity(-0.08)
        
        connectivity = ConnectivityMatrix{2}([
            conn_ee conn_ei;
            conn_ie conn_ii
        ])
        
        nonlinearity_e = SigmoidNonlinearity(a=0.5, θ=9.0)
        nonlinearity_i = SigmoidNonlinearity(a=1.0, θ=15.0)
        nonlinearity = (nonlinearity_e, nonlinearity_i)
        
        params_test = WilsonCowanParameters{2}(
            α = (1.0, 1.0),
            β = (1.0, 1.0),
            τ = (τₑ, τᵢ),
            connectivity = connectivity,
            nonlinearity = nonlinearity,
            stimulus = nothing,
            lattice = lattice_test,
            pop_names = ("E", "I")
        )
        
        result = evaluate_oscillation_quality(params_test)
        
        if result.score > best_score
            global best_score = result.score
            global best_params = params_test
            global best_result = result
            global best_config = "τₑ=$τₑ, τᵢ=$τᵢ"
            println("  → New best: τₑ=$τₑ, τᵢ=$τᵢ, score=$(round(result.score, digits=3))")
        end
    end
end

# Exploration 3: Vary inhibitory threshold and slope
println("\nExploring inhibitory population nonlinearity...")
for θᵢ in [12.0, 13.0, 14.0, 15.0, 16.0]
    for vᵢ in [0.8, 1.0, 1.2]
        lattice_test = PointLattice()
        
        conn_ee = ScalarConnectivity(2.2)
        conn_ei = ScalarConnectivity(-1.5)
        conn_ie = ScalarConnectivity(1.5)
        conn_ii = ScalarConnectivity(-0.08)
        
        connectivity = ConnectivityMatrix{2}([
            conn_ee conn_ei;
            conn_ie conn_ii
        ])
        
        nonlinearity_e = SigmoidNonlinearity(a=0.5, θ=9.0)
        nonlinearity_i = SigmoidNonlinearity(a=vᵢ, θ=θᵢ)
        nonlinearity = (nonlinearity_e, nonlinearity_i)
        
        params_test = WilsonCowanParameters{2}(
            α = (1.0, 1.0),
            β = (1.0, 1.0),
            τ = (10.0, 10.0),
            connectivity = connectivity,
            nonlinearity = nonlinearity,
            stimulus = nothing,
            lattice = lattice_test,
            pop_names = ("E", "I")
        )
        
        result = evaluate_oscillation_quality(params_test)
        
        if result.score > best_score
            global best_score = result.score
            global best_params = params_test
            global best_result = result
            global best_config = "vᵢ=$vᵢ, θᵢ=$θᵢ"
            println("  → New best: vᵢ=$vᵢ, θᵢ=$θᵢ, score=$(round(result.score, digits=3))")
        end
    end
end

println("\n### Best Configuration Found ###\n")
println("Configuration: $best_config")
println("  Score: $(round(best_result.score, digits=3))")
println("  Oscillations: $(best_result.has_osc)")
println("  Peaks: $(best_result.n_peaks)")
if best_result.decay_rate !== nothing
    println("  Decay rate: $(round(best_result.decay_rate, digits=6))")
    if best_result.half_life !== nothing
        println("  Half-life: $(round(best_result.half_life, digits=2)) msec")
    end
end
if best_result.amplitude !== nothing
    println("  Amplitude: $(round(best_result.amplitude, digits=4))")
end
if best_result.frequency !== nothing
    println("  Frequency: $(round(best_result.frequency, digits=6)) Hz")
    println("  Period: $(round(best_result.period, digits=2)) msec")
end

if best_params !== nothing
    println("\nOptimized parameters:")
    println("  τ: $(best_params.τ)")
    println("  α: $(best_params.α)")
    println("  β: $(best_params.β)")
    
    # Extract connectivity values
    conn = best_params.connectivity
    println("  Connectivity:")
    println("    E → E: $(conn.matrix[1,1].weight)")
    println("    I → E: $(conn.matrix[1,2].weight)")
    println("    E → I: $(conn.matrix[2,1].weight)")
    println("    I → I: $(conn.matrix[2,2].weight)")
    
    println("  Nonlinearity E: a=$(best_params.nonlinearity[1].a), θ=$(best_params.nonlinearity[1].θ)")
    println("  Nonlinearity I: a=$(best_params.nonlinearity[2].a), θ=$(best_params.nonlinearity[2].θ)")
end

println("\n### Testing with Sustained Stimulus ###\n")

if best_params !== nothing
    # Test with different stimulus strengths
    for stim_strength in [3.0, 5.0, 8.0, 10.0]
        stim_test = create_constant_stimulus(stim_strength, 5.0, 150.0, PointLattice())
        result_stim = evaluate_oscillation_quality(best_params, stimulus=stim_test)
        
        println("Stimulus strength $stim_strength:")
        println("  Score: $(round(result_stim.score, digits=3))")
        println("  Peaks: $(result_stim.n_peaks)")
        if result_stim.decay_rate !== nothing && result_stim.half_life !== nothing
            println("  Half-life: $(round(result_stim.half_life, digits=2)) msec")
        end
    end
end

# Save optimized parameters to JSON file
if best_params !== nothing
    println("\n### Saving Optimized Parameters ###\n")
    
    # Extract parameters in a serializable format
    conn = best_params.connectivity
    optimized_data = Dict(
        "mode" => "oscillatory_optimized",
        "timestamp" => string(now()),
        "configuration" => best_config,
        "metrics" => Dict(
            "score" => best_result.score,
            "has_oscillations" => best_result.has_osc,
            "n_peaks" => best_result.n_peaks,
            "decay_rate" => best_result.decay_rate,
            "amplitude" => best_result.amplitude,
            "frequency" => best_result.frequency,
            "half_life" => best_result.half_life,
            "period" => best_result.period
        ),
        "parameters" => Dict(
            "tau_e" => best_params.τ[1],
            "tau_i" => best_params.τ[2],
            "alpha_e" => best_params.α[1],
            "alpha_i" => best_params.α[2],
            "beta_e" => best_params.β[1],
            "beta_i" => best_params.β[2],
            "connectivity" => Dict(
                "b_ee" => conn.matrix[1,1].weight,
                "b_ei" => conn.matrix[1,2].weight,
                "b_ie" => conn.matrix[2,1].weight,
                "b_ii" => conn.matrix[2,2].weight
            ),
            "nonlinearity" => Dict(
                "v_e" => best_params.nonlinearity[1].a,
                "theta_e" => best_params.nonlinearity[1].θ,
                "v_i" => best_params.nonlinearity[2].a,
                "theta_i" => best_params.nonlinearity[2].θ
            )
        )
    )
    
    # Determine the output file path
    output_path = joinpath(dirname(@__DIR__), "data", "optimized_parameters.json")
    
    # Save to JSON
    open(output_path, "w") do io
        JSON.print(io, optimized_data, 4)
    end
    
    println("  Saved optimized parameters to: $output_path")
    println("  Configuration: $best_config")
    println("  Score: $(round(best_result.score, digits=3))")
end

println("\n" * "="^70)
println("Optimization Complete")
println("="^70)
