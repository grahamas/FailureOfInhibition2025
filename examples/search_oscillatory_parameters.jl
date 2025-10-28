#!/usr/bin/env julia

"""
Parameter search script to find Wilson-Cowan parameters that produce sustained oscillations.

This script systematically explores the parameter space around the "oscillatory" mode
from Wilson & Cowan 1973 to find parameter combinations that produce sustained
(non-damped) oscillations.

The key parameters that affect oscillatory behavior are:
1. Connectivity strengths (bₑₑ, bᵢₑ, bₑᵢ, bᵢᵢ)
2. Sigmoid parameters (vₑ, vᵢ, θₑ, θᵢ)

Strategy:
- Start with the WCM 1973 oscillatory mode parameters
- Systematically vary parameters that control the balance between excitation and inhibition
- For each parameter set, simulate the dynamics and check for sustained oscillations
- Report parameter sets that produce sustained oscillations
"""

using FailureOfInhibition2025

# Load the WCM 1973 parameter creation functions
include("../test/test_wcm1973_validation.jl")

# Simple Euler integration for testing
function simulate_dynamics(params, A₀, tspan, dt=0.1; external_input=nothing)
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

# Brief pulse stimulus for initiating oscillations
function brief_pulse(t; start_time=5.0, duration=5.0, strength=20.0)
    if start_time <= t < start_time + duration
        return strength
    else
        return 0.0
    end
end

# Count peaks in a signal to detect oscillations
function count_peaks(signal, threshold=0.0)
    peaks = 0
    for i in 2:length(signal)-1
        if signal[i] > signal[i-1] && signal[i] > signal[i+1] && signal[i] > threshold
            peaks += 1
        end
    end
    return peaks
end

# Analyze oscillatory behavior
function analyze_oscillations(times, A_history; 
                              analysis_window=(100.0, 300.0),
                              min_amplitude=0.01,
                              min_peaks=3)
    """
    Analyze whether the dynamics show sustained oscillations.
    
    Returns: (is_oscillatory, num_peaks, amplitude, mean_activity)
    """
    E_activity = [A_history[i, 1, 1] for i in 1:length(times)]
    
    # Focus on late time window to check for sustained oscillations
    idx = findall(t -> analysis_window[1] <= t <= analysis_window[2], times)
    
    if length(idx) < 10
        return (false, 0, 0.0, 0.0)
    end
    
    E_window = E_activity[idx]
    
    # Count peaks
    num_peaks = count_peaks(E_window, 0.05)
    
    # Calculate amplitude
    amplitude = maximum(E_window) - minimum(E_window)
    
    # Calculate mean activity
    mean_activity = sum(E_window) / length(E_window)
    
    # Determine if oscillatory
    is_oscillatory = (num_peaks >= min_peaks) && (amplitude >= min_amplitude)
    
    return (is_oscillatory, num_peaks, amplitude, mean_activity)
end

# Create custom parameters for testing
function create_custom_oscillatory_params(;
    vₑ=0.5, θₑ=9.0,
    vᵢ=1.0, θᵢ=15.0,
    bₑₑ=2.0, bᵢₑ=1.5, bₑᵢ=1.5, bᵢᵢ=0.1)
    
    lattice = PointLattice()
    
    # Create nonlinearity
    nonlinearity_e = SigmoidNonlinearity(a=vₑ, θ=θₑ)
    nonlinearity_i = SigmoidNonlinearity(a=vᵢ, θ=θᵢ)
    nonlinearity = (nonlinearity_e, nonlinearity_i)
    
    # Create connectivity
    conn_ee = ScalarConnectivity(bₑₑ)
    conn_ei = ScalarConnectivity(-bᵢₑ)
    conn_ie = ScalarConnectivity(bₑᵢ)
    conn_ii = ScalarConnectivity(-bᵢᵢ)
    
    connectivity = ConnectivityMatrix{2}([
        conn_ee conn_ei;
        conn_ie conn_ii
    ])
    
    # Create parameters
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.0),
        β = (1.0, 1.0),
        τ = (10.0, 10.0),
        connectivity = connectivity,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    return params
end

# Main search function
function search_oscillatory_parameters()
    println("="^70)
    println("Searching for Parameters that Produce Sustained Oscillations")
    println("="^70)
    
    # Baseline parameters from WCM 1973 oscillatory mode
    baseline = (vₑ=0.5, θₑ=9.0, vᵢ=1.0, θᵢ=15.0, 
                bₑₑ=2.0, bᵢₑ=1.5, bₑᵢ=1.5, bᵢᵢ=0.1)
    
    println("\nBaseline parameters (WCM 1973 Oscillatory Mode):")
    println("  Sigmoid: vₑ=$(baseline.vₑ), θₑ=$(baseline.θₑ)")
    println("           vᵢ=$(baseline.vᵢ), θᵢ=$(baseline.θᵢ)")
    println("  Connectivity: bₑₑ=$(baseline.bₑₑ), bᵢₑ=$(baseline.bᵢₑ)")
    println("                bₑᵢ=$(baseline.bₑᵢ), bᵢᵢ=$(baseline.bᵢᵢ)")
    
    # Test baseline
    println("\nTesting baseline parameters...")
    params_baseline = create_custom_oscillatory_params(;baseline...)
    A₀ = reshape([0.05, 0.05], 1, 2)
    times, A_history = simulate_dynamics(params_baseline, A₀, (0.0, 400.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
    
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
    println("  Result: is_oscillatory=$is_osc, peaks=$num_peaks, amplitude=$(round(amplitude, digits=4))")
    
    # Parameter search
    successful_params = []
    
    println("\n" * "="^70)
    println("Parameter Search")
    println("="^70)
    
    # Strategy 1: Vary connectivity strengths
    println("\n### Strategy 1: Varying Connectivity Strengths ###")
    println("Searching around baseline connectivity values...")
    
    # Increase E-E coupling (stronger excitatory feedback)
    for bₑₑ in [2.2, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]
        params = create_custom_oscillatory_params(;baseline..., bₑₑ=bₑₑ)
        times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
            external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
        
        is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
        
        if is_osc
            println("  ✓ bₑₑ=$bₑₑ: SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
            push!(successful_params, (name="Increased E-E", bₑₑ=bₑₑ, peaks=num_peaks, amplitude=amplitude))
        else
            println("  ✗ bₑₑ=$bₑₑ: damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        end
    end
    
    # Increase I-E coupling (stronger inhibitory feedback to E)
    println("\nVarying I→E coupling (bᵢₑ)...")
    for bᵢₑ in [1.7, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0]
        params = create_custom_oscillatory_params(;baseline..., bᵢₑ=bᵢₑ)
        times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
            external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
        
        is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
        
        if is_osc
            println("  ✓ bᵢₑ=$bᵢₑ: SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
            push!(successful_params, (name="Increased I-E", bᵢₑ=bᵢₑ, peaks=num_peaks, amplitude=amplitude))
        else
            println("  ✗ bᵢₑ=$bᵢₑ: damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        end
    end
    
    # Decrease I-I coupling (weaker inhibitory self-inhibition)
    println("\nVarying I→I coupling (bᵢᵢ)...")
    for bᵢᵢ in [0.05, 0.02, 0.01, 0.0]
        params = create_custom_oscillatory_params(;baseline..., bᵢᵢ=bᵢᵢ)
        times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
            external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
        
        is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
        
        if is_osc
            println("  ✓ bᵢᵢ=$bᵢᵢ: SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
            push!(successful_params, (name="Decreased I-I", bᵢᵢ=bᵢᵢ, peaks=num_peaks, amplitude=amplitude))
        else
            println("  ✗ bᵢᵢ=$bᵢᵢ: damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        end
    end
    
    # Strategy 2: Vary sigmoid parameters
    println("\n### Strategy 2: Varying Sigmoid Parameters ###")
    
    # Increase inhibitory sigmoid steepness
    println("\nVarying inhibitory sigmoid steepness (vᵢ)...")
    for vᵢ in [1.2, 1.5, 2.0, 2.5]
        params = create_custom_oscillatory_params(;baseline..., vᵢ=vᵢ)
        times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
            external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
        
        is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
        
        if is_osc
            println("  ✓ vᵢ=$vᵢ: SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
            push!(successful_params, (name="Increased vᵢ", vᵢ=vᵢ, peaks=num_peaks, amplitude=amplitude))
        else
            println("  ✗ vᵢ=$vᵢ: damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        end
    end
    
    # Decrease inhibitory threshold
    println("\nVarying inhibitory threshold (θᵢ)...")
    for θᵢ in [14.0, 13.0, 12.0, 11.0, 10.0, 9.0, 8.0]
        params = create_custom_oscillatory_params(;baseline..., θᵢ=θᵢ)
        times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
            external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
        
        is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
        
        if is_osc
            println("  ✓ θᵢ=$θᵢ: SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
            push!(successful_params, (name="Decreased θᵢ", θᵢ=θᵢ, peaks=num_peaks, amplitude=amplitude))
        else
            println("  ✗ θᵢ=$θᵢ: damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        end
    end
    
    # Decrease excitatory threshold
    println("\nVarying excitatory threshold (θₑ)...")
    for θₑ in [8.0, 7.0, 6.0, 5.0, 4.0]
        params = create_custom_oscillatory_params(;baseline..., θₑ=θₑ)
        times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
            external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
        
        is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
        
        if is_osc
            println("  ✓ θₑ=$θₑ: SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
            push!(successful_params, (name="Decreased θₑ", θₑ=θₑ, peaks=num_peaks, amplitude=amplitude))
        else
            println("  ✗ θₑ=$θₑ: damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        end
    end
    
    # Strategy 3: Combined parameter adjustments
    println("\n### Strategy 3: Combined Parameter Adjustments ###")
    println("Testing promising combinations...")
    
    # Combination 1: Stronger E-E + weaker I-I
    println("\nCombination 1: bₑₑ=3.0, bᵢᵢ=0.05")
    params = create_custom_oscillatory_params(;baseline..., bₑₑ=3.0, bᵢᵢ=0.05)
    times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
    if is_osc
        println("  ✓ SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        push!(successful_params, (name="Combo: bₑₑ↑ bᵢᵢ↓", bₑₑ=3.0, bᵢᵢ=0.05, peaks=num_peaks, amplitude=amplitude))
    else
        println("  ✗ damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
    end
    
    # Combination 2: Stronger E-E + steeper inhibitory sigmoid
    println("\nCombination 2: bₑₑ=3.0, vᵢ=1.5")
    params = create_custom_oscillatory_params(;baseline..., bₑₑ=3.0, vᵢ=1.5)
    times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
    if is_osc
        println("  ✓ SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        push!(successful_params, (name="Combo: bₑₑ↑ vᵢ↑", bₑₑ=3.0, vᵢ=1.5, peaks=num_peaks, amplitude=amplitude))
    else
        println("  ✗ damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
    end
    
    # Combination 3: Lower inhibitory threshold + weaker I-I
    println("\nCombination 3: θᵢ=10.0, bᵢᵢ=0.02")
    params = create_custom_oscillatory_params(;baseline..., θᵢ=10.0, bᵢᵢ=0.02)
    times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
    if is_osc
        println("  ✓ SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        push!(successful_params, (name="Combo: θᵢ↓ bᵢᵢ↓", θᵢ=10.0, bᵢᵢ=0.02, peaks=num_peaks, amplitude=amplitude))
    else
        println("  ✗ damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
    end
    
    # Combination 4: Much stronger E-E + stronger I-E
    println("\nCombination 4: bₑₑ=5.0, bᵢₑ=3.0")
    params = create_custom_oscillatory_params(;baseline..., bₑₑ=5.0, bᵢₑ=3.0)
    times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
    if is_osc
        println("  ✓ SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        push!(successful_params, (name="Combo: bₑₑ↑↑ bᵢₑ↑", bₑₑ=5.0, bᵢₑ=3.0, peaks=num_peaks, amplitude=amplitude))
    else
        println("  ✗ damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
    end
    
    # Combination 5: Lower both thresholds
    println("\nCombination 5: θₑ=6.0, θᵢ=10.0")
    params = create_custom_oscillatory_params(;baseline..., θₑ=6.0, θᵢ=10.0)
    times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
    if is_osc
        println("  ✓ SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        push!(successful_params, (name="Combo: θₑ↓ θᵢ↓", θₑ=6.0, θᵢ=10.0, peaks=num_peaks, amplitude=amplitude))
    else
        println("  ✗ damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
    end
    
    # Combination 6: All-around boost to activity
    println("\nCombination 6: bₑₑ=4.0, bₑᵢ=2.5, θₑ=5.0, θᵢ=10.0")
    params = create_custom_oscillatory_params(;baseline..., bₑₑ=4.0, bₑᵢ=2.5, θₑ=5.0, θᵢ=10.0)
    times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
    if is_osc
        println("  ✓ SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        push!(successful_params, (name="Combo: boost all", bₑₑ=4.0, bₑᵢ=2.5, θₑ=5.0, θᵢ=10.0, peaks=num_peaks, amplitude=amplitude))
    else
        println("  ✗ damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
    end
    
    # Summary
    println("\n" * "="^70)
    println("Search Summary")
    println("="^70)
    
    if length(successful_params) > 0
        println("\nFound $(length(successful_params)) parameter set(s) with sustained oscillations:")
        println()
        for (i, p) in enumerate(successful_params)
            println("$i. $(p.name)")
            println("   Parameters: ", filter(kv -> kv[1] != :name && kv[1] != :peaks && kv[1] != :amplitude, pairs(p)))
            println("   Peaks: $(p.peaks), Amplitude: $(round(p.amplitude, digits=4))")
            println()
        end
        
        # Recommend best parameter set
        best_idx = argmax([p.amplitude for p in successful_params])
        best = successful_params[best_idx]
        println("RECOMMENDED: $(best.name)")
        println("  This parameter set produces the strongest sustained oscillations.")
        println()
    else
        println("\nNo parameter sets found with sustained oscillations.")
        println("Consider:")
        println("  - Exploring wider parameter ranges")
        println("  - Adjusting initial conditions")
        println("  - Modifying stimulus parameters")
        println()
    end
    
    println("="^70)
    
    return successful_params
end

# Run the search
if abspath(PROGRAM_FILE) == @__FILE__
    search_oscillatory_parameters()
end
