#!/usr/bin/env julia

"""
Improved parameter search for sustained oscillations using baseline-subtracted sigmoids.

This version uses RectifiedZeroedSigmoidNonlinearity which better matches the 
Wilson & Cowan 1973 formulation where S(0) = 0.
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

# Create custom parameters with baseline-subtracted sigmoid
function create_custom_oscillatory_params_v2(;
    vₑ=0.5, θₑ=9.0,
    vᵢ=1.0, θᵢ=15.0,
    bₑₑ=2.0, bᵢₑ=1.5, bₑᵢ=1.5, bᵢᵢ=0.1,
    τ=10.0,
    use_baseline_subtracted=true)
    
    lattice = PointLattice()
    
    # Create nonlinearity - use RectifiedZeroedSigmoid for baseline subtraction
    if use_baseline_subtracted
        nonlinearity_e = RectifiedZeroedSigmoidNonlinearity(a=vₑ, θ=θₑ)
        nonlinearity_i = RectifiedZeroedSigmoidNonlinearity(a=vᵢ, θ=θᵢ)
    else
        nonlinearity_e = SigmoidNonlinearity(a=vₑ, θ=θₑ)
        nonlinearity_i = SigmoidNonlinearity(a=vᵢ, θ=θᵢ)
    end
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
        τ = (τ, τ),
        connectivity = connectivity,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    return params
end

# Main search function
function search_oscillatory_parameters_v2()
    println("="^70)
    println("Parameter Search v2: Baseline-Subtracted Sigmoids")
    println("="^70)
    
    # Baseline parameters from WCM 1973 oscillatory mode
    baseline = (vₑ=0.5, θₑ=9.0, vᵢ=1.0, θᵢ=15.0, 
                bₑₑ=2.0, bᵢₑ=1.5, bₑᵢ=1.5, bᵢᵢ=0.1, τ=10.0)
    
    println("\nBaseline parameters:")
    println("  Sigmoid: vₑ=$(baseline.vₑ), θₑ=$(baseline.θₑ)")
    println("           vᵢ=$(baseline.vᵢ), θᵢ=$(baseline.θᵢ)")
    println("  Connectivity: bₑₑ=$(baseline.bₑₑ), bᵢₑ=$(baseline.bᵢₑ)")
    println("                bₑᵢ=$(baseline.bₑᵢ), bᵢᵢ=$(baseline.bᵢᵢ)")
    println("  Time constant: τ=$(baseline.τ)")
    
    successful_params = []
    
    # Test with both sigmoid types
    println("\n### Testing Sigmoid Types ###")
    
    println("\n1. Standard sigmoid (no baseline subtraction):")
    params_std = create_custom_oscillatory_params_v2(;baseline..., use_baseline_subtracted=false)
    A₀ = reshape([0.1, 0.1], 1, 2)
    times, A_history = simulate_dynamics(params_std, A₀, (0.0, 400.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
    println("  Result: peaks=$num_peaks, amplitude=$(round(amplitude, digits=4))")
    
    println("\n2. Baseline-subtracted sigmoid (rectified zeroed):")
    params_sub = create_custom_oscillatory_params_v2(;baseline..., use_baseline_subtracted=true)
    times, A_history = simulate_dynamics(params_sub, A₀, (0.0, 400.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
    println("  Result: peaks=$num_peaks, amplitude=$(round(amplitude, digits=4))")
    
    # Search with baseline-subtracted sigmoid
    println("\n### Parameter Search with Baseline-Subtracted Sigmoid ###")
    
    # Strategy 1: Vary time constant (faster dynamics may support oscillations)
    println("\nVarying time constant (τ)...")
    for τ in [8.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]
        params = create_custom_oscillatory_params_v2(;baseline..., τ=τ)
        times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
            external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
        
        is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
        
        if is_osc
            println("  ✓ τ=$τ: SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
            push!(successful_params, (name="τ=$τ", τ=τ, peaks=num_peaks, amplitude=amplitude))
        else
            println("  ✗ τ=$τ: damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        end
    end
    
    # Strategy 2: Stronger connectivity with baseline-subtracted sigmoid
    println("\nVarying E-E connectivity with τ=5.0...")
    for bₑₑ in [3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
        params = create_custom_oscillatory_params_v2(;baseline..., bₑₑ=bₑₑ, τ=5.0)
        times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
            external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
        
        is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
        
        if is_osc
            println("  ✓ bₑₑ=$bₑₑ: SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
            push!(successful_params, (name="bₑₑ=$bₑₑ, τ=5.0", bₑₑ=bₑₑ, τ=5.0, peaks=num_peaks, amplitude=amplitude))
        else
            println("  ✗ bₑₑ=$bₑₑ: damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        end
    end
    
    # Strategy 3: Lower thresholds + faster time constant
    println("\nLowering thresholds with τ=5.0...")
    for (θₑ, θᵢ) in [(8.0, 14.0), (7.0, 13.0), (6.0, 12.0), (5.0, 10.0), (4.0, 8.0)]
        params = create_custom_oscillatory_params_v2(;baseline..., θₑ=θₑ, θᵢ=θᵢ, τ=5.0)
        times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
            external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
        
        is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
        
        if is_osc
            println("  ✓ θₑ=$θₑ, θᵢ=$θᵢ: SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
            push!(successful_params, (name="θₑ=$θₑ, θᵢ=$θᵢ, τ=5.0", θₑ=θₑ, θᵢ=θᵢ, τ=5.0, peaks=num_peaks, amplitude=amplitude))
        else
            println("  ✗ θₑ=$θₑ, θᵢ=$θᵢ: damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        end
    end
    
    # Strategy 4: Combined optimizations
    println("\n### Optimized Combinations ###")
    
    # Combo 1: Fast + strong E-E + weak I-I
    println("\nCombo 1: τ=3.0, bₑₑ=4.0, bᵢᵢ=0.0")
    params = create_custom_oscillatory_params_v2(;baseline..., τ=3.0, bₑₑ=4.0, bᵢᵢ=0.0)
    times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
    if is_osc
        println("  ✓ SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        push!(successful_params, (name="Combo: τ↓ bₑₑ↑ bᵢᵢ=0", τ=3.0, bₑₑ=4.0, bᵢᵢ=0.0, peaks=num_peaks, amplitude=amplitude))
    else
        println("  ✗ damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
    end
    
    # Combo 2: Fast + lower thresholds
    println("\nCombo 2: τ=3.0, θₑ=5.0, θᵢ=10.0")
    params = create_custom_oscillatory_params_v2(;baseline..., τ=3.0, θₑ=5.0, θᵢ=10.0)
    times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
    if is_osc
        println("  ✓ SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        push!(successful_params, (name="Combo: τ↓ θ↓", τ=3.0, θₑ=5.0, θᵢ=10.0, peaks=num_peaks, amplitude=amplitude))
    else
        println("  ✗ damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
    end
    
    # Combo 3: Comprehensive optimization
    println("\nCombo 3: τ=2.0, bₑₑ=5.0, bᵢₑ=3.0, θₑ=4.0, θᵢ=8.0")
    params = create_custom_oscillatory_params_v2(;baseline..., τ=2.0, bₑₑ=5.0, bᵢₑ=3.0, θₑ=4.0, θᵢ=8.0)
    times, A_history = simulate_dynamics(params, A₀, (0.0, 400.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=5.0, duration=10.0, strength=20.0))
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history)
    if is_osc
        println("  ✓ SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        push!(successful_params, (name="Combo: comprehensive", τ=2.0, bₑₑ=5.0, bᵢₑ=3.0, θₑ=4.0, θᵢ=8.0, peaks=num_peaks, amplitude=amplitude))
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
        println()
        println("Note: The Wilson-Cowan model with these parameter ranges may be")
        println("inherently damped. Further exploration needed:")
        println("  - Different stimulus protocols")
        println("  - Spatial models (spatial coupling can sustain oscillations)")
        println("  - Different initial conditions")
        println()
    end
    
    println("="^70)
    
    return successful_params
end

# Run the search
if abspath(PROGRAM_FILE) == @__FILE__
    search_oscillatory_parameters_v2()
end
