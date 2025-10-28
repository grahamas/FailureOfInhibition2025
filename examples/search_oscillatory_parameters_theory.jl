#!/usr/bin/env julia

"""
Theoretically-guided parameter search for sustained oscillations.

This script uses theoretical principles from E-I network oscillations to guide
the parameter search:
1. Excitation must be strong enough to drive activity
2. Inhibition must be delayed relative to excitation (via different time constants)
3. The E-I feedback loop must be strong enough
4. The system must avoid runaway excitation (saturation via sigmoid)
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
        
        # Safety check for numerical stability
        if any(isnan.(A)) || any(isinf.(A))
            @warn "Numerical instability at time $t, aborting simulation"
            return times[1:i-1], A_history[1:i-1, :, :]
        end
        
        A_history[i, :, :] = A
    end
    
    return times, A_history
end

# Brief pulse stimulus
function brief_pulse(t; start_time=5.0, duration=10.0, strength=15.0)
    if start_time <= t < start_time + duration
        return strength
    else
        return 0.0
    end
end

# Count peaks in a signal
function count_peaks(signal, threshold=0.05)
    peaks = 0
    for i in 2:length(signal)-1
        if signal[i] > signal[i-1] && signal[i] > signal[i+1] && signal[i] > threshold
            peaks += 1
        end
    end
    return peaks
end

# Improved oscillation analysis
function analyze_oscillations(times, A_history; 
                              analysis_window=(150.0, 350.0),
                              min_amplitude=0.02,
                              min_peaks=3)
    """
    Analyze whether the dynamics show sustained oscillations.
    """
    if size(A_history, 1) < 10
        return (false, 0, 0.0, 0.0)
    end
    
    E_activity = [A_history[i, 1, 1] for i in 1:size(A_history, 1)]
    
    # Focus on late time window
    idx = findall(t -> analysis_window[1] <= t <= analysis_window[2], times)
    
    if length(idx) < 10
        return (false, 0, 0.0, 0.0)
    end
    
    E_window = E_activity[idx]
    
    # Check for NaN or Inf
    if any(isnan.(E_window)) || any(isinf.(E_window))
        return (false, 0, 0.0, 0.0)
    end
    
    # Count peaks
    num_peaks = count_peaks(E_window, min_amplitude)
    
    # Calculate amplitude
    amplitude = maximum(E_window) - minimum(E_window)
    
    # Calculate mean activity
    mean_activity = sum(E_window) / length(E_window)
    
    # Determine if oscillatory
    is_oscillatory = (num_peaks >= min_peaks) && (amplitude >= min_amplitude)
    
    return (is_oscillatory, num_peaks, amplitude, mean_activity)
end

# Create parameters with differential time constants for E and I
function create_ei_oscillatory_params(;
    vₑ=0.5, θₑ=9.0,
    vᵢ=1.0, θᵢ=15.0,
    bₑₑ=2.0, bᵢₑ=1.5, bₑᵢ=1.5, bᵢᵢ=0.1,
    τₑ=10.0, τᵢ=10.0)
    
    lattice = PointLattice()
    
    # Use standard sigmoid (no baseline subtraction for stability)
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
    
    # Create parameters with different time constants for E and I
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.0),
        β = (1.0, 1.0),
        τ = (τₑ, τᵢ),  # Different time constants!
        connectivity = connectivity,
        nonlinearity = nonlinearity,
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    return params
end

# Main search function
function search_oscillatory_parameters_theory()
    println("="^70)
    println("Theory-Guided Parameter Search for Sustained Oscillations")
    println("="^70)
    
    println("\nKey principle: Differential time constants")
    println("  τᵢ > τₑ: Slower inhibition creates phase lag needed for oscillations")
    println()
    
    successful_params = []
    A₀ = reshape([0.1, 0.1], 1, 2)
    
    # Baseline from WCM 1973
    baseline = (vₑ=0.5, θₑ=9.0, vᵢ=1.0, θᵢ=15.0, 
                bₑₑ=2.0, bᵢₑ=1.5, bₑᵢ=1.5, bᵢᵢ=0.1)
    
    # Strategy 1: Test differential time constants
    println("### Strategy 1: Differential Time Constants (τᵢ/τₑ ratio) ###\n")
    
    for (τₑ, τᵢ) in [(10.0, 15.0), (10.0, 20.0), (8.0, 16.0), (5.0, 15.0), (10.0, 25.0)]
        ratio = τᵢ/τₑ
        params = create_ei_oscillatory_params(;baseline..., τₑ=τₑ, τᵢ=τᵢ)
        times, A_history = simulate_dynamics(params, A₀, (0.0, 500.0), 0.5,
            external_input = t -> brief_pulse(t, start_time=10.0, duration=20.0, strength=15.0))
        
        is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history,
            analysis_window=(200.0, 450.0))
        
        if is_osc
            println("  ✓ τₑ=$τₑ, τᵢ=$τᵢ (ratio=$(round(ratio, digits=2))): SUSTAINED OSCILLATIONS")
            println("    peaks=$num_peaks, amplitude=$(round(amplitude, digits=4))")
            push!(successful_params, (name="τₑ=$τₑ, τᵢ=$τᵢ", τₑ=τₑ, τᵢ=τᵢ, 
                peaks=num_peaks, amplitude=amplitude))
        else
            println("  ✗ τₑ=$τₑ, τᵢ=$τᵢ (ratio=$(round(ratio, digits=2))): damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        end
    end
    
    # Strategy 2: Stronger feedback with differential time constants
    println("\n### Strategy 2: Stronger E-I Feedback Loop ###\n")
    
    for (bₑₑ, bᵢₑ, bₑᵢ) in [(3.0, 2.5, 2.5), (4.0, 3.0, 3.0), (5.0, 4.0, 4.0)]
        params = create_ei_oscillatory_params(;baseline..., τₑ=8.0, τᵢ=18.0, 
            bₑₑ=bₑₑ, bᵢₑ=bᵢₑ, bₑᵢ=bₑᵢ)
        times, A_history = simulate_dynamics(params, A₀, (0.0, 500.0), 0.5,
            external_input = t -> brief_pulse(t, start_time=10.0, duration=20.0, strength=15.0))
        
        is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history,
            analysis_window=(200.0, 450.0))
        
        if is_osc
            println("  ✓ bₑₑ=$bₑₑ, bᵢₑ=$bᵢₑ, bₑᵢ=$bₑᵢ: SUSTAINED OSCILLATIONS")
            println("    peaks=$num_peaks, amplitude=$(round(amplitude, digits=4))")
            push!(successful_params, (name="Strong feedback", bₑₑ=bₑₑ, bᵢₑ=bᵢₑ, bₑᵢ=bₑᵢ,
                τₑ=8.0, τᵢ=18.0, peaks=num_peaks, amplitude=amplitude))
        else
            println("  ✗ bₑₑ=$bₑₑ, bᵢₑ=$bᵢₑ, bₑᵢ=$bₑᵢ: damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        end
    end
    
    # Strategy 3: Lower thresholds + differential time constants
    println("\n### Strategy 3: Lower Thresholds for Higher Gain ###\n")
    
    for (θₑ, θᵢ) in [(7.0, 13.0), (6.0, 12.0), (5.0, 10.0)]
        params = create_ei_oscillatory_params(;baseline..., τₑ=8.0, τᵢ=18.0, 
            θₑ=θₑ, θᵢ=θᵢ)
        times, A_history = simulate_dynamics(params, A₀, (0.0, 500.0), 0.5,
            external_input = t -> brief_pulse(t, start_time=10.0, duration=20.0, strength=15.0))
        
        is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history,
            analysis_window=(200.0, 450.0))
        
        if is_osc
            println("  ✓ θₑ=$θₑ, θᵢ=$θᵢ: SUSTAINED OSCILLATIONS")
            println("    peaks=$num_peaks, amplitude=$(round(amplitude, digits=4))")
            push!(successful_params, (name="Lower thresholds", θₑ=θₑ, θᵢ=θᵢ, 
                τₑ=8.0, τᵢ=18.0, peaks=num_peaks, amplitude=amplitude))
        else
            println("  ✗ θₑ=$θₑ, θᵢ=$θᵢ: damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        end
    end
    
    # Strategy 4: Comprehensive optimization
    println("\n### Strategy 4: Optimized Combinations ###\n")
    
    # Test each combination manually
    println("Combo 1: Balanced (τₑ=8.0, τᵢ=20.0, bₑₑ=3.5, bᵢₑ=3.0, bₑᵢ=3.0, bᵢᵢ=0.05)")
    params = create_ei_oscillatory_params(;baseline..., τₑ=8.0, τᵢ=20.0, bₑₑ=3.5, bᵢₑ=3.0, bₑᵢ=3.0, bᵢᵢ=0.05)
    times, A_history = simulate_dynamics(params, A₀, (0.0, 500.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=10.0, duration=20.0, strength=15.0))
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history, analysis_window=(200.0, 450.0))
    if is_osc
        println("  ✓ SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        push!(successful_params, (name="Balanced", τₑ=8.0, τᵢ=20.0, bₑₑ=3.5, bᵢₑ=3.0, bₑᵢ=3.0, bᵢᵢ=0.05, 
            peaks=num_peaks, amplitude=amplitude))
    else
        println("  ✗ damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
    end
    
    println("\nCombo 2: Strong excitation (τₑ=7.0, τᵢ=18.0, bₑₑ=4.5, bᵢₑ=3.5, bₑᵢ=3.5, bᵢᵢ=0.0)")
    params = create_ei_oscillatory_params(;baseline..., τₑ=7.0, τᵢ=18.0, bₑₑ=4.5, bᵢₑ=3.5, bₑᵢ=3.5, bᵢᵢ=0.0)
    times, A_history = simulate_dynamics(params, A₀, (0.0, 500.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=10.0, duration=20.0, strength=15.0))
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history, analysis_window=(200.0, 450.0))
    if is_osc
        println("  ✓ SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        push!(successful_params, (name="Strong excitation", τₑ=7.0, τᵢ=18.0, bₑₑ=4.5, bᵢₑ=3.5, bₑᵢ=3.5, bᵢᵢ=0.0,
            peaks=num_peaks, amplitude=amplitude))
    else
        println("  ✗ damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
    end
    
    println("\nCombo 3: Large time difference (τₑ=5.0, τᵢ=20.0, bₑₑ=3.0, bᵢₑ=2.5, bₑᵢ=2.5, bᵢᵢ=0.1)")
    params = create_ei_oscillatory_params(;baseline..., τₑ=5.0, τᵢ=20.0, bₑₑ=3.0, bᵢₑ=2.5, bₑᵢ=2.5, bᵢᵢ=0.1)
    times, A_history = simulate_dynamics(params, A₀, (0.0, 500.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=10.0, duration=20.0, strength=15.0))
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history, analysis_window=(200.0, 450.0))
    if is_osc
        println("  ✓ SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        push!(successful_params, (name="Large time difference", τₑ=5.0, τᵢ=20.0, bₑₑ=3.0, bᵢₑ=2.5, bₑᵢ=2.5, bᵢᵢ=0.1,
            peaks=num_peaks, amplitude=amplitude))
    else
        println("  ✗ damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
    end
    
    println("\nCombo 4: Low threshold + slow I (τₑ=8.0, τᵢ=22.0, bₑₑ=3.0, bᵢₑ=2.5, bₑᵢ=2.5, θₑ=6.0, θᵢ=12.0)")
    params = create_ei_oscillatory_params(;baseline..., τₑ=8.0, τᵢ=22.0, bₑₑ=3.0, bᵢₑ=2.5, bₑᵢ=2.5, θₑ=6.0, θᵢ=12.0)
    times, A_history = simulate_dynamics(params, A₀, (0.0, 500.0), 0.5,
        external_input = t -> brief_pulse(t, start_time=10.0, duration=20.0, strength=15.0))
    is_osc, num_peaks, amplitude, mean_act = analyze_oscillations(times, A_history, analysis_window=(200.0, 450.0))
    if is_osc
        println("  ✓ SUSTAINED OSCILLATIONS (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
        push!(successful_params, (name="Low threshold + slow I", τₑ=8.0, τᵢ=22.0, bₑₑ=3.0, bᵢₑ=2.5, bₑᵢ=2.5, θₑ=6.0, θᵢ=12.0,
            peaks=num_peaks, amplitude=amplitude))
    else
        println("  ✗ damped (peaks=$num_peaks, amp=$(round(amplitude, digits=4)))")
    end
    
    # Summary
    println("\n" * "="^70)
    println("Search Summary")
    println("="^70)
    
    if length(successful_params) > 0
        println("\nSuccess! Found $(length(successful_params)) parameter set(s) with sustained oscillations:")
        println()
        for (i, p) in enumerate(successful_params)
            println("$i. $(p.name)")
            # Print main parameters
            if haskey(p, :τₑ) && haskey(p, :τᵢ)
                println("   Time constants: τₑ=$(p.τₑ), τᵢ=$(p.τᵢ)")
            end
            if haskey(p, :bₑₑ)
                println("   Connectivity: bₑₑ=$(p.bₑₑ), bᵢₑ=$(haskey(p, :bᵢₑ) ? p.bᵢₑ : baseline.bᵢₑ), bₑᵢ=$(haskey(p, :bₑᵢ) ? p.bₑᵢ : baseline.bₑᵢ)")
            end
            println("   Oscillations: $(p.peaks) peaks, amplitude=$(round(p.amplitude, digits=4))")
            println()
        end
        
        # Recommend best
        best_idx = argmax([p.amplitude for p in successful_params])
        best = successful_params[best_idx]
        println("RECOMMENDED: $(best.name)")
        println("  This parameter set produces the strongest sustained oscillations.")
        println()
    else
        println("\nNo sustained oscillations found with this parameter search.")
        println("\nNote: Sustained oscillations in Wilson-Cowan models typically require:")
        println("  1. Differential time constants (τᵢ > τₑ)")
        println("  2. Strong E-I feedback coupling")
        println("  3. Appropriate sigmoid gain (lower thresholds)")
        println("  4. OR spatial structure with wave propagation")
        println()
    end
    
    println("="^70)
    
    return successful_params
end

# Run the search
if abspath(PROGRAM_FILE) == @__FILE__
    search_oscillatory_parameters_theory()
end
