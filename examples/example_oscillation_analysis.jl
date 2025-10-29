#!/usr/bin/env julia

"""
Example demonstrating oscillation analysis utilities for point models.

This example shows how to use the oscillation analysis functions to evaluate
oscillatory behavior in Wilson-Cowan point models, including:
- Frequency estimation
- Amplitude measurement
- Decay rate calculation
- Duration assessment

The example uses the WCM 1973 oscillatory mode as a demonstration.
"""

using FailureOfInhibition2025

# Load the WCM 1973 parameter creation functions
include("../test/test_wcm1973_validation.jl")

println("\n" * "="^70)
println("Oscillation Analysis Utilities for Point Models")
println("="^70)

#=============================================================================
Example 1: Basic Oscillation Detection
=============================================================================#

println("\n### Example 1: Basic Oscillation Detection ###\n")

# Create an oscillatory point model (WCM 1973 oscillatory mode)
params_osc = create_point_model_wcm1973(:oscillatory)

println("Model configuration:")
println("  - Mode: Oscillatory (WCM 1973)")
println("  - Populations: $(params_osc.pop_names)")
println("  - Time constants: τ = $(params_osc.τ) msec")

# Initial condition
A₀ = reshape([0.3, 0.2], 1, 2)
tspan = (0.0, 200.0)

# Solve the model
println("\nSolving model from t=$(tspan[1]) to t=$(tspan[2]) msec...")
sol = solve_model(A₀, tspan, params_osc, saveat=0.5)

# Detect oscillations
has_osc, peak_times, peak_values = detect_oscillations(sol, 1)

println("\nOscillation Detection Results:")
println("  - Oscillations detected: $(has_osc)")
println("  - Number of peaks: $(length(peak_times))")

if has_osc
    println("  - First peak at t = $(round(peak_times[1], digits=2)) msec")
    println("  - Last peak at t = $(round(peak_times[end], digits=2)) msec")
    println("  - Peak values range: [$(round(minimum(peak_values), digits=4)), $(round(maximum(peak_values), digits=4))]")
end

#=============================================================================
Example 2: Frequency Analysis
=============================================================================#

println("\n### Example 2: Frequency Analysis ###\n")

# Compute frequency using FFT method
freq_fft, period_fft = compute_oscillation_frequency(sol, 1, method=:fft)

println("FFT Method:")
if freq_fft !== nothing
    println("  - Dominant frequency: $(round(freq_fft, digits=6)) Hz (1/msec)")
    println("  - Period: $(round(period_fft, digits=2)) msec")
    println("  - Oscillation frequency: $(round(1000/period_fft, digits=2)) Hz (if time is in msec)")
else
    println("  - No dominant frequency detected")
end

# Compute frequency using peak detection method
freq_peaks, period_peaks = compute_oscillation_frequency(sol, 1, method=:peaks)

println("\nPeak Detection Method:")
if freq_peaks !== nothing
    println("  - Average frequency: $(round(freq_peaks, digits=6)) Hz (1/msec)")
    println("  - Average period: $(round(period_peaks, digits=2)) msec")
    
    # Compare methods
    if freq_fft !== nothing
        rel_diff = abs(freq_fft - freq_peaks) / freq_fft * 100
        println("  - Relative difference from FFT: $(round(rel_diff, digits=1))%")
    end
else
    println("  - No frequency detected (too few peaks)")
end

#=============================================================================
Example 3: Amplitude Analysis
=============================================================================#

println("\n### Example 3: Amplitude Analysis ###\n")

# Compute amplitude using envelope method
amp_env, envelope = compute_oscillation_amplitude(sol, 1, method=:envelope)

println("Envelope Method:")
if amp_env !== nothing
    println("  - Average peak-to-trough amplitude: $(round(amp_env, digits=4))")
    println("  - Initial envelope value: $(round(envelope[1], digits=4))")
    println("  - Final envelope value: $(round(envelope[end], digits=4))")
else
    println("  - Could not compute envelope amplitude")
end

# Compute amplitude using standard deviation
amp_std, _ = compute_oscillation_amplitude(sol, 1, method=:std)

println("\nStandard Deviation Method:")
println("  - Activity std: $(round(amp_std, digits=4))")

# Compute amplitude using peak mean
amp_peak, _ = compute_oscillation_amplitude(sol, 1, method=:peak_mean)

println("\nPeak Mean Method:")
if amp_peak !== nothing
    println("  - Mean peak amplitude: $(round(amp_peak, digits=4))")
else
    println("  - Could not compute peak mean amplitude")
end

#=============================================================================
Example 4: Decay Analysis
=============================================================================#

println("\n### Example 4: Decay Analysis ###\n")

# Compute decay rate
decay_rate, half_life, decay_envelope = compute_oscillation_decay(sol, 1, method=:exponential)

println("Exponential Decay Fit:")
if decay_rate !== nothing
    println("  - Decay rate λ: $(round(decay_rate, digits=6)) (1/msec)")
    if half_life !== nothing
        println("  - Half-life: $(round(half_life, digits=2)) msec")
        println("  - After 1 half-life, amplitude reduces to 50%")
        println("  - After 2 half-lives, amplitude reduces to 25%")
    end
else
    println("  - No significant decay detected (oscillations sustained)")
end

# Alternative: peak decay method
decay_peak, half_life_peak, peaks = compute_oscillation_decay(sol, 1, method=:peak_decay)

println("\nPeak Decay Method:")
if decay_peak !== nothing
    println("  - Decay rate from peaks: $(round(decay_peak, digits=6)) (1/msec)")
    if half_life_peak !== nothing
        println("  - Half-life from peaks: $(round(half_life_peak, digits=2)) msec")
    end
    println("  - Number of peaks analyzed: $(length(peaks))")
else
    println("  - No decay detected in peak amplitudes")
end

#=============================================================================
Example 5: Duration Analysis
=============================================================================#

println("\n### Example 5: Duration Analysis ###\n")

# Compute oscillation duration
duration, sustained, end_time = compute_oscillation_duration(sol, 1, threshold_ratio=0.1)

println("Duration Analysis (threshold = 10% of initial amplitude):")
if duration !== nothing
    println("  - Duration: $(round(duration, digits=2)) msec")
    println("  - Sustained throughout: $(sustained)")
    
    if !sustained && end_time !== nothing
        println("  - End time: $(round(end_time, digits=2)) msec")
        pct = duration / (tspan[2] - tspan[1]) * 100
        println("  - Percentage of simulation: $(round(pct, digits=1))%")
    end
else
    println("  - Could not determine duration (no oscillations detected)")
end

# Compare with different thresholds
duration_5, sustained_5, _ = compute_oscillation_duration(sol, 1, threshold_ratio=0.05)
duration_20, sustained_20, _ = compute_oscillation_duration(sol, 1, threshold_ratio=0.2)

println("\nSensitivity to threshold:")
if duration_5 !== nothing
    println("  - Duration (5% threshold): $(round(duration_5, digits=2)) msec")
end
if duration !== nothing
    println("  - Duration (10% threshold): $(round(duration, digits=2)) msec")
end
if duration_20 !== nothing
    println("  - Duration (20% threshold): $(round(duration_20, digits=2)) msec")
end

#=============================================================================
Example 6: Complete Analysis Summary
=============================================================================#

println("\n### Example 6: Complete Analysis Summary ###\n")

println("Analyzing both populations (E and I):")
println()

for pop_idx in 1:2
    pop_name = params_osc.pop_names[pop_idx]
    println("Population: $pop_name")
    
    # Detection
    has_osc, _, _ = detect_oscillations(sol, pop_idx)
    println("  ✓ Has oscillations: $has_osc")
    
    # Frequency
    freq, period = compute_oscillation_frequency(sol, pop_idx, method=:peaks)
    if freq !== nothing
        println("  ✓ Frequency: $(round(freq, digits=6)) Hz, Period: $(round(period, digits=2)) msec")
    end
    
    # Amplitude
    amp, _ = compute_oscillation_amplitude(sol, pop_idx, method=:envelope)
    if amp !== nothing
        println("  ✓ Amplitude: $(round(amp, digits=4))")
    end
    
    # Decay
    decay, half_life, _ = compute_oscillation_decay(sol, pop_idx, method=:exponential)
    if decay !== nothing
        if half_life !== nothing
            println("  ✓ Decay rate: $(round(decay, digits=6)) (1/msec), Half-life: $(round(half_life, digits=2)) msec")
        else
            println("  ✓ Decay rate: $(round(decay, digits=6)) (1/msec)")
        end
    else
        println("  ✓ No significant decay")
    end
    
    # Duration
    dur, sus, _ = compute_oscillation_duration(sol, pop_idx)
    if dur !== nothing
        println("  ✓ Duration: $(round(dur, digits=2)) msec, Sustained: $sus")
    end
    
    println()
end

#=============================================================================
Example 7: Comparison Across Modes
=============================================================================#

println("### Example 7: Comparison Across WCM 1973 Modes ###\n")

# Test all three modes
modes = [:active_transient, :oscillatory, :steady_state]
mode_names = ["Active Transient", "Oscillatory", "Steady-State"]

println("Comparing oscillatory behavior across three WCM 1973 modes:\n")

for (mode, mode_name) in zip(modes, mode_names)
    params = create_point_model_wcm1973(mode)
    sol_mode = solve_model(A₀, (0.0, 200.0), params, saveat=0.5)
    
    has_osc, peak_times, _ = detect_oscillations(sol_mode, 1)
    freq, period = compute_oscillation_frequency(sol_mode, 1, method=:peaks)
    
    println("$mode_name Mode:")
    println("  - Oscillations detected: $has_osc")
    
    if has_osc
        println("  - Number of peaks: $(length(peak_times))")
        if freq !== nothing
            println("  - Period: $(round(period, digits=2)) msec")
        end
    end
    println()
end

#=============================================================================
Summary
=============================================================================#

println("="^70)
println("Summary")
println("="^70)
println()
println("This example demonstrated how to:")
println("  ✓ Detect oscillations in point model time series")
println("  ✓ Estimate oscillation frequency using FFT and peak detection")
println("  ✓ Measure oscillation amplitude using multiple methods")
println("  ✓ Compute decay rates and half-lives for damped oscillations")
println("  ✓ Determine how long oscillations persist")
println("  ✓ Analyze multiple populations")
println("  ✓ Compare oscillatory behavior across different model configurations")
println()
println("These utilities are particularly useful for:")
println("  - Characterizing oscillatory dynamics in neural population models")
println("  - Parameter exploration and optimization")
println("  - Validating model behavior against experimental data")
println("  - Comparing different modeling approaches")
println()
println("="^70)
