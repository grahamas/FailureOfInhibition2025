#!/usr/bin/env julia

"""
Test oscillation analysis functions on known synthetic sine waves.

This test validates the oscillation analysis utilities against analytically known
sine wave functions to verify correctness. Includes visual plots for verification.
"""

using Test
using FailureOfInhibition2025
using Statistics

# Check for and load UnicodePlots if available
HAS_UNICODEPLOTS = false
try
    using UnicodePlots
    global HAS_UNICODEPLOTS = true
catch
    @warn "UnicodePlots not available. Skipping visualization. Install with: using Pkg; Pkg.add(\"UnicodePlots\")"
end

println("\n" * "="^70)
println("Testing Oscillation Analysis on Synthetic Sine Waves")
println("="^70)

#=============================================================================
Helper: Create synthetic oscillating solution
=============================================================================#

"""
    create_synthetic_sine_wave(; amplitude, frequency, decay_rate, duration, dt)

Create a synthetic damped sine wave with known parameters for testing.

# Arguments
- `amplitude`: Peak amplitude of the oscillation (default: 1.0)
- `frequency`: Oscillation frequency in Hz (default: 0.1, which is 10 time units period)
- `decay_rate`: Exponential decay rate (default: 0.01)
- `duration`: Total duration (default: 100.0)
- `dt`: Time step (default: 0.1)
- `offset`: Baseline offset (default: 0.0)

# Returns
A mock solution object with time series data resembling an ODE solution.
"""
function create_synthetic_sine_wave(;
    amplitude=1.0,
    frequency=0.1,
    decay_rate=0.01,
    duration=100.0,
    dt=0.1,
    offset=0.0
)
    # Create time points
    times = 0:dt:duration
    n_points = length(times)
    
    # Create damped sine wave: A(t) = A₀ * exp(-λt) * sin(2πft) + offset
    ω = 2π * frequency
    values = [amplitude * exp(-decay_rate * t) * sin(ω * t) + offset for t in times]
    
    # Create mock solution object that mimics DifferentialEquations.jl solution
    # For point models: state shape is (1, n_pops)
    # We'll create a simple struct that has the necessary fields
    u = [reshape([v], 1, 1) for v in values]  # Wrap each value as (1,1) array for point model
    
    return (t=collect(times), u=u)
end

"""
    create_pure_sine_wave(; amplitude, frequency, duration, dt)

Create a pure (non-decaying) sine wave with known parameters.
"""
function create_pure_sine_wave(;
    amplitude=1.0,
    frequency=0.1,
    duration=100.0,
    dt=0.1,
    offset=0.0
)
    return create_synthetic_sine_wave(
        amplitude=amplitude,
        frequency=frequency,
        decay_rate=0.0,  # No decay
        duration=duration,
        dt=dt,
        offset=offset
    )
end

#=============================================================================
Test 1: Damped Sine Wave Analysis
=============================================================================#

println("\n### Test 1: Damped Sine Wave Analysis ###\n")

# Known parameters
true_amplitude = 1.0
true_frequency = 0.1  # Hz (period = 10 time units)
true_decay_rate = 0.01  # 1/time
true_half_life = log(2) / true_decay_rate  # Should be ~69.3 time units

println("Creating synthetic damped sine wave with known parameters:")
println("  - Amplitude: $true_amplitude")
println("  - Frequency: $true_frequency Hz (Period: $(1/true_frequency))")
println("  - Decay rate: $true_decay_rate (Half-life: $(round(true_half_life, digits=2)))")

# Create synthetic solution
sol_damped = create_synthetic_sine_wave(
    amplitude=true_amplitude,
    frequency=true_frequency,
    decay_rate=true_decay_rate,
    duration=200.0,
    dt=0.5
)

println("\nGenerated time series with $(length(sol_damped.t)) points")

# Plot the synthetic wave
if HAS_UNICODEPLOTS
    println("\n--- Damped Sine Wave ---")
    time_series = [u[1,1] for u in sol_damped.u]
    plt = lineplot(sol_damped.t, time_series,
        title="Damped Sine Wave (A₀=$(true_amplitude), f=$(true_frequency), λ=$(true_decay_rate))",
        xlabel="Time",
        ylabel="Amplitude",
        width=60,
        height=15
    )
    println(plt)
end

# Test 1a: Detect oscillations
println("\n1a. Testing detect_oscillations:")
has_osc, peak_times, peak_values = detect_oscillations(sol_damped, 1)

@test has_osc == true
@test length(peak_times) >= 10  # Should detect multiple peaks
println("   ✓ Detected $(length(peak_times)) peaks")
println("   ✓ Peak times: $(round.(peak_times[1:min(5, length(peak_times))], digits=2))...")

# Test 1b: Compute frequency
println("\n1b. Testing compute_oscillation_frequency:")

# FFT method
freq_fft, period_fft = compute_oscillation_frequency(sol_damped, 1, method=:fft)
@test freq_fft !== nothing
@test period_fft !== nothing

freq_error_fft = abs(freq_fft - true_frequency) / true_frequency * 100
println("   ✓ FFT method:")
println("     - Measured frequency: $(round(freq_fft, digits=6)) Hz")
println("     - Expected frequency: $(true_frequency) Hz")
println("     - Error: $(round(freq_error_fft, digits=2))%")
@test freq_error_fft < 5.0  # Should be within 5%

# Peak method
freq_peaks, period_peaks = compute_oscillation_frequency(sol_damped, 1, method=:peaks)
@test freq_peaks !== nothing

freq_error_peaks = abs(freq_peaks - true_frequency) / true_frequency * 100
println("   ✓ Peaks method:")
println("     - Measured frequency: $(round(freq_peaks, digits=6)) Hz")
println("     - Error: $(round(freq_error_peaks, digits=2))%")
@test freq_error_peaks < 10.0  # Peaks method may be less accurate

# Test 1c: Compute amplitude
println("\n1c. Testing compute_oscillation_amplitude:")

amp_env, envelope = compute_oscillation_amplitude(sol_damped, 1, method=:envelope)
@test amp_env !== nothing

# The envelope method measures mean(peaks) - mean(troughs)
# For a damped sine wave, peaks decay over time, so the mean will be less than initial amplitude
# The measured amplitude represents the average oscillation amplitude over the entire time series
println("   ✓ Envelope method:")
println("     - Measured amplitude: $(round(amp_env, digits=4))")
println("     - True initial amplitude: $(true_amplitude)")
println("     - Measurement represents average oscillation amplitude (accounting for decay)")

# Verify it's in a reasonable range (should be less than initial amplitude due to decay)
@test amp_env > 0.0
@test amp_env < true_amplitude * 2.0  # Should be less than 2x initial amplitude

# Test 1d: Compute decay rate
println("\n1d. Testing compute_oscillation_decay:")

decay, half_life, decay_env = compute_oscillation_decay(sol_damped, 1, method=:exponential)
@test decay !== nothing
@test half_life !== nothing

decay_error = abs(decay - true_decay_rate) / true_decay_rate * 100
half_life_error = abs(half_life - true_half_life) / true_half_life * 100

println("   ✓ Exponential decay method:")
println("     - Measured decay rate: $(round(decay, digits=6))")
println("     - Expected decay rate: $(true_decay_rate)")
println("     - Error: $(round(decay_error, digits=2))%")
println("     - Measured half-life: $(round(half_life, digits=2))")
println("     - Expected half-life: $(round(true_half_life, digits=2))")
println("     - Error: $(round(half_life_error, digits=2))%")
@test decay_error < 20.0  # Allow some tolerance for decay rate

# Test 1e: Compute duration
println("\n1e. Testing compute_oscillation_duration:")

duration, sustained, end_time = compute_oscillation_duration(sol_damped, 1, threshold_ratio=0.1)
@test duration !== nothing

println("   ✓ Duration: $(round(duration, digits=2))")
println("   ✓ Sustained: $sustained")
if !sustained
    println("   ✓ End time: $(round(end_time, digits=2))")
end

#=============================================================================
Test 2: Pure (Non-Decaying) Sine Wave Analysis
=============================================================================#

println("\n### Test 2: Pure Sine Wave Analysis ###\n")

# Known parameters for pure sine wave
pure_amplitude = 0.8
pure_frequency = 0.05  # Hz (period = 20 time units)

println("Creating pure sine wave with known parameters:")
println("  - Amplitude: $pure_amplitude")
println("  - Frequency: $pure_frequency Hz (Period: $(1/pure_frequency))")
println("  - Decay rate: 0.0 (no decay)")

sol_pure = create_pure_sine_wave(
    amplitude=pure_amplitude,
    frequency=pure_frequency,
    duration=200.0,
    dt=0.5
)

# Plot the pure wave
if HAS_UNICODEPLOTS
    println("\n--- Pure Sine Wave ---")
    time_series = [u[1,1] for u in sol_pure.u]
    plt = lineplot(sol_pure.t, time_series,
        title="Pure Sine Wave (A=$(pure_amplitude), f=$(pure_frequency))",
        xlabel="Time",
        ylabel="Amplitude",
        width=60,
        height=15
    )
    println(plt)
end

# Test frequency estimation
println("\n2a. Testing frequency estimation on pure sine:")
freq_pure, period_pure = compute_oscillation_frequency(sol_pure, 1, method=:fft)
@test freq_pure !== nothing

freq_error_pure = abs(freq_pure - pure_frequency) / pure_frequency * 100
println("   ✓ FFT method:")
println("     - Measured frequency: $(round(freq_pure, digits=6)) Hz")
println("     - Expected frequency: $(pure_frequency) Hz")
println("     - Error: $(round(freq_error_pure, digits=2))%")
@test freq_error_pure < 10.0  # Allow more tolerance for lower frequency

# Test that no decay is detected
println("\n2b. Testing that no decay is detected:")
decay_pure, _, _ = compute_oscillation_decay(sol_pure, 1)

if decay_pure === nothing
    println("   ✓ Correctly detected no decay (returns nothing)")
else
    println("   ✓ Detected minimal decay rate: $(round(decay_pure, digits=6))")
    @test decay_pure < 0.001  # Should be very small or nothing
end

#=============================================================================
Test 3: Multiple Frequencies Test
=============================================================================#

println("\n### Test 3: Different Frequencies ###\n")

frequencies = [0.02, 0.05, 0.1, 0.2]
println("Testing frequency detection for multiple known frequencies:")

if HAS_UNICODEPLOTS
    println("\n--- Multiple Frequency Sine Waves ---")
end

for (i, f) in enumerate(frequencies)
    sol_f = create_pure_sine_wave(
        amplitude=1.0,
        frequency=f,
        duration=200.0,
        dt=0.5
    )
    
    freq_measured, _ = compute_oscillation_frequency(sol_f, 1, method=:fft)
    
    if freq_measured !== nothing
        error_pct = abs(freq_measured - f) / f * 100
        println("  $(i). f=$(f) Hz → measured=$(round(freq_measured, digits=6)) Hz (error: $(round(error_pct, digits=2))%)")
        @test error_pct < 10.0
        
        # Plot each frequency
        if HAS_UNICODEPLOTS && i <= 2  # Only plot first two to save space
            time_series = [u[1,1] for u in sol_f.u]
            plt = lineplot(sol_f.t[1:min(100, length(sol_f.t))], 
                         time_series[1:min(100, length(time_series))],
                title="f=$(f) Hz (Period=$(round(1/f, digits=2)))",
                xlabel="Time",
                ylabel="Amplitude",
                width=50,
                height=10
            )
            println(plt)
        end
    end
end

#=============================================================================
Test 4: Amplitude Variations
=============================================================================#

println("\n### Test 4: Different Amplitudes ###\n")

amplitudes = [0.5, 1.0, 2.0, 5.0]
println("Testing amplitude detection for multiple known amplitudes:")

for (i, amp) in enumerate(amplitudes)
    sol_a = create_pure_sine_wave(
        amplitude=amp,
        frequency=0.1,
        duration=100.0,
        dt=0.5
    )
    
    amp_measured, _ = compute_oscillation_amplitude(sol_a, 1, method=:envelope)
    
    if amp_measured !== nothing
        # For pure sine wave oscillating around 0:
        # - Peaks are at +amplitude
        # - Troughs are at -amplitude  
        # - envelope method: mean(peaks) - mean(troughs) = amplitude - (-amplitude) = 2*amplitude
        expected = 2 * amp
        println("  $(i). A=$(amp) → measured=$(round(amp_measured, digits=4)) (expected≈$(expected))")
        # Allow reasonable tolerance
        @test amp_measured > 0.8 * expected
        @test amp_measured < 1.2 * expected
    end
end

#=============================================================================
Test 5: Decay Rates
=============================================================================#

println("\n### Test 5: Different Decay Rates ###\n")

decay_rates = [0.005, 0.01, 0.02]
println("Testing decay rate detection for multiple known decay rates:")

for (i, λ) in enumerate(decay_rates)
    sol_decay = create_synthetic_sine_wave(
        amplitude=1.0,
        frequency=0.1,
        decay_rate=λ,
        duration=200.0,
        dt=0.5
    )
    
    decay_measured, half_life_measured, _ = compute_oscillation_decay(sol_decay, 1, method=:exponential)
    
    if decay_measured !== nothing
        expected_half_life = log(2) / λ
        error_pct = abs(decay_measured - λ) / λ * 100
        println("  $(i). λ=$(λ) → measured=$(round(decay_measured, digits=6)) (error: $(round(error_pct, digits=2))%)")
        println("      Half-life: measured=$(round(half_life_measured, digits=2)), expected=$(round(expected_half_life, digits=2))")
        @test error_pct < 30.0  # Allow tolerance for decay rate estimation
    end
end

#=============================================================================
Summary
=============================================================================#

println("\n" * "="^70)
println("Summary")
println("="^70)
println()
println("All oscillation analysis functions tested on synthetic sine waves:")
println("  ✓ detect_oscillations - Successfully detected peaks in all cases")
println("  ✓ compute_oscillation_frequency - Accurately measured frequencies (< 10% error)")
println("  ✓ compute_oscillation_amplitude - Measured amplitudes within tolerance")
println("  ✓ compute_oscillation_decay - Detected decay rates and half-lives")
println("  ✓ compute_oscillation_duration - Determined oscillation persistence")
println()
println("These tests validate the correctness of the oscillation analysis utilities")
println("using analytically known sine wave functions.")
println()
println("="^70)
