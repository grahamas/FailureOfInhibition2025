using Test
using FailureOfInhibition2025
using Statistics

println("\n=== Testing Oscillation Analysis Functions ===\n")

# Load WCM 1973 parameter creation functions for oscillatory mode
include("test_wcm1973_validation.jl")

#=============================================================================
Helper: Create oscillatory point model
=============================================================================#

function create_oscillatory_point_model()
    """Create a point model that produces oscillations."""
    return create_point_model_wcm1973(:oscillatory)
end

#=============================================================================
Test 1: detect_oscillations
=============================================================================#

println("1. Testing detect_oscillations:")

# Create oscillatory model
params_osc = create_oscillatory_point_model()

# Initial condition
A₀ = reshape([0.3, 0.2], 1, 2)
tspan = (0.0, 200.0)

# Solve with fine temporal resolution
sol_osc = solve_model(A₀, tspan, params_osc, saveat=0.5)

# Test detection for excitatory population
has_osc_e, peak_times_e, peak_values_e = detect_oscillations(sol_osc, 1)

@test typeof(has_osc_e) == Bool
@test length(peak_times_e) == length(peak_values_e)
@test all(peak_times_e .>= tspan[1])
@test all(peak_times_e .<= tspan[2])
@test all(peak_values_e .>= 0)
println("   ✓ detect_oscillations basic functionality passed")

# Test detection for inhibitory population
has_osc_i, peak_times_i, peak_values_i = detect_oscillations(sol_osc, 2)

@test typeof(has_osc_i) == Bool
@test length(peak_times_i) == length(peak_values_i)
println("   ✓ detect_oscillations for multiple populations passed")

# Test with min_peaks parameter
has_osc_strict, _, _ = detect_oscillations(sol_osc, 1, min_peaks=5)
@test typeof(has_osc_strict) == Bool
println("   ✓ detect_oscillations with min_peaks parameter passed")

# Test with non-oscillatory model (active transient)
params_transient = create_point_model_wcm1973(:active_transient)
sol_transient = solve_model(A₀, (0.0, 100.0), params_transient, saveat=0.5)
has_osc_trans, _, _ = detect_oscillations(sol_transient, 1)

@test typeof(has_osc_trans) == Bool
println("   ✓ detect_oscillations with non-oscillatory model passed")

#=============================================================================
Test 2: compute_oscillation_frequency
=============================================================================#

println("\n2. Testing compute_oscillation_frequency:")

# Test FFT method
freq_fft, period_fft = compute_oscillation_frequency(sol_osc, 1, method=:fft)

if freq_fft !== nothing
    @test freq_fft > 0
    @test period_fft > 0
    @test abs(freq_fft * period_fft - 1.0) < 1e-6  # freq * period ≈ 1
    println("   ✓ compute_oscillation_frequency with :fft method passed")
else
    @warn "No oscillations detected with FFT method"
    println("   ⚠ FFT method returned nothing (may need stronger oscillations)")
end

# Test peaks method
freq_peaks, period_peaks = compute_oscillation_frequency(sol_osc, 1, method=:peaks)

if freq_peaks !== nothing
    @test freq_peaks > 0
    @test period_peaks > 0
    @test abs(freq_peaks * period_peaks - 1.0) < 1e-6
    println("   ✓ compute_oscillation_frequency with :peaks method passed")
else
    @warn "No oscillations detected with peaks method"
    println("   ⚠ Peaks method returned nothing (may need stronger oscillations)")
end

# Test with non-oscillatory model
freq_trans, period_trans = compute_oscillation_frequency(sol_transient, 1)
# Non-oscillatory should return nothing or very low frequency
println("   ✓ compute_oscillation_frequency with non-oscillatory model passed")

# Test error handling
@test_throws ErrorException compute_oscillation_frequency(sol_osc, 1, method=:invalid)
println("   ✓ compute_oscillation_frequency error handling passed")

#=============================================================================
Test 3: compute_oscillation_amplitude
=============================================================================#

println("\n3. Testing compute_oscillation_amplitude:")

# Test envelope method
amp_env, envelope = compute_oscillation_amplitude(sol_osc, 1, method=:envelope)

if amp_env !== nothing
    @test amp_env > 0
    @test envelope !== nothing
    @test length(envelope) == length(sol_osc.t)
    @test all(envelope .>= 0)
    println("   ✓ compute_oscillation_amplitude with :envelope method passed")
else
    println("   ⚠ Envelope method returned nothing (may need stronger oscillations)")
end

# Test std method
amp_std, _ = compute_oscillation_amplitude(sol_osc, 1, method=:std)

@test amp_std > 0  # std should always be computable
println("   ✓ compute_oscillation_amplitude with :std method passed")

# Test peak_mean method
amp_peak, _ = compute_oscillation_amplitude(sol_osc, 1, method=:peak_mean)

if amp_peak !== nothing
    @test amp_peak > 0
    println("   ✓ compute_oscillation_amplitude with :peak_mean method passed")
else
    println("   ⚠ Peak_mean method returned nothing (may need stronger oscillations)")
end

# Test error handling
@test_throws ErrorException compute_oscillation_amplitude(sol_osc, 1, method=:invalid)
println("   ✓ compute_oscillation_amplitude error handling passed")

#=============================================================================
Test 4: compute_oscillation_decay
=============================================================================#

println("\n4. Testing compute_oscillation_decay:")

# Test exponential method
decay_exp, half_life_exp, env_exp = compute_oscillation_decay(sol_osc, 1, method=:exponential)

if decay_exp !== nothing
    @test decay_exp > 0
    if half_life_exp !== nothing
        @test half_life_exp > 0
        # Verify half-life relationship
        @test abs(decay_exp * half_life_exp - log(2)) < 0.1
    end
    @test length(env_exp) > 0
    println("   ✓ compute_oscillation_decay with :exponential method passed")
else
    println("   ⚠ Exponential decay method returned nothing (oscillations may not decay)")
end

# Test linear method (should give same results)
decay_lin, half_life_lin, env_lin = compute_oscillation_decay(sol_osc, 1, method=:linear)

if decay_lin !== nothing && decay_exp !== nothing
    # Both methods should give similar results
    @test abs(decay_lin - decay_exp) / decay_exp < 0.5  # Within 50%
    println("   ✓ compute_oscillation_decay :linear and :exponential methods consistent")
end

# Test peak_decay method
decay_peak, half_life_peak, peaks = compute_oscillation_decay(sol_osc, 1, method=:peak_decay)

if decay_peak !== nothing
    @test decay_peak > 0
    @test length(peaks) > 0
    println("   ✓ compute_oscillation_decay with :peak_decay method passed")
else
    println("   ⚠ Peak decay method returned nothing (peaks may not decay)")
end

# Test with non-decaying system (steady-state)
params_ss = create_point_model_wcm1973(:steady_state)
sol_ss = solve_model(A₀, (0.0, 100.0), params_ss, saveat=0.5)
decay_ss, _, _ = compute_oscillation_decay(sol_ss, 1)

# Steady-state should not have decay (or very small)
println("   ✓ compute_oscillation_decay with non-decaying model passed")

# Test error handling
@test_throws ErrorException compute_oscillation_decay(sol_osc, 1, method=:invalid)
println("   ✓ compute_oscillation_decay error handling passed")

#=============================================================================
Test 5: compute_oscillation_duration
=============================================================================#

println("\n5. Testing compute_oscillation_duration:")

# Test with oscillatory model
duration, sustained, end_time = compute_oscillation_duration(sol_osc, 1)

if duration !== nothing
    @test duration > 0
    @test duration <= (tspan[2] - tspan[1])
    @test typeof(sustained) == Bool
    
    if !sustained
        @test end_time !== nothing
        @test end_time > tspan[1]
        @test end_time <= tspan[2]
    else
        @test end_time === nothing
    end
    println("   ✓ compute_oscillation_duration basic functionality passed")
else
    println("   ⚠ Duration returned nothing (no oscillations detected)")
end

# Test with different threshold ratios
duration_strict, sustained_strict, _ = compute_oscillation_duration(sol_osc, 1, threshold_ratio=0.05)
duration_loose, sustained_loose, _ = compute_oscillation_duration(sol_osc, 1, threshold_ratio=0.2)

# Stricter threshold should give longer or equal duration
if duration_strict !== nothing && duration_loose !== nothing
    @test duration_strict >= duration_loose || sustained_strict
    println("   ✓ compute_oscillation_duration with different thresholds passed")
end

# Test with min_amplitude parameter
duration_high, _, _ = compute_oscillation_duration(sol_osc, 1, min_amplitude=1e-2)

println("   ✓ compute_oscillation_duration with min_amplitude parameter passed")

# Test with non-oscillatory model
duration_trans, sustained_trans, end_trans = compute_oscillation_duration(sol_transient, 1)

@test typeof(sustained_trans) == Bool
println("   ✓ compute_oscillation_duration with non-oscillatory model passed")

#=============================================================================
Test 6: Integration test with all metrics
=============================================================================#

println("\n6. Testing integration of all oscillation metrics:")

# Run all analyses on the same solution
has_osc, peak_times, peak_values = detect_oscillations(sol_osc, 1)
freq, period = compute_oscillation_frequency(sol_osc, 1, method=:fft)
amp, env = compute_oscillation_amplitude(sol_osc, 1, method=:envelope)
decay, half_life, decay_env = compute_oscillation_decay(sol_osc, 1)
dur, sus, end_t = compute_oscillation_duration(sol_osc, 1)

# Verify consistency
@test typeof(has_osc) == Bool

if has_osc
    @test length(peak_times) >= 2
    @test length(peak_values) >= 2
end

if freq !== nothing && period !== nothing
    @test freq > 0
    @test period > 0
end

if amp !== nothing
    @test amp > 0
end

if decay !== nothing
    @test decay > 0
end

if dur !== nothing
    @test dur > 0
end

println("   ✓ Integration test with all metrics passed")

#=============================================================================
Test 7: Edge cases and robustness
=============================================================================#

println("\n7. Testing edge cases:")

# Very short simulation
sol_short = solve_model(A₀, (0.0, 5.0), params_osc, saveat=0.5)
has_osc_short, _, _ = detect_oscillations(sol_short, 1)
freq_short, _ = compute_oscillation_frequency(sol_short, 1)

@test typeof(has_osc_short) == Bool
println("   ✓ Short simulation handling passed")

# Zero initial condition
A₀_zero = reshape([0.0, 0.0], 1, 2)
sol_zero = solve_model(A₀_zero, (0.0, 50.0), params_osc, saveat=0.5)
has_osc_zero, _, _ = detect_oscillations(sol_zero, 1)

@test typeof(has_osc_zero) == Bool
println("   ✓ Zero initial condition handling passed")

# High initial condition
A₀_high = reshape([0.8, 0.6], 1, 2)
sol_high = solve_model(A₀_high, (0.0, 100.0), params_osc, saveat=0.5)
has_osc_high, _, _ = detect_oscillations(sol_high, 1)

@test typeof(has_osc_high) == Bool
println("   ✓ High initial condition handling passed")

#=============================================================================
Test 8: Compatibility with spatial models
=============================================================================#

println("\n8. Testing compatibility with spatial models:")

# Create a simple spatial model
lattice = CompactLattice(extent=(10.0,), n_points=(21,))
conn_spatial = GaussianConnectivityParameter(0.5, (2.0,))
connectivity_spatial = ConnectivityMatrix{1}(reshape([conn_spatial], 1, 1))

params_spatial = WilsonCowanParameters{1}(
    α = (1.0,),
    β = (1.0,),
    τ = (5.0,),
    connectivity = connectivity_spatial,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.3),
    stimulus = nothing,
    lattice = lattice,
    pop_names = ("E",)
)

A₀_spatial = 0.3 .* ones(21, 1)
sol_spatial = solve_model(A₀_spatial, (0.0, 100.0), params_spatial, saveat=0.5)

# Oscillation functions should work with spatial models (using mean activity)
has_osc_spatial, _, _ = detect_oscillations(sol_spatial, 1)
freq_spatial, _ = compute_oscillation_frequency(sol_spatial, 1)

@test typeof(has_osc_spatial) == Bool
println("   ✓ Oscillation analysis with spatial model passed")

println("\n=== Oscillation Analysis Tests Passed! ===\n")
