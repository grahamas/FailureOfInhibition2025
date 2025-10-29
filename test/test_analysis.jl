using Test
using FailureOfInhibition2025
using Statistics

println("\n=== Testing Traveling Wave Analysis Functions ===\n")

# Helper function to create a simple 1D spatial model
function create_test_spatial_model(n_points=51)
    lattice = CompactLattice(extent=(10.0,), n_points=(n_points,))
    
    # Simple connectivity for testing
    conn = GaussianConnectivityParameter(0.5, (2.0,))
    connectivity = ConnectivityMatrix{1}(reshape([conn], 1, 1))
    
    params = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (5.0,),
        connectivity = connectivity,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.3),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E",)
    )
    
    return params, lattice
end

#=============================================================================
Test 1: detect_traveling_peak
=============================================================================#

println("1. Testing detect_traveling_peak:")

# Create a spatial model
params, lattice = create_test_spatial_model(51)

# Create initial condition with localized activity
A₀ = zeros(51, 1)
A₀[10:15, 1] .= 0.5  # Localized bump

# Solve
tspan = (0.0, 20.0)
sol = solve_model(A₀, tspan, params, saveat=0.5)

# Test detection
has_peak, trajectory, times = detect_traveling_peak(sol, 1, threshold=0.1)

@test typeof(has_peak) == Bool
@test length(trajectory) == length(times)
@test all(trajectory .>= 1)
@test all(trajectory .<= 51)
println("   ✓ detect_traveling_peak basic functionality passed")

# Test error handling for non-spatial model
point_lattice = PointLattice()
point_conn = ConnectivityMatrix{1}(reshape([ScalarConnectivity(0.5)], 1, 1))
point_params = WilsonCowanParameters{1}(
    α = (1.0,),
    β = (1.0,),
    τ = (5.0,),
    connectivity = point_conn,
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.3),
    stimulus = nothing,
    lattice = point_lattice,
    pop_names = ("E",)
)
A₀_point = reshape([0.3], 1, 1)
sol_point = solve_model(A₀_point, (0.0, 10.0), point_params, saveat=0.5)

@test_throws ErrorException detect_traveling_peak(sol_point, 1)
println("   ✓ detect_traveling_peak error handling passed")

#=============================================================================
Test 2: compute_decay_rate
=============================================================================#

println("\n2. Testing compute_decay_rate:")

# Test with spatial model
decay_rate, amplitudes = compute_decay_rate(sol, 1)

@test length(amplitudes) > 0
@test all(amplitudes .>= 0)
@test decay_rate === nothing || decay_rate >= 0
println("   ✓ compute_decay_rate basic functionality passed")

# Test with point model
decay_rate_point, amp_point = compute_decay_rate(sol_point, 1)

@test length(amp_point) > 0
println("   ✓ compute_decay_rate with point model passed")

# Test with fit_window parameter
decay_rate_window, _ = compute_decay_rate(sol, 1, fit_window=(5.0, 15.0))

@test decay_rate_window === nothing || decay_rate_window >= 0
println("   ✓ compute_decay_rate with fit_window passed")

#=============================================================================
Test 3: compute_amplitude
=============================================================================#

println("\n3. Testing compute_amplitude:")

# Test different methods
amp_max = compute_amplitude(sol, 1, method=:max)
amp_peak = compute_amplitude(sol, 1, method=:peak)
amp_mean = compute_amplitude(sol, 1, method=:mean_max)

@test amp_max >= 0
@test amp_peak >= 0
@test amp_mean >= 0
@test amp_max >= amp_mean  # max should be >= mean
println("   ✓ compute_amplitude with all methods passed")

# Test with point model
amp_point = compute_amplitude(sol_point, 1)

@test amp_point >= 0
println("   ✓ compute_amplitude with point model passed")

# Test error handling for invalid method
@test_throws ErrorException compute_amplitude(sol, 1, method=:invalid)
println("   ✓ compute_amplitude error handling passed")

#=============================================================================
Test 4: compute_distance_traveled
=============================================================================#

println("\n4. Testing compute_distance_traveled:")

# Test without lattice (returns index distance)
distance_idx, traj_idx = compute_distance_traveled(sol, 1, nothing, threshold=0.1)

@test distance_idx >= 0
@test length(traj_idx) <= length(sol.t)
println("   ✓ compute_distance_traveled without lattice passed")

# Test with lattice (returns physical distance)
distance_phys, traj_phys = compute_distance_traveled(sol, 1, lattice, threshold=0.1)

@test distance_phys >= 0
@test length(traj_phys) <= length(sol.t)
println("   ✓ compute_distance_traveled with lattice passed")

# Test error handling for non-spatial model
@test_throws ErrorException compute_distance_traveled(sol_point, 1, nothing)
println("   ✓ compute_distance_traveled error handling passed")

#=============================================================================
Test 5: compute_half_max_width
=============================================================================#

println("\n5. Testing compute_half_max_width:")

# Test without lattice
width_idx, half_max_idx, profile_idx = compute_half_max_width(sol, 1, nothing, nothing)

@test width_idx >= 0
@test half_max_idx >= 0
@test length(profile_idx) == 51
println("   ✓ compute_half_max_width without lattice passed")

# Test with lattice
width_phys, half_max_phys, profile_phys = compute_half_max_width(sol, 1, nothing, lattice)

@test width_phys >= 0
@test half_max_phys >= 0
@test length(profile_phys) == 51
println("   ✓ compute_half_max_width with lattice passed")

# Test with specific time index
width_t5, _, _ = compute_half_max_width(sol, 1, 5, lattice)

@test width_t5 >= 0
println("   ✓ compute_half_max_width with time_idx passed")

# Test different baseline methods
width_min, _, _ = compute_half_max_width(sol, 1, nothing, lattice, baseline=:min)
width_zero, _, _ = compute_half_max_width(sol, 1, nothing, lattice, baseline=:zero)
width_custom, _, _ = compute_half_max_width(sol, 1, nothing, lattice, baseline=0.1)

@test width_min >= 0
@test width_zero >= 0
@test width_custom >= 0
println("   ✓ compute_half_max_width with different baselines passed")

# Test error handling for non-spatial model
@test_throws ErrorException compute_half_max_width(sol_point, 1, nothing, nothing)
println("   ✓ compute_half_max_width error handling passed")

#=============================================================================
Test 6: Integration test with realistic traveling wave
=============================================================================#

println("\n6. Testing with realistic traveling wave scenario:")

# Create a model more likely to produce traveling waves
params_wave, lattice_wave = create_test_spatial_model(101)

# Asymmetric initial condition to encourage traveling behavior
A₀_wave = zeros(101, 1)
A₀_wave[5:10, 1] .= 0.8  # Strong localized bump on one side

sol_wave = solve_model(A₀_wave, (0.0, 30.0), params_wave, saveat=0.2)

# Test all metrics together
has_peak_wave, traj_wave, times_wave = detect_traveling_peak(sol_wave, 1, threshold=0.15)
decay_wave, _ = compute_decay_rate(sol_wave, 1)
amp_wave = compute_amplitude(sol_wave, 1)
dist_wave, _ = compute_distance_traveled(sol_wave, 1, lattice_wave, threshold=0.15)
width_wave, _, _ = compute_half_max_width(sol_wave, 1, nothing, lattice_wave)

@test typeof(has_peak_wave) == Bool
@test amp_wave > 0
@test dist_wave >= 0
@test width_wave >= 0
println("   ✓ Integration test with all metrics passed")

#=============================================================================
Test 7: Edge cases and robustness
=============================================================================#

println("\n7. Testing edge cases:")

# Very short simulation
sol_short = solve_model(A₀, (0.0, 1.0), params, saveat=0.5)
decay_short, _ = compute_decay_rate(sol_short, 1)
@test decay_short === nothing || decay_short >= 0
println("   ✓ Short simulation handling passed")

# Zero initial condition
A₀_zero = zeros(51, 1)
sol_zero = solve_model(A₀_zero, (0.0, 10.0), params, saveat=0.5)
amp_zero = compute_amplitude(sol_zero, 1)
@test amp_zero >= 0
println("   ✓ Zero initial condition handling passed")

# Very low threshold
has_peak_low, _, _ = detect_traveling_peak(sol, 1, threshold=0.01)
@test typeof(has_peak_low) == Bool
println("   ✓ Low threshold handling passed")

println("\n=== Traveling Wave Analysis Tests Passed! ===\n")
