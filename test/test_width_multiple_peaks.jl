using Test
using FailureOfInhibition2025

println("\n" * "="^70)
println("Testing compute_half_max_width with Multiple Peaks")
println("="^70)

@testset "Multiple Peaks Width Calculation" begin
    println("\n=== Testing width calculation with multiple traveling waves ===")
    
    # Create a profile with two distinct peaks (simulating two traveling waves)
    profile_multi = zeros(50)
    
    # First peak (taller) at index 15
    for i in 10:20
        profile_multi[i] = 1.0 * exp(-((i - 15)^2) / 8.0)
    end
    
    # Second peak (shorter) at index 35
    for i in 30:40
        profile_multi[i] = 0.7 * exp(-((i - 35)^2) / 8.0)
    end
    
    # Create mock solution
    struct MockSolution
        u::Vector{Matrix{Float64}}
        t::Vector{Float64}
    end
    
    sol = MockSolution([reshape(profile_multi, 50, 1)], [0.0])
    
    # Create lattice for physical units
    lattice = CompactLattice(extent=(10.0,), n_points=(50,))
    
    # Compute width
    width, half_max, profile = compute_half_max_width(sol, 1, 1, lattice)
    
    println("  Profile has two peaks:")
    println("    - Peak 1 at index 15 with amplitude ≈ 1.0")
    println("    - Peak 2 at index 35 with amplitude ≈ 0.7")
    println()
    println("  Results:")
    println("    - Computed width: $(round(width, digits=3))")
    println("    - Half-max level: $(round(half_max, digits=3))")
    
    # The peak is at index 15 (the taller one)
    peak_idx = argmax(profile_multi)
    @test peak_idx == 15
    
    # Find expected width around first peak only
    above_half_max = profile_multi .>= half_max
    left_expected = peak_idx
    right_expected = peak_idx
    
    while left_expected > 1 && above_half_max[left_expected - 1]
        left_expected -= 1
    end
    
    while right_expected < length(profile_multi) && above_half_max[right_expected + 1]
        right_expected += 1
    end
    
    expected_width_indices = right_expected - left_expected + 1
    spatial_scale = 10.0 / (50 - 1)
    expected_width = (expected_width_indices - 1) * spatial_scale
    
    println("    - Expected width: $(round(expected_width, digits=3))")
    println("    - Measured region: indices [$left_expected, $right_expected]")
    println()
    
    # Verify width matches expectation
    @test abs(width - expected_width) < 0.01
    
    # Verify it doesn't span both peaks
    # If it spanned both peaks, the width would be much larger
    max_possible_span = (40 - 10 + 1 - 1) * spatial_scale  # If it included both peaks
    @test width < max_possible_span * 0.6  # Should be much less than full span
    
    println("  ✓ Width correctly measured only the peak at index 15")
    println("  ✓ Did not span across the second peak at index 35")
    
    # Test with only second peak visible (simulate first peak having decayed)
    profile_single = zeros(50)
    for i in 30:40
        profile_single[i] = 1.0 * exp(-((i - 35)^2) / 8.0)
    end
    
    sol_single = MockSolution([reshape(profile_single, 50, 1)], [0.0])
    width_single, _, _ = compute_half_max_width(sol_single, 1, 1, lattice)
    
    println("\n  Testing with only second peak (at index 35):")
    println("    - Width: $(round(width_single, digits=3))")
    @test width_single > 0.0
    
    println("  ✓ Correctly measured second peak when it's the only one")
end

println("\n" * "="^70)
println("Multiple Peaks Width Test Passed!")
println("="^70)
