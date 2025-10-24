#!/usr/bin/env julia

"""
Tests for stimulation functionality
"""

using FailureOfInhibition2025
using Test

# Import internal functions for testing
import FailureOfInhibition2025: stimulate!, CircleStimulus

function test_euclidean_distance()
    println("\n=== Testing Euclidean Distance ===")
    
    # Test 1D distance
    println("\n1. Testing 1D Euclidean distance:")
    @test FailureOfInhibition2025.euclidean_distance(0.0, 3.0) == 3.0
    @test FailureOfInhibition2025.euclidean_distance(5.0, 1.0) == 4.0
    @test FailureOfInhibition2025.euclidean_distance(-2.0, 2.0) == 4.0
    println("   ✓ 1D distance tests passed")
    
    # Test 2D distance
    println("\n2. Testing 2D Euclidean distance:")
    @test FailureOfInhibition2025.euclidean_distance((0.0, 0.0), (3.0, 4.0)) ≈ 5.0
    @test FailureOfInhibition2025.euclidean_distance((1.0, 1.0), (1.0, 1.0)) ≈ 0.0
    @test FailureOfInhibition2025.euclidean_distance((0.0, 0.0), (1.0, 0.0)) ≈ 1.0
    println("   ✓ 2D distance tests passed")
    
    # Test 3D distance
    println("\n3. Testing 3D Euclidean distance:")
    @test FailureOfInhibition2025.euclidean_distance((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) ≈ 1.0
    @test FailureOfInhibition2025.euclidean_distance((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) ≈ sqrt(3.0)
    println("   ✓ 3D distance tests passed")
    
    println("\n=== Euclidean Distance Tests Passed! ===")
end

function test_circle_stimulus_construction()
    println("\n=== Testing CircleStimulus Construction ===")
    
    # Create a simple lattice for construction tests
    lattice = CompactLattice(-5.0, 5.0, 11)
    
    # Test basic construction
    println("\n1. Testing basic CircleStimulus construction:")
    stim = CircleStimulus(
        radius=1.0,
        strength=5.0,
        time_windows=[(0.0, 10.0)],
        lattice=lattice,
        center=nothing,
        baseline=0.0
    )
    @test stim.radius == 1.0
    @test stim.strength == 5.0
    @test length(stim.time_windows) == 1
    @test stim.time_windows[1] == (0.0, 10.0)
    @test stim.center === nothing
    @test stim.baseline == 0.0
    @test stim.lattice === lattice
    println("   ✓ Basic construction passed")
    
    # Test construction with explicit center
    println("\n2. Testing CircleStimulus with explicit center:")
    stim_centered = CircleStimulus(
        radius=2.0,
        strength=3.0,
        time_windows=[(0.0, 5.0)],
        lattice=lattice,
        center=(1.0,),
        baseline=-1.0
    )
    @test stim_centered.center == (1.0,)
    @test stim_centered.radius == 2.0
    @test stim_centered.baseline == -1.0
    println("   ✓ Construction with center passed")
    
    # Test construction with multiple time windows
    println("\n3. Testing CircleStimulus with multiple time windows:")
    stim_multi = CircleStimulus(
        radius=1.0,
        strength=2.0,
        time_windows=[(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)],
        lattice=lattice
    )
    @test length(stim_multi.time_windows) == 3
    @test stim_multi.time_windows[2] == (2.0, 3.0)
    @test stim_multi.baseline == 0.0  # default value
    println("   ✓ Multiple time windows passed")
    
    println("\n=== CircleStimulus Construction Tests Passed! ===")
end

function test_stimulate_1d()
    println("\n=== Testing stimulate! with 1D Lattice ===")
    
    # Create a 1D compact lattice
    println("\n1. Setting up 1D test lattice:")
    lattice = CompactLattice(-5.0, 5.0, 11)  # 1D lattice from -5 to 5 with 11 points
    @test size(lattice) == (11,)
    println("   ✓ 1D lattice created")
    
    # Create stimulus centered at origin
    println("\n2. Testing stimulus at origin:")
    stim = CircleStimulus(
        radius=2.0,
        strength=10.0,
        time_windows=[(0.0, 10.0)],
        lattice=lattice,
        baseline=1.0
    )
    
    dA = zeros(Float64, size(lattice)...)
    A = zeros(Float64, size(lattice)...)
    
    # Apply stimulus at t=5 (within time window)
    stimulate!(dA, A, stim, 5.0)
    
    # Check that baseline is applied everywhere
    @test all(dA .>= stim.baseline)
    
    # Check that points within radius get additional strength
    coords = coordinates(lattice)
    center = (0.0,)
    for (idx, coord) in enumerate(coords)
        dist = FailureOfInhibition2025.euclidean_distance(coord, center)
        if dist <= stim.radius
            @test dA[idx] ≈ stim.baseline + stim.strength
        else
            @test dA[idx] ≈ stim.baseline
        end
    end
    println("   ✓ Stimulus at origin passed")
    
    # Test stimulus outside time window
    println("\n3. Testing stimulus outside time window:")
    dA_outside = zeros(Float64, size(lattice)...)
    stimulate!(dA_outside, A, stim, 15.0)  # t=15 is outside [0,10]
    
    # Only baseline should be applied
    @test all(dA_outside .≈ stim.baseline)
    println("   ✓ Time window check passed")
    
    # Test stimulus with explicit center
    println("\n4. Testing stimulus with offset center:")
    stim_offset = CircleStimulus(
        radius=1.5,
        strength=5.0,
        time_windows=[(0.0, 10.0)],
        lattice=lattice,
        center=(2.0,),
        baseline=0.0
    )
    
    dA_offset = zeros(Float64, size(lattice)...)
    stimulate!(dA_offset, A, stim_offset, 5.0)
    
    # Check that stimulus is centered at x=2
    for (idx, coord) in enumerate(coords)
        dist = FailureOfInhibition2025.euclidean_distance(coord, (2.0,))
        if dist <= stim_offset.radius
            @test dA_offset[idx] ≈ stim_offset.strength
        else
            @test dA_offset[idx] ≈ 0.0
        end
    end
    println("   ✓ Offset center passed")
    
    println("\n=== 1D stimulate! Tests Passed! ===")
end

function test_stimulate_2d()
    println("\n=== Testing stimulate! with 2D Lattice ===")
    
    # Create a 2D compact lattice
    println("\n1. Setting up 2D test lattice:")
    lattice = CompactLattice((-5.0, -5.0), (5.0, 5.0), (11, 11))
    @test size(lattice) == (11, 11)
    println("   ✓ 2D lattice created")
    
    # Create stimulus centered at origin
    println("\n2. Testing 2D stimulus at origin:")
    stim = CircleStimulus(
        radius=2.0,
        strength=10.0,
        time_windows=[(0.0, 10.0)],
        lattice=lattice,
        baseline=1.0
    )
    
    dA = zeros(Float64, size(lattice)...)
    A = zeros(Float64, size(lattice)...)
    
    stimulate!(dA, A, stim, 5.0)
    
    # Check spatial pattern
    coords = coordinates(lattice)
    center = (0.0, 0.0)
    for idx in CartesianIndices(coords)
        coord = coords[idx]
        dist = FailureOfInhibition2025.euclidean_distance(coord, center)
        if dist <= stim.radius
            @test dA[idx] ≈ stim.baseline + stim.strength
        else
            @test dA[idx] ≈ stim.baseline
        end
    end
    println("   ✓ 2D stimulus at origin passed")
    
    # Test with offset center
    println("\n3. Testing 2D stimulus with offset center:")
    stim_offset = CircleStimulus(
        radius=1.5,
        strength=8.0,
        time_windows=[(0.0, 10.0)],
        lattice=lattice,
        center=(2.0, -1.0),
        baseline=0.5
    )
    
    dA_offset = zeros(Float64, size(lattice)...)
    stimulate!(dA_offset, A, stim_offset, 3.0)
    
    for idx in CartesianIndices(coords)
        coord = coords[idx]
        dist = FailureOfInhibition2025.euclidean_distance(coord, (2.0, -1.0))
        if dist <= stim_offset.radius
            @test dA_offset[idx] ≈ stim_offset.baseline + stim_offset.strength
        else
            @test dA_offset[idx] ≈ stim_offset.baseline
        end
    end
    println("   ✓ 2D offset stimulus passed")
    
    # Test that some points are stimulated
    println("\n4. Verifying stimulation pattern:")
    num_stimulated = sum(dA_offset .> stim_offset.baseline)
    @test num_stimulated > 0
    @test num_stimulated < length(dA_offset)  # Not all points stimulated
    println("   ✓ Stimulation pattern verified ($(num_stimulated) points stimulated)")
    
    println("\n=== 2D stimulate! Tests Passed! ===")
end

function test_stimulate_wrapper()
    println("\n=== Testing stimulate Wrapper Function ===")
    
    # Test that the wrapper function exists and works
    println("\n1. Testing stimulate wrapper:")
    lattice = CompactLattice(-3.0, 3.0, 7)
    stim = CircleStimulus(
        radius=1.0,
        strength=5.0,
        time_windows=[(0.0, 10.0)],
        lattice=lattice
    )
    
    dA = zeros(Float64, size(lattice)...)
    A = zeros(Float64, size(lattice)...)
    
    # Call the wrapper function
    stimulate(dA, A, stim, 5.0)
    
    # Verify it had the same effect as stimulate!
    @test any(dA .> 0.0)  # Some points should be stimulated
    println("   ✓ Wrapper function works")
    
    println("\n=== stimulate Wrapper Tests Passed! ===")
end

function test_time_windows()
    println("\n=== Testing Time Window Functionality ===")
    
    lattice = CompactLattice(-2.0, 2.0, 5)
    
    # Create stimulus with multiple disjoint time windows
    println("\n1. Testing multiple time windows:")
    stim = CircleStimulus(
        radius=1.0,
        strength=10.0,
        time_windows=[(0.0, 1.0), (3.0, 4.0), (6.0, 7.0)],
        lattice=lattice
    )
    
    dA = zeros(Float64, size(lattice)...)
    A = zeros(Float64, size(lattice)...)
    
    # Test time inside first window
    stimulate!(dA, A, stim, 0.5)
    @test any(dA .> 0.0)
    println("   ✓ First time window active")
    
    # Test time between windows
    fill!(dA, 0.0)
    stimulate!(dA, A, stim, 2.0)
    @test all(dA .== 0.0)  # baseline is 0, so all should be 0
    println("   ✓ Between windows inactive")
    
    # Test time inside second window
    fill!(dA, 0.0)
    stimulate!(dA, A, stim, 3.5)
    @test any(dA .> 0.0)
    println("   ✓ Second time window active")
    
    # Test time after all windows
    fill!(dA, 0.0)
    stimulate!(dA, A, stim, 10.0)
    @test all(dA .== 0.0)
    println("   ✓ After all windows inactive")
    
    # Test edge cases (at boundaries)
    println("\n2. Testing time window boundaries:")
    fill!(dA, 0.0)
    stimulate!(dA, A, stim, 1.0)  # Right at end of first window
    @test any(dA .> 0.0)
    
    fill!(dA, 0.0)
    stimulate!(dA, A, stim, 3.0)  # Right at start of second window
    @test any(dA .> 0.0)
    println("   ✓ Boundary times work correctly")
    
    println("\n=== Time Window Tests Passed! ===")
end

function test_edge_cases()
    println("\n=== Testing Edge Cases ===")
    
    lattice = CompactLattice(-2.0, 2.0, 5)
    
    # Test with zero radius
    println("\n1. Testing zero radius stimulus:")
    stim_zero = CircleStimulus(
        radius=0.0,
        strength=10.0,
        time_windows=[(0.0, 10.0)],
        lattice=lattice,
        baseline=1.0
    )
    dA = zeros(Float64, size(lattice)...)
    A = zeros(Float64, size(lattice)...)
    stimulate!(dA, A, stim_zero, 5.0)
    # With zero radius, only the exact center point might be stimulated
    # Depending on floating point comparison
    @test all(dA .>= stim_zero.baseline)
    println("   ✓ Zero radius handled")
    
    # Test with very large radius
    println("\n2. Testing large radius stimulus:")
    stim_large = CircleStimulus(
        radius=100.0,
        strength=5.0,
        time_windows=[(0.0, 10.0)],
        lattice=lattice
    )
    fill!(dA, 0.0)
    stimulate!(dA, A, stim_large, 5.0)
    # All points should be stimulated
    @test all(dA .≈ stim_large.strength)
    println("   ✓ Large radius handled")
    
    # Test with negative baseline
    println("\n3. Testing negative baseline:")
    stim_neg = CircleStimulus(
        radius=1.0,
        strength=10.0,
        time_windows=[(0.0, 10.0)],
        lattice=lattice,
        baseline=-5.0
    )
    fill!(dA, 0.0)
    stimulate!(dA, A, stim_neg, 5.0)
    @test all(dA .>= stim_neg.baseline)
    @test any(dA .> stim_neg.baseline)  # Some points get extra strength
    println("   ✓ Negative baseline handled")
    
    println("\n=== Edge Case Tests Passed! ===")
end

function main()
    println("Running stimulation function tests...")
    
    test_euclidean_distance()
    test_circle_stimulus_construction()
    test_stimulate_1d()
    test_stimulate_2d()
    test_stimulate_wrapper()
    test_time_windows()
    test_edge_cases()
    
    println("\n🎉 All stimulation tests completed successfully!")
    println("\nStimulation functionality is ready for use in neural models.")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
