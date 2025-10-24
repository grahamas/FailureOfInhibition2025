#!/usr/bin/env julia

"""
Tests for stimulation functionality
"""

using FailureOfInhibition2025
using Test

# Import internal functions for testing
import FailureOfInhibition2025: stimulate!, CircleStimulus

function test_euclidean_distance()
    # Test 1D distance
    @test FailureOfInhibition2025.euclidean_distance(0.0, 3.0) == 3.0
    @test FailureOfInhibition2025.euclidean_distance(5.0, 1.0) == 4.0
    @test FailureOfInhibition2025.euclidean_distance(-2.0, 2.0) == 4.0
    
    # Test 2D distance
    @test FailureOfInhibition2025.euclidean_distance((0.0, 0.0), (3.0, 4.0)) ≈ 5.0
    @test FailureOfInhibition2025.euclidean_distance((1.0, 1.0), (1.0, 1.0)) ≈ 0.0
    @test FailureOfInhibition2025.euclidean_distance((0.0, 0.0), (1.0, 0.0)) ≈ 1.0
    
    # Test 3D distance
    @test FailureOfInhibition2025.euclidean_distance((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)) ≈ 1.0
    @test FailureOfInhibition2025.euclidean_distance((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) ≈ sqrt(3.0)
end

function test_circle_stimulus_construction()
    # Create a simple lattice for construction tests
    lattice = CompactLattice(-5.0, 5.0, 11)
    
    # Test basic construction
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
    
    # Test construction with explicit center
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
    
    # Test construction with multiple time windows
    stim_multi = CircleStimulus(
        radius=1.0,
        strength=2.0,
        time_windows=[(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)],
        lattice=lattice
    )
    @test length(stim_multi.time_windows) == 3
    @test stim_multi.time_windows[2] == (2.0, 3.0)
    @test stim_multi.baseline == 0.0  # default value
end

function test_stimulate_1d()
    # Create a 1D compact lattice
    lattice = CompactLattice(-5.0, 5.0, 11)  # 1D lattice from -5 to 5 with 11 points
    @test size(lattice) == (11,)
    
    # Create stimulus centered at origin
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
    
    # Test stimulus outside time window
    dA_outside = zeros(Float64, size(lattice)...)
    stimulate!(dA_outside, A, stim, 15.0)  # t=15 is outside [0,10]
    
    # Only baseline should be applied
    @test all(dA_outside .≈ stim.baseline)
    
    # Test stimulus with explicit center
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
end

function test_stimulate_2d()    
    # Create a 2D compact lattice
    lattice = CompactLattice((-5.0, -5.0), (5.0, 5.0), (11, 11))
    @test size(lattice) == (11, 11)
    
    # Create stimulus centered at origin
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
    
    # Test with offset center
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
    
    # Test that some points are stimulated
    num_stimulated = sum(dA_offset .> stim_offset.baseline)
    @test num_stimulated > 0
    @test num_stimulated < length(dA_offset)  # Not all points stimulated
end

function test_time_windows()
    lattice = CompactLattice(-2.0, 2.0, 5)
    
    # Create stimulus with multiple disjoint time windows
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
    
    # Test time between windows
    fill!(dA, 0.0)
    stimulate!(dA, A, stim, 2.0)
    @test all(dA .== 0.0)  # baseline is 0, so all should be 0
    
    # Test time inside second window
    fill!(dA, 0.0)
    stimulate!(dA, A, stim, 3.5)
    @test any(dA .> 0.0)
    
    # Test time after all windows
    fill!(dA, 0.0)
    stimulate!(dA, A, stim, 10.0)
    @test all(dA .== 0.0)
    
    # Test edge cases (at boundaries)
    fill!(dA, 0.0)
    stimulate!(dA, A, stim, 1.0)  # Right at end of first window
    @test any(dA .> 0.0)
    
    fill!(dA, 0.0)
    stimulate!(dA, A, stim, 3.0)  # Right at start of second window
    @test any(dA .> 0.0)
end

function test_edge_cases()
    lattice = CompactLattice(-2.0, 2.0, 5)
    
    # Test with zero radius
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
    
    # Test with very large radius
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
    
    # Test with negative baseline
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
end

function main()    
    test_euclidean_distance()
    test_circle_stimulus_construction()
    test_stimulate_1d()
    test_stimulate_2d()
    test_time_windows()
    test_edge_cases()
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
