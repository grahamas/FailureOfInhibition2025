#!/usr/bin/env julia

"""
Comprehensive tests for space/lattice/coordinates functionality.

Tests include:
- Creating compact (non-periodic) spaces
- Creating periodic spaces  
- Creating coordinates from those spaces
- Creating distance matrices from those coordinates
"""

using FailureOfInhibition2025
using Test

function test_compact_lattice_creation()
    println("=== Testing Compact Lattice Creation ===")
    
    # Test 1D compact lattice
    println("\n1. Testing 1D CompactLattice creation:")
    lat1d = CompactLattice(discrete_lattice(0.0, 10.0, 11))
    @test size(lat1d) == (11,)
    @test start(lat1d) == (0.0,)
    @test stop(lat1d) == (10.0,)
    @test extent(lat1d) == (10.0,)
    @test step(lat1d) == (1.0,)
    println("   ✓ 1D CompactLattice creation tests passed")
    
    # Test 1D compact lattice with extent notation
    println("\n2. Testing 1D CompactLattice with extent notation:")
    lat1d_ext = CompactLattice(extent=(20.0,), n_points=(21,))
    @test size(lat1d_ext) == (21,)
    @test extent(lat1d_ext) == (20.0,)
    # Should be centered at origin
    @test start(lat1d_ext)[1] ≈ -10.0
    @test stop(lat1d_ext)[1] ≈ 10.0
    println("   ✓ 1D CompactLattice extent notation tests passed")
    
    # Test 2D compact lattice
    println("\n3. Testing 2D CompactLattice creation:")
    lat2d = CompactLattice(discrete_lattice((-1.0, -1.0), (1.0, 1.0), (5, 5)))
    @test size(lat2d) == (5, 5)
    @test start(lat2d) == (-1.0, -1.0)
    @test stop(lat2d) == (1.0, 1.0)
    @test extent(lat2d) == (2.0, 2.0)
    @test step(lat2d) == (0.5, 0.5)
    println("   ✓ 2D CompactLattice creation tests passed")
    
    # Test 2D compact lattice with extent notation
    println("\n4. Testing 2D CompactLattice with extent notation:")
    lat2d_ext = CompactLattice(extent=(4.0, 6.0), n_points=(9, 13))
    @test size(lat2d_ext) == (9, 13)
    @test extent(lat2d_ext) == (4.0, 6.0)
    @test start(lat2d_ext)[1] ≈ -2.0
    @test start(lat2d_ext)[2] ≈ -3.0
    @test stop(lat2d_ext)[1] ≈ 2.0
    @test stop(lat2d_ext)[2] ≈ 3.0
    println("   ✓ 2D CompactLattice extent notation tests passed")
    
    println("\n=== All Compact Lattice Creation Tests Passed! ===")
end

function test_periodic_lattice_creation()
    println("\n=== Testing Periodic Lattice Creation ===")
    
    # Test 1D periodic lattice
    println("\n1. Testing 1D PeriodicLattice creation:")
    plat1d = PeriodicLattice(extent=(10.0,), n_points=(10,))
    @test size(plat1d) == (10,)
    @test extent(plat1d) == (10.0,)
    # For periodic lattices, the last point should not coincide with first
    coords = coordinates(plat1d)
    @test coords[1][1] ≈ -5.0
    @test coords[end][1] ≈ 4.0  # -5.0 + 9*1.0
    println("   ✓ 1D PeriodicLattice creation tests passed")
    
    # Test 2D periodic lattice
    println("\n2. Testing 2D PeriodicLattice creation:")
    plat2d = PeriodicLattice(extent=(2π, 2π), n_points=(32, 32))
    @test size(plat2d) == (32, 32)
    @test extent(plat2d) == (2π, 2π)
    coords2d = coordinates(plat2d)
    @test coords2d[1,1][1] ≈ -π
    @test coords2d[1,1][2] ≈ -π
    println("   ✓ 2D PeriodicLattice creation tests passed")
    
    # Test that periodic lattice spacing is different from compact
    println("\n3. Testing periodic vs compact spacing:")
    # Compact lattice: spacing = extent/(n-1)
    clat = CompactLattice(extent=(10.0,), n_points=(11,))
    @test step(clat)[1] ≈ 1.0  # 10/(11-1) = 1.0
    
    # Periodic lattice: spacing = extent/n
    plat = PeriodicLattice(extent=(10.0,), n_points=(10,))
    coords_p = coordinates(plat)
    spacing = coords_p[2][1] - coords_p[1][1]
    @test spacing ≈ 1.0  # 10/10 = 1.0
    println("   ✓ Periodic vs compact spacing tests passed")
    
    println("\n=== All Periodic Lattice Creation Tests Passed! ===")
end

function test_coordinates_from_spaces()
    println("\n=== Testing Coordinates from Spaces ===")
    
    # Test coordinates from 1D compact lattice
    println("\n1. Testing coordinates from 1D CompactLattice:")
    lat1d = CompactLattice(discrete_lattice(0.0, 4.0, 5))
    coords1d = coordinates(lat1d)
    @test size(coords1d) == (5,)
    @test coords1d[1] == (0.0,)
    @test coords1d[2] == (1.0,)
    @test coords1d[3] == (2.0,)
    @test coords1d[4] == (3.0,)
    @test coords1d[5] == (4.0,)
    println("   ✓ 1D CompactLattice coordinates tests passed")
    
    # Test coordinates from 2D compact lattice
    println("\n2. Testing coordinates from 2D CompactLattice:")
    lat2d = CompactLattice(discrete_lattice((-1.0, -1.0), (1.0, 1.0), (3, 3)))
    coords2d = coordinates(lat2d)
    @test size(coords2d) == (3, 3)
    @test coords2d[1,1] == (-1.0, -1.0)
    @test coords2d[2,2] == (0.0, 0.0)
    @test coords2d[3,3] == (1.0, 1.0)
    @test coords2d[1,3] == (-1.0, 1.0)
    @test coords2d[3,1] == (1.0, -1.0)
    println("   ✓ 2D CompactLattice coordinates tests passed")
    
    # Test coordinates from 1D periodic lattice
    println("\n3. Testing coordinates from 1D PeriodicLattice:")
    plat1d = PeriodicLattice(extent=(8.0,), n_points=(8,))
    pcoords1d = coordinates(plat1d)
    @test size(pcoords1d) == (8,)
    @test pcoords1d[1][1] ≈ -4.0
    @test pcoords1d[end][1] ≈ 3.0
    # Check uniform spacing
    for i in 1:7
        @test pcoords1d[i+1][1] - pcoords1d[i][1] ≈ 1.0
    end
    println("   ✓ 1D PeriodicLattice coordinates tests passed")
    
    # Test coordinates from 2D periodic lattice
    println("\n4. Testing coordinates from 2D PeriodicLattice:")
    plat2d = PeriodicLattice(extent=(4.0, 6.0), n_points=(4, 6))
    pcoords2d = coordinates(plat2d)
    @test size(pcoords2d) == (4, 6)
    @test pcoords2d[1,1][1] ≈ -2.0
    @test pcoords2d[1,1][2] ≈ -3.0
    println("   ✓ 2D PeriodicLattice coordinates tests passed")
    
    # Test coordinate_axes function
    println("\n5. Testing coordinate_axes function:")
    lat = CompactLattice(discrete_lattice((-2.0, -3.0), (2.0, 3.0), (5, 7)))
    axes = coordinate_axes(lat)
    @test length(axes) == 2
    @test length(axes[1]) == 5
    @test length(axes[2]) == 7
    @test axes[1][1] ≈ -2.0
    @test axes[1][end] ≈ 2.0
    @test axes[2][1] ≈ -3.0
    @test axes[2][end] ≈ 3.0
    println("   ✓ coordinate_axes tests passed")
    
    println("\n=== All Coordinates from Spaces Tests Passed! ===")
end

function test_distance_matrices()
    println("\n=== Testing Distance Matrices (differences) ===")
    
    # Test differences for 1D compact lattice
    println("\n1. Testing differences for 1D CompactLattice:")
    lat1d = CompactLattice(discrete_lattice(0.0, 4.0, 5))
    diffs1d = differences(lat1d)
    @test size(diffs1d) == (5, 5)
    # Distance from point 1 to itself should be 0
    @test diffs1d[1,1] == (0.0,)
    # Distance from point 1 to point 3 should be 2
    @test diffs1d[1,3] == (2.0,)
    # Distance from point 3 to point 1 should be 2 (symmetric)
    @test diffs1d[3,1] == (2.0,)
    # Distance from point 1 to point 5 should be 4
    @test diffs1d[1,5] == (4.0,)
    println("   ✓ 1D CompactLattice differences tests passed")
    
    # Test differences for 2D compact lattice
    println("\n2. Testing differences for 2D CompactLattice:")
    lat2d = CompactLattice(discrete_lattice((0.0, 0.0), (2.0, 2.0), (3, 3)))
    diffs2d = differences(lat2d)
    @test size(diffs2d) == (3, 3, 3, 3)
    # Distance from (0,0) to itself should be (0,0)
    @test diffs2d[1,1,1,1] == (0.0, 0.0)
    # Distance from (0,0) to (1,1) should be (1,1)
    @test diffs2d[1,1,2,2] == (1.0, 1.0)
    # Distance from (0,0) to (2,2) should be (2,2)
    @test diffs2d[1,1,3,3] == (2.0, 2.0)
    println("   ✓ 2D CompactLattice differences tests passed")
    
    # Test differences for 1D periodic lattice
    println("\n3. Testing differences for 1D PeriodicLattice:")
    plat1d = PeriodicLattice(extent=(8.0,), n_points=(8,))
    pdiffs1d = differences(plat1d)
    @test size(pdiffs1d) == (8, 8)
    # Distance from point to itself should be 0
    @test pdiffs1d[1,1] == (0.0,)
    # In periodic lattice, check wrapping behavior
    # Points at opposite ends should be close due to wrapping
    # Point 1 is at -4.0, point 8 is at 3.0
    # Direct distance: 7.0, but wrapped distance should be 1.0 (8.0 - 7.0)
    @test pdiffs1d[1,8] == (1.0,)
    println("   ✓ 1D PeriodicLattice differences tests passed")
    
    # Test differences with reference location
    println("\n4. Testing differences with reference location:")
    lat1d = CompactLattice(discrete_lattice(0.0, 4.0, 5))
    ref_loc = (2.0,)
    diffs_ref = differences(lat1d, ref_loc)
    @test size(diffs_ref) == (5,)
    # Distance from point 1 (0.0) to reference (2.0) should be 2.0
    @test diffs_ref[1] == (2.0,)
    # Distance from point 3 (2.0) to reference (2.0) should be 0.0
    @test diffs_ref[3] == (0.0,)
    # Distance from point 5 (4.0) to reference (2.0) should be 2.0
    @test diffs_ref[5] == (2.0,)
    println("   ✓ differences with reference location tests passed")
    
    # Test differences with reference location for 2D
    println("\n5. Testing 2D differences with reference location:")
    lat2d = CompactLattice(discrete_lattice((-1.0, -1.0), (1.0, 1.0), (3, 3)))
    ref_loc2d = (0.0, 0.0)
    diffs_ref2d = differences(lat2d, ref_loc2d)
    @test size(diffs_ref2d) == (3, 3)
    # Distance from corner (-1,-1) to center (0,0)
    @test diffs_ref2d[1,1] == (1.0, 1.0)
    # Distance from center (0,0) to center (0,0)
    @test diffs_ref2d[2,2] == (0.0, 0.0)
    # Distance from corner (1,1) to center (0,0)
    @test diffs_ref2d[3,3] == (1.0, 1.0)
    println("   ✓ 2D differences with reference location tests passed")
    
    println("\n=== All Distance Matrix Tests Passed! ===")
end

function test_distance_metrics()
    println("\n=== Testing Distance Metric Functions ===")
    
    # Test abs_difference for scalars
    println("\n1. Testing abs_difference for scalars:")
    @test abs_difference((5.0, 1.0)) == 4.0
    @test abs_difference((1.0, 5.0)) == 4.0
    @test abs_difference((3.0, 3.0)) == 0.0
    println("   ✓ abs_difference scalar tests passed")
    
    # Test abs_difference for tuples
    println("\n2. Testing abs_difference for tuples:")
    @test abs_difference(((2.0, 2.0), (5.0, -5.0))) == (3.0, 7.0)
    @test abs_difference(((0.0, 0.0), (0.0, 0.0))) == (0.0, 0.0)
    @test abs_difference(((1.0, 2.0, 3.0), (4.0, 5.0, 6.0))) == (3.0, 3.0, 3.0)
    println("   ✓ abs_difference tuple tests passed")
    
    # Test abs_difference_periodic for scalars
    println("\n3. Testing abs_difference_periodic for scalars:")
    # Direct path is shorter
    @test abs_difference_periodic((1.0, 3.0), (10.0,)) == 2.0
    # Wrapped path is shorter: direct=7, wrapped=10-7=3
    @test abs_difference_periodic((1.0, 8.0), (10.0,)) == 3.0
    # Exactly at half-period
    @test abs_difference_periodic((0.0, 5.0), (10.0,)) == 5.0
    println("   ✓ abs_difference_periodic scalar tests passed")
    
    # Test abs_difference_periodic for tuples
    println("\n4. Testing abs_difference_periodic for tuples:")
    # Test 2D periodic difference
    result = abs_difference_periodic(((0.0, 0.0), (8.0, 8.0)), (10.0, 10.0))
    @test result == (2.0, 2.0)
    
    # Test mixed cases
    result2 = abs_difference_periodic(((1.0, 1.0), (3.0, 9.0)), (10.0, 10.0))
    @test result2 == (2.0, 2.0)  # Direct for x, wrapped for y
    println("   ✓ abs_difference_periodic tuple tests passed")
    
    println("\n=== All Distance Metric Tests Passed! ===")
end

function test_space_properties()
    println("\n=== Testing Space Properties and Methods ===")
    
    # Test ndims function
    println("\n1. Testing ndims function:")
    lat1d = CompactLattice(discrete_lattice(0.0, 10.0, 11))
    lat2d = CompactLattice(discrete_lattice((0.0, 0.0), (10.0, 10.0), (11, 11)))
    @test ndims(lat1d) == 1
    @test ndims(lat2d) == 2
    println("   ✓ ndims tests passed")
    
    # Test zero function
    println("\n2. Testing zero function:")
    zero_arr = zero(lat1d)
    @test size(zero_arr) == (11,)
    @test all(zero_arr .== 0.0)
    
    zero_arr2d = zero(lat2d)
    @test size(zero_arr2d) == (11, 11)
    @test all(zero_arr2d .== 0.0)
    println("   ✓ zero function tests passed")
    
    # Test fft_center_idx
    println("\n3. Testing fft_center_idx:")
    lat = CompactLattice(discrete_lattice((0.0, 0.0), (10.0, 10.0), (11, 11)))
    center_idx = fft_center_idx(lat)
    @test center_idx == CartesianIndex(6, 6)  # floor(11/2) + 1 = 6
    
    lat_even = CompactLattice(discrete_lattice((0.0, 0.0), (10.0, 10.0), (10, 10)))
    center_idx_even = fft_center_idx(lat_even)
    @test center_idx_even == CartesianIndex(6, 6)  # floor(10/2) + 1 = 6
    println("   ✓ fft_center_idx tests passed")
    
    # Test CartesianIndices
    println("\n4. Testing CartesianIndices:")
    lat = CompactLattice(discrete_lattice((0.0, 0.0), (2.0, 2.0), (3, 3)))
    cart_inds = CartesianIndices(lat)
    @test size(cart_inds) == (3, 3)
    @test cart_inds[1,1] == CartesianIndex(1, 1)
    @test cart_inds[3,3] == CartesianIndex(3, 3)
    println("   ✓ CartesianIndices tests passed")
    
    println("\n=== All Space Properties Tests Passed! ===")
end

function run_all_space_tests()
    println("Running comprehensive space/lattice/coordinates tests...")
    
    @testset "Space/Lattice/Coordinates Tests" begin
        
        @testset "Compact Lattice Creation" begin
            test_compact_lattice_creation()
        end
        
        @testset "Periodic Lattice Creation" begin
            test_periodic_lattice_creation()
        end
        
        @testset "Coordinates from Spaces" begin
            test_coordinates_from_spaces()
        end
        
        @testset "Distance Matrices" begin
            test_distance_matrices()
        end
        
        @testset "Distance Metrics" begin
            test_distance_metrics()
        end
        
        @testset "Space Properties" begin
            test_space_properties()
        end
    end
    
    println("\n🎉 All space/lattice/coordinates tests completed successfully!")
end

# Allow running this file directly
if abspath(PROGRAM_FILE) == @__FILE__
    run_all_space_tests()
end
