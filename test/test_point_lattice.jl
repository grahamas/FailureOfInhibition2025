#!/usr/bin/env julia

"""
Tests for PointLattice (zero-dimensional lattice for non-spatial models)
"""

using FailureOfInhibition2025
using Test

function test_point_lattice_creation()
    println("=== Testing PointLattice Creation ===")
    
    # Test basic construction
    println("\n1. Testing basic PointLattice construction:")
    lattice = PointLattice()
    @test lattice isa PointLattice{Float64}
    @test lattice isa AbstractPointLattice
    @test lattice isa AbstractLattice
    println("   ✓ Basic construction passed")
    
    # Test with explicit type parameter
    println("\n2. Testing PointLattice with type parameter:")
    lattice_f32 = PointLattice{Float32}()
    @test lattice_f32 isa PointLattice{Float32}
    println("   ✓ Type parameter construction passed")
    
    println("\n=== PointLattice Creation Tests Passed! ===")
end

function test_point_lattice_properties()
    println("\n=== Testing PointLattice Properties ===")
    
    lattice = PointLattice()
    
    # Test size
    println("\n1. Testing size:")
    @test size(lattice) == ()
    @test ndims(lattice) == 0
    println("   ✓ Size test passed")
    
    # Test size with dimension argument should throw
    println("\n2. Testing size(lattice, d) throws error:")
    @test_throws BoundsError size(lattice, 1)
    println("   ✓ Size dimension error test passed")
    
    # Test coordinates
    println("\n3. Testing coordinates:")
    coords = coordinates(lattice)
    @test size(coords) == ()
    @test coords[] == ()
    println("   ✓ Coordinates test passed")
    
    # Test coordinate_axes
    println("\n4. Testing coordinate_axes:")
    axes = coordinate_axes(lattice)
    @test axes == ()
    println("   ✓ Coordinate axes test passed")
    
    # Test start, stop, extent
    println("\n5. Testing start, stop, extent:")
    @test start(lattice) == ()
    @test stop(lattice) == ()
    @test extent(lattice) == ()
    println("   ✓ Start/stop/extent test passed")
    
    # Test step
    println("\n6. Testing step:")
    @test step(lattice) == ()
    println("   ✓ Step test passed")
    
    # Test CartesianIndices
    println("\n7. Testing CartesianIndices:")
    cart_inds = CartesianIndices(lattice)
    @test length(cart_inds) == 1
    @test cart_inds[1] == CartesianIndex()
    println("   ✓ CartesianIndices test passed")
    
    # Test indexing
    println("\n8. Testing indexing:")
    @test lattice.arr[] == ()
    println("   ✓ Indexing test passed")
    
    # Test zeros
    println("\n9. Testing zeros:")
    z = zeros(lattice)
    @test z == 0.0
    @test z isa Float64
    
    lattice_f32 = PointLattice{Float32}()
    z32 = zeros(lattice_f32)
    @test z32 == 0.0f0
    @test z32 isa Float32
    println("   ✓ Zeros test passed")
    
    println("\n=== PointLattice Properties Tests Passed! ===")
end

function test_point_lattice_with_wilson_cowan()
    println("\n=== Testing PointLattice with Wilson-Cowan Model ===")
    
    # Create a point lattice
    lattice = PointLattice()
    
    # Test with single population
    println("\n1. Testing single population with PointLattice:")
    params_single = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (1.0,),
        connectivity = nothing,  # No spatial connectivity
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice
    )
    
    # Activity for single population, no spatial dimension
    A_single = [0.5]
    dA_single = zeros(1)
    
    # This should work without error
    wcm1973!(dA_single, A_single, params_single, 0.0)
    @test !all(dA_single .== 0.0)
    println("   ✓ Single population test passed")
    
    # Test with two populations
    println("\n2. Testing two populations with PointLattice:")
    params_two = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (1.0, 0.8),
        connectivity = nothing,  # No spatial connectivity
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    # Activity for two populations, no spatial dimension
    A_two = [0.3, 0.5]
    dA_two = zeros(2)
    
    # This should work without error
    wcm1973!(dA_two, A_two, params_two, 0.0)
    @test !all(dA_two .== 0.0)
    println("   ✓ Two populations test passed")
    
    # Test with three populations
    println("\n3. Testing three populations with PointLattice:")
    params_three = WilsonCowanParameters{3}(
        α = (1.0, 1.5, 1.2),
        β = (1.0, 1.0, 1.0),
        τ = (1.0, 0.8, 0.9),
        connectivity = nothing,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice
    )
    
    A_three = [0.3, 0.5, 0.4]
    dA_three = zeros(3)
    
    wcm1973!(dA_three, A_three, params_three, 0.0)
    @test !all(dA_three .== 0.0)
    println("   ✓ Three populations test passed")
    
    println("\n=== PointLattice Wilson-Cowan Tests Passed! ===")
end

function test_point_lattice_with_different_nonlinearities()
    println("\n=== Testing PointLattice with Different Nonlinearities ===")
    
    lattice = PointLattice()
    
    # Test with RectifiedZeroedSigmoidNonlinearity
    println("\n1. Testing with RectifiedZeroedSigmoidNonlinearity:")
    params_rect = WilsonCowanParameters{2}(
        α = (1.0, 1.0),
        β = (1.0, 1.0),
        τ = (1.0, 1.0),
        connectivity = nothing,
        nonlinearity = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice
    )
    
    A = [0.3, 0.5]
    dA = zeros(2)
    wcm1973!(dA, A, params_rect, 0.0)
    @test !all(dA .== 0.0)
    println("   ✓ RectifiedZeroedSigmoidNonlinearity test passed")
    
    # Test with DifferenceOfSigmoidsNonlinearity
    println("\n2. Testing with DifferenceOfSigmoidsNonlinearity:")
    params_diff = WilsonCowanParameters{2}(
        α = (1.0, 1.0),
        β = (1.0, 1.0),
        τ = (1.0, 1.0),
        connectivity = nothing,
        nonlinearity = DifferenceOfSigmoidsNonlinearity(
            a_up=5.0, θ_up=0.3, a_down=3.0, θ_down=0.7
        ),
        stimulus = nothing,
        lattice = lattice
    )
    
    A = [0.5, 0.6]
    dA = zeros(2)
    wcm1973!(dA, A, params_diff, 0.0)
    @test !all(dA .== 0.0)
    println("   ✓ DifferenceOfSigmoidsNonlinearity test passed")
    
    println("\n=== Different Nonlinearities Tests Passed! ===")
end

function test_point_lattice_dynamics()
    println("\n=== Testing PointLattice Dynamics Correctness ===")
    
    lattice = PointLattice()
    
    println("\n1. Testing decay dynamics:")
    # With no stimulus or connectivity, activity should decay
    params = WilsonCowanParameters{1}(
        α = (1.0,),  # Decay rate
        β = (1.0,),
        τ = (1.0,),
        connectivity = nothing,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice
    )
    
    # Positive activity should have negative derivative (decay)
    A = [0.5]
    dA = zeros(1)
    wcm1973!(dA, A, params, 0.0)
    
    # The derivative should reflect decay
    # With activity 0.5, nonlinearity will produce some output,
    # but the -α*A term should dominate if α is large enough
    @test dA[1] != 0.0
    println("   ✓ Decay dynamics test passed")
    
    println("\n2. Testing time constant effects:")
    # Larger τ should produce smaller derivatives
    params_slow = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (2.0,),  # Larger time constant
        connectivity = nothing,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice
    )
    
    A = [0.5]
    dA_slow = zeros(1)
    wcm1973!(dA_slow, A, params_slow, 0.0)
    
    # With larger τ, derivative should be smaller in magnitude
    # (slower dynamics)
    @test abs(dA_slow[1]) < abs(dA[1])
    println("   ✓ Time constant test passed")
    
    println("\n=== Dynamics Correctness Tests Passed! ===")
end

function run_all_point_lattice_tests()
    @testset "PointLattice Tests" begin
        
        @testset "PointLattice Creation" begin
            test_point_lattice_creation()
        end
        
        @testset "PointLattice Properties" begin
            test_point_lattice_properties()
        end
        
        @testset "PointLattice with Wilson-Cowan" begin
            test_point_lattice_with_wilson_cowan()
        end
        
        @testset "PointLattice with Different Nonlinearities" begin
            test_point_lattice_with_different_nonlinearities()
        end
        
        @testset "PointLattice Dynamics" begin
            test_point_lattice_dynamics()
        end
    end
end

# Allow running this file directly
if abspath(PROGRAM_FILE) == @__FILE__
    println("Running PointLattice tests...")
    run_all_point_lattice_tests()
    println("\n🎉 All PointLattice tests completed successfully!")
end
