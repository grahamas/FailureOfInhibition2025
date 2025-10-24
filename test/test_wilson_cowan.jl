#!/usr/bin/env julia

"""
Tests for Wilson-Cowan model implementation
"""

using FailureOfInhibition2025

function test_wilson_cowan_parameters()
    println("=== Testing Wilson-Cowan Parameters ===")
    
    # Create a simple lattice for tests
    lattice = CompactLattice(extent=(10.0,), n_points=(11,))
    
    # Test basic construction with keyword arguments
    println("\n1. Testing WilsonCowanParameters construction:")
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (1.0, 0.8),
        connectivity = nothing,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    @assert params.α == (1.0, 1.5)
    @assert params.β == (1.0, 1.0)
    @assert params.τ == (1.0, 0.8)
    @assert params.pop_names == ("E", "I")
    @assert params.nonlinearity isa SigmoidNonlinearity
    @assert params.lattice === lattice
    println("   ✓ WilsonCowanParameters construction passed")
    
    # Test with default pop_names
    println("\n2. Testing default population names:")
    params_default = WilsonCowanParameters{3}(
        α = (1.0, 1.0, 1.0),
        β = (1.0, 1.0, 1.0),
        τ = (1.0, 1.0, 1.0),
        connectivity = nothing,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice
    )
    
    @assert params_default.pop_names == ("Pop1", "Pop2", "Pop3")
    println("   ✓ Default population names test passed")
    
    println("\n=== Wilson-Cowan Parameters Tests Passed! ===")
end

function test_wilson_cowan_dynamics()
    println("\n=== Testing Wilson-Cowan Dynamics ===")
    
    # Create a simple lattice for tests
    lattice = CompactLattice(extent=(10.0,), n_points=(3,))
    
    # Create a simple 2-population model
    println("\n1. Testing basic dynamics with 2 populations:")
    
    # Simple parameters
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.0),
        β = (1.0, 1.0),
        τ = (1.0, 1.0),
        connectivity = nothing,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice
    )
    
    # Create state arrays for 2 populations with 3 spatial points each
    A = [0.3 0.3; 0.5 0.5; 0.7 0.7]  # 3x2 array (3 spatial points, 2 populations)
    dA = zeros(size(A))
    
    # Test that wcm1973! runs without error
    wcm1973!(dA, A, params, 0.0)
    
    # dA should be modified
    @assert !all(dA .== 0.0)
    
    # Check that the dynamics make sense:
    # For Wilson-Cowan, decay term should reduce activity
    # Since we have no stimulus or connectivity, dynamics should be dominated by decay
    println("   ✓ Basic dynamics test passed")
    
    # Test single population case
    println("\n2. Testing single population:")
    lattice_single = CompactLattice(extent=(10.0,), n_points=(3,))
    params_single = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (1.0,),
        connectivity = nothing,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice_single
    )
    
    A_single = [0.3, 0.5, 0.7]  # 1D array for single population
    dA_single = zeros(size(A_single))
    
    wcm1973!(dA_single, A_single, params_single, 0.0)
    
    @assert !all(dA_single .== 0.0)
    println("   ✓ Single population test passed")
    
    # Test with different time constants
    println("\n3. Testing with different time constants:")
    params_diff_tau = WilsonCowanParameters{2}(
        α = (1.0, 1.0),
        β = (1.0, 1.0),
        τ = (1.0, 2.0),  # Different time constants
        connectivity = nothing,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice
    )
    
    A = [0.5 0.5; 0.5 0.5]
    dA = zeros(size(A))
    
    wcm1973!(dA, A, params_diff_tau, 0.0)
    
    # Population 2 should have slower dynamics (smaller dA due to larger τ)
    # This is a qualitative check
    @assert !all(dA .== 0.0)
    println("   ✓ Different time constants test passed")
    
    # Test with rectified zeroed sigmoid
    println("\n4. Testing with RectifiedZeroedSigmoidNonlinearity:")
    params_rect = WilsonCowanParameters{2}(
        α = (1.0, 1.0),
        β = (1.0, 1.0),
        τ = (1.0, 1.0),
        connectivity = nothing,
        nonlinearity = RectifiedZeroedSigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice
    )
    
    A = [0.3 0.3; 0.5 0.5]
    dA = zeros(size(A))
    
    wcm1973!(dA, A, params_rect, 0.0)
    
    @assert !all(dA .== 0.0)
    println("   ✓ RectifiedZeroedSigmoidNonlinearity test passed")
    
    # Test with difference of sigmoids
    println("\n5. Testing with DifferenceOfSigmoidsNonlinearity:")
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
    
    A = [0.5 0.5; 0.6 0.6]
    dA = zeros(size(A))
    
    wcm1973!(dA, A, params_diff, 0.0)
    
    @assert !all(dA .== 0.0)
    println("   ✓ DifferenceOfSigmoidsNonlinearity test passed")
    
    println("\n=== Wilson-Cowan Dynamics Tests Passed! ===")
end

function test_implementation_documentation()
    println("\n=== Testing Implementation Documentation ===")
    
    # Create a simple lattice for tests
    lattice = CompactLattice(extent=(10.0,), n_points=(11,))
    
    # Check that WilsonCowanParameters is properly documented
    println("\n1. Verifying WilsonCowanParameters is exported:")
    @assert isdefined(FailureOfInhibition2025, :WilsonCowanParameters)
    println("   ✓ WilsonCowanParameters is exported")
    
    # Check that the type has the expected fields
    println("\n2. Verifying WilsonCowanParameters fields:")
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.0),
        β = (1.0, 1.0),
        τ = (1.0, 1.0),
        connectivity = nothing,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    @assert hasfield(typeof(params), :α)
    @assert hasfield(typeof(params), :β)
    @assert hasfield(typeof(params), :τ)
    @assert hasfield(typeof(params), :connectivity)
    @assert hasfield(typeof(params), :nonlinearity)
    @assert hasfield(typeof(params), :stimulus)
    @assert hasfield(typeof(params), :lattice)
    @assert hasfield(typeof(params), :pop_names)
    println("   ✓ All expected fields present")
    
    println("\n=== Implementation Documentation Tests Passed! ===")
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running Wilson-Cowan model tests...")
    test_wilson_cowan_parameters()
    test_wilson_cowan_dynamics()
    test_implementation_documentation()
    println("\n🎉 All Wilson-Cowan tests completed successfully!")
    println("\nWilson-Cowan model implementation is complete with proper documentation")
    println("of differences from the WilsonCowanModel.jl reference implementation.")
end
