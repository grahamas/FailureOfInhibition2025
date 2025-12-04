#!/usr/bin/env julia

"""
Tests to verify that connectivity kernels are pre-computed and only calculated once
when creating WilsonCowanParameters, not on every propagation step.
"""

using FailureOfInhibition2025
using Test

function test_gaussian_connectivity_parameter_converted()
    println("=== Testing Connectivity Kernel Pre-computation ===")
    
    # Create a lattice
    lattice = CompactLattice(extent=(10.0,), n_points=(21,))
    
    # Create connectivity matrix with GaussianConnectivityParameter objects
    conn_ee = GaussianConnectivityParameter{Float64,1}(1.0, (2.0,))
    conn_ei = GaussianConnectivityParameter{Float64,1}(-0.5, (1.5,))
    conn_ie = GaussianConnectivityParameter{Float64,1}(0.8, (2.5,))
    conn_ii = GaussianConnectivityParameter{Float64,1}(-0.3, (1.0,))
    
    connectivity_params = ConnectivityMatrix{2}([
        conn_ee conn_ei;
        conn_ie conn_ii
    ])
    
    println("\n1. Testing that WilsonCowanParameters pre-computes connectivity:")
    
    # Create Wilson-Cowan parameters - this should convert parameters to pre-computed objects
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (1.0, 0.8),
        connectivity = connectivity_params,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    # Check that connectivity has been converted to pre-computed GaussianConnectivity objects
    @test params.connectivity isa ConnectivityMatrix{2}
    
    # Each element should now be a GaussianConnectivity object, not a parameter
    for i in 1:2
        for j in 1:2
            conn = params.connectivity[i, j]
            @test conn isa FailureOfInhibition2025.GaussianConnectivity
            @test !(conn isa GaussianConnectivityParameter)
        end
    end
    
    println("   ✓ Connectivity parameters were converted to pre-computed GaussianConnectivity")
    
    println("\n2. Testing that pre-computed connectivity works correctly:")
    
    # Test that the model works with pre-computed connectivity
    A = zeros(21, 2)
    A[:, 1] .= 0.3
    A[:, 2] .= 0.2
    dA = zeros(21, 2)
    
    # This should use the pre-computed connectivity (no kernel recalculation)
    wcm1973!(dA, A, params, 0.0)
    
    @test !all(dA .== 0.0)  # Derivatives should be computed
    println("   ✓ Model dynamics work correctly with pre-computed connectivity")
    
    println("\n3. Testing that prepare_connectivity function works:")
    
    # Test the prepare_connectivity function directly
    prepared = prepare_connectivity(connectivity_params, lattice)
    
    @test prepared isa ConnectivityMatrix{2}
    for i in 1:2
        for j in 1:2
            @test prepared[i, j] isa FailureOfInhibition2025.GaussianConnectivity
        end
    end
    
    println("   ✓ prepare_connectivity function works correctly")
    
    println("\n=== Connectivity Kernel Pre-computation Tests Passed! ===")
end

function test_scalar_connectivity_unchanged()
    println("\n=== Testing ScalarConnectivity Not Modified ===")
    
    # Create point lattice for scalar connectivity
    lattice = PointLattice()
    
    # Create connectivity matrix with ScalarConnectivity objects
    conn_ee = ScalarConnectivity(1.0)
    conn_ei = ScalarConnectivity(-0.5)
    conn_ie = ScalarConnectivity(0.8)
    conn_ii = ScalarConnectivity(-0.3)
    
    connectivity = ConnectivityMatrix{2}([
        conn_ee conn_ei;
        conn_ie conn_ii
    ])
    
    println("\n1. Testing that ScalarConnectivity is preserved:")
    
    # Create Wilson-Cowan parameters
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (10.0, 8.0),
        connectivity = connectivity,
        nonlinearity = SigmoidNonlinearity(a=1.5, θ=0.3),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    # ScalarConnectivity should remain unchanged
    for i in 1:2
        for j in 1:2
            @test params.connectivity[i, j] isa ScalarConnectivity
        end
    end
    
    println("   ✓ ScalarConnectivity objects are preserved (not modified)")
    
    println("\n2. Testing that model works with ScalarConnectivity:")
    
    # Test that the model works
    A = reshape([0.3, 0.2], 1, 2)
    dA = zeros(1, 2)
    
    wcm1973!(dA, A, params, 0.0)
    
    @test !all(dA .== 0.0)
    println("   ✓ Model works correctly with ScalarConnectivity")
    
    println("\n=== ScalarConnectivity Tests Passed! ===")
end

function test_nothing_connectivity_unchanged()
    println("\n=== Testing Nothing Connectivity Preserved ===")
    
    lattice = CompactLattice(extent=(10.0,), n_points=(21,))
    
    # Create sparse connectivity with some nothing entries
    conn_ee = GaussianConnectivityParameter{Float64,1}(1.0, (2.0,))
    conn_ii = GaussianConnectivityParameter{Float64,1}(-0.3, (1.0,))
    
    sparse_connectivity = ConnectivityMatrix{2}([
        conn_ee   nothing;
        nothing   conn_ii
    ])
    
    println("\n1. Testing that nothing entries are preserved:")
    
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (1.0, 0.8),
        connectivity = sparse_connectivity,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    # Check that nothing entries remain nothing
    @test params.connectivity[1, 1] isa FailureOfInhibition2025.GaussianConnectivity
    @test params.connectivity[1, 2] === nothing
    @test params.connectivity[2, 1] === nothing
    @test params.connectivity[2, 2] isa FailureOfInhibition2025.GaussianConnectivity
    
    println("   ✓ Nothing entries are preserved in sparse connectivity")
    
    println("\n2. Testing model works with sparse connectivity:")
    
    A = zeros(21, 2)
    A[:, 1] .= 0.3
    A[:, 2] .= 0.2
    dA = zeros(21, 2)
    
    wcm1973!(dA, A, params, 0.0)
    
    @test !all(dA .== 0.0)
    println("   ✓ Model works correctly with sparse connectivity")
    
    println("\n=== Nothing Connectivity Tests Passed! ===")
end

function test_non_connectivity_matrix_unchanged()
    println("\n=== Testing Non-ConnectivityMatrix Types Unchanged ===")
    
    lattice = CompactLattice(extent=(10.0,), n_points=(21,))
    
    # Test with nothing connectivity
    params = WilsonCowanParameters{1}(
        α = (1.0,),
        β = (1.0,),
        τ = (1.0,),
        connectivity = nothing,
        nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5),
        stimulus = nothing,
        lattice = lattice,
        pop_names = ("E",)
    )
    
    @test params.connectivity === nothing
    println("   ✓ Nothing connectivity is preserved")
    
    println("\n=== Non-ConnectivityMatrix Tests Passed! ===")
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running connectivity kernel pre-computation tests...")
    test_gaussian_connectivity_parameter_converted()
    test_scalar_connectivity_unchanged()
    test_nothing_connectivity_unchanged()
    test_non_connectivity_matrix_unchanged()
    println("\n🎉 All kernel pre-computation tests completed successfully!")
end
