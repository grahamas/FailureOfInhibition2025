#!/usr/bin/env julia

"""
Tests for ConnectivityMatrix implementation
"""

using FailureOfInhibition2025
using FFTW

function test_connectivity_matrix_construction()
    println("=== Testing ConnectivityMatrix Construction ===")
    
    # Test 2x2 connectivity matrix for 2 populations
    println("\n1. Testing ConnectivityMatrix{2} construction:")
    
    # Create connectivity parameters
    conn_ee = GaussianConnectivityParameter{Float64,1}(1.0, (2.0,))   # E → E
    conn_ei = GaussianConnectivityParameter{Float64,1}(-0.5, (1.5,))  # I → E (inhibitory)
    conn_ie = GaussianConnectivityParameter{Float64,1}(0.8, (2.5,))   # E → I
    conn_ii = GaussianConnectivityParameter{Float64,1}(-0.3, (1.0,))  # I → I (inhibitory)
    
    # Create matrix using array constructor
    matrix = [conn_ee conn_ei; conn_ie conn_ii]
    conn_matrix = ConnectivityMatrix{2}(matrix)
    
    @assert size(conn_matrix) == (2, 2)
    @assert conn_matrix[1, 1] == conn_ee  # E → E
    @assert conn_matrix[1, 2] == conn_ei  # I → E
    @assert conn_matrix[2, 1] == conn_ie  # E → I
    @assert conn_matrix[2, 2] == conn_ii  # I → I
    
    println("   ✓ Matrix construction passed")
    
    # Test tuple of tuples constructor
    println("\n2. Testing tuple of tuples construction:")
    data = ((conn_ee, conn_ei), (conn_ie, conn_ii))
    conn_matrix2 = ConnectivityMatrix{2}(data)
    
    @assert size(conn_matrix2) == (2, 2)
    @assert conn_matrix2[1, 1] == conn_ee
    @assert conn_matrix2[1, 2] == conn_ei
    @assert conn_matrix2[2, 1] == conn_ie
    @assert conn_matrix2[2, 2] == conn_ii
    
    println("   ✓ Tuple construction passed")
    
    # Test with nothing connections (sparse connectivity)
    println("\n3. Testing sparse connectivity (with nothing):")
    sparse_matrix = [conn_ee nothing; conn_ie conn_ii]
    sparse_conn = ConnectivityMatrix{2}(sparse_matrix)
    
    @assert sparse_conn[1, 1] == conn_ee
    @assert sparse_conn[1, 2] === nothing
    @assert sparse_conn[2, 1] == conn_ie
    @assert sparse_conn[2, 2] == conn_ii
    
    println("   ✓ Sparse connectivity passed")
    
    # Test error on wrong size
    println("\n4. Testing size validation:")
    try
        wrong_size = [conn_ee conn_ei conn_ie; conn_ie conn_ii conn_ee]  # 2x3 instead of 2x2
        ConnectivityMatrix{2}(wrong_size)
        @assert false  # Should have thrown an error
    catch e
        @assert e isa ArgumentError
        println("   ✓ Size validation passed (correctly caught error)")
    end
    
    println("\n=== ConnectivityMatrix Construction Tests Passed! ===")
end

function test_connectivity_matrix_indexing_convention()
    println("\n=== Testing Connectivity Matrix Indexing Convention ===")
    
    println("\n1. Verifying A_ij maps j → i:")
    
    # Create simple connectivity parameters with different amplitudes
    # to easily distinguish them
    conn_11 = GaussianConnectivityParameter{Float64,1}(1.0, (1.0,))  # Pop1 → Pop1
    conn_12 = GaussianConnectivityParameter{Float64,1}(2.0, (1.0,))  # Pop2 → Pop1
    conn_21 = GaussianConnectivityParameter{Float64,1}(3.0, (1.0,))  # Pop1 → Pop2
    conn_22 = GaussianConnectivityParameter{Float64,1}(4.0, (1.0,))  # Pop2 → Pop2
    
    matrix = [conn_11 conn_12; conn_21 conn_22]
    conn_matrix = ConnectivityMatrix{2}(matrix)
    
    # conn_matrix[i, j] should map from population j to population i
    @assert conn_matrix[1, 1].amplitude == 1.0  # Pop1 → Pop1
    @assert conn_matrix[1, 2].amplitude == 2.0  # Pop2 → Pop1 (source=2, target=1)
    @assert conn_matrix[2, 1].amplitude == 3.0  # Pop1 → Pop2 (source=1, target=2)
    @assert conn_matrix[2, 2].amplitude == 4.0  # Pop2 → Pop2
    
    println("   ✓ Indexing convention verified: [i,j] maps j → i")
    
    println("\n=== Connectivity Matrix Indexing Convention Tests Passed! ===")
end

function test_propagate_activation_with_connectivity_matrix()
    println("\n=== Testing propagate_activation with ConnectivityMatrix ===")
    
    # Create a 1D lattice
    lattice = CompactLattice(extent=(10.0,), n_points=(11,))
    
    # Create connectivity matrix for 2 populations
    # E → E: excitatory, strong
    # I → E: inhibitory, moderate
    # E → I: excitatory, moderate
    # I → I: inhibitory, weak
    conn_ee = GaussianConnectivityParameter{Float64,1}(1.0, (2.0,))
    conn_ei = GaussianConnectivityParameter{Float64,1}(-0.5, (1.5,))
    conn_ie = GaussianConnectivityParameter{Float64,1}(0.8, (2.5,))
    conn_ii = GaussianConnectivityParameter{Float64,1}(-0.3, (1.0,))
    
    matrix = [conn_ee conn_ei; conn_ie conn_ii]
    conn_matrix = ConnectivityMatrix{2}(matrix)
    
    # Pre-compute connectivity (required after removing backward compatibility)
    conn_matrix_prepared = prepare_connectivity(conn_matrix, lattice)
    
    # Create activity state for 2 populations
    # Each population has 11 spatial points
    A = zeros(11, 2)
    A[6, 1] = 1.0  # Spike in E population
    A[6, 2] = 0.5  # Half spike in I population
    
    dA = zeros(11, 2)
    
    # Propagate activation
    println("\n1. Testing propagation with full connectivity matrix:")
    FailureOfInhibition2025.propagate_activation(dA, A, conn_matrix_prepared, 0.0, lattice)
    
    # Check that both populations received input
    @assert !all(dA[:, 1] .== 0.0)  # E population should have non-zero derivative
    @assert !all(dA[:, 2] .== 0.0)  # I population should have non-zero derivative
    
    println("   ✓ Propagation with full matrix passed")
    
    # Test with sparse connectivity (some connections are nothing)
    println("\n2. Testing propagation with sparse connectivity:")
    sparse_matrix = [conn_ee nothing; conn_ie conn_ii]
    sparse_conn = ConnectivityMatrix{2}(sparse_matrix)
    sparse_conn_prepared = prepare_connectivity(sparse_conn, lattice)
    
    A_sparse = zeros(11, 2)
    A_sparse[6, 1] = 1.0  # Only E population active
    dA_sparse = zeros(11, 2)
    
    FailureOfInhibition2025.propagate_activation(dA_sparse, A_sparse, sparse_conn_prepared, 0.0, lattice)
    
    # E population should receive input only from itself (no I → E connection)
    # I population should receive input from E (E → I connection exists)
    @assert !all(dA_sparse[:, 1] .== 0.0)  # E gets input from E
    @assert !all(dA_sparse[:, 2] .== 0.0)  # I gets input from E
    
    println("   ✓ Sparse connectivity propagation passed")
    
    # Test that input is not modified
    println("\n3. Testing that input activity is preserved:")
    A_test = copy(A)
    dA_test = zeros(11, 2)
    FailureOfInhibition2025.propagate_activation(dA_test, A_test, conn_matrix_prepared, 0.0, lattice)
    
    @assert A_test == A  # Input should not be modified
    println("   ✓ Input preservation passed")
    
    println("\n=== propagate_activation Tests Passed! ===")
end

function test_wilson_cowan_with_connectivity_matrix()
    println("\n=== Testing Wilson-Cowan Model with ConnectivityMatrix ===")
    
    # Create lattice
    lattice = CompactLattice(extent=(10.0,), n_points=(11,))
    
    # Create connectivity matrix for E-I model
    conn_ee = GaussianConnectivityParameter{Float64,1}(1.0, (2.0,))
    conn_ei = GaussianConnectivityParameter{Float64,1}(-0.5, (1.5,))
    conn_ie = GaussianConnectivityParameter{Float64,1}(0.8, (2.5,))
    conn_ii = GaussianConnectivityParameter{Float64,1}(-0.3, (1.0,))
    
    connectivity = ConnectivityMatrix{2}([conn_ee conn_ei; conn_ie conn_ii])
    
    # Create other model components
    nonlinearity = SigmoidNonlinearity(a=2.0, θ=0.5)
    stimulus = nothing
    
    # Create Wilson-Cowan parameters
    params = WilsonCowanParameters{2}(
        α = (1.0, 1.5),
        β = (1.0, 1.0),
        τ = (1.0, 0.8),
        connectivity = connectivity,
        nonlinearity = nonlinearity,
        stimulus = stimulus,
        lattice = lattice,
        pop_names = ("E", "I")
    )
    
    println("\n1. Testing wcm1973! with ConnectivityMatrix:")
    
    # Initial state
    A = zeros(11, 2)
    A[:, 1] .= 0.3  # E population
    A[:, 2] .= 0.2  # I population
    dA = zeros(11, 2)
    
    # Run one step of dynamics
    wcm1973!(dA, A, params, 0.0)
    
    # Check that derivatives were computed
    @assert !all(dA .== 0.0)
    @assert !all(dA[:, 1] .== 0.0)
    @assert !all(dA[:, 2] .== 0.0)
    
    println("   ✓ Wilson-Cowan with ConnectivityMatrix passed")
    
    println("\n=== Wilson-Cowan Model Tests Passed! ===")
end

if abspath(PROGRAM_FILE) == @__FILE__
    println("Running ConnectivityMatrix tests...")
    test_connectivity_matrix_construction()
    test_connectivity_matrix_indexing_convention()
    test_propagate_activation_with_connectivity_matrix()
    test_wilson_cowan_with_connectivity_matrix()
    println("\n🎉 All ConnectivityMatrix tests completed successfully!")
end
