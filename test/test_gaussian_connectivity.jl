#!/usr/bin/env julia

"""
Tests for Gaussian connectivity implementation
"""

using FailureOfInhibition2025
using FFTW

function test_gaussian_connectivity_parameter()
    println("=== Testing GaussianConnectivityParameter ===")
    
    # Test 1D case
    println("\n1. Testing 1D GaussianConnectivityParameter:")
    param_1d = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,1}(
        1.0,  # amplitude
        (0.5,)  # spread
    )
    @assert param_1d.amplitude == 1.0
    @assert param_1d.spread == (0.5,)
    @assert typeof(param_1d.spread) == NTuple{1,Float64}
    println("   ✓ 1D GaussianConnectivityParameter construction passed")
    
    # Test 2D case
    println("\n2. Testing 2D GaussianConnectivityParameter:")
    param_2d = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,2}(
        2.5,  # amplitude
        (0.3, 0.7)  # spread
    )
    @assert param_2d.amplitude == 2.5
    @assert param_2d.spread == (0.3, 0.7)
    @assert typeof(param_2d.spread) == NTuple{2,Float64}
    println("   ✓ 2D GaussianConnectivityParameter construction passed")
    
    println("\n=== GaussianConnectivityParameter Tests Passed! ===")
end

function test_apply_connectivity_unscaled()
    println("\n=== Testing apply_connectivity_unscaled ===")
    
    # Test at origin (should be 1.0 for Gaussian)
    println("\n1. Testing at origin:")
    param = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,1}(
        1.0,  # amplitude
        (1.0,)  # spread
    )
    value_at_origin = FailureOfInhibition2025.apply_connectivity_unscaled(param, (0.0,))
    @assert abs(value_at_origin - 1.0) < 1e-10
    println("   ✓ Connectivity at origin is 1.0")
    
    # Test symmetry
    println("\n2. Testing symmetry:")
    param_2d = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,2}(
        1.0,  # amplitude
        (1.0, 1.0)  # spread
    )
    value_pos = FailureOfInhibition2025.apply_connectivity_unscaled(param_2d, (1.0, 0.0))
    value_neg = FailureOfInhibition2025.apply_connectivity_unscaled(param_2d, (-1.0, 0.0))
    @assert abs(value_pos - value_neg) < 1e-10
    println("   ✓ Connectivity is symmetric")
    
    # Test decay with distance
    println("\n3. Testing decay with distance:")
    value_near = FailureOfInhibition2025.apply_connectivity_unscaled(param, (0.5,))
    value_far = FailureOfInhibition2025.apply_connectivity_unscaled(param, (2.0,))
    @assert value_near > value_far
    @assert value_far > 0.0
    @assert value_near < 1.0
    println("   ✓ Connectivity decays with distance")
    
    # Test Gaussian formula: exp(-sum((x/σ)^2)/2)
    println("\n4. Testing Gaussian formula:")
    dist = (1.0,)
    spread = (2.0,)
    param_test = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,1}(
        1.0,  # amplitude
        spread
    )
    expected = exp(-sum((dist ./ spread) .^ 2) / 2)
    actual = FailureOfInhibition2025.apply_connectivity_unscaled(param_test, dist)
    @assert abs(actual - expected) < 1e-10
    println("   ✓ Gaussian formula is correct")
    
    println("\n=== apply_connectivity_unscaled Tests Passed! ===")
end

function test_calculate_kernel()
    println("\n=== Testing calculate_kernel ===")
    
    # Create a simple 1D lattice
    println("\n1. Testing with 1D lattice:")
    lattice_1d = CompactLattice(extent=(10.0,), n_points=(11,))
    param_1d = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,1}(
        1.0,  # amplitude
        (1.0,)  # spread
    )
    
    kernel_1d = FailureOfInhibition2025.calculate_kernel(param_1d, lattice_1d)
    @assert size(kernel_1d) == size(lattice_1d)
    @assert all(kernel_1d .>= 0.0)  # Gaussian kernel is always non-negative
    println("   ✓ 1D kernel calculated successfully")
    
    # Find center and check it's the maximum
    println("\n2. Testing kernel properties:")
    center_idx = FailureOfInhibition2025.fft_center_idx(lattice_1d)
    center_value = kernel_1d[center_idx]
    @assert center_value >= maximum(kernel_1d) * 0.99  # Allow small numerical error
    println("   ✓ Maximum is at center")
    
    # Test 2D case
    println("\n3. Testing with 2D lattice:")
    lattice_2d = CompactLattice(extent=(10.0, 10.0), n_points=(11, 11))
    param_2d = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,2}(
        1.0,  # amplitude
        (1.0, 1.0)  # spread
    )
    
    kernel_2d = FailureOfInhibition2025.calculate_kernel(param_2d, lattice_2d)
    @assert size(kernel_2d) == size(lattice_2d)
    @assert all(kernel_2d .>= 0.0)
    println("   ✓ 2D kernel calculated successfully")
    
    println("\n=== calculate_kernel Tests Passed! ===")
end

function test_gaussian_connectivity_construction()
    println("\n=== Testing GaussianConnectivity Construction ===")
    
    # Test 1D case
    println("\n1. Testing 1D GaussianConnectivity:")
    lattice_1d = CompactLattice(extent=(10.0,), n_points=(11,))
    param_1d = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,1}(
        1.0,  # amplitude
        (1.0,)  # spread
    )
    
    gc_1d = FailureOfInhibition2025.GaussianConnectivity(param_1d, lattice_1d)
    @assert !isnothing(gc_1d.fft_op)
    @assert !isnothing(gc_1d.ifft_op)
    @assert !isnothing(gc_1d.kernel_fft)
    @assert !isnothing(gc_1d.buffer_real)
    @assert !isnothing(gc_1d.buffer_complex)
    @assert !isnothing(gc_1d.buffer_shift)
    println("   ✓ 1D GaussianConnectivity constructed successfully")
    
    # Test 2D case
    println("\n2. Testing 2D GaussianConnectivity:")
    lattice_2d = CompactLattice(extent=(10.0, 10.0), n_points=(11, 11))
    param_2d = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,2}(
        1.0,  # amplitude
        (1.0, 1.0)  # spread
    )
    
    gc_2d = FailureOfInhibition2025.GaussianConnectivity(param_2d, lattice_2d)
    @assert !isnothing(gc_2d.fft_op)
    @assert !isnothing(gc_2d.ifft_op)
    @assert !isnothing(gc_2d.kernel_fft)
    @assert size(gc_2d.buffer_real) == size(lattice_2d)
    println("   ✓ 2D GaussianConnectivity constructed successfully")
    
    println("\n=== GaussianConnectivity Construction Tests Passed! ===")
end

function test_propagate_activation()
    println("\n=== Testing propagate_activation ===")
    
    # Test 1D propagation
    println("\n1. Testing 1D propagation:")
    lattice_1d = CompactLattice(extent=(10.0,), n_points=(11,))
    param_1d = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,1}(
        1.0,  # amplitude
        (1.0,)  # spread
    )
    gc_1d = FailureOfInhibition2025.GaussianConnectivity(param_1d, lattice_1d)
    
    # Create input activation with a single spike
    A_1d = zeros(11)
    A_1d[6] = 1.0  # Spike in the middle
    dA_1d = zeros(11)
    
    # Propagate
    FailureOfInhibition2025.propagate_activation(dA_1d, A_1d, gc_1d, 0.0)
    
    # Check that activation was propagated
    @assert !all(dA_1d .== 0.0)  # Should have non-zero values
    @assert maximum(abs.(dA_1d)) > 0.0
    println("   ✓ 1D propagation produced non-zero output")
    
    # Test 2D propagation
    println("\n2. Testing 2D propagation:")
    lattice_2d = CompactLattice(extent=(10.0, 10.0), n_points=(11, 11))
    param_2d = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,2}(
        1.0,  # amplitude
        (1.0, 1.0)  # spread
    )
    gc_2d = FailureOfInhibition2025.GaussianConnectivity(param_2d, lattice_2d)
    
    # Create input activation with a single spike
    A_2d = zeros(11, 11)
    A_2d[6, 6] = 1.0  # Spike in the middle
    dA_2d = zeros(11, 11)
    
    # Propagate
    FailureOfInhibition2025.propagate_activation(dA_2d, A_2d, gc_2d, 0.0)
    
    # Check that activation was propagated
    @assert !all(dA_2d .== 0.0)
    @assert maximum(abs.(dA_2d)) > 0.0
    println("   ✓ 2D propagation produced non-zero output")
    
    # Test that activation is preserved (input not modified)
    println("\n3. Testing input preservation:")
    A_test = copy(A_1d)
    dA_test = zeros(11)
    FailureOfInhibition2025.propagate_activation(dA_test, A_test, gc_1d, 0.0)
    @assert A_test == A_1d  # Input should not be modified
    println("   ✓ Input activation is preserved")
    
    println("\n=== propagate_activation Tests Passed! ===")
end

function test_fftshift()
    println("\n=== Testing fftshift! ===")
    
    println("\n1. Testing basic fftshift!:")
    input = [1.0, 2.0, 3.0, 4.0, 5.0]
    output = similar(input)
    FailureOfInhibition2025.fftshift!(output, input)
    
    # For length 5, floor(5/2) = 2, so shift by 2
    # circshift by 2 moves: [1,2,3,4,5] -> [4,5,1,2,3]
    expected = [4.0, 5.0, 1.0, 2.0, 3.0]
    @assert output == expected
    println("   ✓ fftshift! works correctly")
    
    println("\n=== fftshift! Tests Passed! ===")
end

function main()
    println("Running Gaussian connectivity tests...")
    
    test_gaussian_connectivity_parameter()
    test_apply_connectivity_unscaled()
    test_calculate_kernel()
    test_gaussian_connectivity_construction()
    test_propagate_activation()
    test_fftshift()
    
    println("\n🎉 All Gaussian connectivity tests completed successfully!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
