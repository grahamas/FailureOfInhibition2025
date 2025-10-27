#!/usr/bin/env julia

"""
Tests for Gaussian connectivity implementation
"""

using FailureOfInhibition2025
using FFTW

function test_gaussian_connectivity_parameter()
    # Test 1D case
    param_1d = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,1}(
        1.0,  # amplitude
        (0.5,)  # spread
    )
    @assert param_1d.amplitude == 1.0
    @assert param_1d.spread == (0.5,)
    @assert typeof(param_1d.spread) == NTuple{1,Float64}
    
    # Test 2D case
    param_2d = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,2}(
        2.5,  # amplitude
        (0.3, 0.7)  # spread
    )
    @assert param_2d.amplitude == 2.5
    @assert param_2d.spread == (0.3, 0.7)
    @assert typeof(param_2d.spread) == NTuple{2,Float64}
end

function test_apply_connectivity_unscaled()    
    # Test at origin (should be 1.0 for Gaussian)
    param = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,1}(
        1.0,  # amplitude
        (1.0,)  # spread
    )
    value_at_origin = FailureOfInhibition2025.apply_connectivity_unscaled(param, (0.0,))
    @assert abs(value_at_origin - 1.0) < 1e-10
    
    # Test symmetry
    param_2d = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,2}(
        1.0,  # amplitude
        (1.0, 1.0)  # spread
    )
    value_pos = FailureOfInhibition2025.apply_connectivity_unscaled(param_2d, (1.0, 0.0))
    value_neg = FailureOfInhibition2025.apply_connectivity_unscaled(param_2d, (-1.0, 0.0))
    @assert abs(value_pos - value_neg) < 1e-10
    
    # Test decay with distance
    value_near = FailureOfInhibition2025.apply_connectivity_unscaled(param, (0.5,))
    value_far = FailureOfInhibition2025.apply_connectivity_unscaled(param, (2.0,))
    @assert value_near > value_far
    @assert value_far > 0.0
    @assert value_near < 1.0
    
    # Test Gaussian formula: exp(-sum((x/σ)^2)/2)
    dist = (1.0,)
    spread = (2.0,)
    param_test = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,1}(
        1.0,  # amplitude
        spread
    )
    expected = exp(-sum((dist ./ spread) .^ 2) / 2)
    actual = FailureOfInhibition2025.apply_connectivity_unscaled(param_test, dist)
    @assert abs(actual - expected) < 1e-10
end

function test_calculate_kernel()    
    # Create a simple 1D lattice
    lattice_1d = CompactLattice(extent=(10.0,), n_points=(11,))
    param_1d = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,1}(
        1.0,  # amplitude
        (1.0,)  # spread
    )
    
    kernel_1d = FailureOfInhibition2025.calculate_kernel(param_1d, lattice_1d)
    @assert size(kernel_1d) == size(lattice_1d)
    @assert all(kernel_1d .>= 0.0)  # Gaussian kernel is always non-negative
    
    # Find center and check it's the maximum
    center_idx = FailureOfInhibition2025.fft_center_idx(lattice_1d)
    center_value = kernel_1d[center_idx]
    @assert center_value >= maximum(kernel_1d) * 0.99  # Allow small numerical error
    
    # Test 2D case
    lattice_2d = CompactLattice(extent=(10.0, 10.0), n_points=(11, 11))
    param_2d = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,2}(
        1.0,  # amplitude
        (1.0, 1.0)  # spread
    )
    
    kernel_2d = FailureOfInhibition2025.calculate_kernel(param_2d, lattice_2d)
    @assert size(kernel_2d) == size(lattice_2d)
    @assert all(kernel_2d .>= 0.0)

    center_idx = FailureOfInhibition2025.fft_center_idx(lattice_2d)
    center_value = kernel_2d[center_idx]
    @assert center_value >= maximum(kernel_2d) * 0.99
end

function test_gaussian_connectivity_construction()    
    # Test 1D case
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
    
    # Test 2D case
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
end

function test_propagate_activation()    
    # Test 1D propagation
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
    @assert maximum(dA_1d) > 0.0
    
    # Test 2D propagation
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
    @assert maximum(dA_2d) > 0.0
    
    # Test that activation is preserved (input not modified)
    A_test = copy(A_1d)
    dA_test = zeros(11)
    FailureOfInhibition2025.propagate_activation(dA_test, A_test, gc_1d, 0.0)
    @assert A_test == A_1d  # Input should not be modified    
end

function test_fftshift()
    input = [1.0, 2.0, 3.0, 4.0, 5.0]
    output = similar(input)
    FailureOfInhibition2025.fftshift!(output, input)
    
    # For length 5, floor(5/2) = 2, so shift by 2
    # circshift by 2 moves: [1,2,3,4,5] -> [4,5,1,2,3]
    expected = [4.0, 5.0, 1.0, 2.0, 3.0]
    @assert output == expected
end

"""
Test that connectivity kernels are normalized to have unit integral
before the amplitude is applied.

For a properly normalized kernel:
- The discrete sum of kernel values (approximating the integral) should equal the amplitude
- Changing the amplitude should scale the kernel proportionally
- The normalization should be independent of the spread parameter
"""
function test_kernel_normalization()
    
    # Test 1D normalization with various amplitudes
    # Use a large enough lattice to capture the full Gaussian
    lattice_1d = CompactLattice(extent=(40.0,), n_points=(401,))
    
    # Test with amplitude = 1.0 (unit amplitude)
    param_unit = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,1}(1.0, (1.0,))
    kernel_unit = FailureOfInhibition2025.calculate_kernel(param_unit, lattice_1d)
    sum_unit = sum(kernel_unit)
    
    # Kernel with unit amplitude should sum to approximately 1.0
    @assert abs(sum_unit - 1.0) < 0.01
    
    # Test with amplitude = 2.5
    param_scaled = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,1}(2.5, (1.0,))
    kernel_scaled = FailureOfInhibition2025.calculate_kernel(param_scaled, lattice_1d)
    sum_scaled = sum(kernel_scaled)
    
    # Kernel sum should equal the amplitude
    @assert abs(sum_scaled - 2.5) < 0.01
    
    # Ratio should match amplitude ratio
    @assert abs((sum_scaled / sum_unit) - 2.5) < 0.01
    
    # Test with different spread but same amplitude
    param_wide = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,1}(2.5, (3.0,))
    kernel_wide = FailureOfInhibition2025.calculate_kernel(param_wide, lattice_1d)
    sum_wide = sum(kernel_wide)
    
    # Sum should still equal amplitude regardless of spread
    @assert abs(sum_wide - 2.5) < 0.01
    
    # Test 2D normalization
    lattice_2d = CompactLattice(extent=(20.0, 20.0), n_points=(101, 101))
    
    param_2d_unit = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,2}(1.0, (1.5, 2.0))
    kernel_2d_unit = FailureOfInhibition2025.calculate_kernel(param_2d_unit, lattice_2d)
    sum_2d_unit = sum(kernel_2d_unit)
    
    # 2D kernel with unit amplitude should sum to approximately 1.0
    @assert abs(sum_2d_unit - 1.0) < 0.01
    
    param_2d_scaled = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,2}(3.5, (1.5, 2.0))
    kernel_2d_scaled = FailureOfInhibition2025.calculate_kernel(param_2d_scaled, lattice_2d)
    sum_2d_scaled = sum(kernel_2d_scaled)
    
    # 2D kernel sum should equal the amplitude
    @assert abs(sum_2d_scaled - 3.5) < 0.01
    
    # Test with negative amplitude (inhibitory connection)
    param_inhibit = FailureOfInhibition2025.GaussianConnectivityParameter{Float64,1}(-1.5, (1.0,))
    kernel_inhibit = FailureOfInhibition2025.calculate_kernel(param_inhibit, lattice_1d)
    sum_inhibit = sum(kernel_inhibit)
    
    # Negative amplitude should result in negative sum
    @assert abs(sum_inhibit - (-1.5)) < 0.01
end

if abspath(PROGRAM_FILE) == @__FILE__
    test_gaussian_connectivity_parameter()
    test_apply_connectivity_unscaled()
    test_calculate_kernel()
    test_gaussian_connectivity_construction()
    test_propagate_activation()
    test_fftshift()
    test_kernel_normalization()
end
