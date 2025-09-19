

struct WithDistances{C}
    connectivity::C
    distances
end

struct GaussianConnectivityParameter{T,N}
    amplitude::T
    spread::NTuple{T,N}
end

struct GaussianConnectivity
    fft_op
    ifft_op
    kernel_fft
    buffer_real
    buffer_complex
    buffer_shift
end

function GaussianConnectivity(param::GaussianConnectivityParameter, lattice)
    kernel = calculate_kernel(param, lattice)
    fft_op = plan_rfft(kernel; flags=(FFTW.PATIENT | FFTW.UNALIGNED))
    kernel_fft = fft_op * kernel
    ifft_op = plan_irfft(kernel_fft; flags=(FFTW.PATIENT | FFTW.UNALIGNED))
    buffer_real = similar(kernel)
    buffer_complex = similar(kernel_fft)
    buffer_shift = similar(buffer_real)
    GaussianConnectivity(fft_op, ifft_op, kernel_fft, buffer_real, buffer_complex, buffer_shift)
end

function fft_center_idx(arr)
    CartesianIndex(floor.(Ref(Int), size(arr) ./ 2) .+ 1)
end

function calculate_kernel(conn, lattice)
    # Kernel has ZERO DIST at its center (or floor(extent/2) + 1)
    fft_centered_differences = differences(lattice, coordinates(lattice)[fft_center_idx(lattice)])
    unnormed_kernel = apply_connectivity(conn.connectivity, fft_centered_differences, step(lattice), fft_centered_differences)
    return unnormed_kernel .* prod(step(lattice))
end

function apply_connectivity_unscaled(conn::GaussianConnectivityParameter{T,N_CDT}, coord_differences::Tup) where {T,N_CDT, Tup<:NTuple{N_CDT,T}}
    exp(
        -sum( (coord_differences ./ conn.spread) .^ 2) / 2
    )
end

function apply_connectivity(connectivity::CONN, diffs::DIFFS, step_size::NTuple{N_CDT,T}, center_diffs::DIFFS) where {T,N_ARR,N_CDT,CDT<:NTuple{N_CDT,T},DIFFS<:AbstractArray{CDT,N_ARR},CONN<:GaussianConnectivityParameter{T,N_CDT}}
    unscaled = apply_connectivity_unscaled.(Ref(connectivity), diffs)
    scaling_diffs = apply_connectivity_unscaled.(Ref(connectivity), center_diffs)
    sum_scaling = sum(scaling_diffs)
    return connectivity.amplitude .* unscaled ./ (sqrt(prod(connectivity.spread .^ 2)) .* (2π)^(N_CDT/2))
    #return connectivity.amplitude .* (unscaled ./ sum_scaling)
end

function fftshift!(output::AbstractVector, input::AbstractVector)
    circshift!(output, input, (floor(Int, length(output) / 2)))
end

"""
    propagate_activation(dA, A, connectivity::GaussianConnectivityParameter{T,N}, t)

Propagates activation `A` through a Gaussian connectivity kernel, writing the result into `dA`.
Uses FFTs for efficient convolution. The `connectivity` parameter defines the kernel properties, and `t` is a time parameter (not directly used in this implementation).

FIXME should cache buffers and (i)fft ops
"""
function propagate_activation(dA, A, connectivity::GaussianConnectivityParameter{T,N}, t) where {T,N}
    kernel = calculate_kernel(dA, connectivity) .* prod(step(dA)) # FIXME step(dA) might need to be step(coordinates(dA))
    kernel_fftd = rfft(kern_dx)
    single_pop = population(dA, 1)
    fft_op = plan_rfft(single_pop; flags=(FFTW.PATIENT | FFTW.UNALIGNED))
    ifft_op = plan_irfft(fft_op * single_pop, size(dA,1); flags=(FFTW.PATIENT | FFTW.UNALIGNED))
    FFTAction(kernel_fftd, similar(kernel_fftd), similar(kern), fft_op, ifft_op)
    # Create buffers
    buffer_complex = similar(kernel_fftd)
    buffer_real = similar(kernel)
    buffer_shift = similar(buffer_real)
    gc = GaussianConnectivity(fft_op, ifft_op, buffer_real, buffer_complex, buffer_shift)
    propagate_activation(dA, A, gc, t)

function propagate_activation(dA, A, c::GaussianConnectivity, t)
    # Compute fft, multiply by kernel, and invert
    mul!(c.buffer_complex, c.fft_op, A)
    c.buffer_complex .*= c.kernel_fft
    mul!(c.buffer_real, c.ifft_op, c.buffer_complex)
    fftshift!(c.buffer_shift, c.buffer_real)
    dA .+= c.buffer_shift
end
