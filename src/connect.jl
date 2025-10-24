using FFTW
import LinearAlgebra: mul!

struct WithDistances{C}
    connectivity::C
    distances
end

struct GaussianConnectivityParameter{T,N}
    amplitude::T
    spread::NTuple{N,T}
end

"""
    ConnectivityMatrix{P}

A PxP matrix of connectivity objects for P populations.

The connectivity matrix follows matrix multiplication conventions:
`connectivity[i,j]` maps the activity of population j into population i.

For example, in a 2-population (E, I) model:
- connectivity[1,1]: E → E (excitatory self-connection)
- connectivity[1,2]: I → E (inhibitory to excitatory)
- connectivity[2,1]: E → I (excitatory to inhibitory)
- connectivity[2,2]: I → I (inhibitory self-connection)

Each element can be:
- A connectivity parameter (e.g., GaussianConnectivityParameter)
- A GaussianConnectivity object (pre-computed)
- `nothing` (no connection)
"""
struct ConnectivityMatrix{P,C}
    matrix::Matrix{C}
    
    function ConnectivityMatrix{P}(matrix::Matrix{C}) where {P,C}
        size(matrix) == (P, P) || throw(ArgumentError("Matrix must be $(P)x$(P), got $(size(matrix))"))
        new{P,C}(matrix)
    end
end

# Constructor from a tuple of tuples for easier construction
function ConnectivityMatrix{P}(data::NTuple{P,NTuple{P,C}}) where {P,C}
    matrix = Matrix{C}(undef, P, P)
    for i in 1:P
        for j in 1:P
            matrix[i,j] = data[i][j]
        end
    end
    ConnectivityMatrix{P}(matrix)
end

# Indexing support
Base.getindex(cm::ConnectivityMatrix, i, j) = cm.matrix[i, j]
Base.size(cm::ConnectivityMatrix) = size(cm.matrix)

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
    ifft_op = plan_irfft(kernel_fft, size(kernel, 1); flags=(FFTW.PATIENT | FFTW.UNALIGNED))
    buffer_real = similar(kernel)
    buffer_complex = similar(kernel_fft)
    buffer_shift = similar(buffer_real)
    GaussianConnectivity(fft_op, ifft_op, kernel_fft, buffer_real, buffer_complex, buffer_shift)
end

function fft_center_idx(arr)
    CartesianIndex(floor.(Ref(Int), size(arr) ./ 2) .+ 1)
end

function calculate_kernel(conn::GaussianConnectivityParameter, lattice)
    # Kernel has ZERO DIST at its center (or floor(extent/2) + 1)
    fft_centered_differences = differences(lattice, coordinates(lattice)[fft_center_idx(lattice)])
    unnormed_kernel = apply_connectivity(conn, fft_centered_differences, step(lattice), fft_centered_differences)
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

function fftshift!(output::AbstractArray, input::AbstractArray)
    # For multi-dimensional arrays, shift by half the size in each dimension
    shift_amounts = floor.(Int, size(output) ./ 2)
    circshift!(output, input, shift_amounts)
end

"""
    propagate_activation(dA, A, ::Nothing, t, lattice=nothing)

No-op connectivity propagation when connectivity is nothing.
"""
propagate_activation(dA, A, ::Nothing, t, lattice=nothing) = nothing

"""
    propagate_activation(dA, A, connectivity::GaussianConnectivity, t)

Propagates activation through a pre-computed GaussianConnectivity object.
Used for testing and simple single-population cases.
"""
function propagate_activation(dA, A, c::GaussianConnectivity, t)
    # Compute fft, multiply by kernel, and invert
    mul!(c.buffer_complex, c.fft_op, A)
    c.buffer_complex .*= c.kernel_fft
    mul!(c.buffer_real, c.ifft_op, c.buffer_complex)
    fftshift!(c.buffer_shift, c.buffer_real)
    dA .+= c.buffer_shift
end

"""
    propagate_activation(dA, A, connectivity::ConnectivityMatrix{P}, t, lattice)

Propagates activation through a connectivity matrix.

For each population pair (i,j), applies connectivity[i,j] which maps 
the activity of population j to influence on population i.
Follows matrix multiplication convention: A_ij maps j → i.
"""
function propagate_activation(dA, A, connectivity::ConnectivityMatrix{P}, t, lattice) where {P}
    # For each target population i
    for i in 1:P
        dAi = population(dA, i)
        
        # Sum contributions from all source populations j
        for j in 1:P
            conn_ij = connectivity[i, j]
            
            # Skip if no connection exists
            if conn_ij === nothing
                continue
            end
            
            # Get source population activity
            Aj = population(A, j)
            
            # Create temporary buffer for this connection's contribution
            temp_contribution = zero(dAi)
            
            # Propagate from source j to target i
            propagate_activation_single(temp_contribution, Aj, conn_ij, t, lattice)
            
            # Add to target population's derivative
            dAi .+= temp_contribution
        end
    end
end

"""
    propagate_activation_single(dA, A, connectivity, t, lattice)

Propagates activation for a single population-to-population connection.
This is a helper function used by the ConnectivityMatrix propagation.
"""
function propagate_activation_single(dA, A, connectivity::GaussianConnectivityParameter, t, lattice)
    # Create GaussianConnectivity if given parameter
    gc = GaussianConnectivity(connectivity, lattice)
    propagate_activation_single(dA, A, gc, t, lattice)
end

function propagate_activation_single(dA, A, c::GaussianConnectivity, t, lattice)
    # Compute fft, multiply by kernel, and invert
    mul!(c.buffer_complex, c.fft_op, A)
    c.buffer_complex .*= c.kernel_fft
    mul!(c.buffer_real, c.ifft_op, c.buffer_complex)
    fftshift!(c.buffer_shift, c.buffer_real)
    dA .+= c.buffer_shift
end
