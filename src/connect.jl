using FFTW
import LinearAlgebra: mul!

struct WithDistances{C}
    connectivity::C
    distances
end

"""
    GaussianConnectivityParameter{T,N}

Parameters for a Gaussian connectivity kernel in N dimensions.

# Fields
- `amplitude::T`: The amplitude or strength of the connection. This represents
  the integral of the connectivity kernel over the entire space.
- `spread::NTuple{N,T}`: The spread (standard deviation) of the Gaussian in each dimension.

# Normalization
The connectivity kernel is normalized such that its integral over the entire
space equals the amplitude parameter. Specifically, the kernel is first normalized
to have unit integral (∫ kernel(x) dx = 1.0 when discretized), and then multiplied 
by the amplitude.

This means:
- For amplitude = 1.0, the integral ∫ kernel(x) dx = 1.0
- For amplitude = A, the integral ∫ kernel(x) dx = A
- The normalization is independent of the spread parameter

The normalization uses the analytical formula for a multivariate Gaussian:
    kernel(x) = amplitude * exp(-||x/σ||²/2) / (√(∏σᵢ²) * (2π)^(N/2))

where σ = (σ₁, σ₂, ..., σₙ) is the spread vector, σᵢ are its individual components,
and N is the number of dimensions.

# Examples
```julia
# 1D excitatory connection with amplitude 1.5 and spread 2.0
conn_e = GaussianConnectivityParameter{Float64,1}(1.5, (2.0,))

# 2D inhibitory connection with amplitude -0.5 and anisotropic spread
conn_i = GaussianConnectivityParameter{Float64,2}(-0.5, (1.0, 1.5))
```
"""
struct GaussianConnectivityParameter{T,N}
    amplitude::T
    spread::NTuple{N,T}
end

"""
    ScalarConnectivity{T}

A simple scalar connectivity coefficient for non-spatial (point) models.

For point models where there is no spatial structure, connectivity is just
a scalar weight that multiplies the source population's activity.

# Fields
- `weight::T`: The connectivity weight/coefficient

# Example
```julia
# For a 2-population point model with E→E, E→I, I→E, I→I connections
conn_ee = ScalarConnectivity(1.0)    # E → E (excitatory)
conn_ei = ScalarConnectivity(-0.5)   # I → E (inhibitory)
conn_ie = ScalarConnectivity(0.8)    # E → I (excitatory)
conn_ii = ScalarConnectivity(-0.3)   # I → I (inhibitory)

connectivity = ConnectivityMatrix{2}([
    conn_ee conn_ei;
    conn_ie conn_ii
])
```
"""
struct ScalarConnectivity{T}
    weight::T
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

"""
    prepare_connectivity(connectivity::ConnectivityMatrix{P}, lattice)

Prepare a ConnectivityMatrix by pre-computing GaussianConnectivity objects from
GaussianConnectivityParameter objects. This ensures that connectivity kernels and
FFT plans are only calculated once, rather than on every propagation step.

Returns a new ConnectivityMatrix with GaussianConnectivity objects where the
input had GaussianConnectivityParameter objects.
"""
function prepare_connectivity(connectivity::ConnectivityMatrix{P}, lattice) where {P}
    prepared_matrix = Matrix{Union{GaussianConnectivity, ScalarConnectivity, Nothing}}(undef, P, P)
    
    for i in 1:P
        for j in 1:P
            conn = connectivity[i, j]
            if conn isa GaussianConnectivityParameter
                # Pre-compute the GaussianConnectivity object
                prepared_matrix[i, j] = GaussianConnectivity(conn, lattice)
            else
                # Keep ScalarConnectivity and nothing as-is
                prepared_matrix[i, j] = conn
            end
        end
    end
    
    return ConnectivityMatrix{P}(prepared_matrix)
end

# No preparation needed for non-ConnectivityMatrix types
prepare_connectivity(connectivity, lattice) = connectivity

struct GaussianConnectivity{T,N}
    amplitude::T
    spread::NTuple{N,T}
    fft_op
    ifft_op
    kernel_fft
    buffer_real
    buffer_complex
    buffer_shift
end

function GaussianConnectivity(param::GaussianConnectivityParameter{T,N}, lattice) where {T,N}
    kernel = calculate_kernel(param, lattice)
    fft_op = plan_rfft(kernel; flags=(FFTW.PATIENT | FFTW.UNALIGNED))
    kernel_fft = fft_op * kernel
    ifft_op = plan_irfft(kernel_fft, size(kernel, 1); flags=(FFTW.PATIENT | FFTW.UNALIGNED))
    buffer_real = similar(kernel)
    buffer_complex = similar(kernel_fft)
    buffer_shift = similar(buffer_real)
    GaussianConnectivity{T,N}(param.amplitude, param.spread, fft_op, ifft_op, kernel_fft, buffer_real, buffer_complex, buffer_shift)
end

function fft_center_idx(arr)
    CartesianIndex(floor.(Ref(Int), size(arr) ./ 2) .+ 1)
end

"""
    calculate_kernel(conn::GaussianConnectivityParameter, lattice)

Calculate a discretized Gaussian connectivity kernel on a lattice.

The kernel is computed such that when discretized on the lattice:
1. The kernel is first normalized to have unit integral (∑ kernel * dx = 1.0)
2. Then scaled by the amplitude parameter

This ensures that the discrete sum approximates the continuous integral:
    ∑ kernel[i] * Δx ≈ amplitude

# Arguments
- `conn::GaussianConnectivityParameter`: Connectivity parameters (amplitude and spread)
- `lattice`: Spatial lattice defining the discretization

# Returns
Discretized kernel array with the same size as the lattice.

# Note
The kernel is centered at the FFT center index (floor(extent/2) + 1) to be
compatible with FFT-based convolution operations.
"""
function calculate_kernel(conn::GaussianConnectivityParameter, lattice)
    # Kernel has ZERO DIST at its center (or floor(extent/2) + 1)
    fft_centered_differences = differences(lattice, coordinates(lattice)[fft_center_idx(lattice)])
    unnormed_kernel = apply_connectivity(conn, fft_centered_differences, step(lattice), fft_centered_differences)
    return unnormed_kernel .* prod(step(lattice))
end

"""
    apply_connectivity_unscaled(conn::GaussianConnectivityParameter, coord_differences)

Compute the unscaled, unnormalized Gaussian kernel value at a given distance.

Returns exp(-||x/σ||²/2) where x is the coordinate difference and σ is the spread.
This is the raw Gaussian shape before normalization and amplitude scaling.
"""
function apply_connectivity_unscaled(conn::GaussianConnectivityParameter{T,N_CDT}, coord_differences::Tup) where {T,N_CDT, Tup<:NTuple{N_CDT,T}}
    exp(
        -sum( (coord_differences ./ conn.spread) .^ 2) / 2
    )
end

"""
    apply_connectivity(connectivity, diffs, step_size, center_diffs)

Apply Gaussian connectivity with proper normalization.

Computes the normalized Gaussian kernel values:
    kernel(x) = amplitude * exp(-||x/σ||²/2) / (√(∏σᵢ²) * (2π)^(N/2))

The normalization constant (√(∏σᵢ²) * (2π)^(N/2)) ensures that the continuous
integral of the Gaussian equals 1.0 before the amplitude is applied.

# Arguments
- `connectivity::GaussianConnectivityParameter`: Connectivity parameters
- `diffs`: Array of coordinate differences where kernel should be evaluated
- `step_size`: Discretization step size (unused but kept for interface compatibility)
- `center_diffs`: Coordinate differences (unused but kept for interface compatibility)

# Returns
Array of normalized kernel values scaled by the amplitude.
"""
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
    propagate_activation(dA, A, ::Nothing, t, lattice)

No-op connectivity propagation when connectivity is nothing.
"""
propagate_activation(dA, A, ::Nothing, t, lattice) = nothing

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

# Note
GaussianConnectivityParameter objects must be pre-computed into GaussianConnectivity
objects using prepare_connectivity() before being used in propagation. This is
automatically done by the WilsonCowanParameters constructor during model initialization.
"""
function propagate_activation_single(dA, A, c::GaussianConnectivity, t, lattice)
    # Compute fft, multiply by kernel, and invert
    mul!(c.buffer_complex, c.fft_op, A)
    c.buffer_complex .*= c.kernel_fft
    mul!(c.buffer_real, c.ifft_op, c.buffer_complex)
    fftshift!(c.buffer_shift, c.buffer_real)
    dA .+= c.buffer_shift
end

"""
    propagate_activation_single(dA, A, connectivity::ScalarConnectivity, t, lattice)

Propagates activation for scalar connectivity in point models.

For point models, this simply multiplies the source activity by the connectivity
weight and adds it to the derivative.
"""
function propagate_activation_single(dA, A, c::ScalarConnectivity, t, lattice)
    # For scalar connectivity, just multiply activity by weight
    dA .+= c.weight .* A
end
