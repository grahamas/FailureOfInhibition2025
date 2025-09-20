"""
    `AbstractSpace{T,N_ARR,N_CDT}`
        with distance-type `T`,
        storable in array of dimension `N_ARR`,
        and with point-coordinates of dimension `N_CDT`.
    """
abstract type AbstractSpace{T,N_ARR,N_CDT} end

"""
    coordinates(space::AbstractSpace)

Return an object in the shape of the space where each element is the coordinate of that element.
"""
coordinates(space::AbstractSpace) = error("undefined.")

"""
    differences(space::AbstractSpace)

Return the distances between every pair of points in `space`.
This computes a matrix where each element [i,j] contains the distance 
between coordinate i and coordinate j in the space.
"""
function differences(space::AbstractSpace{T}) where T
    edges = Iterators.product(coordinates(space), coordinates(space))
    return difference.(Ref(space), edges)
end
"""
    differences(space::AbstractSpace, reference_location::NTuple)

Return the distances from each point in the space to a reference location.
This is more efficient than computing all pairwise distances when you only 
need distances from a single reference point.

# Arguments
- `space`: The spatial lattice
- `reference_location`: Coordinates of the reference point to measure distances from

# Returns
An array with the same shape as the space, where each element contains 
the distance from that coordinate to the reference location.
"""
function differences(space::AbstractSpace{T,N_ARR,N_CDT},
                     reference_location::NTuple{N_CDT,T}) where {T,N_ARR,N_CDT}
    edges = ((coord, reference_location) for coord in coordinates(space))
    return difference.(Ref(space), edges)
end

# Extend Base methods to AbstractSpace types
import Base: step, zero, length, size, ndims

"""
    zero(::Type{NTuple{N,T}})

Create a zero tuple of length N with element type T.
"""
zero(::Type{NTuple{N,T}}) where {N,T} = NTuple{N,T}(zero(T) for _ in 1:N)

"""
    zero(space::AbstractSpace)

Create a zero array with the same dimensions and element type as the space.
"""
zero(space::AbstractSpace{T}) where {T} = zeros(T,size(space)...)

"""
    ndims(space::AbstractSpace)

Return the number of dimensions of the space.
"""
ndims(space::AbstractSpace) = length(size(space))

"""
    fft_center_idx(space::AbstractSpace)

Return the CartesianIndex corresponding to the center of the space for FFT operations.
This is useful when working with Fourier transforms where the zero frequency 
component is typically placed at the center.
"""
function fft_center_idx(space::AbstractSpace)
    CartesianIndex(floor.(Ref(Int), size(space) ./ 2) .+ 1)
end
export fft_center_idx

# Include file definitions of various spaces
include("space/metrics.jl")
include("space/abstract_lattices.jl")
include("space/compact_lattices.jl")
include("space/periodic_lattices.jl")
#include("space/lattice_augmentations.jl")